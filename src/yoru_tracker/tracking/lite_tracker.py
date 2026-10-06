# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""YORU Lite: an online tracker for low-latency, live use.

Lite looks only at the current frame's detections and what it remembers of
earlier frames.  It runs no neural network of its own, never waits for a
future frame and never revises a decision it has already reported, which is
what makes it usable inside a closed loop.

One update::

    predict every track's centre (constant velocity, dt from frame IDs)
    primary association     every confirmed track, visible or lost, in one
                            assignment: strict gate, distance from the
                            prediction + IoU + long axis
    recovery association    confirmed tracks still unmatched, against what is
                            left: gate widening with time missing, measured
                            from the prediction or the last sighting,
                            recently lost preferred
    candidate association   tentative tracks against what is left
    age every unmatched track, retire those past max_age
    start a tentative track for every detection still unmatched
    hand-over (known population only): a candidate seen min_hits times while
                            every ID is taken becomes the nearest lost track,
                            however far away -- an animal that jumped -- and
                            the track takes over the candidate's motion

Lost tracks compete in the primary pass on equal terms.  Holding them back
for a later pass looks like giving established tracks priority, but it lets a
visible track whose own detection was missed take a returning animal's
detection uncontested -- an ID swap the benchmark's high-density and
courtship scenarios show plainly.  The recovery pass only ever sees what the
primary pass could not place.  Newcomers come last, so a one-frame false
detection never takes a detection from an identity that already exists.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Iterable, List, Optional, Tuple

import numpy as np

from yoru_tracker.core.capabilities import TrackerInfo
from yoru_tracker.core.config import TrackerConfig
from yoru_tracker.core.tracker_base import TRACKER_API_VERSION, TrackerBase
from yoru_tracker.core.types import (
    Detection,
    EventKind,
    Track,
    TrackEvent,
    TrackingResult,
)
from yoru_tracker.tracking.association import CostModel, Probe, associate, solve
from yoru_tracker.tracking.geometry import center_distance
from yoru_tracker.tracking.lifecycle import Lifecycle, TrackRecord

logger = logging.getLogger(__name__)
event_logger = logging.getLogger("yoru_tracker.events")

#: Extra cost per unit of ``missed_frames / max_age`` in the recovery pass,
#: so that between two lost tracks equally close to a detection the one lost
#: more recently takes it.
_RECOVERY_AGE_PENALTY = 0.1


class LiteTracker(TrackerBase):
    info = TrackerInfo(
        name="YORU Lite",
        version="0.1.0",
        api_version=TRACKER_API_VERSION,
        description="Online motion + geometry tracker; no appearance model.",
        realtime_capable=True,
        requires_frame=False,
        requires_gpu=False,
        supports_obb=True,
        supports_variable_population=True,
        deterministic=True,
    )

    def __init__(self, config: Optional[TrackerConfig] = None):
        super().__init__(config)
        self._costs = CostModel.from_config(self.config.association)
        self.reset()

    # -- TrackerBase ----------------------------------------------------

    def reset(self) -> None:
        had_tracks = bool(getattr(self, "_records", None))
        self._records: List[TrackRecord] = []
        self._lifecycle = Lifecycle(self.config.lifecycle)
        self._next_key = 0
        self._last_frame_id: Optional[int] = None
        self._updates = 0
        # A reset is reported with the next frame's events, so that it shows
        # up in the same stream as the IDs it invalidated.
        self._pending = [TrackEvent(EventKind.RESET, -1, None, "tracker reset")] if had_tracks else []
        if had_tracks:
            logger.info("Tracker reset: every track dropped, IDs restart at 0")

    def update(self, detections: Iterable[Detection], frame_id: int,
               timestamp: Optional[float] = None, frame=None) -> TrackingResult:
        frame_id = int(frame_id)
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError(
                f"frame_id {frame_id} does not follow {self._last_frame_id}: frames "
                f"must be given in increasing order (call reset() to start over)"
            )
        dt = 1 if self._last_frame_id is None else frame_id - self._last_frame_id
        self._last_frame_id = frame_id
        self._updates += 1

        valid: List[Detection] = []
        rejected: List[Detection] = []
        for det in detections:
            (valid if det.is_valid() else rejected).append(det)

        events = [TrackEvent(e.kind, frame_id, e.track_id, e.detail) for e in self._pending]
        self._pending = []

        for record in self._records:
            record.predict(dt)

        confirmed = [r for r in self._records if r.track_id is not None]
        tentative = [r for r in self._records if r.track_id is None]

        remaining = list(range(len(valid)))
        matched = {}  # record key -> (detection index, cost)
        self._match(confirmed, self._primary_probe, valid, remaining, matched)
        if self.config.association.recovery:
            leftover = [r for r in confirmed if r.key not in matched]
            self._match(leftover, self._recovery_probe, valid, remaining, matched)
        self._match(tentative, self._primary_probe, valid, remaining, matched)

        survivors = []
        reported = {}  # record key -> detection index, for confirmed tracks seen this frame
        index_of = {key: hit[0] for key, hit in matched.items()}
        for record in self._records:
            hit = matched.get(record.key)
            if hit is None:
                keep, evs = self._lifecycle.unmatched(record, frame_id)
                events.extend(evs)
                if keep:
                    survivors.append(record)
                continue
            index, cost = hit
            gap = record.missed_frames
            record.absorb(valid[index], frame_id, cost)
            events.extend(self._lifecycle.matched(record, frame_id, gap))
            survivors.append(record)
            if record.track_id is not None:
                reported[record.key] = index

        first_frame = self._updates == 1
        population = self.config.lifecycle.population
        lost_now = [r for r in survivors if r.track_id is not None and r.key not in index_of]
        if population:
            # Room is limited: the likeliest detections become tracks first.
            remaining.sort(key=lambda j: (-valid[j].confidence, j))
        for index in remaining:
            pending = sum(1 for r in survivors if r.track_id is None)
            if not self._lifecycle.has_room(0 if first_frame else pending):
                # Every ID is taken.  A leftover is then only worth following
                # as a candidate to hand to a lost track, and only while
                # there are lost tracks without one.
                if not population or pending >= len(lost_now):
                    continue
            record = TrackRecord(self._next_key, valid[index], frame_id, self.config.kalman)
            self._next_key += 1
            events.extend(self._lifecycle.born(record, frame_id, first_frame))
            survivors.append(record)
            index_of[record.key] = index
            if record.track_id is not None:
                reported[record.key] = index
        if population:
            self._hand_over(survivors, lost_now, index_of, valid, frame_id, events, reported)
        self._records = survivors

        max_age = self.config.lifecycle.max_age
        confirmed = sorted(
            (r for r in self._records if r.track_id is not None),
            key=lambda r: r.track_id,
        )
        tracked = tuple(
            r.tracked(valid[reported[r.key]] if r.key in reported else None, max_age)
            for r in confirmed
        )
        claimed = set(reported.values())
        unassigned = tuple(d for j, d in enumerate(valid) if j not in claimed)

        if self.config.log_events:
            for event in events:
                event_logger.info("frame %d: track %s %s %s", event.frame_id,
                                  event.track_id, event.kind.value, event.detail)

        return TrackingResult(
            frame_id=frame_id,
            timestamp=timestamp,
            tracked=tracked,
            unassigned=unassigned,
            rejected=tuple(rejected),
            events=tuple(events),
        )

    def tracks(self) -> Tuple[Track, ...]:
        max_age = self.config.lifecycle.max_age
        confirmed = sorted((r for r in self._records if r.track_id is not None),
                           key=lambda r: r.track_id)
        tentative = [r for r in self._records if r.track_id is None]
        return tuple(r.summary(max_age) for r in confirmed + tentative)

    # -- association passes ---------------------------------------------

    def _primary_probe(self, record: TrackRecord) -> Probe:
        box = record.predicted_box
        return Probe(box=box, class_id=record.class_id, anchors=((box[0], box[1]),),
                     gate=self.config.association.max_distance)

    def _recovery_probe(self, record: TrackRecord) -> Probe:
        cfg = self.config.association
        box = record.predicted_box
        # The track can have gone further the longer it has been unseen,
        # up to the configured bound.
        scale = min(float(record.missed_frames + 1), cfg.recovery_gate_scale)
        max_age = max(1, self.config.lifecycle.max_age)
        return Probe(
            box=box,
            class_id=record.class_id,
            anchors=((box[0], box[1]), record.last_seen_center),
            gate=cfg.max_distance * scale,
            penalty=_RECOVERY_AGE_PENALTY * record.missed_frames / max_age,
        )

    def _hand_over(self, survivors, lost, index_of, detections, frame_id, events, reported) -> None:
        """Known population: give lost tracks the candidates that cannot get an ID.

        A candidate qualifies once it has been seen ``min_hits`` times -- the
        same evidence a newcomer needs for an ID of its own -- so a one-frame
        false detection is never mistaken for a missing animal.  Lost tracks
        and qualified candidates are paired by distance with no gate: with
        the animal count known, the far one is still the missing one.
        """
        min_hits = self.config.lifecycle.min_hits
        ready = [r for r in survivors
                 if r.track_id is None and r.hits >= min_hits and r.key in index_of]
        lost = [r for r in lost if r.key not in index_of]
        if not ready or not lost or self._lifecycle.has_room():
            return
        class_aware = self.config.association.class_aware
        cost = np.full((len(lost), len(ready)), np.inf)
        for i, track in enumerate(lost):
            anchors = (track.motion.position, track.last_seen_center)
            for k, candidate in enumerate(ready):
                if class_aware and candidate.class_id != track.class_id:
                    continue
                cost[i, k] = min(center_distance(a, candidate.last_seen_center) for a in anchors)
        for row, col, distance in solve(cost, math.inf):
            track, candidate = lost[row], ready[col]
            index = index_of.pop(candidate.key)
            # This frame was already counted as missed; the track is found in it.
            gap = track.missed_frames - 1
            # The candidate's motion has followed the animal since it came
            # back; the track's own would read the jump as speed.
            track.absorb(detections[index], frame_id, None, motion=candidate.motion)
            for event in self._lifecycle.matched(track, frame_id, gap):
                events.append(dataclasses.replace(
                    event, detail=f"{event.detail}, {distance:.0f} px from where it was lost"))
            survivors.remove(candidate)
            index_of[track.key] = index
            reported[track.key] = index

    def _match(self, records, probe_of, detections, remaining, matched) -> None:
        """Associate *records* with the still-unclaimed detections, in place."""
        if not records or not remaining:
            return
        probes = [probe_of(r) for r in records]
        candidates = [detections[j] for j in remaining]
        result = associate(probes, candidates, self._costs)
        taken = set()
        for row, col, cost in result.matches:
            index = remaining[col]
            matched[records[row].key] = (index, cost)
            taken.add(index)
        remaining[:] = [j for j in remaining if j not in taken]
