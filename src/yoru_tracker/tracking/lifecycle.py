# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""One track's internal state, and the rules that move it between states.

::

    new detection ──▶ TENTATIVE ──(min_hits matches)──▶ ACTIVE ◀──┐
                          │                               │       │ matched
                       missed                          missed     │
                          ▼                               ▼       │
                      discarded                         LOST ─────┘
                                                          │
                                         missed > max_age ▼
                                                       retired

A track is given its public ID only when it is confirmed, so a one-frame
false detection never uses up a number and IDs stay contiguous.  The single
exception is the first frame after a reset, whose detections are confirmed at
once: there was no earlier frame they could have waited in.

With a known population (``lifecycle.population`` > 0) the animals cannot
leave, so a LOST track is never retired -- it waits, and after max_age stops
being extrapolated -- and no ID beyond the population is ever given out.

Counts (``hits``, ``age``, ``missed_frames``) are in tracker updates -- frames
the tracker was actually given -- so a live run that skips camera frames ages
its tracks exactly as the same detections replayed from a file would.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from yoru_tracker.core.config import KalmanConfig, LifecycleConfig
from yoru_tracker.core.types import (
    Box,
    Detection,
    EventKind,
    Track,
    TrackedDetection,
    TrackEvent,
    TrackState,
)
from yoru_tracker.tracking.geometry import box_size
from yoru_tracker.tracking.kalman import ConstantVelocityModel, StationaryModel

#: Weight of the newest detection in a track's running confidence.
_CONFIDENCE_SMOOTHING = 0.3


class TrackRecord:
    """Everything the tracker keeps about one track.  Never leaves the tracker."""

    __slots__ = (
        "key", "track_id", "state", "class_id", "class_name", "motion",
        "shape", "last_box", "hits", "age", "missed_frames",
        "first_frame_id", "last_seen_frame_id", "detection_confidence",
        "last_cost",
    )

    def __init__(self, key: int, detection: Detection, frame_id: int,
                 kalman: KalmanConfig):
        self.key = key
        self.track_id: Optional[int] = None
        self.state = TrackState.TENTATIVE
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        model = ConstantVelocityModel if kalman.enabled else StationaryModel
        self.motion = model(
            detection.cx, detection.cy, box_size(detection.box),
            process_noise=kalman.process_noise,
            measurement_noise=kalman.measurement_noise,
        )
        self.shape = (detection.w, detection.h, detection.angle)
        self.last_box: Box = detection.box
        self.hits = 1
        self.age = 1
        self.missed_frames = 0
        self.first_frame_id = frame_id
        self.last_seen_frame_id = frame_id
        self.detection_confidence = detection.confidence
        self.last_cost: Optional[float] = None

    # -- motion ---------------------------------------------------------

    @property
    def size(self) -> float:
        return box_size(self.last_box)

    @property
    def predicted_box(self) -> Box:
        x, y = self.motion.position
        return (x, y, *self.shape)

    @property
    def last_seen_center(self) -> Tuple[float, float]:
        return self.last_box[0], self.last_box[1]

    def predict(self, dt: float) -> None:
        self.motion.predict(dt, self.size)
        self.age += 1

    def hold(self) -> None:
        """Stop extrapolating: a track lost this long is expected where it is."""
        state = getattr(self.motion, "state", None)
        if state is not None:
            state[2:] = 0.0

    def absorb(self, detection: Detection, frame_id: int, cost: Optional[float], *,
               motion=None) -> None:
        """Take *detection* as this frame's observation.

        *motion*, if given, replaces the track's motion model instead of being
        corrected by the detection: on a hand-over, the candidate's model
        has followed the animal since it reappeared.
        """
        size = box_size(detection.box)
        if motion is not None:
            self.motion = motion
        elif self.missed_frames and self.motion.implausible(detection.cx, detection.cy, size):
            # Found again far from anywhere its motion allowed: the gap held
            # a jump or a stop, not speed.  Corrected towards the detection,
            # the model would take the distance for velocity and throw the
            # next predictions -- and the exported vx, vy -- far off.
            self.motion.restart(detection.cx, detection.cy, size)
        else:
            self.motion.update(detection.cx, detection.cy, size)
        self.shape = (detection.w, detection.h, detection.angle)
        self.last_box = detection.box
        # A track reports the class of its latest detection: with matching
        # across classes, an animal's behaviour class can change under one ID.
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.hits += 1
        self.missed_frames = 0
        self.last_seen_frame_id = frame_id
        self.last_cost = cost
        self.detection_confidence = (
            (1.0 - _CONFIDENCE_SMOOTHING) * self.detection_confidence
            + _CONFIDENCE_SMOOTHING * detection.confidence
        )

    # -- reporting ------------------------------------------------------

    def track_confidence(self, max_age: int) -> float:
        """Running detector confidence, discounted while the track is unseen."""
        fade = 1.0 - self.missed_frames / (max_age + 1.0)
        return max(0.0, min(1.0, self.detection_confidence * fade))

    def tracked(self, detection: Optional[Detection], max_age: int) -> TrackedDetection:
        predicted = detection is None
        return TrackedDetection(
            track_id=self.track_id,
            track_state=self.state,
            box=self.predicted_box if predicted else detection.box,
            predicted=predicted,
            detection=detection,
            class_id=self.class_id,
            class_name=self.class_name,
            track_confidence=self.track_confidence(max_age),
            velocity=self.motion.velocity,
            hits=self.hits,
            age=self.age,
            missed_frames=self.missed_frames,
            association_cost=None if predicted else self.last_cost,
        )

    def summary(self, max_age: int) -> Track:
        return Track(
            track_id=self.track_id,
            track_state=self.state,
            class_id=self.class_id,
            class_name=self.class_name,
            box=self.predicted_box if self.missed_frames else self.last_box,
            velocity=self.motion.velocity,
            hits=self.hits,
            age=self.age,
            missed_frames=self.missed_frames,
            track_confidence=self.track_confidence(max_age),
            first_frame_id=self.first_frame_id,
            last_seen_frame_id=self.last_seen_frame_id,
        )


class Lifecycle:
    """State transitions and ID allocation, per :class:`LifecycleConfig`."""

    def __init__(self, config: LifecycleConfig):
        self.config = config
        self.next_id = 0
        #: Confirmed tracks currently held (active or lost).
        self.live = 0

    def has_room(self, pending: int = 0) -> bool:
        """May another track be confirmed, with *pending* tentative ones waiting?"""
        population = self.config.population
        return not population or self.live + pending < population

    def _confirm(self, record: TrackRecord, frame_id: int) -> TrackEvent:
        record.track_id = self.next_id
        self.next_id += 1
        self.live += 1
        record.state = TrackState.ACTIVE
        return TrackEvent(EventKind.CREATED, frame_id, record.track_id,
                          f"class {record.class_name or record.class_id}")

    def born(self, record: TrackRecord, frame_id: int, first_frame: bool) -> List[TrackEvent]:
        if (first_frame or record.hits >= self.config.min_hits) and self.has_room():
            return [self._confirm(record, frame_id)]
        return []

    def matched(self, record: TrackRecord, frame_id: int, gap: int) -> List[TrackEvent]:
        """After :meth:`TrackRecord.absorb`; *gap* is how long it had been missing."""
        if record.state is TrackState.TENTATIVE:
            if record.hits >= self.config.min_hits and self.has_room():
                return [self._confirm(record, frame_id)]
            return []
        if record.state is TrackState.LOST:
            record.state = TrackState.ACTIVE
            return [TrackEvent(EventKind.RECOVERED, frame_id, record.track_id,
                               f"after {gap} frame(s)")]
        return []

    def unmatched(self, record: TrackRecord, frame_id: int) -> Tuple[bool, List[TrackEvent]]:
        """Age an unmatched track.  Returns ``(keep, events)``."""
        if record.state is TrackState.TENTATIVE:
            # Never confirmed, never numbered: nothing to report.
            return False, []
        record.missed_frames += 1
        events = []
        if record.state is TrackState.ACTIVE:
            record.state = TrackState.LOST
            events.append(TrackEvent(EventKind.LOST, frame_id, record.track_id))
        if record.missed_frames > self.config.max_age:
            if self.config.population:
                # It cannot have left: keep it, but stop moving it.
                if record.missed_frames == self.config.max_age + 1:
                    record.hold()
                return True, events
            self.live -= 1
            events.append(TrackEvent(
                EventKind.RETIRED, frame_id, record.track_id,
                f"unmatched for {record.missed_frames} frame(s)",
            ))
            return False, events
        return True, events
