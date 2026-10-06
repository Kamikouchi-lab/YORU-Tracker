# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""YORU's own frame-to-frame ID matching, kept as the yardstick for Lite.

YORU's video analysis gives IDs by matching each frame's detection centres to
the previous frame's with the Hungarian algorithm
(``yoru.libs.analysis.match_to_previous``).  This tracker calls that very
function rather than a copy of it, so the benchmark compares Lite with what
YORU users actually get today: no motion model, no memory beyond one frame,
and an ID lost the moment a detection is missed.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from yoru_tracker.core.capabilities import TrackerInfo
from yoru_tracker.core.config import TrackerConfig
from yoru_tracker.core.tracker_base import TRACKER_API_VERSION, TrackerBase
from yoru_tracker.core.types import Detection, TrackedDetection, TrackingResult, TrackState


class BaselineTracker(TrackerBase):
    info = TrackerInfo(
        name="YORU baseline (centre-distance Hungarian)",
        version="1",
        api_version=TRACKER_API_VERSION,
        description="Previous/current centre distance only; for comparison.",
        realtime_capable=True,
        supports_obb=False,
        supports_variable_population=True,
    )

    def __init__(self, config: Optional[TrackerConfig] = None):
        super().__init__(config)
        # Imported here, not at module level: yoru.libs.analysis is YORU's
        # video-analysis module and brings its GUI-side imports with it.
        from yoru.libs.analysis import match_to_previous

        self._match = match_to_previous
        self.reset()

    def reset(self) -> None:
        self._previous: List[Tuple[int, Tuple[float, float]]] = []
        self._next_id = 0
        self._hits = {}
        self._first = {}
        self._last_frame_id: Optional[int] = None

    def update(self, detections: Iterable[Detection], frame_id: int,
               timestamp: Optional[float] = None, frame=None) -> TrackingResult:
        frame_id = int(frame_id)
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError(
                f"frame_id {frame_id} does not follow {self._last_frame_id}: "
                f"frames must be given in increasing order"
            )
        self._last_frame_id = frame_id
        valid, rejected = [], []
        for det in detections:
            (valid if det.is_valid() else rejected).append(det)

        max_dist = self.config.association.max_distance
        matches = self._match(
            [center for _, center in self._previous],
            [d.center for d in valid],
            max_dist if max_dist > 0 else None,
        )
        tracked, current = [], []
        for det, previous_index in zip(valid, matches):
            if previous_index >= 0:
                track_id = self._previous[previous_index][0]
            else:
                track_id = self._next_id
                self._next_id += 1
                self._first[track_id] = frame_id
            self._hits[track_id] = self._hits.get(track_id, 0) + 1
            current.append((track_id, det.center))
            tracked.append(TrackedDetection(
                track_id=track_id,
                track_state=TrackState.ACTIVE,
                box=det.box,
                predicted=False,
                detection=det,
                class_id=det.class_id,
                class_name=det.class_name,
                track_confidence=det.confidence,
                velocity=(0.0, 0.0),
                hits=self._hits[track_id],
                age=frame_id - self._first[track_id] + 1,
                missed_frames=0,
            ))
        self._previous = current
        return TrackingResult(
            frame_id=frame_id,
            timestamp=timestamp,
            tracked=tuple(sorted(tracked, key=lambda t: t.track_id)),
            rejected=tuple(rejected),
        )
