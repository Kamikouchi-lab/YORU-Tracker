# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Detections grouped by frame, and the one loop that tracks them.

Video, batch and stored-detection tracking all end in :func:`track_frames`,
and the live runtime feeds the tracker the same way one frame at a time.
There is no second tracking path whose decisions could drift from this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

from yoru_tracker.core.tracker_base import TrackerBase
from yoru_tracker.core.types import Detection, TrackingResult


@dataclass(frozen=True)
class FrameDetections:
    frame_id: int
    timestamp: Optional[float]
    detections: Tuple[Detection, ...]


def track_frames(
    tracker: TrackerBase,
    frames: Iterable[FrameDetections],
    *,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[TrackingResult]:
    """Feed *frames* to *tracker* in order and collect the results."""
    results = []
    for frame in frames:
        if should_stop is not None and should_stop():
            break
        results.append(tracker.update(frame.detections, frame.frame_id, frame.timestamp))
    return results
