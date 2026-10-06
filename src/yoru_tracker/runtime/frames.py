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
    """One frame's detections.

    *observed* is false for a video frame the detector never ran on -- in a
    YORU real-time recording the detector takes only some of the camera's
    frames.  Such a frame is kept so that frames line up with the video, but
    it is not an empty frame: nothing was looked for, so nothing was missed.
    """

    frame_id: int
    timestamp: Optional[float]
    detections: Tuple[Detection, ...]
    observed: bool = True


def track_frames(
    tracker: TrackerBase,
    frames: Iterable[FrameDetections],
    *,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[TrackingResult]:
    """Feed *frames* to *tracker* in order and collect the results.

    One result per frame.  An unobserved frame is not given to the tracker --
    to it the frame does not exist, and the next one is a longer step -- and
    its result is empty.
    """
    results = []
    for frame in frames:
        if should_stop is not None and should_stop():
            break
        if not frame.observed:
            results.append(TrackingResult(frame.frame_id, frame.timestamp))
            continue
        results.append(tracker.update(frame.detections, frame.frame_id, frame.timestamp))
    return results
