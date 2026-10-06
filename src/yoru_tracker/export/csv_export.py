# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Tracks as CSV, one row per track per frame.

A row starts with YORU's own detection columns (``DETECTION_COLUMNS``, the
``*_detect.csv`` layout) in the same order, so code that reads YORU's
detections by position reads these rows too; the tracking columns follow::

    x1 y1 x2 y2 confidence class class_name total_time cx cy w h angle
    frame_id track_id track_state predicted vx vy track_confidence association_cost

``total_time`` is the frame's timestamp in seconds (empty when unknown),
``angle`` is in radians, ``vx``/``vy`` in pixels per frame.  Rows the tracker
only predicted -- a lost track's estimated position -- are left out unless
asked for; when included they have ``predicted`` = 1 and an empty
``confidence``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

from yoru.libs.detector_base import DETECTION_COLUMNS

from yoru_tracker.core.types import TrackedDetection, TrackingResult

TRACKING_COLUMNS = (
    "frame_id", "track_id", "track_state", "predicted",
    "vx", "vy", "track_confidence", "association_cost",
)
TRACK_COLUMNS = (*DETECTION_COLUMNS, *TRACKING_COLUMNS)


def _blank(value):
    return "" if value is None else value


def track_row(tracked: TrackedDetection, frame_id: int, timestamp: Optional[float]) -> list:
    det = tracked.detection
    if det is not None:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
    else:
        x1, y1, x2, y2 = tracked.aabb
    cx, cy, w, h, angle = tracked.box
    vx, vy = tracked.velocity
    return [
        x1, y1, x2, y2,
        _blank(tracked.confidence), tracked.class_id, tracked.class_name,
        _blank(timestamp),
        cx, cy, w, h, angle,
        frame_id, tracked.track_id, tracked.track_state.value,
        int(tracked.predicted), vx, vy, tracked.track_confidence,
        _blank(tracked.association_cost),
    ]


class TrackCsvWriter:
    """Write results as they come, for live runs that may stop at any time."""

    def __init__(self, path, *, include_predicted: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.include_predicted = include_predicted
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(TRACK_COLUMNS)
        self.rows = 0

    def write(self, result: TrackingResult) -> None:
        for tracked in result.tracked:
            if tracked.predicted and not self.include_predicted:
                continue
            self._writer.writerow(track_row(tracked, result.frame_id, result.timestamp))
            self.rows += 1

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_tracks_csv(path, results: Iterable[TrackingResult], *,
                     include_predicted: bool = False) -> int:
    """Write every result to *path*; returns the number of rows."""
    with TrackCsvWriter(path, include_predicted=include_predicted) as writer:
        for result in results:
            writer.write(result)
        return writer.rows
