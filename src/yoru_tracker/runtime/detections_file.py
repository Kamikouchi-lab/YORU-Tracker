# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Detections stored on disk, so a video is tracked again without its detector.

Running the detector is the slow part of tracking a video; tracking itself
takes a fraction of a millisecond per frame.  Saving the detections once lets
the tracker settings be tuned and re-run in seconds, and makes a tracking run
exactly repeatable on another machine with no GPU.

Three layouts are read, told apart by their header:

``yoru-tracker``
    Written by :func:`write_detections_csv`: ``frame_id`` followed by YORU's
    ``DETECTION_COLUMNS`` (``total_time`` holding the timestamp).  A frame in
    which nothing was detected is a row with only ``frame_id`` and
    ``total_time`` filled in, so the tracker sees every frame it saw before;
    a frame it never saw has no row at all.

``yoru-analysis``
    YORU's video-analysis table (``frame, x1, ..., x_center, y_center, w, h,
    angle, confidence, class, class_name``).  Every frame between the first
    and last row was analysed, so frame IDs missing from the table are frames
    with no detections and are fed to the tracker empty.

``yoru-realtime``
    YORU's live ``*_detect.csv``.  Its rows carry the capture time of the
    frame the detector ran on, and a detection is written again with every
    video frame until the next one is ready; repeats are removed and each
    distinct capture time becomes one frame.  With the matching ``*_log.csv``
    beside it, frame IDs are the recorded video's frame numbers.  Frames on
    which the detector found nothing cannot be recovered from this file.

Files saved from a spreadsheet often start with a byte-order mark; it is
skipped.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from yoru.libs.detector_base import DETECTION_COLUMNS

from yoru_tracker.core.types import Detection
from yoru_tracker.runtime.frames import FrameDetections

DETECTIONS_FILE_COLUMNS = ("frame_id", *DETECTION_COLUMNS)

_ANALYSIS_PREFIX = ("frame", "x1", "y1", "x2", "y2", "x_center", "y_center")


def write_detections_csv(path, frames: Iterable[FrameDetections]) -> int:
    """Write *frames* in the ``yoru-tracker`` layout; returns the frame count.

    Unobserved frames are left out, so read back they are again gaps the
    tracker steps over, not empty frames in which every animal was missed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    blank = [""] * len(DETECTION_COLUMNS)
    time_index = DETECTION_COLUMNS.index("total_time")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(DETECTIONS_FILE_COLUMNS)
        for frame in frames:
            if not frame.observed:
                continue
            count += 1
            timestamp = "" if frame.timestamp is None else frame.timestamp
            if not frame.detections:
                row = list(blank)
                row[time_index] = timestamp
                writer.writerow([frame.frame_id, *row])
                continue
            for d in frame.detections:
                writer.writerow([
                    frame.frame_id, d.x1, d.y1, d.x2, d.y2, d.confidence,
                    d.class_id, d.class_name, timestamp,
                    d.cx, d.cy, d.w, d.h, d.angle,
                ])
    return count


def _float_or_none(text) -> Optional[float]:
    if text is None or str(text).strip() == "":
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def _group(rows: Iterable[Tuple[int, Optional[float], Optional[Detection]]]) -> List[FrameDetections]:
    frames: dict = {}
    times: dict = {}
    for frame_id, timestamp, det in rows:
        bucket = frames.setdefault(frame_id, [])
        if timestamp is not None:
            times.setdefault(frame_id, timestamp)
        if det is not None:
            bucket.append(det)
    return [FrameDetections(fid, times.get(fid), tuple(frames[fid])) for fid in sorted(frames)]


def _read_tracker_layout(reader) -> List[FrameDetections]:
    rows = []
    for line in reader:
        if not line or not "".join(line).strip():
            continue
        frame_id = int(float(line[0]))
        rest = line[1:]
        timestamp = _float_or_none(rest[DETECTION_COLUMNS.index("total_time")])
        det = Detection.from_row(rest) if rest[0].strip() else None
        rows.append((frame_id, timestamp, det))
    return _group(rows)


def _read_analysis_layout(header, reader) -> List[FrameDetections]:
    col = {name: i for i, name in enumerate(header)}
    rows = []
    for line in reader:
        if not line or not "".join(line).strip():
            continue
        frame_id = int(float(line[col["frame"]]))
        det = Detection(
            float(line[col["x1"]]), float(line[col["y1"]]),
            float(line[col["x2"]]), float(line[col["y2"]]),
            float(line[col["confidence"]]), int(float(line[col["class"]])),
            line[col["class_name"]],
            float(line[col["x_center"]]), float(line[col["y_center"]]),
            float(line[col["w"]]), float(line[col["h"]]), float(line[col["angle"]]),
        )
        rows.append((frame_id, None, det))
    frames = _group(rows)
    if not frames:
        return frames
    by_id = {f.frame_id: f for f in frames}
    first, last = frames[0].frame_id, frames[-1].frame_id
    return [by_id.get(i, FrameDetections(i, None, ())) for i in range(first, last + 1)]


def _one_repeat(rows):
    """The shortest prefix of *rows* that, repeated, makes up all of them."""
    n = len(rows)
    for period in range(1, n + 1):
        if n % period == 0 and all(rows[i] == rows[i % period] for i in range(n)):
            return rows[:period]
    return rows


def _read_realtime_layout(reader, log_path: Optional[Path]) -> List[FrameDetections]:
    time_col = DETECTION_COLUMNS.index("total_time")
    # Consecutive rows with one capture time are that frame's detections,
    # written once per video frame until the next result arrived.
    runs: List[Tuple[str, list]] = []
    for line in reader:
        if not line:
            continue
        stamp = line[time_col].strip()
        if runs and runs[-1][0] == stamp:
            runs[-1][1].append(tuple(line))
        else:
            runs.append((stamp, [tuple(line)]))
    unique = []
    seen = set()
    for stamp, rows in runs:
        if stamp in seen:
            continue
        seen.add(stamp)
        unique.append((stamp, [Detection.from_row(r) for r in _one_repeat(rows)]))

    frame_of = {}
    if log_path is not None and log_path.is_file():
        with open(log_path, newline="", encoding="utf-8-sig") as f:
            log = csv.reader(f)
            next(log, None)
            for line in log:
                if len(line) >= 2:
                    frame_of.setdefault(line[1].strip(), int(float(line[0])))
    frames = []
    for index, (stamp, dets) in enumerate(unique):
        timestamp = _float_or_none(stamp)
        frame_id = frame_of.get(stamp, index if not frame_of else None)
        if frame_id is None:
            continue
        frames.append(FrameDetections(frame_id, timestamp, tuple(dets)))
    frames.sort(key=lambda f: f.frame_id)
    return frames


def realtime_log_path(path) -> Optional[Path]:
    """The ``*_log.csv`` YORU writes beside a real-time ``*_detect.csv``, if it is there."""
    path = Path(path)
    suffix = "_detect.csv"
    if not path.name.endswith(suffix):
        return None
    log = path.with_name(path.name[:-len(suffix)] + "_log.csv")
    return log if log.is_file() else None


def load_detections(path) -> Tuple[str, List[FrameDetections]]:
    """``(layout, frames)`` for a detections CSV in any of the three layouts."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = tuple(h.strip() for h in next(reader, ()))
        if header[:len(DETECTIONS_FILE_COLUMNS)] == DETECTIONS_FILE_COLUMNS:
            return "yoru-tracker", _read_tracker_layout(reader)
        if header[:len(_ANALYSIS_PREFIX)] == _ANALYSIS_PREFIX:
            return "yoru-analysis", _read_analysis_layout(header, reader)
        if header[:len(DETECTION_COLUMNS)] == DETECTION_COLUMNS:
            return "yoru-realtime", _read_realtime_layout(reader, realtime_log_path(path))
    raise ValueError(
        f"{path.name}: not a detections file YORU Tracker can read "
        f"(header starts {', '.join(header[:4]) or '<empty>'})"
    )


def align_to_video(layout: str, frames: Sequence[FrameDetections],
                   fps: float) -> List[FrameDetections]:
    """*frames* as one entry per video frame, from frame 0 to the last one read.

    What a frame missing from the file means depends on who wrote it.  YORU's
    video analysis ran the detector on every frame, so there it is a frame in
    which nothing was found.  A YORU real-time recording leaves out the frames
    its detector never ran on, and this application's own file the frames its
    tracker never saw; those are filled in unobserved, so tracking the
    aligned frames makes exactly the decisions tracking the file itself does.
    """
    by_id = {f.frame_id: f for f in frames}
    observed = layout == "yoru-analysis"
    return [by_id.get(i) or FrameDetections(i, i / fps, (), observed)
            for i in range(max(by_id, default=-1) + 1)]
