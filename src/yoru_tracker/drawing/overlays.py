# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Drawing tracks on a frame.

Every overlay has a switch in :class:`OverlayOptions`; the defaults show
track boxes, IDs and short trails and nothing else, which stays readable with
a dozen animals.  Drawing reads results and never feeds anything back, so
what is shown cannot change what is tracked.

A trail point remembers whether it was only predicted.  With predicted
positions hidden, a trail runs through the places the animal was seen, and a
lost track -- whose box is hidden -- shows no trail either.
"""

from __future__ import annotations

import colorsys
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from yoru.libs.obb import obb_corners

from yoru_tracker.core.types import Detection, TrackingResult

Point = Tuple[float, float]
#: ``(x, y, predicted)``: where a track was, and whether that was a prediction.
TrailPoint = Tuple[float, float, bool]

_GOLDEN = 0.6180339887498949
_DETECTION_COLOR = (170, 170, 170)


@dataclass(frozen=True)
class OverlayOptions:
    detections: bool = False
    track_boxes: bool = True
    track_ids: bool = True
    predicted: bool = False
    velocity: bool = False
    trajectories: bool = True
    confidence: bool = False
    trail_length: int = 30


def track_color(track_id: int) -> Tuple[int, int, int]:
    """A BGR colour for *track_id*, the same every run and well apart from its neighbours."""
    hue = (0.08 + track_id * _GOLDEN) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _scale(img) -> int:
    return max(1, int(round(min(img.shape[:2]) / 360)))


def _polygon(box) -> np.ndarray:
    return np.array(obb_corners(box), dtype=np.int32).reshape(-1, 1, 2)


def _dashed_polygon(img, box, color, thickness, dash=8):
    pts = obb_corners(box)
    for i in range(4):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % 4]
        length = math.hypot(x2 - x1, y2 - y1)
        steps = max(1, int(length // dash))
        for k in range(0, steps, 2):
            a, b = k / steps, min(1.0, (k + 1) / steps)
            p = (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a))
            q = (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b))
            cv2.line(img, p, q, color, thickness, cv2.LINE_AA)


def _label(img, text, anchor, color, scale):
    font_scale = 0.45 * scale
    thickness = max(1, scale)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = int(anchor[0])
    y = int(anchor[1]) - 4 * scale
    x = max(0, min(x, img.shape[1] - tw - 4))
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base - 2), color, -1)
    luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    ink = (20, 20, 20) if luminance > 140 else (245, 245, 245)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, ink,
                thickness, cv2.LINE_AA)


def draw_detections(img, detections: Iterable[Detection], color=_DETECTION_COLOR,
                    *, labels: bool = True):
    """Detector boxes as they came, before tracking."""
    scale = _scale(img)
    for det in detections:
        cv2.polylines(img, [_polygon(det.box)], True, color, scale, cv2.LINE_AA)
        if labels:
            name = det.class_name or str(det.class_id)
            top_left = min(obb_corners(det.box), key=lambda p: (p[1], p[0]))
            _label(img, f"{name} {det.confidence:.2f}", top_left, color, scale)
    return img


class TrailBook:
    """The last few centres of every track, for drawing trajectories live."""

    def __init__(self, length: int = 30):
        self.length = max(1, int(length))
        self._trails: Dict[int, deque] = {}

    def add(self, result: TrackingResult) -> None:
        present = set()
        for t in result.tracked:
            present.add(t.track_id)
            trail = self._trails.setdefault(t.track_id, deque(maxlen=self.length))
            trail.append((t.cx, t.cy, t.predicted))
        for track_id in list(self._trails):
            if track_id not in present:
                del self._trails[track_id]

    def clear(self) -> None:
        self._trails.clear()

    def trails(self) -> Dict[int, List[TrailPoint]]:
        return {k: list(v) for k, v in self._trails.items()}


def trails_from_results(results: Sequence[TrackingResult], index: int,
                        length: int) -> Dict[int, List[TrailPoint]]:
    """Trails ending at ``results[index]``, for scrubbing through a video."""
    if not results or index < 0:
        return {}
    index = min(index, len(results) - 1)
    alive = {t.track_id for t in results[index].tracked}
    trails: Dict[int, List[TrailPoint]] = {k: [] for k in alive}
    for result in results[max(0, index - length + 1): index + 1]:
        for t in result.tracked:
            if t.track_id in trails:
                trails[t.track_id].append((t.cx, t.cy, t.predicted))
    return trails


def draw_tracking(img, result: Optional[TrackingResult], options: OverlayOptions = OverlayOptions(),
                  *, trails: Optional[Dict[int, List[TrailPoint]]] = None,
                  detections: Optional[Iterable[Detection]] = None):
    """Draw *result* on *img* (in place) and return it."""
    scale = _scale(img)
    if options.detections and detections is not None:
        draw_detections(img, detections, labels=False)
    if result is None:
        return img

    if options.trajectories and trails:
        # Lost tracks, whose boxes are not drawn, get no trail either.
        hidden = set() if options.predicted else {t.track_id for t in result.tracked
                                                  if t.predicted}
        for track_id, points in trails.items():
            if track_id in hidden:
                continue
            shown = [p[:2] for p in points if options.predicted or not p[2]]
            if len(shown) < 2:
                continue
            pts = np.array(shown, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, track_color(track_id), scale, cv2.LINE_AA)

    for t in result.tracked:
        if t.predicted and not options.predicted:
            continue
        color = track_color(t.track_id)
        if options.track_boxes:
            if t.predicted:
                _dashed_polygon(img, t.box, color, scale)
            else:
                cv2.polylines(img, [_polygon(t.box)], True, color, 2 * scale, cv2.LINE_AA)
        if options.velocity:
            vx, vy = t.velocity
            tip = (int(t.cx + 5 * vx), int(t.cy + 5 * vy))
            cv2.arrowedLine(img, (int(t.cx), int(t.cy)), tip, color, scale, cv2.LINE_AA,
                            tipLength=0.3)
        if options.track_ids:
            text = str(t.track_id)
            if options.confidence:
                text += f" {t.track_confidence:.2f}"
            if t.predicted:
                text += " ?"
            top_left = min(obb_corners(t.box), key=lambda p: (p[1], p[0]))
            _label(img, text, top_left, color, scale)
    return img
