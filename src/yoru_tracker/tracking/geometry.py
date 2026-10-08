# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Box geometry used to compare a track with a detection.

Boxes are YORU's ``(cx, cy, w, h, angle)``; corners and the point-in-box
test come from ``yoru.libs.obb`` so the convention is the one the detectors
use.  Pure
Python: a frame has a handful of animals, and gating (see
:mod:`yoru_tracker.tracking.association`) means only a few pairs per frame
ever reach the polygon code.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from yoru.libs.obb import obb_corners, point_in_obb

from yoru_tracker.core.types import Box

Point = Tuple[float, float]

_HALF_PI = math.pi / 2.0
_UPRIGHT_EPS = 1e-9


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def box_size(box: Box) -> float:
    """A single length for a box: the square root of its area."""
    return math.sqrt(max(box[2], 0.0) * max(box[3], 0.0))


def long_side(box: Box) -> float:
    return max(box[2], box[3])


def polygon_area(points: Sequence[Point]) -> float:
    """Unsigned area (shoelace)."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _signed_area(points: Sequence[Point]) -> float:
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def clip_convex(subject: Sequence[Point], clip: Sequence[Point]) -> List[Point]:
    """Intersection of two convex polygons (Sutherland-Hodgman).

    Works for either winding: the side counted as "inside" an edge of *clip*
    is taken from the sign of *clip*'s own area.
    """
    orientation = 1.0 if _signed_area(clip) >= 0 else -1.0
    output = list(subject)
    n = len(clip)
    for i in range(n):
        if not output:
            break
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % n]
        ex, ey = bx - ax, by - ay

        def side(p, ax=ax, ay=ay, ex=ex, ey=ey):
            return orientation * (ex * (p[1] - ay) - ey * (p[0] - ax))

        def crossing(p, q, sp, sq):
            t = sp / (sp - sq)
            return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

        current, output = output, []
        prev = current[-1]
        s_prev = side(prev)
        for point in current:
            s_point = side(point)
            if s_point >= 0:
                if s_prev < 0:
                    output.append(crossing(prev, point, s_prev, s_point))
                output.append(point)
            elif s_prev >= 0:
                output.append(crossing(prev, point, s_prev, s_point))
            prev, s_prev = point, s_point
    return output


def _is_upright(box: Box) -> bool:
    # A multiple of pi/2 is upright too, with w and h trading places.
    return abs(math.sin(2.0 * box[4])) < _UPRIGHT_EPS


def _upright_extent(box: Box) -> Tuple[float, float, float, float]:
    cx, cy, w, h, angle = box
    if abs(math.cos(angle)) < 0.5:  # rotated by ~pi/2: sides swap
        w, h = h, w
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def obb_iou(a: Box, b: Box) -> float:
    """Intersection over union of two oriented boxes, in [0, 1]."""
    area_a = max(a[2], 0.0) * max(a[3], 0.0)
    area_b = max(b[2], 0.0) * max(b[3], 0.0)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    # Boxes whose circumscribed circles do not meet cannot overlap.
    reach = (math.hypot(a[2], a[3]) + math.hypot(b[2], b[3])) / 2.0
    if center_distance(a, b) >= reach:
        return 0.0
    if _is_upright(a) and _is_upright(b):
        ax1, ay1, ax2, ay2 = _upright_extent(a)
        bx1, by1, bx2, by2 = _upright_extent(b)
        iw = min(ax2, bx2) - max(ax1, bx1)
        ih = min(ay2, by2) - max(ay1, by1)
        inter = iw * ih if iw > 0 and ih > 0 else 0.0
    else:
        inter = polygon_area(clip_convex(obb_corners(a), obb_corners(b)))
    union = area_a + area_b - inter
    return max(0.0, min(1.0, inter / union)) if union > 0 else 0.0


def long_axis(box: Box) -> float:
    """Direction of the box's longer side, folded into [0, pi).

    An axis, not a heading: a box and the same box turned by pi have the same
    long axis, and nothing here can tell which end is the head.
    """
    angle = box[4] if box[2] >= box[3] else box[4] + _HALF_PI
    return angle % math.pi


def axis_difference(a: Box, b: Box) -> float:
    """Angle between two boxes' long axes, in [0, pi/2]."""
    d = abs(long_axis(a) - long_axis(b)) % math.pi
    return min(d, math.pi - d)


def elongation(box: Box) -> float:
    """0 for a square, approaching 1 for a thin line."""
    longer = long_side(box)
    if longer <= 0:
        return 0.0
    return 1.0 - min(box[2], box[3]) / longer


def size_difference(a: Box, b: Box) -> float:
    """``|ln(area_a / area_b)|``, capped at 1."""
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    if area_a <= 0 or area_b <= 0:
        return 1.0
    return min(1.0, abs(math.log(area_a / area_b)))


def moved(box: Box, cx: float, cy: float) -> Box:
    """*box* with its centre placed at ``(cx, cy)``."""
    return (float(cx), float(cy), box[2], box[3], box[4])


def contains(box: Box, point: Point) -> bool:
    """Is *point* inside *box*, edge included?"""
    return point_in_obb(point[0], point[1], box)
