"""Box geometry: oriented IoU, long-axis comparison, the 180-degree ambiguity."""

from __future__ import annotations

import math

import pytest

from yoru_tracker.tracking.geometry import (
    axis_difference,
    clip_convex,
    contains,
    elongation,
    long_axis,
    obb_iou,
    polygon_area,
    size_difference,
)


def test_identical_boxes():
    box = (50, 50, 40, 16, 0.3)
    assert obb_iou(box, box) == pytest.approx(1.0)


def test_disjoint_boxes():
    assert obb_iou((0, 0, 10, 10, 0), (100, 0, 10, 10, 0)) == 0.0


def test_upright_half_overlap():
    # Two 10x10 squares shifted by 5: intersection 50, union 150.
    assert obb_iou((0, 0, 10, 10, 0), (5, 0, 10, 10, 0)) == pytest.approx(1 / 3)


def test_square_and_its_45_degree_rotation():
    # The intersection is a regular octagon: IoU = 1/sqrt(2).
    a = (0, 0, 2, 2, 0.0)
    b = (0, 0, 2, 2, math.pi / 4)
    assert obb_iou(a, b) == pytest.approx(1 / math.sqrt(2), rel=1e-9)


def test_rotated_overlap_matches_the_upright_case_when_rotated_together():
    a, b = (0, 0, 10, 10, 0), (5, 0, 10, 10, 0)
    # Rotate the whole scene by 30 degrees: IoU is unchanged.
    t = math.radians(30)
    rot = lambda x, y: (x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t))  # noqa: E731
    ra = (*rot(*a[:2]), 10, 10, t)
    rb = (*rot(*b[:2]), 10, 10, t)
    assert obb_iou(ra, rb) == pytest.approx(1 / 3, rel=1e-9)


@pytest.mark.parametrize("box_b", [
    (50, 50, 40, 16, 0.3 + math.pi),            # same rectangle, turned by pi
    (50, 50, 16, 40, 0.3 + math.pi / 2),         # w and h swapped, turned by pi/2
    (50, 50, 16, 40, 0.3 - math.pi / 2),
])
def test_angle_wrap_describes_the_same_box(box_b):
    box_a = (50, 50, 40, 16, 0.3)
    assert obb_iou(box_a, box_b) == pytest.approx(1.0)
    assert axis_difference(box_a, box_b) == pytest.approx(0.0, abs=1e-12)


def test_upright_fast_path_handles_a_quarter_turn():
    # Angle pi/2 is upright with the sides exchanged.
    assert obb_iou((0, 0, 40, 10, 0.0), (0, 0, 10, 40, math.pi / 2)) == pytest.approx(1.0)


def test_axis_difference_is_an_axis_not_a_heading():
    a = (0, 0, 40, 10, 0.0)
    assert axis_difference(a, (0, 0, 40, 10, math.pi / 2)) == pytest.approx(math.pi / 2)
    assert axis_difference(a, (0, 0, 40, 10, math.pi)) == pytest.approx(0.0, abs=1e-12)
    assert axis_difference(a, (0, 0, 40, 10, 0.1)) == pytest.approx(0.1)
    assert axis_difference(a, (0, 0, 40, 10, math.pi - 0.1)) == pytest.approx(0.1)
    assert 0 <= long_axis((0, 0, 10, 40, 3.0)) < math.pi


def test_elongation():
    assert elongation((0, 0, 10, 10, 0)) == 0.0
    assert elongation((0, 0, 40, 10, 0)) == pytest.approx(0.75)


def test_size_difference_is_symmetric_and_capped():
    a, b = (0, 0, 10, 10, 0), (0, 0, 20, 10, 0)
    assert size_difference(a, b) == pytest.approx(math.log(2))
    assert size_difference(b, a) == pytest.approx(math.log(2))
    assert size_difference(a, (0, 0, 1000, 1000, 0)) == 1.0


def test_clipping_works_for_either_winding():
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    shifted = [(1, 1), (3, 1), (3, 3), (1, 3)]
    assert polygon_area(clip_convex(square, shifted)) == pytest.approx(1.0)
    assert polygon_area(clip_convex(square, shifted[::-1])) == pytest.approx(1.0)


def test_degenerate_boxes_have_no_overlap():
    assert obb_iou((0, 0, 0, 10, 0), (0, 0, 10, 10, 0)) == 0.0


def test_contains_includes_the_edge_and_turns_with_the_box():
    upright = (0.0, 0.0, 40.0, 16.0, 0.0)
    assert contains(upright, (19.0, 7.0)) and contains(upright, (20.0, 8.0))
    assert not contains(upright, (0.0, 9.0))
    turned = (0.0, 0.0, 40.0, 16.0, math.pi / 2)  # the long side now runs along y
    assert contains(turned, (0.0, 19.0)) and not contains(turned, (19.0, 0.0))
