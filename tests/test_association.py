"""Gating, cost and assignment, without a tracker around them."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from yoru_tracker.tracking.association import CostModel, Probe, associate, pair_cost, solve

from conftest import det

MODEL = CostModel()


def probe(cx, cy, gate=100.0, cls=0, anchors=None, penalty=0.0, w=40.0, h=16.0, angle=0.0):
    box = (cx, cy, w, h, angle)
    return Probe(box=box, class_id=cls, anchors=anchors or ((cx, cy),), gate=gate,
                 penalty=penalty)


def test_outside_the_distance_gate_is_not_a_candidate():
    assert pair_cost(probe(0, 0, gate=50), det(60, 0), MODEL) is None
    assert pair_cost(probe(0, 0, gate=50), det(40, 0), MODEL) is not None


def test_class_aware_gate():
    aware = CostModel(class_aware=True)
    assert pair_cost(probe(0, 0, cls=0), det(1, 0, cls=1), aware) is None
    assert pair_cost(probe(0, 0, cls=0), det(1, 0, cls=1), CostModel(class_aware=False)) is not None


def test_min_iou_gate_rejects_far_non_overlapping_pairs_only():
    model = CostModel(min_iou=0.3)
    assert pair_cost(probe(0, 0), det(45, 0), model) is None       # no overlap, 45 px apart
    assert pair_cost(probe(0, 0), det(5, 0), model) is not None     # overlapping
    # Little overlap but centres within half a body length: kept.
    assert pair_cost(probe(0, 0, w=40, h=4), det(0, 15, w=40, h=4), model) is not None


def test_closer_and_better_aligned_is_cheaper():
    p = probe(0, 0, angle=0.0)
    near = pair_cost(p, det(2, 0), MODEL)
    far = pair_cost(p, det(20, 0), MODEL)
    turned = pair_cost(p, det(2, 0, angle=math.pi / 2), MODEL)
    assert near < far
    assert near < turned


def test_the_nearest_anchor_is_used():
    p = probe(0, 0, anchors=((0.0, 0.0), (200.0, 0.0)), gate=50)
    assert pair_cost(p, det(198, 0), MODEL) is not None


def test_penalty_is_added():
    assert pair_cost(probe(0, 0, penalty=0.5), det(1, 0), MODEL) == pytest.approx(
        pair_cost(probe(0, 0), det(1, 0), MODEL) + 0.5)


def _brute_force(cost, limit):
    """Optimum of: each pair costs c, each unmatched side costs limit/2."""
    n, m = cost.shape
    best = math.inf
    for k in range(min(n, m) + 1):
        for rows in itertools.permutations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                if any(not np.isfinite(cost[r, c]) or cost[r, c] >= limit
                       for r, c in zip(rows, cols)):
                    continue
                total = sum(cost[r, c] for r, c in zip(rows, cols)) + (n + m - 2 * k) * limit / 2
                best = min(best, total)
    return best


@pytest.mark.parametrize("seed", range(25))
def test_solve_finds_the_optimum(seed):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(1, 5, size=2)
    cost = rng.uniform(0, 3, size=(n, m))
    cost[rng.random((n, m)) < 0.3] = np.inf
    limit = 2.0
    pairs = solve(cost, limit)
    total = sum(c for _, _, c in pairs) + (n + m - 2 * len(pairs)) * limit / 2
    assert total == pytest.approx(_brute_force(cost, limit))
    assert all(c < limit for _, _, c in pairs)


def _most_pairs(cost):
    """``(pairs, total)`` of the largest set of allowed pairs, cheapest among those."""
    n, m = cost.shape
    for k in range(min(n, m), 0, -1):
        totals = [
            sum(cost[r, c] for r, c in zip(rows, cols))
            for rows in itertools.combinations(range(n), k)
            for cols in itertools.permutations(range(m), k)
            if all(np.isfinite(cost[r, c]) for r, c in zip(rows, cols))
        ]
        if totals:
            return k, min(totals)
    return 0, 0.0


@pytest.mark.parametrize("seed", range(25))
def test_solve_without_a_limit_pairs_all_the_gates_allow(seed):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(1, 5, size=2)
    cost = rng.uniform(0, 300, size=(n, m))
    cost[rng.random((n, m)) < 0.5] = np.inf
    pairs = solve(cost, math.inf)
    assert (len(pairs), sum(c for *_, c in pairs)) == pytest.approx(_most_pairs(cost))


def test_solve_without_a_limit_survives_rows_competing_for_one_column():
    # Both rows can only take column 0; on its own the solver calls this infeasible.
    cost = np.array([[5.0, np.inf], [3.0, np.inf]])
    assert solve(cost, math.inf) == [(1, 0, 3.0)]


def test_a_pair_dearer_than_two_unmatched_is_not_made():
    # The forced matching would pair A with its second choice to give B a
    # detection; leaving B unmatched is cheaper.
    cost = np.array([[0.1, 0.9], [1.9, np.inf]])
    assert solve(cost, 2.0) == [(0, 0, 0.1)]


def test_associate_reports_leftovers():
    result = associate([probe(0, 0), probe(500, 500)], [det(1, 0), det(900, 0)], MODEL)
    assert [(r, c) for r, c, _ in result.matches] == [(0, 0)]
    assert result.unmatched_tracks == (1,)
    assert result.unmatched_detections == (1,)


def test_empty_inputs():
    assert associate([], [det(0, 0)], MODEL).unmatched_detections == (0,)
    assert associate([probe(0, 0)], [], MODEL).unmatched_tracks == (0,)
