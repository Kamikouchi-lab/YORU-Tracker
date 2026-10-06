# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Matching tracks to detections: gate, cost, optimal assignment.

Each track is described to this module by a :class:`Probe` -- where it is
expected, how far it may have gone, which class it is -- so the matching
rules can be tested without a tracker around them.

Gating comes first.  A pair outside a gate is not merely expensive, it is not
a candidate at all, so the assignment can never be pushed into an implausible
match just because every detection has to go somewhere.  Among the candidates
the assignment minimises total cost, where leaving a track and a detection
both unmatched costs ``distance_weight + iou_weight``: a pair dearer than
that is not worth making, even if it is inside the gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from yoru_tracker.core.config import AssociationConfig
from yoru_tracker.core.types import Box, Detection
from yoru_tracker.tracking.geometry import (
    axis_difference,
    center_distance,
    elongation,
    long_side,
    moved,
    obb_iou,
    size_difference,
)

Point = Tuple[float, float]

_HALF_PI = math.pi / 2.0


@dataclass(frozen=True)
class Probe:
    """One track, as association sees it.

    *anchors* are the points a detection's distance is measured from, the
    track's box being placed at whichever is nearest; the first is the motion
    prediction, a second can be the last place the track was seen.  *gate*
    is the furthest a detection may be from every anchor, in pixels.
    *penalty* is added to every pair this track forms.
    """

    box: Box
    class_id: int
    anchors: Tuple[Point, ...]
    gate: float
    penalty: float = 0.0


@dataclass(frozen=True)
class CostModel:
    distance_weight: float = 1.0
    iou_weight: float = 1.0
    axis_weight: float = 0.25
    size_weight: float = 0.0
    min_iou: float = 0.0
    class_aware: bool = True

    @classmethod
    def from_config(cls, config: AssociationConfig) -> "CostModel":
        return cls(
            distance_weight=config.distance_weight,
            iou_weight=config.iou_weight,
            axis_weight=config.axis_weight,
            size_weight=config.size_weight,
            min_iou=config.min_iou,
            class_aware=config.class_aware,
        )

    @property
    def match_limit(self) -> float:
        """A pair costing this much or more is left unmatched."""
        return self.distance_weight + self.iou_weight


def pair_cost(probe: Probe, detection: Detection, model: CostModel) -> Optional[float]:
    """Cost of matching *detection* to the track *probe*; ``None`` if gated out."""
    if model.class_aware and probe.class_id != detection.class_id:
        return None
    best_anchor, distance = None, math.inf
    for anchor in probe.anchors:
        d = center_distance(anchor, detection.center)
        if d < distance:
            best_anchor, distance = anchor, d
    if best_anchor is None or distance > probe.gate:
        return None

    reference = moved(probe.box, *best_anchor)
    det_box = detection.box
    need_iou = model.iou_weight > 0 or model.min_iou > 0
    iou = obb_iou(reference, det_box) if need_iou else 0.0
    if (model.min_iou > 0 and iou < model.min_iou
            and distance > 0.5 * max(long_side(reference), long_side(det_box))):
        return None

    cost = model.distance_weight * (distance / probe.gate if probe.gate > 0 else 0.0)
    cost += model.iou_weight * (1.0 - iou)
    if model.axis_weight > 0:
        # Disagreeing long axes only count when both boxes have one.
        weight = min(elongation(reference), elongation(det_box))
        cost += model.axis_weight * weight * axis_difference(reference, det_box) / _HALF_PI
    if model.size_weight > 0:
        cost += model.size_weight * size_difference(reference, det_box)
    return cost + probe.penalty


def cost_matrix(probes: Sequence[Probe], detections: Sequence[Detection],
                model: CostModel) -> np.ndarray:
    """``len(probes) x len(detections)`` costs; ``inf`` marks a gated pair."""
    cost = np.full((len(probes), len(detections)), np.inf)
    for i, probe in enumerate(probes):
        for j, det in enumerate(detections):
            c = pair_cost(probe, det, model)
            if c is not None:
                cost[i, j] = c
    return cost


def solve(cost: np.ndarray, limit: float) -> List[Tuple[int, int, float]]:
    """Minimum-cost assignment; pairs at or above *limit* stay unmatched.

    Capping every cost at *limit* before solving is the same optimisation as
    letting each side stay unmatched at ``limit / 2``.  Returns
    ``(row, column, cost)`` triples in row order.
    """
    if cost.size == 0 or not np.isfinite(cost).any():
        return []
    capped = np.minimum(cost, limit)
    rows, cols = linear_sum_assignment(capped)
    return [
        (int(r), int(c), float(cost[r, c]))
        for r, c in zip(rows, cols)
        if np.isfinite(cost[r, c]) and cost[r, c] < limit
    ]


@dataclass(frozen=True)
class Assignment:
    matches: Tuple[Tuple[int, int, float], ...]
    unmatched_tracks: Tuple[int, ...]
    unmatched_detections: Tuple[int, ...]


def associate(probes: Sequence[Probe], detections: Sequence[Detection],
              model: CostModel) -> Assignment:
    matches = solve(cost_matrix(probes, detections, model), model.match_limit)
    used_rows = {r for r, _, _ in matches}
    used_cols = {c for _, c, _ in matches}
    return Assignment(
        matches=tuple(matches),
        unmatched_tracks=tuple(i for i in range(len(probes)) if i not in used_rows),
        unmatched_detections=tuple(j for j in range(len(detections)) if j not in used_cols),
    )
