# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""How well a tracker kept identities, measured against ground truth.

Both inputs are per-frame lists of ``(id, x, y)`` centres.  A ground-truth
object and a tracker output are the same animal in a frame when their centres
are within *match_distance*.  Within a frame, a pairing made in the previous
frame is kept while it stays within that distance (an identity is not
reconsidered just because a slightly closer output appeared); the rest are
paired by minimum total distance.

Reported, besides the usual counts:

``id_switches``
    A ground-truth animal matched to a different output ID than last time.
``fragmentations``
    Its matched run interrupted and resumed (by any ID).
``recoveries`` / ``recovery_opportunities``
    Of those resumptions, how many came back with the same ID as before the
    interruption -- the point of keeping lost tracks.
``false_new_ids``
    An animal that already had an ID picked up a brand-new one.
``mean_identity_retention``
    Mean span, in frames, over which an animal kept one ID.
``idf1``
    ID F1: the share of outputs and ground truth that agree under the best
    one-to-one mapping between output IDs and animals over the whole run.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Entry = Tuple[int, float, float]


@dataclass(frozen=True)
class TrackingMetrics:
    frames: int
    gt_objects: int
    predictions: int
    matches: int
    misses: int
    false_positives: int
    id_switches: int
    fragmentations: int
    recoveries: int
    recovery_opportunities: int
    false_new_ids: int
    gt_ids: int
    predicted_ids: int
    mean_identity_retention: float
    idf1: float

    @property
    def recall(self) -> float:
        return self.matches / self.gt_objects if self.gt_objects else 1.0

    @property
    def precision(self) -> float:
        return self.matches / self.predictions if self.predictions else 1.0

    @property
    def mota(self) -> float:
        if not self.gt_objects:
            return 1.0
        return 1.0 - (self.misses + self.false_positives + self.id_switches) / self.gt_objects

    @property
    def recovery_rate(self) -> float:
        if not self.recovery_opportunities:
            return 1.0
        return self.recoveries / self.recovery_opportunities

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(recall=self.recall, precision=self.precision, mota=self.mota,
                    recovery_rate=self.recovery_rate)
        return data


def _pair(gt: Sequence[Entry], pred: Sequence[Entry], keep: Mapping[int, int],
          limit: float) -> List[Tuple[int, int]]:
    """Index pairs ``(gt_index, pred_index)`` for one frame."""
    pairs = []
    used_g, used_p = set(), set()
    pred_index = {p[0]: j for j, p in enumerate(pred)}
    for i, (gid, gx, gy) in enumerate(gt):
        j = pred_index.get(keep.get(gid))
        if j is not None and math.hypot(gx - pred[j][1], gy - pred[j][2]) <= limit:
            pairs.append((i, j))
            used_g.add(i)
            used_p.add(j)
    rest_g = [i for i in range(len(gt)) if i not in used_g]
    rest_p = [j for j in range(len(pred)) if j not in used_p]
    if rest_g and rest_p:
        cost = np.array([[math.hypot(gt[i][1] - pred[j][1], gt[i][2] - pred[j][2])
                          for j in rest_p] for i in rest_g])
        capped = np.minimum(cost, limit * 2.0)
        rows, cols = linear_sum_assignment(capped)
        for r, c in zip(rows, cols):
            if cost[r, c] <= limit:
                pairs.append((rest_g[r], rest_p[c]))
    return pairs


def evaluate(gt_frames: Mapping[int, Sequence[Entry]],
             pred_frames: Mapping[int, Sequence[Entry]],
             match_distance: float) -> TrackingMetrics:
    frame_ids = sorted(set(gt_frames) | set(pred_frames))
    last_pred_of: Dict[int, int] = {}      # gt id -> output id last matched
    was_matched: Dict[int, bool] = {}       # gt id -> matched in its last present frame
    seen_pred = set()
    runs: Dict[int, List[List[int]]] = {}  # gt id -> [[pred id, first frame, last frame]]
    pair_frames: Dict[Tuple[int, int], int] = {}

    gt_total = pred_total = matches = idsw = frag = rec = rec_ops = false_new = 0
    gt_ids, pred_ids = set(), set()
    gt_count: Dict[int, int] = {}
    pred_count: Dict[int, int] = {}

    for fid in frame_ids:
        gt = list(gt_frames.get(fid, ()))
        pred = list(pred_frames.get(fid, ()))
        gt_total += len(gt)
        pred_total += len(pred)
        for g in gt:
            gt_ids.add(g[0])
            gt_count[g[0]] = gt_count.get(g[0], 0) + 1
        for p in pred:
            pred_ids.add(p[0])
            pred_count[p[0]] = pred_count.get(p[0], 0) + 1

        # ID-level co-occurrence for IDF1, independent of the frame matching.
        for g in gt:
            for p in pred:
                if math.hypot(g[1] - p[1], g[2] - p[2]) <= match_distance:
                    key = (g[0], p[0])
                    pair_frames[key] = pair_frames.get(key, 0) + 1

        pairs = _pair(gt, pred, last_pred_of, match_distance)
        matched_gt = set()
        for i, j in pairs:
            gid, pid = gt[i][0], pred[j][0]
            matched_gt.add(gid)
            matches += 1
            previous = last_pred_of.get(gid)
            if previous is not None and previous != pid:
                idsw += 1
                if pid not in seen_pred:
                    false_new += 1
            if previous is not None and not was_matched.get(gid, True):
                frag += 1
                rec_ops += 1
                if previous == pid:
                    rec += 1
            last_pred_of[gid] = pid
            gid_runs = runs.setdefault(gid, [])
            if gid_runs and gid_runs[-1][0] == pid:
                gid_runs[-1][2] = fid
            else:
                gid_runs.append([pid, fid, fid])
        for g in gt:
            was_matched[g[0]] = g[0] in matched_gt
        seen_pred.update(p[0] for p in pred)

    spans = [last - first + 1 for gid_runs in runs.values() for _, first, last in gid_runs]
    retention = sum(spans) / len(spans) if spans else 0.0

    idtp = 0
    if pair_frames:
        g_list = sorted(gt_ids)
        p_list = sorted(pred_ids)
        g_index = {g: i for i, g in enumerate(g_list)}
        p_index = {p: j for j, p in enumerate(p_list)}
        overlap = np.zeros((len(g_list), len(p_list)))
        for (g, p), n in pair_frames.items():
            overlap[g_index[g], p_index[p]] = n
        rows, cols = linear_sum_assignment(-overlap)
        idtp = int(overlap[rows, cols].sum())
    denom = gt_total + pred_total
    idf1 = 2.0 * idtp / denom if denom else 1.0

    return TrackingMetrics(
        frames=len(frame_ids),
        gt_objects=gt_total,
        predictions=pred_total,
        matches=matches,
        misses=gt_total - matches,
        false_positives=pred_total - matches,
        id_switches=idsw,
        fragmentations=frag,
        recoveries=rec,
        recovery_opportunities=rec_ops,
        false_new_ids=false_new,
        gt_ids=len(gt_ids),
        predicted_ids=len(pred_ids),
        mean_identity_retention=retention,
        idf1=idf1,
    )


def predictions_from_results(results: Iterable, *, include_predicted: bool = False
                             ) -> Dict[int, List[Entry]]:
    """``{frame_id: [(track_id, cx, cy), ...]}`` from tracking results."""
    out: Dict[int, List[Entry]] = {}
    for result in results:
        out[result.frame_id] = [
            (t.track_id, t.cx, t.cy) for t in result.tracked
            if include_predicted or not t.predicted
        ]
    return out
