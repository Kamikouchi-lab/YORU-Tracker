"""Metrics on hand-made cases, scenario reproducibility, and the regression gate.

The regression gate (§64): on the synthetic scenarios, Lite must not get
worse than the reference recorded in ``tests/data/lite_reference.json``, and
must stay far ahead of YORU's baseline.  After an intended improvement,
regenerate the reference with::

    python -m yoru_tracker bench --json tests/data/lite_reference.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yoru_tracker.evaluation.benchmark import format_table, run_benchmark
from yoru_tracker.evaluation.metrics import evaluate
from yoru_tracker.evaluation.scenarios import SCENARIOS, make_scenario

REFERENCE = Path(__file__).parent / "data" / "lite_reference.json"


def _gt(n, ids=(1, 2)):
    return {f: [(i, 100.0 * i + f, 50.0) for i in ids] for f in range(n)}


def test_perfect_tracking():
    gt = _gt(10)
    pred = {f: [(i + 10, x, y) for i, x, y in v] for f, v in gt.items()}
    m = evaluate(gt, pred, 5.0)
    assert (m.id_switches, m.fragmentations, m.false_new_ids) == (0, 0, 0)
    assert m.idf1 == pytest.approx(1.0) and m.mota == pytest.approx(1.0)
    assert m.mean_identity_retention == 10


def test_a_swap_is_two_switches():
    gt = _gt(10)
    pred = {}
    for f, v in gt.items():
        swap = f >= 5
        pred[f] = [((2 if swap else 1) if i == 1 else (1 if swap else 2), x, y) for i, x, y in v]
    m = evaluate(gt, pred, 5.0)
    assert m.id_switches == 2 and m.false_new_ids == 0
    assert m.idf1 == pytest.approx(0.5)


def test_new_id_after_a_gap_is_a_failed_recovery():
    gt = {f: [(1, float(f), 0.0)] for f in range(10)}
    pred = {f: [(7 if f < 4 else 8, float(f), 0.0)] for f in range(10) if f not in (4, 5)}
    m = evaluate(gt, pred, 2.0)
    assert m.misses == 2
    assert (m.fragmentations, m.recovery_opportunities, m.recoveries) == (1, 1, 0)
    assert m.false_new_ids == 1 and m.id_switches == 1


def test_same_id_after_a_gap_is_a_recovery():
    gt = {f: [(1, float(f), 0.0)] for f in range(10)}
    pred = {f: [(7, float(f), 0.0)] for f in range(10) if f not in (4, 5)}
    m = evaluate(gt, pred, 2.0)
    assert (m.fragmentations, m.recoveries, m.id_switches) == (1, 1, 0)


def test_one_output_is_never_matched_to_two_animals():
    # Output 5 follows animal 1, then animal 2; then both stand next to it.
    gt = {0: [(1, 0.0, 0.0), (2, 100.0, 0.0)],
          1: [(1, 500.0, 500.0), (2, 0.0, 0.0)],
          2: [(1, 1.0, 0.0), (2, -1.0, 0.0)]}
    pred = {0: [(5, 0.0, 0.0), (6, 100.0, 0.0)],
            1: [(5, 0.0, 0.0)],
            2: [(5, 0.0, 0.0)]}
    m = evaluate(gt, pred, 20.0)
    assert m.matches == 4 and m.false_positives == 0 and m.misses == 2


def test_false_positives_and_misses():
    gt = {0: [(1, 0.0, 0.0)], 1: [(1, 1.0, 0.0)]}
    pred = {0: [(1, 0.0, 0.0), (2, 90.0, 90.0)], 1: []}
    m = evaluate(gt, pred, 2.0)
    assert (m.false_positives, m.misses, m.matches) == (1, 1, 1)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_are_reproducible(name):
    a, b = make_scenario(name, seed=3), make_scenario(name, seed=3)
    assert a.frames == b.frames and a.gt == b.gt
    assert make_scenario(name, seed=4).frames != a.frames


def test_benchmark_table_renders():
    rows = run_benchmark(scenarios=["single"])
    text = format_table(rows)
    assert "single" in text and "lite+N" in text and "baseline" in text


@pytest.fixture(scope="module")
def benchmark_rows():
    return run_benchmark(seed=0)


def test_lite_does_not_regress_against_the_reference(benchmark_rows):
    reference = {(r["scenario"], r["tracker"]): r for r in json.loads(REFERENCE.read_text())}
    worse = []
    for row in benchmark_rows:
        ref = reference.get((row.scenario, row.tracker))
        if ref is None or row.tracker == "baseline":
            continue
        m = row.metrics
        if m.id_switches > ref["id_switches"]:
            worse.append(f"{row.scenario}/{row.tracker}: ID switches {ref['id_switches']} -> {m.id_switches}")
        if m.false_new_ids > ref["false_new_ids"]:
            worse.append(f"{row.scenario}/{row.tracker}: new IDs {ref['false_new_ids']} -> {m.false_new_ids}")
        if m.idf1 < ref["idf1"] - 1e-6:
            worse.append(f"{row.scenario}/{row.tracker}: IDF1 {ref['idf1']:.4f} -> {m.idf1:.4f}")
    assert not worse, "Tracking got worse:\n" + "\n".join(worse)


def test_lite_beats_the_yoru_baseline(benchmark_rows):
    total, by_scenario = {}, {}
    for row in benchmark_rows:
        total[row.tracker] = total.get(row.tracker, 0) + row.metrics.id_switches
        by_scenario.setdefault(row.scenario, {})[row.tracker] = row.metrics.id_switches
    worse = [name for name, n in by_scenario.items() if n["lite"] > n["baseline"]]
    assert not worse, f"more ID switches than YORU's baseline in {worse}"
    # dense_arena, hard for any tracker without appearance, dominates the totals.
    assert total["lite"] * 5 < total["baseline"]
    assert total["lite+N"] <= total["lite"] + 2


def test_the_spare_boxes_are_what_their_scenarios_say():
    spanning = make_scenario("spanning_box")
    spans = [d for f in spanning.frames for d in f.detections if d.confidence < 0.6]
    # Around a pair: taller than two animals side by side, or longer than two end to end.
    assert spans and all(d.h > 30 or d.w > 70 for d in spans)
    twins = [d for f in make_scenario("class_duplicates").frames for d in f.detections
             if d.class_id == 1]
    assert twins and all(d.class_name == "wing_extension" for d in twins)


def test_lite_is_fast(benchmark_rows):
    lite = [r.ms_per_frame for r in benchmark_rows if r.tracker == "lite"]
    # Generous: a real detector takes several milliseconds per frame.
    assert sum(lite) / len(lite) < 2.0
