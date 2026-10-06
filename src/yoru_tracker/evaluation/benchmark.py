# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Run trackers over the synthetic scenarios and report each one separately.

    yoru-tracker bench                       # Lite vs the YORU baseline, all scenarios
    yoru-tracker bench --json out.json       # machine-readable, for regression checks

Every tracking change should be judged on this table -- ID switches,
fragmentation, recovery, runtime -- scenario by scenario, not on how one
difficult video happens to look.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from yoru_tracker.core.config import TrackerConfig
from yoru_tracker.core.registry import create_tracker
from yoru_tracker.evaluation.metrics import TrackingMetrics, evaluate, predictions_from_results
from yoru_tracker.evaluation.scenarios import SCENARIOS, Scenario, make_scenario
from yoru_tracker.runtime.frames import track_frames


@dataclass(frozen=True)
class BenchmarkRow:
    scenario: str
    tracker: str
    metrics: TrackingMetrics
    ms_per_frame: float

    def to_dict(self) -> dict:
        return {"scenario": self.scenario, "tracker": self.tracker,
                "ms_per_frame": self.ms_per_frame, **self.metrics.to_dict()}


def with_population(config: TrackerConfig, population: int) -> TrackerConfig:
    return dataclasses.replace(
        config, lifecycle=dataclasses.replace(config.lifecycle, population=population))


def score(scenario: Scenario, config: TrackerConfig) -> BenchmarkRow:
    tracker = create_tracker(config)
    t0 = time.perf_counter()
    results = track_frames(tracker, scenario.frames)
    elapsed = time.perf_counter() - t0
    metrics = evaluate(scenario.gt, predictions_from_results(results), scenario.match_distance)
    return BenchmarkRow(scenario.name, config.mode, metrics,
                        1000.0 * elapsed / max(1, len(scenario.frames)))


def run_benchmark(
    configs: Optional[Dict[str, TrackerConfig]] = None,
    scenarios: Optional[Iterable[str]] = None,
    seed: int = 0,
    known_population: bool = True,
) -> List[BenchmarkRow]:
    """Score every configuration on every scenario.

    *configs* maps a label to a configuration; by default Lite and the YORU
    baseline with their default settings.  With *known_population*, every
    Lite configuration is also scored as ``<label>+N``: told the number of
    animals on the scenarios that keep it fixed, unchanged on the others.
    """
    configs = configs or {"lite": TrackerConfig(mode="lite"),
                          "baseline": TrackerConfig(mode="baseline")}
    rows = []
    for name in scenarios or SCENARIOS:
        scenario = make_scenario(name, seed)
        for label, config in configs.items():
            variants = [(label, config)]
            if known_population and config.mode == "lite" and not config.lifecycle.population:
                n = scenario.population or 0
                variants.append((f"{label}+N", with_population(config, n)))
            for tag, variant in variants:
                row = score(scenario, variant)
                rows.append(BenchmarkRow(row.scenario, tag, row.metrics, row.ms_per_frame))
    return rows


_COLUMNS = (
    ("scenario", "{:<18}", lambda r: r.scenario),
    ("tracker", "{:<9}", lambda r: r.tracker),
    ("IDSW", "{:>5}", lambda r: r.metrics.id_switches),
    ("frag", "{:>5}", lambda r: r.metrics.fragmentations),
    ("newID", "{:>5}", lambda r: r.metrics.false_new_ids),
    ("recov", "{:>7}", lambda r: f"{r.metrics.recoveries}/{r.metrics.recovery_opportunities}"),
    ("IDF1", "{:>6}", lambda r: f"{r.metrics.idf1:.3f}"),
    ("MOTA", "{:>6}", lambda r: f"{r.metrics.mota:.3f}"),
    ("recall", "{:>6}", lambda r: f"{r.metrics.recall:.3f}"),
    ("prec", "{:>6}", lambda r: f"{r.metrics.precision:.3f}"),
    ("IDs", "{:>4}", lambda r: f"{r.metrics.predicted_ids}"),
    ("GT", "{:>3}", lambda r: f"{r.metrics.gt_ids}"),
    ("ms/f", "{:>6}", lambda r: f"{r.ms_per_frame:.3f}"),
)


def format_table(rows: Sequence[BenchmarkRow]) -> str:
    header = " ".join(fmt.format(name) for name, fmt, _ in _COLUMNS)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(" ".join(fmt.format(get(row)) for _, fmt, get in _COLUMNS))
    totals = {}
    for row in rows:
        t = totals.setdefault(row.tracker, [0, 0, 0, 0.0, 0])
        t[0] += row.metrics.id_switches
        t[1] += row.metrics.fragmentations
        t[2] += row.metrics.false_new_ids
        t[3] += row.ms_per_frame
        t[4] += 1
    lines.append("-" * len(header))
    for tracker, (idsw, frag, new, ms, n) in totals.items():
        lines.append(f"{'TOTAL':<18} {tracker:<9} {idsw:>5} {frag:>5} {new:>5}"
                     f"{'':>36}mean ms/frame {ms / n:.3f}")
    return "\n".join(lines)
