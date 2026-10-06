# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Synthetic behavioural scenarios with known identities.

Each scenario moves fly-sized animals (40 x 16 px) through a 640 x 480 arena
and produces what a detector would report for them: centre jitter, size
jitter, occasional misses -- and, where the scenario calls for it, the hard
cases: animals crossing at speed, touching, lying on top of each other,
dropping out for several frames, entering and leaving.  Ground truth is the
animals' true positions, so every tracker decision can be scored.

One aggregate score hides exactly the failures that matter, so scenarios are
scored separately (see :mod:`yoru_tracker.evaluation.benchmark`).  All
randomness comes from the seed: the same seed gives the same scenario, frame
for frame, on every machine.

Detections are upright boxes by default, as most YORU models produce;
``obb_crossing`` uses oriented boxes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from yoru.libs.obb import normalize_angle, obb_to_aabb

from yoru_tracker.core.types import Box, Detection
from yoru_tracker.runtime.frames import FrameDetections

ARENA = (640, 480)
LENGTH = 40.0
WIDTH = 16.0
FPS = 30.0

#: (gt_id, cx, cy, heading) for every animal present in one frame.
State = Tuple[int, float, float, float]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    gt: Dict[int, List[Tuple[int, float, float]]]
    frames: List[FrameDetections]
    match_distance: float = LENGTH / 2.0
    #: Number of animals present throughout, or None where they come and go.
    population: Optional[int] = None


@dataclass(frozen=True)
class DetectorModel:
    jitter: float = 1.0           # centre noise, px
    size_jitter: float = 1.0      # px
    angle_jitter: float = 0.05    # rad
    miss_rate: float = 0.01       # independent misses per animal per frame
    obb: bool = False
    merge_distance: float = 0.0   # animals closer than this are seen as one box
    false_positive_rate: float = 0.0


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

def _heading(dx, dy, previous):
    return math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else previous


def random_walk(rng, frames, start, *, speed=3.0, turn=0.25, bounds=None) -> List[Tuple[float, float, float]]:
    """A smooth random walk that turns back at the arena wall."""
    x0, y0, x1, y1 = bounds or (LENGTH, LENGTH, ARENA[0] - LENGTH, ARENA[1] - LENGTH)
    x, y = start
    theta = rng.uniform(-math.pi, math.pi)
    out = []
    for _ in range(frames):
        theta += rng.normal(0.0, turn)
        v = max(0.0, speed + rng.normal(0.0, speed * 0.3))
        nx, ny = x + v * math.cos(theta), y + v * math.sin(theta)
        if not x0 <= nx <= x1:
            theta = math.pi - theta
            nx = min(max(nx, x0), x1)
        if not y0 <= ny <= y1:
            theta = -theta
            ny = min(max(ny, y0), y1)
        x, y = nx, ny
        out.append((x, y, theta))
    return out


def straight(frames, start, end) -> List[Tuple[float, float, float]]:
    (xa, ya), (xb, yb) = start, end
    theta = math.atan2(yb - ya, xb - xa)
    return [(xa + (xb - xa) * t / max(1, frames - 1), ya + (yb - ya) * t / max(1, frames - 1), theta)
            for t in range(frames)]


def waypoints(frames, points: Sequence[Tuple[float, float]], holds: Sequence[int] = ()):
    """Constant-speed path through *points*; ``holds[i]`` frames spent still at point i."""
    holds = list(holds) + [0] * (len(points) - len(holds))
    legs = [math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    moving = frames - sum(holds)
    total = sum(legs) or 1.0
    out = []
    heading = 0.0
    for i, (x, y) in enumerate(points):
        out.extend([(x, y, heading)] * holds[i])
        if i == len(points) - 1:
            break
        n = max(1, round(moving * legs[i] / total))
        (xa, ya), (xb, yb) = points[i], points[i + 1]
        heading = _heading(xb - xa, yb - ya, heading)
        out.extend((xa + (xb - xa) * k / n, ya + (yb - ya) * k / n, heading) for k in range(n))
    while len(out) < frames:
        out.append(out[-1])
    return out[:frames]


# ---------------------------------------------------------------------------
# Detection model
# ---------------------------------------------------------------------------

def _box(cx, cy, heading) -> Box:
    return (cx, cy, LENGTH, WIDTH, normalize_angle(heading))


def _inside(cx, cy) -> bool:
    return 0.0 <= cx < ARENA[0] and 0.0 <= cy < ARENA[1]


def _detect(rng, states: List[State], model: DetectorModel, drop: Callable[[int, int], bool],
            frame: int) -> List[Detection]:
    visible = [s for s in states if _inside(s[1], s[2])]
    groups: List[List[State]] = []
    if model.merge_distance > 0:
        for s in visible:
            for g in groups:
                if any(math.hypot(s[1] - o[1], s[2] - o[2]) < model.merge_distance for o in g):
                    g.append(s)
                    break
            else:
                groups.append([s])
    else:
        groups = [[s] for s in visible]

    dets = []
    for group in groups:
        if len(group) == 1:
            gid, cx, cy, heading = group[0]
            if drop(gid, frame) or rng.random() < model.miss_rate:
                continue
            cx += rng.normal(0.0, model.jitter)
            cy += rng.normal(0.0, model.jitter)
            w = LENGTH + rng.normal(0.0, model.size_jitter)
            h = WIDTH + rng.normal(0.0, model.size_jitter)
            angle = normalize_angle(heading + rng.normal(0.0, model.angle_jitter))
            box = (cx, cy, w, h, angle)
        else:
            # Touching animals: one box around both.
            corners = [obb_to_aabb(_box(cx, cy, hd)) for _, cx, cy, hd in group]
            x1 = min(c[0] for c in corners)
            y1 = min(c[1] for c in corners)
            x2 = max(c[2] for c in corners)
            y2 = max(c[3] for c in corners)
            box = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0.0)
        conf = float(min(0.99, max(0.3, rng.normal(0.85, 0.05))))
        if model.obb:
            dets.append(Detection.from_obb(*box, confidence=conf, class_id=0, class_name="fly"))
        else:
            dets.append(Detection.from_xyxy(*obb_to_aabb(box), confidence=conf,
                                            class_id=0, class_name="fly"))
    if model.false_positive_rate and rng.random() < model.false_positive_rate:
        cx, cy = rng.uniform(20, ARENA[0] - 20), rng.uniform(20, ARENA[1] - 20)
        dets.append(Detection.from_xyxy(cx - 12, cy - 6, cx + 12, cy + 6,
                                        confidence=0.4, class_id=0, class_name="fly"))
    return dets


def build(name: str, description: str, tracks: Dict[int, List[Optional[Tuple[float, float, float]]]],
          rng, model: DetectorModel = DetectorModel(),
          drop: Callable[[int, int], bool] = lambda gid, frame: False,
          closed: bool = True) -> Scenario:
    """Turn per-animal paths (``None`` = absent that frame) into a scenario.

    *closed*: the animals stay in the arena throughout, so the population is
    the number of paths.
    """
    n = max(len(p) for p in tracks.values())
    gt: Dict[int, List[Tuple[int, float, float]]] = {}
    frames = []
    for f in range(n):
        states = []
        for gid, path in tracks.items():
            if f < len(path) and path[f] is not None:
                x, y, heading = path[f]
                states.append((gid, x, y, heading))
        gt[f] = [(gid, x, y) for gid, x, y, _ in states if _inside(x, y)]
        frames.append(FrameDetections(f, f / FPS, tuple(_detect(rng, states, model, drop, f))))
    return Scenario(name, description, gt, frames, population=len(tracks) if closed else None)


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------

N = 200
CENTER = (ARENA[0] / 2, ARENA[1] / 2)


def _single(rng):
    return build("single", "one animal, random walk",
                 {0: random_walk(rng, N, CENTER)}, rng)


def _far_apart(rng):
    tracks = {}
    w, h = ARENA
    for i, (qx, qy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        bounds = (qx * w / 2 + LENGTH, qy * h / 2 + LENGTH,
                  (qx + 1) * w / 2 - LENGTH, (qy + 1) * h / 2 - LENGTH)
        start = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        tracks[i] = random_walk(rng, N, start, bounds=bounds)
    return build("far_apart", "four animals, each in its own quadrant", tracks, rng)


def _fast_crossing(rng):
    tracks = {
        0: straight(N, (60, 140), (580, 340)),
        1: straight(N, (60, 340), (580, 140)),
    }
    # Speed up: compress the crossing into the middle of the clip.
    fast = {k: straight(60, p[0][:2], p[-1][:2]) for k, p in tracks.items()}
    pad = (N - 60) // 2
    out = {k: [p[0]] * pad + p + [p[-1]] * (N - 60 - pad) for k, p in fast.items()}
    return build("fast_crossing", "two animals cross at ~9 px/frame", out, rng)


def _parallel(rng):
    a = straight(N, (60, 220), (580, 220))
    b = straight(N, (60, 252), (580, 252))
    return build("parallel_motion", "two animals walk side by side, 32 px apart",
                 {0: a, 1: b}, rng)


def _close_approach(rng):
    a = waypoints(N, [(120, 240), (300, 240), (300, 240), (120, 180)], holds=[0, 40, 0, 0])
    b = waypoints(N, [(520, 240), (340, 240), (340, 240), (520, 300)], holds=[0, 40, 0, 0])
    return build("close_approach", "two animals meet head to head, pause, separate",
                 {0: a, 1: b}, rng)


def _courtship(rng):
    female = random_walk(rng, N, CENTER, speed=1.5, turn=0.15)
    male = []
    for i, (x, y, theta) in enumerate(female):
        orbit = theta + math.pi + 0.8 * math.sin(i / 12.0)
        mx, my = x + 1.3 * LENGTH * math.cos(orbit), y + 1.3 * LENGTH * math.sin(orbit)
        male.append((mx, my, _heading(x - mx, y - my, theta)))
    return build("courtship", "one animal follows and circles another at ~1.3 body lengths",
                 {0: female, 1: male}, rng)


def _contact(rng):
    a = waypoints(N, [(140, 240), (306, 240), (306, 240), (160, 330)], holds=[0, 20, 0, 0])
    b = waypoints(N, [(500, 240), (334, 240), (334, 240), (480, 150)], holds=[0, 20, 0, 0])
    return build("contact", "two animals touch for ~20 frames; the detector sees one box",
                 {0: a, 1: b}, rng, DetectorModel(merge_distance=LENGTH))


def _temporary_overlap(rng):
    a = straight(N, (100, 230), (540, 250))
    b = straight(N, (540, 236), (100, 244))
    return build("temporary_overlap", "two animals pass over each other, both still detected",
                 {0: a, 1: b}, rng)


def _complete_overlap(rng):
    a = waypoints(N, [(120, 240), (320, 240), (320, 240), (520, 140)], holds=[0, 15, 0, 0])
    b = waypoints(N, [(320, 400), (320, 242), (320, 242), (130, 120)], holds=[0, 15, 0, 0])
    return build("complete_overlap", "one animal lies on another for ~15 frames, then both leave",
                 {0: a, 1: b}, rng, DetectorModel(merge_distance=LENGTH * 0.6))


def _detector_dropout(rng):
    tracks = {i: random_walk(rng, N, (160 + 160 * i, 240)) for i in range(3)}
    gaps = {0: [(40, 44), (120, 126)], 1: [(70, 73), (150, 155)], 2: [(95, 100)]}

    def drop(gid, frame):
        return any(a <= frame < b for a, b in gaps.get(gid, ()))

    return build("detector_dropout", "three animals, each missed for bursts of 3-6 frames",
                 tracks, rng, drop=drop)


def _entry(rng):
    tracks = {0: random_walk(rng, N, (200, 240)), 1: random_walk(rng, N, (440, 240))}
    tracks[2] = [None] * 60 + straight(N - 60, (-30, 120), (400, 160))
    tracks[3] = [None] * 120 + straight(N - 120, (670, 400), (420, 360))
    return build("animal_entry", "two animals present; two more walk in at frames 60 and 120",
                 tracks, rng, closed=False)


def _exit(rng):
    tracks = {i: random_walk(rng, N, (160 + 110 * i, 240)) for i in range(4)}
    tracks[0] = straight(90, (200, 240), (-40, 240)) + [None] * (N - 90)
    tracks[3] = waypoints(N, [(480, 240), (480, 240), (690, 320)], holds=[60, 0, 0])
    return build("animal_exit", "four animals; two walk out of the arena", tracks, rng,
                 closed=False)


def _high_density(rng):
    tracks = {}
    for i in range(12):
        start = (rng.uniform(60, ARENA[0] - 60), rng.uniform(60, ARENA[1] - 60))
        tracks[i] = random_walk(rng, N, start, speed=2.5)
    return build("high_density", "twelve animals in one arena, frequent near encounters",
                 tracks, rng, DetectorModel(miss_rate=0.02, false_positive_rate=0.05))


def _jump(rng):
    a = random_walk(rng, N, (200, 240), speed=2.0)
    b = waypoints(N, [(440, 140), (450, 160), (300, 380), (320, 400)], holds=[0, 0, 0, 0])
    # The second animal covers (450,160) -> (300,380) in a single frame.
    b = (straight(100, (440, 140), (450, 160)) + [(300, 380, 2.0)]
         + straight(N - 101, (300, 380), (330, 400)))
    return build("jump", "an animal jumps 270 px (about 7 body lengths) in one frame",
                 {0: a, 1: b}, rng)


def _long_contact(rng):
    a = waypoints(N, [(140, 240), (306, 240), (306, 240), (160, 330)], holds=[0, 70, 0, 0])
    b = waypoints(N, [(500, 240), (334, 240), (334, 240), (480, 150)], holds=[0, 70, 0, 0])
    return build("long_contact", "two animals stay in contact for ~70 frames (longer than "
                 "max_age), seen as one box", {0: a, 1: b}, rng,
                 DetectorModel(merge_distance=LENGTH))


def _obb_crossing(rng):
    a = waypoints(N, [(100, 120), (540, 360)])
    b = waypoints(N, [(100, 360), (540, 120)])
    return build("obb_crossing", "two animals cross diagonally; oriented-box detector",
                 {0: a, 1: b}, rng, DetectorModel(obb=True))


SCENARIOS: Dict[str, Callable] = {
    "single": _single,
    "far_apart": _far_apart,
    "fast_crossing": _fast_crossing,
    "parallel_motion": _parallel,
    "close_approach": _close_approach,
    "courtship": _courtship,
    "contact": _contact,
    "temporary_overlap": _temporary_overlap,
    "complete_overlap": _complete_overlap,
    "detector_dropout": _detector_dropout,
    "animal_entry": _entry,
    "animal_exit": _exit,
    "high_density": _high_density,
    "obb_crossing": _obb_crossing,
    "jump": _jump,
    "long_contact": _long_contact,
}


def make_scenario(name: str, seed: int = 0) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}. Available: {', '.join(SCENARIOS)}")
    # One generator per scenario, seeded from both the seed and the name, so
    # adding a scenario never changes the others.
    rng = np.random.default_rng([seed, sum(ord(c) * 31 ** i for i, c in enumerate(name)) % (2 ** 32)])
    return SCENARIOS[name](rng)
