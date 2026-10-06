# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Tracker configuration: one versioned, validated, serialisable value.

The configuration is saved next to every result (see
:mod:`yoru_tracker.export.metadata`), so a tracking run can be repeated from
its output alone.  On disk it looks like::

    config_version: 1
    tracker:
      mode: lite
      lifecycle:
        max_age: 10
        min_hits: 2
        population: 0
      association:
        distance_weight: 1.0
        iou_weight: 1.0
        axis_weight: 0.25
        max_distance: 100
        ...

Loading is strict.  An unknown key, a value of the wrong type or a value out
of range is an error naming the offending key -- a misspelt ``max_age`` that
silently fell back to its default would change every ID in the output.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: Version of the on-disk layout below.  Bump it when a key is renamed or
#: its meaning changes, and teach :meth:`TrackerConfig.from_dict` the old one.
CONFIG_VERSION = 1

__all__ = [
    "AdvancedConfig",
    "AssociationConfig",
    "CONFIG_VERSION",
    "ConfigError",
    "KalmanConfig",
    "LifecycleConfig",
    "TrackerConfig",
]


class ConfigError(ValueError):
    """The configuration is invalid; ``problems`` lists every reason."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class LifecycleConfig:
    #: Updates a confirmed track may go unmatched before it is retired.  While
    #: unmatched it is LOST: still predicted, still recoverable.  10 rather
    #: than 5: on the benchmark, 5 loses the ID of every animal the detector
    #: misses for six frames, and 10 costs nothing elsewhere.
    max_age: int = 10
    #: Matches a new track needs before it is given an ID.  1 confirms on
    #: first sight.
    min_hits: int = 2
    #: Number of animals that are always present, when known (a closed
    #: chamber with two flies: 2).  0 means unknown: animals may enter and
    #: leave, and a track unseen for max_age is retired.  With a known
    #: number, a track is never retired, a detection nobody claims goes to
    #: the nearest lost track however far away it is (a fly that jumped), and
    #: no more IDs than this are ever given out.
    population: int = 0


@dataclass(frozen=True)
class AssociationConfig:
    """How detections are matched to tracks.

    The cost of a candidate pair is
    ``distance_weight * d/gate + iou_weight * (1 - IoU) + axis_weight * axis
    + size_weight * size``, each term in [0, 1].  A pair costing
    ``distance_weight + iou_weight`` or more is not made.  None of the weights
    is a universal constant; benchmark before changing them.
    """

    distance_weight: float = 1.0
    iou_weight: float = 1.0
    #: Long-axis disagreement, weighted by how elongated both boxes are -- a
    #: near-square box has no axis to disagree about.
    axis_weight: float = 0.25
    #: Log area ratio.  Off by default.
    size_weight: float = 0.0
    #: Gate: a detection further than this (pixels) from where a track is
    #: predicted to be cannot be matched to it.
    max_distance: float = 100.0
    #: Gate: a pair with IoU below this whose centres are more than half a
    #: body length apart cannot be matched.  0 disables it.
    min_iou: float = 0.0
    #: Never match a track to a detection of another class.  Off by default:
    #: YORU classes are often behaviours ("solo", "copulation"), which an
    #: animal moves between while staying the same animal.  Turn it on when
    #: classes are different kinds of animal that must never share an ID.
    class_aware: bool = False
    #: Run the recovery pass, which looks for lost tracks with a gate that
    #: widens with the time they have been missing.
    recovery: bool = True
    #: Upper bound of that widening gate, in multiples of max_distance.
    recovery_gate_scale: float = 3.0


@dataclass(frozen=True)
class KalmanConfig:
    """Constant-velocity motion model on the box centre.

    Both noise levels are relative to the box size (the square root of its
    area), so one setting serves a 20 px fly and a 200 px mouse.  Disabled,
    a track is predicted to stay where it was last seen.
    """

    enabled: bool = True
    #: Acceleration noise, in box sizes per frame squared.
    process_noise: float = 0.2
    #: Detector centre noise, in box sizes.  What matters most is the ratio
    #: of the two: with process noise ten times this, the velocity follows
    #: every jitter of the detector and animals crossing at speed swap IDs;
    #: at twice this it is smooth enough to carry them through.
    measurement_noise: float = 0.1


@dataclass(frozen=True)
class AdvancedConfig:
    """Options of the Advanced tracker.  Not accepted by any other mode."""

    reid: bool = False
    occlusion_recovery: bool = False


@dataclass(frozen=True)
class TrackerConfig:
    mode: str = "lite"
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    #: Write every track event (created, lost, recovered, retired) to the
    #: log.  Off by default: it is a line per event.
    log_events: bool = False

    # -- validation -----------------------------------------------------

    def problems(self) -> list:
        """Every reason this configuration is invalid; empty when it is fine."""
        out = []
        if not isinstance(self.mode, str) or not self.mode.strip():
            out.append("mode must be a non-empty string")
        lc, ac, kc = self.lifecycle, self.association, self.kalman
        if lc.max_age < 0:
            out.append("lifecycle.max_age must be >= 0")
        if lc.min_hits < 1:
            out.append("lifecycle.min_hits must be >= 1")
        if lc.population < 0:
            out.append("lifecycle.population must be >= 0 (0 = not known)")
        for name in ("distance_weight", "iou_weight", "axis_weight", "size_weight"):
            if getattr(ac, name) < 0:
                out.append(f"association.{name} must be >= 0")
        if ac.distance_weight + ac.iou_weight <= 0:
            # Their sum is also the cost above which a pair is not made.
            out.append("association: distance_weight + iou_weight must be positive")
        if ac.max_distance <= 0:
            out.append("association.max_distance must be > 0 (pixels)")
        if not 0.0 <= ac.min_iou <= 1.0:
            out.append("association.min_iou must be within [0, 1]")
        if ac.recovery_gate_scale < 1.0:
            out.append("association.recovery_gate_scale must be >= 1")
        if kc.process_noise <= 0:
            out.append("kalman.process_noise must be > 0")
        if kc.measurement_noise <= 0:
            out.append("kalman.measurement_noise must be > 0")
        if self.mode != "advanced":
            for name in ("reid", "occlusion_recovery"):
                if getattr(self.advanced, name):
                    out.append(
                        f"advanced.{name} is an option of the Advanced tracker; "
                        f"mode {self.mode!r} does not provide it"
                    )
        return out

    def validate(self) -> "TrackerConfig":
        problems = self.problems()
        if problems:
            raise ConfigError(problems)
        return self

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        return {"config_version": CONFIG_VERSION, "tracker": dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackerConfig":
        """Inverse of :meth:`to_dict`; a bare ``tracker`` mapping is accepted too."""
        if not isinstance(data, Mapping):
            raise ConfigError([f"expected a mapping, got {type(data).__name__}"])
        if "tracker" in data:
            extra = sorted(set(data) - {"tracker", "config_version"})
            if extra:
                raise ConfigError([f"unknown top-level key(s): {', '.join(extra)}"])
            version = data.get("config_version", CONFIG_VERSION)
            if version != CONFIG_VERSION:
                raise ConfigError([
                    f"config_version {version!r} is not supported "
                    f"(this YORU Tracker reads version {CONFIG_VERSION})"
                ])
            data = data["tracker"]
        problems = []
        config = _build(cls, data, "", problems)
        if problems:
            raise ConfigError(problems)
        return config.validate()

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> "TrackerConfig":
        import yaml

        return cls.from_dict(yaml.safe_load(text) or {})

    def save(self, path) -> Path:
        """Write YAML, or JSON when *path* ends in ``.json``."""
        path = Path(path)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        else:
            path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "TrackerConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return cls.from_dict(json.loads(text))
        return cls.from_yaml(text)


def _build(cls, data, prefix, problems):
    """Construct dataclass *cls* from *data*, collecting every problem."""
    if not isinstance(data, Mapping):
        problems.append(f"{prefix or 'tracker'} must be a mapping")
        return cls()
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    for key in unknown:
        problems.append(f"unknown key {prefix}{key}")
    kwargs = {}
    for name, f in fields.items():
        if name not in data:
            continue
        value = data[name]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        key = f"{prefix}{name}"
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value, f"{key}.", problems)
            continue
        expected = type(default)
        if expected is bool:
            ok = isinstance(value, bool)
        elif expected is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif expected is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            value = float(value) if ok else value
        else:
            ok = isinstance(value, expected)
        if not ok:
            problems.append(
                f"{key} must be {expected.__name__}, got {type(value).__name__} {value!r}"
            )
            continue
        kwargs[name] = value
    return cls(**kwargs)
