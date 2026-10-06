# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Detection through YORU: load a model, turn its output into ``Detection``.

YORU-Tracker has no detector code of its own.  Every model YORU can load --
YOLOv5 checkpoints from YORU v1, YOLOv8/11, RT-DETR, torchvision, ONNX -- is
loaded by ``yoru.libs.plugins.get_detector`` and its output normalised here,
so nothing downstream ever sees a detector-specific object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, Tuple

from yoru_tracker.core.types import Detection


class Detector(Protocol):
    """Anything that turns a BGR image into detections."""

    names: Mapping[int, str]

    def detect(self, frame) -> List[Detection]: ...


@dataclass(frozen=True)
class DetectorConfig:
    model_path: str = ""
    backend: str = "auto"
    conf_thresh: float = 0.25
    iou_thresh: float = 0.45
    #: Class IDs dropped before tracking.
    exclude_classes: Tuple[int, ...] = field(default=())

    def to_dict(self) -> dict:
        data = asdict(self)
        data["exclude_classes"] = list(self.exclude_classes)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectorConfig":
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown detector setting(s): {', '.join(sorted(unknown))}")
        values = dict(data)
        if "exclude_classes" in values:
            values["exclude_classes"] = tuple(int(c) for c in values["exclude_classes"])
        return cls(**values)


class YoruDetector:
    """A YORU detector plugin behind the :class:`Detector` protocol."""

    def __init__(self, backend_detector, config: DetectorConfig):
        self._detector = backend_detector
        self.config = config
        self._exclude = frozenset(config.exclude_classes)

    @classmethod
    def load(cls, config: DetectorConfig) -> "YoruDetector":
        """Load ``config.model_path``.  Errors propagate: there is no fallback model."""
        if not config.model_path:
            raise ValueError("No detector model selected")
        from yoru.libs.plugins import get_detector

        detector = get_detector(
            config.backend, config.model_path,
            conf_thresh=config.conf_thresh, iou_thresh=config.iou_thresh,
        )
        return cls(detector, config)

    @property
    def names(self) -> Dict[int, str]:
        return dict(self._detector.names)

    def detect(self, frame) -> List[Detection]:
        out = []
        for raw in self._detector.detect(frame):
            det = Detection.from_yoru(raw)
            if det.class_id not in self._exclude:
                out.append(det)
        return out
