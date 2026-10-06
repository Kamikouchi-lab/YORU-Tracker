# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""What the views share: the tracker settings, the detector, remembered paths.

Remembered between sessions in ``~/.yoru/yoru_tracker_gui.json`` (beside
YORU's own state file): the last model, folders, and tracker settings.  A
corrupt or unreadable file is logged and replaced by defaults -- it only
holds conveniences, never anything a result depends on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, List, Optional

from yoru_tracker.core.config import ConfigError, TrackerConfig
from yoru_tracker.runtime.detection import DetectorConfig, YoruDetector

logger = logging.getLogger(__name__)

STATE_FILENAME = "yoru_tracker_gui.json"


def state_file() -> Path:
    from yoru.libs.user_paths import get_yoru_home

    return get_yoru_home() / STATE_FILENAME


@dataclass
class AppState:
    tracker_config: TrackerConfig = field(default_factory=TrackerConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    output_dir: str = ""
    last_video: str = ""
    last_folder: str = ""
    camera_id: int = 0
    _listeners: List[Callable[[TrackerConfig], None]] = field(default_factory=list, repr=False)

    # -- tracker settings -------------------------------------------------

    def set_tracker_config(self, config: TrackerConfig) -> None:
        config = config.validate()
        if config == self.tracker_config:
            return
        self.tracker_config = config
        logger.info("Tracker settings changed (mode %s)", config.mode)
        for listener in list(self._listeners):
            listener(config)

    def on_tracker_config(self, callback: Callable[[TrackerConfig], None]) -> None:
        self._listeners.append(callback)

    # -- detector -------------------------------------------------------

    def set_detector(self, **changes) -> None:
        self.detector = replace(self.detector, **changes)

    def load_detector(self) -> YoruDetector:
        return YoruDetector.load(self.detector)

    # -- persistence ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "tracker_config": self.tracker_config.to_dict(),
            "detector": self.detector.to_dict(),
            "output_dir": self.output_dir,
            "last_video": self.last_video,
            "last_folder": self.last_folder,
            "camera_id": self.camera_id,
        }

    def save(self, path: Optional[Path] = None) -> None:
        try:
            (path or state_file()).write_text(json.dumps(self.to_dict(), indent=2),
                                              encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not save GUI state: %s", exc)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppState":
        path = path or state_file()
        state = cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return state
        except (OSError, ValueError) as exc:
            logger.warning("GUI state %s unreadable, using defaults: %s", path, exc)
            return state
        try:
            state.tracker_config = TrackerConfig.from_dict(data.get("tracker_config", {}))
        except ConfigError as exc:
            logger.warning("Saved tracker settings rejected, using defaults: %s", exc)
        try:
            state.detector = DetectorConfig.from_dict(data.get("detector", {}))
        except (TypeError, ValueError) as exc:
            logger.warning("Saved detector settings rejected: %s", exc)
        state.output_dir = str(data.get("output_dir", ""))
        state.last_video = str(data.get("last_video", ""))
        state.last_folder = str(data.get("last_folder", ""))
        try:
            state.camera_id = int(data.get("camera_id", 0))
        except (TypeError, ValueError):
            state.camera_id = 0
        return state
