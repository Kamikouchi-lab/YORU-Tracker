# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The JSON written beside every tracks CSV: how to get the same tracks again.

It records the tracker (name, version, API version), its full configuration,
the detector settings, the input, and the software versions.  Handing the
file back to :func:`config_from_metadata` returns the exact configuration,
and with the saved detections CSV the run can be repeated without a GPU.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from yoru_tracker.core.config import TrackerConfig
from yoru_tracker.core.tracker_base import TRACKER_API_VERSION, TrackerBase
from yoru_tracker.export.csv_export import TRACK_COLUMNS

METADATA_VERSION = 1


def _yoru_version() -> str:
    try:
        import yoru

        return str(yoru.__version__)
    except Exception:  # pragma: no cover - yoru is a hard dependency
        return "unknown"


def build_metadata(
    tracker: TrackerBase,
    *,
    detector: Optional[Mapping[str, Any]] = None,
    source: Optional[Mapping[str, Any]] = None,
    outputs: Optional[Mapping[str, Any]] = None,
    summary: Optional[Mapping[str, Any]] = None,
) -> dict:
    from yoru_tracker import __version__

    return {
        "metadata_version": METADATA_VERSION,
        "created_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "software": {
            "yoru_tracker": __version__,
            "yoru": _yoru_version(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "tracker": {
            **tracker.info.to_dict(),
            "tracker_api_version": TRACKER_API_VERSION,
        },
        "tracker_config": tracker.config.to_dict(),
        "detector": dict(detector) if detector is not None else None,
        "source": dict(source) if source is not None else None,
        "outputs": dict(outputs) if outputs is not None else None,
        "summary": dict(summary) if summary is not None else None,
        "track_columns": list(TRACK_COLUMNS),
    }


def write_metadata(path, metadata: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_metadata(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_from_metadata(metadata: Mapping[str, Any]) -> TrackerConfig:
    """The tracker configuration a result was produced with."""
    return TrackerConfig.from_dict(metadata["tracker_config"])
