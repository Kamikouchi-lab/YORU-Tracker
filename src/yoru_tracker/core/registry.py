# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Tracker modes: the built-in ones and any installed as plugins.

A plugin package registers a :class:`TrackerBase` subclass under the entry
point group ``yoru_tracker.backends``::

    [project.entry-points."yoru_tracker.backends"]
    my_tracker = "my_tracker.plugin:MyTracker"

Nothing here falls back.  Asking for a mode that is not installed, failed to
import, or was written against another tracker API version is an error that
says which of those it was -- a tracker silently swapped for another would
hand out different IDs for the same animals.
"""

from __future__ import annotations

import importlib
import logging
from importlib import metadata
from typing import Dict, List, Optional, Type

from yoru_tracker.core.config import TrackerConfig
from yoru_tracker.core.tracker_base import TRACKER_API_VERSION, TrackerBase

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "yoru_tracker.backends"

#: Built-in modes, imported on first use.
_BUILTIN: Dict[str, str] = {
    "lite": "yoru_tracker.tracking.lite_tracker:LiteTracker",
    "baseline": "yoru_tracker.tracking.baseline:BaselineTracker",
}

#: Modes that are part of the design but not of this release.  Asking for
#: one says so instead of reporting an unknown name.
_PLANNED: Dict[str, str] = {
    "advanced": (
        "The Advanced tracker (Re-ID, occlusion-aware recovery) is planned but "
        "not part of this release of YORU Tracker; use mode 'lite'."
    ),
}


class TrackerUnavailableError(RuntimeError):
    """The requested tracker mode cannot be created."""


class IncompatibleTrackerError(TrackerUnavailableError):
    """A plugin was written against another tracker API version."""


def _load(target: str):
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _plugin_entry_points() -> Dict[str, metadata.EntryPoint]:
    try:
        eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - broken metadata in the environment
        logger.exception("Could not read %s entry points", ENTRY_POINT_GROUP)
        return {}
    return {ep.name: ep for ep in eps}


def available_modes() -> List[str]:
    """Modes :func:`create_tracker` can build: built-ins first, then plugins."""
    plugins = sorted(n for n in _plugin_entry_points() if n not in _BUILTIN)
    return [*_BUILTIN, *plugins]


def planned_modes() -> Dict[str, str]:
    """Modes that exist in the design only, with the reason they are missing."""
    return dict(_PLANNED)


def check_compatible(cls, name: str) -> None:
    """Raise unless *cls* is a tracker written for this API version."""
    if not (isinstance(cls, type) and issubclass(cls, TrackerBase)):
        raise TrackerUnavailableError(
            f"Tracker {name!r} ({cls!r}) is not a TrackerBase subclass"
        )
    required = getattr(cls, "api_version", None)
    if required != TRACKER_API_VERSION:
        raise IncompatibleTrackerError(
            f"Tracker {name!r} requires tracker API v{required}; "
            f"this application provides v{TRACKER_API_VERSION}"
        )
    if getattr(cls, "info", None) is None:
        raise TrackerUnavailableError(f"Tracker {name!r} declares no TrackerInfo")


def get_tracker_class(mode: str) -> Type[TrackerBase]:
    if mode in _BUILTIN:
        cls = _load(_BUILTIN[mode])
    else:
        ep = _plugin_entry_points().get(mode)
        if ep is None:
            if mode in _PLANNED:
                raise TrackerUnavailableError(_PLANNED[mode])
            raise TrackerUnavailableError(
                f"Unknown tracker mode {mode!r}. Available: {', '.join(available_modes())}"
            )
        try:
            cls = ep.load()
        except Exception as exc:
            raise TrackerUnavailableError(
                f"Tracker plugin {mode!r} ({ep.value}) failed to load: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    check_compatible(cls, mode)
    return cls


def create_tracker(config: Optional[TrackerConfig] = None) -> TrackerBase:
    """Build the tracker that ``config.mode`` names."""
    config = config or TrackerConfig()
    tracker = get_tracker_class(config.mode)(config)
    logger.debug("Tracker initialized: %s %s (mode %s)",
                tracker.info.name, tracker.info.version, config.mode)
    return tracker
