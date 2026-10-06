# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The public tracking API: data types, configuration, the tracker contract."""

from yoru_tracker.core.capabilities import TrackerInfo
from yoru_tracker.core.config import ConfigError, TrackerConfig
from yoru_tracker.core.registry import (
    IncompatibleTrackerError,
    TrackerUnavailableError,
    available_modes,
    create_tracker,
)
from yoru_tracker.core.tracker_base import TRACKER_API_VERSION, TrackerBase
from yoru_tracker.core.types import (
    Detection,
    EventKind,
    Track,
    TrackedDetection,
    TrackEvent,
    TrackingResult,
    TrackState,
)

__all__ = [
    "ConfigError",
    "Detection",
    "EventKind",
    "IncompatibleTrackerError",
    "TRACKER_API_VERSION",
    "Track",
    "TrackedDetection",
    "TrackerBase",
    "TrackerConfig",
    "TrackerInfo",
    "TrackerUnavailableError",
    "TrackEvent",
    "TrackingResult",
    "TrackState",
    "available_modes",
    "create_tracker",
]
