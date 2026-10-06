"""The public tracker API: types, the base class, the registry, version checks."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from yoru.libs.detector_base import DETECTION_COLUMNS

from yoru_tracker import TRACKER_API_VERSION
from yoru_tracker.core import (
    Detection,
    IncompatibleTrackerError,
    TrackerBase,
    TrackerConfig,
    TrackerInfo,
    TrackerUnavailableError,
    TrackState,
    available_modes,
    create_tracker,
)
from yoru_tracker.core import registry
from yoru_tracker.tracking.lite_tracker import LiteTracker

from conftest import det


def test_api_version_is_one():
    assert TRACKER_API_VERSION == 1
    assert LiteTracker.api_version == TRACKER_API_VERSION


def test_tracker_base_is_abstract():
    with pytest.raises(TypeError):
        TrackerBase()


def test_update_signature_matches_the_design():
    import inspect

    params = list(inspect.signature(TrackerBase.update).parameters)
    assert params == ["self", "detections", "frame_id", "timestamp", "frame"]


def test_lite_declares_its_capabilities():
    info = LiteTracker.info
    assert isinstance(info, TrackerInfo)
    assert info.realtime_capable and not info.requires_frame and not info.requires_gpu
    assert info.supports_obb and info.deterministic


def test_builtin_modes():
    modes = available_modes()
    assert modes[:2] == ["lite", "baseline"]


def test_create_tracker_follows_the_mode():
    assert isinstance(create_tracker(TrackerConfig(mode="lite")), LiteTracker)
    assert create_tracker(TrackerConfig(mode="baseline")).info.name.startswith("YORU baseline")


def test_advanced_is_refused_explicitly_not_replaced_by_lite():
    with pytest.raises(TrackerUnavailableError, match="Advanced"):
        create_tracker(TrackerConfig(mode="advanced"))


def test_unknown_mode_lists_what_is_available():
    with pytest.raises(TrackerUnavailableError, match="lite"):
        create_tracker(TrackerConfig(mode="no-such-tracker"))


class _OldPlugin(TrackerBase):
    api_version = 99
    info = TrackerInfo("old", "0", 99)

    def reset(self):
        pass

    def update(self, detections, frame_id, timestamp=None, frame=None):
        raise AssertionError("must never run")


def test_a_plugin_for_another_api_version_is_refused():
    with pytest.raises(IncompatibleTrackerError, match="requires tracker API v99"):
        registry.check_compatible(_OldPlugin, "old")


def test_entry_point_plugins_are_discovered_and_checked(monkeypatch):
    class _EP:
        name = "old"
        value = "tests:_OldPlugin"

        def load(self):
            return _OldPlugin

    monkeypatch.setattr(registry, "_plugin_entry_points", lambda: {"old": _EP()})
    assert "old" in available_modes()
    with pytest.raises(IncompatibleTrackerError):
        create_tracker(TrackerConfig(mode="old"))


def test_a_plugin_that_fails_to_import_says_why(monkeypatch):
    class _EP:
        name = "broken"
        value = "broken.plugin:Tracker"

        def load(self):
            raise ImportError("no module named broken")

    monkeypatch.setattr(registry, "_plugin_entry_points", lambda: {"broken": _EP()})
    with pytest.raises(TrackerUnavailableError, match="no module named broken"):
        create_tracker(TrackerConfig(mode="broken"))


# -- data types ---------------------------------------------------------------

def test_detection_from_a_yoru_dict_without_obb():
    d = Detection.from_yoru({"x1": 10, "y1": 20, "x2": 50, "y2": 40, "conf": 0.8,
                             "class_id": 1, "class_name": "solo"})
    assert d.box == (30.0, 30.0, 40.0, 20.0, 0.0)
    assert (d.class_id, d.class_name, d.confidence) == (1, "solo", 0.8)


def test_detection_from_a_yoru_dict_with_obb():
    d = Detection.from_yoru({"x1": 0, "y1": 0, "x2": 60, "y2": 60, "conf": 0.5,
                             "class_id": 0, "class_name": "fly",
                             "cx": 30, "cy": 30, "w": 40, "h": 10, "angle": 0.5})
    assert d.box == (30.0, 30.0, 40.0, 10.0, 0.5)


def test_detection_from_a_yoru_row():
    row = [10, 20, 50, 40, 0.7, 2, "x", 1.25, 30, 30, 40, 20, 0.0]
    assert len(row) == len(DETECTION_COLUMNS)
    d = Detection.from_row(row)
    assert d.class_id == 2 and d.center == (30.0, 30.0)


@pytest.mark.parametrize("bad", [
    dict(w=0.0), dict(h=-1.0), dict(cx=math.nan), dict(angle=math.inf), dict(confidence=math.nan),
])
def test_invalid_detections_are_recognised(bad):
    d = dataclasses.replace(det(10, 10), **bad)
    assert not d.is_valid()


def test_results_are_plain_serialisable_values():
    tracker = create_tracker()
    result = tracker.update([det(10, 10), det(100, 10)], 0, timestamp=0.0)
    text = json.dumps(result.to_dict())
    assert '"track_state": "active"' in text
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.tracked[0].track_id = 5
    # No filter, history or other internals reachable from the output.
    for field in dataclasses.fields(result.tracked[0]):
        value = getattr(result.tracked[0], field.name)
        assert isinstance(value, (int, float, str, bool, tuple, TrackState, Detection, type(None)))


def test_tracks_summary_includes_tentative_tracks():
    tracker = create_tracker()
    tracker.update([det(10, 10)], 0)
    tracker.update([det(12, 10), det(300, 300)], 1)  # second one is new, tentative
    summary = tracker.tracks()
    assert [t.track_id for t in summary] == [0, None]
    assert summary[1].track_state is TrackState.TENTATIVE
