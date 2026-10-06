"""Tracker configuration: defaults, validation, serialisation, versioning."""

from __future__ import annotations

import dataclasses
import json

import pytest

from yoru_tracker.core.config import CONFIG_VERSION, ConfigError, TrackerConfig


def test_defaults_are_valid_and_follow_the_design():
    c = TrackerConfig().validate()
    assert c.mode == "lite"
    assert c.lifecycle.min_hits == 2
    assert c.association.distance_weight == 1.0
    assert c.association.iou_weight == 1.0
    assert c.association.axis_weight == 0.25
    assert c.association.max_distance == 100.0
    assert c.kalman.enabled
    assert not c.advanced.reid and not c.advanced.occlusion_recovery


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_round_trip_through_a_file(tmp_path, suffix):
    c = TrackerConfig()
    c = dataclasses.replace(c, lifecycle=dataclasses.replace(c.lifecycle, max_age=7, population=3))
    path = c.save(tmp_path / f"tracker{suffix}")
    assert TrackerConfig.load(path) == c


def test_dict_round_trip_is_versioned():
    data = TrackerConfig().to_dict()
    assert data["config_version"] == CONFIG_VERSION
    assert TrackerConfig.from_dict(data) == TrackerConfig()
    # A bare tracker section is accepted too.
    assert TrackerConfig.from_dict(data["tracker"]) == TrackerConfig()


def test_partial_files_take_defaults_for_the_rest():
    c = TrackerConfig.from_yaml("tracker:\n  lifecycle:\n    max_age: 3\n")
    assert c.lifecycle.max_age == 3
    assert c.association == TrackerConfig().association


def test_a_misspelt_key_is_an_error_not_a_silent_default():
    with pytest.raises(ConfigError, match="max_agee"):
        TrackerConfig.from_yaml("tracker:\n  lifecycle:\n    max_agee: 3\n")


@pytest.mark.parametrize("yaml_text, fragment", [
    ("tracker:\n  lifecycle:\n    max_age: 2.5\n", "lifecycle.max_age must be int"),
    ("tracker:\n  lifecycle:\n    max_age: true\n", "lifecycle.max_age must be int"),
    ("tracker:\n  kalman:\n    enabled: 1\n", "kalman.enabled must be bool"),
    ("tracker:\n  association:\n    max_distance: far\n", "max_distance must be float"),
    ("tracker:\n  association:\n    max_distance: 0\n", "max_distance must be > 0"),
    ("tracker:\n  lifecycle:\n    min_hits: 0\n", "min_hits must be >= 1"),
    ("tracker:\n  lifecycle:\n    population: -1\n", "population must be >= 0"),
    ("tracker:\n  association:\n    min_iou: 1.5\n", "min_iou"),
    ("tracker:\n  association:\n    distance_weight: 0\n    iou_weight: 0\n", "must be positive"),
])
def test_invalid_values_are_named(yaml_text, fragment):
    with pytest.raises(ConfigError, match=fragment.replace(".", r"\.")):
        TrackerConfig.from_yaml(yaml_text)


def test_every_problem_is_reported_at_once():
    with pytest.raises(ConfigError) as exc:
        TrackerConfig.from_yaml("tracker:\n  lifecycle:\n    max_age: -1\n    min_hits: 0\n")
    assert len(exc.value.problems) == 2


def test_a_future_config_version_is_refused():
    data = TrackerConfig().to_dict()
    data["config_version"] = CONFIG_VERSION + 1
    with pytest.raises(ConfigError, match="config_version"):
        TrackerConfig.from_dict(data)


def test_advanced_options_are_not_silently_ignored_by_lite():
    with pytest.raises(ConfigError, match="Advanced tracker"):
        TrackerConfig.from_yaml("tracker:\n  mode: lite\n  advanced:\n    reid: true\n")


def test_the_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        TrackerConfig().mode = "baseline"


def test_json_is_plain_data():
    json.dumps(TrackerConfig().to_dict())
