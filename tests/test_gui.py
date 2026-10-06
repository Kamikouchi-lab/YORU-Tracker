"""The GUI: every screen builds, switches and reads its settings back.

The in-process tests need a DearPyGui context but no window.  The native
test opens the real window for a few frames; it is opt-in
(``YORU_TRACKER_GUI_TESTS=1``) because it needs a desktop session.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import threading

import dearpygui.dearpygui as dpg
import pytest

from yoru_tracker.core.config import TrackerConfig
from yoru_tracker.gui.main_window import TrackerApp
from yoru_tracker.gui.state import AppState
from yoru_tracker.gui import tracker_settings


@pytest.fixture
def app(tmp_path):
    dpg.create_context()
    application = TrackerApp(AppState())
    application.build()
    yield application
    for view in (application.video, application.realtime, application.batch):
        view.shutdown()   # not app.shutdown(): that would save state to the real ~/.yoru
    dpg.destroy_context()


def test_every_view_builds_and_switches(app):
    for name in ("home", "video", "realtime", "batch", "home"):
        app.show_view(name)
        assert app.current == name
        shown = [n for n in app.views if dpg.is_item_shown(
            "view_home" if n == "home" else f"view_{n}")]
        assert shown == [name]
        app.tick()


def test_home_offers_the_three_workflows_and_settings(app):
    labels = {dpg.get_item_label(i) for i in dpg.get_all_items()
              if dpg.get_item_type(i) == "mvAppItemType::mvButton"}
    assert {"Realtime Tracking", "Video Tracking", "Batch Tracking", "Settings"} <= labels


def test_settings_round_trip(app):
    config = TrackerConfig()
    config = dataclasses.replace(
        config, lifecycle=dataclasses.replace(config.lifecycle, max_age=7, population=2),
        association=dataclasses.replace(config.association, max_distance=55.5, iou_weight=0.1))
    app.settings.show_config(config)
    assert app.settings.read_config() == config


def test_applying_invalid_settings_is_refused_with_a_reason(app):
    app.settings.show_config(TrackerConfig())
    dpg.set_value(tracker_settings._tag("association", "max_distance"), 0.0)
    assert not app.settings.apply()
    assert "max_distance" in dpg.get_value("setting_status")
    assert app.state.tracker_config == TrackerConfig()


def test_applied_settings_reach_the_views(app):
    app.show_view("video")
    app.settings.show_config(TrackerConfig())
    dpg.set_value(tracker_settings._tag("lifecycle", "max_age"), 4)
    assert app.settings.apply()
    assert app.state.tracker_config.lifecycle.max_age == 4
    assert "max_age 4" in dpg.get_value("video_tracker_summary")


def test_video_view_opens_and_scrubs_a_video(app, video_file):
    app.show_view("video")
    app.video.open_video(str(video_file))
    assert dpg.get_item_configuration("video_slider")["max_value"] == 39
    app.video.goto(10, user=True)
    app.tick()
    assert dpg.get_value("video_frame_label").startswith("Frame 11/40")


def test_state_survives_a_restart(tmp_path):
    state = AppState()
    state.set_detector(model_path="model.pt", conf_thresh=0.4)
    state.set_tracker_config(TrackerConfig.from_yaml("tracker:\n  lifecycle:\n    population: 3\n"))
    state.output_dir = "out"
    path = tmp_path / "state.json"
    state.save(path)
    loaded = AppState.load(path)
    assert loaded.detector.model_path == "model.pt" and loaded.detector.conf_thresh == 0.4
    assert loaded.tracker_config.lifecycle.population == 3
    assert loaded.output_dir == "out"


def test_a_corrupt_state_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    assert AppState.load(path).tracker_config == TrackerConfig()


@pytest.mark.parametrize("text", ["null", "[]", "3"])
def test_a_state_file_that_is_not_an_object_falls_back_to_defaults(tmp_path, text):
    path = tmp_path / "state.json"
    path.write_text(text)
    assert AppState.load(path).tracker_config == TrackerConfig()


def test_detector_settings_are_fixed_when_a_run_starts():
    state = AppState()
    state.set_detector(model_path="model.pt", conf_thresh=0.4)
    _load, settings = state.detector_loader()
    state.set_detector(conf_thresh=0.7)       # changed while the model loads
    assert settings["conf_thresh"] == 0.4


def test_the_open_video_is_kept_while_it_is_busy(app, video_file, tmp_path):
    from conftest import write_video

    app.show_view("video")
    app.video.open_video(str(video_file))
    errors = []
    app.report_error = lambda context, exc: errors.append(context)
    gate = threading.Event()
    app.video._start_task("re-track", gate.wait, lambda result: None)
    try:
        app.video.open_video(str(write_video(tmp_path / "other.avi", frames=5)))
        assert dpg.get_value("video_path") == str(video_file)
        dpg.set_value("video_vflip", True)
        app.video._flips_changed()
        assert dpg.get_value("video_vflip") is False and not app.video.source.v_flip
        assert len(errors) == 2
    finally:
        gate.set()
        app.video._task.join(timeout=5)
    app.tick()
    assert app.video._task is None


def test_scrubbing_back_shows_the_frame_the_results_belong_to(app, tmp_path, monkeypatch,
                                                              late_seeks):
    import time

    import numpy as np

    from conftest import BlobDetector, read_all, write_video

    path = write_video(tmp_path / "v.avi", frames=60)
    frames, _ = read_all(path)
    monkeypatch.setattr(app.state, "detector_loader", lambda: (BlobDetector, {}))
    app.state.set_detector(model_path="blobs")
    app.show_view("video")
    app.video.open_video(str(path))
    app.video.start()
    deadline = time.time() + 20
    while app.video.job.running and time.time() < deadline:
        app.tick()
    app.tick()
    assert app.video.run.complete
    for index in (50, 10, 33):                  # each a seek, which lands 3 frames late
        app.video.goto(index, user=True)
        app.tick()
        assert np.array_equal(app.video._frame_cache[1], frames[index]), index


def test_the_live_event_log_has_the_events_of_frames_it_never_drew(app, tmp_path, monkeypatch):
    import time

    from conftest import BlobDetector, write_video

    # Long enough not to loop (and start the blobs over) during the test.
    video = write_video(tmp_path / "live.avi", frames=400, fps=200.0)
    monkeypatch.setattr(app.state, "detector_loader", lambda: (BlobDetector, {}))
    app.state.set_detector(model_path="blobs")
    app.show_view("realtime")
    dpg.set_value("rt_source", "Video file (real time)")
    dpg.set_value("rt_video", str(video))
    app.realtime.start()
    rt = app.realtime.rt
    deadline = time.time() + 10
    while rt.stats().processed_frames < 10 and rt.running and time.time() < deadline:
        time.sleep(0.01)
    # The first frame, where both blobs got their IDs, was never drawn.
    app.realtime.tick(time.perf_counter())
    app.realtime.stop()
    assert rt.error is None
    created = [line for line in dpg.get_value("rt_events").splitlines() if "created" in line]
    assert any("#0" in line for line in created) and any("#1" in line for line in created)


@pytest.mark.gui
@pytest.mark.skipif(os.environ.get("YORU_TRACKER_GUI_TESTS") != "1",
                    reason="set YORU_TRACKER_GUI_TESTS=1 to open a real window")
@pytest.mark.parametrize("view", ["home", "video", "realtime", "batch"])
def test_the_real_window_opens_and_closes(view, tmp_path):
    shot = tmp_path / f"{view}.png"
    proc = subprocess.run(
        [sys.executable, "-m", "yoru_tracker", "gui", "--view", view,
         "--exit-after", "40", "--screenshot", str(shot)],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert shot.is_file() and shot.stat().st_size > 0
