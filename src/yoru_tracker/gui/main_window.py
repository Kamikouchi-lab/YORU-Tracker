# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The YORU Tracker window: start screen, the three workflows, settings.

One viewport, one view at a time -- Realtime, Video or Batch tracking, each
filling the window -- chosen from the start screen or the Go menu.  The
window shell (screen-fitted size, Japanese-capable font, the Window menu) is
YORU's ``GuiSession``; the screens are the tracker's own, built from YORU's
primitives rather than from any YORU screen.

DearPyGui callbacks are run on this thread, from the render loop, so they
can touch the views' state without locks.  Long work -- loading a model,
tracking a video, a batch -- runs on worker threads that the views poll.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import dearpygui.dearpygui as dpg
import numpy as np
from yoru.libs.gui_error import GuiErrorMixin

from yoru_tracker import TRACKER_API_VERSION, __version__
from yoru_tracker.gui import widgets
from yoru_tracker.gui.batch_view import BatchView
from yoru_tracker.gui.realtime_view import RealtimeView
from yoru_tracker.gui.state import AppState
from yoru_tracker.gui.theme import apply_tracker_theme
from yoru_tracker.gui.tracker_settings import TrackerSettingsWindow
from yoru_tracker.gui.video_view import VideoView
from yoru_tracker.logs import log_exception, log_file, setup_logging

logger = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parents[1] / "assets"
HOME = "view_home"
HOME_W = 620
HOME_H = 800
LOGO_W = 500

_WORKFLOWS = (
    ("realtime", "Realtime Tracking",
     "Camera, detector and tracker live, with IDs, trails and latency."),
    ("video", "Video Tracking",
     "Track a video, step through every frame, inspect tracks, export CSV."),
    ("batch", "Batch Tracking",
     "Track every video in a folder with one model and one set of settings."),
)


class TrackerApp(GuiErrorMixin):
    def __init__(self, state: Optional[AppState] = None):
        self.state = state or AppState.load()
        self.current = None
        self.session = None
        self.backends = ["auto"]
        self._backend_task = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Create every window.  Needs a DearPyGui context, not a viewport."""
        self.themes = apply_tracker_theme()
        with dpg.texture_registry() as registry:
            self._load_logo(registry)
            self.video = VideoView(self)
            self.realtime = RealtimeView(self)
            self.batch = BatchView(self)
            self._build_home()
            self.video.build(registry)
            self.realtime.build(registry)
            self.batch.build(registry)
        self.views = {"home": None, "video": self.video, "realtime": self.realtime,
                      "batch": self.batch}
        self.settings = TrackerSettingsWindow(self.state, self.themes, self.report_error)
        self.settings.build()
        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=lambda s, key: self._on_key(key))
        self.state.on_tracker_config(self._tracker_config_changed)

    def _load_logo(self, registry) -> None:
        path = ASSETS / "logo_dark.png"
        try:
            from PIL import Image

            image = Image.open(path).convert("RGBA")
            data = (np.asarray(image, dtype=np.float32) / 255.0).ravel()
            self.logo_size = image.size
        except Exception as exc:  # the start screen still works without it
            log_exception(f"Could not load the logo {path}", exc)
            self.logo_size = (4, 4)
            data = np.zeros(4 * 4 * 4, dtype=np.float32)
        dpg.add_static_texture(self.logo_size[0], self.logo_size[1], data,
                               tag="logo_texture", parent=registry)

    def _build_home(self) -> None:
        import yoru

        with dpg.window(tag=HOME, show=False, no_title_bar=True, no_move=True,
                        no_resize=True, no_collapse=True):
            dpg.bind_item_theme(HOME, self.themes.window)
            with dpg.child_window(tag="home_content", width=HOME_W, height=HOME_H, border=False,
                                  no_scrollbar=True):
                logo_h = int(LOGO_W * self.logo_size[1] / self.logo_size[0])
                dpg.add_image("logo_texture", width=LOGO_W, height=logo_h,
                              indent=(HOME_W - LOGO_W) // 2)
                dpg.add_spacer(height=4)
                sub = dpg.add_text("Identities and trajectories for YORU detections",
                                   indent=120)
                dpg.bind_item_theme(sub, self.themes.muted)
                dpg.add_spacer(height=10)
                for name, label, text in _WORKFLOWS:
                    btn = dpg.add_button(label=label, width=440, height=58, indent=90,
                                         tag=f"home_{name}",
                                         callback=lambda s, a, u: self.show_view(u),
                                         user_data=name)
                    dpg.bind_item_theme(btn, self.themes.big_button)
                    note = dpg.add_text(text, indent=100)
                    dpg.bind_item_theme(note, self.themes.muted)
                    dpg.add_spacer(height=6)
                settings = dpg.add_button(label="Settings", width=180, height=34, indent=220,
                                          callback=lambda: self.open_settings())
                dpg.bind_item_theme(settings, self.themes.big_button)
                dpg.add_spacer(height=14)
                foot = dpg.add_text(
                    f"YORU Tracker {__version__}   |   YORU {yoru.__version__}   |   "
                    f"tracker API v{TRACKER_API_VERSION}", indent=110)
                dpg.bind_item_theme(foot, self.themes.muted)
                log = dpg.add_text(f"Log: {log_file()}", indent=40, wrap=HOME_W - 60)
                dpg.bind_item_theme(log, self.themes.muted)

    def build_menus(self) -> None:
        """Extra viewport menus, next to GuiSession's Window menu."""
        with dpg.menu(label="Go"):
            dpg.add_menu_item(label="Home", callback=lambda: self.show_view("home"))
            for name, label, _ in _WORKFLOWS:
                dpg.add_menu_item(label=label, callback=lambda s, a, u: self.show_view(u),
                                  user_data=name)
        with dpg.menu(label="Tracker"):
            dpg.add_menu_item(label="Settings...", callback=lambda: self.open_settings())
        with dpg.menu(label="Help"):
            dpg.add_menu_item(label="Open log folder", callback=lambda: self._open_log_folder())
            dpg.add_menu_item(label=f"YORU Tracker {__version__} (tracker API v{TRACKER_API_VERSION})",
                              enabled=False)

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def show_view(self, name: str) -> None:
        if name not in self.views:
            raise ValueError(f"Unknown view {name!r}")
        for other in self.views:
            dpg.hide_item(_window(other))
        dpg.show_item(_window(name))
        self.current = name
        view = self.views[name]
        if view is not None:
            view.on_show()
        self.layout()
        title = "YORU Tracker" if name == "home" else f"YORU Tracker - {dict((n, l) for n, l, _ in _WORKFLOWS)[name]}"
        try:
            dpg.set_viewport_title(title)
        except Exception:
            pass

    def layout(self) -> None:
        if self.session is not None:
            x0, top, width, height = self.session.content_region()
        else:
            x0, top, width, height = 0, 0, 1400, 900
        for name in self.views:
            tag = _window(name)
            dpg.set_item_pos(tag, [x0, top])
            dpg.configure_item(tag, width=width, height=height)
        dpg.set_item_pos("home_content", [max(0, (width - HOME_W) // 2), max(0, (height - HOME_H) // 2)])
        view = self.views.get(self.current)
        if view is not None:
            view.layout(width, height)

    def open_settings(self) -> None:
        self.settings.open()

    def _tracker_config_changed(self, config) -> None:
        for view in (self.video, self.realtime, self.batch):
            view.on_tracker_config(config)

    def _on_key(self, key) -> None:
        if self.current == "video":
            self.video.on_key(key)

    # ------------------------------------------------------------------
    # Shared actions
    # ------------------------------------------------------------------

    def choose_model(self, tag: str) -> None:
        path = widgets.ask_open_file("Select a detector model", widgets.MODEL_TYPES,
                                     current=self.state.detector.model_path)
        if not path:
            return
        self.state.set_detector(model_path=path)
        for t in ("video_model", "rt_model", "batch_model"):
            if dpg.does_item_exist(t):
                dpg.set_value(t, path)

    def report_error(self, context: str, exc: BaseException) -> None:
        """Log to ``yoru_tracker.log`` and show the error in a popup."""
        detail = f"{type(exc).__name__}: {exc}"
        print(f"[yoru-tracker] {context}: {detail}", file=sys.stderr, flush=True)
        log_exception(context, exc)
        self._show_error_popup(context, detail)

    # GuiErrorMixin's entry point, routed to the tracker's own log.
    _report_error = report_error

    def _open_log_folder(self) -> None:
        folder = log_file().parent
        try:
            if sys.platform == "win32":
                os.startfile(folder)  # noqa: S606 - opening a folder the user asked for
            else:
                import subprocess

                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])
        except Exception as exc:
            self.report_error(f"Could not open {folder}", exc)

    def start_background(self) -> None:
        """List the detector backends without holding up the first frame."""
        def backends():
            from yoru.libs.plugins import list_detector_backends

            return list_detector_backends()

        self._backend_task = widgets.BackgroundTask("backends", backends)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def tick(self) -> None:
        now = time.perf_counter()
        task = self._backend_task
        if task is not None and task.done:
            self._backend_task = None
            if task.error is None:
                self.backends = task.result
                dpg.configure_item("video_backend", items=self.backends)
            else:
                log_exception("Listing detector backends failed", task.error)
        for name, view in self.views.items():
            if view is not None:
                try:
                    view.tick(now)
                except Exception as exc:
                    self.report_error(f"{name} view failed", exc)

    def shutdown(self) -> None:
        for view in (self.video, self.realtime, self.batch):
            try:
                view.shutdown()
            except Exception as exc:
                log_exception("Shutdown cleanup failed", exc)
        self.state.save()


def _window(name: str) -> str:
    return HOME if name == "home" else f"view_{name}"


def run(exit_after: Optional[int] = None, screenshot: Optional[str] = None,
        view: Optional[str] = None) -> int:
    """Open the window and run until it is closed."""
    from yoru.gui_layout import GuiSession

    setup_logging()
    logger.info("YORU Tracker %s starting", __version__)
    app = TrackerApp()
    session = GuiSession("yoru_tracker", "YORU Tracker", width=1440, height=920,
                         min_width=1100, min_height=760)
    session.begin()
    # Callbacks run on this thread, from the loop below.
    dpg.configure_app(manual_callback_management=True)
    try:
        app.build()
        session.add_layout_menu(extra_builder=app.build_menus)
        session.finish(on_resize=app.layout)
        app.session = session
        app.show_view(view or "home")
        app.start_background()
        frames = 0
        shot_at = (exit_after or 30) - 8
        while dpg.is_dearpygui_running():
            jobs = dpg.get_callback_queue()
            if jobs:
                dpg.run_callbacks(jobs)
            app.tick()
            dpg.render_dearpygui_frame()
            frames += 1
            if screenshot and frames == shot_at:
                dpg.output_frame_buffer(file=screenshot)
            if exit_after is not None and frames >= exit_after:
                break
    finally:
        try:
            app.shutdown()
            session.save()
        finally:
            dpg.destroy_context()
    logger.info("YORU Tracker closed")
    return 0
