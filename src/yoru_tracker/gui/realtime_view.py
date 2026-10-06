# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Realtime tracking: camera, detector, tracker and display, side by side.

Left, the frame with the detector's boxes as they came; right, the same frame
with tracks, IDs and trails; below, the tracker's status and latency.  A
video file can stand in for the camera (played in real time) to try settings
without an animal under the lens.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from collections import deque
from pathlib import Path

import dearpygui.dearpygui as dpg

from yoru_tracker.drawing.overlays import OverlayOptions, TrailBook, draw_detections, draw_tracking
from yoru_tracker.gui import widgets
from yoru_tracker.gui.status_panel import EventLog, TrackTable, _ids

logger = logging.getLogger(__name__)

WINDOW = "view_realtime"
TEX_RAW = "rt_texture_raw"
TEX_TRACK = "rt_texture_track"
TEX_W, TEX_H = 640, 480
STATUS_W = 520
STATUS_H = 234
EVENT_LINES = 40

_OVERLAYS = (
    ("track_boxes", "Track boxes"),
    ("track_ids", "IDs"),
    ("trajectories", "Trails"),
    ("predicted", "Predicted (lost)"),
    ("velocity", "Velocity"),
    ("confidence", "Confidence"),
)

_STATUS_FIELDS = (
    ("mode", "Tracker mode"),
    ("active", "Active IDs"),
    ("lost", "Lost IDs"),
    ("capture_fps", "Camera FPS"),
    ("processed_fps", "Detector FPS"),
    ("detector_ms", "Detector latency"),
    ("tracker_ms", "Tracking latency"),
    ("wait_ms", "Queue wait"),
    ("draw_ms", "Draw time"),
    ("end_to_end_ms", "Capture to screen"),
    ("dropped", "Dropped frames"),
    ("resets", "Tracker resets"),
    ("recording", "Recording"),
)


class RealtimeView:
    def __init__(self, app):
        self.app = app
        self.state = app.state
        self.themes = app.themes
        self.rt = None
        self.trails = TrailBook(30)
        self._last_key = None
        self._generation = 0
        self._reported = False
        self._last_status = 0.0
        self._recording_path = None
        #: Events of every processed frame, put here by the processing thread.
        #: The display draws only the newest snapshot; the events of frames
        #: it skipped would go with them.  As long as the log shows, so what
        #: a stalled window (a file dialog) lets pile up stays bounded.
        self._inbox = deque(maxlen=EVENT_LINES)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, texture_registry) -> None:
        blank = widgets.blank_texture(TEX_W, TEX_H)
        dpg.add_dynamic_texture(TEX_W, TEX_H, blank, tag=TEX_RAW, parent=texture_registry)
        dpg.add_dynamic_texture(TEX_W, TEX_H, blank, tag=TEX_TRACK, parent=texture_registry)
        with dpg.window(tag=WINDOW, show=False, no_title_bar=True, no_move=True,
                        no_resize=True, no_collapse=True):
            dpg.bind_item_theme(WINDOW, self.themes.window)
            self._build_controls()
            with dpg.group(horizontal=True):
                with dpg.group():
                    t = dpg.add_text("Raw / detection")
                    dpg.bind_item_theme(t, self.themes.header)
                    dpg.add_image(TEX_RAW, tag="rt_image_raw", width=TEX_W, height=TEX_H)
                with dpg.group():
                    t = dpg.add_text("Tracking")
                    dpg.bind_item_theme(t, self.themes.header)
                    dpg.add_image(TEX_TRACK, tag="rt_image_track", width=TEX_W, height=TEX_H)
            with dpg.group(horizontal=True):
                defaults = OverlayOptions()
                for name, label in _OVERLAYS:
                    dpg.add_checkbox(label=label, tag=f"rt_ov_{name}",
                                     default_value=getattr(defaults, name))
            self._build_status()

    def _build_controls(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_text("Source")
            dpg.add_radio_button(["Camera", "Video file (real time)"], tag="rt_source",
                                 default_value="Camera", horizontal=True)
            dpg.add_text("  Camera ID")
            dpg.add_input_int(tag="rt_camera", width=90, default_value=self.state.camera_id,
                              min_value=0, min_clamped=True)
            dpg.add_input_text(tag="rt_video", readonly=True, hint="video for real-time playback",
                               width=280)
            dpg.add_button(label="Browse", callback=lambda: self.choose_video())
        with dpg.group(horizontal=True):
            dpg.add_text("Model ")
            dpg.add_input_text(tag="rt_model", readonly=True, hint="path/to/model.pt", width=360)
            dpg.add_button(label="Browse", callback=lambda: self.app.choose_model("rt_model"))
            dpg.add_text("  Conf.")
            dpg.add_input_float(tag="rt_conf", width=80, step=0, format="%.2f",
                                default_value=self.state.detector.conf_thresh,
                                callback=lambda s, v: self.state.set_detector(
                                    conf_thresh=float(f"{v:.4g}")))
            dpg.add_text("  ")
            dpg.add_text("", tag="rt_tracker_summary")
            dpg.add_button(label="Settings...", callback=lambda: self.app.open_settings())
        with dpg.group(horizontal=True):
            start = dpg.add_button(label="Start", tag="rt_start", width=100, height=30,
                                   callback=lambda: self.start())
            dpg.bind_item_theme(start, self.themes.primary_button)
            dpg.add_button(label="Stop", tag="rt_stop", width=80, height=30, enabled=False,
                           callback=lambda: self.stop())
            dpg.add_button(label="Reset tracker", tag="rt_reset", width=130, height=30,
                           enabled=False, callback=lambda: self.reset())
            with dpg.tooltip("rt_reset"):
                dpg.add_text("Drops every track; IDs restart at 0. Uses the current tracker "
                             "settings.", wrap=320)
            dpg.add_text("    Record tracks to")
            dpg.add_input_text(tag="rt_outdir", readonly=True, hint="output folder", width=240)
            dpg.add_button(label="Browse", callback=lambda: self.choose_outdir())
            dpg.add_button(label="Start recording", tag="rt_record", width=140, enabled=False,
                           callback=lambda: self.toggle_recording())
            dpg.add_checkbox(label="incl. predicted", tag="rt_inc_pred")
        dpg.add_text("", tag="rt_message")
        dpg.bind_item_theme("rt_message", self.themes.muted)

    def _build_status(self) -> None:
        with dpg.group(horizontal=True):
            with dpg.child_window(tag="rt_status", width=STATUS_W, height=STATUS_H, border=True,
                                  no_scrollbar=True):
                t = dpg.add_text("Tracking status")
                dpg.bind_item_theme(t, self.themes.header)
                half = (len(_STATUS_FIELDS) + 1) // 2
                with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=128)
                    dpg.add_table_column(init_width_or_weight=1.0)
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=128)
                    dpg.add_table_column(init_width_or_weight=1.0)
                    for i in range(half):
                        with dpg.table_row():
                            for key, label in _STATUS_FIELDS[i::half][:2]:
                                muted = dpg.add_text(label)
                                dpg.bind_item_theme(muted, self.themes.muted)
                                dpg.add_text("-", tag=f"rt_stat_{key}")
            with dpg.group(tag="rt_status_tracks"):
                dpg.add_text("Tracks")
                self.table = TrackTable("rt_tracks", height=STATUS_H - 26)
                self.table.build()
            with dpg.group():
                dpg.add_text("Events")
                self.events = EventLog("rt_events", length=EVENT_LINES)
                self.events.build(height=STATUS_H - 26)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def layout(self, width: int, height: int) -> None:
        # Controls (three rows and a message), image captions, overlay row,
        # the status block and the window's own padding.
        avail_h = height - STATUS_H - 230
        image_w, image_h = widgets.fit_size((width - 40) / 2, avail_h, TEX_W / TEX_H)
        for tag in ("rt_image_raw", "rt_image_track"):
            dpg.configure_item(tag, width=image_w, height=image_h)
        rest = max(300, width - STATUS_W - 50)
        dpg.configure_item("rt_tracks", width=int(rest * 0.6))
        dpg.configure_item("rt_events", width=int(rest * 0.4))

    def on_show(self) -> None:
        self.refresh_summary()
        dpg.set_value("rt_model", self.state.detector.model_path)
        dpg.set_value("rt_conf", self.state.detector.conf_thresh)
        if self.state.output_dir and not dpg.get_value("rt_outdir"):
            dpg.set_value("rt_outdir", self.state.output_dir)

    def refresh_summary(self) -> None:
        c = self.state.tracker_config
        dpg.set_value("rt_tracker_summary", f"Tracker: {c.mode}")

    def on_tracker_config(self, config) -> None:
        self.refresh_summary()
        if self.rt is not None and self.rt.running and config != self.rt.tracker_config:
            dpg.set_value("rt_message", "Tracker settings changed. They apply when you press "
                                        "Reset tracker (IDs restart) or start again.")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def choose_video(self) -> None:
        path = widgets.ask_open_file("Video to play as a live source", widgets.VIDEO_TYPES,
                                     current=self.state.last_video)
        if path:
            dpg.set_value("rt_video", path)
            dpg.set_value("rt_source", "Video file (real time)")

    def choose_outdir(self) -> None:
        path = widgets.ask_directory("Folder for recorded tracks",
                                     current=dpg.get_value("rt_outdir"))
        if path:
            dpg.set_value("rt_outdir", path)
            self.state.output_dir = path

    def start(self) -> None:
        from yoru_tracker.core.registry import get_tracker_class
        from yoru_tracker.runtime.realtime import RealtimeTracking
        from yoru_tracker.runtime.sources import CameraSource, PacedVideoSource

        if self.rt is not None and self.rt.running:
            return
        config = self.state.tracker_config
        try:
            info = get_tracker_class(config.mode).info
        except Exception as exc:
            self.app.report_error("Tracker unavailable", exc)
            return
        if not info.realtime_capable:
            self.app.report_error("Tracker cannot run live", ValueError(
                f"{info.name} is not realtime-capable; choose another mode in Settings."))
            return
        if not self.state.detector.model_path:
            self.app.report_error("No detector model selected", ValueError("Select a model first."))
            return
        if dpg.get_value("rt_source") == "Camera":
            camera_id = int(dpg.get_value("rt_camera"))
            self.state.camera_id = camera_id
            factory = lambda: CameraSource(camera_id)  # noqa: E731
            description = f"camera {camera_id}"
        else:
            path = dpg.get_value("rt_video")
            if not path:
                self.app.report_error("No video selected", ValueError(
                    "Choose a video to play as the live source, or select Camera."))
                return
            factory = lambda: PacedVideoSource(path)  # noqa: E731
            description = f"video {Path(path).name} played in real time"
        load, settings = self.state.detector_loader()
        # A box of this run's own: a thread of an earlier run that is still
        # finishing cannot put its events into this one's log.
        inbox = deque(maxlen=EVENT_LINES)
        try:
            self.rt = RealtimeTracking(factory, load, config, detector_settings=settings,
                                       source_description=description,
                                       on_result=lambda s: inbox.extend(s.result.events))
        except Exception as exc:
            self.app.report_error("Could not start live tracking", exc)
            return
        self._inbox = inbox
        self.trails.clear()
        self.events.clear()
        self._last_key = None
        self._generation = 0
        self._reported = False
        self.rt.start()
        dpg.set_value("rt_message", f"Starting: {description}, loading the model...")
        self._buttons()

    def stop(self) -> None:
        if self.rt is not None:
            self.rt.stop(timeout=3.0)
            dpg.set_value("rt_message", "Stopped.")
        self._buttons()

    def reset(self) -> None:
        if self.rt is None or not self.rt.running:
            return
        try:
            self.rt.reset_tracker(self.state.tracker_config)
        except Exception as exc:  # the settings cannot run live; the run goes on
            self.app.report_error("Tracker not reset", exc)
            return
        note = " Recording continues in a new file." if self.rt.recording else ""
        dpg.set_value("rt_message", "Tracker reset: every track dropped, IDs restart at 0." + note)

    def toggle_recording(self) -> None:
        if self.rt is None:
            return
        if self.rt.recording:
            self.rt.stop_recording()
            dpg.set_value("rt_message", "Recording stopped.")
        else:
            out = dpg.get_value("rt_outdir")
            if not out:
                self.app.report_error("No output folder", ValueError("Choose where to save tracks."))
                return
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(out) / f"live_{stamp}_tracks.csv"
            try:
                self.rt.start_recording(path, include_predicted=dpg.get_value("rt_inc_pred"))
            except Exception as exc:
                self.app.report_error("Could not start recording", exc)
                return
            self._recording_path = path
            dpg.set_value("rt_message", f"Recording to {path}")
        self._buttons()

    def _buttons(self) -> None:
        running = self.rt is not None and self.rt.running
        dpg.configure_item("rt_start", enabled=not running)
        dpg.configure_item("rt_stop", enabled=running)
        dpg.configure_item("rt_reset", enabled=running)
        dpg.configure_item("rt_record", enabled=running)
        recording = self.rt is not None and self.rt.recording
        dpg.configure_item("rt_record", label="Stop recording" if recording else "Start recording")

    def overlay_options(self) -> OverlayOptions:
        return OverlayOptions(**{name: bool(dpg.get_value(f"rt_ov_{name}"))
                                 for name, _ in _OVERLAYS})

    # ------------------------------------------------------------------
    # Per frame
    # ------------------------------------------------------------------

    def tick(self, now: float) -> None:
        rt = self.rt
        if rt is None:
            return
        if not rt.running and not self._reported:
            self._reported = True
            if rt.error is not None:
                self.app.report_error("Live tracking stopped", rt.error)
                dpg.set_value("rt_message", f"Stopped by an error: {rt.error}")
            rt.stop(timeout=1.0)
            self._buttons()
        if rt.running and rt.phase == "running" and dpg.get_value("rt_message").startswith("Starting"):
            dpg.set_value("rt_message", f"Running: {rt.source_description}")

        snap = rt.latest()
        if snap is not None and (snap.frame_id, snap.generation) != self._last_key:
            self._last_key = (snap.frame_id, snap.generation)
            draw_started = time.perf_counter()
            if snap.generation != self._generation:
                self._generation = snap.generation
                self.trails.clear()
            self.trails.add(snap.result)
            options = self.overlay_options()
            raw = draw_detections(snap.image.copy(), snap.detections)
            tracked = draw_tracking(snap.image.copy(), snap.result, options,
                                    trails=self.trails.trails())
            dpg.set_value(TEX_RAW, widgets.texture_data(widgets.letterbox(raw, TEX_W, TEX_H)))
            dpg.set_value(TEX_TRACK, widgets.texture_data(widgets.letterbox(tracked, TEX_W, TEX_H)))
            rt.report_drawn(snap, draw_started)

        events = []
        while self._inbox:
            events.append(self._inbox.popleft())
        self.events.add(events)

        if now - self._last_status >= 0.25:
            self._last_status = now
            self._update_status(snap)

    def _update_status(self, snap) -> None:
        rt = self.rt
        stats = rt.stats()
        result = snap.result if snap is not None else None
        path = rt.recording_path
        if path is not None and path != self._recording_path:
            self._recording_path = path      # only a reset moves the recording on
            dpg.set_value("rt_message", f"Tracker reset: IDs restart at 0; recording continues "
                                        f"in {path}")
        values = {
            "mode": rt.tracker.info.name,
            "active": _ids(result.active_ids) if result else "-",
            "lost": _ids(result.lost_ids) if result else "-",
            "capture_fps": f"{stats.capture_fps:.1f}",
            "processed_fps": f"{stats.processed_fps:.1f}",
            "detector_ms": f"{stats.detector_ms:.1f} ms",
            "tracker_ms": f"{stats.tracker_ms:.2f} ms",
            "wait_ms": f"{stats.wait_ms:.1f} ms",
            "draw_ms": f"{stats.draw_ms:.1f} ms",
            "end_to_end_ms": f"{stats.end_to_end_ms:.1f} ms",
            "dropped": f"{stats.dropped_frames} of {stats.dropped_frames + stats.processed_frames}",
            "resets": str(rt.generation),
            "recording": f"{stats.recording_rows} rows" if rt.recording else "off",
        }
        for key, value in values.items():
            dpg.set_value(f"rt_stat_{key}", value)
        self.table.show(result)

    def shutdown(self) -> None:
        if self.rt is not None:
            self.rt.stop(timeout=3.0)
