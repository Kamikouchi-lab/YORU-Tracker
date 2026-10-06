# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Video tracking: run a video through detector and tracker, then inspect it.

Tracking is one pass from the first frame to the last; the timeline only
chooses which frame is shown.  While the pass runs the view follows it, and
any frame already processed can be inspected -- its tracks, their state and
match cost, the events of that frame.  Changing the tracker settings and
pressing Re-track tracks the stored detections again in seconds.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import dearpygui.dearpygui as dpg

from yoru_tracker.drawing.overlays import OverlayOptions, draw_tracking, trails_from_results
from yoru_tracker.gui import widgets
from yoru_tracker.gui.status_panel import EventLog, TrackTable
from yoru_tracker.runtime.sources import VideoFileSource

logger = logging.getLogger(__name__)

WINDOW = "view_video"
TEXTURE = "video_texture"
TEX_W, TEX_H = 960, 720
CONTROLS_W = 410

_OVERLAYS = (
    ("track_boxes", "Track boxes"),
    ("track_ids", "IDs"),
    ("trajectories", "Trails"),
    ("detections", "Detections"),
    ("predicted", "Predicted (lost)"),
    ("velocity", "Velocity"),
    ("confidence", "Confidence"),
)


class VideoView:
    def __init__(self, app):
        self.app = app
        self.state = app.state
        self.themes = app.themes
        self.source: Optional[VideoFileSource] = None
        self.job = None
        self.run = None
        self.index = 0
        self.playing = False
        self.follow = False
        self._last_step = 0.0
        self._dirty = True
        self._frame_cache = (None, None)
        self._task = None
        self._task_label = ""
        self._job_reported = False
        self._settings_changed = False

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, texture_registry) -> None:
        dpg.add_dynamic_texture(TEX_W, TEX_H, widgets.blank_texture(TEX_W, TEX_H),
                                tag=TEXTURE, parent=texture_registry)
        with dpg.window(tag=WINDOW, show=False, no_title_bar=True, no_move=True,
                        no_resize=True, no_collapse=True, no_scrollbar=True):
            dpg.bind_item_theme(WINDOW, self.themes.window)
            with dpg.group(horizontal=True):
                with dpg.child_window(tag="video_controls", width=CONTROLS_W, border=False):
                    self._build_controls()
                with dpg.child_window(tag="video_display", border=False, no_scrollbar=True):
                    self._build_display()

    def _path_row(self, label, tag, hint, callback):
        dpg.add_text(label)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag=tag, readonly=True, hint=hint, width=CONTROLS_W - 110)
            dpg.add_button(label="Browse", width=80, callback=callback)

    def _build_controls(self) -> None:
        widgets.heading("Input", self.themes.header)
        self._path_row("Video", "video_path", "path/to/video.mp4", lambda: self.choose_video())
        self._path_row("Detector model", "video_model", "path/to/model.pt",
                       lambda: self.app.choose_model("video_model"))
        with dpg.group(horizontal=True):
            dpg.add_text("Backend")
            dpg.add_combo(["auto"], tag="video_backend", default_value="auto", width=130,
                          callback=lambda s, v: self.state.set_detector(backend=v))
            dpg.add_text("Conf.")
            dpg.add_input_float(tag="video_conf", width=80, step=0, format="%.2f",
                                default_value=self.state.detector.conf_thresh,
                                callback=lambda s, v: self.state.set_detector(
                                    conf_thresh=float(f"{v:.4g}")))
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Flip vertical", tag="video_vflip",
                             callback=lambda: self._flips_changed())
            dpg.add_checkbox(label="Flip horizontal", tag="video_hflip",
                             callback=lambda: self._flips_changed())

        dpg.add_spacer(height=6)
        widgets.heading("Tracking", self.themes.header)
        with dpg.group(horizontal=True):
            dpg.add_text("", tag="video_tracker_summary")
            dpg.add_button(label="Settings...", callback=lambda: self.app.open_settings())
        with dpg.group(horizontal=True):
            run_btn = dpg.add_button(label="Run tracking", tag="video_run", width=150,
                                     height=32, callback=lambda: self.start())
            dpg.bind_item_theme(run_btn, self.themes.primary_button)
            dpg.add_button(label="Stop", tag="video_stop", width=70, height=32,
                           callback=lambda: self.stop(), enabled=False)
        dpg.add_progress_bar(tag="video_progress", default_value=0.0, width=-1, overlay="")
        dpg.add_text("Select a video and a model.", tag="video_status", wrap=CONTROLS_W - 20)
        dpg.bind_item_theme("video_status", self.themes.muted)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Re-track", tag="video_retrack", width=110, enabled=False,
                           callback=lambda: self.retrack())
            dpg.add_button(label="Load detections CSV...", tag="video_load_dets",
                           callback=lambda: self.load_detections())
        dpg.add_text("", tag="video_retrack_note", wrap=CONTROLS_W - 20)
        dpg.bind_item_theme("video_retrack_note", self.themes.warn)

        dpg.add_spacer(height=6)
        widgets.heading("Export", self.themes.header)
        self._path_row("Output folder", "video_outdir", "(beside the video)",
                       lambda: self.choose_outdir())
        dpg.add_checkbox(label="Include predicted rows of lost tracks", tag="video_inc_pred")
        dpg.add_checkbox(label="Save detections (re-track later without the model)",
                         tag="video_save_dets", default_value=True)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Export CSV", tag="video_export", width=120, enabled=False,
                           callback=lambda: self.export())
            dpg.add_button(label="Render video", tag="video_render", width=120, enabled=False,
                           callback=lambda: self.render())
        dpg.add_text("", tag="video_export_status", wrap=CONTROLS_W - 20)
        dpg.bind_item_theme("video_export_status", self.themes.muted)

    def _build_display(self) -> None:
        dpg.add_image(TEXTURE, tag="video_image", width=TEX_W, height=TEX_H)
        dpg.add_slider_int(tag="video_slider", min_value=0, max_value=0, width=-1,
                           callback=lambda s, v: self.goto(v, user=True))
        with dpg.group(horizontal=True):
            dpg.add_button(label="|<", width=36, callback=lambda: self.goto(0, user=True))
            dpg.add_button(label="<", width=36, callback=lambda: self.step(-1))
            dpg.add_button(label="Play", tag="video_play", width=70, callback=lambda: self.toggle_play())
            dpg.add_button(label=">", width=36, callback=lambda: self.step(1))
            dpg.add_button(label=">|", width=36, callback=lambda: self.goto(10 ** 9, user=True))
            dpg.add_text("", tag="video_frame_label")
        with dpg.group(horizontal=True):
            defaults = OverlayOptions()
            for name, label in _OVERLAYS:
                dpg.add_checkbox(label=label, tag=f"video_ov_{name}",
                                 default_value=getattr(defaults, name),
                                 callback=lambda: self._mark_dirty())
        with dpg.group(horizontal=True, tag="video_inspect"):
            with dpg.group():
                dpg.add_text("Tracks in this frame")
                self.table = TrackTable("video_tracks", height=170)
                self.table.build()
            with dpg.group():
                dpg.add_text("Events in this frame")
                self.events = EventLog("video_events")
                self.events.build(height=170)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def layout(self, width: int, height: int) -> None:
        display_w = max(200, width - CONTROLS_W - 30)
        image_w, image_h = widgets.fit_size(display_w, height - 330, TEX_W / TEX_H)
        dpg.configure_item("video_image", width=image_w, height=image_h)
        dpg.configure_item("video_controls", height=height - 20)
        dpg.configure_item("video_display", height=height - 20)
        table_w = max(200, int(display_w * 0.58))
        dpg.configure_item("video_tracks", width=table_w)
        dpg.configure_item("video_events", width=max(150, display_w - table_w - 20))

    def on_show(self) -> None:
        self.refresh_summary()
        dpg.set_value("video_model", self.state.detector.model_path)
        dpg.set_value("video_backend", self.state.detector.backend)
        dpg.set_value("video_conf", self.state.detector.conf_thresh)
        if self.state.output_dir and not dpg.get_value("video_outdir"):
            dpg.set_value("video_outdir", self.state.output_dir)

    def refresh_summary(self) -> None:
        c = self.state.tracker_config
        dpg.set_value("video_tracker_summary",
                      f"{c.mode}  max_age {c.lifecycle.max_age}  "
                      f"max_dist {c.association.max_distance:g}px")

    def on_tracker_config(self, config) -> None:
        self.refresh_summary()
        if self.run is not None and config != self.run.tracker_config:
            self._settings_changed = True
            dpg.set_value("video_retrack_note",
                          "Tracker settings changed: press Re-track to apply them to this video.")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def choose_video(self) -> None:
        path = widgets.ask_open_file("Select a video", widgets.VIDEO_TYPES,
                                     current=self.state.last_video)
        if path:
            self.open_video(path)

    def open_video(self, path: str) -> None:
        if self._busy():
            # A re-track or load finishing later would hang its results on
            # whatever video is open by then.
            self._report_busy()
            return
        try:
            source = VideoFileSource(path, v_flip=dpg.get_value("video_vflip"),
                                     h_flip=dpg.get_value("video_hflip"))
        except Exception as exc:
            self.app.report_error(f"Could not open {path}", exc)
            return
        if self.source is not None:
            self.source.close()
        self.source = source
        self.state.last_video = path
        self.job = None
        self.run = None
        self._set_run_buttons()
        dpg.set_value("video_path", path)
        info = source.info
        dpg.configure_item("video_slider", max_value=max(0, info.frame_count - 1))
        dpg.set_value("video_status",
                      f"{Path(path).name}: {info.frame_count} frames, {info.width}x{info.height}, "
                      f"{info.fps:.2f} fps")
        dpg.set_value("video_progress", 0.0)
        dpg.configure_item("video_progress", overlay="")
        dpg.set_value("video_retrack_note", "")
        self.goto(0)

    def choose_outdir(self) -> None:
        path = widgets.ask_directory("Output folder", current=dpg.get_value("video_outdir"))
        if path:
            dpg.set_value("video_outdir", path)
            self.state.output_dir = path

    def _flips_changed(self) -> None:
        path = dpg.get_value("video_path")
        if not path:
            return
        if self._busy():
            # The open video keeps its flips; the boxes must keep showing them.
            dpg.set_value("video_vflip", self.source.v_flip)
            dpg.set_value("video_hflip", self.source.h_flip)
            self._report_busy()
            return
        had_results = self.run is not None
        self.open_video(path)
        if had_results:
            dpg.set_value("video_status", "Flip changed: the previous results no longer "
                                          "match the frames and were cleared.")

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def start(self) -> None:
        from yoru_tracker.runtime.video import VideoJob

        if self.source is None:
            self.app.report_error("No video selected", ValueError("Select a video first."))
            return
        detector_config = self.state.detector
        if not detector_config.model_path:
            self.app.report_error("No detector model selected", ValueError("Select a model first."))
            return
        if self._busy():
            return
        load, settings = self.state.detector_loader()
        self.job = VideoJob(
            self.source.path, load, self.state.tracker_config,
            v_flip=self.source.v_flip, h_flip=self.source.h_flip,
            detector_settings=settings,
        )
        self.run = self.job.run
        self._job_reported = False
        self._settings_changed = False
        dpg.set_value("video_retrack_note", "")
        self.follow = True
        self.playing = False
        dpg.configure_item("video_play", label="Play")
        self.job.start()
        self._set_run_buttons()

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop(timeout=0.5)

    def retrack(self) -> None:
        from yoru_tracker.runtime.video import retrack

        if self.run is None or not self.run.processed or self._busy():
            return
        run, config = self.run, self.state.tracker_config
        self._start_task("re-track", lambda: retrack(run, config), self._retracked)

    def _retracked(self, run) -> None:
        self.run = run
        self._settings_changed = False
        dpg.set_value("video_retrack_note", "")
        dpg.set_value("video_status",
                      f"Re-tracked {run.processed} frames with mode {run.tracker_config.mode}: "
                      f"{len(run.track_ids())} track IDs, "
                      f"{run.stats.tracker_ms:.3f} ms/frame.")
        self._mark_dirty()

    def load_detections(self) -> None:
        if self.source is None:
            self.app.report_error("No video selected",
                                  ValueError("Select the video the detections belong to first."))
            return
        if self._busy():
            return
        path = widgets.ask_open_file("Detections CSV for this video",
                                     (("CSV", "*.csv"), ("All files", "*.*")),
                                     current=dpg.get_value("video_outdir") or self.source.path)
        if not path:
            return
        info, config = self.source.info, self.state.tracker_config

        def work():
            from yoru_tracker.core.registry import create_tracker
            from yoru_tracker.runtime.detections_file import (
                align_to_video,
                load_detections,
                realtime_log_path,
            )
            from yoru_tracker.runtime.frames import track_frames
            from yoru_tracker.runtime.video import RunStats, VideoTracking

            layout, frames = load_detections(path)
            if layout == "yoru-realtime" and realtime_log_path(path) is None:
                raise ValueError(
                    f"{Path(path).name} is a YORU real-time detections file without its "
                    f"*_log.csv beside it; only the log says which video frame each "
                    f"detection belongs to.")
            if frames and info.frame_count and frames[-1].frame_id >= info.frame_count:
                raise ValueError(
                    f"{Path(path).name} has detections up to frame {frames[-1].frame_id}, "
                    f"but the video has {info.frame_count} frames: it belongs to another video.")
            # Frames must line up with the video's: frame i at index i.
            aligned = align_to_video(layout, frames, info.fps)
            tracker = create_tracker(config)
            t0 = time.perf_counter()
            results = track_frames(tracker, aligned)
            stats = RunStats(len(results), 0.0, time.perf_counter() - t0)
            return VideoTracking(info, config, {"detections_csv": path, "layout": layout},
                                 aligned, results, stats, complete=True, tracker=tracker)

        self._start_task("load detections", work, self._retracked)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _outdir(self) -> str:
        return dpg.get_value("video_outdir") or str(Path(self.source.path).parent)

    def export(self) -> None:
        from yoru_tracker.runtime.video import export_run

        if self.run is None or not self.run.processed or self._busy():
            return
        run, out = self.run, self._outdir()
        include = dpg.get_value("video_inc_pred")
        save_dets = dpg.get_value("video_save_dets")
        self._start_task("export", lambda: export_run(
            run, out, include_predicted=include, save_detections=save_dets), self._exported)

    def _exported(self, written) -> None:
        names = ", ".join(Path(p).name for p in written.values())
        dpg.set_value("video_export_status", f"Wrote {names} to {Path(next(iter(written.values()))).parent}")

    def render(self) -> None:
        from yoru_tracker.runtime.video import output_paths, render_video

        if self.run is None or not self.run.processed or self._busy():
            return
        run = self.run
        out = output_paths(self._outdir(), Path(run.info.path).stem)["video"]
        options = self.overlay_options()
        v, h = self.source.v_flip, self.source.h_flip
        self._start_task("render", lambda: render_video(run, out, options, v_flip=v, h_flip=h),
                         lambda path: dpg.set_value("video_export_status", f"Rendered {path}"))

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    def _busy(self) -> bool:
        return (self._task is not None) or (self.job is not None and self.job.running)

    def _report_busy(self) -> None:
        what = self._task_label if self._task is not None else "tracking"
        self.app.report_error("The current video is still in use", RuntimeError(
            f"Wait for {what} to finish, or stop it, first."))

    def _start_task(self, label, fn, on_done) -> None:
        self._task = widgets.BackgroundTask(label, fn)
        self._task_label = label
        self._task_done = on_done
        dpg.set_value("video_export_status", f"{label.capitalize()}...")
        self._set_run_buttons()

    def _poll_task(self) -> None:
        task = self._task
        if task is None or not task.done:
            return
        self._task = None
        if task.error is not None:
            dpg.set_value("video_export_status", f"{self._task_label.capitalize()} failed.")
            self.app.report_error(f"{self._task_label.capitalize()} failed", task.error)
        else:
            dpg.set_value("video_export_status", "")
            self._task_done(task.result)
        self._set_run_buttons()

    def _set_run_buttons(self) -> None:
        running = self.job is not None and self.job.running
        has_results = self.run is not None and self.run.processed > 0
        busy = self._task is not None
        dpg.configure_item("video_run", enabled=not running and not busy)
        dpg.configure_item("video_stop", enabled=running)
        for tag in ("video_retrack", "video_export", "video_render"):
            dpg.configure_item(tag, enabled=has_results and not running and not busy)
        dpg.configure_item("video_load_dets", enabled=not running and not busy)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def overlay_options(self) -> OverlayOptions:
        return OverlayOptions(**{name: bool(dpg.get_value(f"video_ov_{name}"))
                                 for name, _ in _OVERLAYS})

    def _mark_dirty(self) -> None:
        self._dirty = True

    def goto(self, index: int, user: bool = False) -> None:
        if self.source is None:
            return
        last = max(0, self.source.info.frame_count - 1)
        index = max(0, min(int(index), last))
        if user:
            self.follow = False
        if index != self.index:
            self.index = index
            self._dirty = True
        dpg.set_value("video_slider", index)

    def step(self, delta: int) -> None:
        self.playing = False
        dpg.configure_item("video_play", label="Play")
        self.goto(self.index + delta, user=True)

    def toggle_play(self) -> None:
        if self.source is None:
            return
        self.playing = not self.playing
        self.follow = False
        self._last_step = time.perf_counter()
        dpg.configure_item("video_play", label="Pause" if self.playing else "Play")

    def on_key(self, key) -> None:
        if key == dpg.mvKey_Spacebar:
            self.toggle_play()
        elif key == dpg.mvKey_Left:
            self.step(-1)
        elif key == dpg.mvKey_Right:
            self.step(1)
        elif key == dpg.mvKey_Home:
            self.goto(0, user=True)
        elif key == dpg.mvKey_End:
            self.goto(10 ** 9, user=True)

    # ------------------------------------------------------------------
    # Per frame
    # ------------------------------------------------------------------

    def tick(self, now: float) -> None:
        self._poll_task()
        job = self.job
        if job is not None:
            run = job.run
            total = max(1, run.info.frame_count)
            if job.running or not self._job_reported:
                done = run.processed
                dpg.set_value("video_progress", min(1.0, done / total))
                dpg.configure_item("video_progress", overlay=f"{done}/{run.info.frame_count}")
                dpg.set_value("video_status",
                              f"{job.phase}: {done}/{run.info.frame_count} frames, "
                              f"{len(run.results[-1].tracked) if run.results else 0} tracks now; "
                              f"detector {run.stats.detector_ms:.1f} ms, "
                              f"tracker {run.stats.tracker_ms:.3f} ms per frame")
                if self.follow and done:
                    self.goto(done - 1)
            if not job.running and not self._job_reported:
                self._job_reported = True
                self.follow = False
                if job.error is not None:
                    self.app.report_error("Video tracking failed", job.error)
                    dpg.set_value("video_status", f"Failed: {job.error}")
                else:
                    dpg.set_value("video_status",
                                  f"{job.phase.capitalize()}: {run.processed} frames, "
                                  f"{len(run.track_ids())} track IDs. "
                                  f"Detector {run.stats.detector_ms:.1f} ms, "
                                  f"tracker {run.stats.tracker_ms:.3f} ms per frame.")
                self._set_run_buttons()
                self._dirty = True

        if self.playing and self.source is not None:
            interval = 1.0 / max(1.0, self.source.info.fps)
            if now - self._last_step >= interval:
                self._last_step = now
                limit = self.source.info.frame_count - 1
                if self.index >= limit:
                    self.toggle_play()
                else:
                    self.goto(self.index + 1)

        if self._dirty:
            self._dirty = False
            self.render_frame()

    def _frame(self, index):
        cached_index, cached = self._frame_cache
        if cached_index == index and cached is not None:
            return cached
        frame = self.source.frame_at(index)
        self._frame_cache = (index, frame)
        return frame

    def render_frame(self) -> None:
        if self.source is None:
            return
        frame = self._frame(self.index)
        info = self.source.info
        if frame is None:
            return
        image = frame.copy()
        run = self.run
        result = None
        shown = None
        if run is not None and self.index < run.processed:
            # On a frame the detector never ran on, the latest result before it.
            shown = run.shown_index(self.index)
        if shown is not None:
            result = run.results[shown]
            options = self.overlay_options()
            trails = (trails_from_results(run.results, shown, options.trail_length)
                      if options.trajectories else None)
            draw_tracking(image, result, options, trails=trails,
                          detections=run.frames[shown].detections)
        dpg.set_value(TEXTURE, widgets.texture_data(widgets.letterbox(image, TEX_W, TEX_H)))
        label = f"Frame {self.index + 1}/{info.frame_count}   t = {self.index / info.fps:.2f} s"
        if result is not None:
            label += (f"   active {len(result.active_ids)}   lost {len(result.lost_ids)}")
            if shown != self.index:
                label += f"   (not detected; tracks of frame {shown + 1})"
        elif run is not None:
            label += ("   (not detected)" if self.index < run.processed
                      else "   (not tracked yet)")
        dpg.set_value("video_frame_label", label)
        self.table.show(result)
        self.events.show(result.events if result is not None and shown == self.index else ())

    def shutdown(self) -> None:
        if self.job is not None:
            self.job.stop(timeout=3.0)
        if self._task is not None:
            self._task.join(timeout=3.0)
        if self.source is not None:
            self.source.close()
