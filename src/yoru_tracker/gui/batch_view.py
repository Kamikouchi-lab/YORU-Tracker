# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Batch tracking: every video in a folder, one row of status per file."""

from __future__ import annotations

import logging
from pathlib import Path

import dearpygui.dearpygui as dpg

from yoru_tracker.gui import widgets
from yoru_tracker.gui.theme import TEXT_ERROR, TEXT_MUTED, TEXT_OK, TEXT_WARN

logger = logging.getLogger(__name__)

WINDOW = "view_batch"
CONTROLS_W = 420

_STATUS_COLORS = {
    "pending": (*TEXT_MUTED, 255),
    "running": (*TEXT_WARN, 255),
    "done": (*TEXT_OK, 255),
    "stopped": (*TEXT_WARN, 255),
    "failed": (*TEXT_ERROR, 255),
}


class BatchView:
    def __init__(self, app):
        self.app = app
        self.state = app.state
        self.themes = app.themes
        self.videos = []
        self.job = None
        self._reported = False
        self._last_refresh = 0.0

    def build(self, texture_registry) -> None:
        with dpg.window(tag=WINDOW, show=False, no_title_bar=True, no_move=True,
                        no_resize=True, no_collapse=True):
            dpg.bind_item_theme(WINDOW, self.themes.window)
            with dpg.group(horizontal=True):
                with dpg.child_window(tag="batch_controls", width=CONTROLS_W, border=False):
                    self._build_controls()
                with dpg.child_window(tag="batch_files", border=False):
                    t = dpg.add_text("Files")
                    dpg.bind_item_theme(t, self.themes.header)
                    with dpg.table(tag="batch_table", header_row=True, resizable=True,
                                   borders_innerH=True, borders_outerH=True,
                                   borders_innerV=True, borders_outerV=True,
                                   row_background=True, scrollY=True, height=420,
                                   policy=dpg.mvTable_SizingStretchProp):
                        for label, weight in (("File", 3.0), ("Status", 1.0), ("Frames", 1.2),
                                              ("Track IDs", 0.8), ("Message", 3.0)):
                            dpg.add_table_column(label=label, init_width_or_weight=weight)
                    dpg.add_spacer(height=6)
                    t = dpg.add_text("Failure report")
                    dpg.bind_item_theme(t, self.themes.header)
                    dpg.add_input_text(tag="batch_report", multiline=True, readonly=True,
                                       width=-1, height=160)

    def _build_controls(self) -> None:
        widgets.heading("Input", self.themes.header)
        dpg.add_text("Video folder")
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="batch_folder", readonly=True, hint="folder of videos",
                               width=CONTROLS_W - 110)
            dpg.add_button(label="Browse", width=80, callback=lambda: self.choose_folder())
        dpg.add_checkbox(label="Include subfolders", tag="batch_recursive",
                         callback=lambda: self.scan())
        dpg.add_text("Detector model")
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="batch_model", readonly=True, hint="path/to/model.pt",
                               width=CONTROLS_W - 110)
            dpg.add_button(label="Browse", width=80,
                           callback=lambda: self.app.choose_model("batch_model"))
        with dpg.group(horizontal=True):
            dpg.add_text("Conf.")
            dpg.add_input_float(tag="batch_conf", width=80, step=0, format="%.2f",
                                default_value=self.state.detector.conf_thresh,
                                callback=lambda s, v: self.state.set_detector(
                                    conf_thresh=float(f"{v:.4g}")))
        dpg.add_spacer(height=6)
        widgets.heading("Tracking", self.themes.header)
        with dpg.group(horizontal=True):
            dpg.add_text("", tag="batch_tracker_summary")
            dpg.add_button(label="Settings...", callback=lambda: self.app.open_settings())
        dpg.add_text("Output folder")
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="batch_outdir", readonly=True, hint="results folder",
                               width=CONTROLS_W - 110)
            dpg.add_button(label="Browse", width=80, callback=lambda: self.choose_outdir())
        dpg.add_checkbox(label="Include predicted rows of lost tracks", tag="batch_inc_pred")
        dpg.add_spacer(height=6)
        with dpg.group(horizontal=True):
            run = dpg.add_button(label="Run batch", tag="batch_run", width=150, height=32,
                                 callback=lambda: self.start())
            dpg.bind_item_theme(run, self.themes.primary_button)
            dpg.add_button(label="Stop", tag="batch_stop", width=70, height=32, enabled=False,
                           callback=lambda: self.stop())
        dpg.add_progress_bar(tag="batch_progress", default_value=0.0, width=-1, overlay="")
        dpg.add_text("Select a folder of videos.", tag="batch_status", wrap=CONTROLS_W - 20)
        dpg.bind_item_theme("batch_status", self.themes.muted)

    def layout(self, width: int, height: int) -> None:
        dpg.configure_item("batch_controls", height=height - 20)
        dpg.configure_item("batch_files", height=height - 20)
        dpg.configure_item("batch_table", height=max(160, height - 260))

    def on_show(self) -> None:
        self.refresh_summary()
        dpg.set_value("batch_model", self.state.detector.model_path)
        dpg.set_value("batch_conf", self.state.detector.conf_thresh)
        if self.state.output_dir and not dpg.get_value("batch_outdir"):
            dpg.set_value("batch_outdir", self.state.output_dir)

    def refresh_summary(self) -> None:
        c = self.state.tracker_config
        dpg.set_value("batch_tracker_summary",
                      f"{c.mode}  max_age {c.lifecycle.max_age}  "
                      f"max_dist {c.association.max_distance:g}px")

    def on_tracker_config(self, config) -> None:
        self.refresh_summary()

    # ------------------------------------------------------------------

    def choose_folder(self) -> None:
        path = widgets.ask_directory("Folder of videos", current=self.state.last_folder)
        if path:
            dpg.set_value("batch_folder", path)
            self.state.last_folder = path
            self.scan()

    def choose_outdir(self) -> None:
        path = widgets.ask_directory("Results folder", current=dpg.get_value("batch_outdir"))
        if path:
            dpg.set_value("batch_outdir", path)
            self.state.output_dir = path

    def scan(self) -> None:
        from yoru_tracker.runtime.batch import find_videos

        folder = dpg.get_value("batch_folder")
        if not folder:
            return
        try:
            self.videos = find_videos(folder, recursive=dpg.get_value("batch_recursive"))
        except Exception as exc:
            self.app.report_error("Could not list the folder", exc)
            return
        widgets.set_table_rows("batch_table", [(Path(v).name, "pending", "", "", "")
                                               for v in self.videos])
        dpg.set_value("batch_status", f"{len(self.videos)} video(s) found.")

    def start(self) -> None:
        from yoru_tracker.runtime.batch import BatchJob

        if self.job is not None and self.job.running:
            return
        problems = []
        if not self.videos:
            problems.append("no videos selected")
        if not self.state.detector.model_path:
            problems.append("no detector model")
        if not dpg.get_value("batch_outdir"):
            problems.append("no output folder")
        if problems:
            self.app.report_error("Cannot start the batch", ValueError("; ".join(problems)))
            return
        load, settings = self.state.detector_loader()
        self.job = BatchJob(self.videos, load, self.state.tracker_config,
                            dpg.get_value("batch_outdir"), detector_settings=settings,
                            include_predicted=dpg.get_value("batch_inc_pred"))
        self._reported = False
        dpg.set_value("batch_report", "")
        self.job.start()
        self._buttons()

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop(timeout=0.5)
            dpg.set_value("batch_status", "Stopping after the current frame...")

    def _buttons(self) -> None:
        running = self.job is not None and self.job.running
        dpg.configure_item("batch_run", enabled=not running)
        dpg.configure_item("batch_stop", enabled=running)

    def _refresh_table(self) -> None:
        items = self.job.items
        rows, colors = [], []
        for item in items:
            frames = f"{item.frames}/{item.total_frames}" if item.total_frames else (
                str(item.frames) if item.frames else "")
            rows.append((item.name, item.status, frames,
                         item.track_ids if item.status == "done" else "", item.error))
            colors.append(_STATUS_COLORS.get(item.status))
        widgets.set_table_rows("batch_table", rows, colors)
        finished = sum(1 for i in items if i.status in ("done", "failed", "stopped"))
        current = next((i for i in items if i.status == "running"), None)
        fraction = finished / max(1, len(items))
        if current is not None and current.total_frames:
            fraction += current.frames / current.total_frames / max(1, len(items))
        dpg.set_value("batch_progress", min(1.0, fraction))
        dpg.configure_item("batch_progress", overlay=f"{finished}/{len(items)} files")
        status = f"{self.job.phase}"
        if current is not None:
            status += f": {current.name}"
        dpg.set_value("batch_status", status)

    def tick(self, now: float) -> None:
        job = self.job
        if job is None:
            return
        if job.running and now - self._last_refresh >= 0.3:
            self._last_refresh = now
            self._refresh_table()
        if not job.running and not self._reported:
            from yoru_tracker.runtime.batch import failure_report

            self._reported = True
            self._refresh_table()
            if job.error is not None:
                self.app.report_error("Batch tracking failed", job.error)
            dpg.set_value("batch_report", failure_report(job.items))
            done = sum(1 for i in job.items if i.status == "done")
            failed = sum(1 for i in job.items if i.status == "failed")
            dpg.set_value("batch_status", f"{job.phase.capitalize()}: {done} done, {failed} failed.")
            self._buttons()

    def shutdown(self) -> None:
        if self.job is not None:
            self.job.stop(timeout=3.0)
