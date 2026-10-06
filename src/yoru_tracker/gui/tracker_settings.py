# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The tracker settings window.

Shows the settings worth changing day to day; the rest sit under "More" or
are reachable only through a YAML file.  Values set in a loaded file but not
shown here are kept, never reset to defaults behind the user's back.  Applying
does not re-run anything by itself: each view says what applying means there
(re-track the video, reset the live tracker).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable

import dearpygui.dearpygui as dpg

from yoru_tracker.core.config import ConfigError, TrackerConfig
from yoru_tracker.core.registry import available_modes, get_tracker_class, planned_modes
from yoru_tracker.gui import widgets
from yoru_tracker.gui.state import AppState
from yoru_tracker.gui.theme import Themes

logger = logging.getLogger(__name__)

WINDOW = "tracker_settings_window"

# (section, field, label, kind, help)
_FIELDS = (
    ("lifecycle", "population", "Number of animals (0 = ?)", int,
     "How many animals are always in view, if known (e.g. 2 flies in a chamber). "
     "Then no ID is ever retired or added beyond this, and an animal that jumps "
     "or reappears far away gets its own ID back. 0: animals may come and go."),
    ("lifecycle", "max_age", "Max age (frames)", int,
     "Frames a track may go undetected before it is retired. Within this it keeps its ID."),
    ("lifecycle", "min_hits", "Min hits", int,
     "Detections a new track needs before it gets an ID (1 = immediately)."),
    ("association", "max_distance", "Max distance (px)", float,
     "Furthest a detection may be from where a track is predicted to be."),
    ("association", "distance_weight", "Distance weight", float, "Cost of centre distance."),
    ("association", "iou_weight", "IoU weight", float, "Cost of poor box overlap."),
    ("association", "axis_weight", "Axis weight", float,
     "Cost of disagreeing long axes (elongated boxes only; head and tail are not told apart)."),
    ("association", "class_aware", "Match within class only", bool,
     "Never give a track a detection of another class."),
    ("association", "recovery", "Recovery pass for lost tracks", bool,
     "Look for lost tracks again with a gate that widens the longer they are missing."),
    ("kalman", "enabled", "Motion prediction (Kalman)", bool,
     "Predict where each track moves; off = expect it where it was last seen."),
)

_MORE = (
    ("association", "size_weight", "Size weight", float, "Cost of differing box areas."),
    ("association", "min_iou", "Min IoU", float,
     "Reject pairs with less overlap than this whose centres are over half a body apart (0 = off)."),
    ("association", "recovery_gate_scale", "Recovery gate (x max dist.)", float,
     "How far the recovery gate may widen."),
    ("kalman", "process_noise", "Process noise (x size)", float, "Expected acceleration."),
    ("kalman", "measurement_noise", "Measurement noise (x size)", float, "Detector jitter."),
    (None, "log_events", "Log every track event", bool,
     "Write created/lost/recovered/retired events to the log file."),
)


def _tag(section, name):
    return f"setting_{section or 'root'}_{name}"


class TrackerSettingsWindow:
    def __init__(self, state: AppState, themes: Themes, report_error: Callable):
        self.state = state
        self.themes = themes
        self.report_error = report_error

    # -- building -------------------------------------------------------

    def build(self) -> None:
        with dpg.window(label="Tracker settings", tag=WINDOW, show=False, width=520,
                        height=640, no_collapse=True, pos=(160, 80)):
            dpg.bind_item_theme(WINDOW, self.themes.window)
            with dpg.group(horizontal=True):
                dpg.add_text("Mode")
                dpg.add_combo(available_modes(), tag="setting_mode", width=180,
                              callback=lambda: self._describe_mode())
            dpg.add_text("", tag="setting_mode_info", wrap=490)
            dpg.bind_item_theme("setting_mode_info", self.themes.muted)
            planned = planned_modes()
            if planned:
                note = dpg.add_text("Planned: " + ", ".join(sorted(planned)), wrap=490)
                dpg.bind_item_theme(note, self.themes.muted)
            dpg.add_spacer(height=4)
            widgets.heading("Settings", self.themes.header)
            for spec in _FIELDS:
                self._field(*spec)
            with dpg.collapsing_header(label="More", default_open=False):
                for spec in _MORE:
                    self._field(*spec)
            dpg.add_spacer(height=6)
            dpg.add_text("", tag="setting_status", wrap=490)
            with dpg.group(horizontal=True):
                apply_btn = dpg.add_button(label="Apply", width=90, callback=lambda: self.apply())
                dpg.bind_item_theme(apply_btn, self.themes.primary_button)
                dpg.add_button(label="Load...", width=80, callback=lambda: self.load_file())
                dpg.add_button(label="Save...", width=80, callback=lambda: self.save_file())
                dpg.add_button(label="Defaults", width=80, callback=lambda: self.show_config(TrackerConfig()))
                dpg.add_button(label="Close", width=70, callback=lambda: dpg.hide_item(WINDOW))
        self.show_config(self.state.tracker_config)

    def _field(self, section, name, label, kind, help_text):
        tag = _tag(section, name)
        with dpg.group(horizontal=True):
            if kind is bool:
                dpg.add_checkbox(label=label, tag=tag)
            else:
                dpg.add_text(f"{label:<28}")
                if kind is int:
                    dpg.add_input_int(tag=tag, width=120, step=1, min_value=0, min_clamped=True)
                else:
                    dpg.add_input_float(tag=tag, width=120, step=0, format="%.3f")
        with dpg.tooltip(tag):
            dpg.add_text(help_text, wrap=360)

    # -- values ---------------------------------------------------------

    def show_config(self, config: TrackerConfig) -> None:
        self._base = config
        dpg.set_value("setting_mode", config.mode)
        for section, name, *_ in _FIELDS + _MORE:
            owner = getattr(config, section) if section else config
            dpg.set_value(_tag(section, name), getattr(owner, name))
        self._describe_mode()
        self._status("")

    def read_config(self) -> TrackerConfig:
        """The configuration the widgets describe; raises ConfigError if invalid."""
        base = self._base
        changes = {"mode": dpg.get_value("setting_mode")}
        sections = {}
        for section, name, _label, kind, _help in _FIELDS + _MORE:
            value = kind(dpg.get_value(_tag(section, name)))
            if kind is float:
                # The widget holds a float32: 0.1 comes back as 0.10000000149.
                value = float(f"{value:.6g}")
            if section is None:
                changes[name] = value
            else:
                sections.setdefault(section, {})[name] = value
        for section, values in sections.items():
            changes[section] = dataclasses.replace(getattr(base, section), **values)
        return dataclasses.replace(base, **changes).validate()

    def _describe_mode(self) -> None:
        mode = dpg.get_value("setting_mode")
        try:
            info = get_tracker_class(mode).info
            text = f"{info.name} {info.version}: {info.description}"
        except Exception as exc:
            text = str(exc)
        dpg.set_value("setting_mode_info", text)

    def _status(self, text, theme=None) -> None:
        dpg.set_value("setting_status", text)
        dpg.bind_item_theme("setting_status", theme or self.themes.muted)

    # -- actions --------------------------------------------------------

    def open(self) -> None:
        self.show_config(self.state.tracker_config)
        dpg.show_item(WINDOW)
        dpg.focus_item(WINDOW)

    def apply(self) -> bool:
        try:
            config = self.read_config()
        except ConfigError as exc:
            self._status("Not applied: " + "; ".join(exc.problems), self.themes.error)
            return False
        self.state.set_tracker_config(config)
        self._base = config
        self._status("Applied.", self.themes.ok)
        return True

    def load_file(self) -> None:
        path = widgets.ask_open_file("Load tracker settings", widgets.CONFIG_TYPES)
        if not path:
            return
        try:
            config = TrackerConfig.load(path)
        except Exception as exc:
            self.report_error(f"Could not load tracker settings from {path}", exc)
            return
        logger.info("Configuration loaded from %s", path)
        self.show_config(config)
        self._status(f"Loaded {path}. Press Apply to use it.", self.themes.warn)

    def save_file(self) -> None:
        try:
            config = self.read_config()
        except ConfigError as exc:
            self._status("Not saved: " + "; ".join(exc.problems), self.themes.error)
            return
        path = widgets.ask_save_file("Save tracker settings", widgets.CONFIG_TYPES, ".yaml")
        if not path:
            return
        try:
            config.save(path)
        except Exception as exc:
            self.report_error(f"Could not save tracker settings to {path}", exc)
            return
        self._status(f"Saved {path}.", self.themes.ok)
