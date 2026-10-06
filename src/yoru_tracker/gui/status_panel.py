# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Tracking status and track inspection, shared by the live and video views."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import dearpygui.dearpygui as dpg

from yoru_tracker.core.types import TrackEvent, TrackingResult
from yoru_tracker.drawing.overlays import track_color
from yoru_tracker.gui import widgets

_TRACK_COLUMNS = ("ID", "state", "class", "conf", "hits", "missed", "vx", "vy", "cost")


def _ids(ids) -> str:
    ids = list(ids)
    if not ids:
        return "-"
    text = ", ".join(str(i) for i in ids[:24])
    return text + (f" ... (+{len(ids) - 24})" if len(ids) > 24 else "")


class TrackTable:
    """One row per track in a frame: what the tracker thinks of each animal."""

    def __init__(self, tag: str, height: int = 220):
        self.tag = tag
        self.height = height

    def build(self) -> None:
        with dpg.table(tag=self.tag, header_row=True, resizable=True, borders_innerH=True,
                       borders_outerH=True, borders_innerV=True, borders_outerV=True,
                       row_background=True, scrollY=True, height=self.height,
                       policy=dpg.mvTable_SizingStretchProp):
            for name in _TRACK_COLUMNS:
                dpg.add_table_column(label=name)

    def show(self, result: Optional[TrackingResult]) -> None:
        rows, colors = [], []
        for t in (result.tracked if result else ()):
            vx, vy = t.velocity
            rows.append((
                t.track_id, t.track_state.value, t.class_name or t.class_id,
                f"{t.track_confidence:.2f}", t.hits, t.missed_frames,
                f"{vx:+.1f}", f"{vy:+.1f}",
                "" if t.association_cost is None else f"{t.association_cost:.2f}",
            ))
            b, g, r = track_color(t.track_id)
            colors.append((r, g, b, 255) if not t.predicted else (150, 160, 180, 255))
        widgets.set_table_rows(self.tag, rows, colors)


class EventLog:
    """The most recent track events, newest first."""

    def __init__(self, tag: str, length: int = 12):
        self.tag = tag
        self._events = deque(maxlen=length)

    def build(self, height: int = 140) -> None:
        dpg.add_input_text(tag=self.tag, multiline=True, readonly=True, width=-1, height=height)

    def clear(self) -> None:
        self._events.clear()
        dpg.set_value(self.tag, "")

    def add(self, events: Iterable[TrackEvent]) -> None:
        added = False
        for e in events:
            who = "-" if e.track_id is None else f"#{e.track_id}"
            self._events.appendleft(f"frame {e.frame_id:>6}  {who:>5}  {e.kind.value:<9} {e.detail}")
            added = True
        if added:
            dpg.set_value(self.tag, "\n".join(self._events))

    def show(self, events: Iterable[TrackEvent]) -> None:
        self._events.clear()
        self.add(events)
        if not self._events:
            dpg.set_value(self.tag, "")
