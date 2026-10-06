# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Small pieces every view uses: frame textures, file dialogs, headings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

BACKGROUND_BGR = (42, 24, 18)  # the window colour, so letterbox bars disappear

VIDEO_TYPES = (("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.mpg *.mpeg"), ("All files", "*.*"))
MODEL_TYPES = (("Detector models", "*.pt *.pth *.onnx"), ("All files", "*.*"))
CONFIG_TYPES = (("Tracker settings", "*.yaml *.yml *.json"), ("All files", "*.*"))


def letterbox(frame, width: int, height: int):
    """*frame* scaled to fit ``width x height``, centred on the window colour."""
    canvas = np.empty((height, width, 3), np.uint8)
    canvas[:] = BACKGROUND_BGR
    if frame is None or frame.size == 0:
        return canvas
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    x0, y0 = (width - new_w) // 2, (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def texture_data(frame_bgr) -> np.ndarray:
    """A BGR frame as the flat float32 RGBA data a dynamic texture takes."""
    rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    return (rgba.astype(np.float32) * (1.0 / 255.0)).ravel()


def blank_texture(width: int, height: int) -> np.ndarray:
    return texture_data(letterbox(None, width, height))


def fit_size(avail_w: float, avail_h: float, aspect: float) -> Tuple[int, int]:
    """Largest ``(w, h)`` of the given aspect inside the available area."""
    avail_w, avail_h = max(32.0, avail_w), max(32.0, avail_h)
    if avail_w / avail_h > aspect:
        return int(avail_h * aspect), int(avail_h)
    return int(avail_w), int(avail_w / aspect)


def heading(text: str, theme) -> int:
    item = dpg.add_text(text)
    dpg.bind_item_theme(item, theme)
    dpg.add_separator()
    return item


# -- file dialogs (tkinter, as YORU's own windows use) ----------------------

def _tk_root():
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def _initial_dir(current: Optional[str]) -> Optional[str]:
    if not current:
        return None
    p = Path(current)
    return str(p if p.is_dir() else p.parent) if p.exists() or p.parent.exists() else None


def ask_open_file(title: str, filetypes: Sequence[Tuple[str, str]] = (("All files", "*.*"),),
                  current: Optional[str] = None) -> str:
    from tkinter import filedialog

    root = _tk_root()
    try:
        return filedialog.askopenfilename(title=title, filetypes=list(filetypes),
                                          initialdir=_initial_dir(current)) or ""
    finally:
        root.destroy()


def ask_directory(title: str, current: Optional[str] = None) -> str:
    from tkinter import filedialog

    root = _tk_root()
    try:
        return filedialog.askdirectory(title=title, initialdir=_initial_dir(current)) or ""
    finally:
        root.destroy()


def ask_save_file(title: str, filetypes: Sequence[Tuple[str, str]], default_ext: str,
                  current: Optional[str] = None) -> str:
    from tkinter import filedialog

    root = _tk_root()
    try:
        return filedialog.asksaveasfilename(title=title, filetypes=list(filetypes),
                                            defaultextension=default_ext,
                                            initialdir=_initial_dir(current)) or ""
    finally:
        root.destroy()


def set_table_rows(table, rows: Iterable[Sequence[object]], colors: Optional[Sequence] = None) -> None:
    """Replace every row of *table*."""
    for child in dpg.get_item_children(table, 1) or []:
        dpg.delete_item(child)
    for i, row in enumerate(rows):
        with dpg.table_row(parent=table):
            for cell in row:
                item = dpg.add_text(str(cell))
                if colors is not None and colors[i] is not None:
                    dpg.configure_item(item, color=colors[i])


class BackgroundTask:
    """Run *fn* on a thread; poll ``done`` from the render loop.

    The GUI never waits on one: it checks ``done`` each frame and then reads
    ``result`` or ``error``.  Errors are kept, not raised, so the view that
    started the task decides how to report them.
    """

    def __init__(self, name: str, fn):
        import threading

        self.name = name
        self.result = None
        self.error = None
        self._done = threading.Event()

        def work():
            try:
                self.result = fn()
            except BaseException as exc:  # handed to the GUI to report
                self.error = exc
            finally:
                self._done.set()

        self._thread = threading.Thread(target=work, name=f"yoru-tracker-{name}", daemon=True)
        self._thread.start()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def join(self, timeout=None) -> None:
        self._thread.join(timeout)
