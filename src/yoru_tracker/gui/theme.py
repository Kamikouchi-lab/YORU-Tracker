# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""YORU's theme with the tracker's accent.

The base is YORU's own DearPyGui theme (``yoru.gui_base.apply_default_theme``),
so the two applications look like siblings.  On top of it every tracker
window gets a teal accent -- the colour of the motion trail in the logo --
on its buttons, sliders and check marks, so it is never mistaken for the
normal YORU window.
"""

from __future__ import annotations

from dataclasses import dataclass

import dearpygui.dearpygui as dpg

#: Trail teal, from the logo.
ACCENT = (72, 201, 190)
ACCENT_DIM = (28, 112, 118)
ACCENT_HOVER = (40, 140, 146)
#: YORU's moon.
MOON = (242, 216, 79)
TEXT_MUTED = (150, 165, 190)
TEXT_WARN = (255, 190, 110)
TEXT_ERROR = (255, 120, 120)
TEXT_OK = (130, 220, 160)


@dataclass(frozen=True)
class Themes:
    window: int
    primary_button: int
    big_button: int
    header: int
    muted: int
    warn: int
    error: int
    ok: int
    moon: int


def _text_theme(color):
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, color, category=dpg.mvThemeCat_Core)
    return theme


def apply_tracker_theme() -> Themes:
    from yoru.gui_base import apply_default_theme

    apply_default_theme()

    with dpg.theme() as window:
        with dpg.theme_component(dpg.mvAll):
            core = dpg.mvThemeCat_Core
            dpg.add_theme_color(dpg.mvThemeCol_Button, ACCENT_DIM, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ACCENT_HOVER, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (22, 90, 96), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, ACCENT, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, ACCENT, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (120, 225, 215), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, ACCENT_DIM, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_Header, (30, 86, 100), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, ACCENT_HOVER, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, ACCENT_DIM, category=core)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, (40, 95, 110), category=core)
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, (26, 52, 74), category=core)

    with dpg.theme() as primary:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (36, 150, 142), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (52, 178, 168), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (28, 120, 114), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (245, 250, 250), category=dpg.mvThemeCat_Core)

    with dpg.theme() as big:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (26, 44, 72), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (30, 86, 100), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, ACCENT_DIM, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Border, (60, 130, 140), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8, category=dpg.mvThemeCat_Core)

    return Themes(
        window=window,
        primary_button=primary,
        big_button=big,
        header=_text_theme(ACCENT),
        muted=_text_theme(TEXT_MUTED),
        warn=_text_theme(TEXT_WARN),
        error=_text_theme(TEXT_ERROR),
        ok=_text_theme(TEXT_OK),
        moon=_text_theme(MOON),
    )
