# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Build the YORU Tracker logo from the YORU logo.

    python tools/make_logo.py [path/to/YORU_logo.png]

The YORU wordmark is kept as it is -- same letters, same moon, same mouse and
fly -- so the application reads as part of the YORU family.  Below it,
"TRACKER" and a tracking motif: a motion trail of past positions running into
a bounding box with an ID tag.  Two variants are written to
``src/yoru_tracker/assets/``: ``logo_dark.png`` (transparent, light ink, for
the GUI's dark theme) and ``logo_light.png`` (for documents).

The wordmark is recoloured by writing every pixel as a mix of the logo's
three inks (paper white, navy, moon yellow) and remixing it with new ones, so
the anti-aliased edges stay smooth in the new colours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "yoru_tracker" / "assets"

PAPER = np.array([255.0, 255.0, 255.0])
NAVY = np.array([1.0, 27.0, 67.0])
YELLOW = np.array([242.0, 216.0, 79.0])

#: Accent of the tracker: the trail colour, used nowhere in YORU itself.
TEAL_DARK_BG = (72, 201, 190)
TEAL_LIGHT_BG = (0, 133, 128)


def _default_logo() -> Path:
    import yoru

    return Path(yoru.__file__).resolve().parents[1] / "logos" / "YORU_logo.png"


def _unmix(rgb: np.ndarray) -> np.ndarray:
    """Weights of (paper, navy, yellow) per pixel, least squares, clipped."""
    basis = np.stack([PAPER, NAVY, YELLOW], axis=1)  # 3x3
    weights = np.linalg.solve(basis, rgb.reshape(-1, 3).T).T
    weights = np.clip(weights, 0.0, None)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    return weights.reshape(*rgb.shape[:2], 3)


def _wordmark(src: Path, navy_to, yellow_to, transparent: bool) -> Image.Image:
    image = Image.open(src).convert("RGB")
    w, h = image.size
    # The tagline sits in the bottom sixth; the wordmark is everything above.
    image = image.crop((0, 0, w, int(h * 0.80)))
    rgb = np.asarray(image, dtype=np.float64)
    weights = _unmix(rgb)
    ink = weights[..., 1:2] * np.array(navy_to, float) + weights[..., 2:3] * np.array(yellow_to, float)
    coverage = weights[..., 1] + weights[..., 2]
    if transparent:
        color = np.where(coverage[..., None] > 1e-3, ink / np.maximum(coverage[..., None], 1e-3), 0)
        alpha = np.clip(coverage, 0.0, 1.0) * 255.0
        out = np.dstack([color, alpha]).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGBA")
    out = (weights[..., 0:1] * PAPER + ink).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB").convert("RGBA")


def _font(size: int):
    for name in ("bahnschrift.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(name, size)
        except OSError:
            continue
        try:
            font.set_variation_by_name("SemiBold")
        except Exception:
            pass
        return font
    return ImageFont.load_default()


def _compose(mark: Image.Image, ink, accent, box_color, tag_ink, background) -> Image.Image:
    width = 1600
    scale = width / mark.width
    mark = mark.resize((width, int(mark.height * scale)), Image.LANCZOS)
    band = 250
    canvas = Image.new("RGBA", (width, mark.height + band), background)
    canvas.alpha_composite(mark, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # "TRACKER", letter-spaced, under the wordmark and right of the motif.
    font = _font(140)
    letters = "TRACKER"
    spacing = 44
    boxes = [draw.textbbox((0, 0), c, font=font) for c in letters]
    widths = [b[2] - b[0] for b in boxes]
    glyph_top = min(b[1] for b in boxes)
    glyph_bottom = max(b[3] for b in boxes)
    text_w = sum(widths) + spacing * (len(letters) - 1)
    motif_w = 380
    x = (width - text_w - motif_w) // 2 + motif_w
    top = mark.height + 50 - glyph_top
    text_x0 = x
    for c, cw, b in zip(letters, widths, boxes):
        draw.text((x - b[0], top), c, font=font, fill=ink)
        x += cw + spacing
    mid = top + (glyph_top + glyph_bottom) / 2

    # The motif: past positions along a smooth trail, running into the box
    # around the current position, labelled with its ID -- what the
    # application draws on every animal.
    bw, bh = 96, 72
    bx = text_x0 - 70 - bw / 2
    by = mid + 6
    x0 = text_x0 - motif_w
    xs = np.linspace(x0, bx - bw / 2 - 18, 8)
    ys = by + 30 * np.sin((xs - x0) / (xs[-1] - x0) * np.pi * 1.25 + 0.4)
    ys += by - ys[-1]
    path = [(float(a), float(b)) for a, b in zip(xs, ys)]
    trail = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(trail)
    tdraw.line(path, fill=(*accent[:3], 150), width=6, joint="curve")
    for i, (px, py) in enumerate(path):
        r = 5 + 1.2 * i
        alpha = int(70 + 185 * i / (len(path) - 1))
        tdraw.ellipse((px - r, py - r, px + r, py + r), fill=(*accent[:3], alpha))
    canvas.alpha_composite(trail)
    draw.rounded_rectangle((bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2),
                           radius=12, outline=box_color, width=8)
    draw.ellipse((bx - 11, by - 11, bx + 11, by + 11), fill=accent)
    tag = _font(44)
    tb = draw.textbbox((0, 0), "ID", font=tag)
    tag_w, tag_h = tb[2] - tb[0] + 28, tb[3] - tb[1] + 18
    tx, ty = bx - bw / 2, by - bh / 2 - tag_h + 4
    draw.rounded_rectangle((tx, ty, tx + tag_w, ty + tag_h), radius=8, fill=box_color)
    draw.text((tx + 14 - tb[0], ty + 9 - tb[1]), "ID", font=tag, fill=tag_ink)
    return canvas


def main(argv) -> None:
    src = Path(argv[1]) if len(argv) > 1 else _default_logo()
    ASSETS.mkdir(parents=True, exist_ok=True)

    dark_mark = _wordmark(src, navy_to=(236, 240, 248), yellow_to=tuple(YELLOW), transparent=True)
    dark = _compose(dark_mark, ink=(236, 240, 248, 255), accent=(*TEAL_DARK_BG, 255),
                    box_color=(242, 216, 79, 255), tag_ink=(16, 22, 40, 255),
                    background=(0, 0, 0, 0))
    dark.save(ASSETS / "logo_dark.png", optimize=True)

    light_mark = _wordmark(src, navy_to=tuple(NAVY), yellow_to=tuple(YELLOW), transparent=False)
    light = _compose(light_mark, ink=(1, 27, 67, 255), accent=(*TEAL_LIGHT_BG, 255),
                     box_color=(1, 27, 67, 255), tag_ink=(255, 255, 255, 255),
                     background=(255, 255, 255, 255))
    light.convert("RGB").save(ASSETS / "logo_light.png", optimize=True)
    print(f"wrote {ASSETS / 'logo_dark.png'} and {ASSETS / 'logo_light.png'}")


if __name__ == "__main__":
    main(sys.argv)
