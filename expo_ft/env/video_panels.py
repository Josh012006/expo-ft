"""Frames of the demo / eval / replay videos: two cameras on top, the taxels below, a text banner under it all.

    +---------------------+---------------------+
    |  exterior camera    |   wrist camera      |   224 px
    +---------------------+---------------------+
    |  taxels: left finger 7x4 | right finger 7x4 |   224 px   <- compose_frame()  (448 x 448)
    +-------------------------------------------+
    |  banner: 3 lines of text                  |    48 px   <- text_banner()    (448 x 496, a multiple of 16)
    +-------------------------------------------+

One module for the three video writers (isaaclab_env.py, replay_demos.py, collect_waypoint_demos.py), so that all the
videos of the project look the same. No Isaac Sim, only numpy (and PIL for the text, drawn only if it is installed).

The taxel array is the server's observation `extra/tactile`: shape (2, rows, cols), [0] left finger, [1] right finger,
net contact force of each zone in N, row 0 = fingertip end (drawn at the BOTTOM of each grid, like looking at the pad
from the palm with the fingertips down). A robot without taxels gives `None`: the panel then says so, and the video keeps
its 448 x 448 size, so that videos of different robots can be compared side by side.
"""
from __future__ import annotations

import numpy as np

PANEL_H = 224
BANNER_H = 48
TACTILE_VMAX = 5.0      # N per zone at the top of the colour scale (fixed: frames of one video, or of two videos, compare)

_KEYS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_COLORS = np.array([(18, 18, 40), (40, 80, 220), (40, 200, 120), (240, 220, 50), (230, 40, 30)], dtype=np.float64)


def colormap(x):
    """x in [0, 1] (any shape) -> uint8 colours (..., 3): dark blue (no contact) .. green .. yellow .. red."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.stack([np.interp(x, _KEYS, _COLORS[:, c]) for c in range(3)], axis=-1).astype(np.uint8)


def _pil():
    try:
        from PIL import Image, ImageDraw
        return Image, ImageDraw
    except ImportError:
        return None, None


def tactile_stats(tactile):
    """(max over all zones, max left, max right) in N, or None without data."""
    if tactile is None:
        return None
    t = np.asarray(tactile, dtype=np.float64)
    return float(t.max()), float(t[0].max()), float(t[1].max())


def tactile_panel(tactile, width=448, height=PANEL_H, vmax=TACTILE_VMAX):
    """(height, width, 3) uint8 image of the two taxel grids. `tactile` None -> placeholder."""
    img = np.full((height, width, 3), 12, dtype=np.uint8)
    Image, ImageDraw = _pil()
    texts = []                                            # (x, y, text, colour), drawn at the end with PIL
    if tactile is None:
        texts.append((width // 2 - 62, height // 2 - 5, "no tactile data (robot without taxels)", (130, 130, 130)))
    else:
        t = np.asarray(tactile, dtype=np.float64)
        n_rows, n_cols = t.shape[1], t.shape[2]
        half = width // 2
        top, bottom = 22, height - 22                    # grid area: below the title, above the colour bar
        cell_h = (bottom - top) // n_rows
        cell_w = (half - 28) // n_cols
        for side, name in enumerate(("LEFT", "RIGHT")):
            x0 = side * half + (half - cell_w * n_cols) // 2
            y_bottom = top + cell_h * n_rows
            colours = colormap(t[side] / max(vmax, 1e-9))            # (rows, cols, 3)
            for r in range(n_rows):
                y1 = y_bottom - r * cell_h                          # row 0 (fingertip) at the bottom
                for c in range(n_cols):
                    xa = x0 + c * cell_w
                    img[y1 - cell_h + 1:y1 - 1, xa + 1:xa + cell_w - 1] = colours[r, c]
            texts.append((x0, 5, f"{name}  max {t[side].max():5.2f} N", (230, 230, 230)))
        # colour bar: 0 .. vmax
        bar_w = 140
        bx = (width - bar_w) // 2
        img[height - 14:height - 8, bx:bx + bar_w] = colormap(np.linspace(0, 1, bar_w))[None]
        texts.append((bx - 30, height - 17, "0 N", (160, 160, 160)))
        texts.append((bx + bar_w + 6, height - 17, f"{vmax:g} N", (160, 160, 160)))
    if Image is not None and texts:
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        for x, y, s, colour in texts:
            draw.text((x, y), s, fill=colour)
        img = np.asarray(pil)
    return img


def compose_frame(ext, wrist, tactile, vmax=TACTILE_VMAX):
    """Exterior | wrist side by side on top, the taxel panel below: (2 * h, 2 * w, 3) uint8, 448 x 448 for 224 px cameras."""
    top = np.concatenate([np.asarray(ext, dtype=np.uint8), np.asarray(wrist, dtype=np.uint8)], axis=1)
    return np.concatenate([top, tactile_panel(tactile, width=top.shape[1], height=top.shape[0], vmax=vmax)], axis=0)


def text_banner(lines, width, height=BANNER_H):
    """(height, width, 3) uint8 banner with up to three lines [(text, (r, g, b)), ...]; None if PIL is missing."""
    Image, ImageDraw = _pil()
    if Image is None:
        return None
    banner = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(banner)
    step = max((height - 8) // max(len(lines), 1), 11)
    for i, (text, colour) in enumerate(lines):
        draw.text((4, 4 + i * step), text, fill=colour)
    return np.asarray(banner)


def add_banner(frame, lines, height=BANNER_H):
    """frame with the banner underneath (frame alone when PIL is missing)."""
    banner = text_banner(lines, frame.shape[1], height)
    return frame if banner is None else np.concatenate([frame, banner], axis=0)
