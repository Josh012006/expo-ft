"""Frames of the demo / eval / replay videos: two cameras on top, the taxels below, a text banner under it all.

    +---------------------+---------------------+
    |  exterior camera    |   wrist camera      |   224 px
    +---------------------+---------------------+
    |  taxels: left 7x4 | right 7x4 | readings  |   224 px   <- compose_frame()  (448 x 448)
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
CONTACT_N = 0.05        # a zone above this force counts as "in contact"

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


def panel_kwargs(obs):
    """What the taxel panel shows, taken from one observation of the server: the taxel grids (None without taxels), the
    norm of the hand force sensor (N) and the finger motor effort (N.m, Robotiq only). Missing entries are None."""
    tactile, force, effort = obs.get("extra/tactile"), obs.get("extra/ft_force"), obs.get("extra/gripper_effort")
    return {
        "tactile": None if tactile is None else np.asarray(tactile, dtype=np.float32),
        "hand_force": None if force is None else float(np.linalg.norm(np.asarray(force, dtype=np.float64).reshape(-1)[:3])),
        "gripper_effort": None if effort is None else float(np.asarray(effort, dtype=np.float64).reshape(-1)[0]),
    }


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)       # Pillow >= 10.1; older versions: the small bitmap font
    except TypeError:
        return ImageFont.load_default()


def tactile_panel(tactile, width=448, height=PANEL_H, vmax=TACTILE_VMAX, hand_force=None, gripper_effort=None):
    """(height, width, 3) uint8 panel: the two taxel grids (square cells, no gap: the pad is 26 x 40 mm for 4 x 7 zones)
    in the middle, a colour bar (0 .. vmax N), readings on the right. `tactile` None -> placeholder."""
    img = np.full((height, width, 3), 12, dtype=np.uint8)
    Image, ImageDraw = _pil()
    if tactile is None:
        if Image is not None:
            pil = Image.fromarray(img)
            ImageDraw.Draw(pil).text((width // 2 - 62, height // 2 - 5), "no tactile data (robot without taxels)", fill=(130, 130, 130))
            img = np.asarray(pil)
        return img
    t = np.asarray(tactile, dtype=np.float64)
    n_rows, n_cols = t.shape[1], t.shape[2]
    cell = max(8, (height - 70) // n_rows)                  # 22 px for a 224 px panel
    gw, gh, gap, x_first, top = n_cols * cell, n_rows * cell, 18, 14, 40
    x_grid = [x_first, x_first + gw + gap]
    for side in (0, 1):
        colours = colormap(t[side] / max(vmax, 1e-9))       # (rows, cols, 3)
        for r in range(n_rows):
            y1 = top + gh - r * cell                        # row 0 (fingertip) at the bottom
            for c in range(n_cols):
                img[y1 - cell:y1, x_grid[side] + c * cell:x_grid[side] + (c + 1) * cell] = colours[r, c]
        img[top - 1, x_grid[side]:x_grid[side] + gw] = img[top + gh, x_grid[side]:x_grid[side] + gw] = 70   # frame
    bar_x, bar_w = x_grid[1] + gw + 10, max(8, round(height * 12 / 224))         # colour bar: same height as the grids
    img[top:top + gh, bar_x:bar_x + bar_w] = colormap(np.linspace(1.0, 0.0, gh))[:, None, :]      # vmax at the top
    img[top - 1, bar_x:bar_x + bar_w] = img[top + gh, bar_x:bar_x + bar_w] = 70
    if Image is None:
        return img
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    big, small = _font(max(10, round(height * 15 / 224))), _font(max(9, round(height * 13 / 224)))
    for side, name in enumerate(("LEFT", "RIGHT")):
        draw.text((x_grid[side] + gw // 2 - 4 * len(name) * height // 224 - 2, 14), name, fill=(255, 255, 255), font=big)
    in_contact = [int((t[i] > CONTACT_N).sum()) for i in (0, 1)]
    if all(in_contact):
        state, colour = "BOTH FINGERS", (120, 230, 140)
    elif any(in_contact):
        state, colour = "ONE FINGER ONLY", (255, 190, 90)
    else:
        state, colour = "NO CONTACT", (150, 150, 150)
    lines = [(state, colour),
             (f"zones max  L {t[0].max():4.1f}  R {t[1].max():4.1f} N", (230, 230, 230)),
             (f"zones sum  L {t[0].sum():4.1f}  R {t[1].sum():4.1f} N", (230, 230, 230)),
             (f"in contact L {in_contact[0]}/{n_rows * n_cols}  R {in_contact[1]}/{n_rows * n_cols}", (230, 230, 230))]
    if hand_force is not None:
        lines.append((f"hand force  {hand_force:5.2f} N", (230, 230, 230)))
    if gripper_effort is not None:
        lines.append((f"gripper effort  {gripper_effort:5.1f} Nm", (230, 230, 230)))
    x_lab = bar_x + bar_w + 4
    draw.text((bar_x - 2, top - 20), "N", fill=(130, 150, 255), font=small)
    for frac in (0.0, 0.5, 1.0):                                 # ticks: 0, vmax / 2, vmax (the colours saturate above vmax)
        y = top + round(gh * (1.0 - frac))
        draw.line([(bar_x + bar_w, y), (bar_x + bar_w + 3, y)], fill=(130, 150, 255))
        draw.text((x_lab, y - 7), f"{vmax * frac:g}", fill=(130, 150, 255), font=small)
    x_text, step = x_lab + 4 * len(f"{vmax:g}") * round(height / 224) + 34, round(height * 22 / 224)
    for i, (text, col) in enumerate(lines):
        draw.text((x_text, 44 + i * step), text, fill=col, font=small)
    return np.asarray(pil)


def compose_frame(ext, wrist, tactile, vmax=TACTILE_VMAX, hand_force=None, gripper_effort=None):
    """Exterior | wrist side by side on top, the taxel panel below: (2 * h, 2 * w, 3) uint8, 448 x 448 for 224 px cameras."""
    top = np.concatenate([np.asarray(ext, dtype=np.uint8), np.asarray(wrist, dtype=np.uint8)], axis=1)
    panel = tactile_panel(tactile, width=top.shape[1], height=top.shape[0], vmax=vmax,
                          hand_force=hand_force, gripper_effort=gripper_effort)
    return np.concatenate([top, panel], axis=0)


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
