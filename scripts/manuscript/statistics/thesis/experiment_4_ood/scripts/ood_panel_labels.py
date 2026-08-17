#!/usr/bin/env python3
r"""Shared label rendering for the OOD TP/FP/FN qualitative panels.

Why this module exists
----------------------
The first version of these figures burned the labels in at a fixed cv2 font scale
(0.55 -> ~11 px cap height) on a ~1500 px canvas. What matters on the printed page is
not the pixel size but the *ratio* of text height to figure width, because that ratio
is what survives ``\includegraphics``:

    printed_pt = text_em_px / figure_width_px * textwidth_pt * latex_width_frac

For the thesis (a4paper, inner=3.5cm outer=3cm -> textwidth 14.5cm = 412.6pt) the old
labels came out at ~3 pt on a full-width figure and ~2.3 pt inside a 0.48\linewidth
subfigure. Unreadable in print.

So every label here is sized *backwards from the target printed point size*: pick the
em size in pixels that renders as TARGET_PT once LaTeX scales the PNG. That makes the
labels come out the same physical size in the thesis regardless of canvas resolution,
which is why ``latex_width_frac`` must be passed per figure (1.0 for a full-width
figure, 0.48 for the MassID45 subfigures).

Long labels cannot simply be grown -- they would overflow the panel -- so they are
wrapped/split into short lines (``wrap_chunks``) and only then shrunk as a last resort
(``fit_em``). Text is DejaVu Sans Bold via PIL rather than the cv2 Hershey stroke font,
which holds up much better at print sizes.

All functions take and return OpenCV-style BGR uint8 arrays.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- print geometry
TEXTWIDTH_PT = 412.6   # thesis \textwidth: 14.5cm (a4 - 3.5cm inner - 3cm outer)
TARGET_PT = 10.0       # primary labels (model name + counts): ~caption size
SECONDARY_PT = 8.0     # figure caption strip + legend: footnote size
MIN_EM_PX = 6.0

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SEP_GRAY = (180, 180, 180)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]
FONT_FILE = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)

_cache = {}


def font(em_px):
    em_px = max(int(round(em_px)), 4)
    if em_px not in _cache:
        _cache[em_px] = (ImageFont.truetype(FONT_FILE, em_px) if FONT_FILE
                         else ImageFont.load_default())
    return _cache[em_px]


def em_for(fig_w, latex_width_frac, target_pt=TARGET_PT):
    """Em size in px that prints at `target_pt` for a figure `fig_w` px wide that
    LaTeX renders at `latex_width_frac` x \\textwidth."""
    return max(MIN_EM_PX, target_pt / (TEXTWIDTH_PT * latex_width_frac) * fig_w)


def printed_pt(em_px, fig_w, latex_width_frac):
    """Inverse of `em_for` -- what a given em size will actually measure in print."""
    return em_px / fig_w * TEXTWIDTH_PT * latex_width_frac


def text_w(s, em_px):
    b = font(em_px).getbbox(s)
    return b[2] - b[0]


def fit_em(lines, avail_w, em_px, min_em=MIN_EM_PX):
    """Shrink `em_px` until the widest line fits `avail_w`. Last resort only -- prefer
    splitting long labels into more lines so the print size stays on target."""
    lines = [s for s in lines if s]
    while em_px > min_em and lines and max(text_w(s, em_px) for s in lines) > avail_w:
        em_px *= 0.97
    return em_px


def wrap_chunks(chunks, avail_w, em_px, joiner=" | "):
    """Greedily pack ' | '-separated chunks into as few lines as fit `avail_w` at
    `em_px`, so the text stays at its target print size instead of being shrunk."""
    lines, cur = [], ""
    for c in chunks:
        cand = c if not cur else cur + joiner + c
        if cur and text_w(cand, em_px) > avail_w:
            lines.append(cur)
            cur = c
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _band(width, height, bg):
    return Image.new("RGB", (width, max(height, 1)), bg)


def _to_bgr(pil_img):
    return np.array(pil_img)[:, :, ::-1].copy()


def text_band(width, lines, em_px, pad_frac=0.22, line_frac=1.22, bg=WHITE, fg=BLACK):
    """A white label strip of `width` px holding `lines` at `em_px`."""
    em = int(round(em_px))
    f = font(em)
    line_h = int(round(em * line_frac))
    pad = int(round(em * pad_frac))
    img = _band(width, 2 * pad + len(lines) * line_h, bg)
    d = ImageDraw.Draw(img)
    for i, s in enumerate(lines):
        d.text((pad + int(round(em * 0.15)), pad + i * line_h), s, font=f, fill=fg)
    return _to_bgr(img)


def _legend_metrics(entries, em_px):
    """(per-entry widths, minimum inter-entry gap) at `em_px`."""
    em = int(round(em_px))
    sw = int(round(em * 0.95))
    gap = int(round(em * 0.45))
    return [sw + gap + text_w(t, em) for _, t in entries], int(round(em * 0.9))


def legend_fits(width, entries, em_px, pad_frac=0.30):
    w, min_gap = _legend_metrics(entries, em_px)
    pad = int(round(em_px * pad_frac))
    return sum(w) + (len(entries) - 1) * min_gap <= width - 2 * pad


def legend_band(width, entries, em_px, pad_frac=0.30, bg=WHITE, fg=BLACK):
    """One-row colour key. `entries` is [(bgr_colour, label), ...]. Entries are measured
    and the slack spread between them, so the last label can never run off the canvas
    (equal-width columns used to clip it on the narrow figures). Swatches get a thin
    black outline so yellow stays visible on white."""
    while em_px > MIN_EM_PX and not legend_fits(width, entries, em_px, pad_frac):
        em_px *= 0.97
    em = int(round(em_px))
    f = font(em)
    pad = int(round(em * pad_frac))
    sw = int(round(em * 0.95))
    tgap = int(round(em * 0.45))
    widths, min_gap = _legend_metrics(entries, em)
    slack = (width - 2 * pad) - sum(widths)
    gap = max(min_gap, slack / max(len(entries) - 1, 1)) if len(entries) > 1 else 0

    img = _band(width, 2 * pad + max(sw, int(round(em * 1.22))), bg)
    d = ImageDraw.Draw(img)
    x, y = float(pad), pad
    for (bgr, label), w in zip(entries, widths):
        xi = int(round(x))
        d.rectangle([xi, y, xi + sw, y + sw], fill=tuple(reversed(bgr)), outline=fg, width=1)
        d.text((xi + sw + tgap, y - int(round(em * 0.06))), label, font=f, fill=fg)
        x += w + gap
    return _to_bgr(img)


def pick_legend(width, variants, em_px):
    """First legend wording that fits at `em_px` without shrinking, else the shortest."""
    for entries in variants:
        if legend_fits(width, entries, em_px):
            return entries
    return variants[-1]


def compose(panels, banner_lines, caption_chunks, latex_width_frac,
            legend_variants=None, sep_frac=0.008,
            target_pt=TARGET_PT, secondary_pt=SECONDARY_PT, verbose=True, tag=""):
    """Assemble the final figure.

    panels          : [left_bgr, right_bgr]
    banner_lines    : [[lines for left], [lines for right]] -- drawn above each panel
    caption_chunks  : chunks for the top strip, wrapped to fit
    latex_width_frac: fraction of \\textwidth the figure is included at
    legend_variants : list of [(bgr, label), ...] wordings, longest first
    """
    sep = max(6, int(round(sum(p.shape[1] for p in panels) * sep_frac)))
    pan_w = [p.shape[1] for p in panels]
    fig_w = sum(pan_w) + sep * (len(panels) - 1)

    # One em for both banners (they must match), fitted to the narrower panel.
    em_main = em_for(fig_w, latex_width_frac, target_pt)
    flat = [s for lines in banner_lines for s in lines]
    em_main = fit_em(flat, min(pan_w) - 2 * int(round(em_main * 0.37)), em_main)

    em_sec = em_for(fig_w, latex_width_frac, secondary_pt)
    cap_lines = wrap_chunks(caption_chunks, fig_w - 2 * int(round(em_sec * 0.37)), em_sec)

    blocks = []
    for p, lines in zip(panels, banner_lines):
        blocks.append(np.vstack([text_band(p.shape[1], lines, em_main), p]))
    h = max(b.shape[0] for b in blocks)
    blocks = [np.vstack([b, np.full((h - b.shape[0], b.shape[1], 3), 255, np.uint8)])
              for b in blocks]

    row = blocks[0]
    for b in blocks[1:]:
        row = np.hstack([row, np.full((h, sep, 3), SEP_GRAY, np.uint8), b])

    parts = [text_band(fig_w, cap_lines, em_sec), row]
    if legend_variants:
        parts.append(legend_band(fig_w, pick_legend(fig_w, legend_variants, em_sec), em_sec))
    out = np.vstack(parts)

    if verbose:
        print(f"    {tag}canvas {out.shape[1]}x{out.shape[0]} px, "
              f"@{latex_width_frac:g}\\textwidth -> labels "
              f"{printed_pt(em_main, fig_w, latex_width_frac):.1f}pt / "
              f"{printed_pt(em_sec, fig_w, latex_width_frac):.1f}pt")
    return out
