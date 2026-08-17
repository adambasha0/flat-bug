#!/usr/bin/env python3
r"""
TP / FP / FN qualitative comparison per OOD dataset (refactor of visualize_false_negatives.py).

Driven by the fb_eval_greedy `--coco-matching` CSVs, so the GT<->prediction matching is
exactly the one behind the AP numbers. Per image, two panels (SAM3-ft | FlatBug-L), each
with every box color-coded by correctness at IoU>=0.5:

    GREEN  = TP  (GT correctly detected)
    YELLOW = FN  (GT missed, or detected with IoU<0.5)
    RED    = FP  (false positive: spurious pred, or a pred with IoU<0.5)

Panel label shows real counts: "SAM3-ft / TP=.. FP=.. FN=.. / P=.. R=..".
bbox panels for all 3 datasets; MASK panels for MassID45 (real GT masks).
Urban Insects is cropped to its densest region (the full 4096px scan is too dense).

Labels are rendered by ood_panel_labels, which sizes text backwards from the target
printed point size in the thesis -- see that module's docstring. Each dataset therefore
declares `latex_frac`, the fraction of \textwidth the figure is \includegraphics'd at
(1.0 for a full-width figure, 0.48 for the MassID45 subfigure pair). Getting that wrong
is what made the first version's labels render at ~2-3pt on the printed page.

If the source images are not reachable (they live on the GPU box), use
relabel_ood_panels.py instead: it rebuilds the label bands on the already-rendered
deliverable PNGs from the same CSVs.
"""
import ast
import json
import os
import sys
from collections import defaultdict  # noqa: F401  (kept: used by downstream tweaks)

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ood_panel_labels as L

# The matched CSVs ship with this deliverable, but the raw dataset images and the
# prediction COCO JSONs do not (they live on the evaluation host). Point SAM3_ROOT at
# a checkout that has them; without it only the CSV-driven parts below will work.
# For label-only changes on the already-rendered panels use relabel_ood_panels.py,
# which needs nothing but this folder.
DELIV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("SAM3_ROOT", "/data/sam3")
D = os.environ.get("OOD_CSV_DIR", DELIV)          # per-dataset CSVs: <D>/<key>/csv/
OUT = os.environ.get("OOD_FIG_DIR", os.path.join(DELIV, "_qualitative_rerender"))
TP_C = (0, 200, 0); FN_C = (0, 215, 255); FP_C = (40, 40, 255)   # BGR: green / yellow / red
IOU_T = 0.5
CONF = 0.5   # operating point: only show predictions the model is >=50% confident about
             # (the models' raw score floors are 0.005 -> showing all FPs floods the panel)

PANEL_W = 1100   # max px per panel (was 760): more print resolution for the crops. Text
                 # size is width-relative, so raising this does not shrink the labels.
BOX_PX = 2.6     # box stroke width *after* the panel is resized down to PANEL_W

LEGEND_VARIANTS = [
    [(TP_C, "TP correct"), (FN_C, "FN missed GT"), (FP_C, "FP false positive")],
    [(TP_C, "TP correct"), (FN_C, "FN missed"), (FP_C, "FP spurious")],
]

DS = {
 "pest24": dict(images=f"{ROOT}/datasets/Pest24/test_images", ext=".jpg", crop=None,
                gtrange=(6, 16), latex_frac=1.0),
 "urban_insects": dict(images=f"{ROOT}/datasets/urban_insects_prep/images", ext=".jpg",
                       crop=1700, gtrange=(50, 1e9), latex_frac=1.0),
 "massid45": dict(images=f"{ROOT}/datasets/massid45_prep/images", ext=".png", crop=None,
    gtrange=(4, 15), latex_frac=0.48,   # placed in a 0.48\linewidth subfigure pair
    masks=dict(gt=f"{ROOT}/datasets/massid45_prep/massid45_gt_coco.json",
               sam3=f"{ROOT}/output/results/sam3_massid45/coco_instances.json",
               flatbug=f"{ROOT}/output/results/flatbug_massid45/coco_instances.json")),
}


def pbox(s):
    try: return [float(v) for v in ast.literal_eval(str(s))]
    except Exception: return None


def find_img(images, stem, ext):
    for s in (stem, stem.zfill(7)):
        p = os.path.join(images, s + ext)
        if os.path.exists(p): return p
    return None


def classify(rows, conf_t=CONF):
    """rows: DataFrame for one image, at confidence operating point conf_t.
    -> list[(kind, gt_id, pred_id, box_xyxy)], counts. A prediction below conf_t is
    suppressed (not drawn), so its GT (if any) counts as a miss (FN)."""
    out = []; tp = fp = fn = 0
    for _, r in rows.iterrows():
        c = 0.0 if pd.isna(r.conf2) else float(r.conf2)
        has_gt = (r.idx_1 != -1); has_pred = (r.idx_2 != -1); pred_ok = has_pred and c >= conf_t
        if has_gt and has_pred:
            if pred_ok and r.IoU_bb >= IOU_T:
                out.append(('tp', int(r.idx_1), int(r.idx_2), pbox(r.bbox_1))); tp += 1
            else:
                out.append(('fn', int(r.idx_1), -1, pbox(r.bbox_1))); fn += 1
                if pred_ok and r.IoU_bb < IOU_T:
                    out.append(('fp', -1, int(r.idx_2), pbox(r.bbox_2))); fp += 1
        elif has_gt:
            out.append(('fn', int(r.idx_1), -1, pbox(r.bbox_1))); fn += 1
        elif pred_ok:
            out.append(('fp', -1, int(r.idx_2), pbox(r.bbox_2))); fp += 1
    return out, tp, fp, fn


def crop_win(items, W, H, win):
    gt = [b for k, _, _, b in items if k in ('tp', 'fn') and b]
    if not gt: return (0, 0, min(win, W), min(win, H))
    cx = np.array([b[0] / 2 + b[2] / 2 for b in gt]); cy = np.array([b[1] / 2 + b[3] / 2 for b in gt])
    best = None
    for i in range(len(cx)):
        x0 = int(np.clip(cx[i] - win / 2, 0, max(0, W - win))); y0 = int(np.clip(cy[i] - win / 2, 0, max(0, H - win)))
        x1 = min(W, x0 + win); y1 = min(H, y0 + win)
        c = int(np.sum((cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)))
        if best is None or c > best[0]: best = (c, (x0, y0, x1, y1))
    return best[1]


def color(kind): return {'tp': TP_C, 'fn': FN_C, 'fp': FP_C}[kind]


def draw_panel(img, items, cw, poly_map=None, thick=3):
    # Crop FIRST, then draw with a single offset (cw origin). Drawing offset boxes on the
    # full image *and* cropping double-offsets them out of view (the earlier Urban bug).
    if cw is not None:
        base = img[cw[1]:cw[3], cw[0]:cw[2]].copy(); ox, oy = cw[0], cw[1]
    else:
        base = img.copy(); ox, oy = 0, 0
    if poly_map is not None:  # mask mode
        ov = base.copy()
        for kind, gid, pid, _ in items:
            seg = poly_map(kind, gid, pid)
            if not isinstance(seg, list): continue
            for poly in seg:
                if len(poly) < 6: continue
                pts = (np.array(poly, np.float32).reshape(-1, 2) - np.array([ox, oy], np.float32)).astype(np.int32)
                if kind in ('tp', 'fp'): cv2.fillPoly(ov, [pts], color(kind))
                cv2.polylines(base, [pts], True, color(kind), max(1, int(round(thick * 0.7))))
        cv2.addWeighted(ov, 0.45, base, 0.55, 0, base)
    else:
        for kind, gid, pid, b in items:
            if not b: continue
            cv2.rectangle(base, (int(b[0] - ox), int(b[1] - oy)), (int(b[2] - ox), int(b[3] - oy)), color(kind), thick)
    return base


def render(name, cfg, n=3):
    s3 = pd.read_csv(os.path.join(D, name, "csv", f"{name}_sam3.csv"), sep=';', dtype={'image': str})
    fb = pd.read_csv(os.path.join(D, name, "csv", f"{name}_flatbug.csv"), sep=';', dtype={'image': str})
    gtcount = s3[s3.idx_1 != -1].groupby('image').size()
    lo, hi = cfg['gtrange']
    stems = [st for st in sorted(gtcount.index) if lo <= gtcount[st] <= hi
             and find_img(cfg['images'], st, cfg['ext']) and st in set(fb.image)]
    if len(stems) > n: stems = stems[::max(1, len(stems) // (n * 3))][:n]
    else: stems = stems[:n]
    modes = [("bbox", False)]
    if 'masks' in cfg:
        modes.append(("mask", True))
        gP = {a['id']: a.get('segmentation') for a in json.load(open(cfg['masks']['gt']))['annotations']}
        sP = {a['id']: a.get('segmentation') for a in json.load(open(cfg['masks']['sam3']))['annotations']}
        fP = {a['id']: a.get('segmentation') for a in json.load(open(cfg['masks']['flatbug']))['annotations']}
    od = os.path.join(OUT, name); os.makedirs(od, exist_ok=True); wrote = []
    for st in stems:
        ip = find_img(cfg['images'], st, cfg['ext']); img = cv2.imread(ip)
        if img is None: continue
        H, W = img.shape[:2]
        for mode, is_mask in modes:
            panels, banners = [], []
            for model, df in [("SAM3-ft", s3), ("FlatBug-L", fb)]:
                items, tp, fp, fn = classify(df[df.image == st])
                cw = crop_win(items, W, H, cfg['crop']) if cfg['crop'] else None
                pm = None
                if is_mask:
                    src = sP if model == "SAM3-ft" else fP
                    def pm(kind, gid, pid, src=src):
                        if kind == 'fn': return gP.get(gid)
                        return src.get(pid)
                # thicken strokes so they survive the downscale to PANEL_W
                src_w = (cw[2] - cw[0]) if cw else W
                pan = draw_panel(img, items, cw, poly_map=pm,
                                 thick=max(2, int(round(BOX_PX * max(1.0, src_w / PANEL_W)))))
                if pan.shape[1] > PANEL_W:
                    sc = PANEL_W / pan.shape[1]
                    pan = cv2.resize(pan, (int(pan.shape[1] * sc), int(pan.shape[0] * sc)),
                                     interpolation=cv2.INTER_AREA)
                P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
                panels.append(pan)
                banners.append([model, f"TP={tp}  FP={fp}  FN={fn}", f"P={P:.2f}  R={R:.2f}"])
            chunks = ([name, mode, f"preds @ conf >= {CONF}"]
                          + (["densest crop"] if cfg['crop'] else []) + [st])
            combo = L.compose(panels, banners, chunks, cfg['latex_frac'],
                              legend_variants=LEGEND_VARIANTS, tag=f"{name}_{mode}_{st}: ")
            fn_out = os.path.join(od, f"{name}_{mode}_{st}.png")
            cv2.imwrite(fn_out, combo); wrote.append(fn_out)
    print(f"{name}: {len(wrote)} figures")
    [print("   ", os.path.basename(w)) for w in wrote]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, cfg in DS.items():
        render(name, cfg, 3)
    print("DONE", OUT)
