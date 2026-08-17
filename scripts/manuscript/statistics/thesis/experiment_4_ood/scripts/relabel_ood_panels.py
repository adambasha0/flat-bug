#!/usr/bin/env python3
r"""Re-render the label bands of the already-generated OOD qualitative panels.

The panels in <deliverable>/<dataset>/visualizations/ were produced by
ood_tp_fp_fn_examples.py with the old fixed-size cv2 labels, which come out at ~2-3pt
once LaTeX scales the PNG to \linewidth -- illegible in print. Regenerating them from
scratch needs the raw datasets and prediction JSONs, which live on the GPU box
(/data/sam3) and are not part of this deliverable.

This script fixes the labels without the raw data: it keeps the rendered *image*
content untouched, strips the old label bands off, recomputes every number from the
matching CSVs that ship in the deliverable (same classify() as the generator, so the
labels are identical in content), and re-composes with the print-sized labels of
ood_panel_labels.

Old layout being reversed (all fixed pixel heights):
    rows 0..27          caption strip
    rows 28..57         per-panel banner strips
    rows 58..H-31       the two panels + 12px gray separator
    rows H-30..H-1      legend strip

    python3 relabel_ood_panels.py                 # rewrite all panels -> visualizations_thesis/
    python3 relabel_ood_panels.py --thesis-copy    # ...and copy the 4 thesis figures over
"""
import argparse
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ood_panel_labels as L
from ood_tp_fp_fn_examples import CONF, DS, LEGEND_VARIANTS, classify

DELIV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# geometry of the old burned-in layout
OLD_CAP_H, OLD_BANNER_H, OLD_LEGEND_H, OLD_SEP_W = 28, 30, 30, 12

MODELS = [("SAM3-ft", "sam3"), ("FlatBug-L", "flatbug")]

# panels promoted into the thesis, and the filenames they must keep there
THESIS_MAP = {
    "urban_insects_bbox_W_UV_LUZI_150824_B307": "exp4_urban_insects_bbox_panel.png",
    "pest24_bbox_0000010": "exp4_pest24_bbox_panel.png",
    "massid45_bbox_GRLLN6_0_2255_2665_2767_3177": "exp4_massid45_bbox_panel.png",
    "massid45_mask_GRLLN6_0_2255_2665_2767_3177": "exp4_massid45_mask_panel.png",
}
# Where --thesis-copy writes the four promoted panels. Set THESIS_IMAGE_DIR to the
# thesis repo's qualitative-examples folder; otherwise they land beside the deliverable.
THESIS_DIR = os.environ.get("THESIS_IMAGE_DIR",
                            os.path.join(DELIV, "_thesis_panels"))

_csv_cache = {}


def load_csv(ds, model_key):
    key = (ds, model_key)
    if key not in _csv_cache:
        _csv_cache[key] = pd.read_csv(f"{DELIV}/{ds}/csv/{ds}_{model_key}.csv",
                                      sep=';', dtype={'image': str})
    return _csv_cache[key]


def split_panels(img):
    """Drop the old label bands and return the two panel images."""
    H, W = img.shape[:2]
    content = img[OLD_CAP_H + OLD_BANNER_H:H - OLD_LEGEND_H]
    pw = (W - OLD_SEP_W) // 2
    left, right = content[:, :pw], content[:, pw + OLD_SEP_W:pw + OLD_SEP_W + pw]
    # sanity: the strip we removed must have been the white/gray label furniture
    band = img[:OLD_CAP_H + OLD_BANNER_H]
    assert np.mean(np.all(band >= 250, axis=2)) > 0.5, "unexpected layout: top band not white"
    return [left, right]


def parse_name(ds, fname):
    base = os.path.splitext(fname)[0]
    for mode in ("bbox", "mask"):
        pre = f"{ds}_{mode}_"
        if base.startswith(pre):
            return mode, base[len(pre):]
    return None, None


def relabel(ds, cfg, out_dir):
    src_dir = f"{DELIV}/{ds}/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    wrote = []
    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".png"):
            continue
        mode, stem = parse_name(ds, fname)
        if mode is None:
            print(f"    skip (unparsed name): {fname}")
            continue
        img = cv2.imread(os.path.join(src_dir, fname))
        panels = split_panels(img)

        banners = []
        for label, key in MODELS:
            df = load_csv(ds, key)
            rows = df[df.image == stem]
            if rows.empty:
                raise SystemExit(f"no CSV rows for {ds}/{key} image {stem!r}")
            _, tp, fp, fn = classify(rows)
            P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
            banners.append([label, f"TP={tp}  FP={fp}  FN={fn}", f"P={P:.2f}  R={R:.2f}"])

        chunks = ([ds, mode, f"preds @ conf >= {CONF}"]
                   + (["densest crop"] if cfg['crop'] else []) + [stem])
        combo = L.compose(panels, banners, chunks, cfg['latex_frac'],
                          legend_variants=LEGEND_VARIANTS, tag=f"{fname}: ")
        dst = os.path.join(out_dir, fname)
        cv2.imwrite(dst, combo)
        wrote.append(dst)
    return wrote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="visualizations_thesis",
                    help="sibling dir of visualizations/ to write into")
    ap.add_argument("--thesis-copy", action="store_true",
                    help=f"also copy the {len(THESIS_MAP)} promoted panels into the thesis repo")
    ap.add_argument("--thesis-dir", default=THESIS_DIR)
    args = ap.parse_args()

    produced = {}
    for ds, cfg in DS.items():
        print(f"{ds} (@{cfg['latex_frac']:g}\\textwidth):")
        for p in relabel(ds, cfg, f"{DELIV}/{ds}/{args.out_name}"):
            produced[os.path.splitext(os.path.basename(p))[0]] = p

    if args.thesis_copy:
        print(f"\ncopying into {args.thesis_dir}")
        os.makedirs(args.thesis_dir, exist_ok=True)
        for stem, dst_name in THESIS_MAP.items():
            src = produced.get(stem)
            if src is None:
                raise SystemExit(f"thesis panel not produced: {stem}")
            dst = os.path.join(args.thesis_dir, dst_name)
            shutil.copyfile(src, dst)
            h, w = cv2.imread(dst).shape[:2]
            print(f"    {dst_name}  <- {os.path.basename(src)}  ({w}x{h})")
    print("DONE")


if __name__ == "__main__":
    main()
