#!/usr/bin/env python3
"""
FP-to-GT Overlap Analysis for SAM3 False Positive Characterisation.

For each FP prediction in the evaluation CSV, this script computes:
  1. max_gt_iou       — maximum IoU with any GT bounding box in the same image
  2. max_gt_contain   — maximum "containment in GT" = intersection / area(FP)
                        i.e. what fraction of the FP box lies INSIDE some GT box

These two metrics let us classify each FP into mutually exclusive categories:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  max_gt_contain  │  max_gt_iou  │  Interpretation                      │
  ├──────────────────┼──────────────┼──────────────────────────────────────┤
  │  >= 0.50         │  < 0.50      │  Sub-part detection (wing, leg, etc.)│
  │                  │              │  FP is inside a GT insect but small  │
  ├──────────────────┼──────────────┼──────────────────────────────────────┤
  │  >= 0.50         │  >= 0.50     │  Duplicate / redundant detection     │
  │                  │              │  (essentially the same object twice) │
  ├──────────────────┼──────────────┼──────────────────────────────────────┤
  │  < 0.50          │  > 0.0       │  Partial overlap (edge / shadow FP)  │
  ├──────────────────┼──────────────┼──────────────────────────────────────┤
  │  0.0             │  0.0         │  Pure hallucination (no insect near) │
  └─────────────────────────────────────────────────────────────────────────┘

Usage
-----
  cd /home/dolma/repo/flat-bug

  python3 scripts/manuscript/statistics/analyze_fp_gt_overlap.py \\
    --csv scripts/manuscript/statistics/data/finetuned/lora/anti_hallucination_lora_v3_checkpoint_6_mask_5_score_005_using_greed_eval_with_bb.csv

  # Compare two CSVs side by side:
  python3 scripts/manuscript/statistics/analyze_fp_gt_overlap.py \\
    --csv <path_a.csv> <path_b.csv>

  # Filter by size (default sqrt(area) >= 32 to match main eval):
  python3 scripts/manuscript/statistics/analyze_fp_gt_overlap.py \\
    --csv <path.csv> --min-size 32

Output
------
  Printed report to stdout.
  Optional CSV with per-FP metrics: --save-fp-csv <path>.

Requirements: pandas, numpy (both already installed in sam3env / flatbug env).
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────

def parse_bbox(s) -> np.ndarray | None:
    """Parse '[x1, y1, x2, y2]' string to float array, return None if empty."""
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        vals = ast.literal_eval(s)
        return np.array(vals, dtype=np.float32)
    except Exception:
        return None


def iou_and_containment(fp_box: np.ndarray, gt_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised IoU and containment (intersection / area_fp) of one FP box vs
    many GT boxes.

    Parameters
    ----------
    fp_box  : (4,)  [x1, y1, x2, y2]
    gt_boxes: (N,4) [x1, y1, x2, y2]

    Returns
    -------
    ious       : (N,)
    containment: (N,)  — what fraction of fp_box lies inside each GT box
    """
    if gt_boxes.size == 0:
        return np.array([]), np.array([])

    fp_area = max((fp_box[2] - fp_box[0]) * (fp_box[3] - fp_box[1]), 0.0)

    ix1 = np.maximum(fp_box[0], gt_boxes[:, 0])
    iy1 = np.maximum(fp_box[1], gt_boxes[:, 1])
    ix2 = np.minimum(fp_box[2], gt_boxes[:, 2])
    iy2 = np.minimum(fp_box[3], gt_boxes[:, 3])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)

    gt_areas = np.maximum(
        (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]),
        0.0
    )
    union = fp_area + gt_areas - inter
    iou = np.where(union > 0, inter / union, 0.0)
    contain = np.where(fp_area > 0, inter / fp_area, 0.0)

    return iou, contain


# ──────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────

def analyse_csv(csv_path: str, min_size: float = 32.0) -> pd.DataFrame:
    """
    Load evaluation CSV and compute per-FP max GT IoU and max GT containment.

    Returns a DataFrame with one row per FP prediction.
    """
    print(f"\nLoading {csv_path} ...")
    df = pd.read_csv(csv_path, sep=';', low_memory=False)
    print(f"  Total rows: {len(df):,}")

    # ── size filter ──────────────────────────────────────────────────────────
    def effective_area(row):
        return row['contourArea_1'] if row['idx_1'] != -1 else row['contourArea_2']

    df['_area'] = df.apply(effective_area, axis=1)
    df['_size'] = np.sqrt(df['_area'])
    df = df[df['_size'] >= min_size].copy()
    print(f"  After size >= {min_size} filter: {len(df):,} rows")

    # ── separate GT and FP rows ───────────────────────────────────────────────
    # GT rows: any row that has an idx_1 entry (TP matches + FN unmatched)
    gt_rows = df[df['idx_1'] != -1].copy()
    # FP rows: idx_1 == -1, idx_2 != -1
    fp_rows = df[(df['idx_1'] == -1) & (df['idx_2'] != -1)].copy()

    print(f"  GT annotations used for reference: {len(gt_rows):,}")
    print(f"  FP predictions to analyse:         {len(fp_rows):,}")

    # ── build per-image GT bbox lookup ──────────────────────────────────────
    gt_rows['_bbox_parsed'] = gt_rows['bbox_1'].apply(parse_bbox)
    gt_by_image: dict[str, np.ndarray] = {}
    for img, grp in gt_rows.groupby('image'):
        valid = [b for b in grp['_bbox_parsed'] if b is not None]
        if valid:
            gt_by_image[img] = np.stack(valid)  # (N,4)
        else:
            gt_by_image[img] = np.empty((0, 4), dtype=np.float32)

    # ── compute metrics for each FP ─────────────────────────────────────────
    fp_rows = fp_rows.copy()
    fp_rows['_bbox2_parsed'] = fp_rows['bbox_2'].apply(parse_bbox)

    max_gt_ious = np.zeros(len(fp_rows), dtype=np.float32)
    max_gt_contains = np.zeros(len(fp_rows), dtype=np.float32)

    for i, (_, row) in enumerate(fp_rows.iterrows()):
        fp_box = row['_bbox2_parsed']
        if fp_box is None:
            continue
        gt_boxes = gt_by_image.get(row['image'], np.empty((0, 4), dtype=np.float32))
        if gt_boxes.size == 0:
            continue
        ious, contains = iou_and_containment(fp_box, gt_boxes)
        if ious.size > 0:
            max_gt_ious[i] = ious.max()
            max_gt_contains[i] = contains.max()

        if (i + 1) % 50_000 == 0:
            print(f"    ... processed {i+1:,}/{len(fp_rows):,} FPs", flush=True)

    fp_rows = fp_rows.copy()
    fp_rows['max_gt_iou'] = max_gt_ious
    fp_rows['max_gt_contain'] = max_gt_contains
    fp_rows['conf'] = fp_rows['conf2']
    fp_rows['pred_area'] = fp_rows['contourArea_2']
    fp_rows['pred_size'] = np.sqrt(fp_rows['pred_area'].clip(lower=0))

    return fp_rows


# ──────────────────────────────────────────────
# Classification & reporting
# ──────────────────────────────────────────────

CONTAIN_THRESH = 0.50  # FP >= 50% inside a GT box → sub-part or duplicate
IOU_MATCH_THRESH = 0.50  # IoU >= 50% → essentially the same object


def classify_fp(row) -> str:
    c = row['max_gt_contain']
    iou = row['max_gt_iou']
    if c >= CONTAIN_THRESH and iou >= IOU_MATCH_THRESH:
        return 'duplicate'       # same object, high overlap
    elif c >= CONTAIN_THRESH:
        return 'sub_part'        # inside GT box but small (wing/leg/thorax)
    elif iou > 0.0:
        return 'edge_overlap'    # partial spatial proximity (< 50% containment)
    else:
        return 'hallucination'   # no insect anywhere near


def print_report(fp_df: pd.DataFrame, csv_label: str):
    n = len(fp_df)
    print("\n" + "=" * 70)
    print(f"FP-GT OVERLAP REPORT — {csv_label}")
    print("=" * 70)
    print(f"Total FP predictions analysed: {n:,}")

    fp_df = fp_df.copy()
    fp_df['_class'] = fp_df.apply(classify_fp, axis=1)

    counts = fp_df['_class'].value_counts()

    # ── 1. Category breakdown ─────────────────────────────────────────────
    print("\n── FP Category Breakdown ──────────────────────────────────────")
    category_order = ['sub_part', 'duplicate', 'edge_overlap', 'hallucination']
    category_desc = {
        'sub_part':      'Sub-part of insect (>= 50% inside GT, IoU < 50%)',
        'duplicate':     'Duplicate detection (>= 50% inside GT, IoU >= 50%)',
        'edge_overlap':  'Edge / shadow overlap (< 50% inside GT, IoU > 0)',
        'hallucination': 'Pure hallucination (no overlap with any GT)',
    }
    for cat in category_order:
        cnt = counts.get(cat, 0)
        pct = 100 * cnt / n if n > 0 else 0
        print(f"  {cat:<18} {cnt:>8,}   {pct:5.1f}%   — {category_desc[cat]}")

    # ── 2. max_gt_contain distribution ───────────────────────────────────
    print("\n── max_gt_contain distribution (fraction of FP inside any GT box) ──")
    bins = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.01]
    labels = ['0%', '1%', '5%', '10%', '20%', '30%', '50%', '70%', '90%']
    contain = fp_df['max_gt_contain'].values
    for lo, hi, lbl in zip(bins[:-1], bins[1:], labels):
        mask = (contain >= lo) & (contain < hi)
        cnt = mask.sum()
        pct = 100 * cnt / n if n > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  [{lbl:>3}–{labels[labels.index(lbl)+1] if lbl != '90%' else '100%'}): "
              f"{cnt:>8,}  {pct:5.1f}%  {bar}")

    # ── 3. max_gt_iou distribution ────────────────────────────────────────
    print("\n── max_gt_iou distribution (standard IoU with nearest GT box) ──")
    ious = fp_df['max_gt_iou'].values
    for lo, hi, lbl in zip(bins[:-1], bins[1:], labels):
        mask = (ious >= lo) & (ious < hi)
        cnt = mask.sum()
        pct = 100 * cnt / n if n > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  [{lbl:>3}–{labels[labels.index(lbl)+1] if lbl != '90%' else '100%'}): "
              f"{cnt:>8,}  {pct:5.1f}%  {bar}")

    # ── 4. "How many FPs are sub-part?" at different containment thresholds
    print("\n── Containment filter simulation: FPs suppressed vs TPs lost ──")
    print("  (assuming TP predictions have high scores so won't be suppressed)")
    print(f"  {'Threshold':>10}  {'FPs removed':>12}  {'% of FP':>9}")
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        sub = (contain >= thresh).sum()
        pct = 100 * sub / n if n > 0 else 0
        print(f"  {thresh:>10.2f}  {sub:>12,}  {pct:>8.1f}%")

    # ── 5. Confidence score breakdown by category ─────────────────────────
    print("\n── Confidence score stats by FP category ──────────────────────")
    print(f"  {'Category':<18}  {'Count':>8}  {'Mean conf':>10}  {'Median':>8}  {'p90':>8}")
    for cat in category_order:
        sub = fp_df[fp_df['_class'] == cat]['conf'].dropna()
        if len(sub) == 0:
            print(f"  {cat:<18}  {'0':>8}")
            continue
        print(f"  {cat:<18}  {len(sub):>8,}  "
              f"{sub.mean():>10.4f}  {sub.median():>8.4f}  {sub.quantile(0.9):>8.4f}")

    # ── 6. Size breakdown for sub-parts ──────────────────────────────────
    sub_part_df = fp_df[fp_df['_class'] == 'sub_part']
    if len(sub_part_df) > 0:
        print("\n── Sub-part FP size distribution (sqrt area in pixels) ────────")
        sizes = sub_part_df['pred_size']
        print(f"  Mean:   {sizes.mean():.1f} px")
        print(f"  Median: {sizes.median():.1f} px")
        print(f"  p25:    {sizes.quantile(0.25):.1f} px")
        print(f"  p75:    {sizes.quantile(0.75):.1f} px")
        print(f"  p90:    {sizes.quantile(0.90):.1f} px")
        size_bins = [32, 50, 75, 100, 150, 200, 300, 500, 1e9]
        size_lbls = ['32', '50', '75', '100', '150', '200', '300', '500+']
        for lo, hi, lbl in zip(size_bins[:-1], size_bins[1:], size_lbls):
            mask = (sizes >= lo) & (sizes < hi)
            cnt = mask.sum()
            pct = 100 * cnt / len(sub_part_df)
            print(f"  size [{lbl:>4}+): {cnt:>7,}  {pct:5.1f}%")

    print()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__
    )
    parser.add_argument(
        '--csv', nargs='+', required=True,
        help='Path(s) to evaluation CSV file(s) (semicolon-separated, greedy eval format)'
    )
    parser.add_argument(
        '--min-size', type=float, default=32.0,
        help='Minimum sqrt(area) to include (default: 32.0)'
    )
    parser.add_argument(
        '--save-fp-csv', type=str, default=None,
        help='If set, save per-FP metrics to this CSV path'
    )
    args = parser.parse_args()

    all_fp_dfs = []
    for csv_path in args.csv:
        fp_df = analyse_csv(csv_path, min_size=args.min_size)
        label = Path(csv_path).stem
        print_report(fp_df, label)
        all_fp_dfs.append(fp_df)

    if args.save_fp_csv and all_fp_dfs:
        combined = pd.concat(all_fp_dfs, ignore_index=True)
        out_cols = ['image', 'idx_2', 'bbox_2', 'contourArea_2', 'pred_size',
                    'conf', 'max_gt_iou', 'max_gt_contain']
        out_cols = [c for c in out_cols if c in combined.columns]
        combined[out_cols].to_csv(args.save_fp_csv, index=False)
        print(f"Per-FP metrics saved to: {args.save_fp_csv}")


if __name__ == '__main__':
    main()
