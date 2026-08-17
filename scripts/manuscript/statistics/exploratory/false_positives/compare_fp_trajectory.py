#!/usr/bin/env python3
"""
Compare FP trajectories between two greedy-eval CSV files.

Reads the semicolon-separated CSVs produced by the FlatBug evaluation
pipeline (greedy bbox-IoU matching already done):
  - FP rows:  idx_1 == -1,  idx_2 != -1
  - TP rows:  idx_1 != -1,  idx_2 != -1
  - FN rows:  idx_1 != -1,  idx_2 == -1
Confidence score is taken from the `conf2` column.

Usage:
    python3 scripts/manuscript/statistics/compare_fp_trajectory.py \
        --csv1 data/finetuned/lora/fine_tuned_sam3_with_lora_checkpoint_10_mask_5_score_005_with_bb.csv \
        --csv2 data/finetuned/lora/anti_hallucination_lora_v3_checkpoint_6_mask_5_score_005_using_greed_eval_with_bb.csv \
        --label1 "Baseline ckpt10" \
        --label2 "v3 anti-hallucination ckpt6"
"""

import argparse
import numpy as np
import pandas as pd

# Confidence bins for FP breakdown
FP_BINS = [
    (0.001, 0.005),
    (0.005, 0.010),
    (0.010, 0.020),
    (0.020, 0.050),
    (0.050, 0.100),
    (0.100, 0.200),
    (0.200, 0.500),
    (0.500, 1.001),
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_csv(path, min_area=1024.0):
    """Parse a greedy-eval CSV and return TP/FP/FN arrays.

    min_area: minimum contourArea_2 (in pixels²) to include a prediction.
              Default 1024 = 32×32 px side, matching analyze_fp_gt_overlap.py.
    """
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip()

    for col in ('idx_1', 'idx_2', 'conf2', 'IoU', 'contourArea_2', 'contourArea_1'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['contourArea_2'] = df.get('contourArea_2', pd.Series(0.0, index=df.index)).fillna(0.0)

    is_fp = (df['idx_1'] == -1) & (df['idx_2'] != -1)
    is_tp = (df['idx_1'] != -1) & (df['idx_2'] != -1)
    is_fn = (df['idx_1'] != -1) & (df['idx_2'] == -1)

    # Apply size filter to predictions (FP + TP)
    large_enough = df['contourArea_2'] >= min_area
    df_fp = df[is_fp & large_enough]
    df_tp = df[is_tp & large_enough]
    df_fn = df[is_fn]

    fp_scores = df_fp['conf2'].dropna().values.astype(float)
    tp_scores = df_tp['conf2'].dropna().values.astype(float)
    tp_ious   = df_tp['IoU'].fillna(0).values.astype(float)
    n_fn = len(df_fn)
    n_gt = len(df_tp) + n_fn

    return {
        'fp_scores': fp_scores,
        'tp_scores': tp_scores,
        'tp_ious':   tp_ious,
        'n_fp':  len(fp_scores),
        'n_tp':  len(tp_scores),
        'n_fn':  n_fn,
        'n_gt':  n_gt,
        'n_pred': len(fp_scores) + len(tp_scores),
    }


def analyze(path, label, min_area=1024.0):
    data = load_csv(path, min_area)
    n_tp, n_fp = data['n_tp'], data['n_fp']
    n_pred, n_gt = data['n_pred'], data['n_gt']
    precision = n_tp / n_pred if n_pred > 0 else 0.0
    recall    = n_tp / n_gt   if n_gt   > 0 else 0.0
    tp_above50 = int((data['tp_ious'] >= 0.5).sum()) if len(data['tp_ious']) > 0 else 0
    return {
        'label': label,
        **data,
        'precision':      precision,
        'recall':         recall,
        'tp_above_iou50': tp_above50,
    }


# ── Printing helpers ──────────────────────────────────────────────────────────

def _pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) > 0 else 0.0


def print_summary(r):
    tp_s, fp_s, iou = r['tp_scores'], r['fp_scores'], r['tp_ious']
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY: {r['label']}")
    print(f"{'=' * 72}")
    print(f"  Predictions:  {r['n_pred']:>10,d}   |  GT annotations: {r['n_gt']:>10,d}")
    print(f"  TP:           {r['n_tp']:>10,d}   |  FP:             {r['n_fp']:>10,d}")
    print(f"  FN:           {r['n_fn']:>10,d}")
    print(f"  Precision:    {r['precision']:>10.4f}   |  Recall:         {r['recall']:.4f}")
    if len(iou) > 0:
        print(f"  TP mean IoU:  {iou.mean():>10.4f}   |  TP >= IoU 0.5:  {r['tp_above_iou50']:,d}")
    if len(tp_s) > 0:
        print(f"  TP conf mean: {tp_s.mean():>10.4f}   |  TP conf median: {np.median(tp_s):.4f}")
    if len(fp_s) > 0:
        print(f"  FP conf mean: {fp_s.mean():>10.4f}   |  FP conf median: {np.median(fp_s):.4f}")
        print(f"  FP conf p90:  {_pct(fp_s, 90):>10.4f}   |  FP conf p95:    {_pct(fp_s, 95):.4f}")


def print_fp_table(fp_scores, label):
    total = len(fp_scores)
    print(f"\n{'=' * 72}")
    print(f"  FP Confidence Breakdown: {label}")
    print(f"  Total FPs: {total:,d}")
    print(f"{'=' * 72}")
    print(f"  {'Conf Range':<18s} | {'FP Count':>10s} | {'% total':>8s} | {'Cumul %':>8s}")
    print("  " + "-" * 58)
    cum = 0
    for lo, hi in FP_BINS:
        count = int(((fp_scores >= lo) & (fp_scores < hi)).sum())
        cum += count
        pct     = 100.0 * count / total if total > 0 else 0.0
        cum_pct = 100.0 * cum   / total if total > 0 else 0.0
        print(f"  {lo:.3f} – {hi:<7.3f}      | {count:>10,d} | {pct:>7.2f}% | {cum_pct:>7.2f}%")


def print_comparison(r1, r2):
    L1 = r1['label'][:18]
    L2 = r2['label'][:18]

    def _d(v1, v2, fmt, higher_is_better=True):
        """Format a delta with ✓/✗ indicator."""
        d = v2 - v1
        sign = "+" if d > 0 else ""
        good = (d > 0) == higher_is_better
        mark = "" if abs(d) < 1e-9 else (" ✓" if good else " ✗")
        if fmt == ",d":
            return f"{sign}{int(d):,d}{mark}"
        return f"{sign}{d:{fmt}}{mark}"

    # ── Summary comparison ──────────────────────────────────────────────────
    print(f"\n{'=' * 82}")
    print(f"  COMPARISON: '{r1['label']}'  vs  '{r2['label']}'")
    print(f"{'=' * 82}")
    print(f"  {'Metric':<24s} | {L1:>18s} | {L2:>18s} | {'Delta':>12s}")
    print("  " + "-" * 78)
    rows = [
        ("Predictions",    r1['n_pred'],         r2['n_pred'],         ",d",   False),
        ("True Positives", r1['n_tp'],            r2['n_tp'],           ",d",   True),
        ("False Positives",r1['n_fp'],            r2['n_fp'],           ",d",   False),
        ("False Negatives",r1['n_fn'],            r2['n_fn'],           ",d",   False),
        ("Precision",      r1['precision'],       r2['precision'],      ".4f",  True),
        ("Recall",         r1['recall'],          r2['recall'],         ".4f",  True),
    ]
    if len(r1['tp_ious']) > 0 and len(r2['tp_ious']) > 0:
        rows += [
            ("TP mean IoU",   r1['tp_ious'].mean(),  r2['tp_ious'].mean(), ".4f",  True),
            ("TP >= IoU 0.5", r1['tp_above_iou50'],  r2['tp_above_iou50'],",d",   True),
        ]
    fp1, fp2 = r1['fp_scores'], r2['fp_scores']
    if len(fp1) > 0 and len(fp2) > 0:
        rows += [
            ("FP conf mean",  fp1.mean(),           fp2.mean(),           ".4f",  False),
            ("FP conf median",np.median(fp1),        np.median(fp2),       ".4f",  False),
            ("FP conf p90",   _pct(fp1, 90),         _pct(fp2, 90),        ".4f",  False),
            ("FP conf p95",   _pct(fp1, 95),         _pct(fp2, 95),        ".4f",  False),
        ]
    for name, v1, v2, fmt, hib in rows:
        v1_str = f"{v1:>18,d}" if fmt == ",d" else f"{v1:>18.4f}"
        v2_str = f"{v2:>18,d}" if fmt == ",d" else f"{v2:>18.4f}"
        print(f"  {name:<24s} | {v1_str} | {v2_str} | {_d(v1, v2, fmt, hib):>12s}")

    # ── FP confidence bin comparison ────────────────────────────────────────
    print(f"\n{'=' * 82}")
    print(f"  FP CONFIDENCE BIN COMPARISON  (✓ = fewer FPs in that bin = good)")
    print(f"{'=' * 82}")
    print(f"  {'Conf Range':<18s} | {L1:>14s} | {L2:>14s} | {'Delta':>10s} | Direction")
    print("  " + "-" * 74)
    for lo, hi in FP_BINS:
        c1 = int(((fp1 >= lo) & (fp1 < hi)).sum())
        c2 = int(((fp2 >= lo) & (fp2 < hi)).sum())
        d  = c2 - c1
        direction = "fewer ✓" if d < 0 else ("more  ✗" if d > 0 else "same  –")
        sign = "+" if d > 0 else ""
        print(f"  {lo:.3f} – {hi:<7.3f}      | {c1:>14,d} | {c2:>14,d} | {sign}{d:>9,d} | {direction}")

    # ── Threshold simulation ────────────────────────────────────────────────
    tp1, tp2   = r1['tp_scores'], r2['tp_scores']
    n_gt1, n_gt2 = r1['n_gt'], r2['n_gt']
    print(f"\n{'=' * 82}")
    print(f"  THRESHOLD IMPACT SIMULATION")
    print(f"{'=' * 82}")
    print(f"  {'':12s}  {'——— ' + L1 + ' ———':>34s}   {'——— ' + L2 + ' ———':>34s}")
    print(f"  {'Threshold':<12s}  {'FP':>8s} {'TP':>8s} {'Recall':>8s} {'Prec':>8s}   {'FP':>8s} {'TP':>8s} {'Recall':>8s} {'Prec':>8s}")
    print("  " + "-" * 90)
    for t in [0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500]:
        fp1t = int((fp1 >= t).sum());  tp1t = int((tp1 >= t).sum())
        fp2t = int((fp2 >= t).sum());  tp2t = int((tp2 >= t).sum())
        p1  = tp1t / (tp1t + fp1t) if (tp1t + fp1t) > 0 else 0.0
        rc1 = tp1t / n_gt1          if n_gt1 > 0 else 0.0
        p2  = tp2t / (tp2t + fp2t) if (tp2t + fp2t) > 0 else 0.0
        rc2 = tp2t / n_gt2          if n_gt2 > 0 else 0.0
        print(f"  >= {t:<7.3f}   {fp1t:>8,d} {tp1t:>8,d} {rc1:>8.4f} {p1:>8.4f}   {fp2t:>8,d} {tp2t:>8,d} {rc2:>8.4f} {p2:>8.4f}")

    # ── Plain-language conclusion ────────────────────────────────────────────
    pct_fp = 100.0 * (r2['n_fp'] - r1['n_fp']) / r1['n_fp'] if r1['n_fp'] > 0 else 0.0
    recall_delta = r2['recall'] - r1['recall']
    fp_dir = "REDUCED" if pct_fp < 0 else "INCREASED"
    print(f"\n{'=' * 82}")
    print(f"  CONCLUSION")
    print(f"{'=' * 82}")
    print(f"  FPs {fp_dir} by {abs(pct_fp):.1f}%  ({r1['n_fp']:,d} → {r2['n_fp']:,d}, Δ={r2['n_fp']-r1['n_fp']:+,d})")
    print(f"  Recall change:     {recall_delta:+.4f}  ({r1['recall']:.4f} → {r2['recall']:.4f})")
    if len(fp1) > 0 and len(fp2) > 0:
        med1, med2 = float(np.median(fp1)), float(np.median(fp2))
        conf_dir = "LOWER" if med2 < med1 else "HIGHER"
        print(f"  FP confidence is   {conf_dir} in '{r2['label']}'")
        print(f"    median: {med1:.4f} → {med2:.4f}  (Δ={med2-med1:+.4f})")
        print(f"    mean:   {fp1.mean():.4f} → {fp2.mean():.4f}  (Δ={fp2.mean()-fp1.mean():+.4f})")
        print(f"    p90:    {_pct(fp1, 90):.4f} → {_pct(fp2, 90):.4f}  (Δ={_pct(fp2,90)-_pct(fp1,90):+.4f})")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare FP confidence trajectories between two greedy-eval CSV files"
    )
    parser.add_argument('--csv1',  required=True,  help="First eval CSV (semicolon-separated)")
    parser.add_argument('--csv2',  required=True,  help="Second eval CSV")
    parser.add_argument('--label1', default="exp1", help="Label for first experiment")
    parser.add_argument('--label2', default="exp2", help="Label for second experiment")
    parser.add_argument('--min-area', type=float, default=1024.0,
                        help="Min contourArea_2 in px² to include a prediction (default: 1024 = 32px side)")
    args = parser.parse_args()

    print(f"Loading {args.csv1} ...")
    r1 = analyze(args.csv1, args.label1, args.min_area)
    print_summary(r1)
    print_fp_table(r1['fp_scores'], r1['label'])

    print(f"\nLoading {args.csv2} ...")
    r2 = analyze(args.csv2, args.label2, args.min_area)
    print_summary(r2)
    print_fp_table(r2['fp_scores'], r2['label'])

    print_comparison(r1, r2)


if __name__ == "__main__":
    main()
