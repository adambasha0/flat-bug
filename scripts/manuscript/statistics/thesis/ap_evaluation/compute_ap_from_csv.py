#!/usr/bin/env python3
"""
Compute AP50, AP50-90, Precision, and Recall (Bbox + Mask) from a
greedy-eval CSV produced by the FlatBug evaluation pipeline.

The CSV format (semicolon-separated):
  image ; idx_1 ; idx_2 ; bbox_1 ; bbox_2 ; contourArea_1 ; contourArea_2 ;
  IoU (mask IoU) ; conf2 ; IoU_bb (bbox IoU)

Row types:
  TP  — idx_1 != -1  AND  idx_2 != -1   (matched GT ↔ prediction)
  FP  — idx_1 == -1  AND  idx_2 != -1   (prediction with no GT match)
  FN  — idx_1 != -1  AND  idx_2 == -1   (GT with no prediction)

Metrics reported:
  • Precision / Recall at the given confidence threshold (BBox and Mask @ IoU ≥ 0.5)
  • AP50   — COCO 101-point interpolation, IoU ≥ 0.50
  • AP75   — IoU ≥ 0.75
  • AP50-90 — mean AP over IoU ∈ [0.50, 0.55 … 0.95]
  (Separately for BBox using IoU_bb and Mask using IoU columns)

Usage:
    python3 compute_ap_from_csv.py \\
        --csv  <path_to_eval.csv> \\
        --threshold 0.05

    # Compare two experiments:
    python3 compute_ap_from_csv.py \\
        --csv  exp1.csv \\
        --csv2 exp2.csv \\
        --label1 "Baseline" --label2 "v3 LoRA" \\
        --threshold 0.05
"""

import argparse
import numpy as np
import pandas as pd

# IoU thresholds used for COCO AP50-90
COCO_IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
MIN_AREA = 1024.0  # 32×32 px², ignore tiny dust artefacts


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', low_memory=False)
    df.columns = df.columns.str.strip()

    for col in ('idx_1', 'idx_2', 'conf2', 'IoU', 'contourArea_1', 'contourArea_2'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # IoU_bb is empty for FP rows — fill with 0.0
    df['IoU_bb'] = pd.to_numeric(df.get('IoU_bb', 0), errors='coerce').fillna(0.0)
    df['IoU']    = df['IoU'].fillna(0.0)

    df['contourArea_2'] = df.get('contourArea_2', pd.Series(0.0)).fillna(0.0)
    df['contourArea_1'] = df.get('contourArea_1', pd.Series(0.0)).fillna(0.0)

    return df


def split_rows(df: pd.DataFrame):
    """Return (predictions_df, n_gt) where predictions have conf2 and IoU values."""
    large_pred = df['contourArea_2'] >= MIN_AREA

    is_pred = (df['idx_2'].notna()) & (df['idx_2'] != -1)
    is_gt   = (df['idx_1'].notna()) & (df['idx_1'] != -1)

    preds = df[is_pred & large_pred].copy()
    # n_gt: unique GT annotations — rows where idx_1 is valid (TP + FN)
    # Count per image to avoid double-counting (a GT matched to a prediction
    # appears in exactly one row; an unmatched GT also appears in exactly one row)
    n_gt = int(is_gt.sum())

    # For TP rows, IoU is the mask IoU of the match; for FP it is 0.
    # Mark whether each prediction was matched at all
    preds['matched'] = (preds['idx_1'] != -1) & preds['idx_1'].notna()

    return preds, n_gt


# ── AP computation ────────────────────────────────────────────────────────────

def compute_ap_at_iou(preds_sorted: pd.DataFrame, n_gt: int,
                      iou_col: str, iou_thresh: float) -> float:
    """
    101-point interpolated AP (COCO style).

    preds_sorted: predictions DataFrame sorted by conf2 DESCENDING.
    n_gt:         total GT count in the dataset.
    iou_col:      'IoU' (mask) or 'IoU_bb' (bbox).
    iou_thresh:   e.g. 0.50, 0.75, 0.95.
    """
    if len(preds_sorted) == 0 or n_gt == 0:
        return 0.0

    # A prediction is TP if it was matched to a GT AND the IoU meets the threshold
    tp_flags = (preds_sorted['matched']) & (preds_sorted[iou_col] >= iou_thresh)
    tp_cum = np.cumsum(tp_flags.values.astype(int))
    fp_cum = np.cumsum((~tp_flags).values.astype(int))

    recall_arr    = tp_cum / n_gt
    precision_arr = tp_cum / (tp_cum + fp_cum)

    # 101-point interpolation over recall ∈ [0, 1]
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 101):
        mask = recall_arr >= r
        ap += (precision_arr[mask].max() if mask.any() else 0.0)
    return ap / 101.0


def compute_metrics(preds: pd.DataFrame, n_gt: int, conf_threshold: float,
                    iou_col: str, iou_match: float = 0.50):
    """
    Returns a dict with AP50, AP75, AP50-90, precision, recall
    computed on predictions with conf2 >= conf_threshold.
    """
    filtered = preds[preds['conf2'] >= conf_threshold].copy()
    filtered_sorted = filtered.sort_values('conf2', ascending=False)

    # --- Single-point precision / recall at the operating threshold ---
    tp_at_t = int(
        (filtered_sorted['matched'] & (filtered_sorted[iou_col] >= iou_match)).sum()
    )
    fp_at_t = len(filtered_sorted) - tp_at_t
    fn_at_t = n_gt - tp_at_t

    prec   = tp_at_t / (tp_at_t + fp_at_t) if (tp_at_t + fp_at_t) > 0 else 0.0
    recall = tp_at_t / n_gt                  if n_gt > 0             else 0.0

    # --- AP at individual IoU thresholds ---
    ap50 = compute_ap_at_iou(filtered_sorted, n_gt, iou_col, 0.50)
    ap75 = compute_ap_at_iou(filtered_sorted, n_gt, iou_col, 0.75)

    # --- AP50-90 (COCO mean AP) ---
    ap_values = [
        compute_ap_at_iou(filtered_sorted, n_gt, iou_col, t)
        for t in COCO_IOU_THRESHOLDS
    ]
    ap50_90 = float(np.mean(ap_values))

    return {
        'n_pred':    len(filtered_sorted),
        'n_tp':      tp_at_t,
        'n_fp':      fp_at_t,
        'n_fn':      fn_at_t,
        'n_gt':      n_gt,
        'precision': prec,
        'recall':    recall,
        'AP50':      ap50,
        'AP75':      ap75,
        'AP50_90':   ap50_90,
        'ap_per_iou': list(zip(COCO_IOU_THRESHOLDS.tolist(), ap_values)),
    }


# ── Printing ──────────────────────────────────────────────────────────────────

SEP = "=" * 72

def print_metrics(label: str, conf_threshold: float,
                  bbox_m: dict, mask_m: dict):
    print(f"\n{SEP}")
    print(f"  RESULTS: {label}  (confidence threshold ≥ {conf_threshold})")
    print(SEP)
    print(f"  {'Metric':<28s}  {'BBox':>12s}  {'Mask':>12s}")
    print("  " + "-" * 56)

    rows = [
        ("GT annotations",   f"{bbox_m['n_gt']:,}",      f"{mask_m['n_gt']:,}"),
        ("Predictions kept", f"{bbox_m['n_pred']:,}",    f"{mask_m['n_pred']:,}"),
        ("True Positives",   f"{bbox_m['n_tp']:,}",      f"{mask_m['n_tp']:,}"),
        ("False Positives",  f"{bbox_m['n_fp']:,}",      f"{mask_m['n_fp']:,}"),
        ("False Negatives",  f"{bbox_m['n_fn']:,}",      f"{mask_m['n_fn']:,}"),
        ("Precision (IoU≥0.50)", f"{bbox_m['precision']:.4f}",  f"{mask_m['precision']:.4f}"),
        ("Recall   (IoU≥0.50)", f"{bbox_m['recall']:.4f}",     f"{mask_m['recall']:.4f}"),
        ("AP50",             f"{bbox_m['AP50']:.4f}",    f"{mask_m['AP50']:.4f}"),
        ("AP75",             f"{bbox_m['AP75']:.4f}",    f"{mask_m['AP75']:.4f}"),
        ("AP50-90 (COCO mAP)", f"{bbox_m['AP50_90']:.4f}", f"{mask_m['AP50_90']:.4f}"),
    ]

    for name, bval, mval in rows:
        print(f"  {name:<28s}  {bval:>12s}  {mval:>12s}")

    print(f"\n  AP per IoU threshold:")
    print(f"  {'IoU':>8s}  {'BBox AP':>10s}  {'Mask AP':>10s}")
    print("  " + "-" * 34)
    for (t, bap), (_, map_) in zip(bbox_m['ap_per_iou'], mask_m['ap_per_iou']):
        print(f"  {t:>8.2f}  {bap:>10.4f}  {map_:>10.4f}")


def print_comparison(label1, label2, conf_threshold,
                     bbox_m1, mask_m1, bbox_m2, mask_m2):
    L1, L2 = label1[:16], label2[:16]
    print(f"\n{SEP}")
    print(f"  COMPARISON: '{label1}'  vs  '{label2}'  (threshold ≥ {conf_threshold})")
    print(SEP)

    def d(v1, v2, higher_is_better=True):
        delta = v2 - v1
        sign  = "+" if delta > 0 else ""
        good  = (delta > 0) == higher_is_better
        mark  = (" ✓" if good else " ✗") if abs(delta) > 1e-6 else ""
        return f"{sign}{delta:.4f}{mark}"

    def di(v1, v2, higher_is_better=False):
        delta = v2 - v1
        sign  = "+" if delta > 0 else ""
        good  = (delta > 0) == higher_is_better
        mark  = (" ✓" if good else " ✗") if abs(delta) > 0 else ""
        return f"{sign}{int(delta):,}{mark}"

    header = f"  {'Metric':<28s}  {L1+' B':>10s}  {L2+' B':>10s}  {'ΔBbox':>8s}  {L1+' M':>10s}  {L2+' M':>10s}  {'ΔMask':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def row(name, k, is_int=False, hib=True):
        v1b, v2b = bbox_m1[k], bbox_m2[k]
        v1m, v2m = mask_m1[k], mask_m2[k]
        if is_int:
            print(f"  {name:<28s}  {int(v1b):>10,}  {int(v2b):>10,}  {di(v1b,v2b,hib):>8s}  "
                  f"{int(v1m):>10,}  {int(v2m):>10,}  {di(v1m,v2m,hib):>8s}")
        else:
            print(f"  {name:<28s}  {v1b:>10.4f}  {v2b:>10.4f}  {d(v1b,v2b,hib):>8s}  "
                  f"{v1m:>10.4f}  {v2m:>10.4f}  {d(v1m,v2m,hib):>8s}")

    row("True Positives",     'n_tp',      is_int=True, hib=True)
    row("False Positives",    'n_fp',      is_int=True, hib=False)
    row("False Negatives",    'n_fn',      is_int=True, hib=False)
    row("Precision (IoU≥.50)","precision", hib=True)
    row("Recall   (IoU≥.50)", "recall",    hib=True)
    row("AP50",               "AP50",      hib=True)
    row("AP75",               "AP75",      hib=True)
    row("AP50-90 (COCO mAP)", "AP50_90",   hib=True)

    print(f"\n  AP per IoU threshold:")
    print(f"  {'IoU':>6s}  {'B-base':>8s}  {'B-exp':>8s}  {'ΔBbox':>8s}  {'M-base':>8s}  {'M-exp':>8s}  {'ΔMask':>8s}")
    print("  " + "-" * 62)
    for (t, bap1), (_, bap2), (_, map1), (_, map2) in zip(
            bbox_m1['ap_per_iou'], bbox_m2['ap_per_iou'],
            mask_m1['ap_per_iou'], mask_m2['ap_per_iou']):
        db = bap2 - bap1; dm = map2 - map1
        db_s = f"{'+' if db>=0 else ''}{db:.4f}{'✓' if db>0 else ('✗' if db<0 else '')}"
        dm_s = f"{'+' if dm>=0 else ''}{dm:.4f}{'✓' if dm>0 else ('✗' if dm<0 else '')}"
        print(f"  {t:>6.2f}  {bap1:>8.4f}  {bap2:>8.4f}  {db_s:>8s}  {map1:>8.4f}  {map2:>8.4f}  {dm_s:>8s}")


# ── Threshold sweep ───────────────────────────────────────────────────────────

def sweep_thresholds(preds: pd.DataFrame, n_gt: int, label: str):
    """
    Sweep confidence thresholds and report AP50 (BBox+Mask), Precision,
    Recall, and F1 at each step.  Highlights optimal points for every metric.
    """
    # Dense grid: finer near the action zone, coarser at the extremes
    thresholds = np.unique(np.concatenate([
        np.linspace(0.005, 0.02,  30),
        np.linspace(0.02,  0.10,  40),
        np.linspace(0.10,  0.30,  20),
        np.linspace(0.30,  1.00,  15),
    ]))

    print(f"\n{SEP}")
    print(f"  THRESHOLD SWEEP — {label}")
    print(f"  (AP50 computed with 101-point COCO interpolation at each threshold)")
    print(SEP)
    print(f"  {'Threshold':>10s}  {'Bbox AP50':>9s}  {'Mask AP50':>9s}  "
          f"{'Bbox Prec':>9s}  {'Mask Prec':>9s}  "
          f"{'Bbox Rec':>8s}  {'Mask Rec':>8s}  "
          f"{'Bbox F1':>8s}  {'Preds':>8s}")
    print("  " + "-" * 90)

    results = []
    total = len(thresholds)
    for i, t in enumerate(thresholds):
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{total}", flush=True)
        bm = compute_metrics(preds, n_gt, t, iou_col='IoU_bb')
        mm = compute_metrics(preds, n_gt, t, iou_col='IoU')
        bf1 = (2 * bm['precision'] * bm['recall'] / (bm['precision'] + bm['recall'])
               if (bm['precision'] + bm['recall']) > 0 else 0.0)
        mf1 = (2 * mm['precision'] * mm['recall'] / (mm['precision'] + mm['recall'])
               if (mm['precision'] + mm['recall']) > 0 else 0.0)
        results.append({
            't':        t,
            'bap50':    bm['AP50'],
            'map50':    mm['AP50'],
            'bap7590':  bm['AP50_90'],
            'bprec':    bm['precision'],
            'mprec':    mm['precision'],
            'brec':     bm['recall'],
            'mrec':     mm['recall'],
            'bf1':      bf1,
            'mf1':      mf1,
            'n_pred':   bm['n_pred'],
        })
        print(f"  {t:>10.4f}  {bm['AP50']:>9.4f}  {mm['AP50']:>9.4f}  "
              f"{bm['precision']:>9.4f}  {mm['precision']:>9.4f}  "
              f"{bm['recall']:>8.4f}  {mm['recall']:>8.4f}  "
              f"{bf1:>8.4f}  {bm['n_pred']:>8,}")

    # ── Find optimal points ────────────────────────────────────────────────
    bap50_vals  = [r['bap50']   for r in results]
    bf1_vals    = [r['bf1']     for r in results]
    bprec_vals  = [r['bprec']   for r in results]
    brec_vals   = [r['brec']    for r in results]
    map50_vals  = [r['map50']   for r in results]
    mf1_vals    = [r['mf1']     for r in results]

    idx_bap50  = int(np.argmax(bap50_vals))
    idx_map50  = int(np.argmax(map50_vals))
    idx_bf1    = int(np.argmax(bf1_vals))
    idx_mf1    = int(np.argmax(mf1_vals))

    # Best precision while keeping recall ≥ 0.85
    high_recall_mask = [r['brec'] >= 0.85 for r in results]
    if any(high_recall_mask):
        idx_prec_r85 = max(
            (i for i, ok in enumerate(high_recall_mask) if ok),
            key=lambda i: bprec_vals[i]
        )
    else:
        idx_prec_r85 = None

    # Best precision while keeping recall ≥ 0.90
    high_recall_mask90 = [r['brec'] >= 0.90 for r in results]
    if any(high_recall_mask90):
        idx_prec_r90 = max(
            (i for i, ok in enumerate(high_recall_mask90) if ok),
            key=lambda i: bprec_vals[i]
        )
    else:
        idx_prec_r90 = None

    print(f"\n{SEP}")
    print(f"  OPTIMAL THRESHOLDS — {label}")
    print(SEP)

    def show_opt(tag, idx, extra=""):
        r = results[idx]
        bf1 = r['bf1']
        print(f"\n  ★ {tag}")
        print(f"    threshold : {r['t']:.4f}")
        print(f"    BBox AP50 : {r['bap50']:.4f}   Mask AP50 : {r['map50']:.4f}")
        print(f"    BBox Prec : {r['bprec']:.4f}   Mask Prec : {r['mprec']:.4f}")
        print(f"    BBox Rec  : {r['brec']:.4f}   Mask Rec  : {r['mrec']:.4f}")
        print(f"    BBox F1   : {bf1:.4f}   Mask F1   : {r['mf1']:.4f}")
        print(f"    Preds kept: {r['n_pred']:,}{extra}")

    show_opt("Maximum BBox AP50", idx_bap50)
    show_opt("Maximum Mask AP50", idx_map50)
    show_opt("Maximum BBox F1",   idx_bf1,
             "  ← balances precision and recall")
    show_opt("Maximum Mask F1",   idx_mf1)
    if idx_prec_r90 is not None:
        show_opt("Best BBox Precision with Recall ≥ 90%", idx_prec_r90)
    if idx_prec_r85 is not None:
        show_opt("Best BBox Precision with Recall ≥ 85%", idx_prec_r85)

    # Precision targets
    print(f"\n  ── Thresholds that first reach precision targets ──")
    print(f"  {'Target Prec':>12s}  {'Threshold':>10s}  {'BBox Prec':>10s}  "
          f"{'BBox Rec':>9s}  {'BBox AP50':>10s}  {'BBox F1':>9s}")
    print("  " + "-" * 68)
    for target in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        hit = next((r for r in results if r['bprec'] >= target), None)
        if hit:
            bf1 = hit['bf1']
            print(f"  {target:>12.0%}  {hit['t']:>10.4f}  {hit['bprec']:>10.4f}  "
                  f"{hit['brec']:>9.4f}  {hit['bap50']:>10.4f}  {bf1:>9.4f}")
        else:
            print(f"  {target:>12.0%}  {'never':>10s}  —")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute AP50, AP75, AP50-90, Precision, Recall from a greedy-eval CSV"
    )
    parser.add_argument('--csv',       required=True,  help="Eval CSV file (semicolon-sep)")
    parser.add_argument('--csv2',      default=None,   help="Optional second CSV for comparison")
    parser.add_argument('--threshold', type=float, default=0.05,
                        help="Confidence threshold — predictions below this are discarded (default: 0.05)")
    parser.add_argument('--sweep',     action='store_true',
                        help="Sweep thresholds to find optimal operating point (ignores --threshold)")
    parser.add_argument('--label1',    default="exp1", help="Label for first CSV")
    parser.add_argument('--label2',    default="exp2", help="Label for second CSV")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    df1   = load_csv(args.csv)
    preds1, n_gt1 = split_rows(df1)
    print(f"  GT annotations: {n_gt1:,}  |  Candidate predictions: {len(preds1):,}")

    if args.sweep:
        sweep_thresholds(preds1, n_gt1, args.label1)
        if args.csv2:
            print(f"\nLoading {args.csv2} ...")
            df2   = load_csv(args.csv2)
            preds2, n_gt2 = split_rows(df2)
            print(f"  GT annotations: {n_gt2:,}  |  Candidate predictions: {len(preds2):,}")
            sweep_thresholds(preds2, n_gt2, args.label2)
        return

    print(f"  Applying confidence threshold ≥ {args.threshold} ...")
    print("  Computing BBox metrics ...")
    bbox_m1 = compute_metrics(preds1, n_gt1, args.threshold, iou_col='IoU_bb')
    print("  Computing Mask metrics ...")
    mask_m1 = compute_metrics(preds1, n_gt1, args.threshold, iou_col='IoU')

    print_metrics(args.label1, args.threshold, bbox_m1, mask_m1)

    if args.csv2:
        print(f"\nLoading {args.csv2} ...")
        df2   = load_csv(args.csv2)
        preds2, n_gt2 = split_rows(df2)
        print(f"  GT annotations: {n_gt2:,}  |  Candidate predictions: {len(preds2):,}")
        print(f"  Applying confidence threshold ≥ {args.threshold} ...")

        print("  Computing BBox metrics ...")
        bbox_m2 = compute_metrics(preds2, n_gt2, args.threshold, iou_col='IoU_bb')
        print("  Computing Mask metrics ...")
        mask_m2 = compute_metrics(preds2, n_gt2, args.threshold, iou_col='IoU')

        print_metrics(args.label2, args.threshold, bbox_m2, mask_m2)
        print_comparison(args.label1, args.label2, args.threshold,
                         bbox_m1, mask_m1, bbox_m2, mask_m2)


if __name__ == "__main__":
    main()
