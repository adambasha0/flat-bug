#!/usr/bin/env python3
"""
Count IoU statistics for evaluation results, including both mask IoU and bbox IoU.
Computes AP50 and AP50-95 using the same PR-curve integration as the R scripts.

USAGE:
    # For SAM3 results:
    python3 scripts/manuscript/statistics/count_iou.py \
        --csv scripts/manuscript/statistics/data/combined_results_sam3_fine_tuning_mask_0005_with_bb.csv

    # For Flatbug results:
    python3 scripts/manuscript/statistics/count_iou.py \
        --csv scripts/manuscript/statistics/data/combined_results_fb.csv

____________________________________________________________________________
>> cd /home/dolma/repo/flat-bug && python3 scripts/manuscript/statistics/count_iou.py --csv scripts/manuscript/statistics/data/combined_results_sam3_20_2_with_bb.csv

File: `scripts/manuscript/statistics/data/combined_results_sam3_20_2_with_bb.csv`
Total rows: 29711
Computing IoU_bb from bounding boxes...

BEFORE size filter:
  unmatched_pred: 8302   unmatched_gt: 4766   matched: 16643

============================================================
AFTER size>=32 filter (same as R script)
============================================================
total rows: 22097
unmatched_pred: 4481   unmatched_gt: 1703   matched: 15913

Computing AP (PR-curve integration, same method as R)...

IoU (mask):
  matched: 15913  (good≥0.5: 11590, bad<0.5: 4323)
  TP=11590  FP=8804  FN=6026
  Precision@all = 0.5683   Recall@all = 0.6579
  AP50 = 54.09%   AP50-95 = 19.60%

IoU_bb (bounding box):
  matched: 15913  (good≥0.5: 13904, bad<0.5: 2009)
  TP=13904  FP=6490  FN=3712
  Precision@all = 0.6818   Recall@all = 0.7893
  AP50 = 72.03%   AP50-95 = 39.88%

============================================================
COMPARISON: IoU (mask) vs IoU_bb (bounding box)
============================================================
                               AP50      AP50-95
  IoU (mask)                 54.09%       19.60%
  IoU_bb (bounding box)      72.03%       39.88%

Discordance (at IoU=0.5):
  IoU_bb >= 0.5 but IoU < 0.5: 2527 (mask quality issue)
  IoU_bb < 0.5 but IoU >= 0.5: 213 (bbox quality issue)
"""
import pandas as pd
import numpy as np
import ast
import argparse


def parse_bbox(bbox_str):
    """Parse bbox string like '[x1, y1, x2, y2]' to list of ints."""
    if pd.isna(bbox_str) or bbox_str == '':
        return None
    try:
        return ast.literal_eval(bbox_str)
    except:
        return None


def compute_bbox_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    if bbox1 is None or bbox2 is None:
        return np.nan
    
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def classify_at_threshold(df, iou_thresh, iou_col='IoU'):
    """Classify rows as TP/FP/FN at a given IoU threshold.
    
    Same logic as the R script's classify_at_threshold:
    - unmatched_gt  → FN
    - unmatched_pred → FP
    - matched with IoU >= thresh → TP
    - matched with IoU < thresh  → FP (pred) + FN (gt, conf=0)
    """
    fn_unmatched = df[df['match_type'] == 'unmatched_gt'].assign(result='FN')
    fp_unmatched = df[df['match_type'] == 'unmatched_pred'].assign(result='FP')
    tp_matched = df[(df['match_type'] == 'matched') & (df[iou_col] >= iou_thresh)].assign(result='TP')
    bad = df[(df['match_type'] == 'matched') & (df[iou_col] < iou_thresh)]
    fp_bad = bad.assign(result='FP')
    fn_bad = bad.assign(result='FN', conf=0)
    return pd.concat([fn_unmatched, fp_unmatched, tp_matched, fp_bad, fn_bad], ignore_index=True)


def integrate_pr_curve(recall, precision):
    """Trapezoidal integration of the PR curve (same as R script)."""
    df = pd.DataFrame({'x': recall, 'y': precision}).sort_values('x')
    if any(df['x'].duplicated()):
        df = df.groupby('x', as_index=False)['y'].max().sort_values('x')
    dx = np.diff(df['x'].values)
    avg_y = (df['y'].values[:-1] + df['y'].values[1:]) / 2
    return float(np.sum(dx * avg_y))


def compute_ap(df, iou_thresh, iou_col='IoU'):
    """Compute AP at a single IoU threshold via PR curve integration."""
    dt = classify_at_threshold(df, iou_thresh, iou_col)
    n_tp = (dt['result'] == 'TP').sum()
    n_fn = (dt['result'] == 'FN').sum()
    if n_tp == 0:
        return 0.0
    dt = dt.sort_values('conf', ascending=False).reset_index(drop=True)
    dt['tp_cum'] = (dt['result'] == 'TP').cumsum()
    dt['fp_cum'] = (dt['result'] == 'FP').cumsum()
    dt['recall'] = dt['tp_cum'] / (n_tp + n_fn)
    dt['precision'] = dt['tp_cum'] / (dt['tp_cum'] + dt['fp_cum'])
    return integrate_pr_curve(dt['recall'].values, dt['precision'].values)


def compute_ap50_95(df, iou_col='IoU'):
    """Compute AP50, AP50-95 (mean over 0.50..0.95 step 0.01)."""
    thresholds = np.arange(0.50, 0.96, 0.01)
    aps = [compute_ap(df, t, iou_col) for t in thresholds]
    return aps[0], float(np.mean(aps))


def print_iou_section(df, iou_col, label, thresh):
    """Print TP/FP/FN classification and AP for one IoU type."""
    iou = pd.to_numeric(df[iou_col], errors='coerce')
    matched_mask = (df['match_type'] == 'matched')
    unmatched_pred = (df['match_type'] == 'unmatched_pred').sum()
    unmatched_gt = (df['match_type'] == 'unmatched_gt').sum()
    matched = matched_mask.sum()
    
    good = (matched_mask & (iou >= thresh)).sum()
    bad  = (matched_mask & (iou < thresh)).sum()
    
    TP = good
    FP = unmatched_pred + bad
    FN = unmatched_gt + bad
    
    print(f"\n{label}:")
    print(f"  matched: {matched}  (good≥{thresh}: {good}, bad<{thresh}: {bad})")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Precision@all = {TP/(TP+FP):.4f}   Recall@all = {TP/(TP+FN):.4f}")
    
    # Compute actual AP (PR curve integration, same as R script)
    ap50, ap50_95 = compute_ap50_95(df, iou_col)
    print(f"  AP50 = {ap50*100:.2f}%   AP50-95 = {ap50_95*100:.2f}%")
    
    return TP, FP, FN, ap50, ap50_95


def main():
    parser = argparse.ArgumentParser(description='Count IoU statistics')
    parser.add_argument('--csv', type=str, 
                        default='scripts/manuscript/statistics/data/combined_results_sam3_20_2.csv',
                        help='Path to CSV file')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='IoU threshold')
    args = parser.parse_args()
    
    fn = args.csv
    df = pd.read_csv(fn, sep=";")

    print(f"\nFile: `{fn}`")
    print(f"Total rows: {len(df)}")
    
    # Compute IoU_bb for matched pairs
    print("Computing IoU_bb from bounding boxes...")
    def compute_row_iou_bb(row):
        if row['idx_1'] == -1 or row['idx_2'] == -1:
            return np.nan
        bbox1 = parse_bbox(row['bbox_1'])
        bbox2 = parse_bbox(row['bbox_2'])
        return compute_bbox_iou(bbox1, bbox2)
    
    df['IoU_bb'] = df.apply(compute_row_iou_bb, axis=1)
    
    if 'idx_1' not in df.columns or 'idx_2' not in df.columns:
        print("idx_1/idx_2 columns not found; cannot compute stats.")
        return
    
    # Assign match_type and conf (same logic as R script)
    df['match_type'] = np.where(
        (df['idx_1'] != -1) & (df['idx_2'] == -1), 'unmatched_gt',
        np.where(
            (df['idx_1'] == -1) & (df['idx_2'] != -1), 'unmatched_pred',
            'matched'))
    df['conf'] = df['conf2'].fillna(0.)
    
    # Raw counts (before filter)
    unmatched_pred = (df['match_type'] == 'unmatched_pred').sum()
    unmatched_gt = (df['match_type'] == 'unmatched_gt').sum()
    matched = (df['match_type'] == 'matched').sum()
    
    print(f"\nBEFORE size filter:")
    print(f"  unmatched_pred: {unmatched_pred}   unmatched_gt: {unmatched_gt}   matched: {matched}")
    
    # Apply size >= 32 filter (same as R script)
    df['area'] = df.apply(lambda r: r['contourArea_1'] if r['idx_1'] != -1 else r['contourArea_2'], axis=1)
    df['size'] = df['area'] ** 0.5
    df_filt = df[df['size'] >= 32].copy()
    
    unmatched_pred_f = (df_filt['match_type'] == 'unmatched_pred').sum()
    unmatched_gt_f = (df_filt['match_type'] == 'unmatched_gt').sum()
    matched_f = (df_filt['match_type'] == 'matched').sum()
    
    print(f"\n{'='*60}")
    print(f"AFTER size>=32 filter (same as R script)")
    print(f"{'='*60}")
    print(f"total rows: {len(df_filt)}")
    print(f"unmatched_pred: {unmatched_pred_f}   unmatched_gt: {unmatched_gt_f}   matched: {matched_f}")
    
    # Compute stats + AP for both IoU types
    print(f"\nComputing AP (PR-curve integration, same method as R)...")
    TP_m, FP_m, FN_m, ap50_m, ap5095_m = print_iou_section(
        df_filt, 'IoU', 'IoU (mask)', args.thresh)
    TP_b, FP_b, FN_b, ap50_b, ap5095_b = print_iou_section(
        df_filt, 'IoU_bb', 'IoU_bb (bounding box)', args.thresh)
    
    # Comparison summary
    iou_filt = pd.to_numeric(df_filt['IoU'], errors='coerce')
    iou_bb_filt = pd.to_numeric(df_filt['IoU_bb'], errors='coerce')
    matched_mask_f = (df_filt['match_type'] == 'matched')
    
    bbox_high_mask_low = (matched_mask_f & (iou_bb_filt >= args.thresh) & (iou_filt < args.thresh)).sum()
    bbox_low_mask_high = (matched_mask_f & (iou_bb_filt < args.thresh) & (iou_filt >= args.thresh)).sum()
    
    print(f"\n{'='*60}")
    print("COMPARISON: IoU (mask) vs IoU_bb (bounding box)")
    print(f"{'='*60}")
    print(f"  {'':22s} {'AP50':>10s}   {'AP50-95':>10s}")
    print(f"  {'IoU (mask)':22s} {ap50_m*100:9.2f}%   {ap5095_m*100:9.2f}%")
    print(f"  {'IoU_bb (bounding box)':22s} {ap50_b*100:9.2f}%   {ap5095_b*100:9.2f}%")
    
    print(f"\nDiscordance (at IoU={args.thresh}):")
    print(f"  IoU_bb >= {args.thresh} but IoU < {args.thresh}: {bbox_high_mask_low} (mask quality issue)")
    print(f"  IoU_bb < {args.thresh} but IoU >= {args.thresh}: {bbox_low_mask_high} (bbox quality issue)")


if __name__ == '__main__':
    main()