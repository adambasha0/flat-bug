#!/usr/bin/env python3
"""
FP Analysis and Confidence Threshold Optimization for Fine-tuned SAM3.

This script analyzes False Positives, especially "hallucinations" (unmatched predictions),
and simulates the effect of raising confidence thresholds on TP/FP tradeoffs.

Usage:
    python3 scripts/manuscript/statistics/analyze_fp_threshold.py

    combined_results_fb_score_001_with_bb.csv: flatbug
        Total rows: 32678,
        After size >= 32 filter: 25658 rows
        IoU (mask): AP50: 96.12% | AP50-95: 62.16%
        IoU (bounding box): AP50: 96.13% | AP50-95: 70.96%

        TP: 17183, FP: 8369 (unmatched: 8042, low_iou: 327)
        FN: 433 (unmatched: 106, low_iou: 327)
        Precision: 0.6725, Recall: 0.9754


    combined_results_sam3_005_with_bb.csv (score: 0.005)
        Total rows: 817968
        After size >= 32 filter: 228089 rows
        IoU (mask): AP50: 37.48% | AP50-95: 15.00%
        IoU (bounding box): AP50: 45.22% | AP50-95: 26.98%

        
        TP: 13085, FP: 214478 (unmatched: 210473, low_iou: 4005)
        FN: 4531 (unmatched: 526, low_iou: 4005)
        Precision: 0.0575, Recall: 0.7428

    combined_results_sam3_ft_mask_05_checkpoint18_score_001_greedy:
        Total rows: 874514
        After size >= 32 filter: 868842 rows
        Precision: 0.0189, Recall: 0.9483
        IoU (mask): AP50: 86.67% | AP50-95: 48.52%
        IoU (bounding box): AP50: 87.07% | AP50-95: 60.08%
        
        TP: 16405, FP: 852306 (unmatched: 851543, low_iou: 763)
        FN: 894 (unmatched: 131, low_iou: 763)
        Precision: 0.0189, Recall: 0.9483

    combined_results_sam3_fine_tuning_with_lora_mask_05_score_001_using_greedy_eval_with_bb.csv
        Total rows: 512721
        After size >= 32 filter: 506549 rows
        Precision: 0.0323, Recall: 0.9442
        IoU (mask): AP50: 86.51% | AP50-95: 48.12%
        IoU (bounding box): AP50: 86.68% | AP50-95: 59.45%
        
        TP: 16333, FP: 490041 (unmatched: 489250, low_iou: 791)
        FN: 966 (unmatched: 175, low_iou: 791)
        Precision: 0.0323, Recall: 0.9442

    scripts/manuscript/statistics/data/finetuned/lora/fine_tuned_sam3_with_lora_checkpoint_10_mask_5_score_005_with_bb.csv
        Total rows: 340124
        After size >= 32 filter: 335903 rows

        TP: 16078, FP: 319599 (unmatched: 318604, low_iou: 995)
        FN: 1221 (unmatched: 226, low_iou: 995)
        Precision: 0.0479, Recall: 0.9294

    scripts/manuscript/statistics/data/finetuned/lora/anti_hallucination_lora_v3_checkpoint_6_mask_5_score_005_using_greed_eval_with_bb.csv
        Total rows: 328606
        After size >= 32 filter: 324381 rows

        TP: 16032, FP: 308112 (unmatched: 307082, low_iou: 1030)
        FN: 1267 (unmatched: 237, low_iou: 1030)
        Precision: 0.0495, Recall: 0.9268

    scripts/manuscript/statistics/data/finetuned/lora/anti_hallucination_lora_v3_checkpoint_6_mask_5_score_30_using_greed_eval_with_bb.csv...
        Total rows: 24675
        After size >= 32 filter: 20555 rows

        TP: 15798, FP: 4049 (unmatched: 3256, low_iou: 793)
        FN: 1501 (unmatched: 708, low_iou: 793)
        Precision: 0.7960, Recall: 0.9132

    scripts/manuscript/statistics/data/finetuned/lora/anti_hallucination_lora_v3_checkpoint_6_mask_5_score_35_using_greed_eval_with_bb.csv
        Total rows: 24414
        After size >= 32 filter: 20295 rows

        TP: 15785, FP: 3775 (unmatched: 2996, low_iou: 779)
        FN: 1514 (unmatched: 735, low_iou: 779)
        Precision: 0.8070, Recall: 0.9125
    
    scripts/manuscript/statistics/data/finetuned/combined_results_sam3_ft_mask_05_checkpoint18_score_30_with_bb.csv
        Total rows: 24894
        After size >= 32 filter: 20203 rows

        TP: 16651, FP: 2964 (unmatched: 2587, low_iou: 377)
        FN: 965 (unmatched: 588, low_iou: 377)
        Precision: 0.8489, Recall: 0.9452

    scripts/manuscript/statistics/data/finetuned/combined_results_sam3_ft_mask_5_checkpoint18_score_35_with_bb.csv
        Total rows: 24552
        After size >= 32 filter: 19688 rows

        TP: 16456, FP: 2620 (unmatched: 2072, low_iou: 548)
        FN: 1160 (unmatched: 612, low_iou: 548)
        Precision: 0.8627, Recall: 0.9342
""" 

import pandas as pd
import numpy as np
import sys

# Configuration
CSV_PATH = "scripts/manuscript/statistics/data/finetuned/lora/anti_hallucination_lora_v3_checkpoint_6_mask_5_score_35_using_greed_eval_with_bb.csv"
IOU_THRESHOLD = 0.5
SIZE_THRESHOLD = 32  # sqrt(area) minimum

def classify_row(row, iou_thresh=0.5):
    """Classify row as TP, FP_unmatched, FP_low_iou, FN_unmatched, FN_low_iou."""
    idx_1 = row['idx_1']  # GT index
    idx_2 = row['idx_2']  # Prediction index
    iou = row['IoU'] if pd.notna(row['IoU']) else 0.0
    
    if idx_1 != -1 and idx_2 == -1:
        return 'FN_unmatched'  # Missed GT
    elif idx_1 == -1 and idx_2 != -1:
        return 'FP_unmatched'  # Hallucination - no GT
    elif idx_1 != -1 and idx_2 != -1:
        if iou >= iou_thresh:
            return 'TP'
        else:
            return 'BAD_MATCH'  # Both FP and FN due to low IoU
    return 'OTHER'


def main():
    print("=" * 80)
    print("FALSE POSITIVE ANALYSIS & CONFIDENCE THRESHOLD OPTIMIZATION")
    print("Fine-tuned SAM3 Model")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, sep=';')
    print(f"Total rows: {len(df)}")
    
    # Apply size filter
    df['area'] = df.apply(lambda r: r['contourArea_1'] if r['idx_1'] != -1 else r['contourArea_2'], axis=1)
    df['size'] = df['area'] ** 0.5
    df = df[df['size'] >= SIZE_THRESHOLD].copy()
    print(f"After size >= {SIZE_THRESHOLD} filter: {len(df)} rows")
    
    # Classify each row
    df['classification'] = df.apply(lambda r: classify_row(r, IOU_THRESHOLD), axis=1)
    
    # Get confidence scores (conf2 is prediction confidence)
    df['conf'] = df['conf2']
    
    # ============================================================
    # TASK 1: Analyze Unmatched FPs (Hallucinations)
    # ============================================================
    print("\n" + "=" * 80)
    print("TASK 1: UNMATCHED FP (HALLUCINATION) ANALYSIS")
    print("=" * 80)
    
    fp_unmatched = df[df['classification'] == 'FP_unmatched'].copy()
    print(f"\nTotal Unmatched FPs (hallucinations): {len(fp_unmatched)}")
    
    if len(fp_unmatched) > 0:
        conf_fp = fp_unmatched['conf'].dropna()
        print(f"\nConfidence Score Statistics for Unmatched FPs:")
        print(f"  Count:        {len(conf_fp)}")
        print(f"  Mean:         {conf_fp.mean():.4f}")
        print(f"  Median:       {conf_fp.median():.4f}")
        print(f"  Min:          {conf_fp.min():.4f}")
        print(f"  Max:          {conf_fp.max():.4f}")
        print(f"  25th pct:     {conf_fp.quantile(0.25):.4f}")
        print(f"  75th pct:     {conf_fp.quantile(0.75):.4f}")
        print(f"  90th pct:     {conf_fp.quantile(0.90):.4f}")
        print(f"  95th pct:     {conf_fp.quantile(0.95):.4f}")
        
        fp_median = conf_fp.median()
        fp_75 = conf_fp.quantile(0.75)
        fp_90 = conf_fp.quantile(0.90)
    else:
        print("  No unmatched FPs found!")
        fp_median, fp_75, fp_90 = 0.5, 0.7, 0.9
    
    # Also analyze FP from low IoU matches
    fp_low_iou = df[df['classification'] == 'BAD_MATCH'].copy()
    print(f"\nTotal Low-IoU Match FPs: {len(fp_low_iou)}")
    if len(fp_low_iou) > 0:
        conf_low = fp_low_iou['conf'].dropna()
        print(f"  Mean conf:    {conf_low.mean():.4f}")
        print(f"  Median conf:  {conf_low.median():.4f}")
    
    # Analyze TPs for comparison
    print("\n" + "-" * 40)
    tps = df[df['classification'] == 'TP'].copy()
    print(f"\nTotal True Positives: {len(tps)}")
    if len(tps) > 0:
        conf_tp = tps['conf'].dropna()
        print(f"\nConfidence Score Statistics for TPs:")
        print(f"  Mean:         {conf_tp.mean():.4f}")
        print(f"  Median:       {conf_tp.median():.4f}")
        print(f"  Min:          {conf_tp.min():.4f}")
        print(f"  Max:          {conf_tp.max():.4f}")
        print(f"  5th pct:      {conf_tp.quantile(0.05):.4f}")
        print(f"  10th pct:     {conf_tp.quantile(0.10):.4f}")
    
    # ============================================================
    # TASK 2: Threshold Tradeoff Simulation
    # ============================================================
    print("\n" + "=" * 80)
    print("TASK 2: CONFIDENCE THRESHOLD TRADEOFF SIMULATION")
    print("=" * 80)
    
    # Candidate thresholds
    if len(fp_unmatched) > 0:
        candidate_thresholds = [
            0.20,  # Current (low)
            fp_median,  # Median of FP
            fp_75,  # 75th percentile
            fp_90,  # 90th percentile
            0.50,  # Middle ground
            0.70,  # High
            0.80,  # Very high
            0.90,  # Extreme
        ]
    else:
        candidate_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    # Remove duplicates and sort
    candidate_thresholds = sorted(set([round(t, 4) for t in candidate_thresholds]))
    
    # Current baseline
    total_tp_baseline = len(tps)
    total_fp_unmatched_baseline = len(fp_unmatched)
    total_fp_low_iou_baseline = len(fp_low_iou)
    total_fn_baseline = len(df[df['classification'] == 'FN_unmatched'])
    
    # Include low IoU as both FP and FN
    total_fp_baseline = total_fp_unmatched_baseline + total_fp_low_iou_baseline
    total_fn_all = total_fn_baseline + total_fp_low_iou_baseline  # Low IoU counts as FN too
    
    precision_baseline = total_tp_baseline / (total_tp_baseline + total_fp_baseline) if (total_tp_baseline + total_fp_baseline) > 0 else 0
    recall_baseline = total_tp_baseline / (total_tp_baseline + total_fn_all) if (total_tp_baseline + total_fn_all) > 0 else 0
    
    print(f"\nBaseline (threshold=!!):")
    print(f"  TP: {total_tp_baseline}, FP: {total_fp_baseline} (unmatched: {total_fp_unmatched_baseline}, low_iou: {total_fp_low_iou_baseline})")
    print(f"  FN: {total_fn_all} (unmatched: {total_fn_baseline}, low_iou: {total_fp_low_iou_baseline})")
    print(f"  Precision: {precision_baseline:.4f}, Recall: {recall_baseline:.4f}")
    
    results = []
    
    for thresh in candidate_thresholds:
        # Count what would be filtered
        fp_deleted = len(fp_unmatched[fp_unmatched['conf'] < thresh])
        fp_low_iou_deleted = len(fp_low_iou[fp_low_iou['conf'] < thresh])
        tp_deleted = len(tps[tps['conf'] < thresh])
        
        # New counts
        new_tp = total_tp_baseline - tp_deleted
        new_fp_unmatched = total_fp_unmatched_baseline - fp_deleted
        new_fp_low_iou = total_fp_low_iou_baseline - fp_low_iou_deleted
        new_fp = new_fp_unmatched + new_fp_low_iou
        new_fn_from_deleted_tp = tp_deleted  # Deleted TPs become FN
        new_fn_total = total_fn_all + new_fn_from_deleted_tp
        
        # Metrics
        new_precision = new_tp / (new_tp + new_fp) if (new_tp + new_fp) > 0 else 0
        new_recall = new_tp / (new_tp + new_fn_total) if (new_tp + new_fn_total) > 0 else 0
        
        precision_gain = (new_precision - precision_baseline) * 100
        recall_loss = (recall_baseline - new_recall) * 100
        tp_loss_pct = (tp_deleted / total_tp_baseline * 100) if total_tp_baseline > 0 else 0
        fp_reduction_pct = (fp_deleted / total_fp_unmatched_baseline * 100) if total_fp_unmatched_baseline > 0 else 0
        
        results.append({
            'threshold': thresh,
            'fp_deleted': fp_deleted,
            'tp_deleted': tp_deleted,
            'new_tp': new_tp,
            'new_fp': new_fp,
            'new_precision': new_precision,
            'new_recall': new_recall,
            'precision_gain_pct': precision_gain,
            'recall_loss_pct': recall_loss,
            'tp_loss_pct': tp_loss_pct,
            'fp_reduction_pct': fp_reduction_pct,
        })
    
    # ============================================================
    # TASK 3: Executive Summary Table
    # ============================================================
    print("\n" + "=" * 80)
    print("TASK 3: EXECUTIVE SUMMARY")
    print("=" * 80)
    
    print("\n### Confidence Threshold Tradeoff Analysis\n")
    print("| Threshold | FP Deleted | TP Lost | TP Loss % | FP Reduction % | Precision | Recall | Prec Gain | Recall Loss |")
    print("|-----------|------------|---------|-----------|----------------|-----------|--------|-----------|-------------|")
    
    for r in results:
        print(f"| {r['threshold']:.4f}    | {r['fp_deleted']:>10d} | {r['tp_deleted']:>7d} | {r['tp_loss_pct']:>8.2f}% | {r['fp_reduction_pct']:>13.1f}% | {r['new_precision']:>9.4f} | {r['new_recall']:.4f} | {r['precision_gain_pct']:>+8.2f}% | {r['recall_loss_pct']:>10.2f}% |")
    
    # Find sweet spot
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    # Find threshold that removes most FP with < 2% TP loss
    sweet_spot = None
    for r in results:
        if r['tp_loss_pct'] < 2.0 and r['fp_reduction_pct'] > 50:
            if sweet_spot is None or r['fp_reduction_pct'] > sweet_spot['fp_reduction_pct']:
                sweet_spot = r
    
    if sweet_spot:
        print(f"\n✅ SWEET SPOT FOUND: threshold = {sweet_spot['threshold']:.4f}")
        print(f"   - Removes {sweet_spot['fp_reduction_pct']:.1f}% of hallucinations")
        print(f"   - Loses only {sweet_spot['tp_loss_pct']:.2f}% of TPs")
        print(f"   - Precision: {sweet_spot['new_precision']:.4f}, Recall: {sweet_spot['new_recall']:.4f}")
    else:
        print("\n⚠️  NO SWEET SPOT: Confidence distributions heavily overlap.")
        print("   Raising threshold will damage recall significantly.")
        print("   Recommendation: Implement Hard Negative Mining fine-tuning.")
    
    # Analyze overlap
    print("\n" + "-" * 40)
    print("DISTRIBUTION OVERLAP ANALYSIS")
    print("-" * 40)
    
    if len(fp_unmatched) > 0 and len(tps) > 0:
        fp_below_50 = (fp_unmatched['conf'] < 0.5).sum()
        tp_below_50 = (tps['conf'] < 0.5).sum()
        fp_above_80 = (fp_unmatched['conf'] >= 0.8).sum()
        tp_above_80 = (tps['conf'] >= 0.8).sum()
        
        print(f"\nConfidence < 0.50:")
        print(f"  FPs: {fp_below_50} ({100*fp_below_50/len(fp_unmatched):.1f}%)")
        print(f"  TPs: {tp_below_50} ({100*tp_below_50/len(tps):.1f}%)")
        
        print(f"\nConfidence >= 0.80:")
        print(f"  FPs: {fp_above_80} ({100*fp_above_80/len(fp_unmatched):.1f}%)")
        print(f"  TPs: {tp_above_80} ({100*tp_above_80/len(tps):.1f}%)")
        
        # Degree of separability
        fp_median_val = fp_unmatched['conf'].median()
        tp_10pct = tps['conf'].quantile(0.10)
        
        print(f"\nSeparability indicator:")
        print(f"  FP median:     {fp_median_val:.4f}")
        print(f"  TP 10th pct:   {tp_10pct:.4f}")
        
        if fp_median_val < tp_10pct:
            print(f"  ✅ GOOD: FP median < TP 10th percentile → Distributions separable")
        else:
            print(f"  ⚠️  POOR: Significant overlap → Hard to threshold without collateral damage")
    
    # Dataset breakdown
    print("\n" + "=" * 80)
    print("BREAKDOWN BY DATASET")
    print("=" * 80)
    
    df['dataset'] = df['image'].apply(lambda x: x.split('_')[0])
    
    print("\n| Dataset | TP | FP_unmatch | FP_low_iou | FN_unmatch | FN_low_iou |")
    print("|---------|-----|------------|------------|------------|------------|")
    
    for dataset, group in df.groupby('dataset'):
        n_tp = (group['classification'] == 'TP').sum()
        n_fp_u = (group['classification'] == 'FP_unmatched').sum()
        n_bad = (group['classification'] == 'BAD_MATCH').sum()
        n_fn_u = (group['classification'] == 'FN_unmatched').sum()
        print(f"| {dataset:<7s} | {n_tp:>3d} | {n_fp_u:>10d} | {n_bad:>10d} | {n_fn_u:>10d} | {n_bad:>10d} |")
    
    # Final recommendations
    print("\n" + "=" * 80)
    print("ACTIONABLE RECOMMENDATIONS TO REACH AP > 90%")
    print("=" * 80)
    
    print("""
Based on this analysis:

1. **Immediate Action - Threshold Tuning:**
   - If sweet spot exists: Apply recommended threshold
   - If not: Keep current threshold, focus on other improvements

2. **Address Low-IoU Matches (Segmentation Quality):**
   - These count as BOTH FP and FN, double-penalizing AP
   - Solution: Already done with MASK_THRESHOLD=0.005
   - Verify with IoU_bb (bounding box IoU) vs IoU (mask IoU)

3. **Reduce Unmatched FPs (Hallucinations):**
   - Option A: Raise confidence threshold (if separable)
   - Option B: Hard Negative Mining - train on FP regions
   - Option C: Post-processing NMS with stricter IoU

4. **Reduce False Negatives (Missed Detections):**
   - Lower SCORE_THRESHOLD in SAM3 (currently 0.2)
   - Use multi-scale pyramid more aggressively
   - Add data augmentation for small/occluded insects

5. **Dataset-Specific Issues:**
   - Identify datasets with highest error rates
   - Analyze if GT annotation quality is a factor
   - Consider dataset-specific calibration
""")


if __name__ == "__main__":
    main()
