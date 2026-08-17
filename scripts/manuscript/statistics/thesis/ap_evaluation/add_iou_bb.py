#!/usr/bin/env python3
"""
Add IoU_bb (bounding box IoU) column to evaluation CSV files.

USAGE:
    # Add IoU_bb column to SAM3 results:
    nohup python3 add_iou_bb.py \
    --input <data-dir>/sam3_ft_round3_ep18_predictions_mask_5_score_35_combined_results.csv \
    --output <data-dir>/sam3_ft_round3_ep18_predictions_mask_5_score_35_combined_results_with_bb.csv \
    > logs/add_iou_bb_sam3.log 2>&1 &

    # Add IoU_bb column to Flatbug results:
    nohup python3 add_iou_bb.py \
        --input <data-dir>/combined_results_fb_score_001.csv \
        --output <data-dir>/combined_results_fb_score_001_with_bb.csv \
        > logs/add_iou_bb_fb.log 2>&1 &

This script computes IoU between bounding boxes (bbox_1 and bbox_2) for matched pairs.
This is useful for distinguishing:
- Localization errors (detected but wrong location): low IoU_bb
- Segmentation quality issues (correct location but poor mask): high IoU_bb, low IoU (mask)

For matched pairs, if IoU (mask) is low but IoU_bb is high, this suggests the mask 
annotations differ in quality/style rather than actual detection failure.

OUTPUT:
- New CSV with additional column 'IoU_bb'
- Summary statistics comparing IoU (mask) vs IoU_bb
"""

import ast
import argparse
import pandas as pd
import numpy as np


def parse_bbox(bbox_str):
    """Parse bbox string like '[x1, y1, x2, y2]' to list of ints."""
    if pd.isna(bbox_str) or bbox_str == '':
        return None
    try:
        return ast.literal_eval(bbox_str)
    except:
        return None


def compute_bbox_iou(bbox1, bbox2):
    """
    Compute IoU between two bounding boxes [x1, y1, x2, y2].
    
    Returns:
        IoU value between 0 and 1, or NaN if either bbox is invalid
    """
    if bbox1 is None or bbox2 is None:
        return np.nan
    
    # Intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Check if boxes don't overlap
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def add_iou_bb(input_path, output_path, sep=';'):
    """Add IoU_bb column to CSV file."""
    
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path, sep=sep)
    print(f"  Loaded {len(df)} rows")
    
    # Check required columns
    required_cols = ['bbox_1', 'bbox_2', 'idx_1', 'idx_2']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Compute IoU_bb for all rows
    print("Computing IoU_bb...")
    
    def compute_row_iou_bb(row):
        # Only compute for matched pairs (both idx_1 and idx_2 != -1)
        if row['idx_1'] == -1 or row['idx_2'] == -1:
            return np.nan
        
        bbox1 = parse_bbox(row['bbox_1'])
        bbox2 = parse_bbox(row['bbox_2'])
        return compute_bbox_iou(bbox1, bbox2)
    
    df['IoU_bb'] = df.apply(compute_row_iou_bb, axis=1)
    
    # Statistics
    matched = df[(df['idx_1'] != -1) & (df['idx_2'] != -1)]
    valid_iou_bb = matched[matched['IoU_bb'].notna()]
    
    print(f"\n{'='*70}")
    print("STATISTICS FOR MATCHED PAIRS")
    print(f"{'='*70}")
    print(f"Total rows: {len(df)}")
    print(f"Matched pairs: {len(matched)}")
    print(f"Valid IoU_bb computed: {len(valid_iou_bb)}")
    
    if len(valid_iou_bb) > 0:
        print(f"\nIoU (mask) statistics:")
        print(f"  Mean:   {valid_iou_bb['IoU'].mean():.4f}")
        print(f"  Median: {valid_iou_bb['IoU'].median():.4f}")
        print(f"  Std:    {valid_iou_bb['IoU'].std():.4f}")
        print(f"  Min:    {valid_iou_bb['IoU'].min():.4f}")
        print(f"  Max:    {valid_iou_bb['IoU'].max():.4f}")
        
        print(f"\nIoU_bb (bbox) statistics:")
        print(f"  Mean:   {valid_iou_bb['IoU_bb'].mean():.4f}")
        print(f"  Median: {valid_iou_bb['IoU_bb'].median():.4f}")
        print(f"  Std:    {valid_iou_bb['IoU_bb'].std():.4f}")
        print(f"  Min:    {valid_iou_bb['IoU_bb'].min():.4f}")
        print(f"  Max:    {valid_iou_bb['IoU_bb'].max():.4f}")
        
        # Correlation
        corr = valid_iou_bb[['IoU', 'IoU_bb']].corr()
        print(f"\nCorrelation between IoU and IoU_bb: {corr.loc['IoU', 'IoU_bb']:.4f}")
        
        # Classification comparison
        iou_thresh = 0.5
        tp_mask = valid_iou_bb['IoU'] >= iou_thresh
        tp_bb = valid_iou_bb['IoU_bb'] >= iou_thresh
        
        print(f"\nClassification at IoU >= {iou_thresh}:")
        print(f"  TP by mask IoU: {tp_mask.sum():6d} ({100*tp_mask.mean():.1f}%)")
        print(f"  TP by bbox IoU: {tp_bb.sum():6d} ({100*tp_bb.mean():.1f}%)")
        
        # Cases where bbox IoU is high but mask IoU is low
        # (suggesting detection is correct but mask annotation quality differs)
        bbox_high_mask_low = (valid_iou_bb['IoU_bb'] >= iou_thresh) & (valid_iou_bb['IoU'] < iou_thresh)
        bbox_low_mask_high = (valid_iou_bb['IoU_bb'] < iou_thresh) & (valid_iou_bb['IoU'] >= iou_thresh)
        both_low = (valid_iou_bb['IoU_bb'] < iou_thresh) & (valid_iou_bb['IoU'] < iou_thresh)
        both_high = (valid_iou_bb['IoU_bb'] >= iou_thresh) & (valid_iou_bb['IoU'] >= iou_thresh)
        
        print(f"\nDiscordance analysis (IoU_bb vs IoU):")
        print(f"  Both >= {iou_thresh}:                 {both_high.sum():6d} ({100*both_high.mean():.1f}%)")
        print(f"  Both < {iou_thresh}:                  {both_low.sum():6d} ({100*both_low.mean():.1f}%)")
        print(f"  IoU_bb >= {iou_thresh}, IoU < {iou_thresh}:   {bbox_high_mask_low.sum():6d} ({100*bbox_high_mask_low.mean():.1f}%) <- Mask quality issue")
        print(f"  IoU_bb < {iou_thresh}, IoU >= {iou_thresh}:   {bbox_low_mask_high.sum():6d} ({100*bbox_low_mask_high.mean():.1f}%) <- Bbox quality issue")
        
        if bbox_high_mask_low.sum() > 0:
            subset = valid_iou_bb[bbox_high_mask_low]
            print(f"\n  Cases with IoU_bb >= {iou_thresh} but IoU < {iou_thresh}:")
            print(f"    Mean IoU:    {subset['IoU'].mean():.4f}")
            print(f"    Mean IoU_bb: {subset['IoU_bb'].mean():.4f}")
            print(f"    ** These are correctly LOCALIZED but have poor mask overlap **")
    
    # Save output
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, sep=sep, index=False)
    print(f"Done!")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Add IoU_bb column to evaluation CSV')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file path (default: input with _with_bb suffix)')
    parser.add_argument('--sep', type=str, default=';',
                        help='CSV separator (default: ;)')
    
    args = parser.parse_args()
    
    if args.output is None:
        # Add _with_bb suffix before .csv extension
        if args.input.endswith('.csv'):
            args.output = args.input[:-4] + '_with_bb.csv'
        else:
            args.output = args.input + '_with_bb.csv'
    
    add_iou_bb(args.input, args.output, args.sep)


if __name__ == '__main__':
    main()
