#!/usr/bin/env python3
"""
Estimate optimal mask dilation to match GT annotation style.

This script analyzes the area ratio between SAM3 predictions and GT masks,
and estimates what dilation kernel would bring SAM3 masks closer to GT.

Usage:
    python3 scripts/manuscript/statistics/estimate_dilation.py
"""

import pandas as pd
import numpy as np
import ast
import cv2


def parse_contour(contour_str):
    """Parse contour string to numpy array."""
    if pd.isna(contour_str) or contour_str == '':
        return None
    try:
        pts = ast.literal_eval(contour_str)
        if isinstance(pts, list) and len(pts) >= 6:
            return np.array(pts).reshape(-1, 2).astype(np.int32)
    except:
        pass
    return None


def dilate_contour(contour, dilation_px, img_shape=(2048, 2048)):
    """Dilate a contour by rendering to mask, dilating, and re-extracting."""
    if contour is None or len(contour) < 3:
        return None, 0
    
    # Create a minimal bounding box for efficiency
    x_min, y_min = contour.min(axis=0)
    x_max, y_max = contour.max(axis=0)
    
    # Add padding for dilation
    pad = dilation_px + 5
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(img_shape[1], x_max + pad)
    y_max = min(img_shape[0], y_max + pad)
    
    # Render contour to mask
    h, w = int(y_max - y_min), int(x_max - x_min)
    if h <= 0 or w <= 0:
        return None, 0
    
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = contour - np.array([x_min, y_min])
    cv2.fillPoly(mask, [shifted_contour], 255)
    
    original_area = int(mask.sum() // 255)
    
    # Apply dilation
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_px+1, 2*dilation_px+1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    dilated_area = int(mask.sum() // 255)
    
    return mask, dilated_area


def estimate_optimal_dilation(df, sample_size=500):
    """Estimate optimal dilation to match GT mask areas."""
    
    # Filter to matched pairs with mask quality issues
    matched = df[(df['idx_1'] != -1) & (df['idx_2'] != -1)].copy()
    
    # Focus on cases where SAM3 is undersized
    area_ratio = matched['contourArea_2'] / matched['contourArea_1']
    undersized = matched[area_ratio < 0.9].copy()
    
    print(f"Total matched pairs: {len(matched)}")
    print(f"Undersized SAM3 masks (<90% of GT): {len(undersized)} ({100*len(undersized)/len(matched):.1f}%)")
    
    # Sample for efficiency
    if len(undersized) > sample_size:
        sample = undersized.sample(sample_size, random_state=42)
    else:
        sample = undersized
    
    print(f"Analyzing {len(sample)} samples...")
    
    # Test different dilation values
    dilation_values = [0, 2, 4, 6, 8, 10, 12, 15, 20]
    
    results = {d: [] for d in dilation_values}
    
    for _, row in sample.iterrows():
        contour_pred = parse_contour(row['contour_2'])
        gt_area = row['contourArea_1']
        
        if contour_pred is None or gt_area <= 0:
            continue
        
        original_area = row['contourArea_2']
        
        for dilation_px in dilation_values:
            _, dilated_area = dilate_contour(contour_pred, dilation_px)
            if dilated_area > 0:
                ratio = dilated_area / gt_area
                results[dilation_px].append(ratio)
    
    print(f"\n{'='*60}")
    print("DILATION ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"\nArea Ratio (SAM3 / GT) after dilation:")
    print(f"{'Dilation':>10s} {'Mean':>10s} {'Median':>10s} {'% ≈ GT':>10s}")
    print(f"{'-'*40}")
    
    optimal_dilation = 0
    best_median = 0
    
    for d in dilation_values:
        if len(results[d]) > 0:
            mean_ratio = np.mean(results[d])
            median_ratio = np.median(results[d])
            pct_close = 100 * np.mean(np.abs(np.array(results[d]) - 1.0) < 0.15)
            
            print(f"{d:>10d} {mean_ratio:>10.3f} {median_ratio:>10.3f} {pct_close:>9.1f}%")
            
            # Track best dilation (closest median to 1.0)
            if abs(median_ratio - 1.0) < abs(best_median - 1.0):
                best_median = median_ratio
                optimal_dilation = d
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDED DILATION: {optimal_dilation} pixels")
    print(f"Expected median area ratio after dilation: {best_median:.3f}")
    print(f"{'='*60}")
    
    return optimal_dilation, results


def main():
    # Load data
    csv_path = 'scripts/manuscript/statistics/data/combined_results_sam3_20_2_with_bb.csv'
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, sep=';')
    
    optimal_d, results = estimate_optimal_dilation(df, sample_size=800)
    
    print(f"""
NEXT STEPS:
1. Apply dilation of {optimal_d}px to SAM3 predictions in sam3_predict.py:
   - Set MASK_DILATION_PIXELS = {optimal_d} in V2_OPTIONS
   - Re-run SAM3 predictions
   
2. Or apply post-hoc dilation to existing predictions:
   python3 scripts/manuscript/statistics/apply_dilation.py \\
       --input predictions/coco_instances.json \\
       --dilation {optimal_d}

3. Re-compute IoU and AP metrics
""")


if __name__ == "__main__":
    main()
