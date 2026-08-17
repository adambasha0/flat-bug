#!/usr/bin/env python3
"""
Visualize failure examples (FP/FN) for SAM3 / flat-bug predictions from
fb_eval-based combined-results CSVs.

USAGE:
    # Run from repo root:
    nohup python3 visualize_failure_examples.py \
        --csv <data-dir>/combined_results_sam3_ft_mask_5_checkpoint18_score_35_with_bb.csv \
        --output-dir src/failure_examples_sam3_ft_round3_ep18_score_35 \
        --images-per-dataset 4 \
        > logs/visualize_sam3_ft_round3_ep18_score_35.log 2>&1 &

This script:
1. Reads a combined results CSV file (';'-separated, columns: image, idx_1, idx_2,
   bbox_1, bbox_2, contourArea_1, contourArea_2, IoU, conf1, conf2[, contour_1,
   contour_2, IoU_bb]). contourArea_1/2 are scalar pixel-area numbers (used only
   for the size filter) - they are NOT polygon geometry. Real per-object masks are
   only available when contour_1/contour_2 columns are present in the CSV (a list
   of [x, y] polygon vertices); when absent, the script silently falls back to
   drawing bounding boxes only.
2. Classifies each row as TP/FP/FN using the mask IoU column (>= --iou-thresh)
3. Distinguishes between:
   - COMPLETELY MISSED: unmatched GT (no prediction at all)
   - COMPLETELY FALSE: unmatched prediction (no GT at all)
   - LOW IoU MATCH: matched but IoU < threshold (segmentation quality issue,
     further split into localization-off vs. mask-only using IoU_bb when present)
4. For each dataset, finds images with:
   - Most total errors (worst overall)
   - Most completely missed GT (worst detection)
   - Least errors (best)
5. Produces visualization images with these colors (GT vs. predicted, not TP-only):
   - Green: GT (ground truth) box/mask - drawn on the TP case, alongside its match
   - Blue: predicted box/mask that is a TP (correctly matched to a GT)
   - Yellow: predicted box/mask with no matching GT at all (FP, unmatched)
   - Red: GT box/mask with no matching prediction at all (FN, unmatched/missed)
   - Orange: matched pair but IoU < threshold (drawn on both GT and prediction,
     labeled with mask IoU / bbox IoU / confidence) - a low-quality/partial detection
6. Outputs to specified directory

SCIENTIFIC NOTE:
When a matched pair has IoU < threshold, it counts as BOTH FP and FN. This is correct
for AP calculation but may seem like "double penalty". This script differentiates:
- Detection failures: completely missed GT or completely false predictions
- Segmentation quality issues: detected but low IoU (could be due to mask annotation quality)
When the CSV has an IoU_bb column, a low-IoU match with high IoU_bb is flagged as a
mask-only quality issue (correct localization, poor mask overlap) rather than a
localization failure.
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_bbox(bbox_str):
    """Parse bbox string like '[x1, y1, x2, y2]' to list of ints.

    Uses json.loads rather than ast.literal_eval: both parse this syntax fine, but
    literal_eval routes through compile()/the ast module which is ~15x slower - real
    money at score=0.005 scale where a single image can carry 10k+ of these strings.
    """
    if pd.isna(bbox_str) or bbox_str == '' or bbox_str == '[]':
        return None
    try:
        val = json.loads(bbox_str)
        return val if val else None
    except (ValueError, json.JSONDecodeError):
        return None


def parse_contour(contour_str):
    """Parse contour string like '[[x1, y1], [x2, y2], ...]' to an (N, 2) int32 array."""
    if pd.isna(contour_str) or contour_str == '' or contour_str == '[]':
        return None
    try:
        pts = json.loads(contour_str)
        if not pts:
            return None
        return np.array(pts, dtype=np.int32)
    except (ValueError, json.JSONDecodeError):
        return None


def classify_row_detailed(row, iou_thresh=0.5, bbox_iou_thresh=0.5, has_iou_bb=False):
    """
    Classify a row with detailed categorization.

    Returns dict with:
        'type': 'TP', 'FP_UNMATCHED', 'FN_UNMATCHED', 'BAD_MATCH'
        'iou': mask IoU value (for matched pairs)
        'iou_bb': bbox IoU value (for matched pairs, if available)
        'mask_only_issue': True if matched, low mask IoU, but bbox IoU is high
            (i.e. correctly localized, poor mask/annotation overlap)
    """
    idx_1 = row['idx_1']  # GT index
    idx_2 = row['idx_2']  # Prediction index
    iou = row['IoU'] if pd.notna(row['IoU']) else 0.0
    iou_bb = row['IoU_bb'] if has_iou_bb and pd.notna(row.get('IoU_bb')) else None

    if idx_1 != -1 and idx_2 == -1:
        return {'type': 'FN_UNMATCHED', 'iou': None, 'iou_bb': None, 'mask_only_issue': False}
    elif idx_1 == -1 and idx_2 != -1:
        return {'type': 'FP_UNMATCHED', 'iou': None, 'iou_bb': None, 'mask_only_issue': False}
    elif idx_1 != -1 and idx_2 != -1:
        if iou >= iou_thresh:
            return {'type': 'TP', 'iou': iou, 'iou_bb': iou_bb, 'mask_only_issue': False}
        else:
            mask_only_issue = iou_bb is not None and iou_bb >= bbox_iou_thresh
            return {'type': 'BAD_MATCH', 'iou': iou, 'iou_bb': iou_bb, 'mask_only_issue': mask_only_issue}
    return {'type': None, 'iou': None, 'iou_bb': None, 'mask_only_issue': False}


def find_image_path(image_name, dataset_dirs):
    """Find the actual image file path."""
    parts = image_name.split('_')
    dataset = parts[0]
    rest = '_'.join(parts[1:])

    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    for base_dir in dataset_dirs:
        dataset_path = Path(base_dir) / dataset
        if dataset_path.exists():
            for ext in extensions:
                img_path = dataset_path / f"{rest}{ext}"
                if img_path.exists():
                    return str(img_path)
                img_path = dataset_path / f"{image_name}{ext}"
                if img_path.exists():
                    return str(img_path)
    return None


MASK_ALPHA = 0.3  # single blend alpha shared by all queued mask fills (see queue_shape)


def queue_shape(overlay, draw_queue, bbox, contour, color, thickness=2, label=None, label_color=None):
    """
    Fill `contour` (if any) onto the shared `overlay` right away (cheap - proportional
    to polygon area, not image size), and queue the outline/bbox/label for later.

    Compositing the overlay into the image is a full-image-sized copy + blend
    (cv2.addWeighted); doing that per-detection is O(n_detections * image_size),
    which is ruinous for score=0.005 images with thousands of detections. Instead
    ALL masks for an image are filled onto one shared overlay and blended in with
    a SINGLE addWeighted call (see visualize_image), then outlines/boxes/labels -
    which are cheap, small-region draws - are applied directly on top afterwards.
    """
    if bbox is None and contour is None:
        return
    if contour is not None and len(contour) > 0:
        cv2.fillPoly(overlay, [contour], color)
    draw_queue.append((bbox, contour, color, thickness, label, label_color))


def render_queued_shapes(img, draw_queue):
    """Draw outlines, bounding boxes, and labels for all queued shapes directly onto img."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    for bbox, contour, color, thickness, label, label_color in draw_queue:
        if contour is not None and len(contour) > 0:
            cv2.polylines(img, [contour], True, color, thickness)

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if label:
            label_c = label_color if label_color else color
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_y = y1 - text_height - 10 if y1 - text_height - 10 >= 0 else y2 + 2
            text_y = y1 - 5 if y1 - text_height - 10 >= 0 else y2 + text_height + 4
            cv2.rectangle(img, (x1, label_y), (x1 + text_width + 4, label_y + text_height + 10), label_c, -1)
            cv2.putText(img, label, (x1 + 2, text_y), font, font_scale, (0, 0, 0), font_thickness)


def visualize_image(image_path, image_data, output_path, iou_thresh=0.5, bbox_iou_thresh=0.5,
                     has_iou_bb=False, has_contours=False):
    """
    Visualize all detections for a single image. Draws filled masks (semi-transparent)
    plus box outlines when contour_1/contour_2 are available, otherwise boxes only.

    Colors (GT vs. predicted, not just TP/FP/FN):
    - Green: GT (ground truth) - drawn for the TP case, alongside its blue match
    - Blue: predicted TP (correctly matched to a GT)
    - Yellow: predicted FP (unmatched, no GT at all)
    - Red: GT FN (unmatched, no prediction at all - the missed object itself)
    - Orange: matched pair with IoU below threshold (drawn on both GT and prediction)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Warning: Could not load image {image_path}")
        return False, {}

    # Color definitions (BGR)
    COLOR_GT = (0, 255, 0)          # Green - GT
    COLOR_PRED_TP = (255, 100, 0)   # Blue - TP predictions
    COLOR_FP = (0, 255, 255)        # Yellow - FP (unmatched)
    COLOR_FN = (0, 0, 255)          # Red - FN (unmatched)
    COLOR_LOW_IOU = (0, 165, 255)   # Orange - Low IoU matches

    stats = {
        'TP': 0,
        'FP_unmatched': 0,
        'FN_unmatched': 0,
        'FP_low_iou': 0,
        'FN_low_iou': 0,
        'mask_only_issue': 0,
    }

    overlay = img.copy()
    draw_queue = []

    for _, row in image_data.iterrows():
        result = classify_row_detailed(row, iou_thresh, bbox_iou_thresh, has_iou_bb)

        bbox_gt = parse_bbox(row['bbox_1'])
        bbox_pred = parse_bbox(row['bbox_2'])
        contour_gt = parse_contour(row['contour_1']) if has_contours else None
        contour_pred = parse_contour(row['contour_2']) if has_contours else None
        iou_val = result['iou']
        conf2 = row.get('conf2')
        conf_str = f"{conf2:.2f}" if pd.notna(conf2) else "?"

        if result['type'] == 'TP':
            queue_shape(overlay, draw_queue, bbox_gt, contour_gt, COLOR_GT, thickness=2)
            label = f"TP IoU={iou_val:.2f} conf={conf_str}" if iou_val is not None else f"TP conf={conf_str}"
            queue_shape(overlay, draw_queue, bbox_pred, contour_pred, COLOR_PRED_TP, thickness=2, label=label)
            stats['TP'] += 1

        elif result['type'] == 'FP_UNMATCHED':
            queue_shape(overlay, draw_queue, bbox_pred, contour_pred, COLOR_FP, thickness=3,
                        label=f"FP (no GT) conf={conf_str}", label_color=COLOR_FP)
            stats['FP_unmatched'] += 1

        elif result['type'] == 'FN_UNMATCHED':
            queue_shape(overlay, draw_queue, bbox_gt, contour_gt, COLOR_FN, thickness=3,
                        label="FN (missed)", label_color=COLOR_FN)
            stats['FN_unmatched'] += 1

        elif result['type'] == 'BAD_MATCH':
            iou_str = f"{iou_val:.2f}" if iou_val is not None else "?"
            iou_bb_str = f"{result['iou_bb']:.2f}" if result['iou_bb'] is not None else None
            issue_tag = "mask-only" if result['mask_only_issue'] else "localization"

            gt_label = f"FN IoU={iou_str}" + (f" bbIoU={iou_bb_str}" if iou_bb_str else "") + f" [{issue_tag}]"
            pred_label = f"FP IoU={iou_str}" + (f" bbIoU={iou_bb_str}" if iou_bb_str else "") + f" conf={conf_str}"

            queue_shape(overlay, draw_queue, bbox_gt, contour_gt, COLOR_FN, thickness=2, label=gt_label, label_color=COLOR_LOW_IOU)
            queue_shape(overlay, draw_queue, bbox_pred, contour_pred, COLOR_LOW_IOU, thickness=2, label=pred_label, label_color=COLOR_LOW_IOU)
            stats['FP_low_iou'] += 1
            stats['FN_low_iou'] += 1
            if result['mask_only_issue']:
                stats['mask_only_issue'] += 1

    # Single full-image blend for ALL queued mask fills, then draw crisp outlines/boxes/labels on top
    cv2.addWeighted(overlay, MASK_ALPHA, img, 1 - MASK_ALPHA, 0, img)
    render_queued_shapes(img, draw_queue)

    total_fp = stats['FP_unmatched'] + stats['FP_low_iou']
    total_fn = stats['FN_unmatched'] + stats['FN_low_iou']

    summary1 = f"TP:{stats['TP']} | FP:{total_fp} (unmatched:{stats['FP_unmatched']}, lowIoU:{stats['FP_low_iou']})"
    summary2 = f"FN:{total_fn} (missed:{stats['FN_unmatched']}, lowIoU:{stats['FN_low_iou']})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (10, 10), (700, 80), (0, 0, 0), -1)
    cv2.putText(img, summary1, (15, 35), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, summary2, (15, 65), font, 0.8, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    return True, stats


def compute_image_stats(df, iou_thresh, bbox_iou_thresh, has_iou_bb):
    """
    Compute detailed per-image statistics using vectorized pandas/numpy ops.

    This is deliberately NOT a per-row Python loop: for low-confidence-threshold
    CSVs (e.g. score=0.005) a single file can have 800k+ rows, and this stats
    pass runs over ALL of them just to pick a handful of images to visualize -
    a row-wise iterrows()/apply() loop there dominates runtime by orders of
    magnitude. Classification is boolean-masked instead; only the (~dozen)
    images actually selected for output go through row-wise contour/box drawing.
    """
    idx_1 = df['idx_1'].to_numpy()
    idx_2 = df['idx_2'].to_numpy()
    iou = df['IoU'].fillna(0.0).to_numpy()

    matched = (idx_1 != -1) & (idx_2 != -1)
    fn_unmatched = (idx_1 != -1) & (idx_2 == -1)
    fp_unmatched = (idx_1 == -1) & (idx_2 != -1)
    tp = matched & (iou >= iou_thresh)
    bad_match = matched & (iou < iou_thresh)

    if has_iou_bb:
        iou_bb = df['IoU_bb'].to_numpy()
        mask_only_issue = bad_match & (np.nan_to_num(iou_bb, nan=-1.0) >= bbox_iou_thresh)
    else:
        mask_only_issue = np.zeros(len(df), dtype=bool)

    per_row = pd.DataFrame({
        'image': df['image'].to_numpy(),
        'dataset': df['dataset'].to_numpy(),
        'n_tp': tp,
        'n_fp_unmatched': fp_unmatched,
        'n_fn_unmatched': fn_unmatched,
        'n_fp_low_iou': bad_match,
        'n_fn_low_iou': bad_match,
        'n_mask_only_issue': mask_only_issue,
    })

    stats_df = per_row.groupby(['image', 'dataset'], as_index=False, sort=False).sum()
    stats_df['n_total'] = per_row.groupby('image', sort=False).size().reindex(stats_df['image']).to_numpy()

    stats_df['n_fp'] = stats_df['n_fp_unmatched'] + stats_df['n_fp_low_iou']
    stats_df['n_fn'] = stats_df['n_fn_unmatched'] + stats_df['n_fn_low_iou']
    stats_df['n_errors'] = stats_df['n_fp'] + stats_df['n_fn']
    stats_df['n_detection_errors'] = stats_df['n_fp_unmatched'] + stats_df['n_fn_unmatched']
    stats_df['n_segmentation_errors'] = stats_df['n_fp_low_iou'] + stats_df['n_fn_low_iou']

    return stats_df


def main():
    parser = argparse.ArgumentParser(description='Visualize failure examples from detection results')
    parser.add_argument('--csv', type=str,
                        default='<data-dir>/sam3_results_score_005_combined_results_with_bb_experiment_1.csv',
                        help='Path to combined results CSV file')
    parser.add_argument('--dataset-dir', type=str,
                        default='flatbug-dataset',
                        help='Path to dataset directory containing images')
    parser.add_argument('--output-dir', type=str,
                        default='src/failure_examples_sam3',
                        help='Output directory for visualization images')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='Mask IoU threshold for TP/FP/FN classification')
    parser.add_argument('--bbox-iou-thresh', type=float, default=0.5,
                        help='Bbox IoU (IoU_bb) threshold used to flag mask-only quality issues')
    parser.add_argument('--size-thresh', type=float, default=32,
                        help='Minimum sqrt(area) to include (matches R script filter)')
    parser.add_argument('--min-conf', type=float, default=0.0,
                        help='Display confidence filter: predictions with conf2 below this are hidden. '
                             'A matched pair whose prediction falls below this is demoted to a plain '
                             'missed-GT (FN) - the GT still shows, the weak prediction does not. A pure '
                             'FP below this is dropped entirely. Does not touch the source CSV/AP-curve '
                             'numbers elsewhere - this only reshapes what THIS script selects and draws, '
                             'so low-score-threshold eval runs (e.g. score=0.005) do not render as an '
                             'unreadable wall of near-zero-confidence boxes.')
    parser.add_argument('--images-per-dataset', type=int, default=4,
                        help='Number of images to select per dataset')
    parser.add_argument('--model-name', type=str, default='SAM3',
                        help='Model name for labeling')

    args = parser.parse_args()

    print(f"=" * 70)
    print(f"Failure Examples Visualization")
    print(f"=" * 70)
    print(f"CSV: {args.csv}")
    print(f"IoU threshold: {args.iou_thresh}")
    print(f"Bbox IoU threshold: {args.bbox_iou_thresh}")
    print(f"Size threshold: {args.size_thresh}")
    print(f"Output: {args.output_dir}")
    print(f"=" * 70)

    print(f"\nLoading {args.csv}...")
    df = pd.read_csv(args.csv, sep=';')
    print(f"  Loaded {len(df)} rows")
    has_iou_bb = 'IoU_bb' in df.columns
    has_contours = 'contour_1' in df.columns and 'contour_2' in df.columns
    print(f"  IoU_bb column present: {has_iou_bb}")
    print(f"  contour_1/contour_2 columns present (real masks): {has_contours}")

    # Apply size filter (area comes from GT contourArea if matched/missed-GT, else prediction contourArea)
    df['area'] = np.where(df['idx_1'].to_numpy() != -1, df['contourArea_1'], df['contourArea_2'])
    df['size'] = df['area'] ** 0.5
    df = df[df['size'] >= args.size_thresh].copy()
    print(f"  After size >= {args.size_thresh} filter: {len(df)} rows")

    if args.min_conf > 0:
        conf2 = df['conf2'].fillna(0.0).to_numpy()
        has_pred = df['idx_2'].to_numpy() != -1
        low_conf = has_pred & (conf2 < args.min_conf)
        drop_fp = low_conf & (df['idx_1'].to_numpy() == -1)
        demote = low_conf & (df['idx_1'].to_numpy() != -1)

        demote_aligned = demote[~drop_fp]
        df = df[~drop_fp].copy()
        df.loc[demote_aligned, 'idx_2'] = -1
        df.loc[demote_aligned, 'bbox_2'] = np.nan
        df.loc[demote_aligned, 'IoU'] = 0.0
        if has_iou_bb:
            df.loc[demote_aligned, 'IoU_bb'] = np.nan
        if has_contours:
            df.loc[demote_aligned, 'contour_2'] = np.nan

        print(f"  Display confidence filter (conf2 >= {args.min_conf}): "
              f"dropped {int(drop_fp.sum())} weak FP rows, demoted {int(demote.sum())} weak matches to FN")

    df['dataset'] = df['image'].str.split('_', n=1).str[0]

    stats_df = compute_image_stats(df, args.iou_thresh, args.bbox_iou_thresh, has_iou_bb)
    print(f"\nFound {len(stats_df)} unique images across {stats_df['dataset'].nunique()} datasets")

    print(f"\n{'='*70}")
    print("GLOBAL STATISTICS")
    print(f"{'='*70}")
    total_tp = stats_df['n_tp'].sum()
    total_fp_unmatched = stats_df['n_fp_unmatched'].sum()
    total_fn_unmatched = stats_df['n_fn_unmatched'].sum()
    total_fp_low_iou = stats_df['n_fp_low_iou'].sum()
    total_fn_low_iou = stats_df['n_fn_low_iou'].sum()
    total_mask_only_issue = stats_df['n_mask_only_issue'].sum()

    total_fp = total_fp_unmatched + total_fp_low_iou
    total_fn = total_fn_unmatched + total_fn_low_iou

    print(f"True Positives (TP):           {total_tp:6d}")
    print(f"False Positives (FP):          {total_fp:6d}")
    if total_fp > 0:
        print(f"  - Unmatched (no GT):         {total_fp_unmatched:6d} ({100*total_fp_unmatched/total_fp:.1f}%)")
        print(f"  - Low IoU (matched):         {total_fp_low_iou:6d} ({100*total_fp_low_iou/total_fp:.1f}%)")
    print(f"False Negatives (FN):          {total_fn:6d}")
    if total_fn > 0:
        print(f"  - Unmatched (missed):        {total_fn_unmatched:6d} ({100*total_fn_unmatched/total_fn:.1f}%)")
        print(f"  - Low IoU (matched):         {total_fn_low_iou:6d} ({100*total_fn_low_iou/total_fn:.1f}%)")

    if total_fp > 0 and total_fn > 0:
        print(f"\n** {100*total_fp_low_iou/total_fp:.1f}% of FP and "
              f"{100*total_fn_low_iou/total_fn:.1f}% of FN are due to low IoU (detected but poor mask quality) **")
    if has_iou_bb and total_fp_low_iou > 0:
        print(f"** Of the low-IoU matches, {total_mask_only_issue} ({100*total_mask_only_issue/total_fp_low_iou:.1f}%) "
              f"have IoU_bb >= {args.bbox_iou_thresh} (correctly localized, mask-only quality issue) **")

    # Select images per dataset - three categories
    selected_images = []

    for dataset, group in stats_df.groupby('dataset'):
        n_per_cat = max(1, args.images_per_dataset // 3)

        worst_total = group.nlargest(n_per_cat, 'n_errors')
        for _, row in worst_total.iterrows():
            selected_images.append({**row.to_dict(), 'category': 'worst_total'})

        worst_detection = group.nlargest(n_per_cat, 'n_detection_errors')
        for _, row in worst_detection.iterrows():
            if row['image'] not in [s['image'] for s in selected_images]:
                selected_images.append({**row.to_dict(), 'category': 'worst_detection'})

        best = group.nsmallest(n_per_cat, 'n_errors')
        for _, row in best.iterrows():
            if row['image'] not in [s['image'] for s in selected_images]:
                selected_images.append({**row.to_dict(), 'category': 'best'})

    print(f"\nSelected {len(selected_images)} images for visualization")

    dataset_dirs = [args.dataset_dir]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for item in selected_images:
        image_name = item['image']
        dataset = item['dataset']
        category = item['category']

        image_path = find_image_path(image_name, dataset_dirs)
        if image_path is None:
            print(f"  Warning: Could not find image for {image_name}")
            continue

        image_data = df[df['image'] == image_name]

        if category == 'worst_total':
            output_filename = f"{dataset}_worst_{item['n_errors']}err_{image_name}.jpg"
        elif category == 'worst_detection':
            output_filename = f"{dataset}_missed_{item['n_detection_errors']}det_{image_name}.jpg"
        else:
            output_filename = f"{dataset}_best_{item['n_errors']}err_{image_name}.jpg"

        output_path = output_dir / dataset / output_filename

        print(f"Processing {image_name} ({category})...")

        success, _ = visualize_image(image_path, image_data, str(output_path),
                                      args.iou_thresh, args.bbox_iou_thresh, has_iou_bb, has_contours)
        if success:
            success_count += 1

    print(f"\n{'='*70}")
    print(f"Done! Generated {success_count} visualization images in {output_dir}")
    print(f"{'='*70}")

    print("\nSummary by dataset:")
    print("-" * 90)
    print(f"{'Dataset':25s} {'TP':>6s} {'FP':>6s} {'FP_det':>7s} {'FP_seg':>7s} {'FN':>6s} {'FN_det':>7s} {'FN_seg':>7s} {'MaskOnly':>8s}")
    print("-" * 90)
    for dataset, group in stats_df.groupby('dataset'):
        print(f"{dataset:25s} {group['n_tp'].sum():6d} {group['n_fp'].sum():6d} "
              f"{group['n_fp_unmatched'].sum():7d} {group['n_fp_low_iou'].sum():7d} "
              f"{group['n_fn'].sum():6d} {group['n_fn_unmatched'].sum():7d} {group['n_fn_low_iou'].sum():7d} "
              f"{group['n_mask_only_issue'].sum():8d}")


if __name__ == '__main__':
    main()
