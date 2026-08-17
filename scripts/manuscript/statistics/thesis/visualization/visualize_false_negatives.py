#!/usr/bin/env python3
"""
Visualize False Negatives (Missed Detections) for SAM3.

Identifies both Unmatched FNs (completely missed GT) and Low-IoU FNs
(detected but poor overlap), draws annotated full-image copies, and saves
them sorted by dataset for targeted review.

USAGE:
    python3 visualize_false_negatives.py \
        --csv <data-dir>/combined_results_sam3_ft_mask_05_checkpoint18_score_001_greedy_with_bb.csv \
        --images-dir fb_yolo/insects/images/val \
        --output-dir src/false_negatives_vis \
        --iou-thresh 0.5

    # Or on remote server:
    python3 visualize_false_negatives.py \
        --csv data/combined_results.csv \
        --images-dir /data/sam3/fb_yolo/insects/images/val \
        --output-dir false_negatives_vis

    # With multiprocessing:
    python3 visualize_false_negatives.py \
        --csv ... --workers 8

LEGEND:
    GREEN box  =  Ground Truth (missed insect)
    RED box    =  Model's failed prediction (Low-IoU only)
    Label shows: FN type, IoU (if matched), confidence (if matched)
"""

import argparse
import ast
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

# ── data location ────────────────────────────────────────────────────────────
# Matched-detection CSVs ship next to the Experiment 1-3 scripts (standalone) or
# in the evaluation working tree's data directory (embedded). Override with
# --csv, or FB_DATA_DIR for the directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_THESIS_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
_STATS_ROOT = os.path.abspath(os.path.join(_THESIS_ROOT, os.pardir))
_DATA_DIR = os.environ.get(
    "FB_DATA_DIR",
    os.path.join(_STATS_ROOT, "data", "thesis_experiments_using_fb_eval_refactored", "with_bb")
    if os.path.isdir(os.path.join(_STATS_ROOT, "helpers"))
    else os.path.join(_THESIS_ROOT, "experiments_1_3", "data"))

DEFAULT_CSV = os.path.join(_DATA_DIR, "sam3_ft_round4_ep18_predictions_mask_05_score_005_combined_results_with_bb_experiment_2.csv")
DEFAULT_IMAGES_DIR = "fb_yolo/insects/images/val"
DEFAULT_OUTPUT_DIR = "src/false_negatives_vis"
DEFAULT_IOU_THRESH = 0.5

# Colors (BGR)
COLOR_GT = (0, 200, 0)             # GREEN - GT box
COLOR_PRED_FAIL = (0, 0, 255)      # RED - failed prediction
COLOR_BANNER_BG = (0, 0, 0)        # Black
COLOR_BANNER_TEXT = (255, 255, 255) # White
COLOR_LABEL_UNMATCHED = (0, 200, 0) # Green label bg
COLOR_LABEL_LOW_IOU = (0, 165, 255) # Orange label bg


# ============================================================================
# HELPERS
# ============================================================================

def parse_bbox(bbox_str):
    """Parse bbox string like '[x1, y1, x2, y2]' to list of floats."""
    if pd.isna(bbox_str) or str(bbox_str).strip() in ('', 'nan', 'None'):
        return None
    try:
        return [float(x) for x in ast.literal_eval(str(bbox_str))]
    except Exception:
        return None


def extract_dataset_name(image_name):
    """Extract dataset name from image filename (first token before '_')."""
    return image_name.split('_')[0]


def find_image_path(image_name, images_dir):
    """Find the image file, trying common extensions."""
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        path = os.path.join(images_dir, f"{image_name}{ext}")
        if os.path.exists(path):
            return path
    return None


def classify_fn(row, iou_thresh):
    """
    Classify a row as a False Negative.
    Returns:
        'FN_UNMATCHED' - GT with no prediction at all
        'FN_LOW_IOU'   - GT matched to prediction but IoU < threshold
        None           - Not a FN
    """
    idx_1 = row['idx_1']
    idx_2 = row['idx_2']
    iou = row['IoU'] if pd.notna(row.get('IoU', np.nan)) else 0.0

    if idx_1 != -1 and idx_2 == -1:
        return 'FN_UNMATCHED'
    elif idx_1 != -1 and idx_2 != -1 and iou < iou_thresh:
        return 'FN_LOW_IOU'
    return None


def draw_label(img, text, x, y, bg_color, text_color=(0, 0, 0)):
    """Draw a text label with background at (x, y)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    h, w = img.shape[:2]
    # Keep label inside image bounds
    lx = max(0, min(x, w - tw - 6))
    ly = max(th + 10, min(y, h - 2))

    cv2.rectangle(img, (lx, ly - th - 8), (lx + tw + 6, ly + 2), bg_color, -1)
    cv2.putText(img, text, (lx + 3, ly - 3), font, scale, text_color, thickness)


def draw_banner(img, text):
    """Draw a summary banner at the top of the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    h, w = img.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    banner_h = th + baseline + 16
    cv2.rectangle(img, (0, 0), (w, banner_h), COLOR_BANNER_BG, -1)
    cv2.putText(img, text, (8, th + 8), font, scale, COLOR_BANNER_TEXT, thickness)


# ============================================================================
# PER-IMAGE PROCESSING
# ============================================================================

def process_image(args):
    """
    Process all FNs for a single image: draw GT (green) and pred (red) boxes,
    save one annotated full-image copy per image.
    Returns (image_name, n_fn_unmatched, n_fn_low_iou, success).
    """
    image_name, fn_rows, images_dir, output_dir, iou_thresh = args

    dataset = extract_dataset_name(image_name)
    image_path = find_image_path(image_name, images_dir)

    if image_path is None:
        return (image_name, 0, 0, False, f"Image not found: {image_name}")

    img = cv2.imread(image_path)
    if img is None:
        return (image_name, 0, 0, False, f"Failed to read: {image_path}")

    annotated = img.copy()

    n_unmatched = 0
    n_low_iou = 0

    for _, row in fn_rows.iterrows():
        fn_type = classify_fn(row, iou_thresh)
        if fn_type is None:
            continue

        bbox_gt = parse_bbox(row['bbox_1'])
        bbox_pred = parse_bbox(row.get('bbox_2', None))
        iou = row['IoU'] if pd.notna(row.get('IoU', np.nan)) else 0.0
        conf = row['conf2'] if pd.notna(row.get('conf2', np.nan)) else 0.0

        if fn_type == 'FN_UNMATCHED':
            n_unmatched += 1
            if bbox_gt is not None:
                gx1, gy1, gx2, gy2 = [int(v) for v in bbox_gt]
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), COLOR_GT, 3)
                draw_label(annotated, "FN (missed)", gx1, gy1 - 2, COLOR_LABEL_UNMATCHED,
                           (255, 255, 255))

        elif fn_type == 'FN_LOW_IOU':
            n_low_iou += 1
            if bbox_gt is not None:
                gx1, gy1, gx2, gy2 = [int(v) for v in bbox_gt]
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), COLOR_GT, 3)
                label_gt = f"GT IoU={iou:.2f}"
                draw_label(annotated, label_gt, gx1, gy1 - 2, COLOR_LABEL_LOW_IOU,
                           (0, 0, 0))

            if bbox_pred is not None:
                px1, py1, px2, py2 = [int(v) for v in bbox_pred]
                cv2.rectangle(annotated, (px1, py1), (px2, py2), COLOR_PRED_FAIL, 3)
                label_pred = f"Pred c={conf:.3f}"
                draw_label(annotated, label_pred, px1, py2 + 2, (0, 0, 200),
                           (255, 255, 255))

    # Banner
    total = n_unmatched + n_low_iou
    banner = f"{dataset} | FN: {total} (missed: {n_unmatched}, low-IoU: {n_low_iou})"
    draw_banner(annotated, banner)

    # Save to dataset subdirectory
    ds_dir = os.path.join(output_dir, dataset)
    os.makedirs(ds_dir, exist_ok=True)

    filename = f"fn_{total}total_{n_unmatched}miss_{n_low_iou}low_{image_name}.jpg"
    out_path = os.path.join(ds_dir, filename)
    cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return (image_name, n_unmatched, n_low_iou, True, out_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize False Negatives (missed detections and low-IoU matches)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV,
                        help='Path to combined results CSV')
    parser.add_argument('--images-dir', type=str, default=DEFAULT_IMAGES_DIR,
                        help='Path to validation images directory')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for annotated images')
    parser.add_argument('--iou-thresh', type=float, default=DEFAULT_IOU_THRESH,
                        help='IoU threshold below which a match is considered FN (default: 0.5)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Parallel workers (0 = auto, 1 = sequential)')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Max images to process (0 = all)')

    args = parser.parse_args()

    print("=" * 80)
    print("FALSE NEGATIVE VISUALIZER")
    print("=" * 80)
    print(f"CSV:         {args.csv}")
    print(f"Images:      {args.images_dir}")
    print(f"Output:      {args.output_dir}")
    print(f"IoU thresh:  {args.iou_thresh}")
    print()

    # ---- Load data ----
    print("Loading CSV...")
    with open(args.csv, 'r') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else ','
    df = pd.read_csv(args.csv, sep=sep)
    print(f"  Total rows: {len(df)}")

    # ---- Identify all FN rows ----
    df['fn_type'] = df.apply(lambda r: classify_fn(r, args.iou_thresh), axis=1)
    fn_df = df[df['fn_type'].notna()].copy()
    print(f"\n  Total False Negatives: {len(fn_df)}")

    n_unmatched_total = (fn_df['fn_type'] == 'FN_UNMATCHED').sum()
    n_low_iou_total = (fn_df['fn_type'] == 'FN_LOW_IOU').sum()
    print(f"    Unmatched (completely missed): {n_unmatched_total}")
    print(f"    Low-IoU (poor overlap):        {n_low_iou_total}")

    # Dataset breakdown
    fn_df['dataset'] = fn_df['image'].apply(extract_dataset_name)
    print(f"\n  FNs by dataset:")
    ds_summary = fn_df.groupby('dataset')['fn_type'].value_counts().unstack(fill_value=0)
    for ds in sorted(ds_summary.index):
        row = ds_summary.loc[ds]
        unmatch = row.get('FN_UNMATCHED', 0)
        low_iou = row.get('FN_LOW_IOU', 0)
        total = unmatch + low_iou
        print(f"    {ds:30s} total={total:5d}  missed={unmatch:5d}  low_iou={low_iou:5d}")

    # ---- Group by image ----
    images_with_fn = fn_df.groupby('image')
    n_images = len(images_with_fn)
    print(f"\n  Images with at least 1 FN: {n_images}")

    if args.max_images > 0:
        # Sort by most FNs first, take top max_images
        fn_counts = fn_df.groupby('image').size().sort_values(ascending=False)
        top_images = fn_counts.head(args.max_images).index
        fn_df = fn_df[fn_df['image'].isin(top_images)]
        images_with_fn = fn_df.groupby('image')
        n_images = len(images_with_fn)
        print(f"  Limited to top {n_images} images (most FNs)")

    # ---- Build task list ----
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = []
    for image_name, group in images_with_fn:
        tasks.append((image_name, group, args.images_dir, args.output_dir, args.iou_thresh))

    # ---- Process ----
    n_workers = args.workers if args.workers > 0 else min(cpu_count(), 8)
    print(f"\nProcessing {len(tasks)} images...")

    success = 0
    failed = 0
    total_unmatched = 0
    total_low_iou = 0
    ds_stats = defaultdict(lambda: {'images': 0, 'unmatched': 0, 'low_iou': 0})

    if n_workers > 1 and len(tasks) > 5:
        print(f"  Using {n_workers} workers...")
        with Pool(n_workers) as pool:
            results = pool.map(process_image, tasks)
    else:
        results = [process_image(t) for t in tasks]

    for res in results:
        image_name, n_un, n_li, ok, msg = res
        ds = extract_dataset_name(image_name)
        if ok:
            success += 1
            total_unmatched += n_un
            total_low_iou += n_li
            ds_stats[ds]['images'] += 1
            ds_stats[ds]['unmatched'] += n_un
            ds_stats[ds]['low_iou'] += n_li
        else:
            failed += 1
            if failed <= 10:
                print(f"  WARN: {msg}")

    # ---- Summary ----
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"  Images processed: {success}")
    print(f"  Images failed:    {failed}")
    print(f"  FN visualized:    {total_unmatched + total_low_iou}")
    print(f"    Unmatched:      {total_unmatched}")
    print(f"    Low-IoU:        {total_low_iou}")
    print(f"  Output:           {args.output_dir}")

    print(f"\n  Per-dataset summary:")
    print(f"  {'Dataset':30s} {'Images':>8s} {'Missed':>8s} {'Low-IoU':>8s} {'Total':>8s}")
    print(f"  {'-'*70}")
    for ds in sorted(ds_stats.keys()):
        s = ds_stats[ds]
        total = s['unmatched'] + s['low_iou']
        print(f"  {ds:30s} {s['images']:8d} {s['unmatched']:8d} {s['low_iou']:8d} {total:8d}")


if __name__ == '__main__':
    main()
