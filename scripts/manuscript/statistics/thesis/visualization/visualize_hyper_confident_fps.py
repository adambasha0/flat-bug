#!/usr/bin/env python3
"""
Visualize Hyper-Confident False Positives (Hallucinations) for SAM3.

Identifies the top-N most confident FPs (unmatched predictions with no GT),
crops contextual patches around each one, and saves annotated crops for
visual inspection of what artifacts are tricking the model.

USAGE:
    python3 visualize_hyper_confident_fps.py \
        --csv <data-dir>/combined_results_sam3_ft_mask_05_checkpoint18_score_001_greedy_with_bb.csv \
        --images-dir fb_yolo/insects/images/val \
        --output-dir src/hyper_confident_fps \
        --top-n 200 \
        --crop-size 512

    # Or on remote server:
    python3 visualize_hyper_confident_fps.py \
        --csv data/combined_results_sam3_ft_mask_05_checkpoint18_score_001_greedy_with_bb.csv \
        --images-dir /data/sam3/fb_yolo/insects/images/val \
        --output-dir hyper_confident_fps \
        --top-n 200
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
DEFAULT_OUTPUT_DIR = "src/hyper_confident_fps"
DEFAULT_TOP_N = 200
DEFAULT_CROP_SIZE = 512

# Colors (BGR)
COLOR_FP_BOX = (0, 0, 255)        # RED for hallucinated bbox
COLOR_BANNER_BG = (0, 0, 0)       # Black background
COLOR_BANNER_TEXT = (255, 255, 255) # White text


# ============================================================================
# HELPERS
# ============================================================================

def parse_bbox(bbox_str):
    """Parse bbox string like '[x1, y1, x2, y2]' to list of floats."""
    if pd.isna(bbox_str) or bbox_str == '':
        return None
    try:
        return [float(x) for x in ast.literal_eval(bbox_str)]
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


def crop_context_around_bbox(img, bbox, crop_size):
    """
    Crop a context patch centered on bbox.
    Returns (crop, offset_x, offset_y) where offsets map bbox coords to crop coords.
    Handles edge cases (bbox near image border) by clamping.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # Center of bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Ensure crop is large enough to contain the full bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    actual_crop_w = max(crop_size, int(bbox_w * 1.5))
    actual_crop_h = max(crop_size, int(bbox_h * 1.5))

    # Top-left of crop
    crop_x1 = int(cx - actual_crop_w / 2)
    crop_y1 = int(cy - actual_crop_h / 2)

    # Clamp to image bounds
    crop_x1 = max(0, min(crop_x1, w - actual_crop_w))
    crop_y1 = max(0, min(crop_y1, h - actual_crop_h))
    crop_x2 = min(w, crop_x1 + actual_crop_w)
    crop_y2 = min(h, crop_y1 + actual_crop_h)

    crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    return crop, crop_x1, crop_y1


def draw_fp_crop(img, bbox, rank, conf, dataset, crop_size):
    """
    Crop around bbox, draw red box and info banner.
    Returns annotated crop or None on failure.
    """
    crop, off_x, off_y = crop_context_around_bbox(img, bbox, crop_size)
    if crop is None or crop.size == 0:
        return None

    x1, y1, x2, y2 = bbox
    # Remap bbox to crop coordinates
    bx1 = int(x1 - off_x)
    by1 = int(y1 - off_y)
    bx2 = int(x2 - off_x)
    by2 = int(y2 - off_y)

    # Clamp to crop bounds
    ch, cw = crop.shape[:2]
    bx1 = max(0, min(bx1, cw - 1))
    by1 = max(0, min(by1, ch - 1))
    bx2 = max(0, min(bx2, cw - 1))
    by2 = max(0, min(by2, ch - 1))

    # Draw RED bounding box (thick)
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), COLOR_FP_BOX, 3)

    # Draw info banner at top
    banner_text = f"Rank #{rank} FP | Conf: {conf:.4f} | {dataset}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(banner_text, font, font_scale, thickness)

    banner_h = th + baseline + 16
    cv2.rectangle(crop, (0, 0), (cw, banner_h), COLOR_BANNER_BG, -1)
    cv2.putText(crop, banner_text, (8, th + 8), font, font_scale, COLOR_BANNER_TEXT, thickness)

    return crop


def process_single_fp(args):
    """Worker function for multiprocessing. Returns (success, output_path) or (False, error)."""
    row, rank, images_dir, output_dir, crop_size = args

    image_name = row['image']
    conf = row['conf2']
    bbox = parse_bbox(row['bbox_2'])
    dataset = extract_dataset_name(image_name)

    if bbox is None:
        return (False, f"No bbox for rank {rank}")

    image_path = find_image_path(image_name, images_dir)
    if image_path is None:
        return (False, f"Image not found: {image_name}")

    img = cv2.imread(image_path)
    if img is None:
        return (False, f"Failed to read: {image_path}")

    crop = draw_fp_crop(img, bbox, rank, conf, dataset, crop_size)
    if crop is None:
        return (False, f"Failed to crop for rank {rank}")

    # Save to dataset subdirectory
    ds_dir = os.path.join(output_dir, dataset)
    os.makedirs(ds_dir, exist_ok=True)

    filename = f"rank{rank:04d}_conf{conf:.4f}_{image_name}.jpg"
    output_path = os.path.join(ds_dir, filename)
    cv2.imwrite(output_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return (True, output_path)


# ============================================================================
# FULL-IMAGE VERSION (TASK 1 BONUS)
# ============================================================================

def process_full_image_fps(image_name, fps_for_image, images_dir, output_dir, crop_size):
    """
    For a given image, produce both:
    - Individual crops per FP
    - One full-image overview with ALL FP boxes drawn
    Returns (n_crops_ok, full_image_ok)
    """
    image_path = find_image_path(image_name, images_dir)
    if image_path is None:
        return 0, False

    img = cv2.imread(image_path)
    if img is None:
        return 0, False

    dataset = extract_dataset_name(image_name)
    ds_dir = os.path.join(output_dir, dataset)
    os.makedirs(ds_dir, exist_ok=True)

    # Full-image overview with all FP boxes
    overview = img.copy()
    for _, row in fps_for_image.iterrows():
        bbox = parse_bbox(row['bbox_2'])
        conf = row['conf2']
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(overview, (x1, y1), (x2, y2), COLOR_FP_BOX, 2)
        label = f"{conf:.3f}"
        cv2.putText(overview, label, (x1, max(y1 - 5, 12)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_FP_BOX, 1)

    n_fps = len(fps_for_image)
    overview_path = os.path.join(ds_dir, f"overview_{n_fps}fps_{image_name}.jpg")
    cv2.imwrite(overview_path, overview, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return n_fps, True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize top-N hyper-confident False Positives (hallucinations)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV,
                        help='Path to combined results CSV')
    parser.add_argument('--images-dir', type=str, default=DEFAULT_IMAGES_DIR,
                        help='Path to validation images directory')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for annotated crops')
    parser.add_argument('--top-n', type=int, default=DEFAULT_TOP_N,
                        help='Number of top FPs to visualize (default: 200)')
    parser.add_argument('--crop-size', type=int, default=DEFAULT_CROP_SIZE,
                        help='Context crop size in pixels (default: 512)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers (0 = auto)')
    parser.add_argument('--full-image', action='store_true',
                        help='Also save full-image overviews with all FP boxes')

    args = parser.parse_args()

    print("=" * 80)
    print("HYPER-CONFIDENT FALSE POSITIVE VISUALIZER")
    print("=" * 80)
    print(f"CSV:        {args.csv}")
    print(f"Images:     {args.images_dir}")
    print(f"Output:     {args.output_dir}")
    print(f"Top-N:      {args.top_n}")
    print(f"Crop size:  {args.crop_size}")
    print()

    # ---- Load data ----
    print("Loading CSV...")
    # Auto-detect separator
    with open(args.csv, 'r') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else ','
    df = pd.read_csv(args.csv, sep=sep)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # ---- Filter to unmatched FPs only ----
    fp = df[(df['idx_1'] == -1) & (df['idx_2'] != -1)].copy()
    print(f"\n  Unmatched FPs (hallucinations): {len(fp)}")

    # Filter out anomalous conf2 values (parsing artifacts > 1.0)
    n_anomalous = (fp['conf2'] > 1.0).sum()
    if n_anomalous > 0:
        print(f"  Filtering {n_anomalous} rows with anomalous conf2 > 1.0")
        fp = fp[fp['conf2'] <= 1.0].copy()
        print(f"  FPs after filtering: {len(fp)}")

    # ---- Sort by confidence descending, take top N ----
    fp = fp.sort_values('conf2', ascending=False).head(args.top_n).copy()
    fp = fp.reset_index(drop=True)
    print(f"\n  Selected top {len(fp)} FPs by confidence")
    print(f"  Confidence range: {fp['conf2'].iloc[-1]:.6f} — {fp['conf2'].iloc[0]:.6f}")

    # Dataset breakdown
    fp['dataset'] = fp['image'].apply(extract_dataset_name)
    ds_counts = fp['dataset'].value_counts()
    print(f"\n  Breakdown by dataset:")
    for ds, count in ds_counts.items():
        print(f"    {ds}: {count}")

    # ---- Create output directory ----
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Process crops ----
    print(f"\nGenerating {len(fp)} annotated crops...")
    n_workers = args.workers if args.workers > 0 else min(cpu_count(), 8)

    tasks = []
    for rank_idx, (_, row) in enumerate(fp.iterrows(), start=1):
        tasks.append((row.to_dict(), rank_idx, args.images_dir, args.output_dir, args.crop_size))

    success = 0
    failed = 0

    if n_workers > 1 and len(tasks) > 10:
        print(f"  Using {n_workers} workers...")
        with Pool(n_workers) as pool:
            results = pool.map(process_single_fp, tasks)
        for ok, msg in results:
            if ok:
                success += 1
            else:
                failed += 1
                if failed <= 5:
                    print(f"  WARN: {msg}")
    else:
        for task in tasks:
            ok, msg = process_single_fp(task)
            if ok:
                success += 1
            else:
                failed += 1
                if failed <= 5:
                    print(f"  WARN: {msg}")

    # ---- Optional full-image overviews ----
    if args.full_image:
        print(f"\nGenerating full-image overviews...")
        images_with_fps = fp.groupby('image')
        overview_ok = 0
        for image_name, group in images_with_fps:
            _, ok = process_full_image_fps(
                image_name, group, args.images_dir, args.output_dir, args.crop_size)
            if ok:
                overview_ok += 1
        print(f"  Full-image overviews: {overview_ok}")

    # ---- Summary ----
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"  Crops saved:   {success}")
    print(f"  Crops failed:  {failed}")
    print(f"  Output dir:    {args.output_dir}")
    print()
    print("Subdirectories by dataset:")
    for ds in sorted(ds_counts.index):
        ds_path = os.path.join(args.output_dir, ds)
        n_files = len(os.listdir(ds_path)) if os.path.exists(ds_path) else 0
        print(f"  {ds}: {n_files} files")


if __name__ == '__main__':
    main()
