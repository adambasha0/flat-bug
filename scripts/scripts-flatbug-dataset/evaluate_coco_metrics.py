#!/usr/bin/env python3
"""
COCO-style evaluation script for flat-bug predictions.

This script computes standard COCO metrics (mAP, AP50, AR) for both bounding boxes
and segmentation masks by comparing predictions against ground truth.

Usage:
    python scripts/scripts-flatbug-dataset/evaluate_coco_metrics.py \
        --predictions-dir ./output \
        --gt-dir ./flatbug-dataset \
        --output ./evaluation_results.csv

Output format:
    DATASET                   | mAP(Box) | AP50(Box) | AR(Box)  || mAP(Seg) | AP50(Seg) | AR(Seg)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Allowed dataset folders to evaluate
ALLOWED_FOLDERS = {
    "NHM-beetles-crops",
    "cao2022",
    "gernat2018",
    "sittinger2023",
    "amarathunga2022",
    "biodiscover-arm",
    "Mothitor",
    "DIRT",
    "Diopsis",
    "AMI-traps",
    "AMT",
    "PeMaToEuroPep",
    "abram2023",
    "anTraX",
    "pinoy2023",
    "sticky-pi",
    "ubc-pitfall-traps",
    #"ALUS",
    "BIOSCAN",
    "DiversityScanner",
    "ArTaxOr",
    "CollembolAI",
    "ubc-scanned-sticky-cards",
}

# Default file names
DEFAULT_GT_FILENAME = "instances_default.json"
DEFAULT_PRED_FILENAME = "flatbug_predictions.json"

# COCO evaluation IoU thresholds
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # [0.50, 0.55, ..., 0.95]

# =============================================================================
# COCO EVALUATION UTILITIES
# =============================================================================

def load_json(filepath: str) -> dict:
    """Load a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def build_image_id_mapping(gt_coco: dict, pred_coco: dict) -> Dict[str, Tuple[int, int]]:
    """
    Build a mapping from file_name to (gt_image_id, pred_image_id).
    
    Since GT and predictions may have different image_ids for the same image,
    we need to match them by file_name.
    """
    gt_filename_to_id = {img['file_name']: img['id'] for img in gt_coco.get('images', [])}
    pred_filename_to_id = {img['file_name']: img['id'] for img in pred_coco.get('images', [])}
    
    # Find common filenames
    common_filenames = set(gt_filename_to_id.keys()) & set(pred_filename_to_id.keys())
    
    mapping = {}
    for filename in common_filenames:
        mapping[filename] = (gt_filename_to_id[filename], pred_filename_to_id[filename])
    
    return mapping


def remap_predictions_to_gt_image_ids(pred_coco: dict, image_id_mapping: Dict[str, Tuple[int, int]]) -> dict:
    """
    Remap prediction image_ids to match GT image_ids.
    """
    # Build pred_id -> gt_id mapping
    pred_to_gt_id = {}
    for filename, (gt_id, pred_id) in image_id_mapping.items():
        pred_to_gt_id[pred_id] = gt_id
    
    # Create new predictions with remapped image_ids
    remapped_pred = {
        'images': [],
        'annotations': [],
        'categories': pred_coco.get('categories', [{'id': 1, 'name': 'insect'}])
    }
    
    # Remap images
    for img in pred_coco.get('images', []):
        if img['id'] in pred_to_gt_id:
            new_img = img.copy()
            new_img['id'] = pred_to_gt_id[img['id']]
            remapped_pred['images'].append(new_img)
    
    # Remap annotations
    for ann in pred_coco.get('annotations', []):
        if ann['image_id'] in pred_to_gt_id:
            new_ann = ann.copy()
            new_ann['image_id'] = pred_to_gt_id[ann['image_id']]
            remapped_pred['annotations'].append(new_ann)
    
    return remapped_pred


def compute_iou_bbox(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two bounding boxes in COCO format [x, y, w, h].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1, y1, x2, y2]
    box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
    box2_xyxy = [x2, y2, x2 + w2, y2 + h2]
    
    # Intersection
    xi1 = max(box1_xyxy[0], box2_xyxy[0])
    yi1 = max(box1_xyxy[1], box2_xyxy[1])
    xi2 = min(box1_xyxy[2], box2_xyxy[2])
    yi2 = min(box1_xyxy[3], box2_xyxy[3])
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def polygon_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """
    Convert polygon segmentation to binary mask.
    """
    import cv2
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for polygon in segmentation:
        if len(polygon) < 6:  # Need at least 3 points
            continue
        pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    
    return mask


def compute_iou_mask(seg1: List[List[float]], seg2: List[List[float]], 
                     height: int, width: int) -> float:
    """
    Compute IoU between two segmentation masks.
    """
    mask1 = polygon_to_mask(seg1, height, width)
    mask2 = polygon_to_mask(seg2, height, width)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using 101-point interpolation (COCO style).
    """
    # Append sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 101-point interpolation
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated_precisions = np.zeros_like(recall_thresholds)
    
    for i, r in enumerate(recall_thresholds):
        # Find precision at recall >= r
        idx = np.where(recalls >= r)[0]
        if len(idx) > 0:
            interpolated_precisions[i] = precisions[idx[0]]
    
    return np.mean(interpolated_precisions)


def evaluate_single_iou(gt_anns: List[dict], pred_anns: List[dict], 
                        iou_threshold: float, eval_type: str = 'bbox',
                        image_sizes: Dict[int, Tuple[int, int]] = None) -> Tuple[float, float]:
    """
    Evaluate at a single IoU threshold.
    
    Returns:
        (precision, recall)
    """
    if len(pred_anns) == 0:
        return 0.0, 0.0
    if len(gt_anns) == 0:
        return 0.0, 1.0  # All predictions are FP, but recall is undefined (set to 1)
    
    # Sort predictions by score (descending)
    pred_anns = sorted(pred_anns, key=lambda x: x.get('score', 1.0), reverse=True)
    
    # Track which GT annotations have been matched
    gt_matched = [False] * len(gt_anns)
    
    tp = 0
    fp = 0
    
    for pred in pred_anns:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_anns):
            if gt_matched[gt_idx]:
                continue
            
            # Must be same image
            if pred['image_id'] != gt['image_id']:
                continue
            
            # Compute IoU
            if eval_type == 'bbox':
                iou = compute_iou_bbox(pred['bbox'], gt['bbox'])
            else:  # segm
                img_id = pred['image_id']
                if image_sizes and img_id in image_sizes:
                    h, w = image_sizes[img_id]
                else:
                    # Estimate from bbox
                    h = int(max(pred['bbox'][1] + pred['bbox'][3], gt['bbox'][1] + gt['bbox'][3]) + 100)
                    w = int(max(pred['bbox'][0] + pred['bbox'][2], gt['bbox'][0] + gt['bbox'][2]) + 100)
                
                pred_seg = pred.get('segmentation', [])
                gt_seg = gt.get('segmentation', [])
                
                if not pred_seg or not gt_seg:
                    iou = 0.0
                else:
                    iou = compute_iou_mask(pred_seg, gt_seg, h, w)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    fn = sum(1 for m in gt_matched if not m)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall


def compute_precision_recall_curve(gt_anns: List[dict], pred_anns: List[dict],
                                   iou_threshold: float, eval_type: str = 'bbox',
                                   image_sizes: Dict[int, Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve at a given IoU threshold.
    """
    if len(pred_anns) == 0 or len(gt_anns) == 0:
        return np.array([0.0]), np.array([0.0])
    
    # Sort predictions by score (descending)
    pred_anns = sorted(pred_anns, key=lambda x: x.get('score', 1.0), reverse=True)
    
    # Precompute IoU matrix per image
    gt_by_image = {}
    for i, gt in enumerate(gt_anns):
        img_id = gt['image_id']
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append((i, gt))
    
    n_gt = len(gt_anns)
    gt_matched = [False] * n_gt
    
    tps = []
    fps = []
    
    for pred in pred_anns:
        img_id = pred['image_id']
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in gt_by_image.get(img_id, []):
            if gt_matched[gt_idx]:
                continue
            
            if eval_type == 'bbox':
                iou = compute_iou_bbox(pred['bbox'], gt['bbox'])
            else:
                if image_sizes and img_id in image_sizes:
                    h, w = image_sizes[img_id]
                else:
                    h = int(max(pred['bbox'][1] + pred['bbox'][3], gt['bbox'][1] + gt['bbox'][3]) + 100)
                    w = int(max(pred['bbox'][0] + pred['bbox'][2], gt['bbox'][0] + gt['bbox'][2]) + 100)
                
                pred_seg = pred.get('segmentation', [])
                gt_seg = gt.get('segmentation', [])
                
                if not pred_seg or not gt_seg:
                    iou = 0.0
                else:
                    iou = compute_iou_mask(pred_seg, gt_seg, h, w)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tps.append(1)
            fps.append(0)
            gt_matched[best_gt_idx] = True
        else:
            tps.append(0)
            fps.append(1)
    
    tps = np.cumsum(tps)
    fps = np.cumsum(fps)
    
    recalls = tps / n_gt
    precisions = tps / (tps + fps)
    
    return recalls, precisions


def evaluate_coco_metrics(gt_coco: dict, pred_coco: dict, 
                          eval_type: str = 'bbox') -> Dict[str, float]:
    """
    Compute COCO-style metrics.
    
    Returns:
        dict with keys: 'mAP', 'AP50', 'AR'
    """
    gt_anns = gt_coco.get('annotations', [])
    pred_anns = pred_coco.get('annotations', [])
    
    # Build image size mapping
    image_sizes = {}
    for img in gt_coco.get('images', []):
        image_sizes[img['id']] = (img.get('height', 2000), img.get('width', 2000))
    
    if len(gt_anns) == 0:
        print(f"  Warning: No GT annotations for {eval_type}")
        return {'mAP': 0.0, 'AP50': 0.0, 'AR': 0.0}
    
    if len(pred_anns) == 0:
        print(f"  Warning: No predictions for {eval_type}")
        return {'mAP': 0.0, 'AP50': 0.0, 'AR': 0.0}
    
    # Compute AP at each IoU threshold
    aps = []
    ap50 = None
    
    for iou_thresh in IOU_THRESHOLDS:
        recalls, precisions = compute_precision_recall_curve(
            gt_anns, pred_anns, iou_thresh, eval_type, image_sizes
        )
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        
        if abs(iou_thresh - 0.5) < 0.01:
            ap50 = ap
    
    mAP = np.mean(aps)
    
    # Compute AR (Average Recall) at IoU=0.5
    _, recall_50 = evaluate_single_iou(gt_anns, pred_anns, 0.5, eval_type, image_sizes)
    
    # Compute AR across IoU thresholds (max recall at each threshold, then average)
    recalls_at_thresholds = []
    for iou_thresh in IOU_THRESHOLDS:
        _, recall = evaluate_single_iou(gt_anns, pred_anns, iou_thresh, eval_type, image_sizes)
        recalls_at_thresholds.append(recall)
    ar = np.mean(recalls_at_thresholds)
    
    return {
        'mAP': mAP,
        'AP50': ap50 if ap50 is not None else aps[0],
        'AR': ar
    }


def evaluate_dataset(gt_path: str, pred_path: str, dataset_name: str) -> Optional[Dict[str, float]]:
    """
    Evaluate a single dataset.
    
    Returns:
        dict with bbox and segm metrics, or None if evaluation fails
    """
    print(f"\nEvaluating {dataset_name}...")
    
    if not os.path.exists(gt_path):
        print(f"  GT file not found: {gt_path}")
        return None
    
    if not os.path.exists(pred_path):
        print(f"  Predictions file not found: {pred_path}")
        return None
    
    try:
        gt_coco = load_json(gt_path)
        pred_coco = load_json(pred_path)
    except Exception as e:
        print(f"  Error loading JSON files: {e}")
        return None
    
    # Build image ID mapping and remap predictions
    image_id_mapping = build_image_id_mapping(gt_coco, pred_coco)
    
    if len(image_id_mapping) == 0:
        print(f"  No matching images between GT and predictions")
        return None
    
    print(f"  Found {len(image_id_mapping)} matching images")
    print(f"  GT annotations: {len(gt_coco.get('annotations', []))}")
    print(f"  Pred annotations: {len(pred_coco.get('annotations', []))}")
    
    # Remap prediction image_ids to match GT
    pred_coco_remapped = remap_predictions_to_gt_image_ids(pred_coco, image_id_mapping)
    
    # Filter GT to only include images that have predictions
    gt_image_ids = set(img['id'] for img in pred_coco_remapped['images'])
    gt_coco_filtered = {
        'images': [img for img in gt_coco['images'] if img['id'] in gt_image_ids],
        'annotations': [ann for ann in gt_coco['annotations'] if ann['image_id'] in gt_image_ids],
        'categories': gt_coco.get('categories', [{'id': 1, 'name': 'insect'}])
    }
    
    # Evaluate bounding boxes
    print("  Computing bbox metrics...")
    bbox_metrics = evaluate_coco_metrics(gt_coco_filtered, pred_coco_remapped, eval_type='bbox')
    
    # Evaluate segmentation masks
    print("  Computing segmentation metrics...")
    segm_metrics = evaluate_coco_metrics(gt_coco_filtered, pred_coco_remapped, eval_type='segm')
    
    return {
        'dataset': dataset_name,
        'mAP_box': bbox_metrics['mAP'],
        'AP50_box': bbox_metrics['AP50'],
        'AR_box': bbox_metrics['AR'],
        'mAP_seg': segm_metrics['mAP'],
        'AP50_seg': segm_metrics['AP50'],
        'AR_seg': segm_metrics['AR'],
        'n_gt': len(gt_coco_filtered['annotations']),
        'n_pred': len(pred_coco_remapped['annotations']),
        'n_images': len(image_id_mapping)
    }


def print_results_table(results: List[Dict[str, float]]):
    """
    Print results in a formatted table.
    """
    if not results:
        print("\nNo results to display.")
        return
    
    # Header
    header = f"{'DATASET':<25} | {'mAP(Box)':>8} | {'AP50(Box)':>9} | {'AR(Box)':>8} || {'mAP(Seg)':>8} | {'AP50(Seg)':>9} | {'AR(Seg)':>8}"
    separator = "-" * len(header)
    
    print("\n" + separator)
    print(header)
    print(separator)
    
    for r in results:
        row = f"{r['dataset']:<25} | {r['mAP_box']:>8.4f} | {r['AP50_box']:>9.4f} | {r['AR_box']:>8.4f} || {r['mAP_seg']:>8.4f} | {r['AP50_seg']:>9.4f} | {r['AR_seg']:>8.4f}"
        print(row)
    
    print(separator)
    
    # Compute averages
    if len(results) > 1:
        avg_mAP_box = np.mean([r['mAP_box'] for r in results])
        avg_AP50_box = np.mean([r['AP50_box'] for r in results])
        avg_AR_box = np.mean([r['AR_box'] for r in results])
        avg_mAP_seg = np.mean([r['mAP_seg'] for r in results])
        avg_AP50_seg = np.mean([r['AP50_seg'] for r in results])
        avg_AR_seg = np.mean([r['AR_seg'] for r in results])
        
        avg_row = f"{'AVERAGE':<25} | {avg_mAP_box:>8.4f} | {avg_AP50_box:>9.4f} | {avg_AR_box:>8.4f} || {avg_mAP_seg:>8.4f} | {avg_AP50_seg:>9.4f} | {avg_AR_seg:>8.4f}"
        print(avg_row)
        print(separator)


def save_results_csv(results: List[Dict[str, float]], output_path: str):
    """
    Save results to a CSV file.
    """
    import csv
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    fieldnames = ['dataset', 'mAP_box', 'AP50_box', 'AR_box', 'mAP_seg', 'AP50_seg', 'AR_seg', 'n_gt', 'n_pred', 'n_images']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate flat-bug predictions using COCO metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python evaluate_coco_metrics.py --predictions-dir ./output --gt-dir ./flatbug-dataset
    python evaluate_coco_metrics.py --predictions-dir ./output --gt-dir ./flatbug-dataset --datasets ALUS ArTaxOr
        """
    )
    
    parser.add_argument('--predictions-dir', type=str, default='./output',
                        help='Directory containing prediction folders (default: ./output)')
    parser.add_argument('--gt-dir', type=str, default='./flatbug-dataset',
                        help='Directory containing GT dataset folders (default: ./flatbug-dataset)')
    parser.add_argument('--output', type=str, default='./evaluation_results.csv',
                        help='Output CSV file path (default: ./evaluation_results.csv)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Specific datasets to evaluate (default: all allowed folders)')
    parser.add_argument('--pred-filename', type=str, default=DEFAULT_PRED_FILENAME,
                        help=f'Prediction JSON filename (default: {DEFAULT_PRED_FILENAME})')
    parser.add_argument('--gt-filename', type=str, default=DEFAULT_GT_FILENAME,
                        help=f'Ground truth JSON filename (default: {DEFAULT_GT_FILENAME})')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COCO-style Evaluation for Flat-bug Predictions")
    print("=" * 80)
    print(f"Predictions directory: {args.predictions_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Output file: {args.output}")
    print(f"Prediction filename: {args.pred_filename}")
    print(f"GT filename: {args.gt_filename}")
    
    # Determine which datasets to evaluate
    if args.datasets:
        datasets_to_eval = [d for d in args.datasets if d in ALLOWED_FOLDERS]
        if len(datasets_to_eval) != len(args.datasets):
            skipped = set(args.datasets) - set(datasets_to_eval)
            print(f"\nWarning: Skipping datasets not in ALLOWED_FOLDERS: {skipped}")
    else:
        # Find datasets that have both GT and predictions
        datasets_to_eval = []
        for folder in ALLOWED_FOLDERS:
            gt_path = os.path.join(args.gt_dir, folder, args.gt_filename)
            pred_path = os.path.join(args.predictions_dir, folder, args.pred_filename)
            if os.path.exists(gt_path) and os.path.exists(pred_path):
                datasets_to_eval.append(folder)
    
    datasets_to_eval = sorted(datasets_to_eval)
    print(f"\nDatasets to evaluate ({len(datasets_to_eval)}): {datasets_to_eval}")
    
    if not datasets_to_eval:
        print("\nNo datasets found to evaluate!")
        print(f"Make sure GT files exist in {args.gt_dir}/<dataset>/{args.gt_filename}")
        print(f"And prediction files exist in {args.predictions_dir}/<dataset>/{args.pred_filename}")
        sys.exit(1)
    
    # Evaluate each dataset
    results = []
    for dataset in datasets_to_eval:
        gt_path = os.path.join(args.gt_dir, dataset, args.gt_filename)
        pred_path = os.path.join(args.predictions_dir, dataset, args.pred_filename)
        
        result = evaluate_dataset(gt_path, pred_path, dataset)
        if result:
            results.append(result)
    
    # Print and save results
    print_results_table(results)
    
    if results:
        save_results_csv(results, args.output)
    
    print(f"\nEvaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successfully evaluated {len(results)}/{len(datasets_to_eval)} datasets")


if __name__ == "__main__":
    main()
