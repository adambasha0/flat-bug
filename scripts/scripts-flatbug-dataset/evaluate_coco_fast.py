#!/usr/bin/env python3
"""
Fast COCO-style evaluation using pycocotools.

This script uses the official pycocotools library which has optimized C extensions
for computing mask IoU, making it orders of magnitude faster than pure Python.

Usage:
    python evaluate_coco_fast.py --predictions-dir ./output --gt-dir ./flatbug-dataset
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np

# Flush output immediately
sys.stdout.reconfigure(line_buffering=True)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("ERROR: pycocotools not installed. Run: pip install pycocotools")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

ALLOWED_FOLDERS = {
    "NHM-beetles-crops", "cao2022", "gernat2018", "sittinger2023",
    "amarathunga2022", "biodiscover-arm", "Mothitor", "DIRT", "Diopsis",
    "AMI-traps","AMT",
    "PeMaToEuroPep", "abram2023", "anTraX", "pinoy2023",
    "sticky-pi", "ubc-pitfall-traps", "ALUS", "BIOSCAN", "DiversityScanner",
    "ArTaxOr", "CollembolAI", "ubc-scanned-sticky-cards",
}

DEFAULT_GT_FILENAME = "instances_default.json"
DEFAULT_PRED_FILENAME = "flatbug_predictions.json"


def remap_predictions(gt_path: str, pred_path: str) -> str:
    """
    Remap prediction image_ids to match GT image_ids (matching by filename).
    Returns path to temporary remapped predictions file.
    """
    with open(gt_path) as f:
        gt = json.load(f)
    with open(pred_path) as f:
        pred = json.load(f)
    
    # Build filename -> gt_image_id mapping
    gt_fname_to_id = {img['file_name']: img['id'] for img in gt.get('images', [])}
    pred_fname_to_id = {img['file_name']: img['id'] for img in pred.get('images', [])}
    
    # Build pred_id -> gt_id mapping
    pred_to_gt_id = {}
    for fname, pred_id in pred_fname_to_id.items():
        if fname in gt_fname_to_id:
            pred_to_gt_id[pred_id] = gt_fname_to_id[fname]
    
    # Remap annotations
    remapped_anns = []
    for ann in pred.get('annotations', []):
        if ann['image_id'] in pred_to_gt_id:
            new_ann = ann.copy()
            new_ann['image_id'] = pred_to_gt_id[ann['image_id']]
            remapped_anns.append(new_ann)
    
    # Create temporary file with remapped predictions
    # pycocotools expects a list of annotations for loadRes
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(remapped_anns, tmp)
    tmp.close()
    
    return tmp.name, len(remapped_anns)


def evaluate_dataset(gt_path: str, pred_path: str, dataset_name: str) -> Optional[Dict]:
    """
    Evaluate a single dataset using pycocotools.
    """
    print(f"\nEvaluating {dataset_name}...", flush=True)
    
    try:
        # Load GT
        coco_gt = COCO(gt_path)
        n_gt = len(coco_gt.getAnnIds())
        n_images = len(coco_gt.getImgIds())
        print(f"  GT: {n_gt} annotations, {n_images} images", flush=True)
        
        # Remap predictions to match GT image IDs
        remapped_path, n_pred = remap_predictions(gt_path, pred_path)
        print(f"  Predictions: {n_pred} annotations", flush=True)
        
        if n_pred == 0:
            print(f"  Warning: No matching predictions for {dataset_name}")
            os.unlink(remapped_path)
            return None
        
        # Load predictions
        coco_dt = coco_gt.loadRes(remapped_path)
        os.unlink(remapped_path)  # Clean up temp file
        
        results = {
            'dataset': dataset_name,
            'n_gt': n_gt,
            'n_pred': n_pred,
            'n_images': n_images
        }
        
        # Evaluate bounding boxes
        print(f"  Computing bbox metrics...", flush=True)
        coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_bbox.evaluate()
        coco_eval_bbox.accumulate()
        coco_eval_bbox.summarize()
        
        results['mAP_box'] = coco_eval_bbox.stats[0]  # AP @ IoU=0.50:0.95
        results['AP50_box'] = coco_eval_bbox.stats[1]  # AP @ IoU=0.50
        results['AR_box'] = coco_eval_bbox.stats[8]    # AR @ IoU=0.50:0.95 (maxDets=100)
        
        # Evaluate segmentation
        print(f"  Computing segmentation metrics...", flush=True)
        coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval_segm.evaluate()
        coco_eval_segm.accumulate()
        coco_eval_segm.summarize()
        
        results['mAP_seg'] = coco_eval_segm.stats[0]
        results['AP50_seg'] = coco_eval_segm.stats[1]
        results['AR_seg'] = coco_eval_segm.stats[8]
        
        return results
        
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def print_results_table(results: List[Dict]):
    """Print formatted results table."""
    header = f"{'DATASET':<25} | {'mAP(Box)':>8} | {'AP50(Box)':>9} | {'AR(Box)':>8} || {'mAP(Seg)':>8} | {'AP50(Seg)':>9} | {'AR(Seg)':>8}"
    sep = "-" * len(header)
    
    print(f"\n{sep}")
    print(header)
    print(sep)
    
    for r in results:
        row = f"{r['dataset']:<25} | {r['mAP_box']:>8.4f} | {r['AP50_box']:>9.4f} | {r['AR_box']:>8.4f} || {r['mAP_seg']:>8.4f} | {r['AP50_seg']:>9.4f} | {r['AR_seg']:>8.4f}"
        print(row)
    
    print(sep)
    
    if len(results) > 1:
        avg = {
            'mAP_box': np.mean([r['mAP_box'] for r in results]),
            'AP50_box': np.mean([r['AP50_box'] for r in results]),
            'AR_box': np.mean([r['AR_box'] for r in results]),
            'mAP_seg': np.mean([r['mAP_seg'] for r in results]),
            'AP50_seg': np.mean([r['AP50_seg'] for r in results]),
            'AR_seg': np.mean([r['AR_seg'] for r in results]),
        }
        avg_row = f"{'AVERAGE':<25} | {avg['mAP_box']:>8.4f} | {avg['AP50_box']:>9.4f} | {avg['AR_box']:>8.4f} || {avg['mAP_seg']:>8.4f} | {avg['AP50_seg']:>9.4f} | {avg['AR_seg']:>8.4f}"
        print(avg_row)
        print(sep)


def save_results_csv(results: List[Dict], output_path: str):
    """Save results to CSV."""
    import csv
    
    fieldnames = ['dataset', 'mAP_box', 'AP50_box', 'AR_box', 'mAP_seg', 'AP50_seg', 'AR_seg', 'n_gt', 'n_pred', 'n_images']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Fast COCO evaluation using pycocotools')
    parser.add_argument('--predictions-dir', type=str, default='./output')
    parser.add_argument('--gt-dir', type=str, default='./flatbug-dataset')
    parser.add_argument('--output', type=str, default='./evaluation_results_fast.csv')
    parser.add_argument('--datasets', type=str, nargs='+', default=None)
    parser.add_argument('--pred-filename', type=str, default=DEFAULT_PRED_FILENAME)
    parser.add_argument('--gt-filename', type=str, default=DEFAULT_GT_FILENAME)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Fast COCO Evaluation (pycocotools)")
    print("=" * 80)
    print(f"Predictions: {args.predictions_dir}")
    print(f"Ground truth: {args.gt_dir}")
    print(f"Output: {args.output}")
    
    # Determine datasets to evaluate
    if args.datasets:
        datasets = [d for d in args.datasets if d in ALLOWED_FOLDERS]
    else:
        datasets = []
        for folder in ALLOWED_FOLDERS:
            gt_path = os.path.join(args.gt_dir, folder, args.gt_filename)
            pred_path = os.path.join(args.predictions_dir, folder, args.pred_filename)
            if os.path.exists(gt_path) and os.path.exists(pred_path):
                datasets.append(folder)
    
    datasets = sorted(datasets)
    print(f"\nDatasets to evaluate ({len(datasets)}): {datasets}", flush=True)
    
    if not datasets:
        print("No datasets found!")
        sys.exit(1)
    
    # Evaluate each dataset
    results = []
    for dataset in datasets:
        gt_path = os.path.join(args.gt_dir, dataset, args.gt_filename)
        pred_path = os.path.join(args.predictions_dir, dataset, args.pred_filename)
        
        result = evaluate_dataset(gt_path, pred_path, dataset)
        if result:
            results.append(result)
    
    # Print and save
    print_results_table(results)
    
    if results:
        save_results_csv(results, args.output)
    
    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successfully evaluated {len(results)}/{len(datasets)} datasets", flush=True)


if __name__ == "__main__":
    main()
