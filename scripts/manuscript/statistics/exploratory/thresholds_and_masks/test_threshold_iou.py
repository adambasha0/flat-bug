#!/usr/bin/env python3
"""
Test whether lowering MASK_THRESHOLD improves IoU against ground truth.

This script:
1. Loads cao2022 ground truth annotations
2. Runs SAM3 inference with different thresholds
3. Matches predictions to GT and computes IoU at each threshold
4. Reports how many predictions cross the 0.5 IoU threshold

Usage:
    conda run -n insects python scripts/manuscript/statistics/test_threshold_iou.py
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict

# SAM3 repo
SAM3_REPO = os.path.expanduser("~/repo/sam3-insect-segmentation")
sys.path.insert(0, SAM3_REPO)

# Paths
BASE_DIR = "/home/dolma/repo/flat-bug"
CAO2022_COCO = os.path.join(BASE_DIR, "flatbug-dataset/cao2022/instances_default.json")
CAO2022_IMAGES = os.path.join(BASE_DIR, "fb_yolo/insects/images/val")

THRESHOLDS_TO_TEST = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50]


def load_coco_annotations(coco_path: str) -> dict:
    """Load COCO annotations and organize by image."""
    with open(coco_path) as f:
        coco = json.load(f)
    
    # Build image id -> filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    
    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        filename = id_to_file[img_id]
        anns_by_image[filename].append(ann)
    
    return anns_by_image, id_to_size, id_to_file


def polygon_to_mask(segmentation, height, width):
    """Convert COCO segmentation to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in segmentation:
        if len(seg) < 6:
            continue
        pts = np.array(seg).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.3):
    """Match predictions to GT using bbox IoU. Returns list of (pred_idx, gt_idx) pairs."""
    matches = []
    used_gt = set()
    
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue
            
            # Compute bbox IoU
            x1 = max(pred_box[0], gt_box[0])
            y1 = max(pred_box[1], gt_box[1])
            x2 = min(pred_box[2], gt_box[2])
            y2 = min(pred_box[3], gt_box[3])
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            inter = (x2 - x1) * (y2 - y1)
            area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            union = area_pred + area_gt - inter
            
            iou = inter / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx))
            used_gt.add(best_gt_idx)
    
    return matches


def run_test():
    print("="*70)
    print("MASK_THRESHOLD vs IoU TEST (cao2022)")
    print("="*70)
    
    # Load SAM3
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import torch
    except ImportError as e:
        print(f"Cannot import SAM3: {e}")
        return
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    bpe_path = os.path.join(SAM3_REPO, "assets/bpe_simple_vocab_16e6.txt.gz")
    print("Loading SAM3 model...")
    model = build_sam3_image_model(bpe_path=bpe_path)
    model.to(device)
    model.eval()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.2)
    print("Model loaded.\n")
    
    # Load GT annotations
    print("Loading cao2022 ground truth...")
    anns_by_image, id_to_size, id_to_file = load_coco_annotations(CAO2022_COCO)
    print(f"Found {len(anns_by_image)} images with annotations\n")
    
    # Find cao2022 images in val set
    test_images = []
    for f in os.listdir(CAO2022_IMAGES):
        if f.startswith("cao2022") and f.endswith((".jpg", ".png")):
            test_images.append(os.path.join(CAO2022_IMAGES, f))
    
    test_images = sorted(test_images)[:10]  # Limit to 10 images for speed
    print(f"Testing on {len(test_images)} images\n")
    
    # Results storage: threshold -> list of IoUs
    results_by_threshold = {t: [] for t in THRESHOLDS_TO_TEST}
    
    for img_path in test_images:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        
        # Get GT annotations for this image
        # Val images: cao2022_000002.jpg -> GT filename: 000002.jpg
        gt_anns = None
        base_name = os.path.splitext(filename)[0]  # cao2022_000002
        # Extract the numeric part after the dataset prefix
        if "_" in base_name:
            gt_key = base_name.split("_", 1)[1] + ".jpg"  # 000002.jpg
        else:
            gt_key = filename
        
        if gt_key in anns_by_image:
            gt_anns = anns_by_image[gt_key]
        
        if gt_anns is None or len(gt_anns) == 0:
            print(f"  No GT annotations found")
            continue
        
        # Build GT masks and boxes
        gt_masks = []
        gt_boxes = []
        for ann in gt_anns:
            if "segmentation" not in ann or not ann["segmentation"]:
                continue
            mask = polygon_to_mask(ann["segmentation"], img_h, img_w)
            gt_masks.append(mask)
            
            # Get bbox [x, y, w, h] -> [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
        
        if len(gt_masks) == 0:
            print(f"  No valid GT masks")
            continue
        
        print(f"  GT: {len(gt_masks)} annotations")
        
        # Run SAM3 inference
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                state = processor.set_image(image)
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt("insects", state)
        
        masks_logits = state.get("masks_logits", state.get("masks", []))
        boxes = state.get("boxes", [])
        
        if len(boxes) == 0:
            print(f"  No detections")
            continue
        
        # Get raw logits
        m_logits = masks_logits.float().detach().cpu().numpy()
        b = boxes.float().detach().cpu().numpy()
        
        if m_logits.ndim == 4:
            m_logits = m_logits.squeeze(1)
        
        print(f"  SAM3: {len(b)} detections")
        
        # Convert boxes to [x1, y1, x2, y2]
        pred_boxes = b.tolist()
        
        # Match predictions to GT
        matches = match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.3)
        print(f"  Matched: {len(matches)} pairs")
        
        # For each threshold, compute IoU for matched pairs
        for thresh in THRESHOLDS_TO_TEST:
            for pred_idx, gt_idx in matches:
                # Binarize mask at this threshold
                mask_bin = (m_logits[pred_idx] > thresh).astype(np.uint8)
                
                # Resize mask to image size if needed
                if mask_bin.shape != (img_h, img_w):
                    mask_bin = cv2.resize(mask_bin, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                # Compute IoU with GT mask
                iou = compute_iou(mask_bin, gt_masks[gt_idx])
                results_by_threshold[thresh].append(iou)
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS: Mask IoU at different thresholds")
    print("="*70)
    
    print(f"\n{'Threshold':>10s} {'Mean IoU':>10s} {'Median':>10s} {'IoU<0.5':>10s} {'IoU>=0.5':>10s} {'% >=0.5':>10s}")
    print("-" * 62)
    
    baseline_below_50 = None
    
    for thresh in THRESHOLDS_TO_TEST:
        ious = results_by_threshold[thresh]
        if len(ious) == 0:
            continue
        
        mean_iou = np.mean(ious)
        median_iou = np.median(ious)
        below_50 = sum(1 for x in ious if x < 0.5)
        above_50 = sum(1 for x in ious if x >= 0.5)
        pct_above = 100 * above_50 / len(ious)
        
        if thresh == 0.50:
            baseline_below_50 = below_50
        
        print(f"{thresh:>10.2f} {mean_iou:>10.3f} {median_iou:>10.3f} {below_50:>10d} {above_50:>10d} {pct_above:>9.1f}%")
    
    # Show improvement from 0.50 to 0.01
    print("\n" + "="*70)
    print("IMPROVEMENT ANALYSIS")
    print("="*70)
    
    if 0.50 in results_by_threshold and 0.01 in results_by_threshold:
        ious_50 = results_by_threshold[0.50]
        ious_01 = results_by_threshold[0.01]
        
        # Count how many crossed 0.5 threshold
        crossed = sum(1 for i50, i01 in zip(ious_50, ious_01) if i50 < 0.5 and i01 >= 0.5)
        total_below = sum(1 for x in ious_50 if x < 0.5)
        
        print(f"\nAt threshold=0.50: {total_below} predictions have IoU < 0.5")
        print(f"At threshold=0.01: {crossed} of those crossed above IoU 0.5")
        print(f"Improvement rate: {100*crossed/total_below if total_below > 0 else 0:.1f}%")
        
        # Show cases in 0.4-0.5 range
        in_range = [(i50, i01) for i50, i01 in zip(ious_50, ious_01) if 0.4 <= i50 < 0.5]
        crossed_from_range = sum(1 for i50, i01 in in_range if i01 >= 0.5)
        
        print(f"\nPredictions with IoU in [0.4, 0.5) at threshold=0.50: {len(in_range)}")
        print(f"Of those, crossed to IoU >= 0.5 at threshold=0.01: {crossed_from_range}")
        
        if in_range:
            print(f"\nDetailed [0.4-0.5] cases:")
            for i, (i50, i01) in enumerate(in_range[:10]):
                delta = i01 - i50
                status = "✓ IMPROVED" if i01 >= 0.5 else "✗ still below"
                print(f"  Case {i+1}: IoU@0.50={i50:.3f} -> IoU@0.01={i01:.3f} ({delta:+.3f}) {status}")


if __name__ == "__main__":
    run_test()
