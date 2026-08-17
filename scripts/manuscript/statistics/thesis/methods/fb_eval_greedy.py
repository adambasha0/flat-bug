import json
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contour_geometry import (format_contour, contour_area, contour_bbox,
                              pairwise_contour_intersection)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def contour_bbox_str(ann):
    """Return '[x1, y1, x2, y2]' derived from the segmentation polygon contour.

    Matches fb_eval.py / compare_groups() behaviour: bboxes are computed from
    the polygon vertices, not taken from the COCO bbox field (which may be
    padded).
    Falls back to the COCO bbox field for RLE segmentations.
    """
    seg = ann.get('segmentation', None)
    if isinstance(seg, list) and len(seg) > 0:
        c = format_contour(seg)
        b = contour_bbox(c)  # [x1, y1, x2, y2] integers
        return f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]"
    # Fallback: use COCO bbox field
    coco_bbox = ann.get('bbox', None)
    if not coco_bbox or len(coco_bbox) != 4:
        return ""
    x, y, w, h = coco_bbox
    return f"[{x}, {y}, {x+w}, {y+h}]"

def get_rle(ann, h, w):
    # Safely extract and format RLE masks for pycocotools
    if 'segmentation' in ann:
        segm = ann['segmentation']
        if isinstance(segm, list) and len(segm) > 0:
            rles = maskUtils.frPyObjects(segm, h, w)
            return maskUtils.merge(rles)
        elif isinstance(segm, dict) and 'counts' in segm:
            if isinstance(segm['counts'], list):
                return maskUtils.frPyObjects(segm, h, w)
            return segm
    return None


def get_mask_area(ann, h, w):
    """Return mask area in pixels, computing from segmentation when 'area' is 0."""
    area = ann.get('area', 0)
    if area and float(area) > 0:
        return float(area)
    rle = get_rle(ann, h, w)
    if rle is not None:
        return float(maskUtils.area(rle))
    return 0.0


def _is_polygon(ann):
    """Return True if the annotation's segmentation is a polygon list (not RLE)."""
    seg = ann.get('segmentation', None)
    return isinstance(seg, list) and len(seg) > 0


def compute_iou_matrix(gt_anns, pred_anns, h, w):
    """Compute IoU matrix of shape (n_gt, n_pred).

    Uses contour IoU for polygon segmentations — identical to fb_eval.py /
    match_geoms() — so matching results are consistent.  Falls back to mask IoU
    for RLE segmentations (e.g. SAM3).
    """
    n_gt   = len(gt_anns)
    n_pred = len(pred_anns)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float32)

    all_polygon = all(_is_polygon(a) for a in gt_anns + pred_anns)
    if all_polygon:
        gt_contours   = [format_contour(a['segmentation']) for a in gt_anns]
        pred_contours = [format_contour(a['segmentation']) for a in pred_anns]
        areas1 = np.array([contour_area(c) for c in gt_contours],   dtype=np.float32)
        areas2 = np.array([contour_area(c) for c in pred_contours], dtype=np.float32)
        intersections = pairwise_contour_intersection(gt_contours, pred_contours)
        union = areas1.reshape(-1, 1) + areas2.reshape(1, -1) - intersections
        return np.where(union > 0, intersections / union, 0.0).astype(np.float32)
    else:
        # Mask IoU fallback for RLE / mixed segmentations
        gt_rles   = [get_rle(a, h, w) for a in gt_anns]
        pred_rles = [get_rle(a, h, w) for a in pred_anns]
        return maskUtils.iou(pred_rles, gt_rles, [0] * n_gt).T.copy().astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Greedy Matcher for FlatBug CSV Generation")
    parser.add_argument('-p', '--predictions', required=True, help="Path to predictions JSON")
    parser.add_argument('-g', '--ground_truth', required=True, help="Path to GT JSON")
    parser.add_argument('-o', '--output', required=True, help="Path to output directory")
    
    # --- ADDED DUMMY ARGUMENTS FOR CLI COMPATIBILITY ---
    parser.add_argument('-I', '--image_directory', required=False, help="Path to image directory (Ignored by Greedy)")
    parser.add_argument('-c', '--coco_predictions', action="store_true", help="COCO format flag (Ignored by Greedy)")
    parser.add_argument('--combine', action="store_true", help="Combine flag (Always true for Greedy)")
    parser.add_argument('--iou-threshold', type=float, default=0.2, dest='iou_threshold',
                        help="IoU threshold for matching (default 0.2, same as fb_eval.py config).")
    parser.add_argument('--coco-matching', action='store_true', dest='coco_matching',
                        help="Use COCO-style confidence-sorted greedy matching instead of the "
                             "default bidirectional matching (which mirrors fb_eval.py / "
                             "match_geoms()).  COCO matching prioritises high-confidence "
                             "predictions when assigning GTs, producing higher AP for models "
                             "that generate many redundant detections (e.g. SAM3). "
                             "Use this flag consistently for both models when comparing "
                             "results outside the fb_eval.py reference framework.")
    # --------------------------------------------------

    args = parser.parse_args()

    if args.coco_matching:
        print("Matching mode: COCO-style confidence-sorted greedy  "
              "(high-confidence predictions get first pick of GTs)")
    else:
        print("Matching mode: bidirectional optimal  "
              "(mirrors fb_eval.py / match_geoms(), geometry-based)")

    print("Loading JSON files...")
    gt_data = load_json(args.ground_truth)
    pred_data = load_json(args.predictions)

    img_dict = {img['id']: img for img in gt_data['images']}

    # Group annotations by image
    min_size = 32  # same as DEFAULT_CFG["MIN_MAX_OBJ_SIZE"][0] used by fb_eval.py
    min_area = min_size  # filter_coco(area=min_size) removes bbox w*h < min_size (not min_size²)

    # Build filename-based lookup for predictions (pred JSON may have different image IDs than GT)
    pred_img_by_filename = {os.path.basename(img['file_name']): img['id'] for img in pred_data.get('images', [])}

    gt_by_img = {img['id']: [] for img in gt_data['images']}
    for ann in gt_data['annotations']:
        # Filter small GT annotations (same as filter_coco(gt_coco, area=min_size) in fb_eval.py)
        if ann['bbox'][2] * ann['bbox'][3] >= min_area:
            gt_by_img[ann['image_id']].append(ann)

    # Map pred annotations by filename (to handle different image_id numbering between GT and pred)
    pred_by_filename = {}
    for ann in pred_data.get('annotations', []):
        pred_by_filename.setdefault(ann['image_id'], []).append(ann)

    # Remap to GT image IDs via filename matching
    pred_by_img = {img['id']: [] for img in gt_data['images']}
    for img in gt_data['images']:
        basename = os.path.basename(img['file_name'])
        pred_img_id = pred_img_by_filename.get(basename)
        if pred_img_id is not None:
            for ann in pred_by_filename.get(pred_img_id, []):
                # Filter small predictions (same as filter_coco in fb_eval.py)
                if ann['bbox'][2] * ann['bbox'][3] >= min_area:
                    pred_by_img[img['id']].append(ann)

    results = []

    for img_id, gt_anns in tqdm(gt_by_img.items(), desc="Matching"):
        # Skip images with no GT annotations (e.g. unannotated background images).
        # Predictions for such images cannot be evaluated and would become spurious FPs,
        # lowering precision artificially — consistent with fb_eval.py (shared_keys).
        if not gt_anns:
            continue
        img_info = img_dict[img_id]
        img_name = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        h, w = img_info['height'], img_info['width']
        pred_anns = pred_by_img[img_id]

        n_gt   = len(gt_anns)
        n_pred = len(pred_anns)

        iou = compute_iou_matrix(gt_anns, pred_anns, h, w)
        iou_orig = iou.copy()

        gt_to_pred    = np.full(n_gt, -1, dtype=np.int32)
        matched_preds = set()

        if n_gt > 0 and n_pred > 0:
            if args.coco_matching:
                # COCO-style confidence-sorted greedy matching:
                # Iterate predictions from highest to lowest confidence; each
                # prediction claims the best available GT with IoU > threshold.
                # High-confidence predictions get priority, which rewards
                # well-calibrated confidence scores and matches the standard COCO
                # evaluation protocol.
                conf_order = np.argsort(
                    [-p_ann.get('score', p_ann.get('conf', 0)) for p_ann in pred_anns]
                )
                matched_gts = set()
                for p_idx in conf_order:
                    best_iou  = args.iou_threshold
                    best_g    = -1
                    for g_idx in range(n_gt):
                        if g_idx not in matched_gts and iou[g_idx, p_idx] > best_iou:
                            best_iou = iou[g_idx, p_idx]
                            best_g   = g_idx
                    if best_g >= 0:
                        gt_to_pred[best_g] = p_idx
                        matched_preds.add(p_idx)
                        matched_gts.add(best_g)
            else:
                # Optimal bidirectional matching — mirrors match_geoms() in eval_utils.py:
                # Process each GT in ascending order of its best IoU (hardest first).
                # A GT-pred pair is only accepted when each is the other's best candidate
                # (bidirectional constraint), rewarding geometric accuracy over confidence.
                max_iou_per_gt = iou.max(axis=1)
                for focus in np.argsort(max_iou_per_gt):
                    candidates = np.where(iou[focus] > args.iou_threshold)[0]
                    if len(candidates) == 0:
                        continue
                    # Bidirectional check: keep only candidates for which focus is
                    # the highest-IoU GT (among still-available GTs)
                    best_gt_for_cand = iou[:, candidates].argmax(axis=0)
                    candidates = candidates[best_gt_for_cand == focus]
                    if len(candidates) == 0:
                        continue
                    best_pred = int(candidates[np.argmax(iou[focus, candidates])])
                    gt_to_pred[focus] = best_pred
                    matched_preds.add(best_pred)
                    iou[:, best_pred] = 0  # prevent re-matching

        # Build result rows
        for g_idx, g_ann in enumerate(gt_anns):
            p_idx = int(gt_to_pred[g_idx])
            if p_idx >= 0:
                p_ann = pred_anns[p_idx]
                results.append({
                    "image": img_name,
                    "idx_1": g_ann['id'],
                    "idx_2": p_ann.get('id', p_idx),
                    "bbox_1": contour_bbox_str(g_ann),
                    "bbox_2": contour_bbox_str(p_ann),
                    "contourArea_1": get_mask_area(g_ann, h, w),
                    "contourArea_2": get_mask_area(p_ann, h, w),
                    "IoU": float(iou_orig[g_idx, p_idx]),
                    "conf1": 1.0,
                    "conf2": p_ann.get('score', p_ann.get('conf', 0))
                })
            else:
                # Unmatched GT → False Negative
                results.append({
                    "image": img_name,
                    "idx_1": g_ann['id'],
                    "idx_2": -1,
                    "bbox_1": contour_bbox_str(g_ann),
                    "bbox_2": "",
                    "contourArea_1": get_mask_area(g_ann, h, w),
                    "contourArea_2": 0,
                    "IoU": 0.0,
                    "conf1": 1.0,
                    "conf2": 0.0
                })

        for p_idx, p_ann in enumerate(pred_anns):
            if p_idx not in matched_preds:
                # Unmatched prediction → False Positive
                results.append({
                    "image": img_name,
                    "idx_1": -1,
                    "idx_2": p_ann.get('id', p_idx),
                    "bbox_1": "",
                    "bbox_2": contour_bbox_str(p_ann),
                    "contourArea_1": 0,
                    "contourArea_2": get_mask_area(p_ann, h, w),
                    "IoU": 0.0,
                    "conf1": float('nan'),
                    "conf2": p_ann.get('score', p_ann.get('conf', 0))
                })

    # Prepare output directory and path
    os.makedirs(args.output, exist_ok=True)
    out_file = os.path.join(args.output, "combined_results.csv")

    # Save exactly as ap_curve.R expects (sep=";")
    df = pd.DataFrame(results)
    df.to_csv(out_file, sep=";", index=False)
    print(f"\nDone! Successfully saved {len(df)} rows to {out_file}")

if __name__ == "__main__":
    main()