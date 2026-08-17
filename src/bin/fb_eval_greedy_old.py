import json
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def convert_bbox(coco_bbox):
    # Convert COCO [x, y, w, h] to R script [x1, y1, x2, y2]
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

def main():
    parser = argparse.ArgumentParser(description="Greedy Matcher for FlatBug CSV Generation")
    parser.add_argument('-p', '--predictions', required=True, help="Path to predictions JSON")
    parser.add_argument('-g', '--ground_truth', required=True, help="Path to GT JSON")
    parser.add_argument('-o', '--output', required=True, help="Path to output directory")
    
    # --- ADDED DUMMY ARGUMENTS FOR CLI COMPATIBILITY ---
    parser.add_argument('-I', '--image_directory', required=False, help="Path to image directory (Ignored by Greedy)")
    parser.add_argument('-c', '--coco_predictions', action="store_true", help="COCO format flag (Ignored by Greedy)")
    parser.add_argument('--combine', action="store_true", help="Combine flag (Always true for Greedy)")
    # --------------------------------------------------

    args = parser.parse_args()

    print("Loading JSON files...")
    gt_data = load_json(args.ground_truth)
    pred_data = load_json(args.predictions)

    img_dict = {img['id']: img for img in gt_data['images']}

    # Group annotations by image
    gt_by_img = {img['id']: [] for img in gt_data['images']}
    for ann in gt_data['annotations']:
        gt_by_img[ann['image_id']].append(ann)

    pred_by_img = {img['id']: [] for img in gt_data['images']}
    for ann in pred_data.get('annotations', []):
        if ann['image_id'] in pred_by_img:
            pred_by_img[ann['image_id']].append(ann)

    results = []

    for img_id, gt_anns in tqdm(gt_by_img.items(), desc="Greedy Matching"):
        img_info = img_dict[img_id]
        img_name = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        h, w = img_info['height'], img_info['width']
        pred_anns = pred_by_img[img_id]

        # 1. THE CURE: Sort by confidence descending
        pred_anns = sorted(pred_anns, key=lambda x: x.get('score', x.get('conf', 0)), reverse=True)

        matched_gts = set()

        if len(gt_anns) > 0 and len(pred_anns) > 0:
            gt_rles = [get_rle(ann, h, w) for ann in gt_anns]
            pred_rles = [get_rle(ann, h, w) for ann in pred_anns]
            iscrowd = [0] * len(gt_anns)
            # Generates massive Mask IoU matrix
            iou_matrix = maskUtils.iou(pred_rles, gt_rles, iscrowd)
        else:
            iou_matrix = np.zeros((len(pred_anns), len(gt_anns)))

        # 2. THE GREEDY MATCH: Iterate through predictions (highest confidence first)
        for p_idx, p_ann in enumerate(pred_anns):
            best_iou = 0.0
            best_g_idx = -1

            if len(gt_anns) > 0:
                for g_idx in range(len(gt_anns)):
                    if g_idx not in matched_gts: # If GT is not stolen yet
                        iou = iou_matrix[p_idx, g_idx]
                        if iou > best_iou:
                            best_iou = iou
                            best_g_idx = g_idx

            # Lock in the match
            if best_g_idx != -1 and best_iou > 0.01:
                matched_gts.add(best_g_idx)
                g_ann = gt_anns[best_g_idx]
                results.append({
                    "image": img_name,
                    "idx_1": g_ann['id'],
                    "idx_2": p_ann.get('id', p_idx),
                    "bbox_1": convert_bbox(g_ann['bbox']),
                    "bbox_2": convert_bbox(p_ann['bbox']),
                    "contourArea_1": g_ann.get('area', 0),
                    "contourArea_2": p_ann.get('area', 0),
                    "IoU": best_iou,
                    "conf2": p_ann.get('score', p_ann.get('conf', 0))
                })
            else:
                # Unmatched Prediction (False Positive)
                results.append({
                    "image": img_name,
                    "idx_1": -1,
                    "idx_2": p_ann.get('id', p_idx),
                    "bbox_1": "",
                    "bbox_2": convert_bbox(p_ann['bbox']),
                    "contourArea_1": 0,
                    "contourArea_2": p_ann.get('area', 0),
                    "IoU": 0.0,
                    "conf2": p_ann.get('score', p_ann.get('conf', 0))
                })

        # 3. UNMATCHED GTs: Any real bug left over is a False Negative
        for g_idx, g_ann in enumerate(gt_anns):
            if g_idx not in matched_gts:
                results.append({
                    "image": img_name,
                    "idx_1": g_ann['id'],
                    "idx_2": -1,
                    "bbox_1": convert_bbox(g_ann['bbox']),
                    "bbox_2": "",
                    "contourArea_1": g_ann.get('area', 0),
                    "contourArea_2": 0,
                    "IoU": 0.0,
                    "conf2": 0.0
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