#!/usr/bin/env python3
"""
Compute AP50 and AP50-95 (mAP) metrics using pycocotools.

Usage:
    python compute_ap_metrics.py --gt <ground_truth.json> --pred <predictions.json>
"""

import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(description='Compute COCO AP metrics')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth COCO JSON')
    parser.add_argument('--pred', type=str, required=True, help='Path to predictions COCO JSON')
    args = parser.parse_args()

    print('Loading ground truth...')
    coco_gt = COCO(args.gt)
    n_gt = len(coco_gt.getAnnIds())
    n_images = len(coco_gt.getImgIds())
    print(f'GT: {n_gt} annotations, {n_images} images')

    print('Loading predictions...')
    with open(args.pred) as f:
        pred_data = json.load(f)
    print(f'Predictions: {len(pred_data["annotations"])} annotations')

    # Build filename -> image_id mapping for both GT and predictions
    gt_fname_to_id = {img['file_name']: img['id'] for img in coco_gt.dataset['images']}
    pred_fname_to_id = {img['file_name']: img['id'] for img in pred_data['images']}

    # Remap prediction image IDs to match GT image IDs (by filename)
    pred_to_gt = {}
    for fname, pred_id in pred_fname_to_id.items():
        if fname in gt_fname_to_id:
            pred_to_gt[pred_id] = gt_fname_to_id[fname]
    
    print(f'Matched {len(pred_to_gt)} images by filename')

    # Remap annotations
    remapped_anns = []
    for ann in pred_data['annotations']:
        if ann['image_id'] in pred_to_gt:
            new_ann = ann.copy()
            new_ann['image_id'] = pred_to_gt[ann['image_id']]
            # pycocotools expects 'score' field, but fb_predict uses 'conf'
            if 'conf' in new_ann and 'score' not in new_ann:
                new_ann['score'] = new_ann['conf']
            # Fix area if it's 0 - compute from bbox (x, y, w, h)
            if new_ann.get('area', 0) == 0 and 'bbox' in new_ann:
                new_ann['area'] = new_ann['bbox'][2] * new_ann['bbox'][3]
            remapped_anns.append(new_ann)
    
    print(f'Remapped {len(remapped_anns)} annotations')

    if len(remapped_anns) == 0:
        print("ERROR: No annotations matched. Check that filenames match between GT and predictions.")
        return

    # Load predictions into COCO format
    coco_dt = coco_gt.loadRes(remapped_anns)

    # Evaluate BBOX
    print('\n' + '='*60)
    print('BOUNDING BOX EVALUATION')
    print('='*60)
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()

    # Evaluate SEGMENTATION
    print('\n' + '='*60)
    print('SEGMENTATION EVALUATION')
    print('='*60)
    coco_eval_seg = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval_seg.evaluate()
    coco_eval_seg.accumulate()
    coco_eval_seg.summarize()

    # Summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'{"Metric":<25} {"BBox":>10} {"Segm":>10}')
    print('-'*45)
    print(f'{"AP50-95 (mAP)":<25} {coco_eval_bbox.stats[0]:>10.4f} {coco_eval_seg.stats[0]:>10.4f}')
    print(f'{"AP50":<25} {coco_eval_bbox.stats[1]:>10.4f} {coco_eval_seg.stats[1]:>10.4f}')
    print(f'{"AP75":<25} {coco_eval_bbox.stats[2]:>10.4f} {coco_eval_seg.stats[2]:>10.4f}')
    print(f'{"AR (maxDets=100)":<25} {coco_eval_bbox.stats[8]:>10.4f} {coco_eval_seg.stats[8]:>10.4f}')
    print('='*60)


if __name__ == '__main__':
    main()
