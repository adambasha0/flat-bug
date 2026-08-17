    #!/usr/bin/env python3
"""
Hard Negative Mining Script for SAM3 Fine-tuning.

Generates "poison" crops of empty backgrounds from TRAINING images to teach the
model what empty backgrounds look like (reducing hallucinations on sticky traps,
shadows, and plastic borders).

=================================================================================
HOW TO RUN (ON REMOTE SERVER):
=================================================================================

1. Activate your environment:
   conda activate sam3_gpu  # or your training environment

2. Generate hard negatives DIRECTLY INTO the train directory:
    python3 scripts/training/generate_hard_negatives.py \
        --images /data/sam3/fb_yolo/insects/images/train \
        --coco /data/sam3/fb_yolo/insects/labels/train/instances_default.json \
        --output /data/sam3/fb_yolo/insects/images/train \
        --output-json /data/sam3/fb_yolo/insects/labels/train/hard_negatives.json \
        --num-crops 500
        --crop-size 1024

   ⚠️  IMPORTANT: --output MUST be the SAME as --images (train directory)!
   The merged COCO JSON expects all images in the same directory.

3. Merge hard_negatives.json with your training JSON:
   python3 scripts/training/generate_hard_negatives.py --merge \
       --main-json /data/sam3/fb_yolo/insects/labels/train/instances_default.json \
       --negative-json /data/sam3/fb_yolo/insects/labels/train/hard_negatives.json \
       --output-json /data/sam3/fb_yolo/insects/labels/train/instances_with_negatives.json

4. Update your training config to use instances_with_negatives.json

   If you saved images to a SEPARATE directory by mistake, copy them:
   cp /data/sam3/fb_yolo/insects/images/hard_negatives_train/*.jpg \
      /data/sam3/fb_yolo/insects/images/train/

=================================================================================
WHAT THIS DOES:
=================================================================================

- Loads training images and their GT annotations
- Prioritizes problem datasets: sticky-pi, pinoy2023, abram2023, etc.
- For each image, attempts random crops that have ZERO overlap with any GT bbox
- Adds 50px safety padding around GT boxes to exclude insect legs/antennae
- Generates a COCO JSON with empty annotations list

This creates a "poison" dataset that teaches SAM3: "these regions contain NO insects"

=================================================================================
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ============================================================================
# CONFIGURATION (easily adjustable)
# ============================================================================

# Default crop size (pixels)
DEFAULT_CROP_SIZE = 1024

# Default number of crops to extract
DEFAULT_NUM_CROPS = 500

# Safety padding around GT boxes (pixels) - excludes antennae/legs
GT_PADDING = 50

# Max attempts per image to find a valid crop
MAX_ATTEMPTS_PER_IMAGE = 50

# Priority datasets (these have more hallucinations)
PRIORITY_DATASETS = [
    "sticky-pi",
    "pinoy2023",
    "abram2023",
    "ubc-scanned-sticky-cards",
    "CollembolAI",
    "PeMaToEuroPep",
    "AMI-traps",
    "BIOSCAN",
    "ArTaxOr",
]

# Weight for priority datasets (higher = more likely to be sampled)
PRIORITY_WEIGHT = 3


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_coco(coco_path: str) -> dict:
    """Load COCO JSON file."""
    with open(coco_path, 'r') as f:
        return json.load(f)


def save_coco(coco: dict, output_path: str):
    """Save COCO JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"Saved COCO JSON to: {output_path}")


def get_image_annotations(coco: dict) -> Dict[int, List[dict]]:
    """Create mapping from image_id to list of annotations."""
    img_to_anns = defaultdict(list)
    for ann in coco.get('annotations', []):
        img_to_anns[ann['image_id']].append(ann)
    return img_to_anns


def get_image_by_id(coco: dict, image_id: int) -> Optional[dict]:
    """Get image metadata by ID."""
    for img in coco['images']:
        if img['id'] == image_id:
            return img
    return None


def get_dataset_from_filename(filename: str) -> str:
    """Extract dataset name from filename (e.g., 'sticky-pi_001.jpg' -> 'sticky-pi')."""
    # Handle filenames like 'sticky-pi_001.jpg' or 'pinoy2023_image001.jpg'
    parts = filename.rsplit('_', 1)
    if len(parts) >= 2:
        return parts[0]
    return filename


def is_priority_dataset(filename: str) -> bool:
    """Check if filename belongs to a priority dataset."""
    for ds in PRIORITY_DATASETS:
        if filename.startswith(ds) or ds in filename:
            return True
    return False


def boxes_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
    """
    Check if two boxes overlap.
    Boxes are in format (x, y, w, h).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to x1, y1, x2, y2 format
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2
    
    # Check for non-overlap conditions
    if x1 >= box2_x2 or box1_x2 <= x2:
        return False
    if y1 >= box2_y2 or box1_y2 <= y2:
        return False
    
    return True


def get_padded_boxes(annotations: List[dict], padding: int = GT_PADDING) -> List[Tuple[int, int, int, int]]:
    """
    Get bounding boxes with safety padding.
    Returns list of (x, y, w, h) tuples.
    """
    boxes = []
    for ann in annotations:
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            # Add padding
            px = max(0, x - padding)
            py = max(0, y - padding)
            pw = w + 2 * padding
            ph = h + 2 * padding
            boxes.append((px, py, pw, ph))
    return boxes


def find_valid_crop(
    img_width: int,
    img_height: int,
    gt_boxes: List[Tuple[int, int, int, int]],
    crop_size: int,
    max_attempts: int = MAX_ATTEMPTS_PER_IMAGE
) -> Optional[Tuple[int, int]]:
    """
    Find a random crop position that doesn't overlap with any GT boxes.
    Returns (x, y) top-left corner or None if no valid crop found.
    """
    if img_width < crop_size or img_height < crop_size:
        return None
    
    for _ in range(max_attempts):
        # Random top-left corner
        x = random.randint(0, img_width - crop_size)
        y = random.randint(0, img_height - crop_size)
        
        crop_box = (x, y, crop_size, crop_size)
        
        # Check against all GT boxes
        overlap = False
        for gt_box in gt_boxes:
            if boxes_overlap(crop_box, gt_box):
                overlap = True
                break
        
        if not overlap:
            return (x, y)
    
    return None


def extract_crop(image_path: str, x: int, y: int, crop_size: int) -> Optional[np.ndarray]:
    """Extract a crop from an image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    crop = img[y:y+crop_size, x:x+crop_size]
    return crop


# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def generate_hard_negatives(
    images_dir: str,
    coco_path: str,
    output_dir: str,
    output_json: str,
    num_crops: int = DEFAULT_NUM_CROPS,
    crop_size: int = DEFAULT_CROP_SIZE,
    seed: int = 42
) -> dict:
    """
    Generate hard negative crops from training images.
    
    Args:
        images_dir: Path to training images
        coco_path: Path to training COCO JSON
        output_dir: Directory to save cropped images
        output_json: Path for output COCO JSON
        num_crops: Number of crops to extract
        crop_size: Size of each crop (square)
        seed: Random seed for reproducibility
    
    Returns:
        COCO dict with hard negatives metadata
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 80)
    print("HARD NEGATIVE MINING FOR SAM3")
    print("=" * 80)
    print(f"\nImages directory: {images_dir}")
    print(f"COCO JSON: {coco_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output JSON: {output_json}")
    print(f"Target crops: {num_crops}")
    print(f"Crop size: {crop_size}x{crop_size}")
    print(f"GT padding: {GT_PADDING}px")
    print()
    
    # Load COCO data
    print("Loading COCO annotations...")
    coco = load_coco(coco_path)
    img_to_anns = get_image_annotations(coco)
    
    print(f"Total images in training set: {len(coco['images'])}")
    print(f"Total annotations: {len(coco.get('annotations', []))}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate images into priority and regular
    priority_images = []
    regular_images = []
    
    for img in coco['images']:
        filename = img['file_name']
        if is_priority_dataset(filename):
            priority_images.append(img)
        else:
            regular_images.append(img)
    
    print(f"\nPriority dataset images: {len(priority_images)}")
    print(f"Regular dataset images: {len(regular_images)}")
    
    # Create weighted sampling pool
    # Priority images appear more times in the pool
    sampling_pool = priority_images * PRIORITY_WEIGHT + regular_images
    random.shuffle(sampling_pool)
    
    print(f"Sampling pool size: {len(sampling_pool)} (priority weight: {PRIORITY_WEIGHT}x)")
    
    # Output COCO structure
    output_coco = {
        "info": {
            "description": "Hard Negative Mining Dataset for SAM3",
            "version": "1.0",
            "year": 2026,
            "contributor": "Auto-generated",
            "date_created": "2026-03-02"
        },
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", [{"id": 1, "name": "insect", "supercategory": ""}]),
        "images": [],
        "annotations": []  # EMPTY - these are hard negatives!
    }
    
    # Track statistics
    stats = {
        'total_attempts': 0,
        'successful_crops': 0,
        'skipped_too_small': 0,
        'skipped_no_valid_region': 0,
        'skipped_file_not_found': 0,
        'by_dataset': defaultdict(int)
    }
    
    crops_extracted = 0
    image_id = 1
    
    # Progress bar
    pbar = tqdm(total=num_crops, desc="Extracting hard negatives")
    
    pool_idx = 0
    while crops_extracted < num_crops and pool_idx < len(sampling_pool) * 2:
        # Cycle through pool
        img_meta = sampling_pool[pool_idx % len(sampling_pool)]
        pool_idx += 1
        stats['total_attempts'] += 1
        
        img_id = img_meta['id']
        filename = img_meta['file_name']
        width = img_meta['width']
        height = img_meta['height']
        
        # Check image size
        if width < crop_size or height < crop_size:
            stats['skipped_too_small'] += 1
            continue
        
        # Get GT boxes with padding
        annotations = img_to_anns.get(img_id, [])
        gt_boxes = get_padded_boxes(annotations, GT_PADDING)
        
        # Find valid crop location
        crop_pos = find_valid_crop(width, height, gt_boxes, crop_size)
        
        if crop_pos is None:
            stats['skipped_no_valid_region'] += 1
            continue
        
        x, y = crop_pos
        
        # Load and crop image
        image_path = os.path.join(images_dir, filename)
        if not os.path.exists(image_path):
            stats['skipped_file_not_found'] += 1
            continue
        
        crop = extract_crop(image_path, x, y, crop_size)
        if crop is None:
            stats['skipped_file_not_found'] += 1
            continue
        
        # Generate output filename
        dataset_name = get_dataset_from_filename(filename)
        base_name = os.path.splitext(filename)[0]
        crop_filename = f"hn_{base_name}_x{x}_y{y}.jpg"
        
        # Save crop
        output_path = os.path.join(output_dir, crop_filename)
        cv2.imwrite(output_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Add to output COCO
        output_coco['images'].append({
            "id": image_id,
            "width": crop_size,
            "height": crop_size,
            "file_name": crop_filename,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
            # Extra metadata for traceability
            "source_image": filename,
            "crop_x": x,
            "crop_y": y,
            "is_hard_negative": True
        })
        
        stats['successful_crops'] += 1
        stats['by_dataset'][dataset_name] += 1
        crops_extracted += 1
        image_id += 1
        
        pbar.update(1)
    
    pbar.close()
    
    # Save output COCO JSON
    save_coco(output_coco, output_json)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nTotal crops extracted: {stats['successful_crops']}")
    print(f"Total attempts: {stats['total_attempts']}")
    print(f"Skipped (too small): {stats['skipped_too_small']}")
    print(f"Skipped (no valid region): {stats['skipped_no_valid_region']}")
    print(f"Skipped (file not found): {stats['skipped_file_not_found']}")
    
    print("\nCrops by dataset:")
    for ds, count in sorted(stats['by_dataset'].items(), key=lambda x: -x[1]):
        print(f"  {ds}: {count}")
    
    print(f"\nOutput images: {output_dir}")
    print(f"Output COCO JSON: {output_json}")
    print(f"Total images in JSON: {len(output_coco['images'])}")
    print(f"Total annotations in JSON: {len(output_coco['annotations'])} (should be 0)")
    
    # Check if output_dir matches images_dir (should be same for merged JSON to work)
    if os.path.abspath(output_dir) != os.path.abspath(images_dir):
        print("\n" + "=" * 80)
        print("⚠️  WARNING: Images saved to DIFFERENT directory than training images!")
        print("=" * 80)
        print(f"\nThe merged COCO JSON expects all images in the same directory.")
        print(f"You MUST copy the hard negative images to the training directory:")
        print(f"\n    cp {output_dir}/*.jpg {images_dir}/")
        print("\nOr re-run with --output set to the training images directory.")
        print("=" * 80)
    
    return output_coco


# ============================================================================
# MERGE FUNCTION
# ============================================================================

def merge_coco_files(
    main_json: str,
    negative_json: str,
    output_json: str
) -> dict:
    """
    Merge the main training COCO JSON with the hard negatives COCO JSON.
    
    Args:
        main_json: Path to main training COCO JSON
        negative_json: Path to hard negatives COCO JSON
        output_json: Path for merged output JSON
    
    Returns:
        Merged COCO dict
    """
    print("=" * 80)
    print("MERGING COCO FILES")
    print("=" * 80)
    
    print(f"\nMain JSON: {main_json}")
    print(f"Negative JSON: {negative_json}")
    print(f"Output JSON: {output_json}")
    
    # Load both files
    main_coco = load_coco(main_json)
    negative_coco = load_coco(negative_json)
    
    print(f"\nMain dataset:")
    print(f"  Images: {len(main_coco['images'])}")
    print(f"  Annotations: {len(main_coco.get('annotations', []))}")
    
    print(f"\nHard negatives dataset:")
    print(f"  Images: {len(negative_coco['images'])}")
    print(f"  Annotations: {len(negative_coco.get('annotations', []))} (should be 0)")
    
    # Find max IDs in main dataset
    max_img_id = max(img['id'] for img in main_coco['images'])
    max_ann_id = max((ann['id'] for ann in main_coco.get('annotations', [])), default=0)
    
    print(f"\nMax image ID in main: {max_img_id}")
    print(f"Max annotation ID in main: {max_ann_id}")
    
    # Remap IDs in negative dataset
    for img in negative_coco['images']:
        old_id = img['id']
        img['id'] = old_id + max_img_id
    
    # Merge
    merged_coco = main_coco.copy()
    merged_coco['images'] = main_coco['images'] + negative_coco['images']
    # Annotations stay the same (negatives have none)
    
    # Update info
    merged_coco['info'] = {
        "description": "FlatBug Training Dataset with Hard Negatives",
        "version": "1.0",
        "year": 2026,
        "contributor": "Auto-generated",
        "date_created": "2026-03-02"
    }
    
    # Save merged file
    save_coco(merged_coco, output_json)
    
    print(f"\nMerged dataset:")
    print(f"  Images: {len(merged_coco['images'])}")
    print(f"  Annotations: {len(merged_coco.get('annotations', []))}")
    print(f"  Hard negative images: {len(negative_coco['images'])}")
    
    print(f"\n✅ Merged COCO saved to: {output_json}")
    print("\nNext step: Update your training config to use this merged JSON file.")
    
    return merged_coco


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Hard Negative Mining dataset for SAM3 fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate hard negatives (SAVE TO TRAIN DIRECTORY!)
  python3 generate_hard_negatives.py \
      --images /data/sam3/fb_yolo/insects/images/train \
      --coco /data/sam3/fb_yolo/insects/labels/train/instances_default.json \
      --output /data/sam3/fb_yolo/insects/images/train \
      --output-json /data/sam3/fb_yolo/insects/labels/train/hard_negatives.json \
      --num-crops 500

  # Merge with main training JSON
  python3 generate_hard_negatives.py --merge \
      --main-json /data/sam3/fb_yolo/insects/labels/train/instances_default.json \
      --negative-json /data/sam3/fb_yolo/insects/labels/train/hard_negatives.json \
      --output-json /data/sam3/fb_yolo/insects/labels/train/instances_with_negatives.json
        """
    )
    
    # Mode selection
    parser.add_argument('--merge', action='store_true',
                        help='Merge mode: combine main JSON with hard negatives JSON')
    
    # Generation arguments
    parser.add_argument('--images', type=str,
                        help='Path to training images directory')
    parser.add_argument('--coco', type=str,
                        help='Path to training COCO JSON file')
    parser.add_argument('--output', type=str,
                        help='Output directory for cropped hard negative images')
    parser.add_argument('--output-json', type=str,
                        help='Output path for hard negatives COCO JSON')
    parser.add_argument('--num-crops', type=int, default=DEFAULT_NUM_CROPS,
                        help=f'Number of hard negative crops to extract (default: {DEFAULT_NUM_CROPS})')
    parser.add_argument('--crop-size', type=int, default=DEFAULT_CROP_SIZE,
                        help=f'Size of each crop in pixels (default: {DEFAULT_CROP_SIZE})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Merge arguments
    parser.add_argument('--main-json', type=str,
                        help='[Merge mode] Path to main training COCO JSON')
    parser.add_argument('--negative-json', type=str,
                        help='[Merge mode] Path to hard negatives COCO JSON')
    
    args = parser.parse_args()
    
    if args.merge:
        # Merge mode
        if not all([args.main_json, args.negative_json, args.output_json]):
            parser.error("Merge mode requires --main-json, --negative-json, and --output-json")
        
        merge_coco_files(args.main_json, args.negative_json, args.output_json)
    
    else:
        # Generation mode
        if not all([args.images, args.coco, args.output, args.output_json]):
            parser.error("Generation mode requires --images, --coco, --output, and --output-json")
        
        generate_hard_negatives(
            images_dir=args.images,
            coco_path=args.coco,
            output_dir=args.output,
            output_json=args.output_json,
            num_crops=args.num_crops,
            crop_size=args.crop_size,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
