#!/usr/bin/env python3
r"""
SAM3 Inference CLI script (FlatBug-compatible).

A CLI for SAM3 inference that produces output compatible with FlatBug's evaluation tools.
Uses FlatBug's tiling and pyramid methodology for consistent comparison.

Usage:
    ``sam3_predict -i INPUT_PATH_OR_DIRECTORY -o OUTPUT_DIRECTORY [OPTIONS]``

Options:
    -h, --help            show this help message and exit
    -i INPUT, --input INPUT
                        An image file or a directory of image files
    -o OUTPUT_DIR, --output OUTPUT_DIR
                        The result directory
    -p INPUT_PATTERN, --input-pattern INPUT_PATTERN
                        The pattern to match the images. Default is '[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$'
    -n MAX_IMAGES, --max-images MAX_IMAGES
                        Maximum number of images to process. Default is None.
    -R, --recursive       Process images nested within subdirectories of the input.
    -g GPU, --gpu GPU     Which device to use for inference. Default is 'cuda:0'.
    --config CONFIG       The config file (YAML).
    --no-overviews        Do not save the overview images.
    --only-overviews      Only save the overviews.
    -S, --no-save         Do not save the results.
    -C, --no-compiled-coco
                        Skip the production of a compiled COCO file.
    -v, --verbose         Verbose mode.
    --bpe-path BPE_PATH   Path to BPE vocabulary file. Default is './assets/bpe_simple_vocab_16e6.txt.gz'
"""

import argparse
import gc
import glob
import json
import math
import os
import re
import sys
from itertools import accumulate
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path for SAM3 imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from dotenv import load_dotenv
from huggingface_hub import login

# Load HuggingFace token
load_dotenv()
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)


# ==========================
# DEFAULT CONFIGURATION (matches FlatBug)
# ==========================
DEFAULT_CFG = {
    "TILE_SIZE": 1024,
    "MINIMUM_TILE_OVERLAP": 384,
    "EDGE_CASE_MARGIN": 16,
    "IMAGE_BOUNDARY_MARGIN": 0,  # Set to 0 to match FlatBug behavior
    "SCORE_THRESHOLD": 0.2,
    "IOU_THRESHOLD": 0.2,
    "MASK_THRESHOLD": 0.35,      # Sigmoid logit threshold for mask binarization
                                  # SAM3 masks are ~49% smaller than FlatBug GT on average.
                                  # Lower = EXPANDS masks (try 0.25-0.40 to grow undersized masks)
                                  # Higher = SHRINKS masks (0.5 is SAM3 default, too tight for FlatBug)
    "MIN_MAX_OBJ_SIZE": (32, 1e8),
    "SCALE_INCREMENT": 2/3,
    "PADDING": 32,
    "PROMPT_PLURAL": "insects",
    "PROMPT_SINGULAR": "insect",
    "CATEGORY_ID": 1,
}

# V2 Enhancement Options (based on FlatBug analysis)
V2_OPTIONS = {
    "USE_CHAIN_APPROX_NONE": True,
    "USE_DYNAMIC_TOLERANCE": True,
    "LARGEST_CONTOUR_ONLY": True,
    "MASK_DILATION_PIXELS": 0,
    "MASK_EROSION_PIXELS": 0,       # Morphological erosion to shrink masks (try 2-5 for tighter masks)
    "POLYGON_EXPANSION_PIXELS": 0,
    "BBOX_PADDING_PIXELS": 0,
    "MIN_MASK_AREA_PIXELS": 3,
    "LINEAR_INTERP_POINTS": 0,
    "USE_IOS_NMS": False,
}

# Visualization settings
MASK_FILL_COLOR_RGBA = (135, 206, 250, 60)
BBOX_COLOR = "#0051FF"
LABEL_TEXT_COLOR = "black"
MASK_BORDER_COLOR_RGBA = (0, 81, 255, 220)
MASK_BORDER_WIDTH = 6


# ==========================
# FLATBUG TILING ALGORITHMS
# ==========================

def equal_allocate_overlaps(total: int, segments: int, size: int) -> List[int]:
    """Generates cumulative positions for placing segments with controlled overlaps."""
    if segments < 2:
        return [0] * segments
    
    overlap = segments * size - total
    partial_overlap, remainder = divmod(overlap, segments - 1)
    distance = size - partial_overlap
    
    return list(accumulate(
        [distance - (1 if i < remainder else 0) for i in range(segments - 1)], 
        initial=0
    ))


def calculate_tile_offsets(
    image_size: Tuple[int, int],
    tile_size: int,
    minimum_overlap: int
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Calculate tile offsets for sliding window with minimum overlap."""
    w, h = image_size
    
    x_n_tiles = math.ceil((w - minimum_overlap) / (tile_size - minimum_overlap)) if w != tile_size else 1
    y_n_tiles = math.ceil((h - minimum_overlap) / (tile_size - minimum_overlap)) if h != tile_size else 1
    
    x_range = equal_allocate_overlaps(w, x_n_tiles, tile_size)
    y_range = equal_allocate_overlaps(h, y_n_tiles, tile_size)
    
    return [((m, n), (j, i)) for n, j in enumerate(y_range) for m, i in enumerate(x_range)]


def calculate_pyramid_scales(
    image_w: int, 
    image_h: int, 
    tile_size: int,
    scale_increment: float = 2/3
) -> List[float]:
    """Calculate pyramid scales following FlatBug methodology."""
    max_dim = max(image_w, image_h)
    scales = []
    
    s = tile_size / max_dim
    
    if s >= 1.0:
        return [1.0]
    
    while s <= 0.9:
        scales.append(s)
        s /= scale_increment
    
    scales.append(1.0)
    
    return sorted(scales)


def filter_by_edge_margin(
    boxes: np.ndarray,
    tile_size: int,
    edge_margin: int,
    tile_x: int,
    tile_y: int,
    layer_w: int,
    layer_h: int
) -> np.ndarray:
    """Filter out detections that are too close to tile edges."""
    if len(boxes) == 0:
        return np.array([], dtype=bool)
    
    is_real_left = (tile_x == 0)
    is_real_top = (tile_y == 0)
    is_real_right = (tile_x + tile_size >= layer_w)
    is_real_bottom = (tile_y + tile_size >= layer_h)
    
    keep = np.ones(len(boxes), dtype=bool)
    
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        
        touches_left = (x0 < edge_margin)
        touches_top = (y0 < edge_margin)
        touches_right = (x1 > tile_size - edge_margin)
        touches_bottom = (y1 > tile_size - edge_margin)
        
        if touches_left and not is_real_left:
            keep[i] = False
        elif touches_top and not is_real_top:
            keep[i] = False
        elif touches_right and not is_real_right:
            keep[i] = False
        elif touches_bottom and not is_real_bottom:
            keep[i] = False
    
    return keep


def filter_by_image_boundary(
    boxes: np.ndarray,
    image_w: int,
    image_h: int,
    margin: int
) -> np.ndarray:
    """Filter out detections that touch the actual image boundaries."""
    if len(boxes) == 0:
        return np.array([], dtype=bool)
    
    if margin <= 0:
        return np.ones(len(boxes), dtype=bool)
    
    keep = np.ones(len(boxes), dtype=bool)
    
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        
        touches_left = (x0 < margin)
        touches_top = (y0 < margin)
        touches_right = (x1 > image_w - margin)
        touches_bottom = (y1 > image_h - margin)
        
        if touches_left or touches_top or touches_right or touches_bottom:
            keep[i] = False
    
    return keep


def filter_by_object_size(
    boxes: np.ndarray,
    min_size: float,
    max_size: float
) -> np.ndarray:
    """Filter boxes by object size (sqrt of bbox area)."""
    if len(boxes) == 0:
        return np.array([], dtype=bool)
    
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    sizes = np.sqrt(areas)
    
    keep = (sizes >= min_size) & (sizes <= max_size)
    
    return keep


def simplify_contour(contour: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
    """Simplify a contour using cv2.approxPolyDP."""
    if len(contour) == 0:
        return contour
    
    if contour.ndim == 2:
        contour = contour.reshape(-1, 1, 2)
    
    simplified = cv2.approxPolyDP(contour.astype(np.float32), epsilon=tolerance, closed=True)
    
    return simplified


def find_contours_flatbug(
    mask: np.ndarray,
    largest_only: bool = True,
    simplify: bool = False,
    tolerance: float = 1.0,
    use_chain_approx_none: bool = True
) -> List[np.ndarray]:
    """Find contours in a binary mask using FlatBug's methodology."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    if mask.max() == 1:
        mask = mask * 255
    
    approx_method = cv2.CHAIN_APPROX_NONE if use_chain_approx_none else cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, approx_method)
    
    if len(contours) == 0:
        return []
    
    if largest_only and len(contours) > 1:
        areas = [cv2.contourArea(c) for c in contours]
        largest_idx = np.argmax(areas)
        contours = [contours[largest_idx]]
    
    if simplify:
        contours = [simplify_contour(c, tolerance) for c in contours]
    
    return contours


def calculate_dynamic_tolerance(
    mask_height: int,
    mask_width: int,
    image_height: int,
    image_width: int
) -> float:
    """Calculate dynamic polygon simplification tolerance based on mask-to-image scale."""
    scale_h = (image_height - 1) / max(mask_height - 1, 1)
    scale_w = (image_width - 1) / max(mask_width - 1, 1)
    
    tolerance = (scale_h + scale_w) / 2 / 2
    
    return max(tolerance, 1.0)


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply morphological dilation to expand a binary mask."""
    if kernel_size <= 0:
        return mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    
    return dilated


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply morphological erosion to shrink a binary mask (tighter boundaries)."""
    if kernel_size <= 0:
        return mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel, iterations=1)
    
    return eroded


def expand_polygon(polygon: List[float], expansion_px: float) -> List[float]:
    """Expand a polygon outward from its centroid."""
    if expansion_px <= 0 or len(polygon) < 6:
        return polygon
    
    pts = np.array(polygon).reshape(-1, 2)
    centroid = pts.mean(axis=0)
    
    directions = pts - centroid
    distances = np.linalg.norm(directions, axis=1, keepdims=True)
    distances = np.maximum(distances, 1e-6)
    unit_dirs = directions / distances
    
    expanded_pts = pts + unit_dirs * expansion_px
    
    return expanded_pts.flatten().tolist()


def linear_interpolate_polygon(polygon: List[float], n_interp: int = 10) -> List[float]:
    """Add interpolated points along polygon edges for smoother scaling."""
    if n_interp <= 0 or len(polygon) < 6:
        return polygon
    
    pts = np.array(polygon).reshape(-1, 2)
    n_pts = len(pts)
    
    interpolated = []
    for i in range(n_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % n_pts]
        
        interpolated.append(p1)
        
        for j in range(1, n_interp + 1):
            t = j / (n_interp + 1)
            interp_pt = p1 + t * (p2 - p1)
            interpolated.append(interp_pt)
    
    result = np.array(interpolated).flatten().tolist()
    return result


def check_min_mask_area(mask: np.ndarray, min_area: int = 3) -> bool:
    """Check if a mask has sufficient pixel area."""
    if mask.max() > 1:
        mask_binary = (mask > 127).astype(np.uint8)
    else:
        mask_binary = mask.astype(np.uint8)
    
    return mask_binary.sum() >= min_area


def compute_ios_matrix(boxes: np.ndarray) -> np.ndarray:
    """Compute IoS (Intersection over Smaller) matrix for a set of boxes."""
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 0))
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    ios_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            ix1 = max(x1[i], x1[j])
            iy1 = max(y1[i], y1[j])
            ix2 = min(x2[i], x2[j])
            iy2 = min(y2[i], y2[j])
            
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
            else:
                intersection = 0.0
            
            min_area = min(areas[i], areas[j])
            if min_area > 0:
                ios = intersection / min_area
            else:
                ios = 0.0
            
            ios_matrix[i, j] = ios
            ios_matrix[j, i] = ios
    
    return ios_matrix


def nms_ios(boxes: np.ndarray, scores: np.ndarray, ios_threshold: float = 0.2) -> np.ndarray:
    """Non-Maximum Suppression using IoS (Intersection over Smaller)."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    
    order = np.argsort(scores)[::-1]
    
    keep = []
    suppressed = set()
    
    ios_matrix = compute_ios_matrix(boxes)
    
    for idx in order:
        if idx in suppressed:
            continue
        
        keep.append(idx)
        
        for other_idx in order:
            if other_idx not in suppressed and other_idx != idx:
                if ios_matrix[idx, other_idx] > ios_threshold:
                    suppressed.add(other_idx)
    
    return np.array(keep, dtype=np.int64)


def pad_bbox(bbox: List[float], padding: int, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Pad a bounding box by the specified amount, clamping to image bounds."""
    x0, y0, x1, y1 = bbox
    
    if padding <= 0:
        return x0, y0, x1, y1
    
    new_x0 = max(0, x0 - padding)
    new_y0 = max(0, y0 - padding)
    new_x1 = min(img_w, x1 + padding)
    new_y1 = min(img_h, y1 + padding)
    
    return new_x0, new_y0, new_x1, new_y1


def mask_to_polygon_v2(
    mask_uint8: np.ndarray, 
    x_off: int, 
    y_off: int, 
    scale: float,
    tile_size: int = 1024,
    use_dynamic_tolerance: bool = True,
    largest_only: bool = True,
    use_chain_approx_none: bool = True
) -> List[List[float]]:
    """Convert binary mask to polygon(s) in global coordinates."""
    if use_dynamic_tolerance:
        mask_h, mask_w = mask_uint8.shape[:2]
        global_tile_size = tile_size / scale
        tolerance = calculate_dynamic_tolerance(mask_h, mask_w, int(global_tile_size), int(global_tile_size))
    else:
        tolerance = 1.0
    
    contours = find_contours_flatbug(
        mask_uint8, 
        largest_only=largest_only,
        simplify=False,
        use_chain_approx_none=use_chain_approx_none
    )
    
    polygons = []
    
    for cnt in contours:
        if len(cnt) < 3:
            continue
        
        points = cnt.reshape(-1, 2).astype(np.float64)
        
        points[:, 0] = (points[:, 0] + x_off) / scale
        points[:, 1] = (points[:, 1] + y_off) / scale
        points = np.maximum(points, 0)
        
        simplified = cv2.approxPolyDP(points.astype(np.float32).reshape(-1, 1, 2), 
                                       epsilon=tolerance, closed=True)
        
        if len(simplified) >= 3:
            polygons.append(simplified.reshape(-1).tolist())
    
    return polygons


# ==========================
# SAM3 INFERENCE
# ==========================

def run_sam3_inference(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    score_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SAM3 inference on a single tile."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.inference_mode():
            state = processor.set_image(image)
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt, state)
    
    # Use raw logits (sigmoid probabilities) instead of pre-binarized masks
    # This allows configurable MASK_THRESHOLD for tighter/looser masks
    masks_logits = state.get("masks_logits", state.get("masks", []))
    boxes = state.get("boxes", [])
    scores = state.get("scores", [])
    
    if len(boxes) > 0:
        b = boxes.float().detach().cpu().numpy()
        s = scores.float().detach().cpu().numpy()
        m = masks_logits.float().detach().cpu().numpy()
        
        if m.ndim == 4:
            m = m.squeeze(1)
        
        s = s.flatten()
        
        keep = s > score_threshold
        return b[keep], s[keep], m[keep]
    
    return np.array([]), np.array([]), np.array([])


def pad_image(image: Image.Image, padding: int) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Add padding to image borders."""
    w, h = image.size
    new_w = w + 2 * padding
    new_h = h + 2 * padding
    
    padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded.paste(image, (padding, padding))
    
    return padded, (padding, padding, padding, padding)


# ==========================
# MAIN PROCESSING
# ==========================

def process_image(
    processor: Sam3Processor,
    image_path: str,
    cfg: Dict[str, Any],
    device: str = "cuda:0",
    verbose: bool = False
) -> Tuple[List[Dict], Dict]:
    """Process a single image using FlatBug methodology."""
    TILE_SIZE = cfg["TILE_SIZE"]
    MINIMUM_TILE_OVERLAP = cfg["MINIMUM_TILE_OVERLAP"]
    EDGE_CASE_MARGIN = cfg["EDGE_CASE_MARGIN"]
    IMAGE_BOUNDARY_MARGIN = cfg.get("IMAGE_BOUNDARY_MARGIN", 0)
    SCORE_THRESHOLD = cfg["SCORE_THRESHOLD"]
    IOU_THRESHOLD = cfg["IOU_THRESHOLD"]
    MIN_SIZE, MAX_SIZE = cfg["MIN_MAX_OBJ_SIZE"]
    SCALE_INCREMENT = cfg["SCALE_INCREMENT"]
    PADDING = cfg["PADDING"]
    PROMPT_PLURAL = cfg["PROMPT_PLURAL"]
    PROMPT_SINGULAR = cfg["PROMPT_SINGULAR"]
    
    USE_DYNAMIC_TOLERANCE = cfg.get("USE_DYNAMIC_TOLERANCE", True)
    LARGEST_CONTOUR_ONLY = cfg.get("LARGEST_CONTOUR_ONLY", True)
    USE_CHAIN_APPROX_NONE = cfg.get("USE_CHAIN_APPROX_NONE", True)
    MASK_DILATION = cfg.get("MASK_DILATION_PIXELS", 0)
    MASK_EROSION = cfg.get("MASK_EROSION_PIXELS", 0)
    MASK_THRESHOLD = cfg.get("MASK_THRESHOLD", 0.5)
    POLYGON_EXPANSION = cfg.get("POLYGON_EXPANSION_PIXELS", 0)
    BBOX_PADDING = cfg.get("BBOX_PADDING_PIXELS", 0)
    MIN_MASK_AREA = cfg.get("MIN_MASK_AREA_PIXELS", 3)
    LINEAR_INTERP_POINTS = cfg.get("LINEAR_INTERP_POINTS", 0)
    USE_IOS_NMS = cfg.get("USE_IOS_NMS", False)
    
    orig_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_image.size
    
    padded_image, pad_lrtb = pad_image(orig_image, PADDING)
    padded_w, padded_h = padded_image.size
    
    scales = calculate_pyramid_scales(padded_w, padded_h, TILE_SIZE, SCALE_INCREMENT)
    
    if verbose:
        print(f"   Scales: {[f'{s:.3f}' for s in scales]}")
    
    all_boxes = []
    all_scores = []
    all_masks_info = []
    
    for scale in reversed(scales):
        is_max_scale = (scale == min(scales))
        
        if verbose:
            print(f"   > Scale {scale:.3f}...", end=" ", flush=True)
        
        current_prompt = PROMPT_SINGULAR if is_max_scale else PROMPT_PLURAL
        
        if scale == 1.0:
            layer_img = padded_image
        else:
            new_w = round(padded_w * scale / 4) * 4
            new_h = round(padded_h * scale / 4) * 4
            layer_img = padded_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        layer_w, layer_h = layer_img.size
        
        offsets = calculate_tile_offsets(
            image_size=(layer_w, layer_h),
            tile_size=TILE_SIZE,
            minimum_overlap=int(MINIMUM_TILE_OVERLAP * scale)
        )
        
        tiles_count = 0
        for (grid_m, grid_n), (tile_y, tile_x) in offsets:
            tiles_count += 1
            
            x_end = min(tile_x + TILE_SIZE, layer_w)
            y_end = min(tile_y + TILE_SIZE, layer_h)
            tile = layer_img.crop((tile_x, tile_y, x_end, y_end))
            
            if tile.size != (TILE_SIZE, TILE_SIZE):
                padded_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
                padded_tile.paste(tile, (0, 0))
                tile = padded_tile
            
            t_boxes, t_scores, t_masks = run_sam3_inference(
                processor, tile, current_prompt, SCORE_THRESHOLD
            )
            
            if len(t_boxes) == 0:
                continue
            
            edge_margin = 0 if is_max_scale else EDGE_CASE_MARGIN
            edge_keep = filter_by_edge_margin(
                t_boxes, TILE_SIZE, edge_margin,
                tile_x, tile_y, layer_w, layer_h
            )
            
            t_boxes = t_boxes[edge_keep]
            t_scores = t_scores[edge_keep]
            t_masks = t_masks[edge_keep]
            
            if len(t_boxes) == 0:
                continue
            
            max_obj_size = 1e9 if is_max_scale else MAX_SIZE
            size_keep = filter_by_object_size(t_boxes, MIN_SIZE, max_obj_size)
            
            t_boxes = t_boxes[size_keep]
            t_scores = t_scores[size_keep]
            t_masks = t_masks[size_keep]
            
            for i in range(len(t_boxes)):
                x0, y0, x1, y1 = t_boxes[i]
                
                gx0 = (x0 + tile_x) / scale - PADDING
                gy0 = (y0 + tile_y) / scale - PADDING
                gx1 = (x1 + tile_x) / scale - PADDING
                gy1 = (y1 + tile_y) / scale - PADDING
                
                gx0 = max(0, gx0)
                gy0 = max(0, gy0)
                gx1 = min(orig_w, gx1)
                gy1 = min(orig_h, gy1)
                
                mask_bin = (t_masks[i] > MASK_THRESHOLD).astype(np.uint8)
                
                if MIN_MASK_AREA > 0 and not check_min_mask_area(mask_bin, MIN_MASK_AREA):
                    continue
                
                if MASK_EROSION > 0:
                    mask_bin = erode_mask(mask_bin, MASK_EROSION)
                
                if MASK_DILATION > 0:
                    mask_bin = dilate_mask(mask_bin, MASK_DILATION)
                
                all_boxes.append([gx0, gy0, gx1, gy1])
                all_scores.append(float(t_scores[i]))
                
                all_masks_info.append({
                    "mask": mask_bin,
                    "x_off": tile_x,
                    "y_off": tile_y,
                    "scale": scale,
                    "padding": PADDING,
                    "tile_size": TILE_SIZE
                })
        
        if scale != 1.0:
            del layer_img
        gc.collect()
        if verbose:
            print(f"Done ({tiles_count} tiles, {len(all_boxes)} candidates)")
    
    if verbose:
        print(f"   > Total candidates: {len(all_boxes)}...", end=" ")
    
    if len(all_boxes) > 0:
        boxes_arr = np.array(all_boxes, dtype=np.float32)
        scores_arr = np.array(all_scores, dtype=np.float32)
        
        if USE_IOS_NMS:
            keep_indices = nms_ios(boxes_arr, scores_arr, IOU_THRESHOLD)
            nms_type = "IoS NMS"
        else:
            boxes_t = torch.tensor(boxes_arr, dtype=torch.float32).to(device)
            scores_t = torch.tensor(scores_arr, dtype=torch.float32).to(device)
            keep_indices = torchvision.ops.nms(boxes_t, scores_t, IOU_THRESHOLD)
            keep_indices = keep_indices.cpu().numpy()
            nms_type = "IoU NMS"
        
        final_boxes = np.array([all_boxes[i] for i in keep_indices])
        final_scores = np.array([all_scores[i] for i in keep_indices])
        final_masks = [all_masks_info[i] for i in keep_indices]
        
        if verbose:
            print(f"-> {len(final_boxes)} (after {nms_type})", end=" ")
        
        if IMAGE_BOUNDARY_MARGIN > 0 and len(final_boxes) > 0:
            boundary_keep = filter_by_image_boundary(
                final_boxes, orig_w, orig_h, IMAGE_BOUNDARY_MARGIN
            )
            final_boxes = final_boxes[boundary_keep]
            final_scores = final_scores[boundary_keep]
            final_masks = [final_masks[i] for i, keep in enumerate(boundary_keep) if keep]
            if verbose:
                print(f"-> {len(final_boxes)} (after boundary filter)")
        else:
            if verbose:
                print()
    else:
        final_boxes = np.array([])
        final_scores = np.array([])
        final_masks = []
        if verbose:
            print("-> 0")
    
    annotations = []
    for idx in range(len(final_boxes)):
        box = final_boxes[idx]
        score = final_scores[idx]
        m_info = final_masks[idx]
        
        mask_uint8 = m_info["mask"]
        
        polys = mask_to_polygon_v2(
            mask_uint8, 
            m_info["x_off"], 
            m_info["y_off"], 
            m_info["scale"],
            tile_size=m_info["tile_size"],
            use_dynamic_tolerance=USE_DYNAMIC_TOLERANCE,
            largest_only=LARGEST_CONTOUR_ONLY,
            use_chain_approx_none=USE_CHAIN_APPROX_NONE
        )
        
        adjusted_polys = []
        for poly in polys:
            adjusted = []
            for i in range(0, len(poly), 2):
                x = float(poly[i] - m_info["padding"])
                y = float(poly[i+1] - m_info["padding"])
                x = max(0.0, min(float(orig_w), x))
                y = max(0.0, min(float(orig_h), y))
                adjusted.extend([x, y])
            if len(adjusted) >= 6:
                if LINEAR_INTERP_POINTS > 0:
                    adjusted = linear_interpolate_polygon(adjusted, LINEAR_INTERP_POINTS)
                
                if POLYGON_EXPANSION > 0:
                    adjusted = expand_polygon(adjusted, POLYGON_EXPANSION)
                    clamped = []
                    for j in range(0, len(adjusted), 2):
                        px = max(0.0, min(float(orig_w), adjusted[j]))
                        py = max(0.0, min(float(orig_h), adjusted[j+1]))
                        clamped.extend([px, py])
                    adjusted = clamped
                adjusted_polys.append(adjusted)
        
        if not adjusted_polys:
            continue
        
        x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        
        if BBOX_PADDING > 0:
            x0, y0, x1, y1 = pad_bbox([x0, y0, x1, y1], BBOX_PADDING, orig_w, orig_h)
        
        bbox = [x0, y0, x1 - x0, y1 - y0]  # [x, y, w, h]
        area = (x1 - x0) * (y1 - y0)
        
        annotations.append({
            "bbox": bbox,
            "segmentation": adjusted_polys,
            "area": float(area),
            "score": float(score),
        })
    
    image_info = {
        "file_name": os.path.basename(image_path),
        "width": orig_w,
        "height": orig_h,
    }
    
    return annotations, image_info


def save_overview(
    image_path: str,
    annotations: List[Dict],
    output_path: str,
    font: Optional[ImageFont.FreeTypeFont] = None
):
    """Save visualization overview image."""
    orig_image = Image.open(image_path).convert("RGBA")
    mask_layer = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)
    
    for ann in annotations:
        for poly in ann["segmentation"]:
            if len(poly) >= 6:
                poly_tuples = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                mask_draw.polygon(poly_tuples, fill=MASK_FILL_COLOR_RGBA)
                try:
                    border_path = poly_tuples + [poly_tuples[0]]
                    mask_draw.line(border_path, fill=MASK_BORDER_COLOR_RGBA, width=MASK_BORDER_WIDTH)
                except Exception:
                    pass
    
    comp_image = Image.alpha_composite(orig_image, mask_layer)
    final_vis = comp_image.convert("RGB")
    bbox_draw = ImageDraw.Draw(final_vis)
    
    for ann in annotations:
        x0, y0, w, h = ann["bbox"]
        x1, y1 = x0 + w, y0 + h
        
        bbox_draw.rectangle([x0, y0, x1, y1], outline=BBOX_COLOR, width=3)
        
        label_txt = f"sam3 ({ann['score']:.2f})"
        if font:
            text_bbox = bbox_draw.textbbox((x0, y0), label_txt, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        else:
            text_w, text_h = 80, 12
        label_y = y0 - text_h - 4
        if label_y < 0:
            label_y = y0 + 4
        
        bbox_draw.rectangle(
            [x0, label_y, x0 + text_w + 6, label_y + text_h + 4],
            fill=BBOX_COLOR
        )
        if font:
            bbox_draw.text((x0 + 3, label_y + 1), label_txt, fill=LABEL_TEXT_COLOR, font=font)
        else:
            bbox_draw.text((x0 + 3, label_y + 1), label_txt, fill=LABEL_TEXT_COLOR)
    
    final_vis.save(output_path, quality=95)


# ==========================
# CLI INTERFACE
# ==========================

def cli_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="SAM3 Inference CLI (FlatBug-compatible output)"
    )
    
    parser.add_argument("-i", "--input", type=str, dest="input", required=True,
                        help="An image file or a directory of image files")
    parser.add_argument("-o", "--output", type=str, dest="output_dir", required=True,
                        help="The result directory")
    parser.add_argument("-p", "--input-pattern", type=str, dest="input_pattern",
                        default=r"[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$",
                        help=r"The pattern to match the images. Default is jpg/jpeg/png.")
    parser.add_argument("-n", "--max-images", type=int, dest="max_images", default=None,
                        help="Maximum number of images to process.")
    parser.add_argument("-R", "--recursive", action="store_true",
                        help="Process images nested within subdirectories of the input.")
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0",
                        help="Which device to use for inference. Default is 'cuda:0'.")
    parser.add_argument("--config", type=str, default=None,
                        help="The config file (YAML).")
    parser.add_argument("--no-overviews", action="store_true",
                        help="Do not save the overview images.")
    parser.add_argument("--only-overviews", action="store_true",
                        help="Only save the overviews.")
    parser.add_argument("-S", "--no-save", action="store_true",
                        help="Do not save the results.")
    parser.add_argument("-C", "--no-compiled-coco", action="store_true",
                        help="Skip the production of a compiled COCO file.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose mode.")
    parser.add_argument("--bpe-path", type=str, default="./assets/bpe_simple_vocab_16e6.txt.gz",
                        help="Path to BPE vocabulary file.")
    
    args = parser.parse_args()
    return vars(args)


def predict(
    input: str,
    output_dir: str,
    input_pattern: str = r"[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$",
    max_images: Optional[int] = None,
    recursive: bool = False,
    gpu: str = "cuda:0",
    config: Optional[str] = None,
    no_overviews: bool = False,
    only_overviews: bool = False,
    no_save: bool = False,
    no_compiled_coco: bool = False,
    verbose: bool = False,
    bpe_path: str = "./assets/bpe_simple_vocab_16e6.txt.gz"
):
    """Main prediction function."""
    # Sanitize paths
    input = os.path.normpath(input)
    output_dir = os.path.normpath(output_dir)
    
    if not os.path.exists(input):
        raise FileNotFoundError(f"Input path '{input}' not found.")
    
    # Load configuration
    if config is not None:
        import yaml
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        # Merge with defaults
        for key, value in DEFAULT_CFG.items():
            if key not in cfg:
                cfg[key] = value
    else:
        cfg = DEFAULT_CFG.copy()
    
    # Add V2 options
    for key, value in V2_OPTIONS.items():
        if key not in cfg:
            cfg[key] = value
    
    device = gpu
    if not torch.cuda.is_available() and "cuda" in device:
        raise ValueError(f"Device '{device}' is not available.")
    
    # Configure output flags
    overviews = not no_overviews
    if only_overviews:
        overviews = output_dir
    
    if verbose:
        print("\n" + "="*60)
        print("SAM3 Inference CLI (FlatBug-compatible)")
        print("="*60)
        print(f"\nConfiguration:")
        for key, value in cfg.items():
            print(f"  {key}: {value}")
        print()
    
    # Initialize SAM3 model
    print("Loading SAM3 Model...")
    model = build_sam3_image_model(bpe_path=bpe_path)
    model.to(device)
    model.eval()
    processor = Sam3Processor(model, device=device, confidence_threshold=cfg["SCORE_THRESHOLD"])
    print("Model Loaded.\n")
    
    # Load font for visualization
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build file list
    if os.path.isfile(input):
        file_iter = [input]
    else:
        file_iter = sorted([
            f for f in glob.glob(os.path.join(input, "**"), recursive=recursive)
            if re.search(input_pattern, f)
        ])
    
    if max_images is not None:
        file_iter = file_iter[:max_images]
    
    if len(file_iter) == 0:
        print("No images found matching the pattern.")
        return
    
    print(f"Processing {len(file_iter)} images...")
    
    # Initialize COCO output
    coco_output = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": cfg["CATEGORY_ID"], "name": cfg["PROMPT_PLURAL"]}]
    }
    
    ann_id = 1
    img_id = 1
    
    from tqdm import tqdm
    
    for img_path in tqdm(file_iter, desc="Processing images", unit="image"):
        filename = os.path.basename(img_path)
        
        try:
            if verbose:
                print(f"\n[{img_id}] {filename}")
            
            annotations, image_info = process_image(
                processor, img_path, cfg, device, verbose=verbose
            )
            
            # Add to COCO output
            coco_output["images"].append({
                "id": img_id,
                "file_name": image_info["file_name"],
                "width": image_info["width"],
                "height": image_info["height"]
            })
            
            for ann in annotations:
                coco_output["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cfg["CATEGORY_ID"],
                    "bbox": ann["bbox"],
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "iscrowd": 0,
                    "conf": ann["score"]  # FlatBug compatibility
                })
                ann_id += 1
            
            # Save overview if requested
            if not no_save and overviews and not only_overviews:
                overview_dir = os.path.join(output_dir, "overviews")
                os.makedirs(overview_dir, exist_ok=True)
                overview_path = os.path.join(overview_dir, f"overview_{os.path.splitext(filename)[0]}.jpg")
                save_overview(img_path, annotations, overview_path, font)
            elif only_overviews:
                os.makedirs(output_dir, exist_ok=True)
                overview_path = os.path.join(output_dir, f"overview_{os.path.splitext(filename)[0]}.jpg")
                save_overview(img_path, annotations, overview_path, font)
            
            if verbose:
                print(f"   Saved {len(annotations)} annotations")
            
        except Exception as e:
            print(f"ERROR processing {filename}: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
        
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            img_id += 1
    
    # Save COCO JSON
    if not no_save and not no_compiled_coco:
        coco_path = os.path.join(output_dir, "coco_instances.json")
        with open(coco_path, "w") as f:
            json.dump(coco_output, f, indent=2)
        print(f"\nSaved COCO file: {coco_path}")
    
    print(f"Total images: {img_id - 1}, Total annotations: {ann_id - 1}")


def main():
    predict(**cli_args())


if __name__ == "__main__":
    main()
