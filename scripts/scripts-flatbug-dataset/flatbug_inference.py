#!/usr/bin/env python3
"""
Flat-bug inference script for processing multiple datasets.

This script iterates over the flatbug datasets and runs the same detection logic
as in the flat-bug.ipynb notebook.

Usage:
    nohup python3 -u scripts/scripts-flatbug-dataset/flatbug_inference.py > scripts/flatbug-logs/flatbug_inference.log 2>&1 &
"""

import os
import sys
import glob
import json
import io
import zipfile
import base64
import uuid
import re
import tempfile
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple, Union, Optional
from tqdm import tqdm

import numpy as np
import torch

try:
    import rawpy
except ImportError:
    rawpy = None
    print("Warning: rawpy not installed. DNG files will not be supported.")

from PIL import Image

# Add the flat_bug source to path if needed
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flat_bug.predictor import Predictor, TensorPredictions

# =============================================================================
# Configuration
# =============================================================================

ALLOWED_FOLDERS = {
    #"NHM-beetles-crops",
    #"cao2022",
    "gernat2018",
    #"sittinger2023",
    #"amarathunga2022",
    #"biodiscover-arm",
    #"Mothitor",
    #"DIRT",
    #"Diopsis",
    # "AMI-traps",
    # "AMT",
    #"PeMaToEuroPep",
    #"abram2023",
    #"anTraX",
    #"pinoy2023",
    #"sticky-pi",
    #"ubc-pitfall-traps",
    # "ALUS",
    #"BIOSCAN",
    #"DiversityScanner",
    #"ArTaxOr",
    #"CollembolAI",
    #"ubc-scanned-sticky-cards",
}

# Default paths - can be overridden via command line
DEFAULT_DATASET_ROOT = "./flatbug-dataset"
DEFAULT_OUTPUT_ROOT = "./output"

# =============================================================================
# Input Parsing
# =============================================================================

IMG_REGEX = re.compile(r'\.(jp[e]{0,1}g|png|dng)$', re.IGNORECASE)


def is_image(file_path: str) -> bool:
    return bool(re.search(IMG_REGEX, file_path)) and os.path.isfile(file_path)


def is_txt(file_path: str) -> bool:
    return file_path.endswith('.txt') and os.path.isfile(file_path)


def is_dir(file_path: str) -> bool:
    return os.path.isdir(file_path)


def is_glob(file_path: str) -> bool:
    return not (is_image(file_path) or is_dir(file_path))


def type_of_path(file_path: str) -> str:
    if is_image(file_path):
        return 'image'
    elif is_txt(file_path):
        return 'txt'
    elif is_dir(file_path):
        return 'dir'
    elif is_glob(file_path):
        return 'glob'
    else:
        return 'unknown'


def get_images(input_path_dir_globs: Union[str, List[str]]) -> List[str]:
    if isinstance(input_path_dir_globs, str):
        input_path_dir_globs = [input_path_dir_globs]
    images = []
    for path in input_path_dir_globs:
        match type_of_path(path):
            case 'image':
                images.append(path)
            case 'txt':
                with open(path, 'r') as f:
                    paths = [path.strip() for path in f.readlines() if len(path.strip()) > 0]
                images.extend(get_images(paths))
            case 'dir':
                # Filter to only include image files, not JSON or other files
                all_files = glob.glob(os.path.join(path, '*'))
                images.extend([f for f in all_files if is_image(f)])
            case 'glob':
                # Filter glob results to only include image files
                all_files = glob.glob(path)
                images.extend([f for f in all_files if is_image(f)])
            case _:
                raise ValueError(f"Unknown path type: {path}")
    if len(images) == 0:
        raise ValueError("No images found")
    return images


# =============================================================================
# Image Processing
# =============================================================================

class ImageEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Base64Image):
            return str(o)
        else:
            return super().default(o)


class Base64Image:
    def __init__(self, path: str):
        self.path = path
        self.bytes = base64.b64encode(open(path, "rb").read())
        self.str = self.bytes.decode("ascii")

    def __str__(self):
        return self.str

    def __bytes__(self):
        return self.bytes

    def __repr__(self) -> str:
        return f"Base64Image({self.path})"


def parse_image(
    images: Optional[Union[np.ndarray, bytes, str, List[Union[np.ndarray, bytes, str]]]],
    device: Union[torch.device, str] = "cpu"
):
    # Cases:
    # List: Recursively parse each image
    if isinstance(images, (list, tuple)):
        return [parse_image(image, device) for image in images]
    # String: Open the image with PIL
    elif isinstance(images, str):
        if re.search(re.compile(r"\.dng$", re.IGNORECASE), images):
            if rawpy is None:
                raise ImportError("rawpy is required to read DNG files. Install with: pip install rawpy")
            with rawpy.imread(images) as raw:
                images = raw.postprocess()
                images = Image.fromarray(images)
        else:
            images = Image.open(images)
    # Bytes: Open the image with PIL using a BytesIO
    elif isinstance(images, bytes):
        images = Image.open(io.BytesIO(images))
    # Numpy array: Do nothing
    elif isinstance(images, np.ndarray):
        pass
    # Other: Raise an error
    else:
        raise ValueError(f"Expected image(s) to be a np.ndarray, string or bytes, or list of these, but got {type(images)}")

    # Convert the image to a numpy array
    image = np.array(images)

    # Handle grayscale images (2D arrays) by converting to RGB (3D arrays)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.concatenate([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        # RGBA -> RGB: drop alpha channel
        image = image[:, :, :3]

    # Convert the image to a torch tensor and change from HWC to CHW
    return torch.from_numpy(image).permute(2, 0, 1).to(device)


# =============================================================================
# File Handling
# =============================================================================

def generate_uuid() -> str:
    return str(uuid.uuid4())[::3]


def save_file(
    content: str,
    name: str,
    dir: str,
    ext: str,
    identifier: Optional[str] = None,
    dtype: str = "text"
) -> str:
    # Is the data raw bytes or text?
    if "text" in dtype:
        dtype = ""
    elif "byte" in dtype:
        dtype = "b"
    # If the UUID is not specified, generate a new one
    if identifier is None:
        identifier = generate_uuid()
    # Construct the path
    path = f"{dir}/{identifier}_{name}.{ext}"
    # Dump the content to the file
    with open(path, f"w{dtype}") as f:
        f.write(content)
    # Return the path
    return path


def zip_files(
    files: List[str],
    name: str,
    dir: str,
    identifier: Optional[str] = None
) -> str:
    # If the UUID is not specified, generate a new one
    if identifier is None:
        identifier = generate_uuid()
    # Construct the path
    path = f"{dir}/{identifier}_{name}.zip"
    # Open the zip file
    with zipfile.ZipFile(path, "w") as z:
        # Add all the files
        for file in files:
            if isinstance(file, Base64Image):
                z.write(file.path)
            else:
                z.write(file)
    # Return the path
    return path


# =============================================================================
# Model Definition (Localizer)
# =============================================================================

class Localizer(Predictor):
    def predict(
        self,
        images: Optional[Union[np.ndarray, bytes, str, List[Union[np.ndarray, bytes, str]]]],
        do_plot: Union[bool, List[bool]] = False,
        include_crops: bool = False,
        outdir: str = "output"
    ) -> dict:
        # Initialize the data
        data = {
            "uuids": [],
            "predictions": [],
            "crops": [],
            "visualizations": []
        }

        if not isinstance(images, (list, tuple)):
            images = [images]
        if not isinstance(do_plot, list):
            if isinstance(do_plot, tuple):
                do_plot = list(do_plot)
            else:
                do_plot = [do_plot]
            if len(do_plot) == 1 and len(images) > 1:
                do_plot = do_plot * len(images)
        if not all([isinstance(plot, bool) for plot in do_plot]):
            raise ValueError(f"Expected do_plot to be a boolean or list of booleans, but got {do_plot}")

        for i, image in enumerate(tqdm(images, desc="Localizing insects", unit="image", leave=True)):
            if isinstance(image, str):
                image_identifier = os.path.splitext(os.path.basename(image))[0]
            else:
                image_identifier = "DUMMY"
            # Fetch the image
            image = parse_image(image, self._device)

            # Generate uuid for the image
            identifier = generate_uuid()

            # Create the output directory
            this_outdir = os.path.join(outdir, identifier)
            if not os.path.exists(this_outdir):
                os.makedirs(this_outdir)

            # Run the model
            predictions: TensorPredictions = self.pyramid_predictions(image, "DUMMY_PATH_STR", scale_before=1)

            # Save crops
            predictions.save_crops(outdir=this_outdir, basename=image_identifier, mask=True, identifier=identifier)
            crops = [Base64Image(crop) if include_crops else crop for crop in glob.glob(os.path.join(this_outdir, f"crop*{identifier}.png"))]

            # Plot the image if requested
            if do_plot[i]:
                visualization_dir = os.path.join(os.path.dirname(this_outdir), "visualization")
                if not os.path.exists(visualization_dir):
                    os.makedirs(visualization_dir)
                predict_image = os.path.join(visualization_dir, f'{identifier}_visualization.jpg')
                predictions.plot(outpath=predict_image, scale=1/2)
            else:
                predict_image = None

            # Append the data
            data["uuids"].append(identifier)
            data["visualizations"].append(predict_image)
            data["crops"].append(crops)
            data["predictions"].append(predictions.json_data)

        # Return the predictions as JSON
        return data


# =============================================================================
# COCO-style Output Generation
# =============================================================================

def predictions_to_coco_format(
    all_predictions: dict,
    image_paths: List[str],
    dataset_name: str
) -> dict:
    """
    Convert predictions to COCO-style format for comparison with ground truth.
    
    Args:
        all_predictions: Output from Localizer.predict() containing uuids, predictions, etc.
        image_paths: List of original image paths processed
        dataset_name: Name of the dataset being processed
    
    Returns:
        COCO-style annotations dict with images, annotations, and categories
    """
    coco_output = {
        "info": {
            "description": f"Flat-bug predictions for {dataset_name}",
            "date_created": datetime.now().isoformat(),
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "insect", "supercategory": "animal"}
        ]
    }
    
    annotation_id = 1
    
    for img_idx, (img_path, pred_data) in enumerate(zip(image_paths, all_predictions["predictions"])):
        # Image entry
        image_id = img_idx + 1
        file_name = os.path.basename(img_path)
        
        image_entry = {
            "id": image_id,
            "file_name": file_name,
            "width": pred_data.get("image_width", 0),
            "height": pred_data.get("image_height", 0),
            "original_path": img_path
        }
        coco_output["images"].append(image_entry)
        
        # Annotation entries for each detection in this image
        boxes = pred_data.get("boxes", [])
        contours = pred_data.get("contours", [])
        confs = pred_data.get("confs", [])
        classes = pred_data.get("classes", [])
        areas = pred_data.get("areas", [])
        scales = pred_data.get("scales", [])
        
        for det_idx in range(len(boxes)):
            # Convert box from [x1, y1, x2, y2] to COCO format [x, y, width, height]
            box = boxes[det_idx]
            x1, y1, x2, y2 = box
            bbox_coco = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
            
            # Get segmentation (contour points as polygon)
            # contours format: [[x1, x2, ...], [y1, y2, ...]] -> flatten to [x1, y1, x2, y2, ...]
            segmentation = []
            if det_idx < len(contours) and contours[det_idx]:
                contour = contours[det_idx]
                if len(contour) == 2 and len(contour[0]) > 0:
                    # Interleave x and y coordinates
                    xs, ys = contour[0], contour[1]
                    seg_points = []
                    for x, y in zip(xs, ys):
                        seg_points.extend([x, y])
                    if len(seg_points) >= 6:  # At least 3 points for valid polygon
                        segmentation = [seg_points]
            
            # Get area
            area = areas[det_idx] if det_idx < len(areas) else (bbox_coco[2] * bbox_coco[3])
            
            # Get confidence score
            score = confs[det_idx] if det_idx < len(confs) else 1.0
            
            # Get class (default to 1 for insect)
            category_id = classes[det_idx] if det_idx < len(classes) else 1
            
            # Get scale
            scale = scales[det_idx] if det_idx < len(scales) else 1.0
            
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": bbox_coco,
                "segmentation": segmentation,
                "area": float(area),
                "score": float(score),
                "scale": float(scale),
                "iscrowd": 0
            }
            coco_output["annotations"].append(annotation_entry)
            annotation_id += 1
    
    return coco_output


# =============================================================================
# Main Processing Functions
# =============================================================================

def get_defaults():
    """Define the model parameters."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    return device, dtype


def process_dataset(
    dataset_name: str,
    dataset_path: str,
    output_root: str,
    model: Localizer,
    do_plot: bool = True,
    include_crops: bool = False
) -> dict:
    """Process a single dataset folder."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"{'='*60}")

    # Get all images in the dataset
    try:
        image_paths = get_images(dataset_path)
    except ValueError as e:
        print(f"No images found in {dataset_path}: {e}")
        return {"dataset": dataset_name, "num_images": 0, "status": "no_images"}

    print(f"Found {len(image_paths)} images")

    # Create output directory for this dataset
    outdir = os.path.join(output_root, dataset_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Run the model
    output = model.predict(
        image_paths,
        do_plot=do_plot,
        include_crops=include_crops,
        outdir=outdir
    )

    # Save the combined output JSON (raw format)
    output_json_path = save_file(
        json.dumps(output, cls=ImageEncoder, indent=2),
        name="instances",
        dir=outdir,
        ext="json",
        dtype="text"
    )

    # Save COCO-style predictions JSON for comparison with ground truth
    coco_predictions = predictions_to_coco_format(output, image_paths, dataset_name)
    coco_json_path = os.path.join(outdir, "flatbug_predictions.json")
    with open(coco_json_path, "w") as f:
        json.dump(coco_predictions, f, indent=2)

    print(f"Results saved to {os.path.abspath(outdir)}")
    print(f"Raw output JSON: {output_json_path}")
    print(f"COCO predictions JSON: {coco_json_path}")

    # Count total detections
    total_detections = sum(len(pred.get("boxes", [])) for pred in output["predictions"])

    return {
        "dataset": dataset_name,
        "num_images": len(image_paths),
        "num_predictions": len(output["predictions"]),
        "total_detections": total_detections,
        "output_dir": outdir,
        "output_json": output_json_path,
        "coco_json": coco_json_path,
        "status": "success"
    }


def main(args):
    """Main function to process all datasets."""
    print(f"\n{'#'*60}")
    print(f"# Flat-bug Dataset Inference Script")
    print(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

    # Get device and dtype
    device, dtype = get_defaults()
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    # Create the model
    print(f"\nLoading model: {args.model}")
    print(f"args: {args}")
    model = Localizer(model=args.model, device=device, dtype=dtype)
    model.set_hyperparameters(
        SCORE_THRESHOLD=args.score_threshold,
        IOU_THRESHOLD=args.iou_threshold,
        MINIMUM_TILE_OVERLAP=args.minimum_tile_overlap,
        EDGE_CASE_MARGIN=args.edge_case_margin,
        MIN_MAX_OBJ_SIZE=(args.min_obj_size, args.max_obj_size),
        TIME=args.time
    )
    print("Model loaded successfully!")

    # Create output root directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Get list of datasets to process
    if args.datasets:
        # Use specified datasets
        datasets_to_process = set(args.datasets) & ALLOWED_FOLDERS
    else:
        # Use all allowed folders
        datasets_to_process = ALLOWED_FOLDERS

    print(f"\nDatasets to process: {len(datasets_to_process)}")
    for ds in sorted(datasets_to_process):
        print(f"  - {ds}")

    # Process each dataset
    results = []
    for dataset_name in sorted(datasets_to_process):
        dataset_path = os.path.join(args.input, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"\nWarning: Dataset path does not exist: {dataset_path}")
            results.append({
                "dataset": dataset_name,
                "status": "not_found",
                "path": dataset_path
            })
            continue

        try:
            result = process_dataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                output_root=args.output,
                model=model,
                do_plot=args.plot,
                include_crops=args.crops
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            traceback.print_exc()
            results.append({
                "dataset": dataset_name,
                "status": "error",
                "error": str(e)
            })

    # Save summary
    summary_path = os.path.join(args.output, "processing_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": args.model,
                "score_threshold": args.score_threshold,
                "iou_threshold": args.iou_threshold,
                "minimum_tile_overlap": args.minimum_tile_overlap,
                "edge_case_margin": args.edge_case_margin,
                "min_obj_size": args.min_obj_size,
                "max_obj_size": args.max_obj_size,
                "device": str(device),
                "dtype": str(dtype)
            },
            "results": results
        }, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"# Processing Complete")
    print(f"# Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Summary saved to: {summary_path}")
    print(f"{'#'*60}\n")

    # Print summary
    print("\nSummary:")
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    not_found = sum(1 for r in results if r.get("status") == "not_found")
    no_images = sum(1 for r in results if r.get("status") == "no_images")
    
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Not Found: {not_found}")
    print(f"  No Images: {no_images}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run flat-bug inference on multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all datasets
    python flatbug_inference.py -i /path/to/datasets -o /path/to/output

    # Process specific datasets
    python flatbug_inference.py -i /path/to/datasets -o /path/to/output --datasets ALUS BIOSCAN

    # Run with custom thresholds
    python flatbug_inference.py -i /path/to/datasets -o /path/to/output --score-threshold 0.3

    # Run in background with logging
    nohup python3 -u flatbug_inference.py -i /data -o /output > inference.log 2>&1 &
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing the dataset folders"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for results"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="flat_bug_L.pt",
        help="Model weights file (default: flat_bug_L.pt)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Specific datasets to process (default: all allowed folders)"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="Score threshold for predictions (default: 0.25)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.2,
        help="IOU threshold for NMS (default: 0.2)"
    )
    parser.add_argument(
        "--minimum-tile-overlap",
        type=int,
        default=384,
        help="Minimum overlap between tiles when splitting the image (default: 384)"
    )
    parser.add_argument(
        "--edge-case-margin",
        type=int,
        default=16,
        help="Edge case margin in pixels (default: 16)"
    )
    parser.add_argument(
        "--min-obj-size",
        type=int,
        default=32,
        help="Minimum object size (default: 32)"
    )
    parser.add_argument(
        "--max-obj-size",
        type=int,
        default=1024,
        help="Maximum object size (default: 1024)" 
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate visualization plots (default: True)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Disable visualization plots"
    )
    parser.add_argument(
        "--crops",
        action="store_true",
        default=False,
        help="Include base64-encoded crops in output"
    )
    parser.add_argument(
        "--time",
        action="store_true",
        default=False,
        help="Enable timing output"
    )

    args = parser.parse_args()
    main(args)
