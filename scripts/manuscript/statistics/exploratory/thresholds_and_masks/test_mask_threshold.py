#!/usr/bin/env python3
"""
Diagnostic script to test whether MASK_THRESHOLD actually changes SAM3 mask output.

This script:
1. Runs SAM3 inference on a few test images
2. Tests different mask thresholds (0.25, 0.35, 0.5, 0.65)
3. Reports mask area statistics for each threshold
4. Verifies the mask logit distribution

Usage:
    python3 scripts/manuscript/statistics/test_mask_threshold.py

conda run -n sam3_gpu pip install opencv-python-headless -q && cd /home/dolma/repo/flat-bug && conda run -n sam3_gpu python scripts/manuscript/statistics/test_mask_threshold.py 2>&1
    
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2

# Check for SAM3 availability
SAM3_REPO = os.path.expanduser("~/repo/sam3-insect-segmentation")
if os.path.exists(SAM3_REPO):
    sys.path.insert(0, SAM3_REPO)
    HAS_SAM3 = True
else:
    HAS_SAM3 = False
    print(f"WARNING: SAM3 repo not found at {SAM3_REPO}")

# Test images
TEST_IMAGES = [
    "fb_yolo/insects/images/val/cao2022_000002.jpg",
    "fb_yolo/insects/images/val/cao2022_000003.jpg",
    "fb_yolo/insects/images/val/cao2022_000026.jpg",
]

THRESHOLDS = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.5, 0.65]


def analyze_mask_logits(logits: np.ndarray, thresholds: list) -> dict:
    """Analyze mask logit distribution and compute areas at different thresholds."""
    results = {
        "logit_min": float(np.nanmin(logits)),
        "logit_max": float(np.nanmax(logits)),
        "logit_mean": float(np.nanmean(logits)),
        "logit_median": float(np.nanmedian(logits)),
        "logit_std": float(np.nanstd(logits)),
        "threshold_areas": {},
        "total_pixels": int(logits.size),
    }
    
    for thresh in thresholds:
        mask_binary = (logits > thresh).astype(np.uint8)
        area = int(mask_binary.sum())
        pct = 100 * area / logits.size
        results["threshold_areas"][thresh] = {
            "area_pixels": area,
            "area_pct": pct
        }
    
    return results


def test_without_sam3():
    """Test analysis functions without SAM3 model."""
    print("\n" + "="*70)
    print("Testing mask_threshold effect using synthetic logits")
    print("="*70)
    
    # Create synthetic mask logits similar to SAM3 output
    # SAM3 outputs logits roughly in range [-10, 10] (before sigmoid)
    # After sigmoid: values close to 0 or 1
    
    np.random.seed(42)
    
    # Simulate a mask with confident predictions (values far from 0)
    # This mimics SAM3's typically high-confidence masks
    h, w = 256, 256
    
    # Create a "confident" mask (values far from threshold)
    confident_logits = np.zeros((h, w), dtype=np.float32)
    # Inner region: high confidence positive (insect)
    confident_logits[80:180, 80:180] = 5.0  # sigmoid(5) ≈ 0.993
    # Outer region: high confidence negative (background)
    confident_logits[:80, :] = -5.0  # sigmoid(-5) ≈ 0.007
    confident_logits[180:, :] = -5.0
    confident_logits[:, :80] = -5.0
    confident_logits[:, 180:] = -5.0
    
    # Add some noise
    confident_logits += np.random.randn(h, w) * 0.5
    
    # Convert to sigmoid probabilities
    confident_probs = 1 / (1 + np.exp(-confident_logits))
    
    print("\n--- Scenario 1: Confident SAM3-like masks ---")
    results = analyze_mask_logits(confident_probs, THRESHOLDS)
    print(f"Logit stats: min={results['logit_min']:.3f}, max={results['logit_max']:.3f}, "
          f"mean={results['logit_mean']:.3f}, median={results['logit_median']:.3f}")
    print("\nMask areas at different thresholds:")
    for thresh, data in results["threshold_areas"].items():
        print(f"  threshold={thresh:.2f}: {data['area_pixels']:>6d} pixels ({data['area_pct']:.1f}%)")
    
    area_25 = results["threshold_areas"][0.25]["area_pixels"]
    area_50 = results["threshold_areas"][0.50]["area_pixels"]
    area_diff = area_25 - area_50
    print(f"\nArea difference (0.25 vs 0.50): {area_diff:+d} pixels ({100*area_diff/area_50:+.1f}%)")
    
    # Create an "uncertain" mask with values closer to threshold
    print("\n--- Scenario 2: Uncertain masks (boundary regions) ---")
    uncertain_logits = np.zeros((h, w), dtype=np.float32)
    # Gradient from center outward (simulates uncertain boundaries)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r = np.sqrt(x**2 + y**2)
    # Values transition smoothly from 1.0 (center) to -1.0 (edge)
    uncertain_logits = 2.0 - 4.0 * r  # Range roughly [-2, 2]
    uncertain_logits = np.clip(uncertain_logits, -3, 3)
    uncertain_logits += np.random.randn(h, w) * 0.3
    
    uncertain_probs = 1 / (1 + np.exp(-uncertain_logits))
    
    results = analyze_mask_logits(uncertain_probs, THRESHOLDS)
    print(f"Logit stats: min={results['logit_min']:.3f}, max={results['logit_max']:.3f}, "
          f"mean={results['logit_mean']:.3f}, median={results['logit_median']:.3f}")
    print("\nMask areas at different thresholds:")
    for thresh, data in results["threshold_areas"].items():
        print(f"  threshold={thresh:.2f}: {data['area_pixels']:>6d} pixels ({data['area_pct']:.1f}%)")
    
    area_25 = results["threshold_areas"][0.25]["area_pixels"]
    area_50 = results["threshold_areas"][0.50]["area_pixels"]
    area_diff = area_25 - area_50
    print(f"\nArea difference (0.25 vs 0.50): {area_diff:+d} pixels ({100*area_diff/area_50:+.1f}%)")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("""
If SAM3 produces CONFIDENT masks (values far from 0.5), then changing
MASK_THRESHOLD from 0.35 to 0.5 will have MINIMAL effect on mask area.

This appears to be the case based on your results - the masks are nearly
identical because SAM3's logits are typically very confident (close to 0 or 1
after sigmoid), so the threshold doesn't cross any pixel values.

To verify this hypothesis, we need to examine SAM3's actual mask logit
distribution on the FlatBug dataset.
""")


def test_with_sam3():
    """Test with actual SAM3 model."""
    print("\n" + "="*70)
    print("Testing mask_threshold effect with REAL SAM3 inference")
    print("="*70)
    
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import torch
    except ImportError as e:
        print(f"Cannot import SAM3: {e}")
        return
    
    # Check device - force CPU if CUDA is incompatible with the GPU
    device = "cpu"
    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works
            torch.zeros(1, device="cuda")
            device = "cuda:0"
        except RuntimeError:
            print("CUDA available but incompatible with GPU, using CPU")
            device = "cpu"
    print(f"Using device: {device}")
    
    # Load model
    bpe_path = os.path.join(SAM3_REPO, "assets/bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(bpe_path):
        print(f"BPE file not found: {bpe_path}")
        return
    
    print("Loading SAM3 model...")
    model = build_sam3_image_model(bpe_path=bpe_path)
    model.to(device)
    model.eval()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.2)
    print("Model loaded.\n")
    
    # Find test images
    base_dir = "/home/dolma/repo/flat-bug"
    available_images = []
    for img_path in TEST_IMAGES:
        full_path = os.path.join(base_dir, img_path)
        if os.path.exists(full_path):
            available_images.append(full_path)
    
    if not available_images:
        print("No test images found!")
        return
    
    print(f"Testing on {len(available_images)} images...")
    
    all_logit_stats = []
    
    for img_path in available_images:
        print(f"\n--- {os.path.basename(img_path)} ---")
        
        image = Image.open(img_path).convert("RGB")
        
        # Run inference
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.inference_mode():
                state = processor.set_image(image)
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt("insects", state)
        
        # Get raw logits
        masks_logits = state.get("masks_logits", state.get("masks", []))
        boxes = state.get("boxes", [])
        
        if len(boxes) == 0:
            print("  No detections")
            continue
        
        # Analyze each mask
        m = masks_logits.float().detach().cpu().numpy()
        if m.ndim == 4:
            m = m.squeeze(1)
        
        print(f"  Found {len(m)} detections")
        
        for i, mask_logits in enumerate(m[:3]):  # First 3 detections
            results = analyze_mask_logits(mask_logits, THRESHOLDS)
            
            print(f"\n  Detection {i+1}:")
            print(f"    Logit stats: min={results['logit_min']:.3f}, max={results['logit_max']:.3f}, "
                  f"mean={results['logit_mean']:.3f}")
            print(f"    Threshold areas:")
            for thresh, data in results["threshold_areas"].items():
                print(f"      {thresh:.2f}: {data['area_pixels']:>6d} px ({data['area_pct']:.1f}%)")
            
            all_logit_stats.append(results)
    
    # Summary
    if all_logit_stats:
        print("\n" + "="*70)
        print("SUMMARY: Effect of threshold on mask area")
        print("="*70)
        
        for thresh in THRESHOLDS:
            areas = [s["threshold_areas"][thresh]["area_pixels"] for s in all_logit_stats]
            print(f"  threshold={thresh:.2f}: mean area = {np.mean(areas):.0f} px")
        
        # Compare 0.35 vs 0.50
        areas_035 = [s["threshold_areas"][0.35]["area_pixels"] for s in all_logit_stats]
        areas_050 = [s["threshold_areas"][0.50]["area_pixels"] for s in all_logit_stats]
        area_diffs = [a35 - a50 for a35, a50 in zip(areas_035, areas_050)]
        pct_diffs = [100 * (a35 - a50) / a50 if a50 > 0 else 0 for a35, a50 in zip(areas_035, areas_050)]
        
        print(f"\n  Threshold 0.35 vs 0.50:")
        print(f"    Mean area increase: {np.mean(area_diffs):.0f} px ({np.mean(pct_diffs):.1f}%)")
        print(f"    Max area increase: {np.max(area_diffs):.0f} px ({np.max(pct_diffs):.1f}%)")


def main():
    print("="*70)
    print("MASK_THRESHOLD DIAGNOSTIC TEST")
    print("="*70)
    
    # Always run synthetic test first
    test_without_sam3()
    
    # Try real SAM3 test if available
    if HAS_SAM3:
        test_with_sam3()
    else:
        print("\nTo test with real SAM3 inference, ensure the SAM3 repo is available at:")
        print(f"  {SAM3_REPO}")


if __name__ == "__main__":
    main()
