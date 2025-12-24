"""
02_image_analysis.py - Image Property Analysis

Analyzes image properties:
- Image dimensions (HR and LR)
- Bit depth and data types
- Pixel value distributions
- Band information

Usage:
    python phase1_data_exploration/02_image_analysis.py
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

try:
    import tifffile
except ImportError:
    print("Installing tifffile...")
    os.system("pip install tifffile")
    import tifffile

try:
    from PIL import Image
except ImportError:
    print("Installing Pillow...")
    os.system("pip install Pillow")
    from PIL import Image

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
HR_DIR = DATASET_DIR / "hr_dataset" / "12bit"
LR_DIR = DATASET_DIR / "lr_dataset"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Number of samples to analyze (for speed)
NUM_SAMPLES = 50


def analyze_tiff(filepath):
    """Analyze a single TIFF file."""
    try:
        img = tifffile.imread(filepath)
        return {
            "shape": img.shape,
            "dtype": str(img.dtype),
            "min": float(np.min(img)),
            "max": float(np.max(img)),
            "mean": float(np.mean(img)),
            "std": float(np.std(img))
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_hr_images(sample_size=NUM_SAMPLES):
    """Analyze HR image properties."""
    print("  Scanning HR images...")
    
    results = {
        "pan": [],
        "ps": [],
        "rgbn": [],
        "rgb": []
    }
    
    locations = [d for d in HR_DIR.iterdir() if d.is_dir()]
    samples = random.sample(locations, min(sample_size, len(locations)))
    
    for loc_dir in tqdm(samples, desc="  Analyzing HR"):
        # Analyze each file type
        for suffix, key in [("_pan.tiff", "pan"), ("_ps.tiff", "ps"), 
                            ("_rgbn.tiff", "rgbn"), ("_rgb.png", "rgb")]:
            filepath = loc_dir / f"{loc_dir.name}{suffix}"
            if filepath.exists():
                if suffix.endswith(".tiff"):
                    results[key].append(analyze_tiff(filepath))
                else:
                    # PNG file
                    try:
                        img = Image.open(filepath)
                        results[key].append({
                            "shape": (img.height, img.width, len(img.getbands())),
                            "mode": img.mode
                        })
                    except Exception as e:
                        results[key].append({"error": str(e)})
    
    return results


def analyze_lr_images(sample_size=NUM_SAMPLES):
    """Analyze LR image properties."""
    print("  Scanning LR images...")
    
    results = {
        "L2A_data": [],
        "CLM": [],
        "CLP": [],
        "dataMask": []
    }
    
    locations = [d for d in LR_DIR.iterdir() if d.is_dir()]
    samples = random.sample(locations, min(sample_size, len(locations)))
    
    for loc_dir in tqdm(samples, desc="  Analyzing LR"):
        l2a_dir = loc_dir / "L2A"
        if not l2a_dir.exists():
            continue
        
        # Find first acquisition
        data_files = list(l2a_dir.glob("*-1-L2A_data.tiff"))
        if data_files:
            results["L2A_data"].append(analyze_tiff(data_files[0]))
        
        clm_files = list(l2a_dir.glob("*-1-CLM.tiff"))
        if clm_files:
            results["CLM"].append(analyze_tiff(clm_files[0]))
        
        clp_files = list(l2a_dir.glob("*-1-CLP.tiff"))
        if clp_files:
            results["CLP"].append(analyze_tiff(clp_files[0]))
        
        mask_files = list(l2a_dir.glob("*-1-dataMask.tiff"))
        if mask_files:
            results["dataMask"].append(analyze_tiff(mask_files[0]))
    
    return results


def summarize_analysis(analysis_list):
    """Summarize analysis results for a file type."""
    if not analysis_list:
        return {"count": 0}
    
    # Filter out errors
    valid = [a for a in analysis_list if "error" not in a]
    errors = [a for a in analysis_list if "error" in a]
    
    if not valid:
        return {"count": 0, "errors": len(errors)}
    
    shapes = [a["shape"] for a in valid if "shape" in a]
    
    summary = {
        "count": len(valid),
        "errors": len(errors)
    }
    
    if shapes:
        # Get unique shapes
        unique_shapes = list(set([str(s) for s in shapes]))
        summary["shapes"] = unique_shapes[:5]  # First 5 unique shapes
        summary["shape_count"] = len(unique_shapes)
    
    if "dtype" in valid[0]:
        dtypes = list(set([a["dtype"] for a in valid]))
        summary["dtypes"] = dtypes
    
    if "min" in valid[0]:
        summary["value_range"] = {
            "min": min([a["min"] for a in valid]),
            "max": max([a["max"] for a in valid]),
            "mean_of_means": round(np.mean([a["mean"] for a in valid]), 2)
        }
    
    return summary


def main():
    print("=" * 60)
    print("WorldStrat Image Analysis")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Analyze HR images
    print(f"\n[1/2] Analyzing HR images (sampling {NUM_SAMPLES} locations)...")
    hr_analysis = analyze_hr_images()
    
    # Analyze LR images
    print(f"\n[2/2] Analyzing LR images (sampling {NUM_SAMPLES} locations)...")
    lr_analysis = analyze_lr_images()
    
    # Summarize results
    results = {
        "hr_images": {
            "pan": summarize_analysis(hr_analysis["pan"]),
            "ps": summarize_analysis(hr_analysis["ps"]),
            "rgbn": summarize_analysis(hr_analysis["rgbn"]),
            "rgb": summarize_analysis(hr_analysis["rgb"])
        },
        "lr_images": {
            "L2A_data": summarize_analysis(lr_analysis["L2A_data"]),
            "CLM": summarize_analysis(lr_analysis["CLM"]),
            "CLP": summarize_analysis(lr_analysis["CLP"]),
            "dataMask": summarize_analysis(lr_analysis["dataMask"])
        },
        "samples_analyzed": NUM_SAMPLES
    }
    
    # Save results
    output_file = OUTPUT_DIR / "image_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}")
    
    # Print summary
    print("\n📊 HR IMAGE SUMMARY")
    for key, val in results["hr_images"].items():
        print(f"  {key}: {val.get('shapes', ['N/A'])[:2]}, dtype={val.get('dtypes', ['N/A'])}")
    
    print("\n📊 LR IMAGE SUMMARY")
    for key, val in results["lr_images"].items():
        print(f"  {key}: {val.get('shapes', ['N/A'])[:2]}, dtype={val.get('dtypes', ['N/A'])}")
    
    return results


if __name__ == "__main__":
    main()

