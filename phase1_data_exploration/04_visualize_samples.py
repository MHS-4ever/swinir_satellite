"""
04_visualize_samples.py - Sample Visualization

Creates visualizations of HR-LR pairs:
- Side-by-side comparisons
- Band visualizations
- Cloud mask overlays

Usage:
    python phase1_data_exploration/04_visualize_samples.py
"""

import os
import random
from pathlib import Path
import numpy as np

try:
    import tifffile
except ImportError:
    os.system("pip install tifffile")
    import tifffile

try:
    import matplotlib.pyplot as plt
except ImportError:
    os.system("pip install matplotlib")
    import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    os.system("pip install Pillow")
    from PIL import Image

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
HR_DIR = DATASET_DIR / "hr_dataset" / "12bit"
LR_DIR = DATASET_DIR / "lr_dataset"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "visualizations"

# Number of samples to visualize
NUM_SAMPLES = 5


def normalize_for_display(img, percentile=2):
    """Normalize image for display using percentile clipping."""
    if img.dtype == np.uint16 or img.max() > 255:
        # 12-bit or 16-bit image
        low = np.percentile(img, percentile)
        high = np.percentile(img, 100 - percentile)
        img = np.clip(img, low, high)
        img = ((img - low) / (high - low) * 255).astype(np.uint8)
    return img


def visualize_hr_lr_pair(location_name, output_path):
    """Visualize a single HR-LR pair."""
    hr_dir = HR_DIR / location_name
    lr_dir = LR_DIR / location_name / "L2A"
    
    if not hr_dir.exists() or not lr_dir.exists():
        print(f"  Skipping {location_name}: directories not found")
        return False
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Location: {location_name}", fontsize=14)
    
    # HR images
    # RGB preview
    rgb_path = hr_dir / f"{location_name}_rgb.png"
    if rgb_path.exists():
        rgb_img = Image.open(rgb_path)
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title(f"HR RGB Preview\n{rgb_img.size}")
    axes[0, 0].axis('off')
    
    # Pan-sharpened
    ps_path = hr_dir / f"{location_name}_ps.tiff"
    if ps_path.exists():
        ps_img = tifffile.imread(ps_path)
        if len(ps_img.shape) == 3:
            # Take first 3 bands as RGB
            display_img = normalize_for_display(ps_img[:, :, :3])
            axes[0, 1].imshow(display_img)
        else:
            axes[0, 1].imshow(normalize_for_display(ps_img), cmap='gray')
        axes[0, 1].set_title(f"HR Pan-Sharpened\n{ps_img.shape}, {ps_img.dtype}")
    axes[0, 1].axis('off')
    
    # RGBN (4 bands)
    rgbn_path = hr_dir / f"{location_name}_rgbn.tiff"
    if rgbn_path.exists():
        rgbn_img = tifffile.imread(rgbn_path)
        if len(rgbn_img.shape) == 3 and rgbn_img.shape[2] >= 3:
            display_img = normalize_for_display(rgbn_img[:, :, :3])
            axes[0, 2].imshow(display_img)
        axes[0, 2].set_title(f"HR RGBN\n{rgbn_img.shape}, {rgbn_img.dtype}")
    axes[0, 2].axis('off')
    
    # LR images (first acquisition)
    lr_data_files = list(lr_dir.glob("*-1-L2A_data.tiff"))
    if lr_data_files:
        lr_img = tifffile.imread(lr_data_files[0])
        # Sentinel-2 bands: typically B2,B3,B4 are RGB
        if len(lr_img.shape) == 3:
            # Assume bands are in order, take first 3 as RGB
            if lr_img.shape[0] < lr_img.shape[2]:
                # Bands first format
                lr_rgb = np.transpose(lr_img[:3], (1, 2, 0))
            else:
                lr_rgb = lr_img[:, :, :3]
            display_img = normalize_for_display(lr_rgb)
            axes[1, 0].imshow(display_img)
        else:
            axes[1, 0].imshow(normalize_for_display(lr_img), cmap='gray')
        axes[1, 0].set_title(f"LR Sentinel-2\n{lr_img.shape}, {lr_img.dtype}")
    axes[1, 0].axis('off')
    
    # Cloud probability
    clp_files = list(lr_dir.glob("*-1-CLP.tiff"))
    if clp_files:
        clp_img = tifffile.imread(clp_files[0])
        axes[1, 1].imshow(clp_img, cmap='Reds', vmin=0, vmax=100)
        axes[1, 1].set_title(f"Cloud Probability\n{clp_img.shape}")
    axes[1, 1].axis('off')
    
    # Cloud mask
    clm_files = list(lr_dir.glob("*-1-CLM.tiff"))
    if clm_files:
        clm_img = tifffile.imread(clm_files[0])
        axes[1, 2].imshow(clm_img, cmap='binary')
        axes[1, 2].set_title(f"Cloud Mask\n{clm_img.shape}")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def main():
    print("=" * 60)
    print("WorldStrat Sample Visualization")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    random.seed(42)
    
    # Get matching locations
    print("\n[1/2] Finding matched HR-LR pairs...")
    hr_locations = set(d.name for d in HR_DIR.iterdir() if d.is_dir())
    lr_locations = set(d.name for d in LR_DIR.iterdir() if d.is_dir())
    matched = list(hr_locations & lr_locations)
    print(f"  - Found {len(matched)} matched pairs")
    
    # Sample locations
    samples = random.sample(matched, min(NUM_SAMPLES, len(matched)))
    
    # Visualize each sample
    print(f"\n[2/2] Creating visualizations for {len(samples)} samples...")
    for i, loc in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {loc}")
        output_path = OUTPUT_DIR / f"sample_{i}_{loc.replace(' ', '_')}.png"
        visualize_hr_lr_pair(loc, output_path)
    
    print(f"\n{'=' * 60}")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

