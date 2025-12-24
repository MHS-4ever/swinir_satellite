"""
05_create_splits.py - Create Train/Val/Test Splits

Creates reproducible data splits:
- Train: 80%
- Validation: 10%
- Test: 10%

Stratified by land cover class if available.

Usage:
    python phase1_data_exploration/05_create_splits.py
"""

import json
import random
from pathlib import Path
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
HR_DIR = DATASET_DIR / "hr_dataset" / "12bit"
LR_DIR = DATASET_DIR / "lr_dataset"
METADATA_FILE = DATASET_DIR / "metadata.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "splits"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42


def get_matched_locations():
    """Get locations that have both HR and LR data."""
    hr_locations = set(d.name for d in HR_DIR.iterdir() if d.is_dir())
    lr_locations = set(d.name for d in LR_DIR.iterdir() if d.is_dir())
    matched = sorted(list(hr_locations & lr_locations))
    return matched


def create_random_splits(locations):
    """Create random train/val/test splits."""
    random.seed(RANDOM_SEED)
    
    # Shuffle locations
    shuffled = locations.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    return train, val, test


def create_stratified_splits(locations, metadata_df):
    """Create stratified splits based on land cover class."""
    random.seed(RANDOM_SEED)
    
    # Get unique locations from metadata
    location_col = metadata_df.columns[0]
    
    # Get land cover class for each location
    if 'IPCC Class' in metadata_df.columns:
        class_col = 'IPCC Class'
    else:
        # Fall back to random splits
        return create_random_splits(locations)
    
    # Create location -> class mapping
    loc_class = metadata_df.groupby(location_col)[class_col].first().to_dict()
    
    # Filter to matched locations only
    loc_class = {loc: cls for loc, cls in loc_class.items() if loc in locations}
    
    # Group locations by class
    class_locations = {}
    for loc, cls in loc_class.items():
        if cls not in class_locations:
            class_locations[cls] = []
        class_locations[cls].append(loc)
    
    # Split each class proportionally
    train, val, test = [], [], []
    
    for cls, locs in class_locations.items():
        random.shuffle(locs)
        n = len(locs)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        
        train.extend(locs[:train_end])
        val.extend(locs[train_end:val_end])
        test.extend(locs[val_end:])
    
    # Handle locations not in metadata
    missing = set(locations) - set(loc_class.keys())
    if missing:
        missing = list(missing)
        random.shuffle(missing)
        n = len(missing)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        
        train.extend(missing[:train_end])
        val.extend(missing[train_end:val_end])
        test.extend(missing[val_end:])
    
    # Shuffle final splits
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def save_split(locations, filepath):
    """Save split to text file."""
    with open(filepath, 'w') as f:
        for loc in sorted(locations):
            f.write(f"{loc}\n")


def main():
    print("=" * 60)
    print("WorldStrat Data Split Creation")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get matched locations
    print("\n[1/3] Finding matched locations...")
    locations = get_matched_locations()
    print(f"  - Total matched locations: {len(locations)}")
    
    # Load metadata for stratified splitting
    print("\n[2/3] Creating splits...")
    if METADATA_FILE.exists():
        metadata_df = pd.read_csv(METADATA_FILE)
        train, val, test = create_stratified_splits(locations, metadata_df)
        split_type = "stratified"
    else:
        train, val, test = create_random_splits(locations)
        split_type = "random"
    
    print(f"  - Split type: {split_type}")
    print(f"  - Train: {len(train)} ({len(train)/len(locations)*100:.1f}%)")
    print(f"  - Val: {len(val)} ({len(val)/len(locations)*100:.1f}%)")
    print(f"  - Test: {len(test)} ({len(test)/len(locations)*100:.1f}%)")
    
    # Save splits
    print("\n[3/3] Saving split files...")
    save_split(train, OUTPUT_DIR / "train.txt")
    save_split(val, OUTPUT_DIR / "val.txt")
    save_split(test, OUTPUT_DIR / "test.txt")
    
    # Save split info
    split_info = {
        "total_locations": len(locations),
        "split_type": split_type,
        "random_seed": RANDOM_SEED,
        "splits": {
            "train": {"count": len(train), "ratio": TRAIN_RATIO},
            "val": {"count": len(val), "ratio": VAL_RATIO},
            "test": {"count": len(test), "ratio": TEST_RATIO}
        }
    }
    
    with open(OUTPUT_DIR / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Splits saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    
    print("\n📄 Output Files:")
    print(f"  - train.txt ({len(train)} locations)")
    print(f"  - val.txt ({len(val)} locations)")
    print(f"  - test.txt ({len(test)} locations)")
    print(f"  - split_info.json")


if __name__ == "__main__":
    main()

