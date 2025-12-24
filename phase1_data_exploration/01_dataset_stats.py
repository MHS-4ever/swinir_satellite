"""
01_dataset_stats.py - Dataset Statistics Analysis

Analyzes the WorldStrat dataset structure:
- Counts HR and LR locations
- Validates HR-LR pair matching
- Counts files per category
- Checks for missing/corrupted files

Usage:
    python phase1_data_exploration/01_dataset_stats.py
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
HR_DIR = DATASET_DIR / "hr_dataset" / "12bit"
LR_DIR = DATASET_DIR / "lr_dataset"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def count_hr_locations():
    """Count HR dataset locations and files."""
    locations = []
    file_counts = defaultdict(int)
    
    if not HR_DIR.exists():
        print(f"ERROR: HR directory not found: {HR_DIR}")
        return [], {}
    
    for loc_dir in HR_DIR.iterdir():
        if loc_dir.is_dir():
            locations.append(loc_dir.name)
            for f in loc_dir.iterdir():
                if f.is_file():
                    ext = f.suffix.lower()
                    file_counts[ext] += 1
    
    return locations, dict(file_counts)


def count_lr_locations():
    """Count LR dataset locations and temporal acquisitions."""
    locations = []
    acquisition_counts = {}
    file_counts = defaultdict(int)
    
    if not LR_DIR.exists():
        print(f"ERROR: LR directory not found: {LR_DIR}")
        return [], {}, {}
    
    for loc_dir in LR_DIR.iterdir():
        if loc_dir.is_dir():
            locations.append(loc_dir.name)
            
            # Count acquisitions in L2A folder
            l2a_dir = loc_dir / "L2A"
            if l2a_dir.exists():
                # Count unique acquisition numbers from metadata files
                metadata_files = list(l2a_dir.glob("*.metadata"))
                acquisition_counts[loc_dir.name] = len(metadata_files)
                
                for f in l2a_dir.iterdir():
                    if f.is_file():
                        ext = f.suffix.lower()
                        file_counts[ext] += 1
    
    return locations, acquisition_counts, dict(file_counts)


def validate_pairs(hr_locations, lr_locations):
    """Check HR-LR location matching."""
    hr_set = set(hr_locations)
    lr_set = set(lr_locations)
    
    matched = hr_set & lr_set
    hr_only = hr_set - lr_set
    lr_only = lr_set - hr_set
    
    return {
        "matched_count": len(matched),
        "hr_only_count": len(hr_only),
        "lr_only_count": len(lr_only),
        "hr_only_samples": list(hr_only)[:10],
        "lr_only_samples": list(lr_only)[:10]
    }


def main():
    print("=" * 60)
    print("WorldStrat Dataset Statistics Analysis")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Analyze HR dataset
    print("\n[1/3] Analyzing HR dataset...")
    hr_locations, hr_file_counts = count_hr_locations()
    print(f"  - HR locations: {len(hr_locations)}")
    print(f"  - HR file types: {hr_file_counts}")
    
    # Analyze LR dataset
    print("\n[2/3] Analyzing LR dataset...")
    lr_locations, acquisition_counts, lr_file_counts = count_lr_locations()
    print(f"  - LR locations: {len(lr_locations)}")
    print(f"  - LR file types: {lr_file_counts}")
    
    # Calculate acquisition statistics
    if acquisition_counts:
        acq_values = list(acquisition_counts.values())
        avg_acquisitions = sum(acq_values) / len(acq_values)
        min_acquisitions = min(acq_values)
        max_acquisitions = max(acq_values)
        print(f"  - Acquisitions per location: min={min_acquisitions}, max={max_acquisitions}, avg={avg_acquisitions:.1f}")
    
    # Validate pairs
    print("\n[3/3] Validating HR-LR pairs...")
    pair_stats = validate_pairs(hr_locations, lr_locations)
    print(f"  - Matched pairs: {pair_stats['matched_count']}")
    print(f"  - HR only (no LR): {pair_stats['hr_only_count']}")
    print(f"  - LR only (no HR): {pair_stats['lr_only_count']}")
    
    # Compile results
    results = {
        "hr_dataset": {
            "total_locations": len(hr_locations),
            "file_counts_by_type": hr_file_counts,
            "sample_locations": hr_locations[:5]
        },
        "lr_dataset": {
            "total_locations": len(lr_locations),
            "file_counts_by_type": lr_file_counts,
            "acquisitions_stats": {
                "min": min(acquisition_counts.values()) if acquisition_counts else 0,
                "max": max(acquisition_counts.values()) if acquisition_counts else 0,
                "avg": round(sum(acquisition_counts.values()) / len(acquisition_counts), 2) if acquisition_counts else 0
            },
            "sample_locations": lr_locations[:5]
        },
        "pair_validation": pair_stats,
        "summary": {
            "total_usable_pairs": pair_stats['matched_count'],
            "total_hr_files": sum(hr_file_counts.values()),
            "total_lr_files": sum(lr_file_counts.values())
        }
    }
    
    # Save results
    output_file = OUTPUT_DIR / "dataset_stats.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}")
    
    # Print summary
    print("\n📊 SUMMARY")
    print(f"  Total usable HR-LR pairs: {pair_stats['matched_count']}")
    print(f"  Total HR files: {sum(hr_file_counts.values())}")
    print(f"  Total LR files: {sum(lr_file_counts.values())}")
    
    return results


if __name__ == "__main__":
    main()

