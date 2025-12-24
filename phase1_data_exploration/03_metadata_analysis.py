"""
03_metadata_analysis.py - Metadata Analysis

Analyzes metadata.csv:
- Geographic distribution (lat/lon)
- Land cover classes
- Cloud coverage statistics
- Temporal information

Usage:
    python phase1_data_exploration/03_metadata_analysis.py
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
METADATA_FILE = DATASET_DIR / "metadata.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def main():
    print("=" * 60)
    print("WorldStrat Metadata Analysis")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("\n[1/5] Loading metadata...")
    if not METADATA_FILE.exists():
        print(f"ERROR: Metadata file not found: {METADATA_FILE}")
        return
    
    df = pd.read_csv(METADATA_FILE)
    print(f"  - Total rows: {len(df)}")
    print(f"  - Columns: {list(df.columns)}")
    
    # Basic statistics
    print("\n[2/5] Analyzing basic statistics...")
    
    # Get unique locations (first column seems to be location name)
    location_col = df.columns[0]
    unique_locations = df[location_col].nunique()
    print(f"  - Unique locations: {unique_locations}")
    
    # Cloud coverage analysis
    print("\n[3/5] Analyzing cloud coverage...")
    cloud_stats = {}
    if 'cloud_cover' in df.columns:
        cloud_stats = {
            "min": float(df['cloud_cover'].min()),
            "max": float(df['cloud_cover'].max()),
            "mean": float(df['cloud_cover'].mean()),
            "median": float(df['cloud_cover'].median()),
            "std": float(df['cloud_cover'].std()),
            "low_cloud_pct": float((df['cloud_cover'] < 10).mean() * 100),
            "high_cloud_pct": float((df['cloud_cover'] > 50).mean() * 100)
        }
        print(f"  - Cloud cover range: {cloud_stats['min']:.1f}% - {cloud_stats['max']:.1f}%")
        print(f"  - Mean cloud cover: {cloud_stats['mean']:.1f}%")
        print(f"  - Low cloud (<10%): {cloud_stats['low_cloud_pct']:.1f}% of samples")
    
    # Geographic analysis
    print("\n[4/5] Analyzing geographic distribution...")
    geo_stats = {}
    if 'lat' in df.columns and 'lon' in df.columns:
        geo_stats = {
            "lat_range": [float(df['lat'].min()), float(df['lat'].max())],
            "lon_range": [float(df['lon'].min()), float(df['lon'].max())],
            "lat_mean": float(df['lat'].mean()),
            "lon_mean": float(df['lon'].mean())
        }
        print(f"  - Latitude range: {geo_stats['lat_range'][0]:.2f} to {geo_stats['lat_range'][1]:.2f}")
        print(f"  - Longitude range: {geo_stats['lon_range'][0]:.2f} to {geo_stats['lon_range'][1]:.2f}")
    
    # Land cover analysis
    print("\n[5/5] Analyzing land cover classes...")
    landcover_stats = {}
    if 'IPCC Class' in df.columns:
        class_counts = df['IPCC Class'].value_counts().to_dict()
        landcover_stats["ipcc_classes"] = {str(k): int(v) for k, v in class_counts.items()}
        print(f"  - IPCC Classes found: {len(class_counts)}")
        for cls, cnt in list(class_counts.items())[:5]:
            print(f"    {cls}: {cnt} samples")
    
    if 'LCCS class' in df.columns:
        lccs_counts = df['LCCS class'].value_counts().to_dict()
        landcover_stats["lccs_classes"] = {str(k): int(v) for k, v in list(lccs_counts.items())[:10]}
    
    if 'SMOD Class' in df.columns:
        smod_counts = df['SMOD Class'].value_counts().to_dict()
        landcover_stats["smod_classes"] = {str(k): int(v) for k, v in smod_counts.items()}
    
    # Compile results
    results = {
        "total_rows": len(df),
        "unique_locations": unique_locations,
        "columns": list(df.columns),
        "cloud_coverage": cloud_stats,
        "geographic": geo_stats,
        "land_cover": landcover_stats,
        "acquisitions_per_location": {
            "min": int(df.groupby(location_col).size().min()),
            "max": int(df.groupby(location_col).size().max()),
            "mean": float(df.groupby(location_col).size().mean())
        }
    }
    
    # Save results
    output_file = OUTPUT_DIR / "metadata_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}")
    
    # Print summary
    print("\n📊 SUMMARY")
    print(f"  Unique locations: {unique_locations}")
    print(f"  Acquisitions per location: {results['acquisitions_per_location']['min']}-{results['acquisitions_per_location']['max']}")
    if cloud_stats:
        print(f"  Low cloud samples (<10%): {cloud_stats['low_cloud_pct']:.1f}%")
    
    return results


if __name__ == "__main__":
    main()

