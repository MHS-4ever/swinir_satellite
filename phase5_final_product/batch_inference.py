"""
batch_inference.py - Batch super-resolution inference

Process multiple images in a directory.

Usage:
    python phase5_final_product/batch_inference.py --input_dir ./inputs --output_dir ./outputs
"""

import argparse
import sys
from pathlib import Path
import time
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase5_final_product.inference import (
    load_model, load_image, save_image, super_resolve
)


def get_image_files(input_dir: Path) -> list:
    """Get all image files in directory."""
    extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f'*{ext}'))
        files.extend(input_dir.glob(f'*{ext.upper()}'))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description='Batch Super-Resolution')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                       help='Input directory with images')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                       help='Output directory for super-resolved images')
    parser.add_argument('--checkpoint', '-c', type=str,
                       default='phase5_final_product/release/model/swinir_satellite_sr_x4.pth',
                       help='Model checkpoint path')
    parser.add_argument('--format', '-f', type=str, default='png',
                       choices=['png', 'jpg', 'tiff'],
                       help='Output format')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--tile_size', type=int, default=64,
                       help='Tile size for large image processing')
    args = parser.parse_args()
    
    print("=" * 50)
    print("SwinIR Batch Super-Resolution")
    print("=" * 50)
    
    # Setup
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get input files
    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"No image files found in: {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images")
    
    # Load model
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        alt_path = PROJECT_ROOT / "phase3_training" / "checkpoints" / "best.pth"
        if alt_path.exists():
            checkpoint_path = alt_path
        else:
            print(f"Error: Checkpoint not found")
            sys.exit(1)
    
    print(f"Loading model...")
    model = load_model(str(checkpoint_path), device)
    print("Model loaded")
    
    # Process images
    print(f"\nProcessing {len(image_files)} images...")
    total_time = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # Load
            image = load_image(str(img_path))
            
            # Process
            start = time.time()
            sr_image = super_resolve(model, image, device, tile_size=args.tile_size)
            total_time += time.time() - start
            
            # Save
            output_name = img_path.stem + f"_sr.{args.format}"
            output_path = output_dir / output_name
            save_image(sr_image, str(output_path))
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Images processed: {len(image_files)}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/len(image_files):.2f}s")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()

