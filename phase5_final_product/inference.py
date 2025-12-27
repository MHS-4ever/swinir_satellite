"""
inference.py - Single image super-resolution inference

Upscale a single satellite image using the trained SwinIR model.

Usage:
    python phase5_final_product/inference.py --input image.tiff --output sr_image.png
    python phase5_final_product/inference.py --input image.png --output sr_image.png --cpu
"""

import argparse
import sys
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase2_model_development.swinir import swinir_small


def load_model(checkpoint_path: str, device: torch.device):
    """Load the SwinIR model from checkpoint."""
    model = swinir_small(upscale=4, in_chans=3, img_size=64, window_size=8)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle both clean and training checkpoints
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def load_image(image_path: str) -> np.ndarray:
    """Load image and convert to RGB numpy array."""
    path = Path(image_path)
    
    if path.suffix.lower() in ['.tiff', '.tif']:
        import tifffile
        img = tifffile.imread(str(path))
        # Handle multi-band images
        if img.ndim == 3:
            if img.shape[2] > 3:
                img = img[:, :, :3]  # Take first 3 bands
            elif img.shape[0] <= 4:  # CHW format
                img = np.transpose(img[:3], (1, 2, 0))
        # Normalize based on data type
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 4095.0  # 12-bit
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / img.max()
    else:
        img = Image.open(image_path).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0
    
    return np.clip(img, 0.0, 1.0)


def save_image(image: np.ndarray, output_path: str):
    """Save image to file."""
    path = Path(output_path)
    
    # Convert to uint8
    img = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    if path.suffix.lower() in ['.tiff', '.tif']:
        import tifffile
        tifffile.imwrite(str(path), img)
    else:
        Image.fromarray(img).save(str(path))


def preprocess(image: np.ndarray, target_size: int = 64) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize to target size if needed
    h, w = image.shape[:2]
    
    if h != target_size or w != target_size:
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        img_pil = img_pil.resize((target_size, target_size), Image.BICUBIC)
        image = np.array(img_pil).astype(np.float32) / 255.0
    
    # Convert to tensor (CHW)
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0)


def postprocess(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output to numpy image."""
    img = tensor.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
    return np.clip(img, 0.0, 1.0)


def super_resolve(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 64,
    tile_overlap: int = 8
) -> np.ndarray:
    """
    Super-resolve an image using the model.
    
    For large images, uses tiled processing to avoid memory issues.
    """
    h, w = image.shape[:2]
    
    # For small images, process directly
    if h <= tile_size and w <= tile_size:
        input_tensor = preprocess(image, tile_size).to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor)
            output_tensor = torch.clamp(output_tensor, 0, 1)
        return postprocess(output_tensor)
    
    # For larger images, use tiled processing
    scale = 4
    output_h, output_w = h * scale, w * scale
    output = np.zeros((output_h, output_w, 3), dtype=np.float32)
    count = np.zeros((output_h, output_w, 1), dtype=np.float32)
    
    step = tile_size - tile_overlap
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            tile = image[y_start:y_end, x_start:x_end]
            
            # Process tile
            input_tensor = preprocess(tile, tile_size).to(device)
            with torch.no_grad():
                output_tensor = model(input_tensor)
                output_tensor = torch.clamp(output_tensor, 0, 1)
            
            sr_tile = postprocess(output_tensor)
            
            # Place in output
            out_y_start = y_start * scale
            out_y_end = y_end * scale
            out_x_start = x_start * scale
            out_x_end = x_end * scale
            
            output[out_y_start:out_y_end, out_x_start:out_x_end] += sr_tile
            count[out_y_start:out_y_end, out_x_start:out_x_end] += 1
    
    # Average overlapping regions
    output = output / np.maximum(count, 1)
    
    return output


def main():
    parser = argparse.ArgumentParser(description='SwinIR Satellite Super-Resolution')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input image path (TIFF, PNG, JPG)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output image path')
    parser.add_argument('--checkpoint', '-c', type=str, 
                       default='phase5_final_product/release/model/swinir_satellite_sr_x4.pth',
                       help='Model checkpoint path')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--tile_size', type=int, default=64,
                       help='Tile size for large image processing')
    args = parser.parse_args()
    
    print("=" * 50)
    print("SwinIR Satellite Super-Resolution")
    print("=" * 50)
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        # Try alternative paths
        alt_path = PROJECT_ROOT / "phase3_training" / "checkpoints" / "best.pth"
        if alt_path.exists():
            checkpoint_path = alt_path
        else:
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    
    print(f"Loading model: {checkpoint_path}")
    model = load_model(str(checkpoint_path), device)
    print("Model loaded successfully")
    
    # Load input image
    input_path = Path(args.input)
    
    # Try to resolve the path
    if not input_path.exists():
        # Check if it's a relative path - try current directory
        if not input_path.is_absolute():
            abs_path = Path.cwd() / input_path
            if abs_path.exists():
                input_path = abs_path
            else:
                # File not found - provide helpful error message
                print(f"Error: Input image not found: {args.input}")
                print(f"\nTried paths:")
                print(f"  - {input_path.absolute()}")
                print(f"  - {abs_path.absolute()}")
                
                # Look for sample images in common locations
                sample_dirs = [
                    PROJECT_ROOT / "phase1_data_exploration" / "outputs" / "visualizations",
                    PROJECT_ROOT / "phase4_evaluation" / "outputs" / "comparison_images",
                    PROJECT_ROOT / "phase5_final_product" / "release" / "examples",
                    PROJECT_ROOT / "phase4_evaluation" / "outputs" / "figures",
                    Path.cwd(),
                ]
                
                print("\nLooking for sample images in common locations...")
                found_samples = []
                for sample_dir in sample_dirs:
                    if sample_dir.exists():
                        for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                            samples = list(sample_dir.glob(f'*{ext}'))
                            if samples:
                                found_samples.extend(samples[:3])  # Show first 3 per directory
                                break  # One extension type per directory is enough
                
                if found_samples:
                    print(f"\nFound {len(found_samples)} sample image(s) you could use:")
                    for i, sample in enumerate(found_samples[:5], 1):  # Show up to 5
                        try:
                            rel_path = sample.relative_to(Path.cwd())
                        except ValueError:
                            rel_path = sample
                        print(f"  {i}. {rel_path}")
                    print(f"\nExample usage:")
                    try:
                        example_path = found_samples[0].relative_to(Path.cwd())
                    except ValueError:
                        example_path = found_samples[0]
                    print(f"  python phase5_final_product/inference.py --input {example_path} --output output.png")
                else:
                    print("\nNo sample images found in common locations.")
                    print("Please provide a valid image path (PNG, JPG, TIFF formats supported).")
                
                sys.exit(1)
    
    print(f"Loading image: {input_path}")
    image = load_image(str(input_path))
    print(f"Input size: {image.shape[1]}x{image.shape[0]}")
    
    # Super-resolve
    print("Processing...")
    start_time = time.time()
    sr_image = super_resolve(model, image, device, tile_size=args.tile_size)
    elapsed = time.time() - start_time
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Output size: {sr_image.shape[1]}x{sr_image.shape[0]} (4x upscale)")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(sr_image, str(output_path))
    print(f"Saved: {output_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

