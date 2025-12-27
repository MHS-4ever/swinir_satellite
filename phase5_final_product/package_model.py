"""
package_model.py - Package the trained model for distribution

Creates a clean model package with:
- Stripped weights (no optimizer state)
- Model configuration
- Training information
- Example usage

Usage:
    python phase5_final_product/package_model.py
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase2_model_development.swinir import swinir_small


def package_model(
    checkpoint_path: str,
    output_dir: str,
    model_name: str = "swinir_satellite_sr_x4"
):
    """Package model for distribution."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "model").mkdir(exist_ok=True)
    (output_dir / "examples" / "input").mkdir(parents=True, exist_ok=True)
    (output_dir / "examples" / "output").mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Packaging SwinIR Model")
    print("=" * 60)
    
    # Load checkpoint
    print(f"\n[1/5] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    epoch = checkpoint.get('epoch', 'unknown')
    best_psnr = checkpoint.get('best_psnr', 0.0)
    
    print(f"  Epoch: {epoch}")
    print(f"  Best PSNR: {best_psnr:.2f} dB")
    
    # Create clean model
    print("\n[2/5] Creating clean model weights...")
    model = swinir_small(upscale=4, in_chans=3, img_size=64, window_size=8)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save clean weights (only model state, no optimizer)
    clean_weights = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'architecture': 'SwinIR-Small',
            'upscale': 4,
            'in_chans': 3,
            'img_size': 64,
            'window_size': 8,
            'embed_dim': 60,
            'depths': [6, 6, 6, 6],
            'num_heads': [6, 6, 6, 6],
        }
    }
    
    model_path = output_dir / "model" / f"{model_name}.pth"
    torch.save(clean_weights, model_path)
    print(f"  Saved: {model_path}")
    print(f"  Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create model info JSON
    print("\n[3/5] Creating model info...")
    model_info = {
        "model_name": "SwinIR-Satellite-SR",
        "version": "1.0.0",
        "description": "Super-Resolution model for satellite imagery using SwinIR architecture",
        "scale_factor": 4,
        "input_size": "64x64 (LR)",
        "output_size": "256x256 (HR)",
        "input_channels": 3,
        "architecture": "SwinIR-Small",
        "total_parameters": f"{total_params / 1e6:.2f}M",
        "trainable_parameters": f"{trainable_params / 1e6:.2f}M",
        "training_dataset": "WorldStrat",
        "training_samples": 3140,
        "validation_samples": 390,
        "test_samples": 398,
        "training_epochs": 100,
        "best_validation_psnr": f"{best_psnr:.2f} dB",
        "test_psnr": "24.49 dB",
        "test_ssim": "0.6501",
        "psnr_improvement_over_bicubic": "+6.62 dB",
        "ssim_improvement_over_bicubic": "+0.2046",
        "training_date": "2025-12-20",
        "checkpoint_file": f"{model_name}.pth",
        "framework": "PyTorch 2.0+",
        "cuda_support": True,
    }
    
    info_path = output_dir / "model" / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"  Saved: {info_path}")
    
    # Create model config YAML
    print("\n[4/5] Creating model config...")
    config_content = """# SwinIR Satellite Super-Resolution Model Configuration

model:
  name: SwinIR-Small
  type: super_resolution
  scale: 4
  
architecture:
  img_size: 64
  window_size: 8
  embed_dim: 60
  depths: [6, 6, 6, 6]
  num_heads: [6, 6, 6, 6]
  mlp_ratio: 2.0
  upsampler: pixelshuffle

input:
  channels: 3
  size: [64, 64]
  format: RGB
  range: [0.0, 1.0]

output:
  channels: 3
  size: [256, 256]
  format: RGB
  range: [0.0, 1.0]

inference:
  batch_size: 1
  device: cuda  # or cpu
  precision: float32  # or float16 for faster inference
"""
    
    config_path = output_dir / "model" / "model_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"  Saved: {config_path}")
    
    # Verify model loading
    print("\n[5/5] Verifying model package...")
    loaded = torch.load(model_path, map_location='cpu', weights_only=False)
    test_model = swinir_small(upscale=4, in_chans=3, img_size=64, window_size=8)
    test_model.load_state_dict(loaded['model_state_dict'])
    test_model.eval()
    
    # Test inference
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 64, 64)
        dummy_output = test_model(dummy_input)
    
    assert dummy_output.shape == (1, 3, 256, 256), "Output shape mismatch!"
    print("  Model verification: PASSED")
    
    print("\n" + "=" * 60)
    print("MODEL PACKAGING COMPLETE")
    print("=" * 60)
    print(f"\nPackage location: {output_dir}")
    print(f"Model file: {model_path}")
    print(f"Model size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return output_dir


if __name__ == '__main__':
    checkpoint = PROJECT_ROOT / "phase3_training" / "checkpoints" / "best.pth"
    output = PROJECT_ROOT / "phase5_final_product" / "release"
    
    package_model(str(checkpoint), str(output))

