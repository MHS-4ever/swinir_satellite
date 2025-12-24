"""
config.py - Hardware-Optimized Configuration

Auto-detects hardware and sets optimal parameters.
Optimized for: RTX 3050 (6GB), i5-13420H (8 cores), 16GB RAM, NVMe SSD

Usage:
    from phase2_model_development.config import Config
    cfg = Config()
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """
    Training configuration optimized for available hardware.
    
    Hardware Profile:
    - GPU: RTX 3050 Laptop (6GB VRAM)
    - CPU: Intel i5-13420H (8 cores, 12 threads)
    - RAM: 16GB
    - Storage: NVMe SSD
    """
    
    # ========================
    # Model Configuration
    # ========================
    model_name: str = "swinir_small"  # Use small model for 6GB VRAM
    upscale: int = 4
    in_channels: int = 3
    img_size: int = 64  # LR input size
    window_size: int = 8
    embed_dim: int = 60  # Small model
    depths: tuple = (6, 6, 6, 6)
    num_heads: tuple = (6, 6, 6, 6)
    
    # ========================
    # Training Configuration
    # ========================
    # Batch size: 4 is optimal for 6GB VRAM with 64x64 input
    # Larger batches will cause OOM
    batch_size: int = 4
    
    # Gradient accumulation to simulate larger batch
    # Effective batch = batch_size * accumulation_steps = 4 * 4 = 16
    accumulation_steps: int = 4
    
    epochs: int = 100
    learning_rate: float = 2e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    
    # ========================
    # Mixed Precision (AMP)
    # ========================
    # CRITICAL: Reduces VRAM usage by ~50%, speeds up training by ~2x
    use_amp: bool = True
    
    # ========================
    # Data Loading (CPU Optimization)
    # ========================
    # Use all 8 CPU cores for parallel data loading
    num_workers: int = 8
    
    # Pin memory for faster CPU→GPU transfer (works great with NVMe)
    pin_memory: bool = True
    
    # Prefetch next batch while GPU processes current
    prefetch_factor: int = 2
    
    # ========================
    # Patch Sizes
    # ========================
    hr_patch_size: int = 256  # 256x256 HR patches
    lr_patch_size: int = 64   # 64x64 LR patches (256/4)
    
    # ========================
    # Loss Configuration
    # ========================
    pixel_loss: str = "l1"  # L1 is memory efficient
    pixel_weight: float = 1.0
    perceptual_weight: float = 0.0  # Disable for faster training, enable later
    
    # ========================
    # Checkpointing
    # ========================
    save_every: int = 10  # Save checkpoint every N epochs
    checkpoint_dir: str = "checkpoints"
    
    # ========================
    # Logging
    # ========================
    log_every: int = 50  # Log every N iterations
    val_every: int = 1   # Validate every N epochs
    
    # ========================
    # Memory Optimization
    # ========================
    # Clear cache periodically to prevent fragmentation
    empty_cache_freq: int = 100
    
    # Gradient checkpointing: trades compute for memory
    # Enable if still getting OOM errors
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        """Validate and adjust config based on available hardware."""
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_name = torch.cuda.get_device_name(0)
            
            print(f"[GPU] {gpu_name} ({gpu_mem:.1f}GB)")
            
            # Auto-adjust batch size based on VRAM
            if gpu_mem < 4:
                self.batch_size = 1
                self.accumulation_steps = 8
                print("  [!] Low VRAM: batch_size=1, accumulation=8")
            elif gpu_mem < 6:
                self.batch_size = 2
                self.accumulation_steps = 4
                print("  [!] Limited VRAM: batch_size=2, accumulation=4")
            elif gpu_mem < 8:
                self.batch_size = 4
                self.accumulation_steps = 4
                print("  [OK] Moderate VRAM: batch_size=4, accumulation=4")
            else:
                self.batch_size = 8
                self.accumulation_steps = 2
                print("  [OK] Good VRAM: batch_size=8, accumulation=2")
        else:
            print("[!] No GPU detected, using CPU (will be slow)")
            self.use_amp = False
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size with accumulation."""
        return self.batch_size * self.accumulation_steps
    
    def print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 50)
        print("Training Configuration")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Batch size: {self.batch_size} (effective: {self.get_effective_batch_size()})")
        print(f"LR patch: {self.lr_patch_size}x{self.lr_patch_size}")
        print(f"HR patch: {self.hr_patch_size}x{self.hr_patch_size}")
        print(f"Scale: {self.upscale}x")
        print(f"Mixed Precision (AMP): {self.use_amp}")
        print(f"Num workers: {self.num_workers}")
        print(f"Learning rate: {self.learning_rate}")
        print("=" * 50)


# VRAM Usage Estimation
def estimate_vram_usage(config: Config) -> dict:
    """Estimate VRAM usage for given config."""
    # Rough estimates based on SwinIR small model
    
    # Model weights: ~4MB for small, ~48MB for medium
    model_mb = 4 if config.model_name == "swinir_small" else 48
    
    # Forward pass activations (rough estimate)
    # Input: batch * 3 * 64 * 64 * 4 bytes = batch * 49KB
    # After upscale: batch * 3 * 256 * 256 * 4 bytes = batch * 786KB
    # Intermediate: ~10x input size for transformers
    input_mb = config.batch_size * 3 * 64 * 64 * 4 / 1e6
    output_mb = config.batch_size * 3 * 256 * 256 * 4 / 1e6
    activations_mb = input_mb * 50  # Rough estimate for transformer
    
    # Gradients: same size as weights
    gradients_mb = model_mb
    
    # Optimizer states: 2x weights for Adam
    optimizer_mb = model_mb * 2
    
    # AMP reduces activation memory by ~50%
    if config.use_amp:
        activations_mb *= 0.5
    
    total_mb = model_mb + activations_mb + gradients_mb + optimizer_mb
    
    return {
        "model_mb": model_mb,
        "activations_mb": activations_mb,
        "gradients_mb": gradients_mb,
        "optimizer_mb": optimizer_mb,
        "total_mb": total_mb,
        "total_gb": total_mb / 1024
    }


if __name__ == "__main__":
    cfg = Config()
    cfg.print_config()
    
    vram = estimate_vram_usage(cfg)
    print(f"\n[VRAM] Estimated Usage: {vram['total_gb']:.2f} GB")

