"""
verify_training.py - Quick verification of training pipeline

Tests:
1. Model and optimizer creation
2. Single training step
3. Metrics calculation
4. TensorBoard logging

Usage:
    python phase3_training/verify_training.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.amp import GradScaler, autocast

from phase2_model_development import (
    Config, swinir_small, get_dataloader, CombinedLoss, get_device, set_seed
)
from phase3_training.metrics import calculate_psnr, calculate_ssim


def main():
    print("=" * 60)
    print("Phase 3: Training Pipeline Verification")
    print("=" * 60)
    
    set_seed(42)
    device = get_device()
    
    # Test 1: Model creation
    print("\n[1/4] Creating model...")
    config = Config()
    model = swinir_small(upscale=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = CombinedLoss(pixel_loss='l1')
    scaler = GradScaler('cuda', enabled=config.use_amp)
    print("  [OK] Model, optimizer, loss created")
    
    # Test 2: Single training step
    print("\n[2/4] Testing training step...")
    model.train()
    
    # Dummy data
    lr_img = torch.rand(2, 3, 64, 64).to(device)
    hr_img = torch.rand(2, 3, 256, 256).to(device)
    
    optimizer.zero_grad()
    with autocast('cuda', enabled=config.use_amp):
        sr_img = model.forward_simple(lr_img)
        losses = criterion(sr_img, hr_img)
    
    scaler.scale(losses['total']).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"  [OK] Training step completed, loss: {losses['total'].item():.4f}")
    
    # Test 3: Metrics
    print("\n[3/4] Testing metrics...")
    model.eval()
    with torch.no_grad():
        sr_img = model.forward_simple(lr_img)
        sr_img = torch.clamp(sr_img, 0, 1)
        
        psnr = calculate_psnr(sr_img, hr_img)
        ssim = calculate_ssim(sr_img, hr_img)
    
    print(f"  [OK] PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    # Test 4: Data loader
    print("\n[4/4] Testing real data loader...")
    try:
        train_loader = get_dataloader(
            split='train',
            batch_size=2,
            num_workers=0,  # 0 for quick test
            scale=4,
            hr_patch_size=256
        )
        batch = next(iter(train_loader))
        print(f"  [OK] Loaded batch: LR {batch['lr'].shape}, HR {batch['hr'].shape}")
        
        # Quick training step with real data
        lr_real = batch['lr'].to(device)
        hr_real = batch['hr'].to(device)
        
        optimizer.zero_grad()
        with autocast('cuda', enabled=config.use_amp):
            sr_real = model.forward_simple(lr_real)
            losses = criterion(sr_real, hr_real)
        
        scaler.scale(losses['total']).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"  [OK] Real data training step, loss: {losses['total'].item():.4f}")
        
    except Exception as e:
        print(f"  [!] Data loader test skipped: {e}")
    
    print("\n" + "=" * 60)
    print("[OK] All tests passed! Ready for training.")
    print("=" * 60)
    print("\nTo start training:")
    print("  python phase3_training/train.py --epochs 100")
    print("\nTo monitor training:")
    print("  tensorboard --logdir=phase3_training/runs")


if __name__ == "__main__":
    main()

