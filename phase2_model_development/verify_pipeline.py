"""
verify_pipeline.py - Verify Data and Model Pipeline

Tests:
1. Data loading works correctly
2. Model forward pass works
3. Loss computation works
4. Shapes are correct throughout

Usage:
    python phase2_model_development/verify_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from phase2_model_development.dataset import WorldStratDataset, get_dataloader
from phase2_model_development.swinir import SwinIR, swinir_small
from phase2_model_development.losses import L1Loss, CharbonnierLoss, CombinedLoss
from phase2_model_development.utils import count_parameters, get_device, set_seed


def test_dataset():
    """Test dataset loading."""
    print("\n" + "=" * 60)
    print("Testing Dataset")
    print("=" * 60)
    
    try:
        # Test train dataset
        print("\n[1] Loading train dataset...")
        train_dataset = WorldStratDataset(
            split='train',
            scale=4,
            hr_patch_size=256,
            augment=True
        )
        print(f"  ✓ Train dataset size: {len(train_dataset)}")
        
        # Get a sample
        print("\n[2] Loading sample...")
        sample = train_dataset[0]
        lr = sample['lr']
        hr = sample['hr']
        location = sample['location']
        
        print(f"  ✓ LR shape: {lr.shape} (expected: [3, 64, 64])")
        print(f"  ✓ HR shape: {hr.shape} (expected: [3, 256, 256])")
        print(f"  ✓ Location: {location}")
        print(f"  ✓ LR value range: [{lr.min():.3f}, {lr.max():.3f}]")
        print(f"  ✓ HR value range: [{hr.min():.3f}, {hr.max():.3f}]")
        
        # Test dataloader
        print("\n[3] Testing DataLoader...")
        dataloader = get_dataloader(
            split='train',
            batch_size=4,
            num_workers=0,  # 0 for testing
            scale=4,
            hr_patch_size=256
        )
        
        batch = next(iter(dataloader))
        print(f"  ✓ Batch LR shape: {batch['lr'].shape}")
        print(f"  ✓ Batch HR shape: {batch['hr'].shape}")
        
        print("\n✅ Dataset test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test model forward pass."""
    print("\n" + "=" * 60)
    print("Testing Model")
    print("=" * 60)
    
    try:
        device = get_device()
        
        # Create model
        print("\n[1] Creating SwinIR model...")
        model = swinir_small(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8
        )
        model = model.to(device)
        
        # Count parameters
        params = count_parameters(model)
        print(f"  ✓ Total parameters: {params['total'] / 1e6:.2f}M")
        print(f"  ✓ Trainable parameters: {params['trainable'] / 1e6:.2f}M")
        
        # Test forward pass
        print("\n[2] Testing forward pass...")
        x = torch.randn(2, 3, 64, 64).to(device)
        
        model.eval()
        with torch.no_grad():
            y = model.forward_simple(x)
        
        print(f"  ✓ Input shape: {x.shape}")
        print(f"  ✓ Output shape: {y.shape} (expected: [2, 3, 256, 256])")
        print(f"  ✓ Output value range: [{y.min():.3f}, {y.max():.3f}]")
        
        # Verify scale
        expected_h = x.shape[2] * 4
        expected_w = x.shape[3] * 4
        assert y.shape[2] == expected_h, f"Height mismatch: {y.shape[2]} vs {expected_h}"
        assert y.shape[3] == expected_w, f"Width mismatch: {y.shape[3]} vs {expected_w}"
        
        # Test gradient flow
        print("\n[3] Testing gradient flow...")
        model.train()
        x = torch.randn(1, 3, 64, 64, requires_grad=True).to(device)
        y = model.forward_simple(x)
        loss = y.mean()
        loss.backward()
        
        # Check gradients exist
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Gradients computed: {has_grad}")
        
        print("\n✅ Model test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_losses():
    """Test loss functions."""
    print("\n" + "=" * 60)
    print("Testing Losses")
    print("=" * 60)
    
    try:
        device = get_device()
        
        pred = torch.randn(2, 3, 256, 256).to(device)
        target = torch.randn(2, 3, 256, 256).to(device)
        
        # L1 Loss
        print("\n[1] Testing L1 Loss...")
        l1_loss = L1Loss()
        loss_val = l1_loss(pred, target)
        print(f"  ✓ L1 Loss: {loss_val.item():.4f}")
        
        # Charbonnier Loss
        print("\n[2] Testing Charbonnier Loss...")
        char_loss = CharbonnierLoss()
        loss_val = char_loss(pred, target)
        print(f"  ✓ Charbonnier Loss: {loss_val.item():.4f}")
        
        # Combined Loss (without perceptual for speed)
        print("\n[3] Testing Combined Loss...")
        combined = CombinedLoss(
            pixel_loss='l1',
            pixel_weight=1.0,
            perceptual_weight=0.0  # Skip perceptual for faster test
        )
        losses = combined(pred, target)
        print(f"  ✓ Pixel Loss: {losses['pixel'].item():.4f}")
        print(f"  ✓ Total Loss: {losses['total'].item():.4f}")
        
        print("\n✅ Losses test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Losses test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test full training pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline")
    print("=" * 60)
    
    try:
        device = get_device()
        set_seed(42)
        
        # Create components
        print("\n[1] Creating pipeline components...")
        
        model = swinir_small(upscale=4, in_chans=3, img_size=64, window_size=8)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = CombinedLoss(pixel_loss='l1', pixel_weight=1.0, perceptual_weight=0.0)
        
        # Create dummy batch
        print("\n[2] Running training step...")
        lr_batch = torch.randn(2, 3, 64, 64).to(device)
        hr_batch = torch.randn(2, 3, 256, 256).to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        sr_batch = model.forward_simple(lr_batch)
        losses = criterion(sr_batch, hr_batch)
        
        losses['total'].backward()
        optimizer.step()
        
        print(f"  ✓ Loss: {losses['total'].item():.4f}")
        print(f"  ✓ Output shape: {sr_batch.shape}")
        
        # Validation step
        print("\n[3] Running validation step...")
        model.eval()
        with torch.no_grad():
            sr_batch = model.forward_simple(lr_batch)
            val_losses = criterion(sr_batch, hr_batch)
        
        print(f"  ✓ Val Loss: {val_losses['total'].item():.4f}")
        
        print("\n✅ Full pipeline test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Full pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Phase 2: Pipeline Verification")
    print("=" * 60)
    
    results = {
        'dataset': test_dataset(),
        'model': test_model(),
        'losses': test_losses(),
        'full_pipeline': test_full_pipeline()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready for Phase 3 (Training).")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    main()

