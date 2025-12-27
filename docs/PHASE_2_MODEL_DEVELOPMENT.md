# Phase 2: Model Development (SwinIR)

## Overview
This phase implements the SwinIR architecture and data pipeline for satellite imagery super-resolution.

**Status**: ✅ COMPLETED

---

## Objectives
- [x] Understand SwinIR architecture
- [x] Implement SwinIR model for satellite imagery
- [x] Create data loading pipeline
- [x] Define loss functions
- [x] Set up hardware-optimized configuration
- [x] Verify complete pipeline

---

## Implementation Summary

### Hardware Configuration
| Component | Specification | Optimization |
|-----------|--------------|--------------|
| GPU | RTX 3050 6GB | Batch size 4, AMP enabled |
| CPU | i5-13420H (8 cores) | 8 data loading workers |
| RAM | 16GB | Pin memory enabled |
| Storage | NVMe SSD | Prefetch factor 2 |

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Model | SwinIR-Small |
| Parameters | 1.24M |
| Input size | 64×64 (LR) |
| Output size | 256×256 (HR) |
| Scale factor | 4x |
| Window size | 8 |
| Embed dim | 60 |
| Depths | [6, 6, 6, 6] |
| Num heads | [6, 6, 6, 6] |

### Training Configuration
| Parameter | Value | Reason |
|-----------|-------|--------|
| Batch size | 4 | Max for 6GB VRAM |
| Accumulation steps | 4 | Effective batch = 16 |
| Mixed Precision | Enabled | 2x speedup, 50% less VRAM |
| Num workers | 8 | All CPU cores |
| Learning rate | 2e-4 | Standard for SwinIR |

---

## Components Created

### File Structure
```
phase2_model_development/
├── __init__.py           # Package exports
├── config.py             # Hardware-optimized configuration
├── dataset.py            # WorldStrat Dataset + DataLoader
├── swinir.py             # SwinIR model architecture
├── losses.py             # L1, Charbonnier, Perceptual losses
├── utils.py              # Checkpoints, utilities
├── verify_pipeline.py    # Pipeline verification
└── README.md             # Phase 2 guide
```

### Key Classes

#### Dataset (`dataset.py`)
```python
WorldStratDataset(split='train', scale=4, hr_patch_size=256)
get_dataloader(split='train', batch_size=4, num_workers=8)
```

#### Model (`swinir.py`)
```python
SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8)
swinir_small(upscale=4)  # 1.24M params
swinir_medium(upscale=4) # ~12M params
```

#### Losses (`losses.py`)
```python
L1Loss()
CharbonnierLoss(eps=1e-6)
PerceptualLoss(layer_weights={'conv5_4': 1.0})
CombinedLoss(pixel_loss='l1', perceptual_weight=0.0)
```

#### Config (`config.py`)
```python
Config()  # Auto-detects hardware and optimizes settings
```

---

## Verification Results

### Test Summary
| Test | Status | Details |
|------|--------|---------|
| Dataset | ✅ PASSED | 3,140 train samples, shapes correct |
| Model | ✅ PASSED | 1.24M params, 4x upscale works |
| Losses | ✅ PASSED | L1, Charbonnier compute correctly |
| Full Pipeline | ✅ PASSED | Training + validation works |

### Detailed Results
```
Dataset:
  - Train size: 3,140 locations
  - LR shape: [3, 64, 64]
  - HR shape: [3, 256, 256]
  - LR range: [0.020, 0.090]
  - HR range: [0.000, 0.176] (after 12-bit normalization fix)

Model:
  - Parameters: 1.24M
  - Input: [batch, 3, 64, 64]
  - Output: [batch, 3, 256, 256]
  - Gradients: flowing correctly

Losses:
  - L1 Loss: ~1.13 (random input)
  - Training Loss: ~0.80 (dummy data)
```

---

## Observations & Notes

### Value Range Analysis
- **LR images**: Sentinel-2 float32, range [0, 1] after clipping
- **HR images**: 12-bit uint16 normalized by 4095 (not 65535) → range [0, 0.18]
- **Fix Applied**: Changed normalization from 65535 to 4095 for proper 12-bit scaling
- **Result**: HR values now in reasonable range for better gradient flow

### Memory Usage
- Estimated VRAM: ~2-3 GB with batch size 4 + AMP
- Safe margin for 6GB GPU
- Can increase batch if needed

### Potential Improvements (Future)
1. ✅ **DONE**: Adjusted HR normalization to 4095 for 12-bit data
2. Add temporal fusion for multi-acquisition LR images
3. Implement perceptual loss for better visual quality
4. Add data augmentation variants

---

## Usage Examples

### Load Dataset
```python
from phase2_model_development import WorldStratDataset, get_dataloader

train_loader = get_dataloader('train', batch_size=4, num_workers=8)
for batch in train_loader:
    lr = batch['lr']  # [4, 3, 64, 64]
    hr = batch['hr']  # [4, 3, 256, 256]
```

### Create Model
```python
from phase2_model_development import swinir_small, Config

cfg = Config()  # Auto-detect hardware
model = swinir_small(upscale=4).cuda()
```

### Training Step
```python
from phase2_model_development import CombinedLoss

criterion = CombinedLoss(pixel_loss='l1')
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Forward pass
sr = model.forward_simple(lr)
losses = criterion(sr, hr)

# Backward pass
losses['total'].backward()
optimizer.step()
```

---

## Next Steps (Phase 3)

1. Implement complete training loop with:
   - Gradient accumulation
   - Mixed precision (AMP)
   - Learning rate scheduling
   - Checkpointing
2. Add validation with metrics (PSNR, SSIM)
3. Implement TensorBoard logging
4. Run baseline training experiment

---

**Phase Start Date**: December 8, 2025  
**Phase End Date**: December 12, 2025  
**Duration**: 5 days  
**Completed By**: ___________
