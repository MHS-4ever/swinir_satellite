# Phase 2: Model Development

## Overview
This phase implements the SwinIR architecture and data pipeline for satellite imagery super-resolution.

**Hardware Optimized For:**
- GPU: RTX 3050 (6GB VRAM)
- CPU: Intel i5-13420H (8 cores)
- RAM: 16GB
- Storage: NVMe SSD

---

## Workflow

```
Step 1: Data Pipeline        → dataset.py
Step 2: SwinIR Model         → swinir.py
Step 3: Loss Functions       → losses.py
Step 4: Model Utilities      → utils.py
Step 5: Configuration        → config.py
Step 6: Verify Pipeline      → verify_pipeline.py
```

---

## Scripts

### 1. Data Pipeline (`dataset.py`)
PyTorch Dataset for WorldStrat HR-LR pairs.

```python
from phase2_model_development.dataset import WorldStratDataset
dataset = WorldStratDataset(split='train', scale=4)
```

---

### 2. SwinIR Model (`swinir.py`)
SwinIR architecture adapted for satellite imagery.

```python
from phase2_model_development.swinir import SwinIR
model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8)
```

---

### 3. Loss Functions (`losses.py`)
Training losses: L1, Perceptual, Charbonnier.

```python
from phase2_model_development.losses import L1Loss, PerceptualLoss
```

---

### 4. Model Utilities (`utils.py`)
Save/load checkpoints, count parameters.

---

### 5. Verify Pipeline (`verify_pipeline.py`)
Test data loading and model forward pass.

```bash
python phase2_model_development/verify_pipeline.py
```

---

## Configuration

Based on Phase 1 analysis and hardware optimization:

| Parameter | Value | Reason |
|-----------|-------|--------|
| Input | 64×64 RGB | Fits in 6GB VRAM |
| Output | 256×256 RGB | 4x upscale |
| Batch size | 4 | Max for 6GB VRAM |
| Accumulation | 4 | Effective batch = 16 |
| Num workers | 8 | Use all CPU cores |
| Mixed Precision | Yes | 2x speedup, 50% less VRAM |
| Model | swinir_small | ~1M params, fits in VRAM |

```python
from phase2_model_development.config import Config
cfg = Config()  # Auto-detects and optimizes for your hardware
```

---

## Model Architecture

```
Input (64×64×3)
    ↓
Shallow Feature Extraction (Conv 3×3)
    ↓
Deep Feature Extraction (6× RSTB blocks)
    ↓
Feature Reconstruction (Conv 3×3)
    ↓
Upsampling (PixelShuffle 4×)
    ↓
Output (256×256×3)
```


