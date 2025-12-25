# Technical Report: Satellite Image Super-Resolution using SwinIR

## Project: Resolution Enhancement of Satellite Imagery

---

## Abstract

This project implements a deep learning-based super-resolution system for satellite imagery using the SwinIR (Swin Transformer for Image Restoration) architecture. We trained the model on the WorldStrat dataset to upscale low-resolution Sentinel-2 satellite images by a factor of 4x. The model achieves a PSNR of **24.49 dB** and SSIM of **0.6501** on the test set, representing a **+6.62 dB** improvement over bicubic interpolation.

---

## 1. Introduction

### 1.1 Problem Statement

Satellite imagery is crucial for environmental monitoring, urban planning, disaster response, and humanitarian applications. However, many satellite sensors provide limited spatial resolution. Super-resolution (SR) techniques can enhance the effective resolution of these images, enabling finer detail analysis.

### 1.2 Objectives

1. Implement a state-of-the-art super-resolution model for satellite imagery
2. Train on real-world satellite data (WorldStrat dataset)
3. Achieve significant quality improvement over traditional interpolation methods
4. Provide a deployable solution for practical use

### 1.3 Key Contributions

- Adapted SwinIR architecture for satellite imagery super-resolution
- Trained on 3,140 satellite image pairs from the WorldStrat dataset
- Achieved **+6.62 dB PSNR** improvement over bicubic interpolation
- Provided comprehensive evaluation across 4 geographic categories

---

## 2. Related Work

### 2.1 Image Super-Resolution

- **SRCNN (2014)**: First CNN-based SR method
- **ESPCN (2016)**: Efficient sub-pixel convolution
- **EDSR (2017)**: Enhanced deep residual networks
- **RCAN (2018)**: Residual channel attention networks
- **SwinIR (2021)**: Swin Transformer-based image restoration

### 2.2 Satellite Image Super-Resolution

- Traditional methods: Bicubic, Lanczos interpolation
- Deep learning: Adapted natural image SR to satellite domain
- Challenges: Multi-spectral data, atmospheric effects, varying scales

---

## 3. Methodology

### 3.1 Dataset: WorldStrat

The WorldStrat dataset contains high-resolution satellite imagery paired with low-resolution Sentinel-2 images.

| Split | Locations | Description |
|-------|-----------|-------------|
| Train | 3,140 | Training data |
| Validation | 390 | Hyperparameter tuning |
| Test | 398 | Final evaluation |
| **Total** | **3,928** | |

**Categories**:
- ASMSpotter: Artisanal mining sites
- Amnesty POI: Points of interest
- Landcover: Various land cover types
- UNHCR: Refugee camp imagery

**Image Specifications**:
- HR (High-Resolution): 256×256, 12-bit, pan-sharpened
- LR (Low-Resolution): 64×64, Sentinel-2 L2A, 10m resolution
- Scale factor: 4×

### 3.2 Model Architecture: SwinIR-Small

We use a lightweight variant of SwinIR optimized for our hardware constraints.

**Architecture Details**:

| Component | Configuration |
|-----------|---------------|
| Embedding dimension | 60 |
| Depths | [6, 6, 6, 6] |
| Attention heads | [6, 6, 6, 6] |
| Window size | 8 |
| MLP ratio | 2.0 |
| Upsampler | Pixel shuffle |
| **Total parameters** | **1.24M** |

**Key Components**:
1. **Shallow Feature Extraction**: 3×3 convolution
2. **Deep Feature Extraction**: 4 Swin Transformer blocks
3. **High-Quality Image Reconstruction**: Pixel shuffle upsampling

### 3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2×10⁻⁴ |
| LR scheduler | Cosine annealing (5 restarts) |
| Batch size | 4 (effective: 16 with accumulation) |
| Epochs | 100 |
| Loss function | L1 loss |
| Mixed precision | Enabled (AMP) |
| Gradient accumulation | 4 steps |

**Hardware**:
- GPU: NVIDIA RTX 3050 6GB
- Training time: ~4 hours

### 3.4 Data Augmentation

- Random horizontal flip
- Random vertical flip
- Random 90° rotations

---

## 4. Experiments

### 4.1 Experimental Setup

**Metrics**:
- **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level accuracy
- **SSIM** (Structural Similarity Index): Measures structural similarity

**Baseline**: Bicubic interpolation (4× upscaling)

### 4.2 Training Progression

| Epoch | Train Loss | Val Loss | Val PSNR | Val SSIM |
|-------|------------|----------|----------|----------|
| 1 | 0.0905 | 0.0756 | 21.33 dB | 0.5039 |
| 20 | 0.0634 | 0.0582 | 23.43 dB | 0.6190 |
| 50 | 0.0619 | 0.0563 | 23.69 dB | 0.6228 |
| 100 | 0.0560 | 0.0548 | 23.91 dB | 0.6308 |
| **Best** | - | - | **23.95 dB** | **0.6321** |

### 4.3 Test Set Evaluation

**Overall Results**:

| Method | PSNR ↑ | SSIM ↑ |
|--------|--------|--------|
| Bicubic | 17.87 dB | 0.4456 |
| **SwinIR (Ours)** | **24.49 dB** | **0.6501** |
| **Improvement** | **+6.62 dB** | **+0.2046** |

**Results by Category**:

| Category | Samples | PSNR (Bicubic) | PSNR (SwinIR) | Improvement |
|----------|---------|----------------|---------------|-------------|
| ASMSpotter | 39 | 23.76 dB | 26.82 dB | +3.06 dB |
| Amnesty POI | 10 | 17.01 dB | 23.81 dB | +6.80 dB |
| Landcover | 241 | 17.78 dB | 23.87 dB | +6.09 dB |
| UNHCR | 108 | 16.03 dB | 25.08 dB | **+9.05 dB** |

---

## 5. Results

### 5.1 Quantitative Results

The SwinIR model achieves significant improvements across all categories:

- **Overall PSNR**: 24.49 dB (+6.62 dB vs bicubic)
- **Overall SSIM**: 0.6501 (+0.2046 vs bicubic)
- **Best category**: UNHCR with +9.05 dB improvement
- **Best sample**: ASMSpotter-13-3-3 with 42.70 dB PSNR

### 5.2 Qualitative Results

Visual comparisons show clear improvements:
- Sharper edges and details
- Better preservation of structural elements
- Reduced blurring artifacts
- More accurate color reproduction

### 5.3 Generalization

The model generalizes well to unseen data:
- Test PSNR (24.49 dB) > Validation PSNR (23.95 dB)
- Consistent performance across all categories

---

## 6. Discussion

### 6.1 Key Findings

1. **Significant improvement**: +6.62 dB PSNR over bicubic baseline
2. **Category-dependent performance**: Best on UNHCR imagery
3. **Efficient training**: Achieved with limited hardware (6GB GPU)
4. **Good generalization**: Test performance exceeds validation

### 6.2 Limitations

1. **Fixed scale factor**: Currently only supports 4× upscaling
2. **RGB only**: Does not utilize all spectral bands
3. **Fixed input size**: Requires 64×64 input patches
4. **No perceptual loss**: Uses L1 loss only

### 6.3 Future Work

1. **Multi-scale SR**: Support for 2×, 4×, 8× upscaling
2. **Multi-spectral**: Utilize all Sentinel-2 bands
3. **Perceptual loss**: Add VGG-based perceptual loss
4. **Larger model**: Try SwinIR-Medium for better quality
5. **Real-world deployment**: Tile-based processing for large images

---

## 7. Conclusion

We successfully implemented a super-resolution system for satellite imagery using the SwinIR architecture. The model achieves a **PSNR of 24.49 dB** on the test set, representing a **+6.62 dB improvement** over bicubic interpolation. The model generalizes well across different geographic categories and is ready for practical deployment.

---

## 8. Appendix

### A. Model Architecture Diagram

```
Input (64×64×3)
    ↓
Conv 3×3 (Shallow Features)
    ↓
┌─────────────────────────────┐
│ Swin Transformer Block ×4  │
│ - Multi-head Self-Attention │
│ - Window Size: 8×8          │
│ - Shifted Windows           │
└─────────────────────────────┘
    ↓
Conv 3×3 (Deep Features)
    ↓
Pixel Shuffle 4× Upsampling
    ↓
Output (256×256×3)
```

### B. Training Command

```bash
python phase3_training/train.py --epochs 100
```

### C. Inference Command

```bash
python phase5_final_product/inference.py --input image.tiff --output sr_image.png
```

### D. Demo Command

```bash
python phase5_final_product/demo.py
```

---

## References

1. Liang, J., et al. "SwinIR: Image Restoration Using Swin Transformer." ICCV Workshops, 2021.
2. Cornebise, J., et al. "WorldStrat: A Large-Scale Dataset for Semantic Segmentation and Change Detection." NeurIPS Datasets and Benchmarks, 2022.
3. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV, 2021.

---

**Report Date**: December 25, 2024  
**Authors**: ML Course Project Team  
**Version**: 1.0

