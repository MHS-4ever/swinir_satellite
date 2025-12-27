# Satellite Imagery Resolution Enhancement using SwinIR

## 🎉 Project Complete!

ML Course Project - Super-resolution of satellite imagery using the SwinIR architecture trained on the WorldStrat dataset.

### Key Results

| Metric | Bicubic (Baseline) | SwinIR (Ours) | Improvement |
|--------|-------------------|---------------|-------------|
| **PSNR** | 17.87 dB | **24.49 dB** | **+6.62 dB** |
| **SSIM** | 0.4456 | **0.6501** | **+0.2046** |

---

## Quick Start

### Run Inference
```bash
python phase5_final_product/inference.py --input image.png --output sr_image.png
```

### Launch Demo
```bash
python phase5_final_product/demo.py
# Open http://localhost:7860
```

---

## Project Structure

```
ml_project/
├── dataset/                      # WorldStrat dataset
│   ├── hr_dataset/               # High-resolution images
│   ├── lr_dataset/               # Low-resolution Sentinel-2
│   └── metadata.csv
├── docs/                         # Phase documentation
├── phase1_data_exploration/      # Data analysis & preprocessing
├── phase2_model_development/     # SwinIR model implementation
├── phase3_training/              # Training pipeline (100 epochs)
│   └── checkpoints/              # Trained models
├── phase4_evaluation/            # Evaluation & metrics
│   └── outputs/                  # Graphs, figures, results
├── phase5_final_product/         # Final packaging
│   ├── release/                  # Packaged model
│   ├── inference.py              # Single image SR
│   ├── batch_inference.py        # Batch processing
│   ├── demo.py                   # Gradio web demo
│   └── TECHNICAL_REPORT.md       # Full documentation
└── README.md
```

---

## Setup

```bash
# Create conda environment
conda create -n swinir python=3.10 -y
conda activate swinir

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy pandas matplotlib pillow tifffile tqdm pyyaml tensorboard gradio
```

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Exploration | ✅ Completed |
| 2 | Model Development | ✅ Completed |
| 3 | Training (100 epochs) | ✅ Completed |
| 4 | Evaluation | ✅ Completed |
| 5 | Final Product | ✅ Completed |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Phase 1: Data Exploration](docs/PHASE_1_DATA_EXPLORATION.md) | Dataset analysis |
| [Phase 2: Model Development](docs/PHASE_2_MODEL_DEVELOPMENT.md) | SwinIR implementation |
| [Phase 3: Training](docs/PHASE_3_TRAINING.md) | Training results |
| [Phase 4: Evaluation](docs/PHASE_4_EVALUATION.md) | Test set metrics |
| [Phase 5: Final Product](docs/PHASE_5_FINAL_PRODUCT.md) | Handoff materials |
| [Technical Report](phase5_final_product/TECHNICAL_REPORT.md) | Full technical details |
| [User Guide](phase5_final_product/USER_GUIDE.md) | Usage instructions |

---

## For Research Team

### Available Materials

- **Technical Report**: `phase5_final_product/TECHNICAL_REPORT.md`
- **LaTeX Tables**: `phase5_final_product/latex_tables.tex`
- **Graphs**: `phase4_evaluation/outputs/graphs/` (8 analysis charts)
- **Figures**: `phase4_evaluation/outputs/figures/` (31 comparison images)
- **Results CSV**: `phase4_evaluation/outputs/data/per_image_results.csv`

---

## Model Details

| Attribute | Value |
|-----------|-------|
| Architecture | SwinIR-Small |
| Parameters | 1.24M |
| Scale factor | 4× |
| Input size | 64×64 |
| Output size | 256×256 |
| Training epochs | 100 |
| Training time | ~4 hours |

---

## Dataset

- **Source**: [WorldStrat on Kaggle](https://www.kaggle.com/datasets/jucor1/worldstrat)
- **Train**: 3,140 locations
- **Validation**: 390 locations
- **Test**: 398 locations

---

## Results by Category

| Category | PSNR (SwinIR) | Improvement |
|----------|---------------|-------------|
| UNHCR | 25.08 dB | **+9.05 dB** |
| ASMSpotter | 26.82 dB | +3.06 dB |
| Landcover | 23.87 dB | +6.09 dB |
| Amnesty POI | 23.81 dB | +6.80 dB |

---

**Project Timeline**: December 3, 2025 - December 29, 2025 (27 days)  
**Project Completed**: December 29, 2025  
**Course**: Machine Learning  
**Team**: Experimental Implementation
