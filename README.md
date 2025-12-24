# Satellite Imagery Resolution Enhancement using SwinIR

ML Course Project - Experimental implementation of SwinIR for super-resolution of satellite imagery using the WorldStrat dataset.

## Project Structure

```
ml_project/
├── dataset/                     # WorldStrat dataset
│   ├── hr_dataset/              # High-resolution images
│   ├── lr_dataset/              # Low-resolution images
│   └── metadata.csv
├── docs/                        # Phase documentation
├── phase1_data_exploration/     # Data analysis & preprocessing scripts
├── phase2_model_development/    # SwinIR model implementation
├── phase3_training/             # Training pipeline & experiments
├── phase4_evaluation/           # Evaluation & metrics
├── phase5_final_product/        # Final packaging & handoff
├── .gitignore
└── README.md
```

## Setup

```bash
# Create conda environment
conda create -n swinir_satellite python=3.10 -y
conda activate swinir_satellite

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install numpy pandas matplotlib pillow tifffile scikit-image opencv-python tqdm pyyaml tensorboard lpips

# Generate requirements.txt
pip freeze > requirements.txt
```

## Documentation

| Phase | Document |
|-------|----------|
| 1 | [Data Exploration](docs/PHASE_1_DATA_EXPLORATION.md) |
| 2 | [Model Development](docs/PHASE_2_MODEL_DEVELOPMENT.md) |
| 3 | [Training](docs/PHASE_3_TRAINING.md) |
| 4 | [Evaluation](docs/PHASE_4_EVALUATION.md) |
| 5 | [Final Product](docs/PHASE_5_FINAL_PRODUCT.md) |

## Dataset

- **Source**: [WorldStrat on Kaggle](https://www.kaggle.com/datasets/jucor1/worldstrat)
- **Format**: 12-bit TIFF images
