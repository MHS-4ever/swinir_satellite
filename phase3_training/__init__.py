# Phase 3: Training & Experimentation
#
# Components:
#   train.py      - Main training script
#   metrics.py    - PSNR, SSIM metrics
#
# Usage:
#   python phase3_training/train.py --epochs 100
#   tensorboard --logdir=phase3_training/runs

from .metrics import calculate_psnr, calculate_ssim, MetricsCalculator
