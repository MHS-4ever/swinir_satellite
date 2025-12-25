# Phase 3: Training

## Status: ✅ COMPLETED

**Best PSNR**: 23.95 dB (Epoch 97)  
**Best SSIM**: 0.6321 (Epoch 94)  
**Training Time**: ~4 hours on RTX 3050 6GB

---

## Quick Start

### Train from scratch
```bash
python phase3_training/train.py --epochs 100
```

### Resume training
```bash
python phase3_training/train.py --epochs 200 --resume phase3_training/checkpoints/latest.pth
```

### View training curves
```bash
tensorboard --logdir=phase3_training/runs
```

---

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script |
| `metrics.py` | PSNR, SSIM metrics |
| `verify_training.py` | Pipeline verification |
| `FINAL_TRAINING_REPORT.md` | Complete analysis |
| `checkpoints/best.pth` | Best model (PSNR: 23.95 dB) |
| `checkpoints/latest.pth` | Final model (E100) |

---

## Training Results

| Metric | Start | End | Best |
|--------|-------|-----|------|
| PSNR | 21.33 dB | 23.91 dB | **23.95 dB** |
| SSIM | 0.5039 | 0.6308 | **0.6321** |
| Val Loss | 0.0756 | 0.0548 | 0.0547 |

---

## Command Options

```bash
python phase3_training/train.py \
    --epochs 100 \           # Number of epochs
    --batch_size 4 \         # Batch size (auto-detected)
    --lr 0.0002 \            # Learning rate
    --resume PATH \          # Resume from checkpoint
    --workers 8              # Data loader workers
```

---

## Next Step → Phase 4

Evaluate the trained model:
```bash
python phase4_evaluation/evaluate.py --checkpoint phase3_training/checkpoints/best.pth
```
