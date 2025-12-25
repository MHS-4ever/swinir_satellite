# Phase 4: Evaluation

## Status: ✅ COMPLETED

### Key Results
- **PSNR**: 24.49 dB (vs 17.87 dB bicubic) → **+6.62 dB improvement**
- **SSIM**: 0.6501 (vs 0.4456 bicubic) → **+0.2046 improvement**

Comprehensive evaluation of the trained SwinIR model on the held-out test set (398 samples).

---

## Quick Start

### Run full evaluation
```bash
python phase4_evaluation/evaluate.py --checkpoint phase3_training/checkpoints/best.pth
```

### Generate detailed visualizations
```bash
python phase4_evaluation/generate_visualizations.py --checkpoint phase3_training/checkpoints/best.pth
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `evaluate.py` | Main evaluation script - metrics, comparisons, graphs |
| `generate_visualizations.py` | Detailed visual comparisons for report figures |

---

## Outputs Generated

### Comparison Images
- `outputs/comparison_images/` - Side-by-side: LR | Bicubic | SwinIR | HR
- `outputs/best_results/` - Top 10 best performing samples
- `outputs/worst_results/` - 10 most challenging samples

### Graphs
- `psnr_distribution.png` - PSNR histogram (Bicubic vs SwinIR)
- `ssim_distribution.png` - SSIM histogram
- `psnr_improvement_distribution.png` - Improvement distribution
- `psnr_vs_ssim_scatter.png` - PSNR vs SSIM scatter plot
- `psnr_by_category.png` - Bar chart by category
- `ssim_by_category.png` - SSIM by category
- `psnr_boxplot_by_category.png` - Box plots
- `improvement_by_category.png` - Improvement by category

### Detailed Figures
- `figures/detailed_*.png` - Detailed comparisons with zoomed crops
- `figures/grid_best_6.png` - Grid of top 6 results
- `figures/grid_worst_6.png` - Grid of 6 challenging samples
- `figures/improvement_showcase.png` - Bicubic vs SwinIR showcase
- `figures/category_*.png` - Per-category samples

### Data Files
- `data/per_image_results.csv` - Per-image metrics (398 rows)
- `data/category_summary.csv` - Category-level summary
- `data/full_results.json` - Complete results in JSON format

### Report
- `EVALUATION_REPORT.md` - Comprehensive markdown report

---

## Command Options

### evaluate.py
```bash
python phase4_evaluation/evaluate.py \
    --checkpoint phase3_training/checkpoints/best.pth \
    --output_dir phase4_evaluation/outputs \
    --save_every 10  # Save comparison every N samples
```

### generate_visualizations.py
```bash
python phase4_evaluation/generate_visualizations.py \
    --checkpoint phase3_training/checkpoints/best.pth \
    --output_dir phase4_evaluation/outputs/figures \
    --num_samples 20
```

---

## Expected Results

Based on training validation:
- **PSNR**: ~23.9 dB (SwinIR) vs ~20 dB (Bicubic)
- **SSIM**: ~0.63 (SwinIR) vs ~0.40 (Bicubic)
- **Improvement**: ~+4 dB PSNR over baseline

---

## Test Set Details

- **Total Samples**: 398 locations
- **Categories**:
  - ASMSpotter: 39 samples
  - Amnesty POI: 10 samples
  - Landcover: 241 samples
  - UNHCR: 108 samples

