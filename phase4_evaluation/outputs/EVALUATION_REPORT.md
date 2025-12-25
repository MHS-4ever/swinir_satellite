# Evaluation Report - SwinIR Satellite Super-Resolution

## Generated: 2025-12-25

---

## Overall Results

| Metric | Bicubic (Baseline) | SwinIR (Ours) | Improvement |
|--------|-------------------|---------------|-------------|
| **PSNR** | 17.87 dB | **24.49 dB** | **+6.62 dB** |
| **SSIM** | 0.4456 | **0.6501** | **+0.2046** |

### Test Set Statistics
- **Total Samples**: 398
- **PSNR Range**: 1.47 - 42.70 dB
- **PSNR Std Dev**: 5.43 dB
- **SSIM Range**: 0.1269 - 0.9860

---

## Results by Category

| Category | Samples | PSNR (Bicubic) | PSNR (SwinIR) | Improvement | SSIM (SwinIR) |
|----------|---------|----------------|---------------|-------------|---------------|
| ASMSpotter | 39 | 23.76 dB | 26.82 dB | +3.06 dB | 0.7715 |
| Amnesty POI | 10 | 17.01 dB | 23.81 dB | +6.80 dB | 0.6009 |
| Landcover | 241 | 17.78 dB | 23.87 dB | +6.09 dB | 0.6286 |
| UNHCR | 108 | 16.03 dB | 25.08 dB | +9.05 dB | 0.6590 |

---

## Best Performing Samples

| Rank | Location | PSNR (dB) | SSIM | Improvement |
|------|----------|-----------|------|-------------|
| 1 | ASMSpotter-13-3-3 | 42.70 | 0.9488 | +8.85 dB |
| 2 | ASMSpotter-12-1-1 | 42.62 | 0.9490 | +7.77 dB |
| 3 | ASMSpotter-18-1-1 | 40.17 | 0.9229 | +6.31 dB |
| 4 | ASMSpotter-14-1-2 | 39.67 | 0.9442 | +5.90 dB |
| 5 | ASMSpotter-16-3-3 | 39.57 | 0.9400 | +5.96 dB |

---

## Challenging Samples (Lowest PSNR)

| Rank | Location | PSNR (dB) | SSIM | Improvement |
|------|----------|-----------|------|-------------|
| 1 | Landcover-774719 | 7.98 | 0.1542 | +-2.52 dB |
| 2 | Landcover-774999 | 6.69 | 0.2626 | +1.17 dB |
| 3 | Landcover-776564 | 5.56 | 0.1269 | +1.83 dB |
| 4 | Landcover-770313 | 3.64 | 0.1867 | +1.59 dB |
| 5 | Landcover-770166 | 1.47 | 0.1535 | +-1.22 dB |

---

## Generated Files

### Comparison Images
- `comparison_images/` - Side-by-side comparisons (every 10th sample)
- `best_results/` - Top 10 best performing samples
- `worst_results/` - 10 most challenging samples

### Graphs
- `graphs/psnr_distribution.png` - PSNR histogram comparison
- `graphs/ssim_distribution.png` - SSIM histogram comparison
- `graphs/psnr_improvement_distribution.png` - Improvement distribution
- `graphs/psnr_vs_ssim_scatter.png` - PSNR vs SSIM scatter plot
- `graphs/psnr_by_category.png` - Per-category PSNR bar chart
- `graphs/ssim_by_category.png` - Per-category SSIM bar chart
- `graphs/psnr_boxplot_by_category.png` - PSNR box plots by category
- `graphs/improvement_by_category.png` - Improvement by category

### Data Files
- `data/per_image_results.csv` - Per-image metrics
- `data/category_summary.csv` - Category-level summary
- `data/full_results.json` - Complete results in JSON format

---

## Conclusion

The SwinIR model achieved significant improvement over bicubic interpolation:
- **+6.62 dB PSNR** improvement on average
- **+0.2046 SSIM** improvement on average

The model performs consistently across all categories, with the best results on structured scenes and the most challenging results on complex natural scenes.
