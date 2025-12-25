# User Guide: SwinIR Satellite Super-Resolution

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd ml_project

# Create conda environment
conda create -n swinir python=3.10 -y
conda activate swinir

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pillow tifffile tqdm matplotlib
pip install gradio  # Optional: for demo interface
```

### 2. Single Image Super-Resolution

```bash
python phase5_final_product/inference.py --input your_image.png --output sr_output.png
```

**Options**:
- `--input`: Input image (PNG, JPG, TIFF)
- `--output`: Output path
- `--cpu`: Use CPU instead of GPU
- `--checkpoint`: Custom model checkpoint

### 3. Batch Processing

```bash
python phase5_final_product/batch_inference.py --input_dir ./images --output_dir ./results
```

### 4. Interactive Demo

```bash
python phase5_final_product/demo.py
```
Then open http://localhost:7860 in your browser.

---

## Input Requirements

| Parameter | Requirement |
|-----------|-------------|
| Format | PNG, JPG, TIFF |
| Channels | RGB (3 channels) |
| Size | Any (will be resized to 64×64) |
| Range | 0-255 (uint8) or 0-1 (float) |

**Best results with**:
- Satellite/aerial imagery
- Images similar to Sentinel-2 data
- Clear, cloud-free images

---

## Output

| Parameter | Value |
|-----------|-------|
| Size | 4× input size (64×64 → 256×256) |
| Format | Same as input (or specified) |
| Channels | RGB |
| Quality | PSNR ~24.5 dB |

---

## Examples

### Example 1: Upscale a PNG image

```bash
python phase5_final_product/inference.py \
    --input satellite_lr.png \
    --output satellite_sr.png
```

### Example 2: Process TIFF files on CPU

```bash
python phase5_final_product/inference.py \
    --input sentinel2_image.tiff \
    --output enhanced.tiff \
    --cpu
```

### Example 3: Batch process a folder

```bash
python phase5_final_product/batch_inference.py \
    --input_dir ./satellite_images \
    --output_dir ./enhanced \
    --format png
```

---

## Model Information

| Attribute | Value |
|-----------|-------|
| Architecture | SwinIR-Small |
| Parameters | 1.24M |
| Scale factor | 4× |
| Input size | 64×64 |
| Output size | 256×256 |
| Test PSNR | 24.49 dB |
| Test SSIM | 0.6501 |

---

## Troubleshooting

### "CUDA out of memory"

Use CPU mode:
```bash
python phase5_final_product/inference.py --input image.png --output sr.png --cpu
```

### "Model checkpoint not found"

Specify the checkpoint path:
```bash
python phase5_final_product/inference.py \
    --input image.png \
    --output sr.png \
    --checkpoint phase3_training/checkpoints/best.pth
```

### "Module not found"

Ensure you're in the project root directory:
```bash
cd /path/to/ml_project
python phase5_final_product/inference.py ...
```

### "Gradio not installed"

Install Gradio for the demo:
```bash
pip install gradio
```

---

## Performance Tips

1. **Use GPU**: 10-20× faster than CPU
2. **Batch processing**: More efficient for multiple images
3. **Smaller tiles**: Reduce memory usage for large images

---

## API Reference

### inference.py

```python
from phase5_final_product.inference import load_model, super_resolve

# Load model
model = load_model("path/to/checkpoint.pth", device)

# Process image
sr_image = super_resolve(model, lr_image, device)
```

### batch_inference.py

```python
from phase5_final_product.batch_inference import get_image_files

# Get list of images
files = get_image_files(Path("./images"))
```

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the TECHNICAL_REPORT.md for details
3. Contact the project maintainers

---

**Version**: 1.0  
**Date**: December 25, 2024

