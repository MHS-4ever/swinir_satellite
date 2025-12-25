"""
generate_visualizations.py - Generate detailed visual comparisons

Creates high-quality visualizations with zoomed crops for report figures.

Usage:
    python phase4_evaluation/generate_visualizations.py --checkpoint phase3_training/checkpoints/best.pth
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase2_model_development.swinir import swinir_small
from phase2_model_development.dataset import WorldStratDataset
from phase3_training.metrics import calculate_psnr, calculate_ssim


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model."""
    model = swinir_small(upscale=4, in_chans=3, img_size=64, window_size=8)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def bicubic_upsample(lr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    return F.interpolate(lr, scale_factor=scale, mode='bicubic', align_corners=False)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.squeeze().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def create_detailed_comparison(
    lr_img: np.ndarray,
    bicubic_img: np.ndarray,
    sr_img: np.ndarray,
    hr_img: np.ndarray,
    location: str,
    psnr_bicubic: float,
    psnr_sr: float,
    ssim_bicubic: float,
    ssim_sr: float,
    save_path: Path,
    crop_box: tuple = (80, 80, 176, 176)  # x1, y1, x2, y2 for crop region
):
    """Create detailed comparison with zoomed crops."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.6], hspace=0.15, wspace=0.05)
    
    # Top row: Full images
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Bottom row: Cropped regions
    ax5 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[1, 1])
    ax7 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[1, 3])
    
    # Upscale LR for display
    lr_upscaled = np.array(Image.fromarray(lr_img).resize((256, 256), Image.NEAREST))
    
    # Full images
    ax1.imshow(lr_upscaled)
    ax1.set_title('LR Input (64x64)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(bicubic_img)
    ax2.set_title(f'Bicubic 4x\nPSNR: {psnr_bicubic:.2f} dB | SSIM: {ssim_bicubic:.4f}', fontsize=12)
    ax2.axis('off')
    
    ax3.imshow(sr_img)
    ax3.set_title(f'SwinIR 4x\nPSNR: {psnr_sr:.2f} dB | SSIM: {ssim_sr:.4f}', fontsize=12)
    ax3.axis('off')
    
    ax4.imshow(hr_img)
    ax4.set_title('HR Ground Truth (256x256)', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Add crop rectangles
    x1, y1, x2, y2 = crop_box
    for ax in [ax1, ax2, ax3, ax4]:
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    # Cropped regions
    lr_crop_box = (x1//4, y1//4, x2//4, y2//4)
    lr_crop = lr_img[lr_crop_box[1]:lr_crop_box[3], lr_crop_box[0]:lr_crop_box[2]]
    lr_crop_up = np.array(Image.fromarray(lr_crop).resize((96, 96), Image.NEAREST))
    
    bicubic_crop = bicubic_img[y1:y2, x1:x2]
    sr_crop = sr_img[y1:y2, x1:x2]
    hr_crop = hr_img[y1:y2, x1:x2]
    
    ax5.imshow(lr_crop_up)
    ax5.set_title('LR Crop (Zoomed)', fontsize=12)
    ax5.axis('off')
    
    ax6.imshow(bicubic_crop)
    ax6.set_title('Bicubic Crop', fontsize=12)
    ax6.axis('off')
    
    ax7.imshow(sr_crop)
    ax7.set_title('SwinIR Crop', fontsize=12)
    ax7.axis('off')
    
    ax8.imshow(hr_crop)
    ax8.set_title('HR Crop', fontsize=12)
    ax8.axis('off')
    
    # Super title
    improvement = psnr_sr - psnr_bicubic
    fig.suptitle(f'{location}\nPSNR Improvement: +{improvement:.2f} dB', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def create_grid_comparison(
    samples: list,
    save_path: Path,
    title: str = "Super-Resolution Results"
):
    """Create grid of multiple samples for paper figure."""
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    
    for i, sample in enumerate(samples):
        if n == 1:
            ax_row = axes
        else:
            ax_row = axes[i]
        
        lr_up = np.array(Image.fromarray(sample['lr']).resize((256, 256), Image.NEAREST))
        
        ax_row[0].imshow(lr_up)
        ax_row[0].set_title('LR' if i == 0 else '', fontsize=12)
        ax_row[0].axis('off')
        if i == 0:
            ax_row[0].set_ylabel('Input', fontsize=12)
        
        ax_row[1].imshow(sample['bicubic'])
        ax_row[1].set_title(f'Bicubic\n{sample["psnr_bicubic"]:.2f} dB' if i == 0 else f'{sample["psnr_bicubic"]:.2f} dB', fontsize=11)
        ax_row[1].axis('off')
        
        ax_row[2].imshow(sample['sr'])
        ax_row[2].set_title(f'SwinIR\n{sample["psnr_sr"]:.2f} dB' if i == 0 else f'{sample["psnr_sr"]:.2f} dB', fontsize=11)
        ax_row[2].axis('off')
        
        ax_row[3].imshow(sample['hr'])
        ax_row[3].set_title('HR (GT)' if i == 0 else '', fontsize=12)
        ax_row[3].axis('off')
        
        # Add location label
        ax_row[0].text(-0.1, 0.5, sample['location'][:20] + '...', 
                      transform=ax_row[0].transAxes, fontsize=10, 
                      verticalalignment='center', rotation=90)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def create_improvement_showcase(
    samples: list,
    save_path: Path
):
    """Create showcase comparing bicubic vs SwinIR side by side."""
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5*n))
    
    for i, sample in enumerate(samples):
        if n == 1:
            ax_row = axes
        else:
            ax_row = axes[i]
        
        # Bicubic
        ax_row[0].imshow(sample['bicubic'])
        ax_row[0].set_title(f'Bicubic: {sample["psnr_bicubic"]:.2f} dB', fontsize=14)
        ax_row[0].axis('off')
        
        # SwinIR
        ax_row[1].imshow(sample['sr'])
        improvement = sample['psnr_sr'] - sample['psnr_bicubic']
        ax_row[1].set_title(f'SwinIR: {sample["psnr_sr"]:.2f} dB (+{improvement:.2f} dB)', fontsize=14, color='green')
        ax_row[1].axis('off')
        
        # HR
        ax_row[2].imshow(sample['hr'])
        ax_row[2].set_title('Ground Truth', fontsize=14)
        ax_row[2].axis('off')
    
    plt.suptitle('Bicubic vs SwinIR Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='phase3_training/checkpoints/best.pth')
    parser.add_argument('--output_dir', type=str,
                       default='phase4_evaluation/outputs/figures')
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Detailed Visualizations")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(str(PROJECT_ROOT / args.checkpoint), device)
    
    # Load test dataset
    dataset = WorldStratDataset(split='test', scale=4, hr_patch_size=256, augment=False)
    
    # Collect results
    all_results = []
    
    # Sample evenly from dataset
    indices = np.linspace(0, len(dataset)-1, args.num_samples, dtype=int)
    
    print(f"\nProcessing {len(indices)} samples...")
    
    for idx in tqdm(indices):
        sample = dataset[idx]
        lr = sample['lr'].unsqueeze(0).to(device)
        hr = sample['hr']
        
        with torch.no_grad():
            sr = model(lr)
            sr = torch.clamp(sr, 0, 1)
            bicubic = bicubic_upsample(lr)
            bicubic = torch.clamp(bicubic, 0, 1)
        
        hr_t = hr.unsqueeze(0).to(device)
        
        psnr_sr = calculate_psnr(sr, hr_t)
        ssim_sr = calculate_ssim(sr, hr_t)
        psnr_bicubic = calculate_psnr(bicubic, hr_t)
        ssim_bicubic = calculate_ssim(bicubic, hr_t)
        
        # Handle both tensor and float returns
        if hasattr(psnr_sr, 'item'):
            psnr_sr = psnr_sr.item()
        if hasattr(ssim_sr, 'item'):
            ssim_sr = ssim_sr.item()
        if hasattr(psnr_bicubic, 'item'):
            psnr_bicubic = psnr_bicubic.item()
        if hasattr(ssim_bicubic, 'item'):
            ssim_bicubic = ssim_bicubic.item()
        
        result = {
            'idx': idx,
            'location': sample['location'],
            'lr': tensor_to_image(sample['lr']),
            'bicubic': tensor_to_image(bicubic.squeeze(0)),
            'sr': tensor_to_image(sr.squeeze(0)),
            'hr': tensor_to_image(hr),
            'psnr_sr': psnr_sr,
            'ssim_sr': ssim_sr,
            'psnr_bicubic': psnr_bicubic,
            'ssim_bicubic': ssim_bicubic,
        }
        all_results.append(result)
        
        # Create detailed comparison
        create_detailed_comparison(
            result['lr'], result['bicubic'], result['sr'], result['hr'],
            result['location'],
            result['psnr_bicubic'], result['psnr_sr'],
            result['ssim_bicubic'], result['ssim_sr'],
            output_dir / f'detailed_{idx:04d}_{sample["location"][:30]}.png'
        )
    
    # Sort by PSNR
    sorted_results = sorted(all_results, key=lambda x: x['psnr_sr'], reverse=True)
    
    # Create grid of best results
    print("\nCreating grid visualizations...")
    create_grid_comparison(
        sorted_results[:6],
        output_dir / 'grid_best_6.png',
        title='Top 6 Best Results'
    )
    
    create_grid_comparison(
        sorted_results[-6:],
        output_dir / 'grid_worst_6.png',
        title='6 Most Challenging Samples'
    )
    
    # Create improvement showcase
    create_improvement_showcase(
        sorted_results[:4],
        output_dir / 'improvement_showcase.png'
    )
    
    # Create category samples
    categories = {}
    for r in all_results:
        cat = r['location'].split('-')[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    for cat, cat_results in categories.items():
        if len(cat_results) > 0:
            create_grid_comparison(
                cat_results[:min(3, len(cat_results))],
                output_dir / f'category_{cat}.png',
                title=f'Category: {cat}'
            )
    
    print(f"\n[OK] Visualizations saved to: {output_dir}")
    print(f"  - Detailed comparisons: {len(all_results)} images")
    print(f"  - Grid visualizations: 4 images")
    print(f"  - Category samples: {len(categories)} images")


if __name__ == '__main__':
    main()

