"""
metrics.py - Evaluation Metrics for Super-Resolution

Implements:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

Usage:
    from phase3_training.metrics import calculate_psnr, calculate_ssim
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0
) -> float:
    """
    Calculate PSNR between predicted and target images.
    
    Args:
        pred: Predicted image tensor (B, C, H, W) or (C, H, W)
        target: Target image tensor
        max_val: Maximum pixel value (1.0 for normalized images)
        
    Returns:
        PSNR value in dB
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    
    mse = F.mse_loss(pred, target, reduction='mean')
    
    if mse == 0:
        return float('inf')
    
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr.item()


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0
) -> float:
    """
    Calculate SSIM between predicted and target images.
    
    Args:
        pred: Predicted image tensor (B, C, H, W)
        target: Target image tensor
        window_size: Size of the gaussian window
        max_val: Maximum pixel value
        
    Returns:
        SSIM value (0 to 1, higher is better)
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    C = pred.shape[1]
    
    # Create gaussian window
    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.view(1, -1) * g.view(-1, 1)
    
    window = gaussian_window(window_size)
    window = window.expand(C, 1, window_size, window_size).to(pred.device)
    
    # Constants for stability
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2
    
    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    
    return ssim_map.mean().item()


class MetricsCalculator:
    """Calculate and track metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.count = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with a batch."""
        with torch.no_grad():
            # Clamp predictions to valid range
            pred = torch.clamp(pred, 0, 1)
            
            batch_size = pred.shape[0]
            
            # Calculate metrics for each image in batch
            for i in range(batch_size):
                self.psnr_sum += calculate_psnr(pred[i], target[i])
                self.ssim_sum += calculate_ssim(pred[i].unsqueeze(0), target[i].unsqueeze(0))
                self.count += 1
    
    def get_metrics(self) -> dict:
        """Get average metrics."""
        if self.count == 0:
            return {'psnr': 0.0, 'ssim': 0.0}
        
        return {
            'psnr': self.psnr_sum / self.count,
            'ssim': self.ssim_sum / self.count
        }


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Create random tensors
    pred = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)
    
    psnr = calculate_psnr(pred, target)
    ssim = calculate_ssim(pred, target)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    # Test with identical images
    identical = torch.rand(2, 3, 256, 256)
    psnr_identical = calculate_psnr(identical, identical)
    ssim_identical = calculate_ssim(identical, identical)
    
    print(f"\nIdentical images:")
    print(f"PSNR: {psnr_identical} (should be inf)")
    print(f"SSIM: {ssim_identical:.4f} (should be ~1.0)")

