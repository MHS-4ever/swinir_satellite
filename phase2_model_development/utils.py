"""
utils.py - Model Utilities

Utility functions for:
- Saving/loading checkpoints
- Model parameter counting
- Learning rate scheduling
- Logging

Usage:
    from phase2_model_development.utils import save_checkpoint, load_checkpoint
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    **kwargs
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        metrics: Optional validation metrics
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    checkpoint.update(kwargs)
    
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load to
        
    Returns:
        Dictionary with checkpoint info (epoch, loss, metrics, etc.)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded: {filepath} (epoch {checkpoint.get('epoch', 'N/A')})")
    
    return checkpoint


def save_model_only(model: nn.Module, filepath: str):
    """Save only model weights (for inference)."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved: {filepath}")


def load_model_only(model: nn.Module, filepath: str, device: str = 'cpu'):
    """Load only model weights."""
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded: {filepath}")
    return model


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    from swinir import swinir_small
    
    model = swinir_small(upscale=4)
    params = count_parameters(model)
    print(f"Total parameters: {params['total'] / 1e6:.2f}M")
    print(f"Trainable parameters: {params['trainable'] / 1e6:.2f}M")
    
    # Test device
    device = get_device()
    
    # Test seed
    set_seed(42)

