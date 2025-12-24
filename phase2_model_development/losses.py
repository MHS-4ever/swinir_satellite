"""
losses.py - Loss Functions for Super-Resolution

Implements various loss functions:
- L1 Loss (pixel-wise)
- L2 Loss (MSE)
- Charbonnier Loss
- Perceptual Loss (VGG-based)

Usage:
    from phase2_model_development.losses import L1Loss, CharbonnierLoss, PerceptualLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.models as models


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) Loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class L2Loss(nn.Module):
    """L2 (Mean Squared Error) Loss."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (differentiable variant of L1).
    
    L = sqrt((pred - target)^2 + eps^2)
    
    More robust to outliers than L1.
    """
    
    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class VGGFeatureExtractor(nn.Module):
    """VGG network for extracting features for perceptual loss."""
    
    def __init__(
        self,
        layer_name_list: List[str] = ['conv3_4', 'conv4_4', 'conv5_4'],
        use_input_norm: bool = True,
        requires_grad: bool = False
    ):
        super().__init__()
        
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Layer name to index mapping for VGG19
        self.layer_name_mapping = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16,
            'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25,
            'conv5_1': 28, 'conv5_2': 30, 'conv5_3': 32, 'conv5_4': 34
        }
        
        # Find max layer index needed
        max_idx = 0
        for layer_name in layer_name_list:
            idx = self.layer_name_mapping.get(layer_name, 0)
            max_idx = max(max_idx, idx)
        
        # Extract features up to max layer
        self.features = nn.Sequential(*list(vgg.features.children())[:max_idx + 1])
        
        # Freeze weights
        if not requires_grad:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # ImageNet normalization
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
    
    def forward(self, x: torch.Tensor) -> dict:
        """Extract features from specified layers."""
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        
        features = {}
        for name, module in self.features._modules.items():
            x = module(x)
            idx = int(name)
            for layer_name, layer_idx in self.layer_name_mapping.items():
                if layer_idx == idx and layer_name in self.layer_name_list:
                    features[layer_name] = x
        
        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG features.
    
    Computes L1/L2 loss between VGG features of predicted and target images.
    """
    
    def __init__(
        self,
        layer_weights: Optional[dict] = None,
        criterion: str = 'l1',
        use_input_norm: bool = True
    ):
        super().__init__()
        
        # Default layer weights
        if layer_weights is None:
            layer_weights = {'conv5_4': 1.0}
        
        self.layer_weights = layer_weights
        layer_names = list(layer_weights.keys())
        
        self.vgg = VGGFeatureExtractor(
            layer_name_list=layer_names,
            use_input_norm=use_input_norm,
            requires_grad=False
        )
        
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        # Extract features
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        # Compute weighted loss
        loss = 0
        for layer_name, weight in self.layer_weights.items():
            loss += weight * self.criterion(
                pred_features[layer_name],
                target_features[layer_name].detach()
            )
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for super-resolution.
    
    L = pixel_weight * L_pixel + perceptual_weight * L_perceptual
    """
    
    def __init__(
        self,
        pixel_loss: str = 'l1',
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        perceptual_layers: Optional[dict] = None
    ):
        super().__init__()
        
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        
        # Pixel loss
        if pixel_loss == 'l1':
            self.pixel_criterion = L1Loss()
        elif pixel_loss == 'l2':
            self.pixel_criterion = L2Loss()
        elif pixel_loss == 'charbonnier':
            self.pixel_criterion = CharbonnierLoss()
        else:
            raise ValueError(f"Unknown pixel loss: {pixel_loss}")
        
        # Perceptual loss
        if perceptual_weight > 0:
            self.perceptual_criterion = PerceptualLoss(
                layer_weights=perceptual_layers or {'conv5_4': 1.0}
            )
        else:
            self.perceptual_criterion = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Compute combined loss and return breakdown."""
        losses = {}
        
        # Pixel loss
        pixel_loss = self.pixel_criterion(pred, target)
        losses['pixel'] = pixel_loss
        
        total_loss = self.pixel_weight * pixel_loss
        
        # Perceptual loss
        if self.perceptual_criterion is not None and self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_criterion(pred, target)
            losses['perceptual'] = perceptual_loss
            total_loss += self.perceptual_weight * perceptual_loss
        
        losses['total'] = total_loss
        
        return losses


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")
    
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    
    # L1 Loss
    l1_loss = L1Loss()
    print(f"L1 Loss: {l1_loss(pred, target).item():.4f}")
    
    # Charbonnier Loss
    char_loss = CharbonnierLoss()
    print(f"Charbonnier Loss: {char_loss(pred, target).item():.4f}")
    
    # Combined Loss
    combined = CombinedLoss(pixel_loss='l1', pixel_weight=1.0, perceptual_weight=0.1)
    losses = combined(pred, target)
    print(f"Combined Loss - Pixel: {losses['pixel'].item():.4f}, Total: {losses['total'].item():.4f}")

