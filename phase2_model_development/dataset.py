"""
dataset.py - WorldStrat Dataset for Super-Resolution

PyTorch Dataset for loading HR-LR satellite image pairs.
Based on Phase 1 analysis findings.

Usage:
    from phase2_model_development.dataset import WorldStratDataset
    dataset = WorldStratDataset(split='train', scale=4)
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from PIL import Image

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
HR_DIR = DATASET_DIR / "hr_dataset" / "12bit"
LR_DIR = DATASET_DIR / "lr_dataset"
SPLITS_DIR = PROJECT_ROOT / "phase1_data_exploration" / "outputs" / "splits"


class WorldStratDataset(Dataset):
    """
    WorldStrat Dataset for satellite super-resolution.
    
    Args:
        split: 'train', 'val', or 'test'
        scale: Upscaling factor (default: 4)
        hr_patch_size: HR patch size for training (default: 256)
        augment: Apply data augmentation (default: True for train)
        max_cloud: Maximum cloud coverage to include (default: 20.0)
    """
    
    def __init__(
        self,
        split: str = 'train',
        scale: int = 4,
        hr_patch_size: int = 256,
        augment: Optional[bool] = None,
        max_cloud: float = 20.0
    ):
        self.split = split
        self.scale = scale
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale
        self.augment = augment if augment is not None else (split == 'train')
        self.max_cloud = max_cloud
        
        # Load split file
        split_file = SPLITS_DIR / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.locations = [line.strip() for line in f if line.strip()]
        
        print(f"[{split.upper()}] Loaded {len(self.locations)} locations")
    
    def __len__(self) -> int:
        return len(self.locations)
    
    def _load_hr_image(self, location: str) -> np.ndarray:
        """Load HR pan-sharpened image (RGB only)."""
        hr_path = HR_DIR / location / f"{location}_ps.tiff"
        if not hr_path.exists():
            raise FileNotFoundError(f"HR image not found: {hr_path}")
        
        # Load 4-band image, take RGB (first 3 bands)
        img = tifffile.imread(hr_path)
        if img.ndim == 3 and img.shape[2] >= 3:
            img = img[:, :, :3]  # Take RGB
        
        # Normalize 12-bit data to [0, 1]
        # 12-bit max = 4095 (not 65535 for 16-bit)
        # This gives better value range for training
        img = img.astype(np.float32) / 4095.0
        
        # Clip to [0, 1] in case any values exceed 12-bit range
        img = np.clip(img, 0.0, 1.0)
        
        return img
    
    def _load_lr_image(self, location: str, acquisition: int = 1) -> np.ndarray:
        """Load LR Sentinel-2 image (RGB bands)."""
        lr_dir = LR_DIR / location / "L2A"
        lr_path = lr_dir / f"{location}-{acquisition}-L2A_data.tiff"
        
        if not lr_path.exists():
            raise FileNotFoundError(f"LR image not found: {lr_path}")
        
        # Load 12-band Sentinel-2 image
        img = tifffile.imread(lr_path)
        
        # Sentinel-2 L2A bands order (typical):
        # B2 (Blue), B3 (Green), B4 (Red), B5, B6, B7, B8 (NIR), B8A, B11, B12, ...
        # We take first 3 bands as RGB (may need adjustment based on actual order)
        if img.ndim == 3:
            if img.shape[0] < img.shape[2]:
                # Bands-first format (C, H, W)
                img = np.transpose(img[:3], (1, 2, 0))
            else:
                # Bands-last format (H, W, C)
                img = img[:, :, :3]
        
        # Already float32, clip to [0, 1]
        img = np.clip(img.astype(np.float32), 0, 1)
        
        return img
    
    def _get_best_acquisition(self, location: str) -> int:
        """Get acquisition with lowest cloud coverage (simplified: use first)."""
        # TODO: Parse metadata to find lowest cloud acquisition
        return 1
    
    def _random_crop(
        self, 
        hr_img: np.ndarray, 
        lr_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random crop HR and corresponding LR patches."""
        hr_h, hr_w = hr_img.shape[:2]
        lr_h, lr_w = lr_img.shape[:2]
        
        # Calculate valid crop region
        # LR determines the crop region, HR is scaled accordingly
        lr_crop_h = min(self.lr_patch_size, lr_h)
        lr_crop_w = min(self.lr_patch_size, lr_w)
        
        lr_top = random.randint(0, lr_h - lr_crop_h) if lr_h > lr_crop_h else 0
        lr_left = random.randint(0, lr_w - lr_crop_w) if lr_w > lr_crop_w else 0
        
        # Scale to HR coordinates
        hr_crop_h = lr_crop_h * self.scale
        hr_crop_w = lr_crop_w * self.scale
        hr_top = int(lr_top * (hr_h / lr_h))
        hr_left = int(lr_left * (hr_w / lr_w))
        
        # Ensure HR crop is within bounds
        hr_top = min(hr_top, hr_h - hr_crop_h)
        hr_left = min(hr_left, hr_w - hr_crop_w)
        
        # Crop
        lr_crop = lr_img[lr_top:lr_top+lr_crop_h, lr_left:lr_left+lr_crop_w]
        hr_crop = hr_img[hr_top:hr_top+hr_crop_h, hr_left:hr_left+hr_crop_w]
        
        return hr_crop, lr_crop
    
    def _resize_images(
        self,
        hr_img: np.ndarray,
        lr_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize images to target sizes."""
        from PIL import Image
        
        # Resize LR to fixed size
        lr_pil = Image.fromarray((lr_img * 255).astype(np.uint8))
        lr_pil = lr_pil.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        lr_img = np.array(lr_pil).astype(np.float32) / 255.0
        
        # Resize HR to corresponding size
        hr_pil = Image.fromarray((hr_img * 255).astype(np.uint8))
        hr_pil = hr_pil.resize((self.hr_patch_size, self.hr_patch_size), Image.BICUBIC)
        hr_img = np.array(hr_pil).astype(np.float32) / 255.0
        
        return hr_img, lr_img
    
    def _augment(
        self,
        hr_img: np.ndarray,
        lr_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random horizontal flip
        if random.random() > 0.5:
            hr_img = np.fliplr(hr_img).copy()
            lr_img = np.fliplr(lr_img).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            hr_img = np.flipud(hr_img).copy()
            lr_img = np.flipud(lr_img).copy()
        
        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            hr_img = np.rot90(hr_img, k).copy()
            lr_img = np.rot90(lr_img, k).copy()
        
        return hr_img, lr_img
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor (C, H, W)."""
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img.copy()).float()
    
    def __getitem__(self, idx: int) -> dict:
        """Get HR-LR pair."""
        location = self.locations[idx]
        
        try:
            # Load images
            hr_img = self._load_hr_image(location)
            acquisition = self._get_best_acquisition(location)
            lr_img = self._load_lr_image(location, acquisition)
            
            # Resize to target sizes
            hr_img, lr_img = self._resize_images(hr_img, lr_img)
            
            # Random crop for training
            if self.split == 'train':
                hr_img, lr_img = self._random_crop(hr_img, lr_img)
            
            # Augmentation
            if self.augment:
                hr_img, lr_img = self._augment(hr_img, lr_img)
            
            # Convert to tensors
            hr_tensor = self._to_tensor(hr_img)
            lr_tensor = self._to_tensor(lr_img)
            
            return {
                'lr': lr_tensor,
                'hr': hr_tensor,
                'location': location
            }
            
        except Exception as e:
            print(f"Error loading {location}: {e}")
            # Return random valid sample on error
            return self.__getitem__(random.randint(0, len(self) - 1))


def get_dataloader(
    split: str,
    batch_size: int = 4,  # Reduced for 6GB VRAM
    num_workers: int = 8,  # Use all CPU cores for faster loading
    **kwargs
) -> torch.utils.data.DataLoader:
    """Create DataLoader for specified split."""
    dataset = WorldStratDataset(split=split, **kwargs)
    
    shuffle = (split == 'train')
    
    # Build DataLoader kwargs
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': True,  # Faster GPU transfer with NVMe SSD
        'drop_last': (split == 'train'),
    }
    
    # prefetch_factor only works with num_workers > 0
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 2  # Prefetch batches while GPU is busy
        dataloader_kwargs['persistent_workers'] = True  # Keep workers alive
    
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


if __name__ == "__main__":
    # Test dataset
    print("Testing WorldStratDataset...")
    
    dataset = WorldStratDataset(split='train', scale=4, hr_patch_size=256)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"LR shape: {sample['lr'].shape}")
    print(f"HR shape: {sample['hr'].shape}")
    print(f"Location: {sample['location']}")

