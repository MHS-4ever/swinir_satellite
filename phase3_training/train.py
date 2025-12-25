"""
train.py - Training Script for SwinIR Satellite Super-Resolution

Features:
- Mixed Precision Training (AMP) for RTX 3050 (6GB VRAM)
- Gradient Accumulation for effective larger batch size
- Cosine Annealing LR scheduler
- TensorBoard logging
- Checkpoint saving (best + latest)
- Validation with PSNR/SSIM metrics

Usage:
    python phase3_training/train.py --epochs 100
    python phase3_training/train.py --resume checkpoints/latest.pth
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from phase2_model_development import (
    Config, swinir_small, WorldStratDataset, get_dataloader,
    CombinedLoss, save_checkpoint, load_checkpoint, set_seed, get_device
)
from phase3_training.metrics import MetricsCalculator, calculate_psnr, calculate_ssim


def parse_args():
    parser = argparse.ArgumentParser(description='Train SwinIR for Satellite SR')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    return parser.parse_args()


class Trainer:
    """Training class with all optimizations."""
    
    def __init__(self, config: Config, exp_name: str = None):
        self.config = config
        self.device = get_device()
        
        # Create directories
        self.exp_name = exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = PROJECT_ROOT / "phase3_training" / "checkpoints"
        self.log_dir = PROJECT_ROOT / "phase3_training" / "runs" / self.exp_name
        self.sample_dir = PROJECT_ROOT / "phase3_training" / "outputs" / "samples" / self.exp_name
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        print("\n[1/4] Creating model...")
        self.model = swinir_small(
            upscale=config.upscale,
            in_chans=config.in_channels,
            img_size=config.img_size,
            window_size=config.window_size
        ).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  Model parameters: {n_params:.2f}M")
        
        # Initialize optimizer
        print("\n[2/4] Creating optimizer...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.99)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr
        )
        
        # Loss function
        self.criterion = CombinedLoss(
            pixel_loss=config.pixel_loss,
            pixel_weight=config.pixel_weight,
            perceptual_weight=config.perceptual_weight
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda', enabled=config.use_amp)
        self.use_amp = config.use_amp
        
        # Metrics calculator
        self.metrics = MetricsCalculator()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_psnr = 0.0
        
        print(f"\n[3/4] Creating data loaders...")
        
    def create_dataloaders(self):
        """Create train and validation dataloaders."""
        self.train_loader = get_dataloader(
            split='train',
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            scale=self.config.upscale,
            hr_patch_size=self.config.hr_patch_size
        )
        
        self.val_loader = get_dataloader(
            split='val',
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            scale=self.config.upscale,
            hr_patch_size=self.config.hr_patch_size
        )
        
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
    
    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        print(f"\nResuming from: {checkpoint_path}")
        checkpoint = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            device=str(self.device)
        )
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        print(f"  Resumed at epoch {self.start_epoch}, best PSNR: {self.best_psnr:.2f}")
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        accum_steps = self.config.accumulation_steps
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            lr_img = batch['lr'].to(self.device, non_blocking=True)
            hr_img = batch['hr'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.use_amp):
                sr_img = self.model.forward_simple(lr_img)
                losses = self.criterion(sr_img, hr_img)
                loss = losses['total'] / accum_steps  # Scale for accumulation
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Optimizer step every accum_steps
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track loss
            epoch_loss += losses['total'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to TensorBoard
            if self.global_step % self.config.log_every == 0:
                self.writer.add_scalar('train/loss', losses['total'].item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Clear cache periodically
            if batch_idx % self.config.empty_cache_freq == 0:
                torch.cuda.empty_cache()
        
        return {'loss': epoch_loss / num_batches}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate the model."""
        self.model.eval()
        self.metrics.reset()
        
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating", ncols=100)):
            lr_img = batch['lr'].to(self.device, non_blocking=True)
            hr_img = batch['hr'].to(self.device, non_blocking=True)
            
            with autocast('cuda', enabled=self.use_amp):
                sr_img = self.model.forward_simple(lr_img)
                losses = self.criterion(sr_img, hr_img)
            
            val_loss += losses['total'].item()
            
            # Clamp output for metrics
            sr_img = torch.clamp(sr_img, 0, 1)
            self.metrics.update(sr_img, hr_img)
            
            # Save sample images (first batch only)
            if batch_idx == 0 and epoch % 10 == 0:
                self._save_samples(lr_img, sr_img, hr_img, epoch)
        
        metrics = self.metrics.get_metrics()
        metrics['loss'] = val_loss / num_batches
        
        return metrics
    
    def _save_samples(self, lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor, epoch: int):
        """Save sample images for visualization."""
        import torchvision.utils as vutils
        
        # Take first 4 images
        n = min(4, lr.shape[0])
        
        # Upsample LR for comparison
        lr_up = torch.nn.functional.interpolate(lr[:n], scale_factor=4, mode='bicubic', align_corners=False)
        lr_up = torch.clamp(lr_up, 0, 1)
        
        # Create comparison grid: LR_upscaled | SR | HR
        comparison = torch.cat([lr_up, sr[:n], hr[:n]], dim=0)
        grid = vutils.make_grid(comparison, nrow=n, normalize=False, padding=2)
        
        # Save image
        save_path = self.sample_dir / f"epoch_{epoch:04d}.png"
        vutils.save_image(grid, save_path)
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print("\n[4/4] Starting training...")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {self.config.batch_size} (effective: {self.config.get_effective_batch_size()})")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Device: {self.device}")
        print("=" * 60)
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Validate
            if (epoch + 1) % self.config.val_every == 0:
                val_metrics = self.validate(epoch)
                
                # Log to TensorBoard
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
                self.writer.add_scalar('val/ssim', val_metrics['ssim'], epoch)
                
                # Print epoch summary
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB")
                print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
                
                # Save best model
                if val_metrics['psnr'] > self.best_psnr:
                    self.best_psnr = val_metrics['psnr']
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        val_metrics['loss'],
                        str(self.checkpoint_dir / 'best.pth'),
                        scheduler=self.scheduler,
                        metrics=val_metrics,
                        global_step=self.global_step,
                        best_psnr=self.best_psnr
                    )
                    print(f"  [NEW BEST] PSNR: {self.best_psnr:.2f} dB")
            
            # Save latest checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    train_metrics['loss'],
                    str(self.checkpoint_dir / 'latest.pth'),
                    scheduler=self.scheduler,
                    global_step=self.global_step,
                    best_psnr=self.best_psnr
                )
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {self.log_dir}")
        
        self.writer.close()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SwinIR Satellite Super-Resolution Training")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Load config (auto-detects hardware)
    config = Config()
    config.epochs = args.epochs
    config.print_config()
    
    # Create trainer
    trainer = Trainer(config, exp_name=args.exp_name)
    trainer.create_dataloaders()
    
    # Resume if specified
    if args.resume:
        trainer.resume(args.resume)
    
    # Train
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()

