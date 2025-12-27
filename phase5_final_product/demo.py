"""
demo.py - Interactive Gradio Demo for SwinIR Super-Resolution

A web-based interface for testing the super-resolution model.

Usage:
    python phase5_final_product/demo.py
    
Then open http://localhost:7860 in your browser.

Requirements:
    pip install gradio
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Fix Windows asyncio issues
if sys.platform == 'win32':
    import warnings
    import logging
    # Suppress asyncio warnings on Windows
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*proactor.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*connection.*')
    # Suppress connection reset errors in logs
    logging.getLogger('asyncio').setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check for gradio
try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Install with: pip install gradio")
    print("Running in CLI mode instead...")
    gr = None

from phase2_model_development.swinir import swinir_small


class SwinIRDemo:
    """Super-Resolution Demo using SwinIR."""
    
    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Find checkpoint
        if checkpoint_path is None:
            paths = [
                PROJECT_ROOT / "phase5_final_product" / "release" / "model" / "swinir_satellite_sr_x4.pth",
                PROJECT_ROOT / "phase3_training" / "checkpoints" / "best.pth",
            ]
            for p in paths:
                if p.exists():
                    checkpoint_path = str(p)
                    break
        
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found")
        
        print(f"Loading model from: {checkpoint_path}")
        print(f"Device: {self.device}")
        
        self.model = swinir_small(upscale=4, in_chans=3, img_size=64, window_size=8)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model."""
        # Ensure RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image.astype(np.float32) / image.max()
        
        # Resize to 64x64
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        img_pil = img_pil.resize((64, 64), Image.BICUBIC)
        image = np.array(img_pil).astype(np.float32) / 255.0
        
        # To tensor
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return tensor.unsqueeze(0).to(self.device)
    
    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to image."""
        img = tensor.squeeze(0).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img
    
    def bicubic_upscale(self, image: np.ndarray) -> np.ndarray:
        """Bicubic baseline."""
        img_pil = Image.fromarray(image)
        img_pil = img_pil.resize((64, 64), Image.BICUBIC)
        img_pil = img_pil.resize((256, 256), Image.BICUBIC)
        return np.array(img_pil)
    
    def super_resolve(self, image: np.ndarray) -> tuple:
        """
        Super-resolve an image.
        
        Returns:
            tuple: (lr_image, bicubic_image, sr_image)
        """
        if image is None:
            return None, None, None
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Get LR version
        lr_img = input_tensor.squeeze(0).cpu().numpy()
        lr_img = np.transpose(lr_img, (1, 2, 0))
        lr_img = np.clip(lr_img * 255, 0, 255).astype(np.uint8)
        lr_img_up = np.array(Image.fromarray(lr_img).resize((256, 256), Image.NEAREST))
        
        # Bicubic baseline
        bicubic_img = np.array(Image.fromarray(lr_img).resize((256, 256), Image.BICUBIC))
        
        # SwinIR super-resolution
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.clamp(output, 0, 1)
        
        sr_img = self.postprocess(output)
        
        return lr_img_up, bicubic_img, sr_img
    
    def launch_gradio(self, share: bool = False, port: int = 7860):
        """Launch Gradio interface."""
        if gr is None:
            print("Gradio not available. Please install with: pip install gradio")
            return
        
        def process(image):
            lr, bicubic, sr = self.super_resolve(image)
            return lr, bicubic, sr
        
        # Create interface
        with gr.Blocks(title="SwinIR Satellite Super-Resolution") as demo:
            gr.Markdown("""
            # SwinIR Satellite Super-Resolution
            
            Upload a satellite image to upscale it by 4x using the trained SwinIR model.
            
            **Model**: SwinIR-Small (1.24M parameters)  
            **Scale**: 4x (64x64 -> 256x256)  
            **Test PSNR**: 24.49 dB (+6.62 dB vs bicubic)
            """)
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input Image", type="numpy")
                    process_btn = gr.Button("Super-Resolve", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### Results")
            
            with gr.Row():
                lr_output = gr.Image(label="LR Input (64x64, upscaled for display)")
                bicubic_output = gr.Image(label="Bicubic 4x (Baseline)")
                sr_output = gr.Image(label="SwinIR 4x (Ours)")
            
            process_btn.click(
                fn=process,
                inputs=input_image,
                outputs=[lr_output, bicubic_output, sr_output]
            )
            
            gr.Markdown("""
            ---
            ### About
            
            This model was trained on the WorldStrat satellite imagery dataset for the 
            ML course project on satellite image super-resolution.
            
            **Training Details**:
            - Architecture: SwinIR-Small
            - Training epochs: 100
            - Training data: 3,140 satellite images
            - Validation PSNR: 23.95 dB
            - Test PSNR: 24.49 dB
            """)
        
        print(f"\nLaunching demo at http://localhost:{port}")
        print("Note: If you see a connection reset error when closing, it's a harmless Windows cleanup issue.\n")
        
        try:
            demo.launch(share=share, server_port=port, show_error=True)
        except KeyboardInterrupt:
            print("\n\nDemo stopped by user. Shutting down...")
        except Exception as e:
            # Suppress Windows connection reset errors during cleanup
            if sys.platform == 'win32':
                if isinstance(e, ConnectionResetError):
                    # Connection reset during cleanup - harmless on Windows
                    print("\nDemo closed successfully.")
                    return
                elif isinstance(e, OSError) and hasattr(e, 'winerror') and e.winerror == 10054:
                    print("\nDemo closed successfully.")
                    return
            # Re-raise other errors
            raise


def main():
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description='SwinIR Demo')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Model checkpoint path')
    parser.add_argument('--port', '-p', type=int, default=7860,
                       help='Server port')
    parser.add_argument('--share', action='store_true',
                       help='Create public link')
    args = parser.parse_args()
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print('\n\nShutting down demo...')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != 'win32':
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        demo = SwinIRDemo(checkpoint_path=args.checkpoint)
        demo.launch_gradio(share=args.share, port=args.port)
    except KeyboardInterrupt:
        print('\n\nDemo stopped by user.')
    except Exception as e:
        if sys.platform == 'win32' and isinstance(e, ConnectionResetError):
            # Windows connection reset during cleanup - safe to ignore
            pass
        else:
            print(f"\nError: {e}")
            raise


if __name__ == '__main__':
    main()

