"""
Image quality metrics for validation and evaluation.

Provides perceptual and pixel-based metrics for assessing generated image quality:
- LPIPS: Learned Perceptual Image Patch Similarity
- MSE: Mean Squared Error
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index (optional, requires scikit-image)
"""

import torch
import logging
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class LPIPSMetric:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) metric.
    
    Measures perceptual similarity between images using a pretrained network.
    Lower values indicate higher perceptual similarity.
    
    Reference: https://arxiv.org/abs/1801.03924
    """
    
    def __init__(
        self, 
        net: str = 'alex',  # 'alex', 'vgg', 'squeeze'
        device: Optional[torch.device] = None,
        use_dropout: bool = False,
    ):
        """
        Initialize LPIPS metric.
        
        Args:
            net: Network backbone ('alex', 'vgg', or 'squeeze')
                 - 'alex' (AlexNet): Fastest, good for most cases
                 - 'vgg' (VGG16): More accurate, slower
                 - 'squeeze' (SqueezeNet): Fastest, lower accuracy
            device: Device to run on (defaults to cuda if available)
            use_dropout: Whether to use dropout in network (for uncertainty estimation)
        """
        try:
            import lpips
        except ImportError:
            raise ImportError(
                "lpips package not found. Install it with: pip install lpips"
            )
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = lpips.LPIPS(net=net, verbose=False).to(self.device)
        self.model.eval()
        self.net_name = net
        
        logger.info(f"Initialized LPIPS metric with {net} backbone on {self.device}")
    
    @torch.no_grad()
    def compute(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute LPIPS distance between two images.
        
        Args:
            img1: First image tensor [B, C, H, W] or [C, H, W]
            img2: Second image tensor [B, C, H, W] or [C, H, W]
            normalize: If True, assumes images are in [-1, 1] range (recommended)
                      If False, assumes images are in [0, 1] range
        
        Returns:
            LPIPS distance (lower is more similar). Shape: [B] or scalar
        """
        # Handle single images
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        # Ensure same shape
        if img1.shape != img2.shape:
            raise ValueError(
                f"Image shapes must match. Got {img1.shape} and {img2.shape}"
            )
        
        # Move to device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # LPIPS expects images in [-1, 1] range
        if not normalize:
            # Convert from [0, 1] to [-1, 1]
            img1 = img1 * 2 - 1
            img2 = img2 * 2 - 1
        
        # Compute LPIPS
        distance = self.model(img1, img2)
        
        # Return scalar if single image, otherwise keep batch dimension
        if distance.numel() == 1:
            return distance.squeeze()
        return distance.squeeze(1).squeeze(1).squeeze(1)  # Remove spatial dims
    
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convenience method for computing LPIPS."""
        return self.compute(img1, img2, **kwargs)


def compute_mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error between two images.
    
    Args:
        img1: First image tensor [B, C, H, W] or [C, H, W]
        img2: Second image tensor [B, C, H, W] or [C, H, W]
    
    Returns:
        MSE value (lower is better). Shape: [B] or scalar
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match. Got {img1.shape} and {img2.shape}")
    
    mse = torch.nn.functional.mse_loss(img1, img2, reduction='none')
    
    # Average over spatial and channel dimensions
    if img1.dim() == 4:  # Batch
        mse = mse.mean(dim=[1, 2, 3])
    else:  # Single image
        mse = mse.mean()
    
    return mse


def compute_psnr(
    img1: torch.Tensor, 
    img2: torch.Tensor, 
    max_value: float = 1.0
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: First image tensor [B, C, H, W] or [C, H, W]
        img2: Second image tensor [B, C, H, W] or [C, H, W]
        max_value: Maximum possible pixel value (1.0 for normalized images)
    
    Returns:
        PSNR in dB (higher is better). Shape: [B] or scalar
    """
    mse = compute_mse(img1, img2)
    
    # Avoid division by zero
    mse = torch.clamp(mse, min=1e-10)
    
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr


class ImageMetrics:
    """
    Collection of image quality metrics for validation.
    
    Computes multiple metrics in one pass for efficiency.
    """
    
    def __init__(
        self,
        compute_lpips: bool = True,
        lpips_net: str = 'alex',
        device: Optional[torch.device] = None,
    ):
        """
        Initialize metrics collection.
        
        Args:
            compute_lpips: Whether to compute LPIPS (requires model loading)
            lpips_net: LPIPS network backbone ('alex', 'vgg', 'squeeze')
            device: Device to run on
        """
        self.compute_lpips_enabled = compute_lpips
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.lpips_metric = None
        if compute_lpips:
            try:
                self.lpips_metric = LPIPSMetric(net=lpips_net, device=self.device)
            except Exception as e:
                logger.warning(f"Failed to initialize LPIPS metric: {e}")
                logger.warning("LPIPS metrics will be disabled")
                self.compute_lpips_enabled = False
    
    @torch.no_grad()
    def compute_all(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        normalize_lpips: bool = True,
        max_value: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute all available metrics between generated and target images.
        
        Args:
            generated: Generated image tensor [B, C, H, W] or [C, H, W]
            target: Target image tensor [B, C, H, W] or [C, H, W]
            normalize_lpips: If True, images are in [-1, 1] for LPIPS
            max_value: Max pixel value for PSNR calculation
        
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # Ensure same shape
        if generated.shape != target.shape:
            logger.warning(
                f"Shape mismatch: generated {generated.shape} vs target {target.shape}. "
                "Skipping metric computation."
            )
            return metrics
        
        # Ensure both tensors are on the same device (move to generated's device)
        device = generated.device
        if target.device != device:
            target = target.to(device)
        
        # MSE
        try:
            mse = compute_mse(generated, target)
            if mse.dim() == 0:  # Scalar
                metrics['mse'] = mse.item()
            else:  # Batch
                metrics['mse'] = mse.mean().item()
        except Exception as e:
            logger.warning(f"Failed to compute MSE: {e}")
        
        # PSNR
        try:
            psnr = compute_psnr(generated, target, max_value=max_value)
            if psnr.dim() == 0:
                metrics['psnr'] = psnr.item()
            else:
                metrics['psnr'] = psnr.mean().item()
        except Exception as e:
            logger.warning(f"Failed to compute PSNR: {e}")
        
        # LPIPS
        if self.compute_lpips_enabled and self.lpips_metric is not None:
            try:
                lpips_val = self.lpips_metric.compute(
                    generated, target, normalize=normalize_lpips
                )
                if lpips_val.dim() == 0:
                    metrics['lpips'] = lpips_val.item()
                else:
                    metrics['lpips'] = lpips_val.mean().item()
            except Exception as e:
                logger.warning(f"Failed to compute LPIPS: {e}")
        
        return metrics
    
    def __call__(self, generated: torch.Tensor, target: torch.Tensor, **kwargs) -> Dict[str, float]:
        """Convenience method for computing all metrics."""
        return self.compute_all(generated, target, **kwargs)


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> Dict[str, float]:
    """
    Format metrics dictionary with optional prefix.
    
    Args:
        metrics: Dictionary of metric names to values
        prefix: Prefix to add to metric names (e.g., "validation/")
    
    Returns:
        Formatted metrics dictionary
    """
    if not prefix:
        return metrics
    
    # Add trailing slash if not present
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    
    return {f"{prefix}{key}": value for key, value in metrics.items()}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample images
    img1 = torch.randn(2, 3, 256, 256)  # Generated
    img2 = torch.randn(2, 3, 256, 256)  # Target
    
    # Initialize metrics
    metrics = ImageMetrics(compute_lpips=True, lpips_net='alex')
    
    # Compute all metrics
    results = metrics.compute_all(
        generated=img1,
        target=img2,
        normalize_lpips=False,  # Images in [-1, 1] range
        max_value=1.0,
    )
    
    print("Computed metrics:")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")

