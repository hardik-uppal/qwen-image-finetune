from torch import Tensor
import torch.nn.functional as F
import torch
import torch.nn as nn


def map_mask_to_latent(image_mask: Tensor) -> Tensor:
    """
    Args:
        image_mask: [B, H, W] - Binary mask in image space
    Returns:
        latent_mask: [B, seq_len] - Weights for packed latent
    """
    B, H, W = image_mask.shape

    # Step 1: VAE-aligned downsampling
    # [B, H, W] → [B, H/8, W/8]
    latent_h, latent_w = H // 8, W // 8
    mask_latent = F.avg_pool2d(
        image_mask.float().unsqueeze(1),
        kernel_size=8, stride=8
    ).squeeze(1)  # [B, latent_h, latent_w]

    # Step 2: Packing simulation
    # [B, latent_h, latent_w] → [B, latent_h//2, latent_w//2, 4]
    # First reshape to separate 2x2 patches, then fold
    patches = mask_latent.reshape(B, latent_h//2, 2, latent_w//2, 2)
    patches = patches.permute(0, 1, 3, 2, 4).contiguous().view(B, latent_h//2, latent_w//2, 4)

    # Step 3: Patch-wise maximum (preserve text regions)
    # [B, latent_h//2, latent_w//2, 4] → [B, latent_h//2, latent_w//2]
    packed_mask = patches.max(dim=-1)[0]

    # Step 4: Flatten to sequence
    # [B, latent_h//2, latent_w//2] → [B, seq_len]
    seq_len = (latent_h // 2) * (latent_w // 2)
    return packed_mask.view(B, seq_len)


class MaskEditLoss(nn.Module):
    def __init__(self, forground_weight=2.0, background_weight=1.0):
        super().__init__()
        self.forground_weight = forground_weight
        self.background_weight = background_weight
        self.last_mask_norm_factor = None  # Track normalization factor for logging

    def forward(self, mask, model_pred, target, weighting=None):
        """
        Compute mask-weighted loss with proper ordering:
        1. Compute per-pixel MSE
        2. Apply min-SNR temporal weighting (if provided)
        3. Apply spatial mask weighting (normalized to mean ≈ 1)
        
        Args:
            mask: [B, seq_len] - Binary mask, 1=edit region, 0=background
            model_pred: [B, seq_len, channels] - Model predictions
            target: [B, seq_len, channels] - Target values
            weighting: [B, seq_len, 1] - Optional temporal weights (e.g., min-SNR)
        Returns:
            torch.Tensor - Weighted loss scalar
        """
        # Step 1: Compute element-wise MSE
        element_loss = (model_pred.float() - target.float()) ** 2

        # Step 2: Apply temporal weighting (min-SNR) if provided
        if weighting is not None:
            element_loss = weighting.float() * element_loss

        # Step 3: Create spatial mask weights (foreground vs background)
        # mask: [B, seq_len] -> weight_mask: [B, seq_len, 1]
        weight_mask = (mask * self.forground_weight + (1 - mask) * self.background_weight)
        
        # Normalize mask weights to mean ≈ 1 (prevents loss scaling issues)
        # This ensures mask weighting doesn't artificially inflate or deflate the loss magnitude
        mask_mean = weight_mask.mean(dim=1, keepdim=True).detach()  # [B, 1]
        self.last_mask_norm_factor = mask_mean.mean().item()  # Store for logging
        weight_mask_normalized = weight_mask / (mask_mean + 1e-8)
        weight_mask_normalized = weight_mask_normalized.unsqueeze(-1)  # [B, seq_len, 1]

        # Apply normalized mask weights
        weighted_loss = element_loss * weight_mask_normalized

        # Aggregate: mean over sequence, then mean over batch
        loss = torch.mean(weighted_loss.reshape(target.shape[0], -1), 1).mean()
        return loss


if __name__ == "__main__":
    original_mask = torch.randn(1, 832, 576)
    mask = torch.randn(1, 1872)
    model_pred = torch.randn(1, 1872, 64)
    target = torch.randn(1, 1872, 64)
    mask2 = map_mask_to_latent(original_mask)
    print(mask2.shape)
    criterion = MaskEditLoss()
    loss = criterion(mask2, model_pred, target)
    print(loss)

    loss = criterion(mask, model_pred, target)
    print(loss)
