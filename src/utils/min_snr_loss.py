"""
Min-SNR Loss Weighting for Diffusion Training

Implements the min-SNR weighting strategy from:
"Efficient Diffusion Training via Min-SNR Weighting Strategy"
https://arxiv.org/abs/2303.09556

This weights the loss by min(SNR(t), γ) to prevent overfitting at extreme noise levels.
"""

import torch
from typing import Optional


def compute_snr(timesteps: torch.Tensor, scheduler) -> torch.Tensor:
    """
    Compute Signal-to-Noise Ratio (SNR) for given timesteps.
    
    For flow matching schedulers (like Qwen), SNR is computed as:
    SNR(t) = ((1 - sigma) / sigma)^2
    
    Args:
        timesteps: Timesteps tensor
        scheduler: The noise scheduler
        
    Returns:
        SNR values for each timestep
    """
    # Get sigmas from scheduler
    alphas = 1 - timesteps / 1000  # For flow matching
    sigmas = timesteps / 1000
    
    # SNR = (alpha / sigma)^2 = ((1 - sigma) / sigma)^2
    snr = (alphas / sigmas) ** 2
    
    return snr


def compute_min_snr_weights(
    timesteps: torch.Tensor,
    scheduler,
    gamma: float = 5.0,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """
    Compute min-SNR loss weights.
    
    Args:
        timesteps: Timesteps for current batch
        scheduler: Noise scheduler
        gamma: Min-SNR gamma parameter (typical: 5.0)
        prediction_type: "epsilon", "v_prediction", or "sample"
        
    Returns:
        Loss weights tensor of shape [batch_size]
    """
    # Compute SNR for each timestep
    snr = compute_snr(timesteps, scheduler)
    
    # Apply min-SNR clipping
    min_snr = torch.clamp(snr, max=gamma)
    
    # Compute weights based on prediction type
    if prediction_type == "epsilon":
        # For epsilon prediction: weight = min(SNR, γ)
        weights = min_snr
    elif prediction_type == "v_prediction":
        # For v prediction: weight = min(SNR, γ) / (SNR + 1)
        weights = min_snr / (snr + 1)
    elif prediction_type == "sample":
        # For sample prediction: weight = min(SNR, γ) / SNR
        weights = min_snr / snr
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")
    
    return weights


def apply_min_snr_loss_weighting(
    loss: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    gamma: float = 5.0,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """
    Apply min-SNR loss weighting to a per-sample loss tensor.
    
    Args:
        loss: Per-sample loss tensor of shape [batch_size] or [batch_size, ...]
        timesteps: Timesteps for current batch
        scheduler: Noise scheduler
        gamma: Min-SNR gamma parameter
        prediction_type: Prediction type of the model
        
    Returns:
        Weighted loss (scalar)
    """
    # Compute weights
    weights = compute_min_snr_weights(timesteps, scheduler, gamma, prediction_type)
    
    # Reshape weights to match loss dimensions
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    
    # Apply weights and take mean
    weighted_loss = (weights * loss).mean()
    
    return weighted_loss


def compute_snr_from_sigmas(sigmas: torch.Tensor) -> torch.Tensor:
    """
    Compute SNR directly from sigma values.
    
    For flow matching: sigma = t/1000, alpha = 1 - t/1000
    SNR = (alpha / sigma)^2 = ((1 - sigma) / sigma)^2
    
    Args:
        sigmas: Sigma values from scheduler
        
    Returns:
        SNR values
    """
    alphas = 1.0 - sigmas
    snr = (alphas / sigmas) ** 2
    return snr


def compute_min_snr_weights_from_sigmas(
    sigmas: torch.Tensor,
    gamma: float = 5.0,
    prediction_type: str = "v_prediction",
    normalize_weights: bool = True,
) -> torch.Tensor:
    """
    Compute min-SNR weights directly from sigma values.
    
    This is more efficient when you already have sigmas computed.
    
    Args:
        sigmas: Sigma values from scheduler [batch_size, 1, 1, ...]
        gamma: Min-SNR gamma parameter
        prediction_type: "epsilon", "v_prediction", or "sample"
        normalize_weights: If True, normalize weights to have mean ≈ 1 (prevents loss collapse)
        
    Returns:
        Loss weights tensor
    """
    # Compute SNR from sigmas
    snr = compute_snr_from_sigmas(sigmas)
    
    # Apply min-SNR clipping
    min_snr = torch.clamp(snr, max=gamma)
    
    # Compute weights based on prediction type
    if prediction_type == "epsilon":
        weights = min_snr
    elif prediction_type == "v_prediction":
        weights = min_snr / (snr + 1)
    elif prediction_type == "sample":
        weights = min_snr / snr
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")
    
    # Flatten weights to [batch_size]
    weights = weights.flatten()
    
    # OPTIONAL: re-normalize so E[w] ≈ 1 across the batch
    # This prevents the weighted loss from collapsing to very small values
    if normalize_weights:
        weights = weights / (weights.mean().detach() + 1e-8)
    
    return weights


if __name__ == "__main__":
    # Test the functions
    import matplotlib.pyplot as plt
    
    # Create test timesteps
    timesteps = torch.linspace(1, 1000, 100)
    
    # Mock scheduler
    class MockScheduler:
        pass
    
    scheduler = MockScheduler()
    
    # Compute SNR
    snr = compute_snr(timesteps, scheduler)
    
    # Compute min-SNR weights for different gammas
    gammas = [1.0, 5.0, 10.0, 20.0]
    
    plt.figure(figsize=(12, 4))
    
    # Plot SNR
    plt.subplot(1, 2, 1)
    plt.plot(timesteps.numpy(), snr.numpy(), label="SNR")
    for gamma in gammas:
        min_snr = torch.clamp(snr, max=gamma)
        plt.plot(timesteps.numpy(), min_snr.numpy(), label=f"min(SNR, {gamma})")
    plt.xlabel("Timestep")
    plt.ylabel("SNR")
    plt.yscale("log")
    plt.legend()
    plt.title("SNR and Min-SNR Clipping")
    plt.grid(True, alpha=0.3)
    
    # Plot weights for v-prediction
    plt.subplot(1, 2, 2)
    for gamma in gammas:
        weights = compute_min_snr_weights(timesteps, scheduler, gamma, "v_prediction")
        plt.plot(timesteps.numpy(), weights.numpy(), label=f"γ={gamma}")
    plt.xlabel("Timestep")
    plt.ylabel("Loss Weight")
    plt.legend()
    plt.title("Min-SNR Weights (v-prediction)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("min_snr_weights.png", dpi=150)
    print("Saved visualization to min_snr_weights.png")
    
    # Test with actual sigmas
    print("\n=== Testing with sigma values ===")
    sigmas_test = torch.tensor([0.001, 0.01, 0.1, 0.5, 0.9, 0.99]).unsqueeze(-1)
    weights_eps = compute_min_snr_weights_from_sigmas(sigmas_test, gamma=5.0, prediction_type="epsilon")
    weights_v = compute_min_snr_weights_from_sigmas(sigmas_test, gamma=5.0, prediction_type="v_prediction")
    
    print(f"Sigmas: {sigmas_test.squeeze().numpy()}")
    print(f"Weights (ε): {weights_eps.numpy()}")
    print(f"Weights (v): {weights_v.numpy()}")

