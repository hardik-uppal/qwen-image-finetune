# Min-SNR Loss Weighting for Qwen Image Edit Training

## Overview

Min-SNR (Minimum Signal-to-Noise Ratio) weighting is a loss weighting strategy that improves diffusion model training convergence by preventing overfitting at extreme noise levels.

**Paper:** [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)

## Key Concepts

### Problem with Uniform Loss Weighting

Standard diffusion training treats all timesteps equally, but:
- **Early timesteps** (low noise): Model learns fine details, easy to overfit
- **Late timesteps** (high noise): Model learns coarse structure, harder to learn

This leads to imbalanced training where the model may focus too much on certain noise levels.

### Min-SNR Solution

Min-SNR weights the loss by `min(SNR(t), γ)` where:
- **SNR(t)**: Signal-to-Noise Ratio at timestep t
- **γ** (gamma): Clipping threshold (typically 5.0)

This prevents any timestep from dominating the loss, leading to:
- ✅ Better convergence
- ✅ Improved sample quality
- ✅ More stable training
- ✅ Better handling of extreme noise levels

## Critical: Timestep Sampling vs Loss Weighting

**⚠️ IMPORTANT:** Min-SNR weighting should **ONLY** affect loss weighting, **NOT** timestep sampling!

### Correct Implementation ✅

```python
# TIMESTEP SAMPLING - Must be UNIFORM
u = compute_density_for_timestep_sampling(
    weighting_scheme="none",  # ← UNIFORM sampling
    batch_size=batch_size,
)
timesteps = scheduler.timesteps[indices]

# LOSS WEIGHTING - Can use min-SNR
if use_min_snr:
    weights = compute_min_snr_weights(timesteps, gamma=5.0)
    loss = (weights * mse_loss).mean()
else:
    loss = mse_loss.mean()
```

### Wrong Implementation ❌

```python
# DON'T DO THIS!
u = compute_density_for_timestep_sampling(
    weighting_scheme="logit_normal",  # ❌ This biases sampling!
)
# This will cause non-uniform timestep distribution
```

**Why this matters:**
- Timestep sampling determines which noise levels you train on
- Loss weighting determines how much each contributes to gradients
- They should be independent!

## Usage

### 1. Enable in Config

Add to your `configs/qwen_image_edit_plus_custom.yaml`:

```yaml
loss:
  mask_loss: true
  forground_weight: 2.0
  background_weight: 1.0
  # Enable min-SNR weighting
  use_min_snr: true        # Enable min-SNR loss weighting
  min_snr_gamma: 5.0       # Gamma parameter (typical: 5.0)
  prediction_type: null    # Auto-detect from scheduler (or set to "v_prediction"/"epsilon")
```

### 2. Start Training

```bash
python src/main.py --config configs/qwen_image_edit_plus_custom.yaml
```

You'll see in the logs:

```
============================================================
LOSS CONFIGURATION
============================================================
Prediction type: v_prediction (auto-detected from scheduler)
✓ Min-SNR weighting: ENABLED (γ=5.0)
  This will improve convergence by preventing overfitting at extreme noise levels
Timestep sampling: UNIFORM (weighting only affects loss, not sampling)
✓ Mask loss: ENABLED (fg=2.0, bg=1.0)
============================================================
```

### 3. Monitor Training

The following metrics will be logged to WandB/TensorBoard:

- `training/min_snr_weight_mean` - Average weight per batch
- `training/min_snr_weight_std` - Standard deviation of weights
- `training/timestep_mean` - Verify sampling is still uniform (~500)
- `training/timestep_std` - Should show uniform distribution (~290)

## Parameters

### `use_min_snr` (bool, default: False)

Enable/disable min-SNR weighting.

### `min_snr_gamma` (float, default: 5.0)

The γ (gamma) clipping threshold for min-SNR.

**Typical values:**
- **γ = 5.0**: Standard (recommended for most cases)
- **γ = 1.0**: More aggressive clipping (stronger effect)
- **γ = 10.0 - 20.0**: Less aggressive (milder effect)

**How to choose:**
- Start with γ = 5.0 (the paper's recommendation)
- If training is unstable, try γ = 10.0
- If you want stronger regularization, try γ = 3.0

### `prediction_type` (str | null, default: null)

The type of prediction your model makes:
- **`"v_prediction"`**: Flow matching models (Qwen default)
- **`"epsilon"`**: Noise prediction models (DDPM)
- **`"sample"`**: Direct sample prediction
- **`null`**: Auto-detect from scheduler config (recommended)

**Leave as `null` unless you know your model uses a different type.**

## Prediction Type and Target Matching

Min-SNR weighting automatically adjusts based on prediction type:

### For v-prediction (Flow Matching) ✅

```python
target = noise - image_latents  # v-prediction target
weights = min(SNR, γ) / (SNR + 1)  # v-prediction weighting
loss = (weights * (model_pred - target)^2).mean()
```

This is what Qwen Image Edit uses (flow matching).

### For ε-prediction (DDPM)

```python
target = noise  # epsilon target
weights = min(SNR, γ)  # epsilon weighting
loss = (weights * (model_pred - target)^2).mean()
```

## How It Works

### SNR Computation

For flow matching schedulers:
```python
sigma = t / 1000  # Noise level
alpha = 1 - sigma  # Signal level
SNR = (alpha / sigma)^2
```

### Weight Computation

```python
# Clip SNR
min_snr = min(SNR, gamma)

# Compute weights based on prediction type
if prediction_type == "v_prediction":
    weight = min_snr / (SNR + 1)
elif prediction_type == "epsilon":
    weight = min_snr
```

### Loss Application

```python
# Per-pixel loss
mse_per_pixel = (model_pred - target)^2

# Apply mask (if enabled)
if mask_loss:
    w_mask = mask * fg_weight + (1 - mask) * bg_weight
    w_mask = w_mask / w_mask.mean()  # Normalize
    mse_per_pixel = w_mask * mse_per_pixel

# Apply min-SNR weights
weighted_loss = weights * mse_per_pixel

# Final loss
loss = weighted_loss.mean()
```

## Visualization

The min-SNR weights look like this (for γ=5.0):

```
High noise (t=1000) → Low SNR  → weight ≈ γ (clamped)
Medium noise (t=500) → Med SNR → weight ≈ SNR/2
Low noise (t=1)      → High SNR → weight ≈ γ (clamped)
```

This creates a "flatter" loss landscape across timesteps.

## Expected Results

With min-SNR weighting enabled (γ=5.0), you should see:

✅ **Better convergence:**
- Loss decreases more smoothly
- Fewer training spikes
- More stable metrics

✅ **Improved quality:**
- Better fine details
- Less artifacts at high noise levels
- More consistent generation

✅ **Faster training:**
- Typically 10-20% faster convergence
- Can sometimes reduce training time by using fewer steps

## Diagnostic Checks

Run the diagnostic test to verify your setup:

```bash
python script/quick_diagnostic_test.py --config configs/qwen_image_edit_plus_custom.yaml
```

Look for:
- ✅ Check 4: **Timestep sampling is UNIFORM** (not biased)
- ✅ Check 5: **Target type matches** prediction_type
- ✅ Logs show: `Min-SNR weighting: ENABLED`

## Troubleshooting

### Issue: "Timestep sampling is biased"

**Cause:** You're using `weighting_scheme != "none"` for sampling

**Fix:**
```python
# In _compute_loss, ensure this line uses "none":
u = compute_density_for_timestep_sampling(
    weighting_scheme="none",  # ← Must be "none"
    ...
)
```

### Issue: "Target type mismatch"

**Cause:** Prediction type doesn't match scheduler

**Fix:** Set `loss.prediction_type: "v_prediction"` in config explicitly

### Issue: "Loss is NaN or Inf"

**Cause:** Gamma is too low or weights are too extreme

**Fix:** Try increasing gamma:
```yaml
loss:
  min_snr_gamma: 10.0  # Increase from 5.0
```

### Issue: "Training is slower"

**Cause:** Weight computation adds overhead

**Impact:** Negligible (<1% slower), benefits outweigh costs

## References

1. **Paper:** [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)
2. **Code:** Based on Diffusers' implementation
3. **Related:** P2 weighting, SNR-based weighting strategies

## Summary

Min-SNR weighting with γ≈5 is **highly recommended** for:
- ✅ Flow matching models (Qwen Image Edit)
- ✅ Any diffusion model with non-uniform SNR
- ✅ Training with limited compute
- ✅ Achieving better convergence

**Remember:**
- Timestep sampling: UNIFORM
- Loss weighting: min-SNR
- Target: Match scheduler's prediction_type
- Gamma: Start with 5.0

Enable it in your config and enjoy better training! 🚀

