# Min-SNR Loss Weighting Improvements

This document describes the improvements made to the min-SNR loss weighting implementation to prevent loss collapse and improve training stability.

## Summary of Changes

### 1. **Dual Loss Logging (Weighted + Unweighted)**

**Problem:** When using min-SNR weighting, the weighted loss can become very small and may not reflect actual model progress.

**Solution:** Now we compute and log both:
- `loss/weighted` - The loss used for training (with min-SNR weighting applied)
- `loss/unweighted` - Raw MSE loss for monitoring real progress

**Code Location:** `src/trainer/qwen_image_edit_trainer.py`, lines 680-739

```python
# Compute unweighted loss (for monitoring real progress)
mse_per_sample = F.mse_loss(model_pred, target, reduction="none").mean(dim=[1, 2])
loss_unweighted = mse_per_sample.mean()

# Log both for comparison
loss_comparison_metrics = {
    'loss/unweighted': loss_unweighted.item(),
    'loss/weighted': loss_result.item(),
}
```

**Benefit:** You can now monitor model quality using unweighted loss while still training with weighted loss.

---

### 2. **Weight Normalization (Prevents Loss Collapse)**

**Problem:** Min-SNR weights can vary significantly across timesteps, potentially causing the weighted loss to collapse to very small values, making training unstable.

**Solution:** Added optional weight normalization that scales weights to have mean ≈ 1:

```python
# Re-normalize so E[w] ≈ 1 across the batch
if normalize_weights:
    weights = weights / (weights.mean().detach() + 1e-8)
```

**Code Location:** `src/utils/min_snr_loss.py`, lines 169-172

**Configuration:**
```yaml
loss:
  use_min_snr: true
  min_snr_gamma: 5.0
  min_snr_normalize_weights: true  # NEW: Enable normalization
```

**Benefit:** Prevents weighted loss from becoming artificially small while maintaining the relative weighting benefits of min-SNR.

---

### 3. **Enhanced Min-SNR Weight Logging**

**Problem:** Limited visibility into how min-SNR weights are behaving during training.

**Solution:** Added comprehensive weight statistics logged every 10 steps:

- `training/min_snr_weight_mean` - Average weight across batch
- `training/min_snr_weight_std` - Weight variation
- `training/min_snr_weight_min` - Minimum weight (low timesteps)
- `training/min_snr_weight_max` - Maximum weight (high timesteps)

**Code Location:** `src/trainer/qwen_image_edit_trainer.py`, lines 713-720

**Benefit:** Monitor if weights are well-distributed or if certain timesteps are dominating.

---

### 4. **Proper Mask Weight Normalization**

**Problem:** Spatial mask weighting (foreground vs background) could artificially scale the loss magnitude, especially when combined with min-SNR weighting.

**Solution:** Normalize spatial mask weights to mean ≈ 1 before applying:

```python
# Create mask weights
weight_mask = (mask * fg_weight + (1 - mask) * bg_weight)

# Normalize to mean ≈ 1
mask_mean = weight_mask.mean(dim=1, keepdim=True).detach()
weight_mask_normalized = weight_mask / (mask_mean + 1e-8)
```

**Code Location:** `src/loss/edit_mask_loss.py`, lines 72-76

**Order of Operations:**
1. Compute per-pixel MSE
2. Apply min-SNR temporal weighting
3. Apply normalized spatial mask weighting
4. Aggregate to scalar loss

**Benefit:** Mask weighting now focuses on relative priorities (foreground vs background) without changing the overall loss scale.

---

### 5. **Timestep Distribution Monitoring**

**Already Implemented:** The code logs timestep statistics every 10 steps:

- `training/timestep_mean` - Should be ≈500 for uniform sampling
- `training/timestep_std` - Spread of sampled timesteps

**Code Location:** `src/trainer/qwen_image_edit_trainer.py`, lines 742-747

**What to Watch:**
- If `timestep_mean` deviates significantly from 500, sampling may be biased
- Check if loss collapse correlates with certain timestep ranges

---

## Configuration Example

Updated config in `configs/qwen_image_edit_plus_custom.yaml`:

```yaml
loss:
  mask_loss: true
  forground_weight: 2.0
  background_weight: 1.0
  
  # Min-SNR loss weighting (recommended for better convergence)
  use_min_snr: true               # Enable min-SNR weighting
  min_snr_gamma: 5.0              # Gamma parameter (typical: 5.0)
  min_snr_normalize_weights: true # Normalize weights to mean ≈ 1
  prediction_type: null           # Auto-detect from scheduler
```

---

## Monitoring Checklist

When training with these improvements, monitor these metrics in W&B:

### Critical Metrics
- [ ] **`loss/unweighted`** - Should decrease steadily (real progress)
- [ ] **`loss/weighted`** - Training loss (may be smaller but should still decrease)
- [ ] **`training/timestep_mean`** - Should stay near 500 (uniform sampling)

### Weight Statistics
- [ ] **`training/min_snr_weight_mean`** - Should be ≈1.0 (if normalization enabled)
- [ ] **`training/min_snr_weight_std`** - Check variance in weights
- [ ] **`loss/mask_norm_factor`** - Spatial mask normalization factor

### Gradient Health
- [ ] **`gradients/lora_norm`** - Should be > 0 and stable (not exploding/vanishing)
- [ ] **`gradients/global_norm`** - Overall gradient magnitude

---

## Technical Details

### Min-SNR Weight Formula (v-prediction)

For Qwen's flow matching with v-prediction:

```python
# Signal-to-Noise Ratio
alpha = 1 - sigma
SNR = (alpha / sigma)^2

# Min-SNR weight
w = min(SNR, γ) / (SNR + 1)

# Optional normalization
w_normalized = w / mean(w)
```

Where:
- `sigma = t/1000` (flow matching parameterization)
- `γ = 5.0` (typical gamma value from paper)

### Loss Computation Flow

```
Input: model_pred, target, timesteps, mask

1. Compute per-sample MSE: [B, seq_len, C] → [B]
   loss_unweighted = mean(MSE)

2. Compute min-SNR weights from timesteps: [B]
   w = min(SNR, γ) / (SNR + 1)
   w = w / mean(w)  # normalize

3. Apply temporal weights:
   loss_temporal = mean(w * MSE)

4. Apply spatial mask weights (normalized):
   mask_weights = (mask * 2.0 + (1-mask) * 1.0)
   mask_weights = mask_weights / mean(mask_weights)
   loss_final = mean(mask_weights * loss_temporal)

5. Train with loss_final, monitor loss_unweighted
```

---

## Expected Behavior

### Before Improvements
- Weighted loss might collapse to very small values (e.g., 1e-5)
- Hard to tell if model is actually improving
- Mask weighting could dominate min-SNR effects

### After Improvements
- Weighted loss stays in reasonable range (similar magnitude to unweighted)
- Clear separation between training loss (weighted) and monitoring loss (unweighted)
- Both types of weighting (temporal + spatial) work together harmoniously
- Easy to spot if quality improves while weighted loss appears flat

---

## References

1. **Min-SNR Paper:** [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)
2. **Weight Normalization:** Common practice in loss reweighting to prevent scale collapse
3. **Flow Matching:** Qwen uses flow matching (v-prediction) instead of traditional DDPM

---

## Files Modified

1. `src/utils/min_snr_loss.py` - Added weight normalization
2. `src/data/config.py` - Added `min_snr_normalize_weights` config option
3. `src/trainer/qwen_image_edit_trainer.py` - Dual loss logging + enhanced metrics
4. `src/loss/edit_mask_loss.py` - Normalized mask weights
5. `configs/qwen_image_edit_plus_custom.yaml` - Updated config with new option

---

## Questions or Issues?

If weighted loss becomes extremely small (< 1e-6), try:
1. Check `min_snr_normalize_weights: true` is enabled
2. Monitor `training/min_snr_weight_mean` - should be ≈1.0
3. Compare `loss/weighted` vs `loss/unweighted` - should be same order of magnitude
4. Check `training/timestep_mean` - should be ≈500 for uniform sampling

If quality seems good but loss looks bad:
- Focus on `loss/unweighted` for quality assessment
- Weighted loss is for training optimization, not quality measurement
- Use fixed validation samples to visually assess quality

