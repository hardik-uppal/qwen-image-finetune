# Quick Debug Reference Card

## 🚨 Loss Spike at Step ~100?

### 1. Check if Data-Related (30 seconds)

Open W&B and look at step ~100:

| Metric | Normal | Problem |
|--------|--------|---------|
| `loss/unweighted` | Steady decrease | Sudden jump |
| `loss/weighted` | Steady decrease | Sudden jump |
| `data/target_magnitude` | Stable (~0.5-2.0) | Jumps > 50% |
| `training/timestep_mean` | ~500 | Deviates > 100 |

**If both losses jump together → DATA ISSUE ✓**
**If only weighted jumps → Min-SNR issue**

---

### 2. Find the Culprit (2 minutes)

```bash
# Run CSV validation
python script/validate_dataset_csv.py workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv

# Check logs
grep "Starting Epoch" logs/training.log  # Find epoch boundaries
grep "NaN\|skipped" logs/training.log   # Find bad data
```

---

### 3. Calculate Epoch Alignment (1 minute)

```python
# Your config values
batch_size = 2
gradient_accumulation = 4
num_gpus = 8
num_samples = 2000

# Effective batch size
effective_batch = batch_size * gradient_accumulation * num_gpus  # = 64

# Steps per epoch
steps_per_epoch = num_samples / effective_batch  # = 31.25

# Check
spike_step = 100
epoch_number = spike_step / steps_per_epoch  # = 3.2

# If ≈ integer, it's an epoch boundary issue!
```

---

## 📊 W&B Metrics Quick Guide

### Critical Metrics (Check These First)

| Metric | What It Means | Healthy Range | Problem If... |
|--------|---------------|---------------|---------------|
| `loss/unweighted` | Real model progress | Decreasing | Jumps or increases |
| `data/target_magnitude` | Latent size | 0.5 - 2.0 | Sudden changes > 50% |
| `training/timestep_mean` | Sampling bias | ~500 | < 400 or > 600 |
| `gradients/lora_norm` | Gradient flow | 0.5 - 5.0 | < 0.01 or > 100 |

### Secondary Metrics (For Deep Dives)

| Metric | What It Means | Look For |
|--------|---------------|----------|
| `loss/weighted` | Training loss | Compare to unweighted |
| `data/target_std` | Batch variance | Sudden spikes |
| `data/mask_mean` | Mask coverage | Should be stable (0-1) |
| `training/min_snr_weight_mean` | Weight normalization | Should be ≈1.0 |
| `loss/mask_norm_factor` | Mask scaling | Should be stable |

---

## 🔧 Common Fixes

### Fix 1: Bad Data in CSV

**Problem:** Validation shows missing/corrupt files

**Solution:**
```bash
# The dataset loader will skip them automatically
# Just make sure this is in config:
skip_on_error: true  # Already enabled
```

### Fix 2: Epoch Boundary Issues

**Problem:** Loss spikes at epoch boundaries

**Solution:**
```yaml
# Fix random seed for reproducibility
# Add to src/main.py:
seed_everything(42)  # Change from 1234

# Or disable shuffle temporarily to test:
data:
  shuffle: false  # Only for debugging!
```

### Fix 3: NaN in Latents

**Problem:** "VAE produced NaN or Inf latents" in logs

**Solution:**
1. Find the problematic image in logs
2. Check image file quality
3. Remove from CSV or fix image
4. Rerun validation

---

## 🎯 Decision Tree

```
Loss spike detected
    │
    ├─→ Both weighted AND unweighted jump?
    │   YES → DATA ISSUE
    │   │     ├─→ Check data/target_magnitude → Changed? → Batch statistics issue
    │   │     ├─→ Check timestep_mean → Changed? → Sampling bias
    │   │     └─→ Check logs for "NaN" → Found? → Corrupt data
    │   
    └─→ Only weighted jumps?
        NO → Min-SNR ISSUE
              ├─→ Check min_snr_normalize_weights: true
              └─→ Check training/min_snr_weight_mean ≈ 1.0
```

---

## 🚀 One-Liners

```bash
# Validate before training
python script/validate_dataset_csv.py train.csv && echo "✓ CSV OK"

# Watch training logs live
tail -f logs/training.log | grep -E "Epoch|NaN|WARNING"

# Check if loss aligns with epochs
python -c "print(f'Epoch {100 / (2000 / (2*4*8)):.1f}')"  # Replace 100 with spike step

# Find last checkpoint
ls -lh /skynas/01/hardik/qwen-lora/qwen-edit-plus-custom/checkpoint-*

# Resume from checkpoint
# Set in config: resume: /path/to/checkpoint
```

---

## 📞 Emergency Checklist

When training is misbehaving:

- [ ] Run `python script/validate_dataset_csv.py train.csv`
- [ ] Check W&B: `loss/unweighted`, `data/target_magnitude`, `training/timestep_mean`
- [ ] Search logs: `grep "NaN\|skipped\|WARNING" logs/training.log`
- [ ] Verify config: `min_snr_normalize_weights: true`
- [ ] Calculate epoch alignment (see formula above)
- [ ] Check if spike = corrupt data or epoch boundary

---

## 💾 Save This

**Bookmark these files:**
- [`TRAINING_DIAGNOSTICS_SUMMARY.md`](TRAINING_DIAGNOSTICS_SUMMARY.md) - Full guide
- [`MIN_SNR_IMPROVEMENTS.md`](MIN_SNR_IMPROVEMENTS.md) - Loss weighting details
- [`DATA_ANOMALY_DETECTION.md`](DATA_ANOMALY_DETECTION.md) - Data debugging

**Key script:**
```bash
python script/validate_dataset_csv.py <csv_path>
```

**Key config:**
```yaml
loss:
  use_min_snr: true
  min_snr_normalize_weights: true  # CRITICAL!
  
data:
  skip_on_error: true  # Skip bad samples
  
logging:
  enhanced_metrics_interval: 10  # Frequent logging
```

---

## 🎓 Remember

1. **Both losses jump together** = Data issue (99% of time)
2. **Check `data/*` metrics first** - They show what changed
3. **Run CSV validation** - Catches most issues before training
4. **Epoch boundaries matter** - Calculate alignment
5. **Logs tell the story** - grep is your friend

---

**Last Updated:** After implementing min-SNR improvements + data anomaly detection
**Related Issues:** Loss spikes, NaN latents, epoch boundary effects

