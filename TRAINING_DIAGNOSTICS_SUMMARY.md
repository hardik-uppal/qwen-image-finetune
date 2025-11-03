# Training Diagnostics Summary

This document provides a quick reference for all the diagnostic tools and improvements made to detect and debug training issues.

## 📊 Problems Addressed

### 1. **CRITICAL: Timestep Sampling Stuck at 20** 🔴

All timesteps were stuck at 20.0 instead of uniform sampling [1, 1000]. This meant the model only learned to denoise at very low noise levels!

**Status:** ✅ **FIXED** - Added `setup_training_scheduler()` to properly initialize training timesteps.

**Document:** [`TIMESTEP_SAMPLING_FIX.md`](TIMESTEP_SAMPLING_FIX.md)

### 2. **Loss Spikes at Step ~100**

Both weighted and unweighted losses jump identically, indicating a **data-related issue** rather than a min-SNR problem.

**Status:** ✅ **Enhanced monitoring and validation** added.

---

## 🛠️ Implemented Solutions

### 1. Min-SNR Loss Improvements ✅

**Document:** [`MIN_SNR_IMPROVEMENTS.md`](MIN_SNR_IMPROVEMENTS.md)

**Key Features:**
- Dual loss logging (weighted + unweighted)
- Weight normalization (prevents loss collapse)
- Enhanced min-SNR statistics
- Proper mask weight normalization

**W&B Metrics:**
- `loss/weighted` - Training loss
- `loss/unweighted` - Real progress metric
- `training/min_snr_weight_mean` - Should be ≈1.0
- `loss/mask_norm_factor` - Mask normalization factor

---

### 2. Data Anomaly Detection ✅

**Document:** [`DATA_ANOMALY_DETECTION.md`](DATA_ANOMALY_DETECTION.md)

**Key Features:**
- VAE latent validation (catches NaN/Inf early)
- CSV dataset validation (skips invalid rows)
- Batch statistics logging (detects anomalies)
- Epoch boundary logging (tracks shuffle)

**W&B Metrics:**
- `data/target_magnitude` - Target latent magnitude
- `data/target_std` - Target variability  
- `data/pred_magnitude` - Model prediction magnitude
- `data/latent_magnitude` - Input latent magnitude
- `data/mask_mean` - Mask coverage (0-1)

**Validation Script:**
```bash
python script/validate_dataset_csv.py workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv
```

---

## 🔍 Quick Debug Workflow

When you see a loss spike:

### Step 1: Check W&B Metrics

```
1. Navigate to your W&B run
2. Look at step ~100 where spike occurs
3. Check these metrics:
```

**Timestep Distribution:**
- `training/timestep_mean` → Should stay ~500
- `training/timestep_std` → Should be consistent

**Batch Statistics:**
- `data/target_magnitude` → Look for sudden jumps
- `data/target_std` → Look for variance spikes
- `data/latent_magnitude` → Should be stable
- `data/mask_mean` → Check mask coverage changes

**Loss Comparison:**
- `loss/weighted` vs `loss/unweighted` → Should move together
- If both jump → Data issue
- If only weighted jumps → Min-SNR issue

### Step 2: Check Training Logs

```bash
# Epoch boundaries
grep "Starting Epoch" training.log

# NaN warnings
grep "NaN" training.log

# Skipped samples
grep "skipped" training.log

# VAE errors
grep "VAE produced" training.log
```

### Step 3: Validate CSV

```bash
python script/validate_dataset_csv.py path/to/your/train.csv
```

Look for:
- ✗ Missing files
- ⚠ NaN values
- ⚠ Corrupt images
- ⚠ Large size variations

### Step 4: Calculate Epoch Alignment

```python
# Your config
batch_size = 2
gradient_accumulation_steps = 4
num_samples = 2000

# Calculate
samples_per_step = batch_size * gradient_accumulation_steps * num_gpus
steps_per_epoch = num_samples // samples_per_step

# Check alignment
if spike_step % steps_per_epoch ≈ 0:
    print("Spike aligns with epoch boundary!")
```

---

## 📝 Files Modified

### Min-SNR Improvements
1. `src/utils/min_snr_loss.py` - Weight normalization
2. `src/data/config.py` - New config parameter
3. `src/trainer/qwen_image_edit_trainer.py` - Dual loss logging
4. `src/loss/edit_mask_loss.py` - Mask normalization
5. `configs/qwen_image_edit_plus_custom.yaml` - Config updated

### Data Anomaly Detection
1. `src/trainer/qwen_image_edit_trainer.py` - VAE validation, batch stats
2. `src/data/dataset.py` - CSV validation
3. `src/trainer/base_trainer.py` - Epoch logging

### New Scripts
1. `script/validate_dataset_csv.py` - CSV validation tool

### Documentation
1. `MIN_SNR_IMPROVEMENTS.md` - Min-SNR details
2. `DATA_ANOMALY_DETECTION.md` - Data debugging guide
3. `TRAINING_DIAGNOSTICS_SUMMARY.md` - This file

---

## 📈 Monitoring Checklist

### Before Training

- [ ] Validate CSV: `python script/validate_dataset_csv.py train.csv`
- [ ] Check for NaN in CSV
- [ ] Verify all image files exist
- [ ] Check image size consistency
- [ ] Confirm `skip_on_error: true` in config
- [ ] **CRITICAL:** Look for "✓ Initialized training timesteps: 1000 timesteps" in startup logs

### During Training (W&B)

**Critical Metrics:**
- [ ] **`training/timestep_mean`** - Should be ~500 (NOT 20!) ⚠️
- [ ] **`training/timestep_min/max`** - Should vary widely (not stuck!)
- [ ] `loss/unweighted` - Should decrease steadily
- [ ] `data/target_magnitude` - Should be stable

**Weight Statistics:**
- [ ] `training/min_snr_weight_mean` - Should be ≈1.0
- [ ] `loss/mask_norm_factor` - Should be stable

**Gradient Health:**
- [ ] `gradients/lora_norm` - Should be > 0 and stable

### After Loss Spike

- [ ] Compare spike_step to epoch boundaries
- [ ] Check `data/*` metrics for anomalies
- [ ] Review logs for warnings/errors
- [ ] Re-validate CSV if issues found

---

## 🎯 Common Patterns

### Pattern 1: Epoch Boundary Spike

**Symptoms:**
- Spike at step = N × steps_per_epoch
- Both weighted/unweighted jump
- `data/target_magnitude` changes

**Cause:** DataLoader shuffle reveals different statistics

**Fix:**
- Check if CSV is sorted (e.g., by size)
- Verify preprocessing consistency
- Consider fixing shuffle seed for debugging

### Pattern 2: Random Spikes

**Symptoms:**
- Spike at random steps
- `data/target_std` spikes
- VAE NaN warnings in logs

**Cause:** Corrupt or problematic images

**Fix:**
- Run CSV validation
- Check logs for specific files
- Use `skip_on_error: true` (already enabled)

### Pattern 3: Gradual Drift

**Symptoms:**
- Loss slowly increases
- No sudden spikes
- `data/mask_mean` drifts

**Cause:** Batch composition changes

**Fix:**
- Monitor `data/mask_mean`
- Check if samples ordered by complexity
- Verify mask normalization working

---

## 🚀 Quick Commands

```bash
# Validate dataset
python script/validate_dataset_csv.py workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv

# Start training
python src/main.py --config configs/qwen_image_edit_plus_custom.yaml

# Monitor logs
tail -f logs/training.log | grep -E "(Starting Epoch|NaN|skipped|WARNING)"

# Check W&B
wandb sync  # if offline
```

---

## 💡 Configuration Tips

### Recommended Settings

```yaml
data:
  skip_on_error: true  # Skip problematic samples
  shuffle: true        # Enable shuffling
  
loss:
  use_min_snr: true
  min_snr_gamma: 5.0
  min_snr_normalize_weights: true  # IMPORTANT: Prevents collapse
  mask_loss: true
  
logging:
  log_gradients: true
  log_parameters: true
  log_memory: true
  enhanced_metrics_interval: 10  # Log frequently for debugging
```

---

## 🔬 Advanced Debugging

### Enable Even More Logging

To get per-step detailed statistics, modify `qwen_image_edit_trainer.py`:

```python
# Change from % 10 == 0 to % 1 == 0 for every step
if self.accelerator.is_main_process and self.global_step % 1 == 0:
    # Log batch stats every step
```

**Warning:** This generates A LOT of data. Only use for debugging specific issues.

### Fix Random Seed for Reproducibility

```python
# src/main.py
seed_everything(42)  # Change 1234 to fixed seed

# In config or code
train_dataloader = DataLoader(
    dataset,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),  # Fixed shuffle order
)
```

This makes data order reproducible for debugging.

---

## 📞 Support

If issues persist:

1. **Check all metrics** in W&B around spike
2. **Run CSV validation** and fix any issues
3. **Review logs** for specific error messages
4. **Share metrics** - Include screenshots of:
   - Loss curves (`loss/weighted`, `loss/unweighted`)
   - Batch stats (`data/target_magnitude`, `data/target_std`)
   - Timestep distribution (`training/timestep_mean`)

---

## ✅ Summary

You now have:

1. ✅ **Min-SNR improvements** - Proper weighting without collapse
2. ✅ **Data validation** - Catches bad data before training
3. ✅ **Comprehensive logging** - Pinpoints exact issues
4. ✅ **Validation tools** - Checks CSV health
5. ✅ **Debug workflow** - Step-by-step problem solving

**Key Insight:** When both `loss/weighted` and `loss/unweighted` jump together, it's **always a data issue**, not a weighting issue. Use the new `data/*` metrics to find exactly which batch statistic changed!

