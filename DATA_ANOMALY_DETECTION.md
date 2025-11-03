# Data Anomaly Detection Guide

## Problem: Loss Spikes at Step ~100

When you see identical sharp jumps in both `train/loss` and `train/smooth_loss` around step ~100, this indicates a **data-related issue**, not a min-SNR weighting problem.

### Common Causes

1. **Epoch Boundary Shuffle**: DataLoader reshuffles at new epoch, suddenly exposing different data statistics
2. **Invalid Data**: NaN or empty latent values in some samples
3. **Inconsistent Image Properties**: Mixed resolutions, corrupt files, missing masks
4. **CSV Issues**: Multiple CSVs with different statistics, or mid-run CSV switching

---

## Implemented Safeguards

### 1. VAE Encoding Validation

**Location:** `src/trainer/qwen_image_edit_trainer.py`, lines 831-836

Validates latents after VAE encoding to catch NaN/Inf early:

```python
# Validate latents for NaN or Inf
if torch.isnan(image_latents).any() or torch.isinf(image_latents).any():
    logging.error(f"NaN or Inf detected in VAE latents!")
    logging.error(f"Input image stats - min: {image.min():.4f}, max: {image.max():.4f}")
    raise ValueError("VAE produced NaN or Inf latents - check input image quality")
```

**What it catches:**
- Corrupt images that produce NaN latents
- Extreme pixel values causing encoding issues
- VAE numerical instability

---

### 2. CSV Dataset Validation

**Location:** `src/data/dataset.py`, lines 293-343

Validates CSV before loading:

```python
# Check for NaN in required columns
nan_targets = df['path_target'].isna().sum()
nan_prompts = df['prompt'].isna().sum()

# Skip rows with NaN in required fields
if pd.isna(row["path_target"]) or pd.isna(row["prompt"]):
    skipped_count += 1
    continue

# Validate at least one control image exists
if len(controls) == 0:
    logging.warning(f"Row {idx} has no control images, skipping")
    skipped_count += 1
    continue
```

**What it catches:**
- Missing or NaN target images
- Missing or NaN prompts
- Empty control image fields
- Invalid mask paths

**Output:**
```
WARNING: Found 5 NaN targets and 2 NaN prompts in CSV workspace/train.csv
WARNING: These rows will be skipped during training
WARNING: Skipped 7 invalid rows out of 2000 total rows
INFO: Loaded 1993 valid samples from workspace/train.csv
```

---

### 3. Batch Statistics Logging

**Location:** `src/trainer/qwen_image_edit_trainer.py`, lines 739-750

Logs batch data statistics every 10 steps:

```python
batch_stats = {
    'data/target_magnitude': target.abs().mean().item(),
    'data/target_std': target.std().item(),
    'data/pred_magnitude': model_pred.abs().mean().item(),
    'data/latent_magnitude': image_latents.abs().mean().item(),
}
if edit_mask is not None:
    batch_stats['data/mask_mean'] = edit_mask.float().mean().item()
```

**Monitored Metrics:**
- `data/target_magnitude` - Target latent magnitude (should be stable ~0.5-2.0)
- `data/target_std` - Target variability (check for sudden spikes)
- `data/pred_magnitude` - Model prediction magnitude
- `data/latent_magnitude` - Input latent magnitude
- `data/mask_mean` - Average mask coverage (0.0-1.0)

**How to use:**
Watch these in W&B around step ~100 where loss spikes. If they change drastically, you've found the culprit!

---

### 4. Epoch Boundary Logging

**Location:** `src/trainer/base_trainer.py`, lines 483-489

Logs at the start of each epoch:

```python
logging.info(f"=" * 60)
logging.info(f"Starting Epoch {epoch} | Global Step: {self.global_step}")
logging.info(f"Dataset size: {len(train_dataloader.dataset)}")
logging.info(f"=" * 60)
```

**What to check:**
- If loss spike occurs exactly at epoch boundaries
- If dataset size changes unexpectedly
- If shuffle behavior is consistent

---

## Debugging Workflow

### Step 1: Check W&B Metrics

When you see a loss spike at step ~100:

1. **Check timestep distribution:**
   - `training/timestep_mean` - Should stay ~500
   - `training/timestep_std` - Should be consistent
   - If these change, sampling is biased

2. **Check batch statistics:**
   - `data/target_magnitude` - Look for sudden jumps
   - `data/target_std` - Look for variance spikes
   - `data/latent_magnitude` - Should be stable
   - `data/mask_mean` - Check mask coverage changes

3. **Check loss comparison:**
   - `loss/weighted` vs `loss/unweighted` - Should move together
   - If unweighted jumps too, it's definitely data-related

### Step 2: Check Logs

Search your logs around step ~100 for:

```bash
# Check for epoch boundaries
grep "Starting Epoch" training.log

# Check for NaN warnings
grep "NaN" training.log

# Check for skipped samples
grep "skipped" training.log

# Check for VAE errors
grep "VAE produced" training.log
```

### Step 3: Validate Your CSV

Run the validation script (see below) to check for:
- Missing files
- NaN values
- Inconsistent image properties
- Corrupt images

### Step 4: Check Dataloader Shuffle

**Key question:** Is your loss spike exactly at an epoch boundary?

Calculate:
```python
samples_per_epoch = len(dataset) // batch_size
steps_per_epoch = samples_per_epoch // gradient_accumulation_steps

# If spike_step ≈ N * steps_per_epoch, it's an epoch boundary issue
```

If spike happens at epoch boundary:
1. Different batches have very different statistics
2. Some "hard" samples cluster together after shuffle
3. Data augmentation introduces variance

**Solution:**
- Ensure consistent data preprocessing
- Check if certain image types cluster in CSV rows
- Consider fixing random seed for shuffle to debug

---

## CSV Validation Script

Create `script/validate_dataset_csv.py`:

```python
#!/usr/bin/env python3
"""
Validate dataset CSV for common issues that cause training anomalies.
"""

import pandas as pd
import logging
from pathlib import Path
import argparse
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_csv(csv_path: str):
    """Validate CSV dataset for common issues."""
    print(f"\n{'='*60}")
    print(f"Validating CSV: {csv_path}")
    print(f"{'='*60}\n")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded CSV with {len(df)} rows")
    
    # Check required columns
    required_cols = ['path_target', 'prompt']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        return False
    print(f"✓ Required columns present: {required_cols}")
    
    # Check for NaN
    nan_count = df[required_cols].isna().sum()
    if nan_count.any():
        print(f"⚠ NaN values found:")
        for col, count in nan_count.items():
            if count > 0:
                print(f"  - {col}: {count} NaN values")
    else:
        print(f"✓ No NaN values in required columns")
    
    # Check control columns
    control_cols = [col for col in df.columns if 'path_control' in col]
    if len(control_cols) == 0:
        print(f"✗ No control columns found (expected path_control or path_control_1, etc.)")
        return False
    print(f"✓ Found {len(control_cols)} control columns: {control_cols}")
    
    # Validate file paths
    print(f"\nValidating file paths...")
    missing_targets = 0
    missing_controls = 0
    corrupt_images = 0
    
    for idx, row in df.iterrows():
        # Check target
        if pd.notna(row['path_target']):
            if not Path(row['path_target']).exists():
                missing_targets += 1
                if missing_targets <= 5:  # Only print first 5
                    print(f"  Row {idx}: Missing target - {row['path_target']}")
            else:
                # Try to open image
                try:
                    img = Image.open(row['path_target'])
                    img.verify()
                except Exception as e:
                    corrupt_images += 1
                    if corrupt_images <= 5:
                        print(f"  Row {idx}: Corrupt image - {row['path_target']}: {e}")
        
        # Check controls
        for col in control_cols:
            if col in row and pd.notna(row[col]):
                if not Path(row[col]).exists():
                    missing_controls += 1
                    if missing_controls <= 5:
                        print(f"  Row {idx}: Missing control - {row[col]}")
    
    print(f"\nFile Validation Results:")
    print(f"  Missing targets: {missing_targets}")
    print(f"  Missing controls: {missing_controls}")
    print(f"  Corrupt images: {corrupt_images}")
    
    if missing_targets + missing_controls + corrupt_images == 0:
        print(f"✓ All files exist and are readable")
    else:
        print(f"⚠ Found {missing_targets + missing_controls + corrupt_images} file issues")
    
    # Check image statistics
    print(f"\nChecking image statistics (sampling first 50 images)...")
    widths, heights, aspects = [], [], []
    
    for idx, row in df.head(50).iterrows():
        if pd.notna(row['path_target']) and Path(row['path_target']).exists():
            try:
                img = Image.open(row['path_target'])
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspects.append(w/h)
            except:
                pass
    
    if len(widths) > 0:
        print(f"  Width range: {min(widths)} - {max(widths)} (mean: {np.mean(widths):.0f})")
        print(f"  Height range: {min(heights)} - {max(heights)} (mean: {np.mean(heights):.0f})")
        print(f"  Aspect ratio range: {min(aspects):.2f} - {max(aspects):.2f}")
        
        if max(widths) / min(widths) > 2 or max(heights) / min(heights) > 2:
            print(f"⚠ Large variation in image sizes detected - may cause batch inconsistency")
    
    print(f"\n{'='*60}")
    print(f"Validation Complete")
    print(f"{'='*60}\n")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset CSV")
    parser.add_argument("csv_path", help="Path to CSV file")
    args = parser.parse_args()
    
    validate_csv(args.csv_path)
```

**Usage:**
```bash
python script/validate_dataset_csv.py workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv
```

---

## Common Patterns

### Pattern 1: Loss Spike at Epoch Boundary

**Symptom:**
- Loss jumps at step = N * steps_per_epoch
- Both weighted and unweighted losses jump
- `data/target_magnitude` changes significantly

**Cause:** DataLoader shuffle exposes batches with different statistics

**Solution:**
- Check if data is sorted in CSV (e.g., by image size)
- Verify preprocessing consistency across all samples
- Consider using stratified sampling

### Pattern 2: Random Loss Spikes

**Symptom:**
- Loss spikes at random steps (not epoch boundaries)
- `data/target_std` shows sudden spikes
- May see VAE NaN warnings in logs

**Cause:** Corrupt or problematic individual images

**Solution:**
- Run CSV validation script
- Check logs for specific file paths causing errors
- Use `skip_on_error: true` in dataset config (already enabled)

### Pattern 3: Gradual Loss Drift

**Symptom:**
- Loss slowly increases over time
- No sudden spikes
- `data/mask_mean` drifts

**Cause:** Batch composition changes (e.g., mask coverage varies)

**Solution:**
- Monitor `data/mask_mean` - should stay consistent
- Check if samples are ordered by mask complexity
- Ensure mask normalization is working

---

## Quick Checklist

When you see a loss spike:

- [ ] Check `training/timestep_mean` - should be ~500
- [ ] Check `data/target_magnitude` - look for jumps
- [ ] Check `data/mask_mean` - look for coverage changes
- [ ] Check logs for "NaN" or "skipped" warnings
- [ ] Run CSV validation script
- [ ] Calculate if spike aligns with epoch boundary
- [ ] Compare `loss/weighted` vs `loss/unweighted` - should move together
- [ ] Check if VAE produced any errors around that step

---

## Summary

The min-SNR improvements ensure weighted loss behaves properly, but **data quality is king**. The new safeguards will:

1. **Catch bad data early** (VAE validation, CSV checks)
2. **Expose anomalies** (batch statistics logging)
3. **Track data flow** (epoch boundary logging)
4. **Prevent silent failures** (skip invalid samples instead of crashing)

Monitor the new `data/*` metrics in W&B to quickly identify when and why loss spikes occur!

