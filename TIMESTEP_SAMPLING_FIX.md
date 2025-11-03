# Critical Fix: Timestep Sampling Stuck at 20

## 🐛 The Bug

**Symptom:** All training timesteps were stuck at 20.0 instead of being sampled uniformly from [1, 1000]

```
INFO:root:Timestep sampling - min: 20.0, max: 20.0, scheduler range: [20.0, 1000.0]
INFO:root:Timestep sampling - min: 20.0, max: 20.0, scheduler range: [20.0, 1000.0]
INFO:root:Timestep sampling - min: 20.0, max: 20.0, scheduler range: [20.0, 1000.0]
```

## ⚠️ Why This Was Critical

When timesteps are stuck at 20:

1. **Model only learns low noise**: Training at t=20 means very little noise
2. **No generalization**: Model doesn't learn to denoise at other noise levels
3. **Poor inference**: Model fails when sampling from t=1000 down to t=1
4. **Wasted training**: Essentially training a near-identity function

This is like training an image classifier on only cat images and expecting it to recognize dogs!

---

## 🔍 Root Cause (Double Whammy!)

### Issue #1: Scheduler Not Initialized for Training

The scheduler was loaded from the pipeline but never initialized for **training**:

```python
# In load_model()
self.scheduler = pipe.scheduler  # ← Loaded with INFERENCE timesteps!
```

The pipeline's scheduler is configured for **inference** (typically 20-50 steps for fast generation), not the full 1000-step training schedule.

**What was happening:**
1. Scheduler loaded with short inference timesteps: `[1000, 980, 960, ..., 40, 20]` (50 steps)
2. Training code samples indices: `u * num_train_timesteps` → `[0.5 * 1000] = 500`
3. Clamped to valid range: `min(500, len(timesteps)-1)` → `min(500, 49) = 49`
4. Always selects last timestep: `scheduler.timesteps[49] = 20`

### Issue #2: Validation Corrupts Training Timesteps! 🎯

**Even worse:** After step ~100, validation sampling calls `scheduler.set_timesteps(20)` which **overwrites** the training timesteps!

```python
# In prepare_predict_timesteps() - called during validation
timesteps, num_inference_steps = retrieve_timesteps(
    self.scheduler,  # ← Same scheduler used for training!
    num_inference_steps=20,  # ← Overwrites with 20-step inference schedule
    device=device,
)
```

**Timeline of disaster:**
1. Training starts (timesteps stuck at 20 from Issue #1)
2. Step 100: Validation runs
3. Validation calls `prepare_predict_timesteps(20)` 
4. This calls `scheduler.set_timesteps(20)` → Overwrites training timesteps
5. Training resumes with corrupted 20-step inference schedule
6. All subsequent training steps stuck at t=20!

---

## ✅ The Fix (Two-Part Solution)

### Fix #1: Initialize Training Timesteps

**Added `setup_training_scheduler()` method**

**Location:** `src/trainer/base_trainer.py`, lines 640-665

```python
def setup_training_scheduler(self):
    """Initialize scheduler timesteps for training."""
    if self.scheduler is None:
        logging.warning("Scheduler not loaded, skipping training scheduler setup")
        return
    
    # Check if scheduler has set_train_timesteps method (custom scheduler)
    if hasattr(self.scheduler, 'set_train_timesteps'):
        num_timesteps = self.scheduler.config.num_train_timesteps
        device = self.accelerator.device if self.accelerator else 'cpu'
        self.scheduler.set_train_timesteps(
            num_timesteps=num_timesteps,
            device=device,
            timestep_type='linear'
        )
        logging.info(f"✓ Initialized training timesteps: {len(self.scheduler.timesteps)} timesteps")
    else:
        # Standard scheduler - manually create training timesteps
        num_timesteps = getattr(self.scheduler.config, 'num_train_timesteps', 1000)
        device = self.accelerator.device if self.accelerator else 'cpu'
        timesteps = torch.linspace(num_timesteps, 1, num_timesteps, device=device)
        self.scheduler.timesteps = timesteps
        logging.info(f"✓ Initialized standard training timesteps: {len(timesteps)} timesteps")
```

**What it does:**
- Creates a full 1000-step linear schedule: `[1000, 999, 998, ..., 3, 2, 1]`
- Replaces the short inference schedule
- Called once during training setup

### Fix #2: Separate Inference Scheduler (The Proper Solution!)

**Created separate scheduler for validation/inference**

**Key changes:**

1. **Added `inference_scheduler` attribute** (`src/trainer/base_trainer.py`, line 88):
```python
self.inference_scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None
```

2. **Create inference scheduler copy** (`src/trainer/base_trainer.py`, lines 667-678):
```python
def _create_inference_scheduler(self):
    """Create a separate scheduler instance for validation/inference."""
    scheduler_class = self.scheduler.__class__
    self.inference_scheduler = scheduler_class.from_config(self.scheduler.config)
    logging.info(f"✓ Created separate inference scheduler: {scheduler_class.__name__}")
```

3. **Use inference scheduler in validation** (`src/trainer/base_trainer.py`, line 1033):
```python
def prepare_predict_timesteps(self, num_inference_steps: int, image_seq_len: int):
    # Use separate inference scheduler (doesn't affect training)
    scheduler_to_use = self.inference_scheduler if self.inference_scheduler is not None else self.scheduler
    
    # This modifies inference_scheduler.timesteps (NOT training scheduler)
    timesteps, num_inference_steps = retrieve_timesteps(scheduler_to_use, ...)
```

4. **Use inference scheduler in sampling** (`src/trainer/qwen_image_edit_trainer.py`, line 1143):
```python
def sampling_from_embeddings(self, embeddings: dict):
    # Use inference scheduler (not training scheduler)
    scheduler = self.inference_scheduler if self.inference_scheduler is not None else self.scheduler
    
    # All scheduler operations use the separate inference scheduler
    scheduler.set_begin_index(0)
    ...
    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
```

**Why this is better:**
- ✅ **Complete isolation**: Training and inference never interfere
- ✅ **Thread-safe**: Can run validation while training (future feature)
- ✅ **No restore logic**: Cleaner, more maintainable
- ✅ **Industry standard**: Most libraries use separate schedulers

### Fix #3: Enhanced Logging

**Location:** `src/trainer/qwen_image_edit_trainer.py`, lines 752-765

Added logging to verify fix:
- `training/timestep_min` - Minimum timestep in batch
- `training/timestep_max` - Maximum timestep in batch  
- Console log every 100 steps showing timestep range

---

## 📊 Expected Behavior After Fix

### Before Fix ❌
```
Timestep sampling - min: 20.0, max: 20.0
training/timestep_mean: 20.0
training/timestep_std: 0.0
```

### After Fix ✅
```
✓ Initialized training timesteps: 1000 timesteps from 1000.0 to 1.0
Timestep range: min=45.0, max=982.0, mean=503.2
training/timestep_mean: 503.2
training/timestep_std: 287.4
training/timestep_min: 45.0
training/timestep_max: 982.0
```

**What to look for:**
- `timestep_mean` should be around **500** (uniform distribution)
- `timestep_min` and `timestep_max` should vary widely (full range coverage)
- `timestep_std` should be around **290** (standard deviation of uniform[1, 1000])

**Startup logs to confirm fix:**
```
✓ Initialized training timesteps: 1000 timesteps from 1000.0 to 1.0
✓ Created separate inference scheduler: FlowMatchEulerDiscreteScheduler
```

---

## 🔬 Technical Details

### Timestep Sampling Process

1. **Generate uniform random values** `u ~ Uniform(0, 1)`:
   ```python
   u = compute_density_for_timestep_sampling(
       weighting_scheme="none",  # Uniform sampling
       batch_size=batch_size,
   )
   ```

2. **Convert to indices**:
   ```python
   indices = (u * self.scheduler.config.num_train_timesteps).long()
   # Example: u=0.5 → indices=500
   ```

3. **Clamp to valid range**:
   ```python
   indices = torch.clamp(indices, 0, len(self.scheduler.timesteps) - 1)
   # Before fix: clamp(500, 0, 49) = 49 (always last timestep!)
   # After fix: clamp(500, 0, 999) = 500 (correct index)
   ```

4. **Select timesteps**:
   ```python
   timesteps = self.scheduler.timesteps[indices]
   # Before fix: timesteps[49] = 20 (always 20!)
   # After fix: timesteps[500] ≈ 500 (uniform sampling)
   ```

### Why Uniform Sampling?

**Critical principle:** Timestep **sampling** should be uniform, even when using min-SNR **weighting**.

```python
# ✅ CORRECT: Uniform sampling + weighted loss
u = compute_density_for_timestep_sampling(weighting_scheme="none")  # Uniform
loss_weighted = min_snr_weights * mse_loss  # Weight the loss

# ❌ WRONG: Biased sampling
u = compute_density_for_timestep_sampling(weighting_scheme="logit_normal")  # Biased!
```

**Reason:** You need to train on all noise levels equally. Min-SNR weighting adjusts **how much you learn** from each level, but you still need to **see** all levels during training.

---

## 🧪 Verification

After this fix, check your logs/W&B for:

### 1. Startup Log
```
✓ Initialized training timesteps: 1000 timesteps from 1000.0 to 1.0
```

### 2. Training Metrics (W&B)
- `training/timestep_mean` ≈ 500 (±50)
- `training/timestep_std` ≈ 290 (±30)
- `training/timestep_min` - Should vary (not stuck at 20!)
- `training/timestep_max` - Should vary (approaching 1000)

### 3. Console Logs (every 100 steps)
```
INFO: Timestep range: min=123.0, max=879.0, mean=512.3
INFO: Scheduler timesteps available: 1000 timesteps from 1000.0 to 1.0
```

---

## 📝 Files Modified

1. **`src/trainer/base_trainer.py`**
   - Added `setup_training_scheduler()` method (lines 639-661)
   - Called in `fit()` method (line 615)

2. **`src/trainer/qwen_image_edit_trainer.py`**
   - Enhanced timestep logging (lines 752-765)
   - Added min/max timestep metrics

---

## 🎓 Key Takeaways

1. **Scheduler initialization matters**: Inference schedules ≠ Training schedules
2. **Always validate timestep distribution**: Check `timestep_mean ≈ 500`
3. **Log early and often**: Timestep metrics revealed this bug
4. **Uniform sampling is critical**: Even with weighted losses

---

## 🔗 Related Concepts

### Difference Between Sampling and Weighting

| Aspect | Timestep Sampling | Loss Weighting |
|--------|------------------|----------------|
| **What** | Which t values to train on | How much to learn from each t |
| **Should be** | Uniform (cover all noise levels) | Can be weighted (min-SNR) |
| **Code location** | `compute_density_for_timestep_sampling` | `compute_min_snr_weights` |
| **Effect on training** | Determines noise level coverage | Balances learning across levels |

### Why This Wasn't Caught Earlier

- Small batch sizes (2-4) made it less obvious
- Loss still decreased (model learned the easy t=20 task)
- No validation sampling to expose the issue
- Timestep logging wasn't initially present

---

## ✅ Verification Checklist

After restarting training:

- [ ] See "✓ Initialized training timesteps: 1000 timesteps" in startup logs
- [ ] `training/timestep_mean` ≈ 500 in W&B
- [ ] `training/timestep_min` varies (not stuck at 20)
- [ ] `training/timestep_max` approaches 1000
- [ ] Console logs show varying timestep ranges

---

## 🚀 Impact on Training

**Before Fix:**
- ❌ Only learned to denoise at t=20 (minimal noise)
- ❌ Poor generalization to other noise levels
- ❌ Failed at inference (starts from t=1000)
- ❌ Essentially wasted training time

**After Fix:**
- ✅ Learns across full noise range [1, 1000]
- ✅ Better generalization
- ✅ Correct inference behavior
- ✅ Proper diffusion training

This is as critical as the loss function itself - without proper timestep sampling, the model cannot learn the diffusion process!

