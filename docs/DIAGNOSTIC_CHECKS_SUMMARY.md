# Diagnostic Checks Implementation Summary

## What Was Created

I've implemented a comprehensive diagnostic system for your Qwen Image Edit training pipeline that performs the 5 critical checks you requested:

### Files Created

1. **`script/diagnostic_checks.py`** - Core diagnostic checker class
   - Implements all 5 checks as reusable methods
   - Automatic logging to WandB/TensorBoard
   - ~500 lines of diagnostic code

2. **`script/quick_diagnostic_test.py`** - Standalone test script
   - Runs diagnostics on a few training steps
   - Quick validation before full training
   - ~400 lines

3. **`script/apply_diagnostic_checks.py`** - Full training integration
   - Patches your trainer to add diagnostics
   - Continuous monitoring during training
   - ~200 lines

4. **`docs/DIAGNOSTIC_CHECKS_README.md`** - Detailed documentation
   - Full explanation of each check
   - How to interpret results
   - Troubleshooting guide

5. **`docs/DIAGNOSTIC_QUICK_REFERENCE.md`** - Quick reference card
   - One-page summary
   - Code snippets for each check
   - Common issues & fixes

## How to Use

### Step 1: Quick Validation (2 minutes)

Run the quick diagnostic test to verify your setup:

```bash
cd /workspace/hardik/test_repos/qwen-image-finetune
python script/quick_diagnostic_test.py --config configs/qwen_image_edit_plus_custom.yaml
```

**What it does:**
- Loads your model and config
- Runs 5 training steps
- Performs all 5 diagnostic checks
- Prints a summary report

**Expected output:**
```
==================================================
DIAGNOSTIC SUMMARY
==================================================
Check 1 (Trainable Params):    ✓ PASS
Check 2 (Gradient Norms):       ✓ PASS
Check 3 (LoRA Hooks):           ✓ PASS
Check 4 (Timestep Sampling):    ✓ PASS
Check 5 (Target & Mask):        ✓ PASS
==================================================
✅ All diagnostic checks passed!
```

### Step 2: Review Results

Check the terminal output for detailed information about each check:

**Check 1 - Trainable Parameters:**
```
trainable_count: 148
total_params: 25,165,824
LoRA params: 148/148
```

**Check 2 - Gradient Norms:**
```
params_with_grad: 148/148
global_grad_norm: 2.456789
```

**Check 3 - LoRA Hooks:**
```
Found 74 modules with LoRA
Target modules pattern: ['to_k', 'to_q', 'to_v', 'to_out.0']
Matched modules by pattern:
  to_k: 18 modules
  to_q: 18 modules
  to_v: 18 modules
  to_out.0: 18 modules
```

**Check 4 - Timestep Sampling:**
```
Mean: 500.0 (expected ~500.0)
Mean deviation: 2.5%
Uniformity score: 0.95 (1.0 = perfect)
```

**Check 5 - Target & Mask:**
```
Target magnitude (abs mean): 0.8234
Scheduler prediction_type: epsilon
Effective weight mean: 1.0234
```

### Step 3: Full Training with Monitoring (Optional)

If you want continuous monitoring during full training:

```bash
python script/apply_diagnostic_checks.py --config configs/qwen_image_edit_plus_custom.yaml --stage fit
```

This will:
- Run your normal training
- Log diagnostic metrics every 10 steps to WandB/TensorBoard
- Show gradients, timesteps, mask stats in real-time

## The 5 Checks Explained

### Check 1: Trainable Parameters & Gradients
**Purpose:** Verify LoRA parameters are set up and receive gradients

**Code equivalent:**
```python
trainable = [(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
print("trainable_count:", len(trainable), "total_params:", sum(s for _,s in trainable))

# After loss.backward():
has_grad = [(n,p.grad is not None) for n,p in model.named_parameters() if p.requires_grad]
print("params_with_grad:", sum(int(x[1]) for x in has_grad))
```

**What to expect:**
- ~148 trainable parameter tensors for LoRA
- ~25M total trainable parameters
- All trainable params should have gradients after backward

### Check 2: Gradient Norms (Before Zeroing)
**Purpose:** Ensure gradients flow properly and are not dying/exploding

**Code equivalent:**
```python
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(
    (p for p in model.parameters() if p.requires_grad and p.grad is not None), 
    1.0
).item()
wandb.log({"gradients/global_norm": grad_norm}, step=global_step)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

**What to expect:**
- Global grad norm: 0.5 - 5.0 (varies by model/LR)
- LoRA grad norm should be similar
- No NaN or Inf values

### Check 3: LoRA Hooks Verification
**Purpose:** Confirm LoRA adapters are applied to the correct modules

**Code equivalent:**
```python
for name, module in model.named_modules():
    if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
        print(f"LoRA found in: {name}")
```

**What to expect:**
- LoRA on attention layers: `to_q`, `to_k`, `to_v`, `to_out.0`
- Number of LoRA modules matches your config
- All LoRA parameters are trainable

### Check 4: Timestep Sampling Uniformity
**Purpose:** Verify timesteps are sampled uniformly (not biased by loss weighting)

**Code equivalent:**
```python
u = compute_density_for_timestep_sampling(
    weighting_scheme="none",  # ← Must be "none" for uniform
    batch_size=batch_size,
    logit_mean=0.0,
    logit_std=1.0,
    mode_scale=1.29,
)
indices = (u * scheduler.config.num_train_timesteps).long()
timesteps = scheduler.timesteps[indices]
```

**What to expect:**
- Mean timestep ~500 (for max 1000)
- Standard deviation ~290
- Uniformity score > 0.85

**IMPORTANT:** Loss weighting (min-SNR, p2) should NOT change timestep sampling. Sampling must be uniform!

### Check 5: Target Type & Mask Normalization
**Purpose:** Verify prediction targets match scheduler and masks are normalized

**Code equivalent:**
```python
# Check target matches scheduler
target = noise - image_latents  # For flow matching
print(f"prediction_type: {scheduler.config.prediction_type}")

# Check mask normalization
w = mask * fg_weight + (1 - mask) * bg_weight
w = w * (w.numel() / (w.sum() + 1e-8))  # ← Normalize mean to 1.0
loss = (w * mse_per_pixel).mean()
```

**What to expect:**
- Target magnitude: 0.1 - 5.0
- Prediction type: "epsilon" or "v_prediction"
- Mask effective weight mean: 0.9 - 1.1

**CRITICAL:** Mask normalization prevents LR-amplified gradients!

## Integration into Your Existing Training

If you want to add these checks to your existing training code (instead of using the wrapper scripts):

### Option A: Minimal Integration (Just Check 2)

Add gradient norm logging after backward:

```python
# In your train_epoch method, after backward but before zero_grad:
def train_epoch(self, epoch, train_dataloader):
    for batch in train_dataloader:
        with self.accelerator.accumulate(self.dit):
            loss = self.training_step(batch)
            self.accelerator.backward(loss)
            
            # === ADD THIS ===
            if self.accelerator.sync_gradients and self.global_step % 10 == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.dit.parameters() if p.requires_grad and p.grad is not None],
                    self.config.train.max_grad_norm,
                ).item()
                self.accelerator.log({"gradients/global_norm": grad_norm}, step=self.global_step)
            # ================
            
            self.optimizer.step()
            self.optimizer.zero_grad()
```

### Option B: Full Integration (All Checks)

Use the DiagnosticChecker class:

```python
from script.diagnostic_checks import DiagnosticChecker

# In your __init__:
def __init__(self, config):
    super().__init__(config)
    self.diagnostic_checker = DiagnosticChecker(self)

# In your _compute_loss (before return):
def _compute_loss(self, embeddings):
    # ... your existing loss computation ...
    
    # Add checks
    if self.global_step % 10 == 0:
        self.diagnostic_checker.check_timestep_sampling(timesteps)
        self.diagnostic_checker.check_target_and_mask(model_pred, target, edit_mask)
    
    return loss

# In your train_epoch (after backward):
def train_epoch(self, epoch, train_dataloader):
    for batch in train_dataloader:
        with self.accelerator.accumulate(self.dit):
            loss = self.training_step(batch)
            self.accelerator.backward(loss)
            
            # Add gradient check
            if self.accelerator.sync_gradients:
                self.diagnostic_checker.check_gradient_norms()
            
            self.clip_gradients()
            self.optimizer.step()
            self.optimizer.zero_grad()
```

## Metrics Logged to WandB/TensorBoard

The diagnostic checks automatically log these metrics:

**Gradient Metrics:**
- `gradients/global_norm` - Overall gradient magnitude
- `gradients/lora_norm` - LoRA-specific gradient magnitude
- `gradients/params_with_grad` - Count of params receiving gradients
- `gradients/max_norm` - Largest individual gradient
- `gradients/mean_norm` - Average gradient magnitude

**Timestep Metrics:**
- `diagnostics/timestep_mean` - Average sampled timestep
- `diagnostics/timestep_uniformity` - Uniformity score (1.0 = perfect)
- `diagnostics/timestep_mean_deviation` - Deviation from expected mean

**Target/Mask Metrics:**
- `diagnostics/target_magnitude` - Magnitude of prediction targets
- `diagnostics/mask_fg_ratio` - Foreground pixel ratio
- `diagnostics/mask_effective_weight` - Mean mask weight after normalization

**Parameter Metrics:**
- `diagnostics/trainable_param_count` - Number of trainable tensors
- `diagnostics/lora_modules` - Number of modules with LoRA

## Interpreting Results

### ✅ All Checks Pass

Your training pipeline is correctly configured! You should see:
- Trainable params: 148 tensors, ~25M elements
- Gradient norm: 0.5 - 5.0
- LoRA on correct modules
- Timestep uniformity > 0.85
- Mask weight mean 0.9 - 1.1

### ⚠️ Warnings

**Timestep mean deviation > 10%:**
- Your sampling is biased
- Change `weighting_scheme="none"` in `compute_density_for_timestep_sampling`
- Loss weighting is separate from sampling!

**Mask weight mean far from 1.0:**
- Add normalization: `w = w * (w.numel() / (w.sum() + 1e-8))`
- Prevents LR-amplified gradients on masked regions

### ❌ Critical Failures

**No trainable parameters:**
- LoRA wasn't added correctly
- Check `model.lora.target_modules` in config
- Verify `add_adapter()` was called

**No gradients after backward:**
- Loss isn't connected to model
- Check for detached tensors
- Verify model is in train mode

**No LoRA modules found:**
- LoRA adapters not applied
- Check PEFT installation
- Verify adapter name matches config

## Troubleshooting

### Issue: "No module named 'script.diagnostic_checks'"

**Fix:**
```bash
cd /workspace/hardik/test_repos/qwen-image-finetune
export PYTHONPATH=/workspace/hardik/test_repos/qwen-image-finetune:$PYTHONPATH
python script/quick_diagnostic_test.py --config configs/qwen_image_edit_plus_custom.yaml
```

### Issue: "Config file not found"

**Fix:** Use absolute path:
```bash
python script/quick_diagnostic_test.py --config /workspace/hardik/test_repos/qwen-image-finetune/configs/qwen_image_edit_plus_custom.yaml
```

### Issue: Out of memory during diagnostic test

**Fix:** Reduce test steps:
```bash
python script/quick_diagnostic_test.py --config configs/qwen_image_edit_plus_custom.yaml --steps 2
```

### Issue: Diagnostic test hangs

**Possible causes:**
1. Cache not built yet - Run cache stage first if `use_cache: true`
2. Dataset loading issue - Check data paths in config
3. Model download in progress - Wait for HuggingFace download to complete

## Next Steps

1. **Run the quick diagnostic test** to verify your setup
2. **Review the output** and address any warnings/failures
3. **Start full training** once all checks pass
4. **Monitor WandB/TensorBoard** for diagnostic metrics during training
5. **Adjust hyperparameters** based on gradient norms and loss curves

## Questions?

- Detailed explanations: See `docs/DIAGNOSTIC_CHECKS_README.md`
- Quick reference: See `docs/DIAGNOSTIC_QUICK_REFERENCE.md`
- Code examples: See `script/diagnostic_checks.py`

The diagnostic checks are designed to catch common training issues early. Run them before every major training run!

