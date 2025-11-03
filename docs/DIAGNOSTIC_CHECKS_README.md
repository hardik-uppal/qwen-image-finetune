# Diagnostic Checks for Qwen Image Edit Training

This document describes the 5 critical diagnostic checks for verifying your Qwen Image Edit training pipeline and how to run them.

## Overview

These checks help diagnose common training issues:

1. **Trainable Parameters** - Verify LoRA parameters are set up correctly
2. **Gradient Norms** - Ensure gradients flow properly through the network
3. **LoRA Hooks** - Confirm LoRA adapters are applied to the right modules
4. **Timestep Sampling** - Check that timesteps are sampled uniformly (not biased)
5. **Target & Mask** - Verify prediction targets and mask normalization are correct

## Quick Start

### Option 1: Quick Diagnostic Test (Recommended)

Run a quick 5-step test to verify all checks:

```bash
python script/quick_diagnostic_test.py --config configs/qwen_image_edit_plus_custom.yaml --steps 5
```

This will:
- Load your model and config
- Run 5 training steps
- Perform all diagnostic checks
- Print a summary report

**Expected output:**
```
✓ Check 1 (Trainable Params):    ✓ PASS
✓ Check 2 (Gradient Norms):       ✓ PASS
✓ Check 3 (LoRA Hooks):           ✓ PASS
✓ Check 4 (Timestep Sampling):    ✓ PASS
✓ Check 5 (Target & Mask):        ✓ PASS
```

### Option 2: Continuous Monitoring During Training

To monitor diagnostics throughout your full training run:

```bash
python script/apply_diagnostic_checks.py --config configs/qwen_image_edit_plus_custom.yaml --stage fit
```

This will run your normal training with diagnostic checks logged every 10 steps to wandb/tensorboard.

## Detailed Check Descriptions

### Check 1: Trainable Parameters

**What it checks:**
- Lists all parameters with `requires_grad=True`
- Counts total trainable parameters
- Verifies LoRA parameters exist

**What to look for:**
```
trainable_count: 148
total_params: 25,165,824
LoRA params: 148/148
```

**Red flags:**
- `trainable_count: 0` - No parameters are trainable!
- `LoRA params: 0/X` - LoRA not applied correctly

### Check 2: Gradient Norms

**What it checks:**
- Verifies gradients exist after `backward()`
- Computes global gradient norm
- Identifies parameters without gradients

**What to look for:**
```
params_with_grad: 148/148
global_grad_norm: 2.456789
```

**Red flags:**
- `params_with_grad: 0` - Gradients not computed!
- `global_grad_norm: 0.0` or very small - Gradients are dying
- Large number of params without grad - Incomplete backward pass

**How to fix:**
```python
# Make sure you're calling backward BEFORE checking gradients
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(
    (p for p in model.parameters() if p.requires_grad and p.grad is not None), 
    1.0
).item()
wandb.log({"gradients/global_norm": grad_norm}, step=global_step)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

### Check 3: LoRA Hooks

**What it checks:**
- Finds all modules with LoRA adapters (`lora_A`, `lora_B`)
- Verifies they match your `target_modules` config
- Confirms LoRA parameters are trainable

**What to look for:**
```
Found 74 modules with LoRA
Target modules pattern: ['to_k', 'to_q', 'to_v', 'to_out.0']
Matched modules by pattern:
  to_k: 18 modules
  to_q: 18 modules
  to_v: 18 modules
  to_out.0: 18 modules
```

**Red flags:**
- `Found 0 modules with LoRA` - LoRA not applied!
- Unmatched modules - LoRA applied to wrong layers
- `LoRA trainable: 0` - LoRA exists but frozen

### Check 4: Timestep Sampling

**What it checks:**
- Collects timesteps from multiple training steps
- Verifies distribution is uniform (not biased)
- Computes uniformity score

**What to look for:**
```
Mean: 500.0 (expected ~500.0)
Mean deviation: 2.5%
Uniformity score: 0.95 (1.0 = perfect)
```

**Red flags:**
- Mean deviation > 10% - Biased sampling
- Uniformity score < 0.7 - Non-uniform distribution
- This suggests you're using weighted sampling when you shouldn't

**How to fix:**
Make sure you're using uniform timestep sampling:
```python
# CORRECT - Uniform sampling
u = compute_density_for_timestep_sampling(
    weighting_scheme="none",  # ← Must be "none" for uniform
    batch_size=batch_size,
    logit_mean=0.0,
    logit_std=1.0,
    mode_scale=1.29,
)
```

**Don't** use cosine/logit-normal schedules for timestep *sampling* - those are for loss weighting only.

### Check 5: Target Type & Mask Normalization

**What it checks:**
- Verifies target tensor is computed correctly
- Checks if it matches scheduler's `prediction_type`
- Validates mask normalization

**What to look for:**
```
Target magnitude (abs mean): 0.8234
Scheduler prediction_type: epsilon
Effective weight mean: 1.0234
```

**Red flags:**
- Target magnitude > 10 or < 0.01 - Wrong target computation
- Effective weight mean far from 1.0 - Poor mask normalization
- Mask values outside [0, 1] - Mask preprocessing error

**How to fix mask normalization:**
```python
# Apply foreground/background weights
fg_mask = mask > 0.5
w = torch.where(fg_mask, fg_weight, bg_weight)

# CRITICAL: Normalize to mean=1.0 per sample to avoid LR-amplified gradients
w = w * (w.numel() / (w.sum() + 1e-8))

# Compute loss
mse_per_pixel = (model_pred - target) ** 2
loss = (w * mse_per_pixel).mean()
```

## Interpreting Results

### All Checks Pass ✅

Your training pipeline is correctly configured! Common metrics you should see:

- **Gradient norm**: 0.5 - 5.0 (varies by model/LR)
- **LoRA trainable**: Should match your config (e.g., 148 for r=32, 4 modules)
- **Timestep uniformity**: > 0.85
- **Mask weight mean**: 0.9 - 1.1

### Some Checks Fail ❌

**If Check 1 fails:**
- Your LoRA adapter wasn't added correctly
- Check `model.lora.target_modules` in config
- Verify you called `add_adapter()` before training

**If Check 2 fails:**
- Loss isn't backpropagating
- Check if loss requires gradient
- Verify model is in train mode
- Check for detached tensors in loss computation

**If Check 3 fails:**
- LoRA adapters on wrong layers (won't train the right weights)
- Check naming - transformers use different names (e.g., `q_proj` vs `to_q`)

**If Check 4 fails:**
- You're using biased timestep sampling
- Change `weighting_scheme="none"` in `compute_density_for_timestep_sampling`
- Loss weighting is separate from sampling!

**If Check 5 fails:**
- Target computation doesn't match scheduler
- For flow matching: `target = noise - image_latents`
- For DDPM: `target = noise` (if predicting noise)
- Check `scheduler.config.prediction_type`

## Integration into Your Training

To add continuous monitoring to your existing training script:

1. **Import the diagnostic checker:**
```python
from script.diagnostic_checks import DiagnosticChecker
```

2. **Initialize in your trainer:**
```python
def __init__(self, config):
    super().__init__(config)
    self.diagnostic_checker = DiagnosticChecker(self)
```

3. **Add checks in training loop:**
```python
def train_epoch(self, epoch, train_dataloader):
    for batch in train_dataloader:
        with self.accelerator.accumulate(self.dit):
            loss = self.training_step(batch)
            self.accelerator.backward(loss)
            
            # === Add diagnostic checks here ===
            if self.accelerator.sync_gradients:
                self.diagnostic_checker.check_gradient_norms()
            # =================================
            
            self.clip_gradients()
            self.optimizer.step()
            self.optimizer.zero_grad()
```

4. **Check _compute_loss method:**
```python
def _compute_loss(self, embeddings):
    # ... your loss computation ...
    
    # === Add diagnostic checks ===
    if self.global_step % 10 == 0:
        self.diagnostic_checker.check_timestep_sampling(timesteps)
        self.diagnostic_checker.check_target_and_mask(model_pred, target, edit_mask)
    # ===========================
    
    return loss
```

## Monitoring in WandB/TensorBoard

The diagnostic checks automatically log metrics to your tracker:

**Gradient metrics:**
- `gradients/global_norm`
- `gradients/lora_norm`
- `gradients/params_with_grad`

**Timestep metrics:**
- `diagnostics/timestep_mean`
- `diagnostics/timestep_uniformity`

**Target/Mask metrics:**
- `diagnostics/target_magnitude`
- `diagnostics/mask_fg_ratio`
- `diagnostics/mask_effective_weight`

**Parameter metrics:**
- `diagnostics/trainable_param_count`
- `diagnostics/lora_modules`

## Troubleshooting

### "No trainable parameters"
→ LoRA wasn't added. Check your `add_adapter()` call and config.

### "No gradients after backward"
→ Loss isn't connected to model. Check for detached tensors or missing backward path.

### "Timesteps not uniform"
→ Change `weighting_scheme="none"` in sampling. Weighting should only affect loss, not sampling.

### "Mask weights not normalized"
→ Add normalization: `w = w * (w.numel() / (w.sum() + 1e-8))`

## References

These checks are based on best practices for diffusion model training:

1. [Diffusers Training Guide](https://huggingface.co/docs/diffusers/training/overview)
2. [LoRA Implementation Guide](https://github.com/microsoft/LoRA)
3. Flow Matching papers on uniform timestep sampling
4. Min-SNR loss weighting (separate from sampling)

## Support

If diagnostic checks reveal issues you can't resolve, please:

1. Run with `--steps 1` to get detailed first-step output
2. Check the full log output
3. Share config and diagnostic summary
4. Include wandb run link if available

