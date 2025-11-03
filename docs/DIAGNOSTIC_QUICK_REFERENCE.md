# Quick Reference: 5 Critical Training Checks

## TL;DR - Run This First

```bash
cd /workspace/hardik/test_repos/qwen-image-finetune
python script/quick_diagnostic_test.py --config configs/qwen_image_edit_plus_custom.yaml
```

This will verify all 5 checks in ~2 minutes.

---

## The 5 Checks

### ✅ Check 1: Trainable Parameters
**What:** List parameters with `requires_grad=True` and verify gradients exist after backward

**Quick check:**
```python
trainable = [(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
print("trainable_count:", len(trainable), "total_params:", sum(s for _,s in trainable))

# After loss.backward():
has_grad = [(n,p.grad is not None) for n,p in model.named_parameters() if p.requires_grad]
print("params_with_grad:", sum(int(x[1]) for x in has_grad))
```

**Expected:** Should see LoRA params (e.g., 148 tensors, ~25M parameters)

**Red flag:** `trainable_count: 0` or `params_with_grad: 0`

---

### ✅ Check 2: Gradient Norms
**What:** Log gradient norms BEFORE zeroing (after backward, before optimizer.step)

**Quick check:**
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

**Expected:** `grad_norm` between 0.5 - 5.0 (varies by model/LR)

**Red flag:** `grad_norm < 1e-8` (dying gradients) or `grad_norm > 100` (exploding)

---

### ✅ Check 3: LoRA Hooks
**What:** Verify LoRA adapters landed on the right modules (matching target_modules)

**Quick check:**
```python
lora_modules = []
for name, module in model.named_modules():
    if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
        lora_modules.append(name)
print(f"Found {len(lora_modules)} LoRA modules")
```

**Expected:** LoRA on `to_q`, `to_k`, `to_v`, `to_out.0` (or whatever you configured)

**Red flag:** `Found 0 LoRA modules` or modules don't match `target_modules` in config

---

### ✅ Check 4: Timestep Sampling
**What:** Verify timesteps are uniformly sampled (not weighted/biased)

**Quick check:**
```python
# Sample timesteps - should be UNIFORM
u = compute_density_for_timestep_sampling(
    weighting_scheme="none",  # ← MUST be "none" for uniform sampling
    batch_size=batch_size,
    logit_mean=0.0,
    logit_std=1.0,
    mode_scale=1.29,
)
indices = (u * scheduler.config.num_train_timesteps).long()
timesteps = scheduler.timesteps[indices]

# Check uniformity
print("timestep_mean:", timesteps.float().mean().item())
print("timestep_std:", timesteps.float().std().item())
```

**Expected:** Mean ~500 (for 1000 timesteps), uniform distribution

**Red flag:** Mean far from midpoint, or high deviation → biased sampling

**Fix:** Use `weighting_scheme="none"` for sampling. Min-SNR weighting should only affect **loss**, not sampling!

---

### ✅ Check 5: Target Type & Mask Normalization
**What:** Verify target matches scheduler's prediction_type and mask weights are normalized

**Quick check:**
```python
# Check target
target = noise - image_latents  # For flow matching
print("target_magnitude:", target.abs().mean().item())
print("prediction_type:", scheduler.config.prediction_type)

# Check mask normalization
if mask is not None:
    w = mask * fg_weight + (1 - mask) * bg_weight
    w = w * (w.numel() / (w.sum() + 1e-8))  # ← Normalize to mean=1.0
    loss = (w * mse_per_pixel).mean()
    print("mask_weight_mean:", w.mean().item())  # Should be ~1.0
```

**Expected:** Target magnitude 0.1-5.0, mask weight mean ~1.0

**Red flag:** Target magnitude > 10 or < 0.01, mask weight mean far from 1.0

**Fix:** Add mask normalization line above to prevent LR-amplified gradients

---

## Common Issues & Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss not decreasing | No gradients flowing | Check #1, #2 |
| Loss decreasing but results bad | LoRA on wrong modules | Check #3 |
| Unstable loss, diverges | Non-uniform timesteps | Check #4, use `weighting_scheme="none"` |
| High foreground loss | Mask not normalized | Check #5, normalize mask weights |
| Out of memory | Not using gradient checkpointing | Enable in config |

---

## Running the Diagnostics

**Quick test (5 steps):**
```bash
python script/quick_diagnostic_test.py --config configs/qwen_image_edit_plus_custom.yaml
```

**Full training with monitoring:**
```bash
python script/apply_diagnostic_checks.py --config configs/qwen_image_edit_plus_custom.yaml --stage fit
```

**View metrics in WandB:**
- `gradients/*` - gradient norms, param counts
- `diagnostics/*` - timesteps, mask stats, target magnitude  
- `training/*` - loss, LR, timesteps

---

## What Good Training Looks Like

✅ **Gradients:**
- Global norm: 0.5 - 5.0
- All trainable params have gradients
- Norm stable across steps

✅ **LoRA:**
- Matches target_modules count
- All LoRA params trainable
- Hooks on attention layers

✅ **Timesteps:**
- Mean ~500 (for 1000 max)
- Uniformity score > 0.85
- No bias toward low/high values

✅ **Loss:**
- Decreases over time
- Foreground/background losses both decrease
- No NaNs or infinite values

✅ **Mask:**
- Effective weight mean 0.9 - 1.1
- Foreground ratio matches your data
- No gradients exploding on masked regions

---

## Next Steps After Diagnostics Pass

1. **Monitor training curves** - Loss should decrease smoothly
2. **Check validation samples** - Visual quality improves
3. **Adjust hyperparameters** - LR, batch size, grad accumulation
4. **Scale up** - If diagnostics pass, can train longer/larger

---

## Getting Help

If diagnostics fail:

1. Run with `--steps 1` to get detailed output
2. Check the full log in terminal
3. Look for specific error messages
4. Review the [detailed README](DIAGNOSTIC_CHECKS_README.md)

For persistent issues, share:
- Config file
- Diagnostic summary output
- WandB run link
- First-step detailed log

