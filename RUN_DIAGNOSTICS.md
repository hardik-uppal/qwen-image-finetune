# 🔍 Quick Start: Run Training Diagnostics

## One Command to Check Everything

```bash
cd /workspace/hardik/test_repos/qwen-image-finetune
python script/quick_diagnostic_test.py --config configs/qwen_image_edit_plus_custom.yaml
```

This will run **5 critical checks** in ~2 minutes:

1. ✅ **Trainable Parameters** - Verify LoRA setup
2. ✅ **Gradient Norms** - Check gradient flow  
3. ✅ **LoRA Hooks** - Confirm adapters on correct modules
4. ✅ **Timestep Sampling** - Verify uniform sampling (not biased)
5. ✅ **Target & Mask** - Check prediction targets and mask normalization

## Expected Output

```
==================================================
CHECK 1: TRAINABLE PARAMETERS
==================================================
trainable_count: 148
total_params: 25,165,824
LoRA params: 148/148

==================================================
CHECK 2: GRADIENT NORMS (after backward)
==================================================
params_with_grad: 148/148
global_grad_norm: 2.456789

==================================================
CHECK 3: LORA HOOKS
==================================================
Found 74 modules with LoRA
Target modules pattern: ['to_k', 'to_q', 'to_v', 'to_out.0']
✓ All LoRA parameters are trainable

==================================================
CHECK 4: TIMESTEP SAMPLING
==================================================
Mean: 500.0 (expected ~500.0)
Uniformity score: 0.95 (1.0 = perfect)
✓ Timestep sampling is uniform

==================================================
CHECK 5: TARGET TYPE & MASK NORMALIZATION
==================================================
Target magnitude (abs mean): 0.8234
Scheduler prediction_type: epsilon
Mask effective weight mean: 1.0234
✓ Mask weights look reasonable

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

## What Each Check Does

| Check | Purpose | Red Flag |
|-------|---------|----------|
| 1. Trainable Params | Verify LoRA parameters exist and receive gradients | `trainable_count: 0` or `params_with_grad: 0` |
| 2. Gradient Norms | Ensure gradients flow properly (not dying/exploding) | `grad_norm < 1e-8` or `> 100` |
| 3. LoRA Hooks | Confirm adapters on correct attention layers | `Found 0 LoRA modules` |
| 4. Timestep Sampling | Verify uniform sampling (not weighted) | Uniformity < 0.7 or mean deviation > 10% |
| 5. Target & Mask | Check prediction targets and mask normalization | Target magnitude > 10 or mask weight far from 1.0 |

## If Checks Fail

**No trainable params?**
→ LoRA wasn't added. Check config and `add_adapter()` call.

**No gradients?**
→ Loss isn't backpropagating. Check for detached tensors.

**LoRA not found?**
→ Adapters not applied. Verify PEFT installation.

**Timesteps not uniform?**
→ Use `weighting_scheme="none"` for sampling. Loss weighting is separate!

**Mask weight far from 1.0?**
→ Normalize: `w = w * (w.numel() / (w.sum() + 1e-8))`

## Detailed Documentation

- **Quick Reference:** `docs/DIAGNOSTIC_QUICK_REFERENCE.md` (1 page)
- **Full Guide:** `docs/DIAGNOSTIC_CHECKS_README.md` (detailed explanations)
- **Implementation:** `docs/DIAGNOSTIC_CHECKS_SUMMARY.md` (how to integrate)

## Advanced Usage

**Run with fewer steps (faster):**
```bash
python script/quick_diagnostic_test.py --config configs/qwen_image_edit_plus_custom.yaml --steps 2
```

**Run full training with continuous monitoring:**
```bash
python script/apply_diagnostic_checks.py --config configs/qwen_image_edit_plus_custom.yaml --stage fit
```

**Integrate into your existing training:**
```python
from script.diagnostic_checks import DiagnosticChecker

# In __init__:
self.diagnostic_checker = DiagnosticChecker(self)

# After backward():
self.diagnostic_checker.check_gradient_norms()
```

---

**Run diagnostics before every training session to catch issues early! 🚀**

