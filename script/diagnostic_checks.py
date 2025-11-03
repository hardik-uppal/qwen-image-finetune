#!/usr/bin/env python3
"""
Diagnostic checks for Qwen Image Edit training pipeline.

This script performs 5 critical checks to verify training correctness:
1. List trainable parameters and verify gradients after backward
2. Log gradient norms before zeroing
3. Verify LoRA hooks landed correctly
4. Check timestep sampling uniformity
5. Verify target type and mask normalization

Usage:
    # Import and call from your trainer
    from script.diagnostic_checks import DiagnosticChecker
    
    # In your trainer __init__:
    self.diagnostic_checker = DiagnosticChecker(self)
    
    # In your training loop (after loss.backward(), before optimizer.step()):
    self.diagnostic_checker.run_all_checks(loss, batch, timesteps)
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class DiagnosticChecker:
    """Comprehensive diagnostic checker for diffusion training."""
    
    def __init__(self, trainer):
        """
        Initialize diagnostic checker.
        
        Args:
            trainer: The trainer instance (BaseTrainer or subclass)
        """
        self.trainer = trainer
        self.timestep_history = []
        self.checked_lora_hooks = False
        self.check_counter = 0
        
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC CHECKER INITIALIZED")
        logger.info("=" * 80)
    
    def run_all_checks(
        self, 
        loss: torch.Tensor,
        batch: Dict[str, Any],
        timesteps: Optional[torch.Tensor] = None,
        model_pred: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        edit_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Run all diagnostic checks.
        
        Should be called in training loop after backward() but before optimizer.step()
        
        Args:
            loss: The computed loss tensor
            batch: Current training batch
            timesteps: Timesteps used for this batch (if available)
            model_pred: Model prediction (if available)
            target: Target tensor (if available)
            edit_mask: Edit mask (if available)
            
        Returns:
            Dictionary with all diagnostic results
        """
        results = {}
        
        # Only run detailed checks periodically to avoid overhead
        self.check_counter += 1
        run_detailed = (self.check_counter % 100 == 1)  # Run on first step and every 100 steps
        
        # Check 1: Trainable parameters (first step only)
        if self.check_counter == 1:
            results['trainable_params'] = self.check_trainable_parameters()
        
        # Check 2: Gradient norms (every step, before zeroing)
        if self.trainer.accelerator.sync_gradients:
            results['grad_norms'] = self.check_gradient_norms()
        
        # Check 3: LoRA hooks (first step only)
        if not self.checked_lora_hooks:
            results['lora_hooks'] = self.check_lora_hooks()
            self.checked_lora_hooks = True
        
        # Check 4: Timestep sampling (periodically)
        if timesteps is not None and run_detailed:
            results['timestep_stats'] = self.check_timestep_sampling(timesteps)
        
        # Check 5: Target type and mask normalization (periodically)
        if run_detailed and model_pred is not None and target is not None:
            results['target_mask'] = self.check_target_and_mask(
                model_pred, target, edit_mask
            )
        
        return results
    
    def check_trainable_parameters(self) -> Dict[str, Any]:
        """
        Check 1: List trainable parameters and their shapes.
        
        Returns:
            Dictionary with trainable parameter statistics
        """
        logger.info("=" * 80)
        logger.info("CHECK 1: TRAINABLE PARAMETERS")
        logger.info("=" * 80)
        
        trainable = [
            (n, p.numel(), p.shape) 
            for n, p in self.trainer.dit.named_parameters() 
            if p.requires_grad
        ]
        
        total_params = sum(s for _, s, _ in trainable)
        
        logger.info(f"Trainable parameter count: {len(trainable)}")
        logger.info(f"Total trainable elements: {total_params:,}")
        
        # Group by type (lora, non-lora)
        lora_params = [(n, s) for n, s, _ in trainable if 'lora' in n.lower()]
        non_lora_params = [(n, s) for n, s, _ in trainable if 'lora' not in n.lower()]
        
        lora_total = sum(s for _, s in lora_params)
        non_lora_total = sum(s for _, s in non_lora_params)
        
        logger.info(f"LoRA parameters: {len(lora_params):,} ({lora_total:,} elements)")
        logger.info(f"Non-LoRA parameters: {len(non_lora_params):,} ({non_lora_total:,} elements)")
        
        # Show first few trainable parameters
        logger.info("\nFirst 10 trainable parameters:")
        for i, (name, numel, shape) in enumerate(trainable[:10]):
            logger.info(f"  {i+1}. {name}: shape={shape}, numel={numel:,}")
        
        if len(trainable) > 10:
            logger.info(f"  ... and {len(trainable) - 10} more")
        
        result = {
            'trainable_count': len(trainable),
            'total_params': total_params,
            'lora_count': len(lora_params),
            'lora_params': lora_total,
            'non_lora_count': len(non_lora_params),
            'non_lora_params': non_lora_total,
        }
        
        # Log to wandb/tensorboard
        if self.trainer.accelerator.is_main_process:
            self.trainer.accelerator.log({
                'diagnostics/trainable_param_count': len(trainable),
                'diagnostics/trainable_total_elements': total_params,
                'diagnostics/lora_param_count': len(lora_params),
                'diagnostics/lora_total_elements': lora_total,
            }, step=self.trainer.global_step)
        
        return result
    
    def check_gradient_norms(self) -> Dict[str, float]:
        """
        Check 2: Compute and log gradient norms BEFORE zeroing.
        
        Should be called after backward() and before optimizer.zero_grad()
        
        Returns:
            Dictionary with gradient norm statistics
        """
        # Collect gradients
        params_with_grad = []
        lora_params_with_grad = []
        grad_norms = []
        lora_grad_norms = []
        
        for name, param in self.trainer.dit.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                params_with_grad.append((name, grad_norm))
                grad_norms.append(grad_norm ** 2)
                
                if 'lora' in name.lower():
                    lora_params_with_grad.append((name, grad_norm))
                    lora_grad_norms.append(grad_norm ** 2)
        
        # Compute global norms
        global_grad_norm = sum(grad_norms) ** 0.5 if grad_norms else 0.0
        lora_grad_norm = sum(lora_grad_norms) ** 0.5 if lora_grad_norms else 0.0
        
        # Count parameters without gradients
        params_without_grad = [
            name for name, param in self.trainer.dit.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        
        # Log summary
        if self.trainer.global_step % 10 == 0:
            logger.info(f"Params with grad: {len(params_with_grad)}/{len(list(self.trainer.dit.parameters()))}")
            logger.info(f"Global grad norm: {global_grad_norm:.6f}")
            logger.info(f"LoRA grad norm: {lora_grad_norm:.6f}")
            
            if params_without_grad:
                logger.warning(f"WARNING: {len(params_without_grad)} trainable params have NO gradients!")
                logger.warning(f"First 5: {params_without_grad[:5]}")
        
        # Log to tracker
        if self.trainer.accelerator.is_main_process:
            metrics = {
                'gradients/global_norm': global_grad_norm,
                'gradients/lora_norm': lora_grad_norm,
                'gradients/params_with_grad': len(params_with_grad),
                'gradients/params_without_grad': len(params_without_grad),
            }
            
            # Add max/min/mean gradient norms
            if grad_norms:
                metrics['gradients/max_norm'] = max(grad_norms) ** 0.5
                metrics['gradients/min_norm'] = min(grad_norms) ** 0.5
                metrics['gradients/mean_norm'] = (sum(grad_norms) / len(grad_norms)) ** 0.5
            
            self.trainer.accelerator.log(metrics, step=self.trainer.global_step)
        
        return {
            'global_grad_norm': global_grad_norm,
            'lora_grad_norm': lora_grad_norm,
            'params_with_grad': len(params_with_grad),
            'params_without_grad': len(params_without_grad),
        }
    
    def check_lora_hooks(self) -> Dict[str, Any]:
        """
        Check 3: Verify LoRA hooks actually landed on the correct modules.
        
        Returns:
            Dictionary with LoRA hook information
        """
        logger.info("=" * 80)
        logger.info("CHECK 3: LORA HOOKS VERIFICATION")
        logger.info("=" * 80)
        
        lora_modules = []
        target_modules = self.trainer.config.model.lora.target_modules
        
        # Find all LoRA modules
        for name, module in self.trainer.dit.named_modules():
            # Check if this module has LoRA adapters
            has_lora = (
                hasattr(module, 'lora_A') or 
                hasattr(module, 'lora_B') or
                any('lora' in pname.lower() for pname, _ in module.named_parameters(recurse=False))
            )
            
            if has_lora:
                lora_modules.append(name)
        
        logger.info(f"Found {len(lora_modules)} modules with LoRA adapters")
        logger.info(f"Target modules pattern: {target_modules}")
        
        # Verify that LoRA is applied to expected modules
        expected_patterns = target_modules
        matched_modules = defaultdict(list)
        
        for module_name in lora_modules:
            for pattern in expected_patterns:
                if pattern in module_name:
                    matched_modules[pattern].append(module_name)
                    break
        
        logger.info("\nLoRA modules by target pattern:")
        for pattern, modules in matched_modules.items():
            logger.info(f"  {pattern}: {len(modules)} modules")
            for mod in modules[:3]:
                logger.info(f"    - {mod}")
            if len(modules) > 3:
                logger.info(f"    ... and {len(modules) - 3} more")
        
        # Check for unexpected LoRA modules
        unmatched = []
        for module_name in lora_modules:
            if not any(pattern in module_name for pattern in expected_patterns):
                unmatched.append(module_name)
        
        if unmatched:
            logger.warning(f"Found {len(unmatched)} LoRA modules NOT matching target patterns:")
            for mod in unmatched[:5]:
                logger.warning(f"  - {mod}")
        
        # Verify LoRA parameters are trainable
        lora_trainable = sum(
            1 for n, p in self.trainer.dit.named_parameters()
            if 'lora' in n.lower() and p.requires_grad
        )
        
        lora_total = sum(
            1 for n, p in self.trainer.dit.named_parameters()
            if 'lora' in n.lower()
        )
        
        logger.info(f"\nLoRA parameters: {lora_trainable}/{lora_total} trainable")
        
        if lora_trainable == 0:
            logger.error("ERROR: No LoRA parameters are trainable!")
        elif lora_trainable < lora_total:
            logger.warning(f"WARNING: Not all LoRA parameters are trainable ({lora_trainable}/{lora_total})")
        else:
            logger.info("✓ All LoRA parameters are trainable")
        
        result = {
            'lora_module_count': len(lora_modules),
            'matched_patterns': {k: len(v) for k, v in matched_modules.items()},
            'unmatched_count': len(unmatched),
            'lora_trainable': lora_trainable,
            'lora_total': lora_total,
        }
        
        # Log to tracker
        if self.trainer.accelerator.is_main_process:
            self.trainer.accelerator.log({
                'diagnostics/lora_modules': len(lora_modules),
                'diagnostics/lora_trainable_params': lora_trainable,
            }, step=self.trainer.global_step)
        
        return result
    
    def check_timestep_sampling(self, timesteps: torch.Tensor) -> Dict[str, float]:
        """
        Check 4: Verify timestep sampling is uniform.
        
        Args:
            timesteps: Timesteps tensor from current batch
            
        Returns:
            Dictionary with timestep statistics
        """
        # Store timesteps for analysis
        self.timestep_history.extend(timesteps.cpu().numpy().tolist())
        
        # Keep only last 1000 timesteps for analysis
        if len(self.timestep_history) > 1000:
            self.timestep_history = self.timestep_history[-1000:]
        
        # Compute statistics
        current_mean = timesteps.float().mean().item()
        current_std = timesteps.float().std().item()
        current_min = timesteps.float().min().item()
        current_max = timesteps.float().max().item()
        
        # Compute statistics on history
        if len(self.timestep_history) >= 100:
            hist_array = np.array(self.timestep_history)
            hist_mean = hist_array.mean()
            hist_std = hist_array.std()
            hist_min = hist_array.min()
            hist_max = hist_array.max()
            
            # Check if distribution looks uniform
            # For uniform [0, max], mean should be ~max/2
            expected_mean = (hist_min + hist_max) / 2
            mean_deviation = abs(hist_mean - expected_mean) / expected_mean
            
            # Compute histogram to check uniformity
            num_bins = 10
            hist, bin_edges = np.histogram(hist_array, bins=num_bins)
            expected_count = len(hist_array) / num_bins
            uniformity_score = 1.0 - np.std(hist - expected_count) / expected_count
            
            logger.info("=" * 80)
            logger.info("CHECK 4: TIMESTEP SAMPLING UNIFORMITY")
            logger.info("=" * 80)
            logger.info(f"Current batch timesteps:")
            logger.info(f"  Mean: {current_mean:.1f}, Std: {current_std:.1f}")
            logger.info(f"  Range: [{current_min:.1f}, {current_max:.1f}]")
            logger.info(f"\nHistorical timesteps (last {len(self.timestep_history)}):")
            logger.info(f"  Mean: {hist_mean:.1f} (expected ~{expected_mean:.1f})")
            logger.info(f"  Std: {hist_std:.1f}")
            logger.info(f"  Range: [{hist_min:.1f}, {hist_max:.1f}]")
            logger.info(f"  Mean deviation: {mean_deviation:.2%}")
            logger.info(f"  Uniformity score: {uniformity_score:.3f} (1.0 = perfect uniform)")
            
            # Warnings
            if mean_deviation > 0.1:
                logger.warning(f"WARNING: Timestep mean deviates {mean_deviation:.2%} from expected!")
                logger.warning("This suggests non-uniform sampling (possibly weighted)")
            
            if uniformity_score < 0.8:
                logger.warning(f"WARNING: Uniformity score is low ({uniformity_score:.3f})")
                logger.warning("This suggests biased timestep sampling")
            
            # Log to tracker
            if self.trainer.accelerator.is_main_process:
                self.trainer.accelerator.log({
                    'diagnostics/timestep_mean': hist_mean,
                    'diagnostics/timestep_std': hist_std,
                    'diagnostics/timestep_uniformity': uniformity_score,
                    'diagnostics/timestep_mean_deviation': mean_deviation,
                }, step=self.trainer.global_step)
            
            return {
                'current_mean': current_mean,
                'current_std': current_std,
                'hist_mean': hist_mean,
                'hist_std': hist_std,
                'uniformity_score': uniformity_score,
                'mean_deviation': mean_deviation,
            }
        
        return {
            'current_mean': current_mean,
            'current_std': current_std,
        }
    
    def check_target_and_mask(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        edit_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Check 5: Verify target type and mask normalization.
        
        Args:
            model_pred: Model prediction tensor
            target: Target tensor
            edit_mask: Optional edit mask tensor
            
        Returns:
            Dictionary with target and mask statistics
        """
        logger.info("=" * 80)
        logger.info("CHECK 5: TARGET TYPE & MASK NORMALIZATION")
        logger.info("=" * 80)
        
        results = {}
        
        # Check target type
        logger.info("Target information:")
        logger.info(f"  Shape: {target.shape}")
        logger.info(f"  Dtype: {target.dtype}")
        logger.info(f"  Range: [{target.min().item():.4f}, {target.max().item():.4f}]")
        logger.info(f"  Mean: {target.mean().item():.4f}, Std: {target.std().item():.4f}")
        
        logger.info("Model prediction information:")
        logger.info(f"  Shape: {model_pred.shape}")
        logger.info(f"  Dtype: {model_pred.dtype}")
        logger.info(f"  Range: [{model_pred.min().item():.4f}, {model_pred.max().item():.4f}]")
        logger.info(f"  Mean: {model_pred.mean().item():.4f}, Std: {model_pred.std().item():.4f}")
        
        # Check scheduler prediction type
        prediction_type = getattr(self.trainer.scheduler.config, 'prediction_type', 'epsilon')
        logger.info(f"\nScheduler prediction type: {prediction_type}")
        
        # Verify target matches prediction type
        # For flow matching, target is typically (noise - image_latents)
        # Check if this looks reasonable
        target_magnitude = target.abs().mean().item()
        logger.info(f"Target magnitude (abs mean): {target_magnitude:.4f}")
        
        if target_magnitude > 10:
            logger.warning("WARNING: Target magnitude is very large (>10)")
            logger.warning("This might indicate incorrect target computation")
        elif target_magnitude < 0.01:
            logger.warning("WARNING: Target magnitude is very small (<0.01)")
            logger.warning("This might indicate incorrect target computation")
        
        results.update({
            'target_mean': target.mean().item(),
            'target_std': target.std().item(),
            'target_min': target.min().item(),
            'target_max': target.max().item(),
            'pred_mean': model_pred.mean().item(),
            'pred_std': model_pred.std().item(),
            'prediction_type': prediction_type,
        })
        
        # Check mask normalization if mask is provided
        if edit_mask is not None:
            logger.info("\nMask information:")
            logger.info(f"  Shape: {edit_mask.shape}")
            logger.info(f"  Dtype: {edit_mask.dtype}")
            logger.info(f"  Range: [{edit_mask.min().item():.4f}, {edit_mask.max().item():.4f}]")
            logger.info(f"  Mean: {edit_mask.mean().item():.4f}")
            
            # Check mask normalization
            # Mask should typically be in [0, 1] range
            if edit_mask.min() < 0 or edit_mask.max() > 1:
                logger.warning(f"WARNING: Mask values outside [0, 1] range!")
            
            # Check foreground/background split
            fg_mask = edit_mask > 0.5
            bg_mask = ~fg_mask
            
            fg_ratio = fg_mask.float().mean().item()
            logger.info(f"  Foreground ratio: {fg_ratio:.2%}")
            logger.info(f"  Background ratio: {(1-fg_ratio):.2%}")
            
            # Check if mask weights are normalized
            # For proper normalization, mean should be close to 1.0
            fg_weight = self.trainer.config.loss.forground_weight
            bg_weight = self.trainer.config.loss.background_weight
            
            logger.info(f"\nMask loss configuration:")
            logger.info(f"  Foreground weight: {fg_weight}")
            logger.info(f"  Background weight: {bg_weight}")
            
            # Compute effective weight per sample
            # w = fg*fg_w + (1-fg)*bg_w
            effective_weights = edit_mask * fg_weight + (1 - edit_mask) * bg_weight
            effective_mean = effective_weights.mean().item()
            
            logger.info(f"  Effective weight mean: {effective_mean:.4f}")
            
            # Check if weights are normalized (should be close to 1.0)
            if abs(effective_mean - 1.0) > 0.1:
                logger.warning(f"WARNING: Effective weight mean ({effective_mean:.4f}) deviates from 1.0")
                logger.warning("Consider normalizing mask weights:")
                logger.warning("  w = w * (w.numel() / (w.sum() + 1e-8))")
            else:
                logger.info("✓ Mask weights are reasonably normalized")
            
            results.update({
                'mask_mean': edit_mask.mean().item(),
                'mask_min': edit_mask.min().item(),
                'mask_max': edit_mask.max().item(),
                'fg_ratio': fg_ratio,
                'effective_weight_mean': effective_mean,
            })
            
            # Log to tracker
            if self.trainer.accelerator.is_main_process:
                self.trainer.accelerator.log({
                    'diagnostics/mask_fg_ratio': fg_ratio,
                    'diagnostics/mask_effective_weight': effective_mean,
                }, step=self.trainer.global_step)
        
        # Log to tracker
        if self.trainer.accelerator.is_main_process:
            self.trainer.accelerator.log({
                'diagnostics/target_magnitude': target_magnitude,
                'diagnostics/target_mean': target.mean().item(),
                'diagnostics/pred_mean': model_pred.mean().item(),
            }, step=self.trainer.global_step)
        
        return results

