#!/usr/bin/env python3
"""
Script to apply diagnostic checks to the Qwen Image Edit training pipeline.

This script patches the trainer to add comprehensive diagnostic checks:
1. Trainable parameters verification
2. Gradient norm logging (before zeroing)
3. LoRA hooks verification  
4. Timestep sampling uniformity check
5. Target type and mask normalization check

Usage:
    python script/apply_diagnostic_checks.py --config configs/qwen_image_edit_plus_custom.yaml --stage fit
    
    # Or with cache stage:
    python script/apply_diagnostic_checks.py --config configs/qwen_image_edit_plus_custom.yaml --stage cache
"""

import argparse
import sys
import logging

# Add project root to path
sys.path.insert(0, '/workspace/hardik/test_repos/qwen-image-finetune')

from src.data.config import load_config_from_yaml
from src.main import import_trainer

# Import diagnostic checker
from script.diagnostic_checks import DiagnosticChecker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def patch_trainer_with_diagnostics(trainer_class):
    """
    Monkey patch the trainer class to add diagnostic checks.
    
    This adds diagnostic checks at critical points in the training loop.
    """
    original_init = trainer_class.__init__
    original_compute_loss = trainer_class._compute_loss
    original_train_epoch = trainer_class.train_epoch
    
    def new_init(self, config):
        """Initialize trainer with diagnostic checker."""
        original_init(self, config)
        
        # Add diagnostic checker
        self.diagnostic_checker = DiagnosticChecker(self)
        logger.info("✓ Diagnostic checker added to trainer")
    
    def new_compute_loss(self, embeddings):
        """Compute loss with diagnostic checks."""
        device = self.accelerator.device
        image_latents = embeddings["image_latents"].to(self.weight_dtype).to(device)
        control_latents = embeddings["control_latents"].to(self.weight_dtype).to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        prompt_embeds_mask = embeddings["prompt_embeds_mask"].to(dtype=torch.int64).to(device)
        batch_size = image_latents.shape[0]
        
        # Compute img_shapes from embeddings
        img_shapes = self._get_image_shapes(embeddings, batch_size)
        if "mask" in embeddings:
            edit_mask = embeddings["mask"]
        else:
            edit_mask = None

        with torch.no_grad():
            noise = torch.randn_like(image_latents, device=device, dtype=self.weight_dtype)

            # Sample timesteps
            from diffusers.training_utils import compute_density_for_timestep_sampling
            u = compute_density_for_timestep_sampling(
                weighting_scheme="none",
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            # Clamp indices to valid range to prevent out-of-bounds access when u = 1.0
            indices = torch.clamp(indices, 0, len(self.scheduler.timesteps) - 1)
            timesteps = self.scheduler.timesteps[indices].to(device=device)

            sigmas = self._get_sigmas(timesteps, n_dim=image_latents.ndim, dtype=image_latents.dtype)
            noisy_model_input = (1.0 - sigmas) * image_latents + sigmas * noise
            packed_input = torch.cat([noisy_model_input, control_latents], dim=1)
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        model_pred = self.dit(
            hidden_states=packed_input,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]
        model_pred = model_pred[:, : image_latents.size(1)]
        
        from diffusers.training_utils import compute_loss_weighting_for_sd3
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
        target = noise - image_latents
        
        # Compute loss
        loss_result = self.forward_loss(model_pred, target, weighting, edit_mask)
        
        # === DIAGNOSTIC CHECKS START ===
        # Run diagnostic checks periodically
        if hasattr(self, 'diagnostic_checker') and self.global_step % 10 == 0:
            # Note: We run checks here BEFORE backward to capture timesteps
            # Gradient checks will run in train_epoch after backward
            self.diagnostic_checker.check_timestep_sampling(timesteps)
            self.diagnostic_checker.check_target_and_mask(model_pred, target, edit_mask)
        # === DIAGNOSTIC CHECKS END ===
        
        # Log timestep statistics
        if self.accelerator.is_main_process and self.global_step % 10 == 0:
            timestep_metrics = {
                'training/timestep_mean': timesteps.float().mean().item(),
                'training/timestep_std': timesteps.float().std().item(),
            }
            self.accelerator.log(timestep_metrics, step=self.global_step)
        
        # Log mask loss breakdown if available
        if isinstance(loss_result, dict):
            if self.accelerator.is_main_process:
                mask_metrics = {}
                if 'fg_loss' in loss_result:
                    mask_metrics['loss/foreground'] = loss_result['fg_loss'].item()
                if 'bg_loss' in loss_result:
                    mask_metrics['loss/background'] = loss_result['bg_loss'].item()
                if 'mask_coverage' in loss_result:
                    mask_metrics['loss/mask_coverage'] = loss_result['mask_coverage'].item()
                
                if mask_metrics:
                    self.accelerator.log(mask_metrics, step=self.global_step)
            
            # Return the actual loss for backprop
            return loss_result['total_loss']
        
        return loss_result
    
    def new_train_epoch(self, epoch, train_dataloader):
        """Training epoch with diagnostic checks."""
        import torch
        
        for _, batch in enumerate(train_dataloader):
            # 检查是否收到中断信号
            if self.training_interrupted:
                logger.info("检测到训练中断信号，保存最后检查点后退出本epoch...")
                self.save_checkpoint(epoch, self.global_step, is_last=True)
                return

            with self.accelerator.accumulate(self.dit):
                loss = self.training_step(batch)
                self.fps_logger.update(
                    batch_size=self.batch_size * self.accelerator.num_processes,
                    num_tokens=None,
                )
                self.accelerator.backward(loss)
                
                # === DIAGNOSTIC CHECK: Gradient norms (BEFORE clipping/zeroing) ===
                if hasattr(self, 'diagnostic_checker') and self.accelerator.sync_gradients:
                    self.diagnostic_checker.check_gradient_norms()
                # === END DIAGNOSTIC CHECK ===
                
                self.clip_gradients()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            # Clear cache periodically to prevent memory buildup with large images
            if self.global_step % 10 == 0:
                torch.cuda.empty_cache()
            
            if self.accelerator.sync_gradients:
                avg_loss = self.accelerator.gather(loss.detach()).mean()
                self.train_loss = avg_loss.item() / self.config.train.gradient_accumulation_steps
                self.running_loss = 0.9 * self.running_loss + 0.1 * self.train_loss
                self.update_progressbar(
                    logs={
                        "loss": self.train_loss,
                        "smooth_loss": self.running_loss,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "fps": self.fps_logger.total_fps(),
                    }
                )
                
                # Log enhanced metrics periodically
                enhanced_interval = getattr(self.config.logging, 'enhanced_metrics_interval', 10)
                if self.global_step % enhanced_interval == 0:
                    self.log_enhanced_metrics(batch_data=batch)
                
                self.save_checkpoint(epoch, self.global_step)
                if self.validation_sampler and self.validation_sampler.should_run_validation(self.global_step):
                    self.fps_logger.pause()
                    try:
                        self.validation_sampler.run_validation_loop(
                            global_step=self.global_step,
                            trainer=self,
                        )
                    except Exception as e:
                        self.accelerator.print(f"Validation sampling failed: {e}")
                    self.fps_logger.resume()
    
    # Apply patches
    trainer_class.__init__ = new_init
    trainer_class._compute_loss = new_compute_loss
    trainer_class.train_epoch = new_train_epoch
    
    logger.info(f"✓ Patched {trainer_class.__name__} with diagnostic checks")
    return trainer_class


def main():
    parser = argparse.ArgumentParser(description="Apply diagnostic checks to Qwen training")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--stage', type=str, default='fit', choices=['fit', 'cache'],
                       help='Training stage (fit or cache)')
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config_from_yaml(args.config)
    
    # Import and patch trainer
    logger.info(f"Importing trainer: {config.trainer}")
    Trainer = import_trainer(config)
    
    # Patch trainer with diagnostics
    logger.info("Patching trainer with diagnostic checks...")
    Trainer = patch_trainer_with_diagnostics(Trainer)
    
    # Create trainer instance
    logger.info("Creating trainer instance...")
    trainer = Trainer(config)
    
    # Load dataset
    logger.info("Loading dataset...")
    from src.data.dataset import loader
    train_dataloader = loader(
        config.data.class_path,
        config.data.init_args,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=config.data.shuffle,
        drop_last=True,
    )
    
    # Run selected stage
    if args.stage == 'cache':
        logger.info("Running cache stage with diagnostics...")
        trainer.cache(train_dataloader)
    else:
        logger.info("Running fit stage with diagnostics...")
        trainer.fit(train_dataloader)
    
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC RUN COMPLETE")
    logger.info("=" * 80)
    logger.info("Check your logs and wandb/tensorboard for diagnostic metrics")


if __name__ == '__main__':
    main()

