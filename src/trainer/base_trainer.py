"""
Abstract Base Trainer for all trainer implementations.
Defines the core interface that all trainers must implement.
"""

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy  # transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
import os
import shutil
import glob
import json
from functools import partial
import numpy as np
from src.utils.sampling import calculate_shift, retrieve_timesteps
from tqdm import tqdm
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict
import logging
import PIL
import signal
import importlib

import safetensors.torch
from src.models.quantize import quantize_model_to_fp8
from src.utils.tools import calculate_sha256_file
from src.data.config import Config
from src.utils.model_summary import print_model_summary_table
from src.utils.lora_utils import FpsLogger
from src.utils.tools import get_git_info
from src.utils.tools import instantiate_class
from src.scheduler.custom_flowmatch_scheduler import FlowMatchEulerDiscreteScheduler
from src.data.cache_manager import EmbeddingCacheManager
from diffusers import FluxKontextPipeline
from src.utils.lora_utils import classify_lora_weight
from src.utils.huggingface import download_lora
from src.trainer.constants import LORA_FILE_BASE_NAME

logger = logging.getLogger(__name__)


def collect_lora_linears(root: nn.Module):
    loras = []
    for m in root.modules():
        if isinstance(m, nn.Linear):
            # Common PEFT LoRA markers
            if (
                hasattr(m, "lora_A")
                or hasattr(m, "lora_B")
                or hasattr(m, "lora_embedding_A")
                or hasattr(m, "lora_embedding_B")
                or hasattr(m, "lora_edit")
            ):  # Custom LoRA marker
                loras.append(m)
                continue
            # Fallback: Linear layer has parameters with "lora" in name, or trainable params besides weight/bias
            names_params = dict(m.named_parameters(recurse=False))
            if any(("lora" in n) for n in names_params.keys()):
                loras.append(m)
                continue
            other_trainables = [p for n, p in names_params.items() if n not in ("weight", "bias") and p.requires_grad]
            if other_trainables:
                loras.append(m)
    return loras


class BaseTrainer(ABC):
    """
    Abstract base class for all trainer implementations.
    Defines the core interface that all trainers must implement.
    """

    def __init__(self, config: Config):
        """Initialize trainer with configuration."""
        self.config = config
        self.accelerator: Optional[Accelerator] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0
        self.scheduler: FlowMatchEulerDiscreteScheduler = None
        self.inference_scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None  # Separate scheduler for validation

        # Common attributes that all trainers should have
        self.weight_dtype = torch.bfloat16
        self.batch_size = self.config.data.batch_size
        self.use_cache = self.config.cache.use_cache
        self.cache_dir = self.config.cache.cache_dir
        self.fps_logger = FpsLogger()
        self.cache_manager = EmbeddingCacheManager(self.cache_dir)
        self.cache_exist = self.cache_manager.exist(self.cache_dir)
        self.quantize = self.config.model.quantize
        self.adapter_name = self.config.model.lora.adapter_name
        self.predict_setted = False

        # Attribute for save_last_checkpoint functionality
        self.training_interrupted = False
        self.log_model_info()
        self.load_preprocessor()
        self.pipeline_class = self.get_pipeline_class()

    @abstractmethod
    def get_pipeline_class(self):
        """return the pipeline class to use the classmethod"""
        return FluxKontextPipeline

    def load_preprocessor(self):
        class_path = self.config.data.init_args.processor.class_path
        init_args = self.config.data.init_args.processor.init_args
        self.preprocessor = instantiate_class(class_path, init_args)

    def __repr__(self) -> str:
        msg = f"{self.__class__.__name__}(config={self.config})"
        return msg

    def setup_signal_handlers(self):
        """Set up signal handlers to capture Ctrl+C interrupts"""

        def signal_handler(signum, frame):
            logging.info("Received interrupt signal, preparing to save final checkpoint...")
            self.training_interrupted = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def log_model_info(self):
        """Log model information."""
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Use Cache: {self.use_cache}")

    def setup_versioned_logging_dir(self):
        """Set up versioned logging directory"""
        base_output_dir = self.config.logging.output_dir
        project_name = self.config.logging.tracker_project_name

        # Create project directory structure: output_dir/project_name/v0
        project_dir = os.path.join(base_output_dir, project_name)

        # If project directory doesn't exist, use v0 directly
        if not os.path.exists(project_dir):
            versioned_dir = os.path.join(project_dir, "v0")
            self.config.logging.output_dir = versioned_dir
            logging.info(f"Creating new training version directory: {versioned_dir}")
            return

        # Find existing versions
        existing_versions = []
        for item in os.listdir(project_dir):
            item_path = os.path.join(project_dir, item)
            if os.path.isdir(item_path) and item.startswith("v") and item[1:].isdigit():
                version_num = int(item[1:])
                existing_versions.append((version_num, item_path))

        # Clean up invalid versions (training steps < 5)
        valid_versions = []
        for version_num, version_path in existing_versions:
            if self._is_valid_training_version(version_path):
                valid_versions.append(version_num)
            else:
                logging.info(f"Removing invalid training version: {version_path}")
                try:
                    shutil.rmtree(version_path)
                except Exception as e:
                    logging.info(f"Failed to remove invalid training version: {version_path}, {e}")

        # Determine new version number
        if valid_versions:
            next_version = max(valid_versions) + 1
        else:
            next_version = 0

        # Create new version directory
        versioned_dir = os.path.join(project_dir, f"v{next_version}")
        self.config.logging.output_dir = versioned_dir
        logging.info(f"Using training version directory: {versioned_dir}")

    def _is_valid_training_version(self, version_path):
        """Returns True if the folder contains a checkpoint"""
        # Check checkpoint directory

        checkpoints = glob.glob(f"{version_path}/*/*.safetensors")
        return len(checkpoints) > 0

    def accelerator_prepare(self, train_dataloader):
        """Prepare accelerator"""
        # from diffusers.loaders import AttnProcsLayers
        # from src.utils.lora_utils import get_lora_layers
        # lora_layers_model = AttnProcsLayers(get_lora_layers(self.dit))
        # Enable gradient checkpointing based on configuration
        if self.config.train.gradient_checkpointing:
            self.dit.enable_gradient_checkpointing()
        if self.config.resume is not None:
            # Try to load optimizer and scheduler state if they exist (only in "last" checkpoints)
            optimizer_path = os.path.join(self.config.resume, "optimizer.bin")
            scheduler_path = os.path.join(self.config.resume, "scheduler.bin")
            
            if os.path.exists(optimizer_path) and os.path.exists(scheduler_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path))
                self.lr_scheduler.load_state_dict(torch.load(scheduler_path))
                logging.info(f"Loaded optimizer and scheduler from {self.config.resume}")
            else:
                logging.info(f"Optimizer/scheduler state not found in {self.config.resume}, will reinitialize (expected for regular checkpoints)")

        # sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)

        plug = getattr(self.accelerator.state, "fsdp_plugin", None)
        if plug is not None:
            from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision

            # torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)

            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)

            plug.use_orig_params = True
            plug.ignored_modules = collect_lora_linears(self.dit)
            plug.limit_all_gathers = True
            # plug.forward_prefetch = False
            # plug.backward_prefetch = BackwardPrefetch.BACKWARD_POST
            plug.forward_prefetch = True
            plug.backward_prefetch = BackwardPrefetch.BACKWARD_PRE  # Prefetch during backward pass
            plug.sync_module_states = True  # Broadcast initialization from main GPU to avoid inconsistencies
            plug.min_num_params = 20_000_000  # 5_000_000
            plug.mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
                cast_forward_inputs=False,
            )
            # from src.models.transformer_qwenimage import QwenImageTransformerBlock  # Your class path
            # plug.auto_wrap_policy = partial(
            #     transformer_auto_wrap_policy,
            #     transformer_layer_cls={QwenImageTransformerBlock},  # Note: keyword argument
            # )
            plug.auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=plug.min_num_params)
            # from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            plug.sharding_strategy = ShardingStrategy.FULL_SHARD  # FULL_SHARD   # SHARD_GRAD_OP

            self.dit = self.dit.to("cpu")
            torch.cuda.empty_cache()
            import gc

            gc.collect()

            self.dit, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                self.dit, self.optimizer, train_dataloader, self.lr_scheduler
            )
        else:
            from diffusers.loaders import AttnProcsLayers
            from src.utils.lora_utils import get_lora_layers

            lora_layers_model = AttnProcsLayers(get_lora_layers(self.dit))
            lora_layers_model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                lora_layers_model, self.optimizer, train_dataloader, self.lr_scheduler
            )
            self.dit = self.dit.to(self.accelerator.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Trackers already initialized in setup_accelerator
        return train_dataloader

    def merge_lora(self):
        """Merge LoRA weights into base model"""
        self.dit.merge_adapter()
        logging.info("Merged LoRA weights into base model")

    def cache(self, train_dataloader):
        """Pre-compute and cache embeddings/latents for training efficiency."""
        logging.info("Starting embedding caching process...")
        self.load_model()
        self.setup_model_device_train_mode(stage="cache", cache=True)
        # Cache for each item (same loop structure as QwenImageEditTrainer)

        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc="cache_embeddings"):
            batch = self.prepare_embeddings(batch, stage="cache")
            self.cache_step(batch)
        self.destroy_models()
        logging.info("Cache completed")

    def destroy_models(self):
        import gc

        if hasattr(self, "text_encoder"):
            self.text_encoder.cpu()
            del self.text_encoder
        if hasattr(self, "text_encoder_2"):
            self.text_encoder_2.cpu()
            del self.text_encoder_2
        if hasattr(self, "vae"):
            self.vae.cpu()
            del self.vae
        if hasattr(self, "vit"):
            self.vit.cpu()
            del self.vit
        torch.cuda.empty_cache()
        gc.collect()

    def setup_validation(self, train_dataloader):
        """Setup validation sampler and cache embeddings"""
        self.validation_sampler = None
        
        # Check if sampling is enabled
        if not self.config.logging.sampling.enable:
            logging.info("Validation sampling disabled in config")
            return
        
        from src.validation.validation_sampler import ValidationSampler
        
        # Create validation sampler
        self.validation_sampler = ValidationSampler(
            config=self.config.logging.sampling,
            accelerator=self.accelerator,
            weight_dtype=self.weight_dtype,
            data_config=self.config.data,
            compute_metrics=self.config.logging.sampling.compute_metrics,
            lpips_net=self.config.logging.sampling.lpips_net,
        )
        
        # Setup validation dataset
        self.validation_sampler.setup_validation_dataset(train_dataloader.dataset)
        
        # Cache embeddings for validation
        if hasattr(self, 'vae') and hasattr(self, 'text_encoder'):
            logging.info("Caching validation embeddings...")
            try:
                self.validation_sampler.cache_embeddings(self)
                logging.info(f"Successfully cached {len(self.validation_sampler.cached_embeddings)} validation samples")
            except Exception as e:
                logging.warning(f"Failed to cache validation embeddings: {e}")
                self.validation_sampler = None
        else:
            logging.warning("Models not loaded, cannot cache validation embeddings")

    def clip_gradients(self):
        """Clip gradients"""
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.dit.parameters(),
                self.config.train.max_grad_norm,
            )

    def training_step(self, batch: dict) -> torch.Tensor:
        """Training step, batch from dataloader"""
        if all(batch["cached"]):
            return self._training_step_cached(batch)
        return self._training_step_compute(batch)

    def _training_step_cached(self, batch: dict) -> torch.Tensor:
        """Training step with cached data"""
        embeddings = self.prepare_cached_embeddings(batch)
        return self._compute_loss(embeddings)

    def _training_step_compute(self, batch: dict) -> torch.Tensor:
        """Training step with real-time data"""
        embeddings = self.prepare_embeddings(batch, stage="fit")
        return self._compute_loss(embeddings)

    @abstractmethod
    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        """Compute loss. returned the flow matching loss tensor"""
        pass

    def forward_loss(self, model_pred, target, weighting=None, edit_mask=None):
        """Compute loss with optional mask-based breakdown"""
        loss_dict = {}
        
        if edit_mask is None:
            if weighting is None:
                loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
            else:
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
            loss_dict['total_loss'] = loss
        else:
            # shape torch.Size([4, 864, 1216]) torch.Size([4, 4104, 64]) torch.Size([4, 4104, 64]) torch.Size([4, 1, 1])
            loss = self.criterion(edit_mask, model_pred, target, weighting)
            loss_dict['total_loss'] = loss
            
            # Compute foreground/background breakdown if mask loss is enabled
            if self.config.loss.mask_loss and edit_mask is not None:
                with torch.no_grad():
                    squared_error = (model_pred.float() - target.float()) ** 2
                    if weighting is not None:
                        squared_error = weighting.float() * squared_error
                    
                    # Reshape to match mask dimensions
                    squared_error_flat = squared_error.reshape(target.shape[0], -1)
                    mask_flat = edit_mask.reshape(edit_mask.shape[0], -1)
                    
                    # Compute foreground and background losses
                    fg_mask = mask_flat > 0.5
                    bg_mask = ~fg_mask
                    
                    if fg_mask.any():
                        fg_loss = squared_error_flat[fg_mask].mean()
                        loss_dict['fg_loss'] = fg_loss
                    
                    if bg_mask.any():
                        bg_loss = squared_error_flat[bg_mask].mean()
                        loss_dict['bg_loss'] = bg_loss
                    
                    # Compute mask coverage
                    mask_coverage = fg_mask.float().mean()
                    loss_dict['mask_coverage'] = mask_coverage
        
        return loss_dict if len(loss_dict) > 1 else loss_dict['total_loss']
    
    def log_enhanced_metrics(self, batch_data=None):
        """Log enhanced metrics including gradients, parameters, and memory usage"""
        metrics = {}
        
        # Log gradient norms
        if hasattr(self.config.logging, 'log_gradients') and self.config.logging.log_gradients:
            try:
                # Compute global gradient norm across all parameters
                total_norm = 0.0
                lora_norm = 0.0
                
                for name, param in self.dit.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                        
                        if 'lora' in name.lower():
                            lora_norm += param_norm ** 2
                
                total_norm = total_norm ** 0.5
                lora_norm = lora_norm ** 0.5
                
                metrics['gradients/global_norm'] = total_norm
                if lora_norm > 0:
                    metrics['gradients/lora_norm'] = lora_norm
            except Exception as e:
                logging.debug(f"Failed to compute gradient norms: {e}")
        
        # Log parameter statistics
        if hasattr(self.config.logging, 'log_parameters') and self.config.logging.log_parameters:
            try:
                lora_params = []
                for name, param in self.dit.named_parameters():
                    if 'lora' in name.lower() and param.requires_grad:
                        lora_params.append(param.data.flatten())
                
                if len(lora_params) > 0:
                    all_lora_params = torch.cat(lora_params)
                    metrics['parameters/mean'] = all_lora_params.mean().item()
                    metrics['parameters/std'] = all_lora_params.std().item()
                    metrics['parameters/max'] = all_lora_params.max().item()
                    metrics['parameters/min'] = all_lora_params.min().item()
            except Exception as e:
                logging.debug(f"Failed to compute parameter statistics: {e}")
        
        # Log memory usage
        if hasattr(self.config.logging, 'log_memory') and self.config.logging.log_memory:
            try:
                if torch.cuda.is_available():
                    # Convert bytes to GB
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    
                    metrics['memory/allocated_gb'] = allocated
                    metrics['memory/reserved_gb'] = reserved
                    metrics['memory/max_allocated_gb'] = max_allocated
            except Exception as e:
                logging.debug(f"Failed to compute memory statistics: {e}")
        
        # Log to accelerator if we have metrics
        if metrics and self.accelerator.is_main_process:
            self.accelerator.log(metrics, step=self.global_step)
        
        return metrics

    def train_epoch(self, epoch, train_dataloader):
        # Log epoch start
        if self.accelerator.is_main_process:
            logging.info(f"=" * 60)
            logging.info(f"Starting Epoch {epoch} | Global Step: {self.global_step}")
            logging.info(f"Dataloader shuffle: {train_dataloader.dataset.generator is not None if hasattr(train_dataloader.dataset, 'generator') else 'N/A'}")
            logging.info(f"Dataset size: {len(train_dataloader.dataset)}")
            logging.info(f"=" * 60)
        
        for _, batch in enumerate(train_dataloader):
            # Check for interrupt signal
            if self.training_interrupted:
                logger.info("Training interruption detected, saving final checkpoint before exiting epoch...")
                # Immediately save "last" checkpoint (even if not at checkpointing_steps interval)
                self.save_checkpoint(epoch, self.global_step, is_last=True)
                return

            with self.accelerator.accumulate(self.dit):
                loss = self.training_step(batch)
                self.fps_logger.update(
                    batch_size=self.batch_size * self.accelerator.num_processes,
                    num_tokens=None,
                )
                self.accelerator.backward(loss)
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
                            trainer=self,  # Pass trainer instance
                        )
                    except Exception as e:
                        self.accelerator.print(f"Validation sampling failed: {e}")
                    self.fps_logger.resume()

    def setup_progressbar(self):
        self.train_loss = 0.0
        self.running_loss = 0.0
        if self.config.resume is not None:
            with open(os.path.join(self.config.resume, "state.json")) as f:
                st = json.load(f)
            self.global_step = st["global_step"]
            self.start_epoch = st["epoch"]
        else:
            self.global_step = 0
            self.start_epoch = 0
        self.progress_bar = tqdm(
            range(self.global_step, self.config.train.max_train_steps),
            desc="fit",
            disable=(not self.accelerator.is_local_main_process),
        )

        # if max_train_steps exist, use is None, use num_epochs

        self.num_epochs = int(self.config.train.max_train_steps / self.batch_size / self.accelerator.num_processes)

    def update_progressbar(self, logs: dict):
        # Log to wandb/tensorboard with clean names
        tracker_logs = {
            "train/loss": logs["loss"],
            "train/smooth_loss": logs["smooth_loss"],
            "train/learning_rate": logs["lr"],
            "train/epoch": logs["epoch"],
            "train/fps": logs["fps"],
        }
        self.accelerator.log(tracker_logs, step=self.global_step)
        
        # Format for progress bar display
        display_logs = {
            "loss": f"{logs['loss']:.3f}",
            "smooth_loss": f"{logs['smooth_loss']:.3f}",
            "lr": f"{logs['lr']:.1e}",
            "epoch": logs["epoch"],
            "fps": f"{logs['fps']:.2f}",
        }
        self.progress_bar.update(1)
        self.global_step += 1
        self.progress_bar.set_postfix(display_logs)

    def fit(self, train_dataloader):
        """Main training loop implementation."""
        self.setup_signal_handlers()
        self.setup_accelerator()
        self.load_model()
        if self.config.resume is not None:
            import glob

            # add the checkpoint in lora.pretrained_weight config
            model_files = glob.glob(os.path.join(self.config.resume, "*.safetensors"))
            if len(model_files) > 0:
                self.config.model.lora.pretrained_weight = model_files[0]
            else:
                self.config.model.lora.pretrained_weight = os.path.join(self.config.resume, LORA_FILE_BASE_NAME)
            logging.info(f"Loaded checkpoint from {self.config.model.lora.pretrained_weight}")
        if self.config.model.quantize:
            self.dit = self.quantize_model(
                self.dit,
                self.config.predict.devices.dit,
            )
        self.__class__.load_pretrain_lora_model(self.dit, self.config, self.adapter_name)

        self.setup_model_device_train_mode(stage="fit", cache=self.use_cache)
        self.configure_optimizers()
        self.setup_criterion()
        self.setup_training_scheduler()  # Initialize scheduler for training
        self.setup_validation(train_dataloader)

        train_dataloader = self.accelerator_prepare(train_dataloader)
        logging.info("***** Running training *****")
        print_model_summary_table(self.dit)

        self.fps_logger.start()
        self.save_train_config()
        self.setup_progressbar()
        current_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.num_epochs):
            current_epoch = epoch
            self.train_epoch(epoch, train_dataloader)
            if self.training_interrupted:
                break

        # Save final checkpoint
        self.save_checkpoint(current_epoch, self.global_step, is_last=True)

        logging.info(f"FPS: {self.fps_logger.last_fps()}")
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

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
            logging.info(f"✓ Initialized training timesteps: {len(self.scheduler.timesteps)} timesteps from {self.scheduler.timesteps[0]:.1f} to {self.scheduler.timesteps[-1]:.1f}")
        else:
            # Standard scheduler - manually create training timesteps
            num_timesteps = getattr(self.scheduler.config, 'num_train_timesteps', 1000)
            device = self.accelerator.device if self.accelerator else 'cpu'
            timesteps = torch.linspace(num_timesteps, 1, num_timesteps, device=device)
            self.scheduler.timesteps = timesteps
            logging.info(f"✓ Initialized standard training timesteps: {len(timesteps)} timesteps")
        
        # Create separate inference scheduler (copy of training scheduler)
        self._create_inference_scheduler()
    
    def _create_inference_scheduler(self):
        """Create a separate scheduler instance for validation/inference.
        
        This prevents validation from corrupting training timesteps.
        """
        if self.scheduler is None:
            return
        
        # Create a copy of the scheduler with the same configuration
        scheduler_class = self.scheduler.__class__
        self.inference_scheduler = scheduler_class.from_config(self.scheduler.config)
        logging.info(f"✓ Created separate inference scheduler: {scheduler_class.__name__}")
    
    def setup_criterion(self):
        if self.config.loss.mask_loss:
            from src.loss.edit_mask_loss import MaskEditLoss

            self.criterion = MaskEditLoss(
                forground_weight=self.config.loss.forground_weight,
                background_weight=self.config.loss.background_weight,
            )
        else:
            self.criterion = nn.MSELoss()
        self.criterion.to(self.accelerator.device)

    def setup_predict(
        self,
    ):
        if self.predict_setted:
            return
        if not hasattr(self, "dit") or self.vae is None:
            logging.info("Loading model...")
            self.load_model()

        if self.config.model.quantize:
            self.dit = self.quantize_model(
                self.dit,
                self.config.predict.devices.dit,
            )

        if self.config.model.lora.pretrained_weight is not None:
            logging.info("load lora from pretrained weight")
            self.load_pretrain_lora_model(self.dit, self.config, self.config.lora_adapter_name, stage="predict")

        self.setup_model_device_train_mode(stage="predict")
        logging.info("setup_model_device_train_mode done")
        print_model_summary_table(self.dit)
        self.predict_setted = True
        logging.info("setup_predict done")

    @abstractmethod
    def prepare_predict_batch_data(self, *args, **kwargs) -> dict:
        """Prepare predict batch data.
        prepare the data to batch dict that can be used to prepare embeddings similar in the training step.
        We want to reuse the same data preparation code in the training step.
        """
        pass

    def predict(
        self,
        **kwargs,
    ):
        """Inference/prediction method.
        Prepare the data, prepare the embeddings, sample the latents, decode the latents to images.
        """
        self.setup_predict()
        batch = self.prepare_predict_batch_data(**kwargs)
        embeddings: dict = self.prepare_embeddings(batch, stage="predict")
        target_height = embeddings["height"]
        target_width = embeddings["width"]
        latents = self.sampling_from_embeddings(embeddings)
        image = self.decode_vae_latent(latents, target_height, target_width)
        output_type = kwargs.get("output_type", "pil")
        if output_type == "pil":
            image = image.detach().permute(0, 2, 3, 1).float().cpu().numpy()
            image = (image * 255).round().astype("uint8")
            if image.shape[-1] == 1:
                # special case for grayscale (single channel) images
                pil_images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in image]
            else:
                pil_images = [PIL.Image.fromarray(image) for image in image]

            return pil_images
        return image

    def setup_accelerator(self):
        """Initialize accelerator and logging configuration"""
        # Setup versioned logging directory
        self.setup_versioned_logging_dir()

        # Set logging_dir to the versioned output directory directly
        # Use project_dir as logging_dir to avoid extra subdirectory creation
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.logging.output_dir,
            logging_dir=self.config.logging.output_dir,
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
            # mixed_precision=self.config.train.mixed_precision,
            mixed_precision="no",  # ← Critical
            log_with=self.config.logging.report_to,
            project_config=accelerator_project_config,
        )

        # Initialize tracker
        if self.config.logging.report_to == "wandb":
            # Create comprehensive config for wandb
            try:
                import wandb
                
                wandb_config = {
                    "learning_rate": float(self.config.optimizer.init_args.get("lr", 0.0001)),
                    "batch_size": int(self.config.data.batch_size),
                    "max_train_steps": int(self.config.train.max_train_steps),
                    "gradient_accumulation_steps": int(self.config.train.gradient_accumulation_steps),
                    "mixed_precision": str(self.config.train.mixed_precision),
                    "lora_r": int(self.config.model.lora.r),
                    "lora_alpha": int(self.config.model.lora.lora_alpha),
                    "model_name": str(self.config.model.pretrained_model_name_or_path),
                    "checkpointing_steps": int(self.config.train.checkpointing_steps),
                    "max_grad_norm": float(self.config.train.max_grad_norm),
                    "mask_loss": bool(self.config.loss.mask_loss),
                    "use_cache": bool(self.config.cache.use_cache),
                }
                
                # Prepare wandb init kwargs (don't include 'project' - accelerator handles it via project_name)
                init_kwargs = {}
                
                # Add optional wandb-specific fields
                if hasattr(self.config.logging, 'wandb_entity') and self.config.logging.wandb_entity:
                    init_kwargs["entity"] = self.config.logging.wandb_entity
                if hasattr(self.config.logging, 'wandb_tags') and self.config.logging.wandb_tags:
                    init_kwargs["tags"] = self.config.logging.wandb_tags
                if hasattr(self.config.logging, 'wandb_notes') and self.config.logging.wandb_notes:
                    init_kwargs["notes"] = self.config.logging.wandb_notes
                
                self.accelerator.init_trackers(
                    project_name=self.config.logging.tracker_project_name,
                    config=wandb_config,
                    init_kwargs={"wandb": init_kwargs}
                )
                
                logging.info("Initialized wandb tracker successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize wandb tracker: {e}")
                # Initialize without extra config if there's an error
                self.accelerator.init_trackers(self.config.logging.tracker_project_name)
        elif self.config.logging.report_to == "tensorboard":
            # Create a simple config dict with only basic types for TensorBoard
            try:
                simple_config = {
                    "learning_rate": float(self.config.optimizer.init_args.get("lr", 0.0001)),
                    "batch_size": int(self.config.data.batch_size),
                    "max_train_steps": int(self.config.train.max_train_steps),
                    "gradient_accumulation_steps": int(self.config.train.gradient_accumulation_steps),
                    "mixed_precision": str(self.config.train.mixed_precision),
                    "lora_r": int(self.config.model.lora.r),
                    "lora_alpha": int(self.config.model.lora.lora_alpha),
                    "model_name": str(self.config.model.pretrained_model_name_or_path),
                    "checkpointing_steps": int(self.config.train.checkpointing_steps),
                }
                self.accelerator.init_trackers("", config=simple_config)
            except Exception as e:
                logging.warning(f"Failed to initialize trackers with config: {e}")
                # Initialize without config if there's an error
                self.accelerator.init_trackers("")
        logging.info(f"Number of devices used in DDP training: {self.accelerator.num_processes}")

        # Set weight data type
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Create output directory
        if self.accelerator.is_main_process and self.config.logging.output_dir is not None:
            os.makedirs(self.config.logging.output_dir, exist_ok=True)

        logging.info(f"Mixed precision: {self.accelerator.mixed_precision}")

    def save_checkpoint(self, epoch, global_step, is_last=False):
        """Save checkpoint"""
        self.fps_logger.pause()
        if not is_last and (self.global_step % self.config.train.checkpointing_steps != 0):
            self.fps_logger.resume()
            return
        if self.accelerator.is_main_process:
            logging.info(f"Saving checkpoint to {self.config.logging.output_dir}")
            save_path = os.path.join(self.config.logging.output_dir, f"checkpoint-{epoch}-{global_step}")
            if is_last:
                save_path = os.path.join(self.config.logging.output_dir, f"checkpoint-last-{epoch}-{global_step}-last")
            os.makedirs(save_path, exist_ok=True)

            self.save_lora(save_path, adapter_name=self.adapter_name)

            state_info = {"global_step": global_step, "epoch": epoch, "is_last": is_last}
            if is_last:
                plug = self.accelerator.state.fsdp_plugin if hasattr(self.accelerator.state, "fsdp_plugin") else None
                if plug is None:
                    self.accelerator.save_state(save_path)  # when save in fsdp, will cause problem
                git_info = get_git_info()
                state_info.update(git_info)
            with open(os.path.join(save_path, "state.json"), "w") as f:
                json.dump(state_info, f)
        self.fps_logger.resume()

    def save_lora(self, save_folder, adapter_name=None):
        """Save LoRA weights"""
        if save_folder.endswith(".safetensors"):
            print(f"Warning: save_folder {save_folder} should a folder")
        if self.accelerator is not None:
            unwrapped_transformer = self.accelerator.unwrap_model(self.dit)
        else:
            unwrapped_transformer = self.dit
        if is_compiled_module(unwrapped_transformer):
            unwrapped_transformer = unwrapped_transformer._orig_mod
        adapter_name = self.adapter_name if adapter_name is None else adapter_name

        lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_transformer, adapter_name=adapter_name)
        )
        # Use FluxKontextPipeline's save method if available, otherwise use generic method
        self.pipeline_class.save_lora_weights(save_folder, lora_state_dict, safe_serialization=True)
        logging.info(f"Saved LoRA weights to {save_folder}")

    def save_train_config(self):
        import yaml

        d_json = self.config.model_dump(mode="json", exclude_none=True)
        train_yaml_file = os.path.join(self.config.logging.output_dir, "train_config.yaml")
        os.makedirs(self.config.logging.output_dir, exist_ok=True)
        with open(train_yaml_file, "w") as f:
            yaml.dump(d_json, f, default_flow_style=False, sort_keys=False)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        from diffusers.optimization import get_scheduler

        trainable_named_params = [(name, param) for name, param in self.dit.named_parameters() if param.requires_grad]
        lora_layers = [param for _, param in trainable_named_params]

        # Log how many parameters are in training and show one example
        if (getattr(self, "accelerator", None) is None) or self.accelerator.is_main_process:
            total_elements = sum(p.numel() for p in lora_layers)
            logging.info(f"Trainable parameters: {len(lora_layers)} tensors, total elements: {total_elements}")
            if len(trainable_named_params) > 0:
                example_name, example_param = trainable_named_params[0]
                logging.info(f"Example trainable param: {example_name}, shape={tuple(example_param.shape)}")
                logging.info(f"Example dtype: {example_param.dtype}")

        # Use optimizer parameters from configuration
        optimizer_config = self.config.optimizer.init_args
        class_path = self.config.optimizer.class_path
        module_name, class_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        logging.info(f"Using optimizer: {cls}, {class_path}")
        self.optimizer = cls(
            lora_layers,
            **optimizer_config,
        )

        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_scheduler.warmup_steps,
            num_training_steps=self.config.train.max_train_steps,
        )

    @classmethod
    def quantize_model(cls, model, device):
        model = quantize_model_to_fp8(
            model,
            engine="bnb",
            verbose=True,
            device=device,
        )
        model = model.to(device)
        return model

    @classmethod
    def add_lora_adapter(cls, transformer: torch.nn.Module, config, adapter_name: str):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=config.model.lora.r,
            lora_alpha=config.model.lora.lora_alpha,
            init_lora_weights=config.model.lora.init_lora_weights,
            target_modules=config.model.lora.target_modules,
        )
        logging.info(f"add_lora_adapter: {lora_config}, {adapter_name}")
        transformer.add_adapter(lora_config, adapter_name=adapter_name)
        transformer.set_adapter(adapter_name)

    @classmethod
    def load_pretrain_lora_model(
        cls,
        transformer: torch.nn.Module,
        config,
        adapter_name: str,
        stage="fit",
    ):
        """
        load the pretrained lora model. Support both local filepath and huggingface repo-id
        Examples of pretrained_weight:
            - TsienDragon/qwen-image-edit-character-composition
            - TsienDragon/qwen-image-edit-character-composition/model.safetensors
            - <local_path>/pytorch_lora_weights.safetensors
            - <local_path>/<filename>.safetensors
        >>>
        """
        pretrained_weight = getattr(config.model.lora, "pretrained_weight", None)
        if pretrained_weight:
            if not os.path.exists(pretrained_weight):
                # if the pretrained_weight is a repo-id, add
                # try as the huggingface repos LORA_FILE_BASE_NAME
                if pretrained_weight.endswith(".safetensors"):
                    repo_id = pretrained_weight.split("/")[:2]
                    filename = pretrained_weight.split("/")[-1]
                else:
                    repo_id = pretrained_weight
                    filename = LORA_FILE_BASE_NAME
                try:
                    pretrained_weight = download_lora(repo_id, filename)
                except Exception as e:
                    logging.warning(f"Failed to download lora from {pretrained_weight}: {e}")
                    pass


            sha256 = calculate_sha256_file(pretrained_weight)
            logging.info(f"sha256 for pretrained_weight: {sha256}")
            lora_type = classify_lora_weight(pretrained_weight)
            # DIFFUSERS can be loaded directly, otherwise, need to add lora first
            if lora_type != "PEFT":
                transformer.load_lora_adapter(pretrained_weight, adapter_name=adapter_name)
                logging.info(f"set_lora: {lora_type} Loaded lora from {pretrained_weight} for {adapter_name}")
            else:
                # add lora first
                # Configure model
                cls.add_lora_adapter(transformer, config, adapter_name)

                missing, unexpected = transformer.load_state_dict(
                    safetensors.torch.load_file(pretrained_weight),
                    strict=False,
                )
                if len(unexpected) > 0:
                    raise ValueError(f"Unexpected keys: {unexpected}")
                logging.info(f"set_lora: {lora_type} Loaded lora from {pretrained_weight} for {adapter_name}")
                logging.info(f"missing keys: {len(missing)}, {missing[0]}")
                # self.load_lora(self.config.model.lora.pretrained_weight)
            logging.info(f"set_lora: Loaded lora from {pretrained_weight}")

        elif stage == "fit":
            cls.add_lora_adapter(transformer, config, adapter_name)

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image from [0,1] to [-1,1]"""
        image = image.to(self.weight_dtype)
        return image * 2.0 - 1.0

    def prepare_predict_timesteps(self, num_inference_steps: int, image_seq_len: int) -> Tuple[torch.Tensor, int]:
        """Prepare timesteps for prediction/validation inference.
        
        Uses a separate inference scheduler to avoid corrupting training timesteps.
        """
        # Use separate inference scheduler (doesn't affect training)
        scheduler_to_use = self.inference_scheduler if self.inference_scheduler is not None else self.scheduler
        
        if scheduler_to_use is None:
            raise ValueError("No scheduler available for inference")
        
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            image_seq_len,
            scheduler_to_use.config.get("base_image_seq_len", 256),
            scheduler_to_use.config.get("max_image_seq_len", 4096),
            scheduler_to_use.config.get("base_shift", 0.5),
            scheduler_to_use.config.get("max_shift", 1.15),
        )
        device = next(self.dit.parameters()).device
        
        # This modifies inference_scheduler.timesteps (NOT training scheduler)
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler_to_use,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        
        logging.debug(f"Prepared {num_inference_steps} inference timesteps using separate scheduler")
        
        return timesteps, num_inference_steps

    @abstractmethod
    def load_model(self, **kwargs):
        """Load and initialize model components."""
        pass

    @abstractmethod
    def encode_prompt(self, *args, **kwargs):
        """Encode text prompts to embeddings. Qwen-Edit pass image and prompt, Flux-Kontext pass prompt"""
        pass

    @abstractmethod
    def prepare_latents(self, *args, **kwargs):
        """Prepare latents for fit & predict. Input usually be control images"""
        pass

    @abstractmethod
    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> Dict[str, torch.Tensor]:
        """Prepare embeddings for prediction. Call vae encoder and prompt encoder
        to get the embeddings. Used in fit & predict
        Update the embeddings keys in batch dict
        """
        return batch

    @abstractmethod
    def prepare_cached_embeddings(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Prepare cached embeddings for prediction. Loaded the cached embeddings from cache.
        Used in fit
        Update the embeddings keys in batch dict
        """
        return batch

    @abstractmethod
    def decode_vae_latent(self, latents: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Decode VAE latent vectors to RGB images. In range [0,1]"""
        pass

    @abstractmethod
    def sampling_from_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sampling from embeddings. Only handle the latent diffusion steps. Output the final latents. Need
        to decode the latents to images.
        """
        pass

    @abstractmethod
    def cache_step(self, data: dict):
        """Cache step"""
        pass

    @abstractmethod
    def setup_model_device_train_mode(self, stage="fit", cache=False):
        pass
