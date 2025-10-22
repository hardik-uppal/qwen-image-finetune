"""
ValidationSampler for monitoring training progress through image sampling.
Supports both FluxKontext and QwenImageEdit trainers.
"""
import logging
from typing import Dict, List, Union
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from accelerate import Accelerator
from src.data.config import SamplingConfig
from src.data.config import DataConfig
import math
import cv2

from src.utils.logger import log_images_auto
from src.utils.tools import sample_indices_per_rank
from src.data.dataset import ImageDataset

logger = logging.getLogger(__name__)


class ValidationSampler:
    """Universal validation sampler that works with any model architecture"""

    def __init__(
        self,
        config: SamplingConfig,
        accelerator: Accelerator,
        weight_dtype: torch.dtype = torch.bfloat16,
        data_config: DataConfig | None = None,
    ):
        self.config = config
        self.data_config = data_config
        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.validation_dataset = None

        # Cached embeddings storage
        self.cached_embeddings = []  # List of cached embedding dictionaries
        self.embeddings_cached = False

        # Internal configuration with sensible defaults
        self._internal_config = {
            "max_image_size": 512,
            "log_prefix": "validation",
            "move_vae_to_cpu_after": True,
            "skip_on_error": True,
            "samples_per_process": 1,
        }
        self.get_accelerator_attribute()
        self.setup_validation_dataset()

    def get_accelerator_attribute(self):
        self.local_rank = self.accelerator.local_process_index  # 本机内 rank
        self.world_size = self.accelerator.num_processes  # 总进程数（= 全局并行数）

    def setup_validation_dataset(self, train_dataset=None):
        """Setup validation dataset from config or training dataset"""
        # only return control and prompt
        if self.config.validation_data is None:
            # Use subset of training dataset
            self._create_subset_from_training(train_dataset)
        elif isinstance(self.config.validation_data, str):
            # Load from dataset path
            self._load_dataset_from_path(self.config.validation_data)
        elif isinstance(self.config.validation_data, list):
            # Use control-prompt pairs
            self._create_dataset_from_pairs(self.config.validation_data)
        else:
            raise ValueError(
                f"Unsupported validation_data format: {type(self.config.validation_data)}"
            )

    def _repeat_datasets(self, train_dataset: Union[Dataset, List[Dict]]):
        dataset_size = len(train_dataset)
        if dataset_size < self.world_size * self.config.num_samples:
            # repeat the dataset to match the size use concat dataset
            repeat_num = math.ceil(
                self.world_size * self.config.num_samples / dataset_size
            )
            if isinstance(train_dataset, List):
                train_dataset = train_dataset * repeat_num
            else:
                train_dataset = ConcatDataset([train_dataset] * repeat_num)
        return train_dataset

    def _create_subset_from_training(self, train_dataset: Union[Dataset, List[Dict]]):
        """Create validation subset from training dataset"""
        if train_dataset is None:
            logger.warning(
                "No training dataset provided for validation subset creation"
            )
            return []
        assert (
            self.data_config is not None
        ), "data_config is required for using training dataset"
        train_dataset = self._repeat_datasets(train_dataset)

        self.selected_indices = sample_indices_per_rank(
            self.accelerator,
            len(train_dataset),
            self.config.num_samples,
            seed=self.config.seed,
            replacement=False,
            global_shuffle=False,
        )

        # Use first few samples as validation set
        self.selected_datasets = []  # List dict
        for i in self.selected_indices:
            data_item = train_dataset[i]
            new_data = {}
            if "prompt" in data_item:
                new_data["prompt"] = data_item["prompt"]
            if (
                "control" in data_item
            ):  # suppose control is a Union[list[tensor], tensor]
                new_data["control"] = data_item["control"]
            self.selected_datasets.append(data_item)

    def _load_dataset_from_path(self, dataset_path: str):
        """Load validation dataset from file path
        
        For validation visualization, we want to load raw images without heavy preprocessing
        to show actual input/output quality, not cropped training data.
        """
        assert (
            self.data_config is not None
        ), "data_config is required to instantiated dataset class"
        
        # Create a copy of init_args with updated dataset_path and minimal preprocessing
        init_args_dict = self.data_config.init_args.model_dump()
        init_args_dict["dataset_path"] = [dataset_path]  # dataset_path expects a list
        
        # For validation visualization, disable aggressive cropping
        # Use the actual image sizes or resize to reasonable size while maintaining aspect ratio
        if "processor" in init_args_dict and init_args_dict["processor"]:
            processor_dict = init_args_dict["processor"]["init_args"]
            # Change from center_crop to resize to preserve full image
            processor_dict["process_type"] = "resize"  # Don't crop for validation visualization
            # Keep target size reasonable but allow full image
            # processor_dict["target_size"] = [768, 768]  # Larger size for better visualization
        
        # Reconstruct the init args object
        from src.data.config import DatasetInitArgs
        updated_init_args = DatasetInitArgs(**init_args_dict)
        
        dataset = ImageDataset(updated_init_args)
        return self._create_subset_from_training(dataset)

    def _create_dataset_from_pairs(self, pairs: List[Dict]):
        """Create validation dataset from control-prompt pairs"""
        validation_samples = []  # List dict

        for pair in pairs:
            if "control" in pair and "prompt" in pair:
                prompt = pair["prompt"]
                with open(pair["prompt"], "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
                control = pair["control"]
                control = cv2.imread(control)
                control = cv2.cvtColor(control, cv2.COLOR_BGR2RGB)
                control = control.transpose(2, 0, 1)
                validation_samples.append(
                    {
                        "control": control,
                        "prompt": prompt,
                    }
                )
        return self._create_subset_from_training(validation_samples)

    def cache_embeddings(self, trainer):
        """Cache embeddings for all validation samples using trainer's methods"""
        self.cached_embeddings = []
        for idx, sample in enumerate(self.selected_datasets):
            # Get VAE and text encodings separately
            vae_encodings = trainer.encode_vae_image_for_validation(sample["control"])
            text_encodings = trainer.encode_prompt_for_validation(sample["prompt"], sample["control"])
            
            # Merge into a single dict for sampling_from_embeddings
            cached_sample = {
                "sample_idx": idx,
                **vae_encodings,  # Unpacks control_latents, height, width, etc.
                **text_encodings,  # Unpacks prompt_embeds, prompt_embeds_mask
            }
            self.cached_embeddings.append(cached_sample)
        logging.info(
            f"rank [{self.local_rank}] Successfully cached embeddings for "
            f"{len(self.cached_embeddings)} samples"
        )
        self.embeddings_cached = True

    def should_run_validation(self, global_step: int) -> bool:
        """Check if validation should run at current step"""
        if self.config.validation_steps <= 0:
            return False
        return global_step % self.config.validation_steps == 0

    def run_validation_loop(self, global_step: int, trainer):
        """Main validation loop using cached embeddings and trainer's methods"""
        if not self.embeddings_cached:
            self.accelerator.print(
                "Warning: Embeddings not cached, skipping validation"
            )
            return

        logger.info(f"Starting validation at step {global_step} with {len(self.cached_embeddings)} samples")
        
        try:
            # Sample from cached embeddings
            for idx, (data, cache_embedding) in enumerate(zip(
                self.selected_datasets, self.cached_embeddings
            )):
                prompt = data["prompt"]
                control = data["control"]  # Can be np.ndarray or Tensor
                
                # Handle both numpy arrays and tensors
                if isinstance(control, torch.Tensor):
                    if control.dim() == 3:  # [C,H,W]
                        control = control.unsqueeze(0)  # Add batch dim -> [1,C,H,W]
                else:
                    control = torch.from_numpy(control).unsqueeze(0)
                
                # Normalize to [-1, 1] range for log_images_auto
                control = control.float()
                
                # Detect range and normalize appropriately
                min_val, max_val = control.min(), control.max()
                logger.info(f"Control image {idx+1} shape: {control.shape}, range: [{min_val:.3f}, {max_val:.3f}]")
                
                if max_val > 10.0:  # Definitely in [0, 255] range
                    control = (control / 255.0) * 2 - 1  # [0, 255] -> [-1, 1]
                    logger.info(f"Normalized from [0, 255] range")
                elif min_val >= 0 and max_val <= 1.0:  # Already in [0, 1] range
                    control = control * 2 - 1  # [0, 1] -> [-1, 1]
                    logger.info(f"Normalized from [0, 1] range")
                else:
                    logger.info(f"Keeping as-is (assumed [-1, 1] range)")
                
                logger.info(f"Logging control image {idx+1}/{len(self.cached_embeddings)}")
                log_images_auto(
                    self.accelerator,
                    f"validation/control_{self.local_rank}_sample{idx}",
                    control,
                    global_step,
                    caption=prompt,
                )
                
                # Log target/ground truth image if available
                if "image" in data:
                    target = data["image"]
                    # Handle tensor/numpy conversion
                    if isinstance(target, torch.Tensor):
                        if target.dim() == 3:
                            target = target.unsqueeze(0)
                    else:
                        target = torch.from_numpy(target).unsqueeze(0)
                    
                    # Normalize target image
                    target = target.float()
                    target_min, target_max = target.min(), target.max()
                    logger.info(f"Target image {idx+1} shape: {target.shape}, range: [{target_min:.3f}, {target_max:.3f}]")
                    
                    if target_max > 10.0:
                        target = (target / 255.0) * 2 - 1
                    elif target_min >= 0 and target_max <= 1.0:
                        target = target * 2 - 1
                    
                    logger.info(f"Logging target/ground truth image {idx+1}/{len(self.cached_embeddings)}")
                    log_images_auto(
                        self.accelerator,
                        f"validation/target_{self.local_rank}_sample{idx}",
                        target,
                        global_step,
                        caption=f"GT: {prompt}",
                    )
                
                # Generate sample using cached embeddings and trainer's model
                logger.info(f"Generating image {idx+1}/{len(self.cached_embeddings)}")
                generated_latents = trainer.sampling_from_embeddings(cache_embedding)
                
                # Decode latents to images
                logger.info(f"Decoding latents to image {idx+1}/{len(self.cached_embeddings)}")
                height = cache_embedding.get("height", 832)
                width = cache_embedding.get("width", 576)
                generated_image = trainer.decode_vae_latent(generated_latents, height, width)
                
                logger.info(f"Generated image shape: {generated_image.shape}, range: [{generated_image.min():.3f}, {generated_image.max():.3f}]")
                logger.info(f"Logging generated image {idx+1}/{len(self.cached_embeddings)}")
                log_images_auto(
                    self.accelerator,
                    f"validation/generated_{self.local_rank}_sample{idx}",
                    generated_image,
                    global_step,
                    caption=f"Pred: {prompt}",
                )
            
            logger.info(f"Validation complete at step {global_step}")

        except Exception as e:
            logger.error(f"Validation sampling failed at step {global_step}: {e}", exc_info=True)
            if not self._internal_config["skip_on_error"]:
                raise
