#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-VL-3B for image-to-prompt generation using lazy loading approach.

This script trains a vision-language model to generate image editing prompts
from control images, using a custom dataset with on-demand image loading.

Features:
    - Multi-GPU training support (DDP)
    - LoRA fine-tuning for memory efficiency
    - Semantic evaluation metrics (BERT Score, ROUGE, Semantic Similarity)
    - Automatic output directory management (resume/overwrite/version)
    - WandB logging for metrics tracking

Usage:
    # Step 1: Prepare dataset (validates images, creates JSONL)
    python script/prepare_qwen25vl_dataset.py --config configs/qwen25vl_prompt_gen.yaml

    # Step 2: Train on prepared dataset
    # Single GPU
    python script/train_qwen25vl_prompt_generation.py --config configs/qwen25vl_prompt_gen.yaml

    # Multi-GPU (4xA100)
    torchrun --nproc_per_node=4 script/train_qwen25vl_prompt_generation.py \
        --config configs/qwen25vl_prompt_gen.yaml

    # With custom run name (recommended for experiment tracking)
    python script/train_qwen25vl_prompt_generation.py \
        --config configs/qwen25vl_prompt_gen.yaml \
        --run-name experiment_v1_high_lr

    # Non-interactive mode (auto-create new version if output_dir exists)
    python script/train_qwen25vl_prompt_generation.py \
        --config configs/qwen25vl_prompt_gen.yaml \
        --non-interactive
    
    # Step 3: Inspect training runs and wandb correlation
    python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora

Optional Dependencies (for semantic metrics):
    pip install bert-score rouge-score sentence-transformers scikit-learn
    
Evaluation Metrics:
    During validation (every eval_steps), the following metrics are computed:
    - eval_loss: Standard cross-entropy loss
    - bert_score_f1: Semantic similarity using BERT embeddings (requires bert-score)
    - rouge1/rouge2/rougeL: Text overlap metrics (requires rouge-score)
    - semantic_similarity: Cosine similarity via sentence embeddings (requires sentence-transformers)
    - avg_pred_length, avg_ref_length, length_ratio: Text statistics
    
    All metrics are automatically logged to WandB for tracking.
"""

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import traceback
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
import safetensors.torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from peft import LoraConfig
from peft.utils.save_and_load import set_peft_model_state_dict

# Optional: Metrics libraries (install if not present)
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Enable better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_next_version_dir(base_dir: str) -> str:
    """
    Find the next available version directory.
    
    Args:
        base_dir: Base directory path (e.g., /path/to/model)
    
    Returns:
        Path with version suffix (e.g., /path/to/model-v2)
    """
    base_path = Path(base_dir)
    base_name = base_path.name
    parent_dir = base_path.parent
    
    # Find existing versions
    version = 2
    while True:
        new_path = parent_dir / f"{base_name}-v{version}"
        if not new_path.exists():
            return str(new_path)
        version += 1


def save_run_metadata(output_dir: str, run_name: str, config: Dict[str, Any], config_path: str) -> None:
    """
    Save training run metadata to output directory for traceability.
    
    This creates a metadata.json file linking:
    - The checkpoint directory
    - The wandb run name and URL
    - The config used
    - Timestamps
    
    Args:
        output_dir: Output directory path
        run_name: Run name used for wandb
        config: Training configuration dict
        config_path: Path to config file
    """
    import socket
    from datetime import datetime
    
    metadata = {
        "run_name": run_name,
        "output_dir": output_dir,
        "config_file": config_path,
        "created_at": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "config_summary": {
            "model": config.get("model", {}).get("model_name"),
            "lora_r": config.get("lora", {}).get("r"),
            "lora_alpha": config.get("lora", {}).get("lora_alpha"),
            "num_epochs": config.get("training", {}).get("num_train_epochs"),
            "learning_rate": config.get("training", {}).get("learning_rate"),
            "batch_size": config.get("training", {}).get("per_device_train_batch_size"),
        },
    }
    
    # Try to get wandb run info if available
    try:
        import wandb
        if wandb.run is not None:
            metadata["wandb"] = {
                "run_id": wandb.run.id,
                "run_name": wandb.run.name,
                "run_url": wandb.run.get_url(),
                "project": wandb.run.project,
                "entity": wandb.run.entity,
            }
            logger.info(f"✓ Wandb run info captured: {metadata['wandb']['run_url']}")
    except Exception as e:
        logger.warning(f"Could not capture wandb info: {e}")
    
    # Save metadata
    metadata_path = Path(output_dir) / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Run metadata saved to: {metadata_path}")


def generate_run_name(
    config_path: str,
    base_output_dir: str,
    cli_run_name: Optional[str],
    config_run_name: Optional[str],
) -> str:
    """
    Generate a run name that is consistent across distributed ranks.

    Args:
        config_path: Path to the YAML config file used for training.
        base_output_dir: Base directory where runs are stored.
        cli_run_name: Run name provided via CLI (if any).
        config_run_name: Run name provided in config (if any).

    Returns:
        A run name string shared across all processes.
    """
    provided = cli_run_name or config_run_name
    if provided:
        return provided

    from datetime import datetime

    auto_name = f"{Path(config_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size <= 1:
        return auto_name

    # Share auto-generated name with other ranks via a temporary file
    key = f"{Path(config_path).resolve()}::{base_output_dir}"
    path_hash = hashlib.md5(key.encode()).hexdigest()[:10]
    run_name_file = Path(f"/tmp/qwen_run_name_{path_hash}.json")

    start_wait = time.time()

    if rank == 0:
        # Remove stale file from previous runs (if any)
        if run_name_file.exists():
            try:
                run_name_file.unlink()
            except OSError:
                pass

        with run_name_file.open("w") as f:
            json.dump({"run_name": auto_name, "created_at": time.time()}, f)
        return auto_name

    # Non-zero ranks: wait for main rank to write the run name
    max_wait = 120.0  # seconds
    poll_interval = 0.25
    while (time.time() - start_wait) < max_wait:
        if run_name_file.exists():
            try:
                with run_name_file.open("r") as fp:
                    metadata = json.load(fp)
            except json.JSONDecodeError:
                time.sleep(poll_interval)
                continue

            created_at = metadata.get("created_at", 0.0)
            if created_at >= start_wait:
                run_name = metadata.get("run_name")
                if run_name:
                    return run_name
        time.sleep(poll_interval)

    raise RuntimeError("Timeout waiting for shared run name from main process")


def load_best_adapter_weights_if_available(trainer: Trainer) -> bool:
    """
    Load the best LoRA adapter weights into the current model if available.

    This is necessary because HuggingFace Trainer looks for `pytorch_model.bin`,
    but PEFT saves adapters as `adapter_model.(safetensors|bin)`.

    Args:
        trainer: Trainer instance after training finishes.

    Returns:
        True if best weights were loaded, False otherwise.
    """
    if not trainer.args.load_best_model_at_end:
        return False

    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    if not best_checkpoint:
        logger.warning("No best model checkpoint recorded; using final training step weights.")
        return False

    best_path = Path(best_checkpoint)
    candidate_files = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.safetensors",
        "pytorch_model.bin",
    ]

    for filename in candidate_files:
        weight_path = best_path / filename
        if not weight_path.exists():
            continue

        if trainer.is_world_process_zero():
            logger.info(f"Loading best adapter weights from {weight_path}")

        if weight_path.suffix == ".safetensors":
            state_dict = safetensors.torch.load_file(weight_path)
        else:
            state_dict = torch.load(weight_path, map_location="cpu")

        set_peft_model_state_dict(trainer.model, state_dict)
        return True

    logger.warning(
        f"Best checkpoint found at {best_checkpoint}, but no adapter weights were present. "
        "Final save will use the last training step weights."
    )
    return False


def handle_output_directory(output_dir: str, non_interactive: bool = False) -> tuple[str, bool]:
    """
    Handle output directory existence and user choice.
    Only prompts on the main process (rank 0) in distributed training.
    
    Args:
        output_dir: Desired output directory path
        non_interactive: If True, skip prompts and auto-create new version if exists
    
    Returns:
        Tuple of (final_output_dir, resume_from_checkpoint)
    """
    # Check if we're in distributed training by looking at environment variables
    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE before processes start
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    is_distributed = world_size > 1
    is_main_process = rank == 0
    
    if is_distributed and is_main_process:
        logger.info(f"[Main Process - Rank {rank}] Handling output directory (other {world_size-1} processes will wait)...")
    elif is_distributed:
        logger.info(f"[Rank {rank}] Waiting for main process to handle output directory...")
    
    # Only main process handles the logic
    if is_main_process:
        output_path = Path(output_dir)
        
        # If directory doesn't exist, create it and proceed
        if not output_path.exists():
            logger.info(f"Output directory does not exist. Will create: {output_dir}")
            output_path.mkdir(parents=True, exist_ok=True)
            result = (output_dir, False)
        else:
            # Directory exists - check if it has checkpoints
            has_checkpoints = any(
                (output_path / d).is_dir() 
                for d in os.listdir(output_path) 
                if d.startswith("checkpoint-")
            )
            
            # In non-interactive mode, auto-create new version
            if non_interactive:
                new_dir = get_next_version_dir(output_dir)
                logger.info(f"Output directory exists. Auto-creating new version: {new_dir}")
                Path(new_dir).mkdir(parents=True, exist_ok=True)
                result = (new_dir, False)
            else:
                result = _prompt_user_for_output_dir(output_dir, output_path, has_checkpoints)
        
        # In distributed mode, write decision to file for other processes
        if is_distributed:
            # Create a unique temporary file based on output directory
            path_hash = hashlib.md5(output_dir.encode()).hexdigest()[:8]
            decision_file = f"/tmp/qwen_training_decision_{path_hash}.json"
            with open(decision_file, "w") as f:
                json.dump({"output_dir": result[0], "resume": result[1]}, f)
            logger.info(f"[Rank 0] Decision saved to {decision_file}")
            return result
        else:
            return result
    else:
        # Non-main processes wait for main process decision
        path_hash = hashlib.md5(output_dir.encode()).hexdigest()[:8]
        decision_file = f"/tmp/qwen_training_decision_{path_hash}.json"
        
        # Poll for decision file (wait up to 60 seconds)
        max_wait = 60
        waited = 0
        while not os.path.exists(decision_file) and waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
        
        if not os.path.exists(decision_file):
            logger.error(f"[Rank {rank}] Timeout waiting for main process decision")
            raise RuntimeError("Timeout waiting for main process output directory decision")
        
        with open(decision_file, "r") as f:
            decision = json.load(f)
        logger.info(f"[Rank {rank}] Received decision from main process: {decision['output_dir']}")
        
        # Clean up decision file after all processes have read it
        # (last process to reach here will clean it up - best effort)
        try:
            if os.path.exists(decision_file):
                os.remove(decision_file)
        except:
            pass  # Ignore errors, another process might have deleted it
        
        return decision["output_dir"], decision["resume"]


def _prompt_user_for_output_dir(output_dir: str, output_path: Path, has_checkpoints: bool) -> tuple[str, bool]:
    """
    Prompt user for output directory handling (internal function).
    
    Args:
        output_dir: Original output directory
        output_path: Path object for output directory
        has_checkpoints: Whether checkpoints exist in the directory
    
    Returns:
        Tuple of (final_output_dir, resume_from_checkpoint)
    """
    # Interactive mode - prompt user
    logger.warning("=" * 80)
    logger.warning(f"OUTPUT DIRECTORY ALREADY EXISTS: {output_dir}")
    logger.warning("=" * 80)
    
    if has_checkpoints:
        logger.info("Found existing checkpoints in directory.")
        logger.info("\nOptions:")
        logger.info("  1. Resume from last checkpoint (continue training)")
        logger.info("  2. Overwrite (delete all and start fresh)")
        logger.info("  3. Create new version directory (e.g., model-v2)")
        logger.info("  4. Cancel")
    else:
        logger.info("Directory exists but no checkpoints found.")
        logger.info("\nOptions:")
        logger.info("  1. Overwrite (delete all and start fresh)")
        logger.info("  2. Create new version directory (e.g., model-v2)")
        logger.info("  3. Cancel")
    
    while True:
        try:
            if has_checkpoints:
                choice = input("\nEnter your choice (1-4): ").strip()
            else:
                choice = input("\nEnter your choice (1-3): ").strip()
            
            if has_checkpoints:
                if choice == "1":
                    logger.info(f"✓ Will resume training from last checkpoint in {output_dir}")
                    return output_dir, True
                elif choice == "2":
                    confirm = input(f"⚠️  This will DELETE all contents of {output_dir}. Are you sure? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        logger.info(f"Deleting {output_dir}...")
                        shutil.rmtree(output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        logger.info("✓ Directory cleared. Starting fresh.")
                        return output_dir, False
                    else:
                        logger.info("Cancelled. Please choose again.")
                        continue
                elif choice == "3":
                    new_dir = get_next_version_dir(output_dir)
                    logger.info(f"✓ Creating new version directory: {new_dir}")
                    Path(new_dir).mkdir(parents=True, exist_ok=True)
                    return new_dir, False
                elif choice == "4":
                    logger.info("Training cancelled by user.")
                    sys.exit(0)
                else:
                    logger.warning("Invalid choice. Please enter 1, 2, 3, or 4.")
            else:
                if choice == "1":
                    confirm = input(f"⚠️  This will DELETE all contents of {output_dir}. Are you sure? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        logger.info(f"Deleting {output_dir}...")
                        shutil.rmtree(output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        logger.info("✓ Directory cleared. Starting fresh.")
                        return output_dir, False
                    else:
                        logger.info("Cancelled. Please choose again.")
                        continue
                elif choice == "2":
                    new_dir = get_next_version_dir(output_dir)
                    logger.info(f"✓ Creating new version directory: {new_dir}")
                    Path(new_dir).mkdir(parents=True, exist_ok=True)
                    return new_dir, False
                elif choice == "3":
                    logger.info("Training cancelled by user.")
                    sys.exit(0)
                else:
                    logger.warning("Invalid choice. Please enter 1, 2, or 3.")
        except (KeyboardInterrupt, EOFError):
            logger.info("\nTraining cancelled by user.")
            sys.exit(0)


class LazySupervisedDataset(Dataset):
    """
    Lazy dataset that loads images on-demand.
    
    Loads metadata from JSONL on initialization, then loads PIL images
    on-the-fly in __getitem__ to avoid memory issues and Arrow serialization problems.
    
    Args:
        jsonl_path: Path to JSONL file with formatted samples
    """
    def __init__(self, jsonl_path: str):
        logger.info(f"Loading dataset from {jsonl_path}")
        
        # Load all metadata (paths + messages) - lightweight
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append({
                    "image_path": sample["image_path"],
                    "messages": sample["messages"],
                })
        
        logger.info(f"Loaded {len(self.samples)} samples metadata from {jsonl_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load image on-demand and return formatted sample."""
        sample = self.samples[idx]
        
        try:
            # Load PIL image on-demand
            image = Image.open(sample["image_path"]).convert("RGB")
            
            # Insert image into messages
            messages = copy.deepcopy(sample["messages"])
            for msg in messages:
                if msg["role"] == "user":
                    for item in msg["content"]:
                        if item["type"] == "image":
                            item["image"] = image
                            break
                    break
            
            return {
                "images": [image],  # Separate field for processor
                "messages": messages
            }
        except Exception as e:
            logger.error(f"ERROR loading image {sample['image_path']}: {e}")
            # Return sample without image (will be filtered or handled by collator)
            raise


class Qwen2VLDataCollator:
    """
    Custom data collator for Qwen2-VL that handles image processing and tokenization.
    
    Properly masks labels so only assistant response tokens are supervised:
    - System/user tokens: masked with -100
    - Image placeholder tokens (<|image_pad|>, <|vision_start|>, <|vision_end|>): masked with -100
    - Assistant response tokens: kept for supervision
    
    Args:
        processor: Qwen2VLProcessor for tokenization and image processing
    """
    def __init__(self, processor):
        self.processor = processor
        
        # Cache special token IDs for masking
        tokenizer = processor.tokenizer
        self.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        
        # Encode the assistant marker to find where supervision should start
        # Note: We include the newline that comes after "assistant"
        self.assistant_marker_ids = tokenizer.encode(
            "<|im_start|>assistant\n", 
            add_special_tokens=False
        )
    
    def __call__(self, features):
        """Process batch of features into model inputs."""
        # Extract images and messages from batch
        images_list = [f["images"] for f in features]
        messages_list = [f["messages"] for f in features]
        
        # Apply chat template to convert messages to text
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            for msg in messages_list
        ]
        
        # Process with Qwen processor (handles both text and images)
        inputs = self.processor(
            text=texts,
            images=images_list,
            padding=True,
            return_tensors="pt"
        )
        
        # Create labels with proper masking
        labels = inputs["input_ids"].clone()
        
        # Mask labels for each sample in the batch
        for i in range(len(labels)):
            labels[i] = self._mask_labels(labels[i])
        
        inputs["labels"] = labels
        
        return inputs
    
    def _mask_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Mask labels so only assistant response is supervised.
        
        Args:
            input_ids: Token IDs for a single sample
            
        Returns:
            Masked labels tensor (same shape as input_ids)
        """
        labels = input_ids.clone()
        
        # Step 1: Find where assistant response starts
        # Look for the pattern: <|im_start|>assistant\n
        assistant_start_idx = None
        marker_len = len(self.assistant_marker_ids)
        
        for i in range(len(input_ids) - marker_len):
            # Check if we found the assistant marker
            if all(input_ids[i + j] == self.assistant_marker_ids[j] for j in range(marker_len)):
                # Assistant content starts AFTER the marker
                assistant_start_idx = i + marker_len
                break
        
        # Step 2: Mask everything before assistant response
        if assistant_start_idx is not None:
            labels[:assistant_start_idx] = -100
        else:
            # If we couldn't find assistant marker, mask everything (safety fallback)
            logger.warning("Could not find assistant marker in sample, masking all labels")
            labels[:] = -100
            return labels
        
        # Step 3: Mask all image-related special tokens (even in assistant response, though rare)
        # This ensures <|image_pad|>, <|vision_start|>, <|vision_end|> are never supervised
        image_token_mask = (
            (input_ids == self.image_token_id) |
            (input_ids == self.vision_start_id) |
            (input_ids == self.vision_end_id)
        )
        labels[image_token_mask] = -100
        
        return labels


class MetricsComputer:
    """
    Computes semantic similarity metrics for text generation evaluation.
    """
    def __init__(self):
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ Loaded sentence transformer for semantic similarity")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.sentence_model = None
        
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            logger.info("✓ Initialized ROUGE scorer")
    
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute various metrics for generated text.
        
        Args:
            predictions: Generated texts
            references: Ground truth texts
        
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # BERT Score (semantic similarity)
        if BERT_SCORE_AVAILABLE and len(predictions) > 0:
            try:
                P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
                metrics["bert_score_precision"] = P.mean().item()
                metrics["bert_score_recall"] = R.mean().item()
                metrics["bert_score_f1"] = F1.mean().item()
            except Exception as e:
                logger.warning(f"BERT Score computation failed: {e}")
        
        # ROUGE scores
        if self.rouge_scorer and len(predictions) > 0:
            try:
                rouge_scores = {
                    'rouge1_f': [],
                    'rouge2_f': [],
                    'rougeL_f': []
                }
                for pred, ref in zip(predictions, references):
                    scores = self.rouge_scorer.score(ref, pred)
                    rouge_scores['rouge1_f'].append(scores['rouge1'].fmeasure)
                    rouge_scores['rouge2_f'].append(scores['rouge2'].fmeasure)
                    rouge_scores['rougeL_f'].append(scores['rougeL'].fmeasure)
                
                metrics["rouge1"] = np.mean(rouge_scores['rouge1_f'])
                metrics["rouge2"] = np.mean(rouge_scores['rouge2_f'])
                metrics["rougeL"] = np.mean(rouge_scores['rougeL_f'])
            except Exception as e:
                logger.warning(f"ROUGE computation failed: {e}")
        
        # Cosine similarity using sentence transformers
        if self.sentence_model and len(predictions) > 0:
            try:
                pred_embeddings = self.sentence_model.encode(predictions)
                ref_embeddings = self.sentence_model.encode(references)
                
                # Compute pairwise cosine similarity
                similarities = []
                for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                    sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                    similarities.append(sim)
                
                metrics["semantic_similarity"] = np.mean(similarities)
            except Exception as e:
                logger.warning(f"Semantic similarity computation failed: {e}")
        
        # Basic text statistics
        if len(predictions) > 0:
            avg_pred_len = np.mean([len(p.split()) for p in predictions])
            avg_ref_len = np.mean([len(r.split()) for r in references])
            metrics["avg_pred_length"] = avg_pred_len
            metrics["avg_ref_length"] = avg_ref_len
            metrics["length_ratio"] = avg_pred_len / max(avg_ref_len, 1)
        
        return metrics


class QwenTrainerWithGeneration(Trainer):
    """
    Custom Trainer that generates text during evaluation for computing semantic metrics.
    """
    def __init__(self, *args, metrics_computer=None, eval_generation_config=None, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_computer = metrics_computer
        self.eval_generation_config = eval_generation_config or {
            "max_new_tokens": 256,
            "do_sample": False,
            "temperature": 1.0,
        }
        # Store processor for generation (not passed to parent Trainer)
        self.processor = processor
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, *args, **kwargs):
        """
        Override evaluation loop to generate text and compute semantic metrics.
        """
        # First run standard evaluation to get loss
        output = super().evaluation_loop(
            dataloader, 
            description, 
            prediction_loss_only=prediction_loss_only,
            *args, 
            **kwargs
        )
        
        # Only compute generation metrics on main process and if we have metrics computer
        if self.is_world_process_zero() and self.metrics_computer is not None:
            try:
                logger.info("Generating predictions for semantic metrics...")
                predictions, references = self._generate_predictions(dataloader)
                
                if len(predictions) > 0:
                    # Compute semantic metrics
                    semantic_metrics = self.metrics_computer.compute_metrics(predictions, references)
                    
                    # Add to output metrics
                    if output.metrics is not None:
                        output.metrics.update(semantic_metrics)
                    
                    # Log sample predictions
                    if len(predictions) > 0:
                        logger.info("=" * 80)
                        logger.info("SAMPLE PREDICTIONS:")
                        for i in range(min(3, len(predictions))):
                            logger.info(f"\n[Sample {i+1}]")
                            logger.info(f"Reference: {references[i][:200]}...")
                            logger.info(f"Generated: {predictions[i][:200]}...")
                        logger.info("=" * 80)
            except Exception as e:
                logger.warning(f"Failed to compute generation metrics: {e}")
        
        return output
    
    def _generate_predictions(self, dataloader, max_samples=50):
        """
        Generate predictions for a subset of evaluation data.
        
        Args:
            dataloader: Evaluation dataloader
            max_samples: Maximum number of samples to generate (for speed)
        
        Returns:
            Tuple of (predictions, references)
        """
        model = self.model
        processor = self.processor  # Use our stored processor
        
        model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if idx * dataloader.batch_size >= max_samples:
                    break
                
                # Get images and create input for generation (without labels)
                # We need to reconstruct the messages from the dataset
                # This is a simplified approach - we'll extract the reference from input_ids
                
                # Move inputs to device
                pixel_values = batch.get("pixel_values")
                image_grid_thw = batch.get("image_grid_thw")
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                
                if pixel_values is not None:
                    pixel_values = pixel_values.to(model.device)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(model.device)
                
                # Extract reference text from labels
                labels = batch.get("labels", input_ids)
                for label_ids in labels:
                    # Decode reference (skip padding)
                    label_ids = label_ids[label_ids != -100]
                    ref_text = processor.decode(label_ids, skip_special_tokens=True)
                    # Extract only the assistant response
                    if "<|im_start|>assistant" in ref_text:
                        ref_text = ref_text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
                    references.append(ref_text)
                
                # Create generation inputs (prompt only, without answer)
                # For generation, we need to truncate input_ids to remove the assistant response
                generation_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                if pixel_values is not None:
                    generation_inputs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    generation_inputs["image_grid_thw"] = image_grid_thw
                
                # Generate
                generated_ids = model.generate(
                    **generation_inputs,
                    max_new_tokens=self.eval_generation_config["max_new_tokens"],
                    do_sample=self.eval_generation_config["do_sample"],
                    temperature=self.eval_generation_config.get("temperature", 1.0),
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
                
                # Decode generated text
                for gen_ids, input_len in zip(generated_ids, input_ids.shape[1]):
                    # Only decode the newly generated tokens
                    gen_text = processor.decode(gen_ids[input_len:], skip_special_tokens=True)
                    predictions.append(gen_text.strip())
        
        model.train()
        return predictions, references


def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for prompt generation")
        parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
        parser.add_argument("--debug", action="store_true", help="Debug mode with limited samples")
        parser.add_argument("--non-interactive", action="store_true", 
                          help="Non-interactive mode: auto-create new version if output_dir exists")
        parser.add_argument("--run-name", type=str, default=None,
                          help="Custom run name for this training session (used for wandb and output dir)")
        args = parser.parse_args()
        
        # Load configuration
        logger.info("=" * 80)
        logger.info("STARTING QWEN2.5-VL TRAINING")
        logger.info("=" * 80)
        config = load_config(args.config)
        logger.info(f"✓ Loaded configuration from {args.config}")
        
        # Extract config sections
        model_config = config.get("model", {})
        lora_config = config.get("lora", {})
        data_config = config.get("data", {})
        training_config = config.get("training", {})
        
        # Setup model name and paths
        model_name = model_config.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct")
        base_output_dir = training_config.get("output_dir", "workspace/qwen25vl-prompt-gen-lora")

        # Generate or use run name for consistent tracking across ranks
        run_name = generate_run_name(
            config_path=args.config,
            base_output_dir=base_output_dir,
            cli_run_name=args.run_name,
            config_run_name=training_config.get("run_name"),
        )
        logger.info(f"Run name: {run_name}")
        
        # Create output directory with run name for traceability
        # Structure: base_output_dir/run_name/
        output_dir = os.path.join(base_output_dir, run_name)
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Base output directory: {base_output_dir}")
        logger.info(f"Full output directory: {output_dir}")
        
        # Handle output directory (check if exists, prompt user, etc.)
        output_dir, resume_from_checkpoint = handle_output_directory(
            output_dir, 
            non_interactive=args.non_interactive
        )
        logger.info(f"Final output directory: {output_dir}")
        if resume_from_checkpoint:
            logger.info("Will resume from last checkpoint")
        
        # Load pre-formatted dataset from JSONL
        # Note: Run prepare_qwen25vl_dataset.py first to create these files
        prepared_data_dir = data_config.get("prepared_data_dir", "workspace/prepared_data")
        train_jsonl = Path(prepared_data_dir) / "train.jsonl"
        eval_jsonl = Path(prepared_data_dir) / "eval.jsonl"
        
        if not train_jsonl.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_jsonl}. "
                "Please run prepare_qwen25vl_dataset.py first to prepare the dataset."
            )
        
        if not eval_jsonl.exists():
            raise FileNotFoundError(
                f"Validation data not found at {eval_jsonl}. "
                "Please run prepare_qwen25vl_dataset.py first to prepare the dataset."
            )
        
        # Initialize lazy loading datasets (no caching, loads PIL images on-demand)
        logger.info("Initializing lazy loading datasets...")
        train_dataset = LazySupervisedDataset(str(train_jsonl))
        eval_dataset = LazySupervisedDataset(str(eval_jsonl))
        
        logger.info(f"✓ Datasets ready: {len(train_dataset)} training, {len(eval_dataset)} validation samples")
        logger.info("Images will be loaded on-demand during training (memory efficient)")
        
        # Setup quantization config if specified
        quantization = model_config.get("quantization", None)
        quantization_config = None
        
        if quantization == "4bit":
            logger.info("Using 4-bit quantization (QLoRA)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            logger.info("No quantization - loading full precision model")
        
        # Setup device map based on quantization
        # - With quantization: device_map="auto" (single GPU only due to BitsAndBytes limitation)
        # - Without quantization: device_map=None (let Accelerate distribute across GPUs)
        device_map = None if quantization_config is None else "auto"
        
        # Load model
        logger.info("=" * 80)
        logger.info(f"LOADING MODEL: {model_name}")
        logger.info("=" * 80)
        logger.info(f"  Quantization: {quantization or 'None (full precision)'}")
        logger.info(f"  Device map: {device_map or 'None (Accelerate will handle distribution)'}")
        logger.info(f"  Dtype: bfloat16")
        logger.info("Starting model load (this may take a few minutes)...")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=device_map,
        )
        logger.info("✓ Model loaded successfully")
        
        # Load processor
        logger.info("Loading processor...")
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
        logger.info("✓ Processor loaded successfully")
        
        # Apply LoRA to model
        from peft import get_peft_model
        
        peft_config = LoraConfig(
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("lora_alpha", 64),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            task_type="CAUSAL_LM",
        )
        
        logger.info(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
                    f"dropout={peft_config.lora_dropout}")
        logger.info(f"Target modules: {peft_config.target_modules}")
        
        # Wrap model with PEFT
        logger.info("Applying LoRA adapters to model...")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Enable input gradients for gradient checkpointing with PEFT
        # This is required when using gradient_checkpointing=True with LoRA
        if training_config.get("gradient_checkpointing", True):
            model.enable_input_require_grads()
            logger.info("✓ Enabled input gradients for gradient checkpointing")
        
        logger.info("✓ LoRA adapters applied")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,  # Consistent run name for wandb tracking
            num_train_epochs=training_config.get("num_train_epochs", 3),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 2),
            learning_rate=training_config.get("learning_rate", 2e-5),
            weight_decay=training_config.get("weight_decay", 0.01),
            warmup_ratio=training_config.get("warmup_ratio", 0.03),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            logging_steps=training_config.get("logging_steps", 10),
            save_steps=training_config.get("save_steps", 500),
            eval_steps=training_config.get("eval_steps", 500),
            save_total_limit=training_config.get("save_total_limit", 3),
            bf16=training_config.get("bf16", True),
            gradient_checkpointing=training_config.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Required for PEFT + gradient checkpointing
            dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
            remove_unused_columns=False,  # Required for vision-language models
            ddp_find_unused_parameters=True,  # Required for DDP + gradient checkpointing + LoRA
            report_to=training_config.get("report_to", "wandb"),
            push_to_hub=training_config.get("push_to_hub", False),
            hub_token=None,  # Transformers 5.0 uses hub_token instead of push_to_hub_token
            eval_strategy="steps",  # Changed from evaluation_strategy (deprecated in transformers 4.x)
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        logger.info("Training arguments configured")
        logger.info(f"  Epochs: {training_args.num_train_epochs}")
        logger.info(f"  Batch size per device: {training_args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {training_args.learning_rate}")
        logger.info(f"  Total steps: ~{len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
        
        # Initialize data collator
        logger.info("Creating data collator...")
        data_collator = Qwen2VLDataCollator(processor)
        logger.info("✓ Data collator created")
        
        # Validate masking on a sample batch
        logger.info("=" * 80)
        logger.info("VALIDATING LABEL MASKING")
        logger.info("=" * 80)
        try:
            # Get a small sample from train dataset
            sample_batch = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
            collated = data_collator(sample_batch)
            
            # Calculate masking statistics
            labels = collated["labels"]
            total_tokens = labels.numel()
            supervised_tokens = (labels != -100).sum().item()
            masked_tokens = (labels == -100).sum().item()
            valid_token_frac = supervised_tokens / total_tokens
            
            logger.info(f"Sample batch masking statistics:")
            logger.info(f"  Total tokens: {total_tokens}")
            logger.info(f"  Supervised tokens (labels != -100): {supervised_tokens}")
            logger.info(f"  Masked tokens (labels == -100): {masked_tokens}")
            logger.info(f"  Valid token fraction: {valid_token_frac:.3f}")
            
            # Verify masking is reasonable
            if valid_token_frac > 0.8:
                logger.warning("⚠️  WARNING: Valid token fraction is very high (>0.8)!")
                logger.warning("   This suggests labels may not be properly masked.")
                logger.warning("   Expected: ~0.3-0.5 for vision-language SFT")
            elif valid_token_frac < 0.01:
                logger.warning("⚠️  WARNING: Valid token fraction is very low (<0.01)!")
                logger.warning("   This suggests too many tokens are masked.")
            else:
                logger.info(f"✓ Valid token fraction looks reasonable for vision-language SFT")
                
        except Exception as e:
            logger.warning(f"Could not validate masking: {e}")
            logger.warning("Proceeding with training anyway...")
        
        # Initialize metrics computer for semantic evaluation
        logger.info("=" * 80)
        logger.info("INITIALIZING METRICS COMPUTER")
        logger.info("=" * 80)
        metrics_computer = MetricsComputer()
        
        # Log which metrics are available
        available_metrics = []
        if BERT_SCORE_AVAILABLE:
            available_metrics.append("BERT Score")
        if ROUGE_AVAILABLE:
            available_metrics.append("ROUGE")
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            available_metrics.append("Semantic Similarity")
        
        if available_metrics:
            logger.info(f"✓ Available semantic metrics: {', '.join(available_metrics)}")
        else:
            logger.warning("⚠️  No semantic metrics available. Install bert-score, rouge-score, and sentence-transformers for better evaluation.")
            logger.warning("   Run: pip install bert-score rouge-score sentence-transformers")
        
        # Generation config for evaluation
        eval_generation_config = training_config.get("eval_generation", {
            "max_new_tokens": 256,
            "do_sample": False,
        })
        
        # Initialize Trainer with generation capabilities
        logger.info("=" * 80)
        logger.info("INITIALIZING TRAINER")
        logger.info("=" * 80)
        
        trainer = QwenTrainerWithGeneration(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processor=processor,  # Pass processor separately (not as tokenizer)
            metrics_computer=metrics_computer,
            eval_generation_config=eval_generation_config,
        )
        logger.info("✓ Trainer initialized successfully")
        logger.info(f"✓ Evaluation will generate up to 50 samples every {training_args.eval_steps} steps")
        
        # Train
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        
        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            # Find last checkpoint
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                checkpoint_path = os.path.join(output_dir, last_checkpoint)
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                trainer.train(resume_from_checkpoint=checkpoint_path)
            else:
                logger.warning("No checkpoints found, starting from scratch")
                trainer.train()
        else:
            trainer.train()

        # Reload best adapter weights (if Trainer couldn't due to PEFT naming)
        best_loaded = load_best_adapter_weights_if_available(trainer)
        if best_loaded and trainer.is_world_process_zero():
            logger.info("✓ Best adapter weights restored prior to final save")

        # Save run metadata after training starts (wandb run is now initialized)
        logger.info("=" * 80)
        logger.info("SAVING RUN METADATA")
        logger.info("=" * 80)
        save_run_metadata(output_dir, run_name, config, args.config)
        
        # Save final model
        logger.info("=" * 80)
        logger.info("SAVING MODEL")
        logger.info("=" * 80)
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        
        # Save config for reproducibility
        config_save_path = Path(output_dir) / "training_config.yaml"
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
        logger.info(f"Saved training config to {config_save_path}")
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Model and adapter saved to: {output_dir}")
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("FATAL ERROR - TRAINING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
