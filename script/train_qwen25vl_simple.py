#!/usr/bin/env python3
"""
Simple Qwen2.5-VL training script that bypasses accelerate issues.
Uses torch.distributed directly for multi-GPU training.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from qwen_vl_utils import process_vision_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

# Default system prompt for image editing instruction generation
DEFAULT_SYSTEM_PROMPT = """You are an expert photo editor. Analyze the provided photograph and describe the visual adjustments that could be made to enhance or transform it. Focus on actionable editing instructions including lighting adjustments, color corrections, object additions/removals, perspective corrections, lens distortion fixes, and compositional improvements."""

DEFAULT_USER_PROMPT = "Describe the edits needed for this image."


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def format_data(row: pd.Series, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Format a CSV row into the TRL chat template format.
    
    Args:
        row: Pandas Series with 'path_control' and 'prompt' columns
        system_prompt: System instruction for the model
        user_prompt: User question to ask about the image
        
    Returns:
        Dict with 'images' and 'messages' keys formatted for TRL SFTTrainer
    """
    try:
        # Load image
        image_path = row["path_control"]
        image = Image.open(image_path).convert("RGB")
        
        # Format in chat template
        return {
            "images": [image],
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": row["prompt"]}],
                },
            ],
        }
    except Exception as e:
        logger.warning(f"Error formatting row with image {row.get('path_control', 'unknown')}: {e}")
        return None


def load_and_format_dataset(
    csv_path: str,
    train_split: float = 0.9,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
    max_samples: Optional[int] = None,
) -> tuple[List[Dict], List[Dict]]:
    """
    Load CSV dataset and format for TRL training.
    
    Args:
        csv_path: Path to CSV file with 'path_control' and 'prompt' columns
        train_split: Fraction of data to use for training (rest for validation)
        system_prompt: System instruction for the model
        user_prompt: User question template
        max_samples: Optional limit on number of samples to load (for debugging)
        
    Returns:
        Tuple of (train_dataset, eval_dataset) as lists of formatted dicts
    """
    logger.info(f"Loading dataset from {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} total samples from CSV")
    
    # Check required columns
    required_cols = ["path_control", "prompt"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Filter out rows with missing values
    df = df.dropna(subset=required_cols)
    logger.info(f"After filtering NaN values: {len(df)} samples")
    
    # Limit samples if specified (for debugging)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        logger.info(f"Limited to {max_samples} samples for debugging")
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train/val
    split_idx = int(len(df) * train_split)
    train_df = df[:split_idx]
    eval_df = df[split_idx:]
    
    logger.info(f"Split dataset: {len(train_df)} train, {len(eval_df)} validation")
    
    # Format datasets
    logger.info("Formatting training data...")
    train_dataset = []
    for idx, row in train_df.iterrows():
        formatted = format_data(row, system_prompt, user_prompt)
        if formatted is not None:
            train_dataset.append(formatted)
    
    logger.info("Formatting validation data...")
    eval_dataset = []
    for idx, row in eval_df.iterrows():
        formatted = format_data(row, system_prompt, user_prompt)
        if formatted is not None:
            eval_dataset.append(formatted)
    
    logger.info(f"Successfully formatted {len(train_dataset)} train and {len(eval_dataset)} eval samples")
    
    return train_dataset, eval_dataset


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for prompt generation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--debug", action="store_true", help="Debug mode with limited samples")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Extract config sections
    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    
    # Setup model name and paths
    model_name = model_config.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct")
    output_dir = training_config.get("output_dir", "workspace/qwen25vl-prompt-gen-lora")
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and format dataset
    max_samples = 100 if args.debug else None
    train_dataset, eval_dataset = load_and_format_dataset(
        csv_path=data_config.get("train_csv"),
        train_split=data_config.get("train_split", 0.9),
        system_prompt=data_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        user_prompt=data_config.get("user_prompt", DEFAULT_USER_PROMPT),
        max_samples=max_samples,
    )
    
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
    
    # Load model
    logger.info(f"Loading model {model_name}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    # Load processor
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    
    # Setup LoRA config
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
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        remove_unused_columns=False,  # Required for vision-language models
        report_to=training_config.get("report_to", "wandb"),
        push_to_hub=training_config.get("push_to_hub", False),
        evaluation_strategy="steps",
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
    
    # Initialize SFTTrainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    # Save config for reproducibility
    config_save_path = Path(output_dir) / "training_config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Saved training config to {config_save_path}")
    
    logger.info("Training complete!")
    logger.info(f"Model and adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()