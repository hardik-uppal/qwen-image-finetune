#!/usr/bin/env python3
"""
Prepare and validate dataset for Qwen2.5-VL prompt generation training.

This script:
1. Loads the source CSV with image paths and prompts
2. Validates that all image files exist and can be opened
3. Formats data into chat template structure
4. Saves to JSONL format ready for training

Usage:
    python script/prepare_qwen25vl_dataset.py --config configs/qwen25vl_prompt_gen.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

# Default prompts
DEFAULT_SYSTEM_PROMPT = """You are an expert photo editor. Analyze the provided photograph and describe the visual adjustments that could be made to enhance or transform it. Focus on actionable editing instructions including lighting adjustments, color corrections, object additions/removals, perspective corrections, lens distortion fixes, and compositional improvements."""

DEFAULT_USER_PROMPT = "Describe the edits needed for this image."


def validate_and_format_row(
    row: pd.Series,
    system_prompt: str,
    user_prompt: str,
) -> Optional[Dict[str, Any]]:
    """
    Validate image exists and format row into chat template.
    
    Args:
        row: Pandas Series with 'path_control' and 'prompt' columns
        system_prompt: System instruction for the model
        user_prompt: User question to ask about the image
        
    Returns:
        Dict with formatted data or None if validation fails
    """
    try:
        # Get image path
        image_path = row["path_control"]
        
        # Validate file exists
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return None
        
        # Try to open image to validate it's readable
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify it's a valid image
        except Exception as e:
            logger.warning(f"Cannot open image {image_path}: {e}")
            return None
        
        # Format in chat template (store path, not PIL Image)
        return {
            "image_path": str(image_path),
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # Placeholder - will load during training
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
        logger.warning(f"Error processing row: {e}")
        return None


def prepare_dataset(
    csv_path: str,
    output_dir: str,
    train_split: float = 0.9,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
    max_samples: Optional[int] = None,
) -> tuple[str, str]:
    """
    Prepare and validate dataset, save to JSONL.
    
    Args:
        csv_path: Path to source CSV file
        output_dir: Directory to save prepared datasets
        train_split: Fraction for training (rest for validation)
        system_prompt: System instruction
        user_prompt: User question template
        max_samples: Optional limit on samples (for debugging)
        
    Returns:
        Tuple of (train_jsonl_path, eval_jsonl_path)
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
    
    # Validate and format all samples
    logger.info("Validating and formatting samples...")
    formatted_samples = []
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        formatted = validate_and_format_row(row, system_prompt, user_prompt)
        if formatted is not None:
            formatted_samples.append(formatted)
        else:
            skipped += 1
    
    logger.info(f"Successfully formatted {len(formatted_samples)} samples")
    logger.info(f"Skipped {skipped} samples due to validation errors")
    
    if len(formatted_samples) == 0:
        raise ValueError("No valid samples found! Check your image paths.")
    
    # Split into train/val
    split_idx = int(len(formatted_samples) * train_split)
    train_samples = formatted_samples[:split_idx]
    eval_samples = formatted_samples[split_idx:]
    
    logger.info(f"Split: {len(train_samples)} train, {len(eval_samples)} validation")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to JSONL
    train_jsonl = output_path / "train.jsonl"
    eval_jsonl = output_path / "eval.jsonl"
    
    logger.info(f"Saving training data to {train_jsonl}")
    with open(train_jsonl, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    
    logger.info(f"Saving validation data to {eval_jsonl}")
    with open(eval_jsonl, "w") as f:
        for sample in eval_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Save metadata
    metadata = {
        "source_csv": csv_path,
        "total_samples": len(df),
        "valid_samples": len(formatted_samples),
        "skipped_samples": skipped,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "train_split": train_split,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    
    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Also save a simple CSV with just valid paths for reference
    valid_paths_csv = output_path / "valid_image_paths.csv"
    valid_df = pd.DataFrame([
        {"image_path": s["image_path"], "prompt": s["messages"][2]["content"][0]["text"]}
        for s in formatted_samples
    ])
    valid_df.to_csv(valid_paths_csv, index=False)
    logger.info(f"Saved valid paths CSV to {valid_paths_csv}")
    
    logger.info("Dataset preparation complete!")
    logger.info(f"  Train: {train_jsonl}")
    logger.info(f"  Eval:  {eval_jsonl}")
    
    return str(train_jsonl), str(eval_jsonl)


def main():
    parser = argparse.ArgumentParser(description="Prepare Qwen2.5-VL dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--debug", action="store_true", help="Debug mode with limited samples")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (defaults to data config or workspace/prepared_data)",
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {args.config}")
    
    # Extract config
    data_config = config.get("data", {})
    csv_path = data_config.get("train_csv")
    
    if not csv_path:
        raise ValueError("Config must specify data.train_csv")
    
    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = data_config.get("prepared_data_dir", "workspace/prepared_data")
    
    # Prepare dataset
    max_samples = 100 if args.debug else None
    prepare_dataset(
        csv_path=csv_path,
        output_dir=output_dir,
        train_split=data_config.get("train_split", 0.9),
        system_prompt=data_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        user_prompt=data_config.get("user_prompt", DEFAULT_USER_PROMPT),
        max_samples=max_samples,
    )


if __name__ == "__main__":
    main()

