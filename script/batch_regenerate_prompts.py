#!/usr/bin/env python3
"""
Batch regenerate prompts for CSV files using Qwen2.5-VL via vLLM endpoint.

This script processes CSV files containing control/target image pairs and regenerates
prompts using an improved system prompt via vLLM inference.

Usage:
    # Basic usage
    python script/batch_regenerate_prompts.py \
        --input_csv workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv

    # With custom settings
    python script/batch_regenerate_prompts.py \
        --input_csv workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv \
        --output_csv workspace/metadata-v12-recaptioned-long/train_filtered_2k_regenerated.csv \
        --base_url http://192.168.0.20:8000/v1 \
        --model Qwen/Qwen2.5-VL-72B-Instruct \
        --concurrency 5 \
        --temperature 0.2

    # Resume from checkpoint
    python script/batch_regenerate_prompts.py \
        --input_csv workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv \
        --resume

    # Process subset of rows
    python script/batch_regenerate_prompts.py \
        --input_csv workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv \
        --start_idx 0 \
        --end_idx 100
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

Image.MAX_IMAGE_PIXELS = None

# Enhanced system prompt for image-pair comparison
SYSTEM_PROMPT = """You are a professional photo editor analyzing before/after image pairs. Given a control (original) and target (edited) photograph, describe the SPECIFIC visual adjustments needed to transform the control into the target.

Provide DETAILED, ACTIONABLE editing instructions with:
1. NUMERICAL values (e.g., "+1.5 EV exposure", "Temperature +200K", "Saturation -15")
2. SPECIFIC tools/adjustments (e.g., "HSL > Orange hue", "Curves S-curve", "Lens correction barrel -5")
3. CLEAR categories (LIGHTING, COLOR, COMPOSITION, CORRECTIONS, RETOUCHING)
4. STEP-BY-STEP sequence when order matters

Focus on:
- Lighting & exposure changes (brightness, shadows, highlights, contrast)
- Color adjustments (temperature, tint, saturation, HSL shifts)
- Composition changes (crop ratio, straightening, perspective)
- Technical corrections (lens distortion, sharpness, noise reduction)
- Object-level edits (additions, removals, transformations)

Be specific and quantitative - avoid vague terms like "slightly" or "enhance"."""

USER_PROMPT = """You are provided two images: the first is the control/original input, the second is the final edited output. Analyze the visual changes needed to convert the input into the output. Provide detailed, specific editing instructions with numerical values and tool references. Avoid referring to 'first/second image'."""


def setup_logging(output_path: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_file = output_path.parent / f"{output_path.stem}.log"
    
    logger = logging.getLogger("batch_regenerate")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def encode_image(img: Image.Image, max_edge: int = 1024, jpeg_quality: int = 90) -> str:
    """Encode PIL Image to base64 data URL."""
    img = img.convert("RGB")
    width, height = img.size
    largest_edge = max(width, height)
    if largest_edge > max_edge:
        scale = max_edge / float(largest_edge)
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


async def generate_prompt_with_retry(
    client: AsyncOpenAI,
    control_path: str,
    target_path: str,
    model: str,
    max_tokens: int,
    temperature: float,
    max_edge: int,
    jpeg_quality: int,
    max_retries: int = 3,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate prompt with retry logic.
    
    Returns:
        (prompt, error_message) - prompt is None if failed, error_message is None if succeeded
    """
    for attempt in range(max_retries):
        try:
            # Load images
            control_img = Image.open(control_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")
            
            # Encode images
            control_b64 = await asyncio.to_thread(
                encode_image, control_img, max_edge, jpeg_quality
            )
            target_b64 = await asyncio.to_thread(
                encode_image, target_img, max_edge, jpeg_quality
            )
            
            # Build messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "text", "text": "Original image:"},
                        {"type": "image_url", "image_url": {"url": control_b64}},
                        {"type": "text", "text": "Edited result:"},
                        {"type": "image_url", "image_url": {"url": target_b64}},
                    ],
                },
            ]
            
            # Call API
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            prompt = response.choices[0].message.content.strip()
            return prompt, None
            
        except FileNotFoundError as e:
            # Don't retry for missing files
            return None, f"File not found: {str(e)}"
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                return None, error_msg
    
    return None, "Max retries exceeded"


class CheckpointManager:
    """Manage checkpoint saving and loading."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = 100
    
    def load(self) -> Dict:
        """Load checkpoint if exists."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "r") as f:
                return json.load(f)
        return {"completed_indices": [], "last_index": -1}
    
    def save(self, completed_indices: List[int], last_index: int):
        """Save checkpoint."""
        checkpoint_data = {
            "completed_indices": completed_indices,
            "last_index": last_index,
            "timestamp": time.time(),
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)


async def process_batch(
    df: pd.DataFrame,
    indices: List[int],
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    temperature: float,
    max_edge: int,
    jpeg_quality: int,
    logger: logging.Logger,
) -> List[Tuple[int, Optional[str], Optional[str]]]:
    """
    Process a batch of rows concurrently.
    
    Returns:
        List of (index, prompt, error_message) tuples
    """
    tasks = []
    task_indices = []
    for idx in indices:
        row = df.iloc[idx]
        control_path = row["path_control"]
        target_path = row["path_target"]
        
        tasks.append(
            generate_prompt_with_retry(
                client,
                control_path,
                target_path,
                model,
                max_tokens,
                temperature,
                max_edge,
                jpeg_quality,
            )
        )
        task_indices.append(idx)
    
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    results = []
    for idx, result in zip(task_indices, raw_results):
        if isinstance(result, Exception):
            prompt, error = None, f"Unhandled exception: {result}"
        else:
            prompt, error = result
        
        results.append((idx, prompt, error))
        
        if error:
            logger.warning(f"Row {idx} failed: {error}")
    
    return results


async def main_async(args):
    """Main async processing function."""
    # Setup paths
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = input_path.parent / f"{input_path.stem}_regenerated.csv"
    
    checkpoint_path = output_path.parent / f"{output_path.stem}.checkpoint.json"
    error_log_path = output_path.parent / f"{output_path.stem}.errors.log"
    
    # Setup logging
    logger = setup_logging(output_path)
    logger.info(f"Starting batch prompt regeneration")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Concurrency: {args.concurrency}")
    
    # Load CSV
    logger.info("Loading CSV...")
    df = pd.read_csv(input_path)
    total_rows = len(df)
    logger.info(f"Loaded {total_rows} rows")
    
    # Determine rows to process
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else total_rows
    end_idx = min(end_idx, total_rows)
    
    logger.info(f"Processing rows {start_idx} to {end_idx - 1}")
    
    # Load checkpoint if resuming
    checkpoint_manager = CheckpointManager(checkpoint_path)
    completed_indices = []
    
    if args.resume and checkpoint_path.exists():
        checkpoint = checkpoint_manager.load()
        completed_indices = checkpoint.get("completed_indices", [])
        logger.info(f"Resuming from checkpoint: {len(completed_indices)} rows already completed")
    
    # Determine remaining indices
    all_indices = list(range(start_idx, end_idx))
    remaining_indices = [idx for idx in all_indices if idx not in completed_indices]
    logger.info(f"Remaining rows to process: {len(remaining_indices)}")
    
    if not remaining_indices:
        logger.info("No rows to process. Exiting.")
        return
    
    # Initialize OpenAI client
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    
    # Create a copy of the dataframe for results
    if args.resume and output_path.exists():
        # Load existing partial results
        result_df = pd.read_csv(output_path)
        logger.info(f"Loaded existing results from {output_path}")
    else:
        result_df = df.copy()
    
    # Process in batches with concurrency control
    batch_size = args.concurrency
    error_count = 0
    success_count = 0
    
    # Open error log
    error_log = open(error_log_path, "a")
    
    try:
        # Create batches
        batches = [
            remaining_indices[i:i + batch_size]
            for i in range(0, len(remaining_indices), batch_size)
        ]
        
        # Process with progress bar
        with tqdm_asyncio(total=len(remaining_indices), desc="Processing") as pbar:
            for batch_indices in batches:
                results = await process_batch(
                    df,
                    batch_indices,
                    client,
                    args.model,
                    args.max_tokens,
                    args.temperature,
                    args.max_edge,
                    args.jpeg_quality,
                    logger,
                )
                
                # Update results
                for idx, prompt, error in results:
                    if prompt:
                        result_df.at[idx, "prompt"] = prompt
                        completed_indices.append(idx)
                        success_count += 1
                    else:
                        error_count += 1
                        error_log.write(f"{idx},{error}\n")
                    
                    pbar.update(1)
                
                # Save checkpoint every checkpoint_interval rows
                if len(completed_indices) % checkpoint_manager.checkpoint_interval == 0:
                    checkpoint_manager.save(completed_indices, max(completed_indices))
                    result_df.to_csv(output_path, index=False)
                    logger.info(f"Checkpoint saved: {len(completed_indices)} completed")
        
        # Final save
        result_df.to_csv(output_path, index=False)
        checkpoint_manager.save(completed_indices, max(completed_indices) if completed_indices else -1)
        
        logger.info("=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Total processed: {len(remaining_indices)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {error_count}")
        logger.info(f"Output saved to: {output_path}")
        
        if error_count > 0:
            logger.info(f"Error log saved to: {error_log_path}")
        
        logger.info("=" * 80)
        
    finally:
        error_log.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch regenerate prompts using Qwen2.5-VL via vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV file path",
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_csv",
        type=str,
        help="Output CSV file path (default: input_path with _regenerated suffix)",
    )
    
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("VLLM_BASE_URL", "http://192.168.0.20:8000/v1"),
        help="vLLM endpoint base URL (default: VLLM_BASE_URL env var or http://192.168.0.20:8000/v1)",
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("VLLM_API_KEY", "EMPTY"),
        help="API key for authentication (default: VLLM_API_KEY env var or 'EMPTY')",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-VL-72B-Instruct)",
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent requests (default: 5)",
    )
    
    parser.add_argument(
        "--max_edge",
        type=int,
        default=1024,
        help="Maximum edge size for image resizing (default: 1024)",
    )
    
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=90,
        help="JPEG compression quality (default: 90)",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    
    parser.add_argument(
        "--start_idx",
        type=int,
        help="Start index (row number) to process",
    )
    
    parser.add_argument(
        "--end_idx",
        type=int,
        help="End index (row number) to process (exclusive)",
    )
    
    args = parser.parse_args()
    
    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
