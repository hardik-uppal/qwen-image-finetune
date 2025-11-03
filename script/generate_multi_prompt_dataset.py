#!/usr/bin/env python3
"""
Generate multiple prompt variations per control-target image pair for DPO and SFT training.

This script processes CSV files and generates N variations of prompts for EACH control variant
(path_control, path_control_1, path_control_2, etc.) paired with the target using Qwen2.5-VL 
via vLLM endpoint. It creates massively expanded datasets suitable for:
- DPO (Direct Preference Optimization): with chosen/rejected pairs per control variant
- SFT (Supervised Fine-Tuning): with multiple high-quality variations per control variant

For a CSV with 1000 rows, each having 6 control images, and 3 variations per control:
    Input: 1000 rows
    Output SFT: 18,000 training samples (1000 × 6 controls × 3 variations)
    Output DPO: ~36,000+ pairs

Usage:
    # Basic usage - generate 3 variations per control variant
    python script/generate_multi_prompt_dataset.py \
        --input_csv workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv \
        --num_variations 3

    # Custom settings for DPO with 5 variations
    python script/generate_multi_prompt_dataset.py \
        --input_csv workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv \
        --output_dir /skynas/01/hardik/qwen-dataset/multi-prompt-v1 \
        --num_variations 5 \
        --base_url http://192.168.0.20:8000/v1 \
        --model Qwen/Qwen2.5-VL-72B-Instruct \
        --concurrency 3

    # Resume from checkpoint
    python script/generate_multi_prompt_dataset.py \
        --input_csv workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv \
        --resume
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

# System prompts for variations
SYSTEM_PROMPT_DETAILED = """You are a professional photo editor analyzing before/after image pairs. Given a control (original) and target (edited) photograph, describe the SPECIFIC visual adjustments needed to transform the control into the target.

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

SYSTEM_PROMPT_CONCISE = """You are a professional photo editor. Analyze the control (original) and target (edited) images, then provide clear, actionable editing instructions to transform the control into the target.

Focus on the most important changes:
- Lighting adjustments (exposure, contrast, shadows, highlights)
- Color corrections (temperature, tint, saturation, vibrance)
- Composition changes (crop, perspective, straightening)
- Technical fixes (sharpness, noise, lens correction)
- Object edits (additions, removals, transformations)

Be specific and use measurable values when possible."""

SYSTEM_PROMPT_TECHNICAL = """You are a technical photo editing specialist. Analyze both images and provide precise, technical editing instructions with numerical parameters.

Structure your response:
1. EXPOSURE: EV adjustments, shadows, highlights, whites, blacks
2. COLOR: Temperature (K), tint, vibrance, saturation, HSL shifts
3. DETAIL: Sharpness, clarity, texture, noise reduction
4. LENS: Distortion correction, vignette, chromatic aberration
5. COMPOSITION: Crop ratio, rotation, perspective adjustments
6. RETOUCHING: Object removals, additions, local adjustments

Use specific values and tool references."""

SYSTEM_PROMPTS = [
    SYSTEM_PROMPT_DETAILED,
    SYSTEM_PROMPT_CONCISE,
    SYSTEM_PROMPT_TECHNICAL,
]

USER_PROMPT = """You are provided two images: the first is the control/original input, the second is the final edited output. Analyze the visual changes needed to convert the input into the output. Provide detailed, specific editing instructions with numerical values and tool references. Avoid referring to 'first/second image'."""


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_file = output_dir / "generation.log"
    
    logger = logging.getLogger("multi_prompt_gen")
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


async def generate_single_prompt(
    client: AsyncOpenAI,
    control_b64: str,
    target_b64: str,
    model: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    max_retries: int = 3,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a single prompt variation.
    
    Returns:
        (prompt, error_message) - prompt is None if failed, error_message is None if succeeded
    """
    for attempt in range(max_retries):
        try:
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
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
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            else:
                return None, error_msg
    
    return None, "Max retries exceeded"


async def generate_prompt_variations(
    client: AsyncOpenAI,
    control_path: str,
    target_path: str,
    model: str,
    max_tokens: int,
    num_variations: int,
    max_edge: int,
    jpeg_quality: int,
    temperature_range: Tuple[float, float],
) -> Tuple[List[str], Optional[str]]:
    """
    Generate multiple prompt variations for a single control-target pair.
    
    Returns:
        (prompts_list, error_message) - prompts_list is empty if failed
    """
    try:
        # Load and encode images once
        control_img = Image.open(control_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        
        control_b64 = await asyncio.to_thread(
            encode_image, control_img, max_edge, jpeg_quality
        )
        target_b64 = await asyncio.to_thread(
            encode_image, target_img, max_edge, jpeg_quality
        )
        
    except FileNotFoundError as e:
        return [], f"File not found: {str(e)}"
    except Exception as e:
        return [], f"Image loading error: {str(e)}"
    
    # Generate variations with different temperatures and system prompts
    tasks = []
    temp_min, temp_max = temperature_range
    
    for i in range(num_variations):
        # Vary temperature across range
        if num_variations == 1:
            temperature = temp_min
        else:
            temperature = temp_min + (temp_max - temp_min) * i / (num_variations - 1)
        
        # Cycle through system prompts
        system_prompt = SYSTEM_PROMPTS[i % len(SYSTEM_PROMPTS)]
        
        tasks.append(
            generate_single_prompt(
                client,
                control_b64,
                target_b64,
                model,
                max_tokens,
                temperature,
                system_prompt,
            )
        )
    
    # Execute all variations concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    prompts = []
    errors = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append(f"Variation {i}: {str(result)}")
        else:
            prompt, error = result
            if prompt:
                prompts.append(prompt)
            elif error:
                errors.append(f"Variation {i}: {error}")
    
    if not prompts:
        return [], f"All variations failed: {'; '.join(errors)}"
    
    return prompts, None


class CheckpointManager:
    """Manage checkpoint saving and loading."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = 50
    
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


async def process_single_row(
    idx: int,
    row: pd.Series,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    num_variations: int,
    max_edge: int,
    jpeg_quality: int,
    temperature_range: Tuple[float, float],
    logger: logging.Logger,
) -> Tuple[int, Dict[str, List[str]], List[str]]:
    """
    Process a single row to generate multiple prompt variations for all control variants.
    
    Returns:
        (index, {control_key: prompts_list}, errors_list)
    """
    target_path = row["path_target"]
    
    # Identify all available control images
    control_paths = {}
    
    # Main control image
    if pd.notna(row.get("path_control")):
        control_paths["path_control"] = row["path_control"]
    
    # Alternative control images (path_control_1, path_control_2, etc.)
    for i in range(1, 10):  # Support up to path_control_9
        control_key = f"path_control_{i}"
        if control_key in row and pd.notna(row[control_key]):
            control_paths[control_key] = row[control_key]
    
    # Generate prompts for each control variant
    results = {}
    errors = []
    
    for control_key, control_path in control_paths.items():
        prompts, error = await generate_prompt_variations(
            client,
            control_path,
            target_path,
            model,
            max_tokens,
            num_variations,
            max_edge,
            jpeg_quality,
            temperature_range,
        )
        
        if prompts:
            results[control_key] = prompts
        else:
            error_msg = f"{control_key}: {error}"
            errors.append(error_msg)
            logger.warning(f"Row {idx} {error_msg}")
    
    return idx, results, errors


async def process_batch(
    df: pd.DataFrame,
    indices: List[int],
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    num_variations: int,
    max_edge: int,
    jpeg_quality: int,
    temperature_range: Tuple[float, float],
    logger: logging.Logger,
) -> List[Tuple[int, Dict[str, List[str]], List[str]]]:
    """
    Process a batch of rows concurrently.
    
    Returns:
        List of (index, {control_key: prompts_list}, errors_list) tuples
    """
    tasks = []
    for idx in indices:
        row = df.iloc[idx]
        tasks.append(
            process_single_row(
                idx,
                row,
                client,
                model,
                max_tokens,
                num_variations,
                max_edge,
                jpeg_quality,
                temperature_range,
                logger,
            )
        )
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Unhandled exception in batch: {result}")
        else:
            processed_results.append(result)
    
    return processed_results


def create_sft_dataset(
    df: pd.DataFrame,
    results: Dict[int, Dict[str, List[str]]],
    output_path: Path,
    logger: logging.Logger,
):
    """Create SFT dataset with expanded rows for all control variants."""
    sft_rows = []
    
    for idx, control_variants in results.items():
        row = df.iloc[idx]
        target_path = row["path_target"]
        
        for control_key, prompts in control_variants.items():
            control_path = row[control_key]
            
            for variation_idx, prompt in enumerate(prompts):
                sft_rows.append({
                    "path_control": control_path,
                    "path_target": target_path,
                    "prompt": prompt,
                    "original_idx": idx,
                    "control_variant": control_key,
                    "variation_idx": variation_idx,
                })
    
    sft_df = pd.DataFrame(sft_rows)
    sft_df.to_csv(output_path, index=False)
    logger.info(f"SFT dataset saved: {len(sft_df)} rows -> {output_path}")


def create_dpo_dataset(
    df: pd.DataFrame,
    results: Dict[int, Dict[str, List[str]]],
    output_path: Path,
    logger: logging.Logger,
):
    """
    Create DPO dataset with chosen/rejected pairs for all control variants.
    
    Strategy: Use lower temperature prompts as "chosen" and higher as "rejected".
    For each control variant with N variations, create multiple pairs.
    """
    dpo_rows = []
    
    for idx, control_variants in results.items():
        row = df.iloc[idx]
        target_path = row["path_target"]
        
        for control_key, prompts in control_variants.items():
            if len(prompts) < 2:
                continue  # Need at least 2 prompts for DPO
            
            control_path = row[control_key]
            
            # Create pairs: first half as chosen, second half as rejected
            mid_point = len(prompts) // 2
            
            # Strategy 1: Pair first half with second half
            for i in range(mid_point):
                chosen_prompt = prompts[i]  # Lower temperature
                rejected_prompt = prompts[mid_point + i] if mid_point + i < len(prompts) else prompts[-1]
                
                dpo_rows.append({
                    "path_control": control_path,
                    "path_target": target_path,
                    "chosen_prompt": chosen_prompt,
                    "rejected_prompt": rejected_prompt,
                    "original_idx": idx,
                    "control_variant": control_key,
                    "pair_idx": i,
                })
            
            # Strategy 2: Also create sequential pairs for diversity
            for i in range(len(prompts) - 1):
                dpo_rows.append({
                    "path_control": control_path,
                    "path_target": target_path,
                    "chosen_prompt": prompts[i],
                    "rejected_prompt": prompts[i + 1],
                    "original_idx": idx,
                    "control_variant": control_key,
                    "pair_idx": f"seq_{i}",
                })
    
    dpo_df = pd.DataFrame(dpo_rows)
    dpo_df.to_csv(output_path, index=False)
    logger.info(f"DPO dataset saved: {len(dpo_df)} pairs -> {output_path}")


async def main_async(args):
    """Main async processing function."""
    # Setup paths
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("/skynas/01/hardik/qwen-dataset") / f"multi-prompt-{input_path.stem}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sft_output_path = output_dir / "sft_dataset.csv"
    dpo_output_path = output_dir / "dpo_dataset.csv"
    checkpoint_path = output_dir / "generation.checkpoint.json"
    error_log_path = output_dir / "generation.errors.log"
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("MULTI-PROMPT DATASET GENERATION")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Variations per pair: {args.num_variations}")
    logger.info(f"Temperature range: {args.temp_min} - {args.temp_max}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info("=" * 80)
    
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
    results_dict = {}
    
    if args.resume and checkpoint_path.exists():
        checkpoint = checkpoint_manager.load()
        completed_indices = checkpoint.get("completed_indices", [])
        logger.info(f"Resuming from checkpoint: {len(completed_indices)} rows completed")
        
        # Load partial results if exist
        if sft_output_path.exists():
            temp_df = pd.read_csv(sft_output_path)
            for orig_idx in temp_df["original_idx"].unique():
                if orig_idx not in results_dict:
                    results_dict[orig_idx] = {}
                
                rows = temp_df[temp_df["original_idx"] == orig_idx]
                for control_variant in rows["control_variant"].unique():
                    variant_rows = rows[rows["control_variant"] == control_variant]
                    results_dict[orig_idx][control_variant] = variant_rows["prompt"].tolist()
    
    # Determine remaining indices
    all_indices = list(range(start_idx, end_idx))
    remaining_indices = [idx for idx in all_indices if idx not in completed_indices]
    logger.info(f"Remaining rows to process: {len(remaining_indices)}")
    
    if not remaining_indices:
        logger.info("No rows to process.")
        if results_dict:
            logger.info("Generating final datasets from existing results...")
            create_sft_dataset(df, results_dict, sft_output_path, logger)
            create_dpo_dataset(df, results_dict, dpo_output_path, logger)
        logger.info("Exiting.")
        return
    
    # Initialize OpenAI client
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    
    # Process in batches
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
        total_expected = len(remaining_indices) * args.num_variations
        with tqdm_asyncio(
            total=len(remaining_indices),
            desc="Processing rows",
            unit="row"
        ) as pbar:
            for batch_indices in batches:
                results = await process_batch(
                    df,
                    batch_indices,
                    client,
                    args.model,
                    args.max_tokens,
                    args.num_variations,
                    args.max_edge,
                    args.jpeg_quality,
                    (args.temp_min, args.temp_max),
                    logger,
                )
                
                # Update results
                for idx, control_variants, errors in results:
                    if control_variants:
                        results_dict[idx] = control_variants
                        completed_indices.append(idx)
                        success_count += 1
                        
                        total_variations = sum(len(prompts) for prompts in control_variants.values())
                        num_variants = len(control_variants)
                        
                        pbar.set_postfix({
                            "success": success_count,
                            "errors": error_count,
                            "variants": num_variants,
                            "prompts": total_variations
                        })
                    
                    if errors:
                        error_count += len(errors)
                        for error in errors:
                            error_log.write(f"{idx},{error}\n")
                        error_log.flush()
                    
                    pbar.update(1)
                
                # Save checkpoint periodically
                if len(completed_indices) % checkpoint_manager.checkpoint_interval == 0:
                    checkpoint_manager.save(completed_indices, max(completed_indices))
                    # Save intermediate results
                    create_sft_dataset(df, results_dict, sft_output_path, logger)
                    create_dpo_dataset(df, results_dict, dpo_output_path, logger)
                    logger.info(f"Checkpoint saved: {len(completed_indices)} completed")
        
        # Final save
        checkpoint_manager.save(
            completed_indices,
            max(completed_indices) if completed_indices else -1
        )
        
        # Generate final datasets
        logger.info("Generating final datasets...")
        create_sft_dataset(df, results_dict, sft_output_path, logger)
        create_dpo_dataset(df, results_dict, dpo_output_path, logger)
        
        # Statistics
        total_prompts = 0
        total_variants = 0
        for control_variants in results_dict.values():
            total_variants += len(control_variants)
            for prompts in control_variants.values():
                total_prompts += len(prompts)
        
        avg_prompts = total_prompts / len(results_dict) if results_dict else 0
        avg_variants = total_variants / len(results_dict) if results_dict else 0
        
        logger.info("=" * 80)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total rows processed: {len(remaining_indices)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {error_count}")
        logger.info(f"Total control variants processed: {total_variants}")
        logger.info(f"Total prompts generated: {total_prompts}")
        logger.info(f"Average control variants per row: {avg_variants:.2f}")
        logger.info(f"Average prompts per row: {avg_prompts:.2f}")
        logger.info(f"SFT dataset: {sft_output_path}")
        logger.info(f"DPO dataset: {dpo_output_path}")
        
        if error_count > 0:
            logger.info(f"Error log: {error_log_path}")
        
        logger.info("=" * 80)
        
    finally:
        error_log.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate multiple prompt variations for DPO and SFT training",
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
        "--output_dir",
        type=str,
        help="Output directory (default: /skynas/01/hardik/qwen-dataset/multi-prompt-{input_stem})",
    )
    
    parser.add_argument(
        "--num_variations",
        type=int,
        default=3,
        help="Number of prompt variations per control-target pair (default: 3)",
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
        "--temp_min",
        type=float,
        default=0.1,
        help="Minimum temperature for variations (default: 0.1)",
    )
    
    parser.add_argument(
        "--temp_max",
        type=float,
        default=0.5,
        help="Maximum temperature for variations (default: 0.5)",
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of concurrent row-level requests (default: 3)",
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

