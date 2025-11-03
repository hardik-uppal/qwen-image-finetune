#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen2.5-VL prompt generation model.

Generates image editing prompts from control images using the fine-tuned LoRA adapter.

Usage:
    # Single image inference
    python script/infer_qwen25vl_prompt_gen.py \
        --adapter_path workspace/qwen25vl-prompt-gen-lora \
        --image path/to/image.jpg

    # Batch inference on validation set
    python script/infer_qwen25vl_prompt_gen.py \
        --adapter_path workspace/qwen25vl-prompt-gen-lora \
        --test_csv workspace/metadata-v12-recaptioned-long/val.csv \
        --output_csv results/generated_prompts.csv \
        --num_samples 100

    # Compare with ground truth (calculate metrics)
    python script/infer_qwen25vl_prompt_gen.py \
        --adapter_path workspace/qwen25vl-prompt-gen-lora \
        --test_csv workspace/metadata-v12-recaptioned-long/val.csv \
        --output_csv results/generated_prompts.csv \
        --compute_metrics
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

DEFAULT_SYSTEM_PROMPT = """You are an expert photo editor. Analyze the provided photograph and describe the visual adjustments that could be made to enhance or transform it. Focus on actionable editing instructions including lighting adjustments, color corrections, object additions/removals, perspective corrections, lens distortion fixes, and compositional improvements."""

DEFAULT_USER_PROMPT = "Describe the edits needed for this image."


def load_model_and_processor(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    adapter_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    Load model, processor, and optionally LoRA adapter.
    
    Args:
        model_name: Base model name or path
        adapter_path: Path to fine-tuned LoRA adapter (optional)
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading base model: {model_name}")
    
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
    
    # Load LoRA adapter if specified
    if adapter_path:
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        logger.info("✓ LoRA adapter loaded successfully")
    else:
        model = base_model
        model.eval()
    
    return model, processor


def generate_prompt_from_image(
    image_path: str,
    model,
    processor,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate editing prompt from a single image.
    
    Args:
        image_path: Path to control image
        model: Loaded Qwen2VL model
        processor: Qwen2VL processor
        system_prompt: System instruction
        user_prompt: User question
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        Generated prompt as string
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Format messages in chat template
    messages = [
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
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
        )
    
    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    
    return output_text[0]


def batch_inference(
    csv_path: str,
    model,
    processor,
    output_csv: str,
    num_samples: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> pd.DataFrame:
    """
    Run batch inference on validation set.
    
    Args:
        csv_path: Path to CSV with 'path_control' column
        model: Loaded model
        processor: Loaded processor
        output_csv: Path to save results
        num_samples: Optional limit on number of samples
        system_prompt: System instruction
        user_prompt: User question
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        
    Returns:
        DataFrame with results
    """
    logger.info(f"Loading test data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if num_samples is not None and len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42)
        logger.info(f"Limited to {num_samples} samples")
    
    logger.info(f"Generating prompts for {len(df)} images...")
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        try:
            image_path = row["path_control"]
            
            # Generate prompt
            generated_prompt = generate_prompt_from_image(
                image_path=image_path,
                model=model,
                processor=processor,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            result = {
                "path_control": image_path,
                "generated_prompt": generated_prompt,
            }
            
            # Add ground truth if available
            if "prompt" in row:
                result["ground_truth_prompt"] = row["prompt"]
            
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Error processing {row.get('path_control', 'unknown')}: {e}")
            results.append({
                "path_control": row.get("path_control", "unknown"),
                "generated_prompt": f"ERROR: {str(e)}",
                "ground_truth_prompt": row.get("prompt", ""),
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")
    
    return results_df


def compute_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute evaluation metrics comparing generated vs ground truth prompts.
    
    Args:
        results_df: DataFrame with 'generated_prompt' and 'ground_truth_prompt' columns
        
    Returns:
        Dictionary of metric scores
    """
    try:
        from evaluate import load
        
        # Filter valid samples
        valid_df = results_df[
            (results_df["generated_prompt"].notna()) &
            (results_df["ground_truth_prompt"].notna()) &
            (~results_df["generated_prompt"].str.startswith("ERROR"))
        ]
        
        if len(valid_df) == 0:
            logger.warning("No valid samples for metric computation")
            return {}
        
        predictions = valid_df["generated_prompt"].tolist()
        references = valid_df["ground_truth_prompt"].tolist()
        
        logger.info(f"Computing metrics on {len(valid_df)} valid samples...")
        
        metrics = {}
        
        # BLEU score
        try:
            bleu = load("bleu")
            bleu_results = bleu.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            metrics["bleu"] = bleu_results["bleu"]
            logger.info(f"BLEU: {metrics['bleu']:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute BLEU: {e}")
        
        # ROUGE score
        try:
            rouge = load("rouge")
            rouge_results = rouge.compute(
                predictions=predictions,
                references=references
            )
            metrics["rouge-1"] = rouge_results["rouge1"]
            metrics["rouge-2"] = rouge_results["rouge2"]
            metrics["rouge-l"] = rouge_results["rougeL"]
            logger.info(f"ROUGE-1: {metrics['rouge-1']:.4f}")
            logger.info(f"ROUGE-2: {metrics['rouge-2']:.4f}")
            logger.info(f"ROUGE-L: {metrics['rouge-l']:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute ROUGE: {e}")
        
        # BERTScore (computationally expensive, optional)
        try:
            bertscore = load("bertscore")
            bert_results = bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
                model_type="microsoft/deberta-base-mnli"
            )
            metrics["bertscore_f1"] = sum(bert_results["f1"]) / len(bert_results["f1"])
            logger.info(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute BERTScore: {e}")
        
        return metrics
        
    except ImportError:
        logger.warning("'evaluate' library not installed. Install with: pip install evaluate")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen2.5-VL")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Base model name")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path for inference")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="CSV file for batch inference")
    parser.add_argument("--output_csv", type=str, default="results/generated_prompts.csv",
                        help="Output CSV path for batch results")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of samples for batch inference")
    parser.add_argument("--compute_metrics", action="store_true",
                        help="Compute evaluation metrics (requires ground truth)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt")
    parser.add_argument("--user_prompt", type=str, default=DEFAULT_USER_PROMPT,
                        help="User prompt")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.image is None and args.test_csv is None:
        parser.error("Must specify either --image or --test_csv")
    
    # Load model and processor
    model, processor = load_model_and_processor(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
    )
    
    # Single image inference
    if args.image:
        logger.info(f"Generating prompt for {args.image}")
        prompt = generate_prompt_from_image(
            image_path=args.image,
            model=model,
            processor=processor,
            system_prompt=args.system_prompt,
            user_prompt=args.user_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("\n" + "="*80)
        print("Generated Prompt:")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
    
    # Batch inference
    if args.test_csv:
        results_df = batch_inference(
            csv_path=args.test_csv,
            model=model,
            processor=processor,
            output_csv=args.output_csv,
            num_samples=args.num_samples,
            system_prompt=args.system_prompt,
            user_prompt=args.user_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        
        # Compute metrics if requested
        if args.compute_metrics:
            metrics = compute_metrics(results_df)
            if metrics:
                print("\n" + "="*80)
                print("Evaluation Metrics:")
                print("="*80)
                for metric_name, score in metrics.items():
                    print(f"{metric_name}: {score:.4f}")
                print("="*80 + "\n")


if __name__ == "__main__":
    main()

