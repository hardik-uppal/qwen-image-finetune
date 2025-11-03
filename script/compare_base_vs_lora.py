#!/usr/bin/env python3
"""
Compare base model vs fine-tuned LoRA model performance.

Runs evaluation on the same dataset with both models and compares:
- Quantitative metrics (BLEU, ROUGE, BERTScore)
- Sample outputs (qualitative comparison)
- Performance statistics

Usage:
    python script/compare_base_vs_lora.py \
        --adapter_path /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
        --test_csv workspace/metadata-v12-recaptioned-long/val.csv \
        --output_dir results/comparison \
        --num_samples 100

    # Full evaluation
    python script/compare_base_vs_lora.py \
        --adapter_path /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
        --test_csv workspace/metadata-v12-recaptioned-long/val.csv \
        --output_dir results/comparison
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

DEFAULT_SYSTEM_PROMPT = """You are an expert photo editor. Analyze the provided photograph and describe the visual adjustments that could be made to enhance or transform it. Focus on actionable editing instructions including lighting adjustments, color corrections, object additions/removals, perspective corrections, lens distortion fixes, and compositional improvements."""

DEFAULT_USER_PROMPT = "Describe the edits needed for this image."


def load_models(
    base_model_name: str,
    adapter_path: str,
):
    """Load both base and fine-tuned models."""
    logger.info("=" * 80)
    logger.info("Loading Models")
    logger.info("=" * 80)
    
    # Load base model
    logger.info(f"Loading base model: {base_model_name}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()
    logger.info("✓ Base model loaded")
    
    # Load processor
    processor = Qwen2_5_VLProcessor.from_pretrained(base_model_name)
    logger.info("✓ Processor loaded")
    
    # Load fine-tuned model
    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    finetuned_model.eval()
    logger.info("✓ Fine-tuned model loaded")
    
    logger.info("")
    return base_model, finetuned_model, processor


def generate_prompt(
    image_path: str,
    model,
    processor,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate prompt for a single image."""
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True if temperature > 0 else False,
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    
    return output_text


def run_evaluation(
    csv_path: str,
    base_model,
    finetuned_model,
    processor,
    num_samples: int = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run evaluation on both models."""
    logger.info("=" * 80)
    logger.info("Running Evaluation")
    logger.info("=" * 80)
    
    # Load data
    df = pd.read_csv(csv_path)
    if num_samples is not None and len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42)
    
    logger.info(f"Evaluating on {len(df)} samples\n")
    
    base_results = []
    finetuned_results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            image_path = row["path_control"]
            ground_truth = row.get("prompt", "")
            
            # Generate with base model
            base_output = generate_prompt(
                image_path, base_model, processor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            # Generate with fine-tuned model
            finetuned_output = generate_prompt(
                image_path, finetuned_model, processor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            base_results.append({
                "path_control": image_path,
                "generated_prompt": base_output,
                "ground_truth_prompt": ground_truth,
            })
            
            finetuned_results.append({
                "path_control": image_path,
                "generated_prompt": finetuned_output,
                "ground_truth_prompt": ground_truth,
            })
            
        except Exception as e:
            logger.warning(f"Error processing {row.get('path_control', 'unknown')}: {e}")
            base_results.append({
                "path_control": row.get("path_control", "unknown"),
                "generated_prompt": f"ERROR: {str(e)}",
                "ground_truth_prompt": row.get("prompt", ""),
            })
            finetuned_results.append({
                "path_control": row.get("path_control", "unknown"),
                "generated_prompt": f"ERROR: {str(e)}",
                "ground_truth_prompt": row.get("prompt", ""),
            })
    
    return pd.DataFrame(base_results), pd.DataFrame(finetuned_results)


def compute_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """Compute evaluation metrics."""
    try:
        from evaluate import load
        
        valid_df = results_df[
            (results_df["generated_prompt"].notna()) &
            (results_df["ground_truth_prompt"].notna()) &
            (~results_df["generated_prompt"].str.startswith("ERROR"))
        ]
        
        if len(valid_df) == 0:
            return {}
        
        predictions = valid_df["generated_prompt"].tolist()
        references = valid_df["ground_truth_prompt"].tolist()
        
        metrics = {}
        
        # BLEU
        try:
            bleu = load("bleu")
            bleu_results = bleu.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            metrics["bleu"] = bleu_results["bleu"]
        except Exception as e:
            logger.warning(f"Could not compute BLEU: {e}")
        
        # ROUGE
        try:
            rouge = load("rouge")
            rouge_results = rouge.compute(
                predictions=predictions,
                references=references
            )
            metrics["rouge-1"] = rouge_results["rouge1"]
            metrics["rouge-2"] = rouge_results["rouge2"]
            metrics["rouge-l"] = rouge_results["rougeL"]
        except Exception as e:
            logger.warning(f"Could not compute ROUGE: {e}")
        
        # BERTScore
        try:
            bertscore = load("bertscore")
            bert_results = bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
                model_type="microsoft/deberta-base-mnli"
            )
            metrics["bertscore_f1"] = sum(bert_results["f1"]) / len(bert_results["f1"])
        except Exception as e:
            logger.warning(f"Could not compute BERTScore: {e}")
        
        return metrics
        
    except ImportError:
        logger.warning("'evaluate' library not installed. Install with: pip install evaluate")
        return {}


def display_comparison(base_metrics: Dict, finetuned_metrics: Dict):
    """Display side-by-side comparison of metrics."""
    logger.info("\n" + "=" * 80)
    logger.info("METRICS COMPARISON")
    logger.info("=" * 80)
    
    if not base_metrics or not finetuned_metrics:
        logger.warning("Metrics not available for comparison")
        return
    
    print(f"\n{'Metric':<20} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for metric_name in base_metrics.keys():
        base_val = base_metrics.get(metric_name, 0)
        ft_val = finetuned_metrics.get(metric_name, 0)
        
        if base_val > 0:
            improvement = ((ft_val - base_val) / base_val) * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "N/A"
        
        print(f"{metric_name:<20} {base_val:<15.4f} {ft_val:<15.4f} {improvement_str:<15}")
    
    print("-" * 70)


def display_sample_outputs(base_df: pd.DataFrame, finetuned_df: pd.DataFrame, num_samples: int = 3):
    """Display sample outputs for qualitative comparison."""
    logger.info("\n" + "=" * 80)
    logger.info(f"SAMPLE OUTPUTS (First {num_samples})")
    logger.info("=" * 80)
    
    for i in range(min(num_samples, len(base_df))):
        print(f"\n{'─' * 80}")
        print(f"Sample {i+1}: {base_df.iloc[i]['path_control']}")
        print(f"{'─' * 80}")
        
        print("\n📌 GROUND TRUTH:")
        print(base_df.iloc[i]['ground_truth_prompt'])
        
        print("\n🔵 BASE MODEL:")
        print(base_df.iloc[i]['generated_prompt'])
        
        print("\n🟢 FINE-TUNED MODEL:")
        print(finetuned_df.iloc[i]['generated_prompt'])


def save_results(
    output_dir: Path,
    base_df: pd.DataFrame,
    finetuned_df: pd.DataFrame,
    base_metrics: Dict,
    finetuned_metrics: Dict,
):
    """Save all results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSVs
    base_df.to_csv(output_dir / "base_model_results.csv", index=False)
    finetuned_df.to_csv(output_dir / "finetuned_model_results.csv", index=False)
    logger.info(f"✓ Results saved to {output_dir}")
    
    # Save metrics
    comparison = {
        "base_model": base_metrics,
        "finetuned_model": finetuned_metrics,
        "improvement": {
            metric: ((finetuned_metrics.get(metric, 0) - base_metrics.get(metric, 0)) / base_metrics.get(metric, 1) * 100)
            if base_metrics.get(metric, 0) > 0 else 0
            for metric in base_metrics.keys()
        }
    }
    
    with open(output_dir / "metrics_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"✓ Metrics saved to {output_dir / 'metrics_comparison.json'}")


def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to fine-tuned LoRA adapter"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="CSV file for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparison",
        help="Directory to save results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of samples (None = all)"
    )
    parser.add_argument(
        "--num_display_samples",
        type=int,
        default=3,
        help="Number of sample outputs to display"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    # Load models
    base_model, finetuned_model, processor = load_models(
        args.base_model,
        args.adapter_path,
    )
    
    # Run evaluation
    base_df, finetuned_df = run_evaluation(
        args.test_csv,
        base_model,
        finetuned_model,
        processor,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # Compute metrics
    logger.info("\nComputing metrics for base model...")
    base_metrics = compute_metrics(base_df)
    
    logger.info("Computing metrics for fine-tuned model...")
    finetuned_metrics = compute_metrics(finetuned_df)
    
    # Display comparison
    display_comparison(base_metrics, finetuned_metrics)
    
    # Display sample outputs
    display_sample_outputs(base_df, finetuned_df, args.num_display_samples)
    
    # Save results
    output_dir = Path(args.output_dir)
    save_results(output_dir, base_df, finetuned_df, base_metrics, finetuned_metrics)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Evaluated {len(base_df)} samples")
    logger.info(f"Results saved to: {output_dir}")
    
    if base_metrics and finetuned_metrics:
        logger.info("\nKey Improvements:")
        for metric in ["bleu", "rouge-l", "bertscore_f1"]:
            if metric in base_metrics and metric in finetuned_metrics:
                base_val = base_metrics[metric]
                ft_val = finetuned_metrics[metric]
                improvement = ((ft_val - base_val) / base_val) * 100
                emoji = "🟢" if improvement > 0 else "🔴"
                logger.info(f"  {emoji} {metric}: {improvement:+.1f}%")
    
    logger.info("\n🎉 Comparison complete!")


if __name__ == "__main__":
    main()

