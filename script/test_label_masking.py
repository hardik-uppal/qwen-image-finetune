#!/usr/bin/env python3
"""
Test script to verify label masking is working correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from transformers import Qwen2_5_VLProcessor

# Import the fixed data collator from training script
from script.train_qwen25vl_prompt_generation import (
    LazySupervisedDataset,
    Qwen2VLDataCollator
)


def test_masking():
    """Test the label masking on actual data."""
    print("=" * 80)
    print("TESTING LABEL MASKING FIX")
    print("=" * 80)
    
    # Load config
    config_path = "configs/qwen25vl_prompt_gen.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_name = config["model"]["model_name"]
    prepared_data_dir = Path(config["data"]["prepared_data_dir"])
    train_jsonl = prepared_data_dir / "train.jsonl"
    
    if not train_jsonl.exists():
        print(f"✗ Training data not found at {train_jsonl}")
        print("  Please run prepare_qwen25vl_dataset.py first")
        return False
    
    print(f"Model: {model_name}")
    print(f"Data: {train_jsonl}")
    
    # Load processor
    print("\nLoading processor...")
    processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
    print("✓ Processor loaded")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = LazySupervisedDataset(str(train_jsonl))
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Initialize collator with FIX
    print("\nInitializing data collator with masking fix...")
    collator = Qwen2VLDataCollator(processor)
    print("✓ Collator initialized")
    
    # Test on multiple samples
    print("\n" + "=" * 80)
    print("TESTING ON SAMPLES")
    print("=" * 80)
    
    num_test_samples = min(5, len(dataset))
    total_stats = {
        "total_tokens": 0,
        "supervised_tokens": 0,
        "masked_tokens": 0,
        "image_tokens_total": 0,
        "image_tokens_masked": 0,
    }
    
    tokenizer = processor.tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    
    for i in range(num_test_samples):
        print(f"\n--- Sample {i} ---")
        
        # Get sample and collate
        sample = dataset[i]
        batch = collator([sample])
        
        # Analyze labels
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        
        total_tokens = len(labels)
        supervised_tokens = (labels != -100).sum().item()
        masked_tokens = (labels == -100).sum().item()
        valid_token_frac = supervised_tokens / total_tokens
        
        # Count image tokens
        image_tokens = (input_ids == image_token_id).sum().item()
        image_tokens_masked = ((input_ids == image_token_id) & (labels == -100)).sum().item()
        
        print(f"Total tokens: {total_tokens}")
        print(f"Supervised: {supervised_tokens}, Masked: {masked_tokens}")
        print(f"Valid token fraction: {valid_token_frac:.3f}")
        print(f"Image tokens: {image_tokens} (masked: {image_tokens_masked})")
        
        # Verify
        if image_tokens_masked == image_tokens:
            print("✓ All image tokens are masked")
        else:
            print(f"✗ ERROR: {image_tokens - image_tokens_masked} image tokens NOT masked!")
        
        if 0.02 <= valid_token_frac <= 0.6:
            print("✓ Valid token fraction is reasonable")
        else:
            print(f"⚠️  Valid token fraction is {'too high' if valid_token_frac > 0.6 else 'too low'}")
        
        # Accumulate stats
        total_stats["total_tokens"] += total_tokens
        total_stats["supervised_tokens"] += supervised_tokens
        total_stats["masked_tokens"] += masked_tokens
        total_stats["image_tokens_total"] += image_tokens
        total_stats["image_tokens_masked"] += image_tokens_masked
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    avg_valid_frac = total_stats["supervised_tokens"] / total_stats["total_tokens"]
    print(f"Across {num_test_samples} samples:")
    print(f"  Total tokens: {total_stats['total_tokens']}")
    print(f"  Supervised tokens: {total_stats['supervised_tokens']}")
    print(f"  Masked tokens: {total_stats['masked_tokens']}")
    print(f"  Average valid token fraction: {avg_valid_frac:.3f}")
    print(f"  Image tokens: {total_stats['image_tokens_total']} (masked: {total_stats['image_tokens_masked']})")
    
    # Final verdict
    print("\n" + "=" * 80)
    all_image_masked = total_stats["image_tokens_masked"] == total_stats["image_tokens_total"]
    reasonable_frac = 0.02 <= avg_valid_frac <= 0.6
    
    if all_image_masked and reasonable_frac:
        print("✓✓✓ PASS: Label masking is working correctly!")
        print(f"✓ All {total_stats['image_tokens_total']} image tokens are masked")
        print(f"✓ Valid token fraction ({avg_valid_frac:.3f}) is reasonable")
        return True
    else:
        print("✗✗✗ FAIL: Label masking has issues!")
        if not all_image_masked:
            print(f"✗ {total_stats['image_tokens_total'] - total_stats['image_tokens_masked']} image tokens are NOT masked")
        if not reasonable_frac:
            print(f"✗ Valid token fraction ({avg_valid_frac:.3f}) is out of range")
        return False


if __name__ == "__main__":
    success = test_masking()
    sys.exit(0 if success else 1)

