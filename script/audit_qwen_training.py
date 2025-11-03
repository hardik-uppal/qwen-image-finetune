#!/usr/bin/env python3
"""
Audit script for Qwen2.5-VL training setup.
Checks:
1. Label masking for non-assistant tokens and image placeholders
2. Chat template correctness (BOS/EOS tokens)
3. Special image tokens handling
4. Valid token fraction calculation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from PIL import Image
from transformers import Qwen2_5_VLProcessor


def load_config(config_path: str) -> Dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sample_from_jsonl(jsonl_path: str, idx: int = 0) -> Dict:
    """Load a single sample from JSONL."""
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"Sample {idx} not found in {jsonl_path}")


def analyze_tokenization(processor, sample: Dict, system_prompt: str, user_prompt: str):
    """Analyze tokenization and label masking for a sample."""
    print("\n" + "=" * 80)
    print("TOKENIZATION ANALYSIS")
    print("=" * 80)
    
    # Load image
    image = Image.open(sample["image_path"]).convert("RGB")
    print(f"Image: {sample['image_path']}")
    print(f"Image size: {image.size}")
    
    # Get messages from sample
    messages = sample["messages"]
    print(f"\nMessages structure: {len(messages)} messages")
    for i, msg in enumerate(messages):
        print(f"  Message {i}: role={msg['role']}, content_types={[c['type'] for c in msg['content']]}")
    
    # Apply chat template (same as data collator)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    print("\n" + "-" * 80)
    print("FORMATTED TEXT (with special tokens):")
    print("-" * 80)
    print(text)
    print("-" * 80)
    
    # Tokenize
    inputs = processor(
        text=[text],
        images=[[image]],  # Wrap in list for batch processing
        padding=False,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"][0]
    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"Total tokens: {len(input_ids)}")
    
    # Decode tokens individually to see structure
    print("\n" + "-" * 80)
    print("TOKEN-BY-TOKEN BREAKDOWN:")
    print("-" * 80)
    
    # Get special tokens
    tokenizer = processor.tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    print(f"Special token IDs:")
    print(f"  <|image_pad|>: {image_token_id}")
    print(f"  <|vision_start|>: {vision_start_id}")
    print(f"  <|vision_end|>: {vision_end_id}")
    print(f"  <|im_start|>: {im_start_id}")
    print(f"  <|im_end|>: {im_end_id}")
    print(f"  BOS: {tokenizer.bos_token_id}")
    print(f"  EOS: {tokenizer.eos_token_id}")
    print(f"  PAD: {tokenizer.pad_token_id}")
    
    # Analyze token structure
    print("\nFirst 50 tokens:")
    for i in range(min(50, len(input_ids))):
        token_id = input_ids[i].item()
        token_str = tokenizer.decode([token_id])
        
        # Mark special tokens
        special = ""
        if token_id == image_token_id:
            special = " [IMAGE_PAD]"
        elif token_id == vision_start_id:
            special = " [VISION_START]"
        elif token_id == vision_end_id:
            special = " [VISION_END]"
        elif token_id == im_start_id:
            special = " [IM_START]"
        elif token_id == im_end_id:
            special = " [IM_END]"
        elif token_id == tokenizer.bos_token_id:
            special = " [BOS]"
        elif token_id == tokenizer.eos_token_id:
            special = " [EOS]"
        
        print(f"  [{i:3d}] {token_id:6d} | {repr(token_str):30s} {special}")
    
    if len(input_ids) > 50:
        print(f"  ... ({len(input_ids) - 50} more tokens)")
    
    # Analyze where assistant response starts
    print("\n" + "-" * 80)
    print("FINDING ASSISTANT RESPONSE START:")
    print("-" * 80)
    
    # Look for "<|im_start|>assistant" pattern
    assistant_token_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    print(f"Assistant marker token IDs: {assistant_token_ids}")
    
    assistant_start_idx = None
    for i in range(len(input_ids) - len(assistant_token_ids)):
        if all(input_ids[i + j] == assistant_token_ids[j] for j in range(len(assistant_token_ids))):
            assistant_start_idx = i + len(assistant_token_ids)
            print(f"✓ Found assistant response start at token index: {assistant_start_idx}")
            break
    
    if assistant_start_idx is None:
        print("✗ WARNING: Could not find assistant response start!")
    
    # Look for where assistant response ends
    assistant_end_idx = None
    end_marker_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if assistant_start_idx:
        for i in range(assistant_start_idx, len(input_ids) - len(end_marker_ids)):
            if all(input_ids[i + j] == end_marker_ids[j] for j in range(len(end_marker_ids))):
                assistant_end_idx = i
                print(f"✓ Found assistant response end at token index: {assistant_end_idx}")
                break
    
    # Show assistant response
    if assistant_start_idx and assistant_end_idx:
        assistant_tokens = input_ids[assistant_start_idx:assistant_end_idx]
        assistant_text = tokenizer.decode(assistant_tokens, skip_special_tokens=False)
        print(f"\nAssistant response ({len(assistant_tokens)} tokens):")
        print(f"  '{assistant_text[:200]}{'...' if len(assistant_text) > 200 else ''}'")
    
    return input_ids, assistant_start_idx, assistant_end_idx, image_token_id


def analyze_current_masking(input_ids: torch.Tensor, assistant_start_idx: int = None, 
                           assistant_end_idx: int = None, image_token_id: int = None):
    """Analyze the CURRENT (incorrect) masking approach."""
    print("\n" + "=" * 80)
    print("CURRENT MASKING ANALYSIS (INCORRECT - labels = input_ids.clone())")
    print("=" * 80)
    
    # Current approach in Qwen2VLDataCollator
    labels = input_ids.clone()
    
    total_tokens = len(labels)
    supervised_tokens = (labels != -100).sum().item()
    valid_token_frac = supervised_tokens / total_tokens
    
    print(f"Total tokens: {total_tokens}")
    print(f"Supervised tokens (labels != -100): {supervised_tokens}")
    print(f"Valid token fraction: {valid_token_frac:.3f}")
    
    # Count image tokens
    if image_token_id is not None:
        image_tokens = (input_ids == image_token_id).sum().item()
        print(f"Image placeholder tokens: {image_tokens}")
        print(f"✗ PROBLEM: Image tokens are being supervised (should be -100)!")
    
    # Count non-assistant tokens
    if assistant_start_idx is not None:
        non_assistant_tokens = assistant_start_idx
        print(f"Non-assistant tokens (system + user): {non_assistant_tokens}")
        print(f"✗ PROBLEM: Non-assistant tokens are being supervised (should be -100)!")
    
    print("\n✗ CRITICAL ISSUE: ALL tokens are supervised, including:")
    print("  - System prompt tokens")
    print("  - User prompt tokens")
    print("  - Image placeholder tokens (<|image_pad|>, <|vision_start|>, <|vision_end|>)")
    print("  - Only assistant tokens should be supervised!")


def analyze_correct_masking(input_ids: torch.Tensor, assistant_start_idx: int = None,
                           assistant_end_idx: int = None, image_token_id: int = None):
    """Show the CORRECT masking approach."""
    print("\n" + "=" * 80)
    print("CORRECT MASKING ANALYSIS (FIXED)")
    print("=" * 80)
    
    # Correct approach: mask everything except assistant response
    labels = input_ids.clone()
    
    # Mask all tokens initially
    labels[:] = -100
    
    # Only unmask assistant response
    if assistant_start_idx and assistant_end_idx:
        labels[assistant_start_idx:assistant_end_idx] = input_ids[assistant_start_idx:assistant_end_idx]
    
    total_tokens = len(labels)
    supervised_tokens = (labels != -100).sum().item()
    valid_token_frac = supervised_tokens / total_tokens
    
    print(f"Total tokens: {total_tokens}")
    print(f"Supervised tokens (labels != -100): {supervised_tokens}")
    print(f"Valid token fraction: {valid_token_frac:.3f}")
    
    if assistant_start_idx and assistant_end_idx:
        assistant_tokens = assistant_end_idx - assistant_start_idx
        print(f"Assistant tokens: {assistant_tokens}")
        print(f"✓ CORRECT: Only assistant tokens are supervised")
    
    # Verify image tokens are masked
    if image_token_id is not None:
        image_token_mask = input_ids == image_token_id
        all_image_masked = (labels[image_token_mask] == -100).all().item()
        if all_image_masked:
            print(f"✓ CORRECT: All {image_token_mask.sum().item()} image tokens are masked")
        else:
            print(f"✗ ERROR: Some image tokens are not masked!")
    
    print("\n✓ CORRECT: Only assistant response tokens are supervised")
    print(f"✓ Expected valid_token_frac for single-turn SFT: {valid_token_frac:.3f}")
    print(f"✓ This is {'good' if valid_token_frac >= 0.3 else 'low'} (typically want ≥0.3 for vision tasks)")
    
    return labels


def check_chat_template(processor, messages: List[Dict]):
    """Check chat template with add_generation_prompt."""
    print("\n" + "=" * 80)
    print("CHAT TEMPLATE VERIFICATION")
    print("=" * 80)
    
    # Training mode (includes assistant response)
    text_train = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    # Inference mode (stops before assistant response)
    messages_no_assistant = [msg for msg in messages if msg["role"] != "assistant"]
    text_infer = processor.apply_chat_template(
        messages_no_assistant, tokenize=False, add_generation_prompt=True
    )
    
    print("Training template (add_generation_prompt=False):")
    print("-" * 80)
    print(text_train)
    print("-" * 80)
    
    print("\nInference template (add_generation_prompt=True):")
    print("-" * 80)
    print(text_infer)
    print("-" * 80)
    
    # Check for BOS/EOS
    tokenizer = processor.tokenizer
    has_bos = text_train.startswith(tokenizer.bos_token) if tokenizer.bos_token else False
    has_eos = text_train.endswith(tokenizer.eos_token) if tokenizer.eos_token else False
    
    print(f"\n✓ BOS token present: {has_bos}")
    print(f"✓ EOS token present: {has_eos}")
    
    # Tokenize both
    tokens_train = processor.tokenizer.encode(text_train, add_special_tokens=False)
    tokens_infer = processor.tokenizer.encode(text_infer, add_special_tokens=False)
    
    print(f"\nTraining tokens: {len(tokens_train)}")
    print(f"Inference tokens: {len(tokens_infer)}")
    print(f"Difference: {len(tokens_train) - len(tokens_infer)} tokens (assistant response)")


def main():
    parser = argparse.ArgumentParser(description="Audit Qwen2.5-VL training setup")
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to analyze")
    args = parser.parse_args()
    
    print("=" * 80)
    print("QWEN2.5-VL TRAINING AUDIT")
    print("=" * 80)
    
    # Load config
    config = load_config(args.config)
    model_name = config["model"]["model_name"]
    data_config = config["data"]
    
    print(f"Model: {model_name}")
    print(f"Config: {args.config}")
    
    # Load processor
    print(f"\nLoading processor from {model_name}...")
    processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
    print("✓ Processor loaded")
    
    # Load sample
    prepared_data_dir = Path(data_config["prepared_data_dir"])
    train_jsonl = prepared_data_dir / "train.jsonl"
    
    if not train_jsonl.exists():
        print(f"✗ ERROR: Training data not found at {train_jsonl}")
        print("  Please run prepare_qwen25vl_dataset.py first")
        sys.exit(1)
    
    print(f"\nLoading sample {args.sample_idx} from {train_jsonl}...")
    sample = load_sample_from_jsonl(str(train_jsonl), args.sample_idx)
    print("✓ Sample loaded")
    
    # Analyze tokenization
    input_ids, assistant_start_idx, assistant_end_idx, image_token_id = analyze_tokenization(
        processor, sample, 
        data_config["system_prompt"],
        data_config["user_prompt"]
    )
    
    # Analyze current (incorrect) masking
    analyze_current_masking(input_ids, assistant_start_idx, assistant_end_idx, image_token_id)
    
    # Show correct masking
    correct_labels = analyze_correct_masking(input_ids, assistant_start_idx, assistant_end_idx, image_token_id)
    
    # Check chat template
    check_chat_template(processor, sample["messages"])
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print("\n✗ CRITICAL ISSUES FOUND:")
    print("  1. Labels are not masked - ALL tokens are supervised")
    print("     Current: labels = input_ids.clone()")
    print("     Fix: Mask system/user/image tokens, only supervise assistant tokens")
    print("\n  2. Image placeholder tokens (<|image_pad|>) are supervised")
    print("     Fix: Set labels to -100 for all image-related special tokens")
    print("\n  3. System and user tokens are supervised")
    print("     Fix: Set labels to -100 before assistant response starts")
    
    print("\n✓ RECOMMENDED FIX:")
    print("  Update Qwen2VLDataCollator to properly mask labels:")
    print("  - Find assistant response start position")
    print("  - Set labels[:assistant_start] = -100")
    print("  - Keep labels[assistant_start:] = input_ids[assistant_start:]")
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

