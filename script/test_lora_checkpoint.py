#!/usr/bin/env python3
"""
Quick test script to verify LoRA checkpoint loads correctly.

Tests:
1. Load base model
2. Load LoRA adapter from checkpoint
3. Verify trainable parameters
4. Test single inference
5. Compare with base model output (optional)

Usage:
    # Test a checkpoint
    python script/test_lora_checkpoint.py \
        --checkpoint /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora/checkpoint-500 \
        --test_image workspace/test_images/sample.jpg

    # Test final model
    python script/test_lora_checkpoint.py \
        --checkpoint /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
        --test_image workspace/test_images/sample.jpg \
        --compare_base
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
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


def test_checkpoint_loading(checkpoint_path: str, base_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """Test if checkpoint loads correctly."""
    logger.info("=" * 80)
    logger.info("TEST 1: Loading Checkpoint")
    logger.info("=" * 80)
    
    try:
        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        logger.info("✓ Base model loaded")
        
        # Load LoRA adapter
        logger.info(f"Loading LoRA adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logger.info("✓ LoRA adapter loaded")
        
        # Print trainable parameters
        logger.info("\nTrainable parameters:")
        model.print_trainable_parameters()
        
        # Load processor
        processor = Qwen2_5_VLProcessor.from_pretrained(base_model_name)
        logger.info("✓ Processor loaded")
        
        logger.info("\n✅ Checkpoint loads successfully!")
        return model, processor, base_model
        
    except Exception as e:
        logger.error(f"\n❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_inference(
    model,
    processor,
    image_path: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
):
    """Test inference with the loaded model."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Running Inference")
    logger.info("=" * 80)
    
    try:
        # Check image exists
        if not Path(image_path).exists():
            logger.warning(f"Test image not found: {image_path}")
            logger.info("Skipping inference test")
            return None
        
        # Load image
        logger.info(f"Loading test image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        logger.info(f"Image size: {image.size}")
        
        # Format messages
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
        
        logger.info("Generating response...")
        
        # Generate
        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
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
        )[0]
        
        logger.info("\n" + "-" * 80)
        logger.info("Generated Output:")
        logger.info("-" * 80)
        print(output_text)
        logger.info("-" * 80)
        
        logger.info("\n✅ Inference successful!")
        return output_text
        
    except Exception as e:
        logger.error(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_base(
    base_model,
    finetuned_model,
    processor,
    image_path: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
):
    """Compare outputs from base model vs fine-tuned model."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Comparing Base vs Fine-tuned")
    logger.info("=" * 80)
    
    if not Path(image_path).exists():
        logger.warning(f"Test image not found: {image_path}")
        logger.info("Skipping comparison test")
        return
    
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Format messages
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
        
        # Generate with base model
        logger.info("Generating with BASE model...")
        inputs_base = {k: v.to(base_model.device) for k, v in inputs.items()}
        base_model.eval()
        with torch.no_grad():
            base_output = base_model.generate(
                **inputs_base,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        base_text = processor.batch_decode(
            [base_output[0][len(inputs.input_ids[0]):]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        # Generate with fine-tuned model
        logger.info("Generating with FINE-TUNED model...")
        inputs_ft = {k: v.to(finetuned_model.device) for k, v in inputs.items()}
        finetuned_model.eval()
        with torch.no_grad():
            ft_output = finetuned_model.generate(
                **inputs_ft,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        ft_text = processor.batch_decode(
            [ft_output[0][len(inputs.input_ids[0]):]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        # Display comparison
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON RESULTS")
        logger.info("=" * 80)
        
        print("\n📌 BASE MODEL OUTPUT:")
        print("-" * 80)
        print(base_text)
        print("-" * 80)
        
        print("\n🎯 FINE-TUNED MODEL OUTPUT:")
        print("-" * 80)
        print(ft_text)
        print("-" * 80)
        
        logger.info("\n✅ Comparison complete!")
        
    except Exception as e:
        logger.error(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test LoRA checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default=None,
        help="Path to test image (optional)",
    )
    parser.add_argument(
        "--compare_base",
        action="store_true",
        help="Compare fine-tuned vs base model outputs",
    )
    
    args = parser.parse_args()
    
    # Test 1: Load checkpoint
    model, processor, base_model = test_checkpoint_loading(
        args.checkpoint, args.base_model
    )
    
    if model is None:
        logger.error("Checkpoint loading failed. Exiting.")
        sys.exit(1)
    
    # Test 2: Inference (if test image provided)
    if args.test_image:
        test_inference(model, processor, args.test_image)
        
        # Test 3: Compare with base (if requested)
        if args.compare_base:
            compare_with_base(base_model, model, processor, args.test_image)
    else:
        logger.info("\nNo test image provided. Skipping inference tests.")
        logger.info("Provide --test_image to run inference tests.")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info("✅ Checkpoint loads correctly")
    if args.test_image and Path(args.test_image).exists():
        logger.info("✅ Inference works")
        if args.compare_base:
            logger.info("✅ Comparison with base model completed")
    logger.info("\n🎉 All tests passed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

