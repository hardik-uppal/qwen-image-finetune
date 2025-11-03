#!/usr/bin/env python3
"""
Test improved system prompts without retraining.

This script lets you compare outputs with:
1. Original prompts
2. Improved prompts (with few-shot examples)
3. Your fine-tuned model with both prompt versions

Usage:
    # Test with fine-tuned model
    python script/test_improved_prompts.py \
        --adapter_path /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
        --image path/to/test/image.jpg
    
    # Compare base vs fine-tuned
    python script/test_improved_prompts.py \
        --adapter_path /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
        --image path/to/test/image.jpg \
        --compare_base
"""

import argparse
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

Image.MAX_IMAGE_PIXELS = None

# Original prompt (from your config)
ORIGINAL_SYSTEM_PROMPT = """You are an expert photo editor. Analyze the provided photograph and describe the visual adjustments that could be made to enhance or transform it. Focus on actionable editing instructions including lighting adjustments, color corrections, object additions/removals, perspective corrections, lens distortion fixes, and compositional improvements."""

# Improved prompt with examples
IMPROVED_SYSTEM_PROMPT = """You are a professional photo editor with 10+ years of experience in:
- Technical corrections (exposure, white balance, lens distortion)
- Color grading and tonal adjustments
- Composition optimization and cropping
- Retouching and image cleanup

Provide DETAILED, ACTIONABLE editing instructions with:
1. SPECIFIC numerical values (e.g., "+1.5 EV", "Temperature +200K")
2. STEP-BY-STEP sequence (numbered or bulleted)
3. TOOL/PANEL references (e.g., "In HSL panel:", "Use Curves:")
4. CLEAR categories (LIGHTING, COLOR, COMPOSITION, etc.)

EXAMPLE OUTPUT FORMAT:

[CATEGORY: PORTRAIT/LANDSCAPE/PRODUCT]

LIGHTING & EXPOSURE:
- Increase exposure by +1.2 EV
- Lift shadows: +30
- Reduce highlights: -20

COLOR GRADING:
- Temperature: +150K (warmer)
- Tint: +5 (add magenta)
- HSL > Orange: Saturation -10

COMPOSITION:
- Crop to 4:5 ratio for Instagram
- Apply rule of thirds grid

CORRECTIONS:
- Straighten horizon (+2.5 degrees)
- Remove lens distortion (barrel -5)

Now analyze the provided image and give similarly detailed instructions."""

USER_PROMPT = "Analyze this photograph professionally and provide specific editing instructions. Include numerical values, tool references, and step-by-step guidance."


def generate(image_path, model, processor, system_prompt, user_prompt, max_tokens=512):
    """Generate output with given prompts."""
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
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
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


def main():
    parser = argparse.ArgumentParser(description="Test improved prompts")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--image", type=str, required=True, help="Test image path")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--compare_base", action="store_true", help="Also compare with base model")
    parser.add_argument("--max_tokens", type=int, default=512)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TESTING IMPROVED PROMPTS")
    print("=" * 80)
    
    # Load models
    print(f"\nLoading base model: {args.base_model}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()
    
    processor = Qwen2_5_VLProcessor.from_pretrained(args.base_model)
    
    print(f"Loading LoRA adapter: {args.adapter_path}")
    finetuned_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    finetuned_model.eval()
    
    print(f"✓ Models loaded\n")
    
    # Test configurations
    tests = []
    
    if args.compare_base:
        tests.extend([
            ("Base Model + Original Prompt", base_model, ORIGINAL_SYSTEM_PROMPT),
            ("Base Model + Improved Prompt", base_model, IMPROVED_SYSTEM_PROMPT),
        ])
    
    tests.extend([
        ("Fine-tuned + Original Prompt", finetuned_model, ORIGINAL_SYSTEM_PROMPT),
        ("Fine-tuned + Improved Prompt", finetuned_model, IMPROVED_SYSTEM_PROMPT),
    ])
    
    # Generate outputs
    results = {}
    for name, model, system_prompt in tests:
        print(f"Generating with: {name}...")
        output = generate(
            args.image, model, processor, system_prompt, USER_PROMPT, args.max_tokens
        )
        results[name] = output
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    for name, output in results.items():
        print(f"\n{'─' * 80}")
        print(f"📝 {name}")
        print(f"{'─' * 80}")
        print(output)
        print()
    
    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    for name, output in results.items():
        word_count = len(output.split())
        has_numbers = any(char.isdigit() for char in output)
        has_structure = any(marker in output for marker in ['STEP', '1.', '2.', '-', 'LIGHTING', 'COLOR'])
        
        print(f"\n{name}:")
        print(f"  Words: {word_count}")
        print(f"  Has numbers: {'✅' if has_numbers else '❌'}")
        print(f"  Has structure: {'✅' if has_structure else '❌'}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if args.compare_base:
        base_original = results["Base Model + Original Prompt"]
        base_improved = results["Base Model + Improved Prompt"]
        ft_original = results["Fine-tuned + Original Prompt"]
        ft_improved = results["Fine-tuned + Improved Prompt"]
        
        # Check if improved prompt helps
        base_improvement = len(base_improved.split()) - len(base_original.split())
        ft_improvement = len(ft_improved.split()) - len(ft_original.split())
        
        if base_improvement > 20:
            print("\n✅ IMPROVED PROMPT HELPS A LOT!")
            print("   → Better prompts alone significantly improve output quality")
            print("   → Recommendation: Use improved prompts with current model")
            print("   → Consider retraining WITH improved prompts for even better results")
        
        if abs(len(ft_original.split()) - len(base_original.split())) < 10:
            print("\n❌ FINE-TUNING NOT HELPING MUCH")
            print("   → Fine-tuned model is too similar to base model")
            print("   → Recommendations:")
            print("      1. Check training data quality (run analyze_training_data_quality.py)")
            print("      2. Increase LoRA rank (64 → 128)")
            print("      3. Train longer (3 → 5 epochs)")
            print("      4. Use improved config (configs/qwen25vl_prompt_gen_improved.yaml)")
        else:
            print("\n✅ FINE-TUNING IS WORKING")
            print("   → Fine-tuned model produces different outputs than base")
            print("   → Try improved prompts to enhance quality further")
    else:
        ft_original = results["Fine-tuned + Original Prompt"]
        ft_improved = results["Fine-tuned + Improved Prompt"]
        improvement = len(ft_improved.split()) - len(ft_original.split())
        
        if improvement > 20:
            print("\n✅ IMPROVED PROMPT HELPS!")
            print("   → Use improved system prompt for better outputs")
            print("   → No retraining needed for this improvement")
        else:
            print("\n🔔 PROMPT CHANGE HAS MINIMAL EFFECT")
            print("   → Model outputs similar regardless of prompt")
            print("   → Possible issues:")
            print("      1. Model hasn't learned enough from fine-tuning")
            print("      2. Training data quality is low")
            print("   → Next steps: Analyze training data and retrain with improved config")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

