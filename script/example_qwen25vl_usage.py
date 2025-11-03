#!/usr/bin/env python3
"""
Simple example showing how to use the fine-tuned Qwen2.5-VL prompt generation model.

This demonstrates:
1. Loading the fine-tuned model
2. Generating a prompt from an image
3. Using the prompt with an image editing model
"""

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info


def load_prompt_generation_model(adapter_path: str, base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """Load the fine-tuned prompt generation model."""
    print(f"Loading base model: {base_model}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Loading adapter: {adapter_path}")
    model.load_adapter(adapter_path)
    model.eval()
    
    processor = Qwen2VLProcessor.from_pretrained(base_model)
    
    return model, processor


def generate_edit_prompt(image_path: str, model, processor, max_tokens: int = 512):
    """Generate an editing prompt from an image."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Create chat messages
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an expert photo editor. Analyze the provided photograph and describe the visual adjustments that could be made to enhance or transform it."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the edits needed for this image."},
            ],
        },
    ]
    
    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate
    print("\nGenerating prompt...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
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
    )
    
    return output_text[0]


def main():
    """Example usage."""
    
    # Configuration
    ADAPTER_PATH = "workspace/qwen25vl-prompt-gen-lora"  # Path to your fine-tuned adapter
    IMAGE_PATH = "path/to/your/image.jpg"  # Path to test image
    
    print("="*80)
    print("Qwen2.5-VL Prompt Generation Example")
    print("="*80)
    
    # Load model
    print("\n1. Loading model...")
    model, processor = load_prompt_generation_model(ADAPTER_PATH)
    print("✓ Model loaded successfully")
    
    # Generate prompt
    print(f"\n2. Analyzing image: {IMAGE_PATH}")
    prompt = generate_edit_prompt(IMAGE_PATH, model, processor)
    
    # Display result
    print("\n" + "="*80)
    print("Generated Edit Prompt:")
    print("="*80)
    print(prompt)
    print("="*80)
    
    # Example: Use this prompt with an image editing model
    print("\n3. Next steps:")
    print("   - Use this prompt with Qwen Image Edit Plus or similar models")
    print("   - Apply the suggested edits using photo editing software")
    print("   - Fine-tune further with DPO for better quality")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    # Note: Update ADAPTER_PATH and IMAGE_PATH before running
    print("\nNote: This is an example script. Please update the following:")
    print("  - ADAPTER_PATH: Path to your fine-tuned model")
    print("  - IMAGE_PATH: Path to your test image")
    print("\nThen run: python script/example_qwen25vl_usage.py")
    
    # Uncomment to run:
    # main()

