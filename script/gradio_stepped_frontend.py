#!/usr/bin/env python3
"""
Stepped Gradio frontend with separate buttons for each stage.

Flow:
1. Upload image + enter request → Generate Instructions (show prompt)
2. Review prompt → Generate Image (show edited image)
3. Review edited image → Upscale (show upscaled image)
"""

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import gradio as gr
import pandas as pd
import yaml
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.clients import VLLMVisionClient, RayServeImageClient, TiledUpscaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
vl_client = None
edit_client = None
upscale_client = None
config = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


async def initialize_clients_async():
    """Initialize all service clients."""
    global vl_client, edit_client, upscale_client, config
    
    logger.info("Initializing clients...")
    
    # VL Client
    vl_config = {
        "base_url": config['services']['vl_endpoint'],
        "model_name": config['models']['qwen_vl']['model_name'],
    }
    vl_client = VLLMVisionClient(vl_config)
    await vl_client.load()
    logger.info(f"✅ VL Client initialized: {vl_config['base_url']}")
    
    # Edit Client
    edit_config = {
        "base_url": config['services']['image_edit_endpoint'],
        "timeout_seconds": config['pipeline']['edit_defaults'].get('timeout_seconds', 900),
    }
    edit_client = RayServeImageClient(edit_config)
    await edit_client.load()
    logger.info(f"✅ Edit Client initialized: {edit_config['base_url']} (timeout: {edit_config['timeout_seconds']}s)")
    
    # Upscale Client (uses edit client with tiling)
    # Create a separate editor client with longer timeout for upscaling
    upscale_timeout = config['pipeline']['tiling'].get('timeout_seconds', 1800)
    upscale_edit_config = {
        "base_url": config['services']['image_edit_endpoint'],
        "timeout_seconds": upscale_timeout,
    }
    upscale_edit_client = RayServeImageClient(upscale_edit_config)
    await upscale_edit_client.load()
    
    upscale_config = {
        "tiling": {
            "tile_width": 512,
            "tile_height": 512,
            "overlap": 64,
            "mask_blur": 8,
        },
        "image_editor_instance": upscale_edit_client,  # Use dedicated client with longer timeout
        "image_editor_config": upscale_edit_config,
    }
    upscale_client = TiledUpscaler(upscale_config)
    await upscale_client.load()  # Initialize the tiled processor
    logger.info(f"✅ Upscale Client initialized (timeout: {upscale_timeout}s)")
    
    return "✅ All clients initialized!"


def initialize_clients():
    """Sync wrapper for client initialization."""
    return asyncio.run(initialize_clients_async())


async def generate_instructions_async(
    input_image: Image.Image,
    target_image: Optional[Image.Image],
    temperature: float,
    max_tokens: int
) -> Tuple[str, str]:
    """Generate editing instructions using VL model by comparing input and target images."""
    logger.info("🔍 Generating instructions...")
    logger.info(f"   Input image: {input_image.size}")
    if target_image:
        logger.info(f"   Target image: {target_image.size}")
    
    # Professional photo editor system prompt
    system_prompt = """You are a professional photo editor with 10+ years of experience in:
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
[CATEGORY: INTERIOR/EXTERIOR/LANDSCAPE]

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

Now analyze the provided images and give similarly detailed instructions."""
    
    # Build user prompt based on whether target image is provided
    if target_image:
        user_prompt = "Analyze these BEFORE (first image) and AFTER (second image) photographs professionally. Provide specific editing instructions with numerical values, tool references, and step-by-step guidance to transform the BEFORE image into the AFTER image."
        # Encode both images as a list for multi-image comparison
        images_to_send = [input_image, target_image]
    else:
        user_prompt = "Analyze this photograph professionally and provide specific editing instructions. Include numerical values, tool references, and step-by-step guidance."
        images_to_send = input_image
    
    try:
        instructions, metrics = await vl_client.generate(
            images_to_send,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )
        
        logger.info(f"✅ Instructions generated ({metrics['inference_time']:.2f}s)")
        logger.info(f"   Instructions: {instructions[:200]}...")
        
        status = f"✅ Generated in {metrics['inference_time']:.2f}s"
        return instructions, status
        
    except Exception as e:
        logger.error(f"❌ Failed to generate instructions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return "", f"❌ Error: {e}"


def generate_instructions_sync(
    input_image: Optional[Image.Image],
    target_image: Optional[Image.Image],
    temperature: float,
    max_tokens: int
) -> Tuple[str, str]:
    """Sync wrapper for instruction generation."""
    if input_image is None:
        return "", "⚠️ Please upload an input image"
    
    logger.info("=" * 80)
    logger.info("🚀 Step 1: Generate Instructions")
    logger.info(f"   Comparison mode: {'Yes' if target_image else 'No'}")
    
    return asyncio.run(generate_instructions_async(
        input_image, target_image, temperature, max_tokens
    ))


async def edit_image_async(
    image: Image.Image,
    instructions: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    # width and height removed - server auto-resizes
    negative_prompt: str
) -> Tuple[Optional[Image.Image], str]:
    """Edit image using the instructions."""
    logger.info("🎨 Editing image...")
    logger.info(f"   Instructions: {instructions[:100]}...")
    
    try:
        edited_image, metrics = await edit_client.edit(
            image,
            instructions,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            # width and height removed - server determines from input image
        )
        
        logger.info(f"✅ Image edited ({metrics['inference_time']:.2f}s, GPU {metrics.get('gpu_id', 'N/A')})")
        
        status = f"✅ Edited in {metrics['inference_time']:.2f}s (GPU {metrics.get('gpu_id', 'N/A')})"
        return edited_image, status
        
    except Exception as e:
        logger.error(f"❌ Failed to edit image: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"❌ Error: {e}"


def edit_image_sync(
    image: Optional[Image.Image],
    instructions: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    # width and height removed - server auto-resizes
    negative_prompt: str
) -> Tuple[Optional[Image.Image], str]:
    """Sync wrapper for image editing."""
    if image is None:
        return None, "⚠️ Please upload an image"
    
    if not instructions.strip():
        return None, "⚠️ Please generate instructions first"
    
    logger.info("=" * 80)
    logger.info("🚀 Step 2: Edit Image")
    
    return asyncio.run(edit_image_async(
        image, instructions, num_inference_steps, guidance_scale,
        seed, negative_prompt  # width and height removed
    ))


async def upscale_image_async(
    image: Image.Image,
    upscale_factor: float
) -> Tuple[Optional[Image.Image], str]:
    """Upscale the edited image."""
    logger.info("🔍 Upscaling image...")
    logger.info(f"   Input size: {image.size}")
    logger.info(f"   Factor: {upscale_factor}x")
    
    try:
        upscaled_image, metrics = await upscale_client.upscale(
            image,
            factor=upscale_factor,
            num_inference_steps=15,
            guidance_scale=3.5
        )
        
        # Extract time from metrics (use inference_time if total_time not available)
        elapsed_time = metrics.get('total_time', metrics.get('inference_time', 0))
        logger.info(f"✅ Image upscaled ({elapsed_time:.2f}s)")
        logger.info(f"   Output size: {upscaled_image.size}")
        
        status = f"✅ Upscaled in {elapsed_time:.2f}s ({image.size} → {upscaled_image.size})"
        return upscaled_image, status
        
    except Exception as e:
        logger.error(f"❌ Failed to upscale image: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"❌ Error: {e}"


def upscale_image_sync(
    image: Optional[Image.Image],
    upscale_factor: float
) -> Tuple[Optional[Image.Image], str]:
    """Sync wrapper for upscaling."""
    if image is None:
        return None, "⚠️ Please generate edited image first"
    
    logger.info("=" * 80)
    logger.info("🚀 Step 3: Upscale Image")
    
    return asyncio.run(upscale_image_async(image, upscale_factor))


def create_ui():
    """Create the Gradio interface."""
    with gr.Blocks(title="Image Editing Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Image Editing Pipeline (Stepped)")
        gr.Markdown("Edit images in three steps: Generate instructions → Edit image → Upscale")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 1: Input Images")
                input_image = gr.Image(
                    label="Input Image (BEFORE)",
                    type="pil",
                    height=350
                )
                target_image = gr.Image(
                    label="Target/Comparison Image (AFTER) - Optional",
                    type="pil",
                    height=350
                )
                gr.Markdown("💡 *Upload a target image for comparison-based instructions, or leave empty for general analysis*")
                
                with gr.Accordion("VL Parameters", open=False):
                    temperature = gr.Slider(0.0, 1.0, 0.2, label="Temperature")
                    max_tokens = gr.Slider(128, 2048, 1024, label="Max Tokens")
                
                generate_btn = gr.Button("🔍 Generate Instructions", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### Step 2: Generated Instructions")
                instructions_text = gr.Textbox(
                    label="Edit Instructions",
                    lines=10,
                    interactive=True,
                    placeholder="Instructions will appear here..."
                )
                instructions_status = gr.Textbox(label="Status", lines=1)
                
                with gr.Accordion("Edit Parameters", open=True):
                    num_inference_steps = gr.Slider(5, 50, 20, step=1, label="Inference Steps")
                    guidance_scale = gr.Slider(1.0, 10.0, 4.0, label="Guidance Scale")
                    seed = gr.Number(-1, label="Seed (-1 for random)")
                    gr.Markdown("ℹ️ **Output Size:** Automatically matches input image size (max 1328px on longer side, aspect ratio preserved)")
                    negative_prompt = gr.Textbox(
                        "blurry, low quality, distorted, artifacts",
                        label="Negative Prompt",
                        lines=2
                    )
                
                edit_btn = gr.Button("🎨 Generate Edited Image", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 3: Edited Image")
                edited_image = gr.Image(
                    label="Edited Result",
                    type="pil",
                    height=400
                )
                edit_status = gr.Textbox(label="Status", lines=1)
                
                with gr.Accordion("Upscale Parameters", open=True):
                    upscale_factor = gr.Slider(1.5, 4.0, 2.0, step=0.5, label="Upscale Factor")
                
                upscale_btn = gr.Button("🔍 Upscale Image", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### Step 4: Final Result")
                upscaled_image = gr.Image(
                    label="Upscaled Result",
                    type="pil",
                    height=400
                )
                upscale_status = gr.Textbox(label="Status", lines=1)
        
        # Wire up the buttons
        generate_btn.click(
            fn=generate_instructions_sync,
            inputs=[input_image, target_image, temperature, max_tokens],
            outputs=[instructions_text, instructions_status]
        )
        
        edit_btn.click(
            fn=edit_image_sync,
            inputs=[
                input_image, instructions_text, num_inference_steps,
                guidance_scale, seed, negative_prompt
            ],
            outputs=[edited_image, edit_status]
        )
        
        upscale_btn.click(
            fn=upscale_image_sync,
            inputs=[edited_image, upscale_factor],
            outputs=[upscaled_image, upscale_status]
        )
    
    return demo


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stepped Gradio Frontend")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/production_pipeline.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Load config
    global config
    config = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")
    
    # Initialize clients
    logger.info("Initializing service clients...")
    initialize_clients()
    logger.info("✅ All clients ready!")
    
    # Create and launch UI
    frontend_config = config.get('frontend', {})
    demo = create_ui()
    
    logger.info(f"Launching Gradio frontend on {frontend_config['host']}:{frontend_config['port']}")
    logger.info(f"Share enabled: {frontend_config.get('share', False)}")
    
    demo.launch(
        server_name=frontend_config['host'],
        server_port=frontend_config['port'],
        share=frontend_config.get('share', False),
        show_error=True,
    )


if __name__ == "__main__":
    main()

