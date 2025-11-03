#!/usr/bin/env python3
"""
Gradio app for generating edit prompts from custom image pairs using Qwen2.5-VL.
"""

import asyncio
import base64
import io
import os
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from openai import AsyncOpenAI
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert photo editor. Given an original control photograph and the final edited result, "
    "describe the exact visual adjustments that transform the original into the result. Focus on actionable "
    "editing instructions suitable for conditioning an image-edit diffusion model."
)

DEFAULT_USER_PROMPT = (
    "You are provided two images: the first is the control/original input, the second is the final edited output. "
    "Summarize the visual changes needed to convert the input into the output. Mention lighting or color shifts, "
    "object additions/removals, stylistic adjustments, prespective distortion fixes, lens distortion fixes, and composition tweaks. Keep the response deatiled (multiple "
    "sentences) and avoid referring to 'first/second image'."
)


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


async def generate_prompt_async(
    control_img: Image.Image,
    target_img: Image.Image,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    max_edge: int = 1024,
    jpeg_quality: int = 90,
) -> str:
    """Generate edit prompt using Qwen2.5-VL via OpenAI-compatible endpoint."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    # Encode images
    control_b64 = await asyncio.to_thread(encode_image, control_img, max_edge, jpeg_quality)
    target_b64 = await asyncio.to_thread(encode_image, target_img, max_edge, jpeg_quality)
    
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
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
    
    return response.choices[0].message.content.strip()


def generate_prompt_sync(
    control_img: Optional[Image.Image],
    target_img: Optional[Image.Image],
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    max_edge: int,
    jpeg_quality: int,
) -> Tuple[str, str]:
    """Synchronous wrapper for Gradio."""
    if control_img is None or target_img is None:
        return "", "⚠️ Please upload both control (original) and target (edited) images."
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        prompt = loop.run_until_complete(
            generate_prompt_async(
                control_img,
                target_img,
                base_url,
                api_key,
                model,
                system_prompt,
                user_prompt,
                max_tokens,
                temperature,
                max_edge,
                jpeg_quality,
            )
        )
        loop.close()
        return prompt, "✅ Prompt generated successfully!"
    except Exception as exc:
        return "", f"❌ Error: {str(exc)}"


def create_interface() -> gr.Blocks:
    """Create Gradio interface."""
    with gr.Blocks(title="Image Edit Prompt Generator") as demo:
        gr.Markdown(
            """
            # 🎨 Image Edit Prompt Generator
            
            Upload a pair of images (original → edited) to generate a descriptive edit prompt using Qwen2.5-VL.
            
            **How to use:**
            1. Configure your vLLM endpoint settings below
            2. Upload the **original/control** image
            3. Upload the **edited/target** image
            4. Click **Generate Prompt** to get the edit description
            """
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🖼️ Images")
                control_image = gr.Image(
                    label="Control Image (Original)",
                    type="pil",
                    height=300,
                )
                target_image = gr.Image(
                    label="Target Image (Edited)",
                    type="pil",
                    height=300,
                )
                
                generate_btn = gr.Button("🚀 Generate Prompt", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False, lines=1)
            
            with gr.Column():
                gr.Markdown("### 📝 Generated Edit Prompt")
                prompt_output = gr.Textbox(
                    label="Edit Prompt",
                    lines=10,
                    placeholder="The generated prompt will appear here...",
                    show_copy_button=True,
                )
        
        with gr.Accordion("⚙️ Model Configuration", open=False):
            with gr.Row():
                base_url = gr.Textbox(
                    label="Base URL",
                    value=os.environ.get("VLLM_BASE_URL", "http://192.168.0.20:8000/v1"),
                    info="OpenAI-compatible endpoint URL",
                )
                api_key = gr.Textbox(
                    label="API Key",
                    value=os.environ.get("VLLM_API_KEY", "EMPTY"),
                    type="password",
                    info="Authentication key if required",
                )
            
            model = gr.Textbox(
                label="Model Name",
                value="Qwen/Qwen2.5-VL-72B-Instruct",
                info="Model served by vLLM",
            )
        
        with gr.Accordion("🎯 Prompt Configuration", open=False):
            system_prompt = gr.Textbox(
                label="System Prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=3,
                info="System instruction to steer the model",
            )
            user_prompt = gr.Textbox(
                label="User Prompt",
                value=DEFAULT_USER_PROMPT,
                lines=4,
                info="Instruction text sent alongside the images",
            )
        
        with gr.Accordion("🔧 Advanced Settings", open=False):
            with gr.Row():
                max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=128,
                    maximum=4096,
                    value=1024,
                    step=128,
                    info="Maximum tokens to generate",
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.2,
                    step=0.1,
                    info="Sampling temperature",
                )
            
            with gr.Row():
                max_edge = gr.Slider(
                    label="Max Edge Size",
                    minimum=512,
                    maximum=2048,
                    value=1024,
                    step=128,
                    info="Resize longest edge before sending",
                )
                jpeg_quality = gr.Slider(
                    label="JPEG Quality",
                    minimum=50,
                    maximum=100,
                    value=90,
                    step=5,
                    info="JPEG compression quality",
                )
        
        # Event handler
        generate_btn.click(
            fn=generate_prompt_sync,
            inputs=[
                control_image,
                target_image,
                base_url,
                api_key,
                model,
                system_prompt,
                user_prompt,
                max_tokens,
                temperature,
                max_edge,
                jpeg_quality,
            ],
            outputs=[prompt_output, status_text],
        )
        
        gr.Markdown(
            """
            ---
            ### 📌 Notes
            - Make sure your vLLM endpoint is running and accessible
            - Default settings point to remote server: `192.168.0.20:8000`
            - Set `VLLM_BASE_URL` and `VLLM_API_KEY` environment variables to override defaults
            - Ensure the remote machine allows connections on port 8000
            """
        )
    
    return demo


def main():
    """Launch the Gradio app."""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()

