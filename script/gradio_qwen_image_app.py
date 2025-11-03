#!/usr/bin/env python3
"""
Gradio app for interacting with Qwen Vision-Language model via vLLM.
Users can submit an image and text prompt to get responses from the model.
"""

import asyncio
import base64
import io
import os
from typing import Optional, Tuple

import gradio as gr
from openai import AsyncOpenAI
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that can analyze images and answer questions about them."
)

DEFAULT_USER_PROMPT = "Describe this image in detail."


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


async def generate_response_async(
    input_img: Image.Image,
    user_prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    max_edge: int = 1024,
    jpeg_quality: int = 90,
) -> str:
    """Generate response using Qwen Vision-Language model via OpenAI-compatible endpoint."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    # Encode image
    image_b64 = await asyncio.to_thread(encode_image, input_img, max_edge, jpeg_quality)
    
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_b64}},
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


def generate_response_sync(
    input_img: Optional[Image.Image],
    user_prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    max_edge: int,
    jpeg_quality: int,
) -> Tuple[str, str]:
    """Synchronous wrapper for Gradio."""
    if input_img is None:
        return "", "⚠️ Please upload an image."
    
    if not user_prompt.strip():
        return "", "⚠️ Please enter a prompt or question."
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(
            generate_response_async(
                input_img,
                user_prompt,
                base_url,
                api_key,
                model,
                system_prompt,
                max_tokens,
                temperature,
                max_edge,
                jpeg_quality,
            )
        )
        loop.close()
        return response, "✅ Response generated successfully!"
    except Exception as exc:
        return "", f"❌ Error: {str(exc)}"


def create_interface() -> gr.Blocks:
    """Create Gradio interface."""
    with gr.Blocks(title="Qwen Vision-Language Model App") as demo:
        gr.Markdown(
            """
            # 🤖 Qwen Vision-Language Model Interface
            
            Upload an image and ask questions or give instructions to the Qwen Vision-Language model served via vLLM.
            
            **How to use:**
            1. Configure your vLLM endpoint settings below
            2. Upload an image
            3. Enter your question or instruction
            4. Click **Generate Response** to get the model's answer
            
            **Example prompts:**
            - "Describe this image in detail"
            - "What objects can you see in this image?"
            - "What is the main subject of this image?"
            - "Analyze the composition and colors in this image"
            """
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🖼️ Input")
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400,
                )
                user_prompt = gr.Textbox(
                    label="Your Question or Instruction",
                    value=DEFAULT_USER_PROMPT,
                    lines=3,
                    placeholder="Enter your question or instruction about the image...",
                )
                
                generate_btn = gr.Button("🚀 Generate Response", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False, lines=1)
            
            with gr.Column():
                gr.Markdown("### 💬 Model Response")
                response_output = gr.Textbox(
                    label="Response",
                    lines=15,
                    placeholder="The model's response will appear here...",
                    show_copy_button=True,
                )
        
        with gr.Accordion("⚙️ Model Configuration", open=True):
            with gr.Row():
                base_url = gr.Textbox(
                    label="vLLM Base URL",
                    value=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
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
                value="Qwen/Qwen2.5-VL-7B-Instruct",
                info="Model served by vLLM",
            )
        
        with gr.Accordion("🎯 Prompt Configuration", open=False):
            system_prompt = gr.Textbox(
                label="System Prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=2,
                info="System instruction to steer the model",
            )
        
        with gr.Accordion("🔧 Advanced Settings", open=False):
            with gr.Row():
                max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=128,
                    maximum=4096,
                    value=2048,
                    step=128,
                    info="Maximum tokens to generate",
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    info="Sampling temperature (higher = more creative)",
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
            fn=generate_response_sync,
            inputs=[
                input_image,
                user_prompt,
                base_url,
                api_key,
                model,
                system_prompt,
                max_tokens,
                temperature,
                max_edge,
                jpeg_quality,
            ],
            outputs=[response_output, status_text],
        )
        
        # Example prompts
        gr.Markdown(
            """
            ---
            ### 📌 Notes
            - Make sure your vLLM endpoint is running and accessible
            - Default settings point to local server: `localhost:8000`
            - Set `VLLM_BASE_URL` and `VLLM_API_KEY` environment variables to override defaults
            - Supported models: Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-72B-Instruct, and other Qwen vision models
            
            ### 🚀 Starting vLLM Server
            
            To start the vLLM server with a Qwen vision model:
            
            ```bash
            # For Qwen2.5-VL-7B-Instruct
            vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000
            
            # For larger models (72B) with more GPUs
            vllm serve Qwen/Qwen2.5-VL-72B-Instruct --host 0.0.0.0 --port 8000 --tensor-parallel-size 4
            ```
            """
        )
    
    return demo


def main():
    """Launch the Gradio app."""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()

