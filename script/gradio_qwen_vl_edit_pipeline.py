#!/usr/bin/env python3
"""
End-to-end Gradio pipeline that:
1) Uses a Qwen2.5-VL endpoint to author edit instructions for an input image.
2) Invokes the Qwen-Image-Edit-Plus model (with optional LoRA) to apply those edits.
3) Sends the generated image through a SUPIR super-resolution stage for 2× upscaling.
"""

import asyncio
import base64
import io
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import torch
from openai import AsyncOpenAI
from PIL import Image

try:
    from diffusers import AutoPipelineForImage2Image
except Exception:  # pragma: no cover - keep the app importable even if diffusers is missing
    AutoPipelineForImage2Image = None

# Ensure the project root is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from script.gradio_image_edit_app import QwenImageEditApp  # noqa: E402  (import after sys.path tweak)

Image.MAX_IMAGE_PIXELS = None


DEFAULT_VL_SYSTEM_PROMPT = (
    "You are a senior photo editor. Given a control image and the user's intent, "
    "produce precise, actionable editing instructions that a diffusion model can follow. "
    "Mention lighting, colors, composition, geometry corrections, and stylistic cues."
)

DEFAULT_VL_USER_TEMPLATE = (
    "User edit request: {edit_request}\n"
    "{extra_context}\n"
    "Describe the exact sequence of edits needed to transform the uploaded image so it satisfies the request. "
    "Write multiple sentences with imperative instructions only."
)

DEFAULT_SUPIR_PROMPT = (
    "ultra sharp, clean edges, natural texture, faithful to the original composition, extremely detailed"
)
DEFAULT_SUPIR_NEGATIVE_PROMPT = "blurry, oversaturated, distorted, artifacts, text, watermark"


def encode_image(image: Image.Image, max_edge: int = 1024, jpeg_quality: int = 90) -> str:
    """Encode a PIL image as a base64 data URL."""
    image = image.convert("RGB")
    width, height = image.size
    largest_edge = max(width, height)
    if largest_edge > max_edge:
        scale = max_edge / float(largest_edge)
        image = image.resize(
            (int(width * scale), int(height * scale)),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


class VisionInstructionGenerator:
    """Thin wrapper around an OpenAI-compatible Qwen2.5-VL endpoint."""

    async def _generate_async(
        self,
        image: Image.Image,
        formatted_prompt: str,
        *,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        max_edge: int,
        jpeg_quality: int,
    ) -> str:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        encoded = await asyncio.to_thread(encode_image, image, max_edge, jpeg_quality)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": formatted_prompt},
                        {"type": "image_url", "image_url": {"url": encoded}},
                    ],
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def generate(
        self,
        image: Optional[Image.Image],
        formatted_prompt: str,
        *,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        max_edge: int = 1024,
        jpeg_quality: int = 90,
    ) -> Tuple[str, str]:
        if image is None:
            return "", "⚠️ Please upload a control image first."
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            prompt = loop.run_until_complete(
                self._generate_async(
                    image,
                    formatted_prompt,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_edge=max_edge,
                    jpeg_quality=jpeg_quality,
                )
            )
            return prompt, "✅ Generated edit instructions with Qwen2.5-VL."
        except Exception as exc:
            return "", f"❌ Qwen2.5-VL prompt generation failed: {exc}"
        finally:
            loop.close()


class ImageEditEngine:
    """Stateful wrapper around the existing QwenImageEditApp helper."""

    def __init__(self):
        self.app: Optional[QwenImageEditApp] = None
        self.current_config: dict = {}

    def load(self, model_name: str, lora_path: str, device: str, dtype: str) -> str:
        normalized_lora = lora_path.strip() if lora_path else ""
        config = {
            "model_name": model_name.strip(),
            "lora_path": normalized_lora,
            "device": device,
            "dtype": dtype,
        }
        if self.app and self.app.is_loaded and self.current_config == config:
            return "✅ Qwen Image Edit model already loaded with this configuration."

        self.app = QwenImageEditApp(
            model_name=config["model_name"],
            lora_path=normalized_lora or None,
            device=device,
            dtype=dtype,
        )
        status = self.app.load_model()
        if status.startswith("✅"):
            self.current_config = config
        return status

    def generate(
        self,
        image: Optional[Image.Image],
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        width: int,
        height: int,
    ) -> Tuple[Optional[Image.Image], str]:
        if not self.app or not self.app.is_loaded:
            return None, "❌ Qwen Image Edit model is not loaded yet."
        return self.app.generate_image(
            input_image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
        )


class SUPIRUpscaler:
    """Loads a SUPIR (or diffusers-compatible) image-to-image pipeline for post-upscaling."""

    def __init__(self):
        self.pipeline = None
        self.current_config: dict = {}

    @staticmethod
    def _torch_dtype(dtype_name: str):
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(dtype_name.lower(), torch.float16)

    def load(
        self,
        model_id: str,
        device: str,
        dtype: str,
        variant: Optional[str] = None,
        enable_tiling: bool = False,
    ) -> str:
        if AutoPipelineForImage2Image is None:
            raise ImportError("diffusers.AutoPipelineForImage2Image is not available. Install/upgrade diffusers.")
        config = {
            "model_id": model_id.strip(),
            "device": device,
            "dtype": dtype,
            "variant": variant or "",
            "enable_tiling": enable_tiling,
        }
        if self.pipeline and self.current_config == config:
            return f"✅ SUPIR already loaded ({model_id}) on {device}."

        pipeline = AutoPipelineForImage2Image.from_pretrained(
            config["model_id"],
            torch_dtype=self._torch_dtype(dtype),
            variant=variant or None,
        )
        if enable_tiling and hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        pipeline.to(device)
        self.pipeline = pipeline
        self.current_config = config
        return f"✅ SUPIR loaded from {model_id} on {device}."

    def upscale(
        self,
        image: Optional[Image.Image],
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        strength: float,
        num_steps: int,
        upscale_factor: float,
    ) -> Image.Image:
        if image is None:
            raise ValueError("No image available to upscale.")
        if not self.pipeline:
            raise RuntimeError("SUPIR pipeline is not loaded.")
        result = self.pipeline(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt or None,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            strength=strength,
        )
        upscaled = result.images[0] if hasattr(result, "images") else result
        if upscale_factor and upscale_factor != 1.0:
            new_size = (
                max(8, int(upscaled.width * upscale_factor)),
                max(8, int(upscaled.height * upscale_factor)),
            )
            upscaled = upscaled.resize(new_size, Image.Resampling.LANCZOS)
        return upscaled


vision_generator = VisionInstructionGenerator()
image_engine = ImageEditEngine()
supir_engine = SUPIRUpscaler()


def format_user_template(template: str, edit_request: str, extra_context: str) -> str:
    template = template or DEFAULT_VL_USER_TEMPLATE
    return (
        template.replace("{edit_request}", edit_request or "Improve the photo.")
        .replace("{extra_context}", extra_context or "")
        .strip()
    )


def load_qwen_model(model_name: str, lora_path: str, device: str, dtype: str) -> str:
    try:
        return image_engine.load(model_name, lora_path, device, dtype)
    except Exception as exc:
        return f"❌ Failed to load Qwen Image Edit model: {exc}"


def load_supir_model(
    model_id: str,
    device: str,
    dtype: str,
    variant: str,
    enable_tiling: bool,
) -> str:
    try:
        return supir_engine.load(model_id, device, dtype, variant or None, enable_tiling)
    except Exception as exc:
        return f"❌ Failed to load SUPIR pipeline: {exc}"


def generate_edit_instructions(
    control_image: Optional[Image.Image],
    edit_request: str,
    extra_context: str,
    prompt_template: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    max_edge: int,
    jpeg_quality: int,
) -> Tuple[str, str, str]:
    formatted_prompt = format_user_template(prompt_template, edit_request, extra_context)
    prompt, status = vision_generator.generate(
        control_image,
        formatted_prompt,
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt or DEFAULT_VL_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        temperature=temperature,
        max_edge=max_edge,
        jpeg_quality=jpeg_quality,
    )
    return prompt, prompt, status


def run_image_editor(
    control_image: Optional[Image.Image],
    manual_prompt: str,
    auto_prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
) -> Tuple[Optional[Image.Image], str]:
    prompt = (manual_prompt or "").strip() or (auto_prompt or "").strip()
    if not prompt:
        return None, "⚠️ Provide a prompt via Qwen-VL or the manual override."
    return image_engine.generate(
        control_image,
        prompt,
        negative_prompt,
        num_steps,
        guidance_scale,
        seed,
        width,
        height,
    )


def run_supir_stage(
    qwen_image: Optional[Image.Image],
    supir_prompt: str,
    supir_negative: str,
    supir_guidance: float,
    supir_strength: float,
    supir_steps: int,
    supir_upscale: float,
    enable_supir: bool,
) -> Tuple[Optional[Image.Image], str]:
    if qwen_image is None:
        return None, "⚠️ Generate an image with Qwen first."
    if not enable_supir:
        return qwen_image, "ℹ️ SUPIR disabled; returning the base Qwen output."

    try:
        upscaled = supir_engine.upscale(
            qwen_image,
            supir_prompt or DEFAULT_SUPIR_PROMPT,
            supir_negative or DEFAULT_SUPIR_NEGATIVE_PROMPT,
            supir_guidance,
            supir_strength,
            supir_steps,
            supir_upscale,
        )
        return upscaled, "✅ SUPIR super-resolution complete."
    except Exception as exc:
        return qwen_image, f"❌ SUPIR failed (showing base image): {exc}"


def run_full_pipeline(
    control_image: Optional[Image.Image],
    edit_request: str,
    extra_context: str,
    prompt_template: str,
    use_manual_prompt: bool,
    manual_prompt: str,
    cached_prompt: str,
    base_url: str,
    api_key: str,
    vl_model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    max_edge: int,
    jpeg_quality: int,
    negative_prompt: str,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
    enable_supir: bool,
    supir_prompt: str,
    supir_negative: str,
    supir_guidance: float,
    supir_strength: float,
    supir_steps: int,
    supir_upscale: float,
) -> Tuple[str, Optional[Image.Image], Optional[Image.Image], str]:
    status_parts = []
    final_prompt = None

    if use_manual_prompt:
        final_prompt = (manual_prompt or "").strip() or (cached_prompt or "").strip()
        if final_prompt:
            status_parts.append("✋ Using manual prompt override.")
        else:
            return cached_prompt, None, None, "❌ Manual override enabled but no prompt text was provided."
    else:
        formatted_prompt = format_user_template(prompt_template, edit_request, extra_context)
        final_prompt, prompt_status = vision_generator.generate(
            control_image,
            formatted_prompt,
            base_url=base_url,
            api_key=api_key,
            model=vl_model,
            system_prompt=system_prompt or DEFAULT_VL_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=temperature,
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
        )
        status_parts.append(prompt_status)
        if not final_prompt:
            return cached_prompt, None, None, prompt_status

    qwen_image, edit_status = image_engine.generate(
        control_image,
        final_prompt,
        negative_prompt,
        num_steps,
        guidance_scale,
        seed,
        width,
        height,
    )
    status_parts.append(edit_status)
    if qwen_image is None:
        return final_prompt, None, None, "\n".join(status_parts)

    supir_image, supir_status = run_supir_stage(
        qwen_image,
        supir_prompt,
        supir_negative,
        supir_guidance,
        supir_strength,
        supir_steps,
        supir_upscale,
        enable_supir,
    )
    status_parts.append(supir_status)
    return final_prompt, qwen_image, supir_image, "\n".join(status_parts)


def create_interface() -> gr.Blocks:
    with gr.Blocks(title="Qwen VL → Qwen Image → SUPIR", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔁 Qwen VL → Qwen Image Edit → SUPIR Pipeline

            1. **Qwen2.5-VL** inspects the reference image plus your intent and writes a detailed edit prompt.
            2. **Qwen-Image-Edit-Plus** (optionally with LoRA) executes that prompt to create the edited image.
            3. **SUPIR** doubles the resolution while preserving fine structure.
            """
        )

        with gr.Accordion("⚙️ Qwen Image Edit Model", open=True):
            with gr.Row():
                model_name = gr.Textbox(
                    label="Model Name or Path",
                    value="Qwen/Qwen-Image-Edit-Plus",
                )
                device = gr.Dropdown(
                    label="Device",
                    choices=["cuda", "cuda:0", "cuda:1", "cpu"],
                    value="cuda" if torch.cuda.is_available() else "cpu",
                )
                dtype = gr.Dropdown(
                    label="Precision",
                    choices=["bfloat16", "float16", "float32"],
                    value="bfloat16",
                )
            lora_path = gr.Textbox(
                label="LoRA Weights (optional)",
                placeholder="path/to/lora.safetensors or HuggingFace repo",
            )
            load_model_btn = gr.Button("🚀 Load Qwen Image Model", variant="primary")
            model_status = gr.Textbox(label="Image Model Status", interactive=False)

        with gr.Accordion("🧩 SUPIR Super Resolution", open=False):
            with gr.Row():
                supir_model_id = gr.Textbox(
                    label="SUPIR Model (Diffusers repo)",
                    value="TencentARC/SUPIR-4X",
                )
                supir_device = gr.Dropdown(
                    label="Device",
                    choices=["cuda", "cuda:0", "cuda:1", "cpu"],
                    value="cuda" if torch.cuda.is_available() else "cpu",
                )
                supir_dtype = gr.Dropdown(
                    label="Precision",
                    choices=["float16", "bfloat16", "float32"],
                    value="float16",
                )
            with gr.Row():
                supir_variant = gr.Textbox(
                    label="Variant (optional)",
                    placeholder="e.g. fp16, bf16",
                )
                supir_enable_tiling = gr.Checkbox(
                    label="Enable VAE tiling (saves memory)", value=True
                )
            load_supir_btn = gr.Button("📈 Load SUPIR", variant="secondary")
            supir_status = gr.Textbox(label="SUPIR Status", interactive=False)

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                control_image = gr.Image(
                    label="Control / Source Image",
                    type="pil",
                    height=400,
                )
                edit_request = gr.Textbox(
                    label="Edit Intent",
                    value="Turn this portrait into a cinematic cyberpunk scene with neon rim lighting.",
                    lines=3,
                )
                extra_context = gr.Textbox(
                    label="Additional Context (optional)",
                    placeholder="Camera metadata, artist references, prohibited changes, etc.",
                    lines=2,
                )
                prompt_template = gr.Textbox(
                    label="Vision-Language Prompt Template",
                    value=DEFAULT_VL_USER_TEMPLATE,
                    lines=4,
                )
                use_manual_prompt = gr.Checkbox(
                    label="Use manual prompt override (skip VL)",
                    value=False,
                )
            with gr.Column(scale=1):
                base_url = gr.Textbox(
                    label="Qwen2.5-VL Base URL",
                    value=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
                )
                api_key = gr.Textbox(
                    label="API Key",
                    value=os.environ.get("VLLM_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")),
                    type="password",
                )
                vl_model = gr.Textbox(
                    label="VL Model",
                    value="Qwen/Qwen2.5-VL-72B-Instruct",
                )
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=DEFAULT_VL_SYSTEM_PROMPT,
                    lines=3,
                )
                temperature = gr.Slider(
                    label="VL Temperature",
                    minimum=0.0,
                    maximum=1.2,
                    value=0.2,
                    step=0.05,
                )
                max_tokens = gr.Slider(
                    label="VL Max Tokens",
                    minimum=64,
                    maximum=2048,
                    value=512,
                    step=32,
                )
                max_edge = gr.Slider(
                    label="Image encode max edge",
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=64,
                )
                jpeg_quality = gr.Slider(
                    label="JPEG encode quality",
                    minimum=50,
                    maximum=100,
                    value=90,
                    step=1,
                )

        with gr.Row():
            generate_prompt_btn = gr.Button("📝 Generate Edit Instructions", variant="primary")
            run_full_pipeline_btn = gr.Button("✨ Run Full Pipeline (VL → Qwen → SUPIR)", variant="primary")

        with gr.Row():
            with gr.Column():
                auto_prompt = gr.Textbox(
                    label="Latest Auto Prompt",
                    lines=8,
                    interactive=False,
                )
                manual_prompt = gr.Textbox(
                    label="Manual Prompt Override (editable)",
                    placeholder="Paste or tweak the generated prompt here...",
                    lines=8,
                )
                prompt_status = gr.Textbox(label="Prompt Status", interactive=False)
            with gr.Column():
                negative_prompt = gr.Textbox(
                    label="Negative Prompt for Qwen",
                    placeholder="undesired elements, artifacts, etc.",
                    lines=3,
                )
                num_steps = gr.Slider(
                    label="Inference Steps",
                    minimum=1,
                    maximum=60,
                    value=24,
                    step=1,
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.1,
                )
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=1024,
                    value=768,
                    step=32,
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=1024,
                    value=768,
                    step=32,
                )
                seed = gr.Slider(
                    label="Seed (-1 for random)",
                    minimum=-1,
                    maximum=2_147_483_647,
                    value=-1,
                    step=1,
                )
                run_editor_btn = gr.Button("🎨 Run Qwen Image Edit Only")
                editor_status = gr.Textbox(label="Image Edit Status", interactive=False)

        gr.Markdown("### Outputs")
        with gr.Row():
            qwen_image = gr.Image(
                label="Qwen Image Edit Output",
                type="pil",
                height=400,
            )
            supir_image = gr.Image(
                label="SUPIR 2× Output",
                type="pil",
                height=400,
            )

        with gr.Accordion("SUPIR Controls", open=False):
            supir_prompt = gr.Textbox(
                label="SUPIR Prompt",
                value=DEFAULT_SUPIR_PROMPT,
                lines=3,
            )
            supir_negative = gr.Textbox(
                label="SUPIR Negative Prompt",
                value=DEFAULT_SUPIR_NEGATIVE_PROMPT,
                lines=2,
            )
            with gr.Row():
                supir_guidance = gr.Slider(
                    label="SUPIR Guidance",
                    minimum=0.1,
                    maximum=10.0,
                    value=4.5,
                    step=0.1,
                )
                supir_strength = gr.Slider(
                    label="SUPIR Strength",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.85,
                    step=0.05,
                )
                supir_steps = gr.Slider(
                    label="SUPIR Steps",
                    minimum=5,
                    maximum=80,
                    value=30,
                    step=1,
                )
                supir_upscale = gr.Slider(
                    label="Upscale Factor",
                    minimum=1.0,
                    maximum=4.0,
                    value=2.0,
                    step=0.1,
                )
            enable_supir = gr.Checkbox(label="Enable SUPIR Super Resolution", value=True)
            run_supir_btn = gr.Button("📈 Run SUPIR on last Qwen output")
            supir_only_status = gr.Textbox(label="SUPIR Run Status", interactive=False)

        pipeline_status = gr.Textbox(label="Pipeline Status", interactive=False, lines=4)

        load_model_btn.click(
            fn=load_qwen_model,
            inputs=[model_name, lora_path, device, dtype],
            outputs=[model_status],
        )

        load_supir_btn.click(
            fn=load_supir_model,
            inputs=[supir_model_id, supir_device, supir_dtype, supir_variant, supir_enable_tiling],
            outputs=[supir_status],
        )

        generate_prompt_btn.click(
            fn=generate_edit_instructions,
            inputs=[
                control_image,
                edit_request,
                extra_context,
                prompt_template,
                base_url,
                api_key,
                vl_model,
                system_prompt,
                temperature,
                max_tokens,
                max_edge,
                jpeg_quality,
            ],
            outputs=[auto_prompt, manual_prompt, prompt_status],
        )

        run_editor_btn.click(
            fn=run_image_editor,
            inputs=[
                control_image,
                manual_prompt,
                auto_prompt,
                negative_prompt,
                num_steps,
                guidance_scale,
                seed,
                width,
                height,
            ],
            outputs=[qwen_image, editor_status],
        )

        run_supir_btn.click(
            fn=run_supir_stage,
            inputs=[
                qwen_image,
                supir_prompt,
                supir_negative,
                supir_guidance,
                supir_strength,
                supir_steps,
                supir_upscale,
                enable_supir,
            ],
            outputs=[supir_image, supir_only_status],
        )

        run_full_pipeline_btn.click(
            fn=run_full_pipeline,
            inputs=[
                control_image,
                edit_request,
                extra_context,
                prompt_template,
                use_manual_prompt,
                manual_prompt,
                auto_prompt,
                base_url,
                api_key,
                vl_model,
                system_prompt,
                temperature,
                max_tokens,
                max_edge,
                jpeg_quality,
                negative_prompt,
                num_steps,
                guidance_scale,
                seed,
                width,
                height,
                enable_supir,
                supir_prompt,
                supir_negative,
                supir_guidance,
                supir_strength,
                supir_steps,
                supir_upscale,
            ],
            outputs=[auto_prompt, qwen_image, supir_image, pipeline_status],
        )

    return demo


def main():
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    main()

