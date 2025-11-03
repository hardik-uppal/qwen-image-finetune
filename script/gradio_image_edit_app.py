#!/usr/bin/env python3
"""
Gradio app for Qwen-Image-Edit-Plus (2509) model inference.
Users can submit an input image with a text prompt and get back an edited/generated image.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import gradio as gr
import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.config import load_config_from_yaml
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer

Image.MAX_IMAGE_PIXELS = None


class QwenImageEditApp:
    """Qwen-Image-Edit-Plus inference application."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-Image-Edit-Plus",
        lora_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        """Initialize the app with model configuration."""
        self.model_name = model_name
        self.lora_path = lora_path
        self.device = device
        self.dtype = dtype
        self.trainer = None
        self.is_loaded = False
        
    def load_model(self, status_callback=None) -> str:
        """Load the model and return status message."""
        if self.is_loaded:
            return "✅ Model already loaded"
        
        try:
            if status_callback:
                status_callback("🔄 Loading model... This may take a few minutes...")
            
            # Create a minimal config for inference
            config_dict = {
                "model": {
                    "class_path": "src.trainer.qwen_image_edit_plus_trainer.QwenImageEditPlusTrainer",
                    "init_args": {
                        "model_name": self.model_name,
                        "device": self.device,
                        "dtype": self.dtype,
                        "gradient_checkpointing": False,
                    }
                }
            }
            
            # Initialize trainer for inference
            from omegaconf import OmegaConf
            config = OmegaConf.create(config_dict)
            
            self.trainer = QwenImageEditPlusTrainer(config)
            self.trainer.setup_predict()
            
            # Load LoRA weights if provided
            if self.lora_path:
                if status_callback:
                    status_callback(f"🔄 Loading LoRA weights from {self.lora_path}...")
                self.trainer.load_lora_weights(self.lora_path)
            
            self.is_loaded = True
            
            device_info = f" on {self.device}" if self.device else ""
            lora_info = f" with LoRA ({self.lora_path})" if self.lora_path else ""
            return f"✅ Model loaded successfully{device_info}{lora_info}"
            
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def generate_image(
        self,
        input_image: Optional[Image.Image],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        seed: int = -1,
        width: int = 512,
        height: int = 512,
    ) -> Tuple[Optional[Image.Image], str]:
        """Generate edited image from input image and prompt."""
        if not self.is_loaded:
            return None, "❌ Model not loaded. Please click 'Load Model' first."
        
        if input_image is None:
            return None, "⚠️ Please upload an input image."
        
        if not prompt.strip():
            return None, "⚠️ Please enter a prompt."
        
        try:
            # Set seed if specified
            if seed >= 0:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            
            # Resize input image if needed
            if input_image.size != (width, height):
                input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Generate image using trainer
            result_images = self.trainer.predict(
                prompt_image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=guidance_scale,
                height=height,
                width=width,
            )
            
            if result_images and len(result_images) > 0:
                return result_images[0], f"✅ Image generated successfully! ({num_inference_steps} steps)"
            else:
                return None, "❌ No image generated"
                
        except Exception as e:
            return None, f"❌ Error during generation: {str(e)}"


# Global app instance
app_instance = None


def initialize_app(
    model_name: str,
    lora_path: str,
    device: str,
    dtype: str,
) -> str:
    """Initialize the application with model configuration."""
    global app_instance
    
    # Clean lora_path
    lora_path = lora_path.strip() if lora_path and lora_path.strip() else None
    
    try:
        app_instance = QwenImageEditApp(
            model_name=model_name,
            lora_path=lora_path,
            device=device,
            dtype=dtype,
        )
        return app_instance.load_model()
    except Exception as e:
        return f"❌ Error initializing app: {str(e)}"


def generate_wrapper(
    input_image: Optional[Image.Image],
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
) -> Tuple[Optional[Image.Image], str]:
    """Wrapper for generate_image to use with Gradio."""
    global app_instance
    
    if app_instance is None:
        return None, "❌ Please initialize the model first by clicking 'Load Model'."
    
    return app_instance.generate_image(
        input_image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        width=width,
        height=height,
    )


def create_interface() -> gr.Blocks:
    """Create Gradio interface."""
    with gr.Blocks(title="Qwen-Image-Edit-Plus App", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 Qwen-Image-Edit-Plus (2509) Image Generation App
            
            Upload an input image and provide editing instructions to generate a new image using the Qwen-Image-Edit-Plus model.
            
            **How to use:**
            1. Configure model settings and click **Load Model** (only needed once)
            2. Upload an input/control image
            3. Enter your editing prompt
            4. Adjust generation parameters if needed
            5. Click **Generate Image** to create the edited result
            """
        )
        
        # Model Configuration Section
        with gr.Accordion("⚙️ Model Configuration", open=True):
            with gr.Row():
                model_name = gr.Textbox(
                    label="Model Name",
                    value="Qwen/Qwen-Image-Edit-Plus",
                    info="HuggingFace model name or local path",
                )
                device = gr.Dropdown(
                    label="Device",
                    choices=["cuda", "cuda:0", "cuda:1", "cpu"],
                    value="cuda" if torch.cuda.is_available() else "cpu",
                    info="Device to run inference on",
                )
            
            with gr.Row():
                dtype = gr.Dropdown(
                    label="Data Type",
                    choices=["bfloat16", "float16", "float32"],
                    value="bfloat16",
                    info="Model precision (bfloat16 recommended)",
                )
                lora_path = gr.Textbox(
                    label="LoRA Weights Path (Optional)",
                    value="",
                    placeholder="Path to LoRA weights or HuggingFace repo (e.g., TsienDragon/qwen-image-edit-plus-lora-face-seg)",
                    info="Leave empty for base model only",
                )
            
            load_model_btn = gr.Button("🚀 Load Model", variant="primary", size="lg")
            load_status = gr.Textbox(label="Model Status", interactive=False, value="Model not loaded")
        
        # Image Generation Section
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")
                input_image = gr.Image(
                    label="Input/Control Image",
                    type="pil",
                    height=400,
                )
                
                prompt = gr.Textbox(
                    label="Editing Prompt",
                    lines=3,
                    placeholder="Describe the edit you want to make...",
                    value="change the image from the face to the face segmentation mask",
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (Optional)",
                    lines=2,
                    placeholder="What you don't want in the image...",
                    value="",
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=400,
                )
                
                generate_btn = gr.Button("✨ Generate Image", variant="primary", size="lg")
                generation_status = gr.Textbox(label="Generation Status", interactive=False)
        
        # Generation Parameters
        with gr.Accordion("🎛️ Generation Parameters", open=False):
            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="Inference Steps",
                    minimum=1,
                    maximum=50,
                    value=20,
                    step=1,
                    info="More steps = better quality but slower",
                )
                guidance_scale = gr.Slider(
                    label="Guidance Scale (CFG)",
                    minimum=1.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.5,
                    info="Higher = more prompt adherence",
                )
            
            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    info="Output image width",
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    info="Output image height",
                )
            
            seed = gr.Slider(
                label="Seed",
                minimum=-1,
                maximum=2147483647,
                value=-1,
                step=1,
                info="Random seed (-1 for random)",
            )
        
        # Example Prompts
        with gr.Accordion("💡 Example Prompts", open=False):
            gr.Markdown(
                """
                **Face Segmentation:**
                - "change the image from the face to the face segmentation mask"
                
                **Style Transfer:**
                - "convert this photo to an oil painting style"
                - "make this image look like a watercolor painting"
                
                **Color Adjustments:**
                - "make the image black and white"
                - "enhance the colors to be more vibrant"
                - "adjust the lighting to golden hour"
                
                **Object Manipulation:**
                - "remove the background and make it white"
                - "add a sunset in the background"
                
                **Quality Enhancement:**
                - "enhance the image quality and sharpness"
                - "remove noise and blur from the image"
                """
            )
        
        # Event handlers
        load_model_btn.click(
            fn=initialize_app,
            inputs=[model_name, lora_path, device, dtype],
            outputs=[load_status],
        )
        
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                num_inference_steps,
                guidance_scale,
                seed,
                width,
                height,
            ],
            outputs=[output_image, generation_status],
        )
        
        # Information Section
        gr.Markdown(
            """
            ---
            ### 📌 Notes
            
            - **First Time Setup**: Click "Load Model" before generating images (this downloads the model if needed)
            - **Memory Requirements**: Requires ~16GB GPU VRAM for base model, less with quantization
            - **LoRA Models**: You can use fine-tuned LoRA weights for specific tasks (e.g., face segmentation, character composition)
            - **Available LoRA Models**:
              - `TsienDragon/qwen-image-edit-plus-lora-face-seg` - Face segmentation
              - Or path to your locally trained LoRA weights
            
            ### 🔧 Model Details
            
            This app uses the Qwen-Image-Edit-Plus (2509) model, which is an enhanced version with:
            - Native multi-image composition support
            - Better edit control and quality
            - Support for various image editing tasks through fine-tuning
            
            For more information, see the [project documentation](../README.md).
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


