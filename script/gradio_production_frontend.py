#!/usr/bin/env python3
"""
Production Gradio frontend for distributed image editing pipeline.

Features:
- Image comparison with before/after slider
- Per-stage metrics tracking
- Batch processing
- A/B testing mode
- Export results
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
from typing import Optional, Tuple, List, Dict, Any

import gradio as gr
import pandas as pd
import yaml
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.clients import VLLMVisionClient, RayServeImageClient, TiledUpscaler
from src.pipeline.registry import get_registry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and tracks pipeline metrics."""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
    
    def track_request(
        self,
        stage: str,
        duration: float,
        **kwargs
    ):
        """Track a pipeline stage execution."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "duration_seconds": round(duration, 2),
            **kwargs
        }
        self.metrics_history.append(entry)
    
    def get_summary(self) -> pd.DataFrame:
        """Get metrics summary as DataFrame."""
        if not self.metrics_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.metrics_history)
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        
        stats = {
            "total_requests": len(df),
            "avg_vl_time": df[df['stage'] == 'vl_generation']['duration_seconds'].mean() if 'vl_generation' in df['stage'].values else 0,
            "avg_edit_time": df[df['stage'] == 'image_editing']['duration_seconds'].mean() if 'image_editing' in df['stage'].values else 0,
            "avg_upscale_time": df[df['stage'] == 'upscaling']['duration_seconds'].mean() if 'upscaling' in df['stage'].values else 0,
            "total_time": df['duration_seconds'].sum(),
        }
        
        return stats
    
    def export_logs(self, path: str):
        """Export metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class ProductionPipeline:
    """Production image editing pipeline with distributed services."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.metrics = MetricsCollector()
        
        # Initialize clients
        vl_config = {
            "base_url": self.config['services']['vl_endpoint'],
            "api_key": self.config['models']['qwen_vl']['api_key'],
            "model": self.config['models']['qwen_vl']['model_name'],
            "system_prompt": self.config['models']['qwen_vl'].get('system_prompt', ''),
        }
        self.vl_client = VLLMVisionClient(vl_config)
        
        edit_config = {
            "base_url": self.config['services']['image_edit_endpoint'],
        }
        self.edit_client = RayServeImageClient(edit_config)
        
        upscale_config = {
            "tiling": self.config['pipeline']['tiling'],
            "image_editor_instance": self.edit_client,
        }
        self.upscaler = TiledUpscaler(upscale_config)
    
    async def initialize(self) -> str:
        """Initialize all clients."""
        status_parts = []
        
        try:
            status = await self.vl_client.load()
            status_parts.append(status)
        except Exception as e:
            status_parts.append(f"❌ VL Client: {e}")
        
        try:
            status = await self.edit_client.load()
            status_parts.append(status)
        except Exception as e:
            status_parts.append(f"❌ Edit Client: {e}")
        
        try:
            status = await self.upscaler.load()
            status_parts.append(status)
        except Exception as e:
            status_parts.append(f"❌ Upscaler: {e}")
        
        return "\n".join(status_parts)
    
    async def generate_instructions(
        self,
        image: Image.Image,
        edit_request: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate editing instructions from VL model."""
        start_time = time.time()
        
        instructions, metrics = await self.vl_client.generate(
            image,
            edit_request,
            **kwargs
        )
        
        duration = time.time() - start_time
        self.metrics.track_request(
            stage="vl_generation",
            duration=duration,
            **metrics
        )
        
        return instructions, metrics
    
    async def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        **kwargs
    ) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Edit image using distributed service."""
        logger.info("🎨 edit_image called")
        logger.info(f"   Prompt: {prompt[:100]}...")
        logger.info(f"   Image size: {image.size}")
        logger.info(f"   Kwargs: {list(kwargs.keys())}")
        
        start_time = time.time()
        
        logger.info("📤 Calling edit_client.edit()...")
        try:
            edited_image, metrics = await self.edit_client.edit(
                image,
                prompt,
                **kwargs
            )
            logger.info(f"✅ edit_client.edit() returned successfully")
            logger.info(f"   Edited image size: {edited_image.size if edited_image else 'None'}")
        except Exception as e:
            logger.error(f"❌ edit_client.edit() failed: {e}")
            import traceback
            logger.error(f"   Traceback:\n{traceback.format_exc()}")
            raise
        
        duration = time.time() - start_time
        self.metrics.track_request(
            stage="image_editing",
            duration=duration,
            **metrics
        )
        
        return edited_image, metrics
    
    async def upscale_image(
        self,
        image: Image.Image,
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Upscale image using tiled processing."""
        start_time = time.time()
        
        upscaled, metrics = await self.upscaler.upscale(
            image,
            **kwargs
        )
        
        duration = time.time() - start_time
        self.metrics.track_request(
            stage="upscaling",
            duration=duration,
            **metrics
        )
        
        return upscaled, metrics
    
    async def run_full_pipeline(
        self,
        image: Image.Image,
        edit_request: str,
        enable_upscale: bool = True,
        **kwargs
    ) -> Tuple[str, Optional[Image.Image], Optional[Image.Image], str, pd.DataFrame]:
        """Run the full pipeline end-to-end."""
        logger.info("📋 Pipeline.run_full_pipeline called")
        logger.info(f"   Image size: {image.size}")
        logger.info(f"   Enable upscale: {enable_upscale}")
        
        status_parts = []
        
        # Step 1: Generate instructions
        try:
            logger.info("🔍 Step 1: Calling generate_instructions...")
            instructions, vl_metrics = await self.generate_instructions(
                image,
                edit_request,
                **kwargs.get('vl_params', {})
            )
            logger.info(f"✅ Instructions generated: {instructions[:100]}...")
            status_parts.append(f"✅ Generated instructions ({vl_metrics['inference_time']:.2f}s)")
        except Exception as e:
            logger.error(f"❌ VL generation failed: {e}")
            import traceback
            logger.error(f"   Traceback:\n{traceback.format_exc()}")
            return "", None, None, f"❌ VL generation failed: {e}", pd.DataFrame()
        
        # Step 2: Edit image
        try:
            logger.info("🎨 Step 2: Calling edit_image...")
            edited_image, edit_metrics = await self.edit_image(
                image,
                instructions,
                **kwargs.get('edit_params', {})
            )
            logger.info(f"✅ Image edited, size: {edited_image.size}")
            status_parts.append(
                f"✅ Image edited ({edit_metrics['inference_time']:.2f}s, "
                f"GPU {edit_metrics.get('gpu_id', 'N/A')})"
            )
        except Exception as e:
            return instructions, None, None, f"{chr(10).join(status_parts)}\n❌ Editing failed: {e}", pd.DataFrame()
        
        # Step 3: Upscale (optional)
        upscaled_image = None
        if enable_upscale and edited_image:
            try:
                upscaled_image, upscale_metrics = await self.upscale_image(
                    edited_image,
                    **kwargs.get('upscale_params', {})
                )
                status_parts.append(
                    f"✅ Image upscaled {upscale_metrics['upscale_factor']}× "
                    f"({upscale_metrics['inference_time']:.2f}s, "
                    f"{upscale_metrics['tiles_processed']} tiles)"
                )
            except Exception as e:
                status_parts.append(f"⚠️ Upscaling failed: {e}")
                upscaled_image = edited_image
        
        # Get metrics summary
        metrics_df = self.metrics.get_summary()
        
        status = "\n".join(status_parts)
        
        return instructions, edited_image, upscaled_image, status, metrics_df


# Global pipeline instance
pipeline: Optional[ProductionPipeline] = None


def initialize_pipeline(config_path: str) -> str:
    """Initialize the pipeline."""
    global pipeline
    
    try:
        pipeline = ProductionPipeline(config_path)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        status = loop.run_until_complete(pipeline.initialize())
        loop.close()
        return status
    except Exception as e:
        return f"❌ Initialization failed: {e}"


def run_full_pipeline_sync(
    input_image: Optional[Image.Image],
    edit_request: str,
    enable_upscale: bool,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
    negative_prompt: str,
    upscale_factor: float,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Optional[Image.Image], Optional[Image.Image], str, pd.DataFrame]:
    """Synchronous wrapper for the full pipeline."""
    global pipeline
    
    if pipeline is None:
        return "", None, None, "❌ Pipeline not initialized", pd.DataFrame()
    
    if input_image is None:
        return "", None, None, "⚠️ Please upload an image", pd.DataFrame()
    
    if not edit_request.strip():
        return "", None, None, "⚠️ Please enter an edit request", pd.DataFrame()
    
    # Prepare parameters
    vl_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    edit_params = {
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "width": width,
        "height": height,
        "negative_prompt": negative_prompt,
    }
    
    upscale_params = {
        "factor": upscale_factor,
        "num_inference_steps": 15,
        "guidance_scale": 3.5,
    }
    
    # Run pipeline in a new event loop (safer for Gradio threads)
    # Gradio runs handlers in threads, so we need to create a fresh event loop
    logger.info("=" * 80)
    logger.info("🚀 Starting run_full_pipeline_sync")
    logger.info(f"   Input image size: {input_image.size}")
    logger.info(f"   Edit request: {edit_request[:100]}...")
    logger.info(f"   Thread ID: {threading.current_thread().ident}")
    logger.info(f"   Thread name: {threading.current_thread().name}")
    
    try:
        logger.info("🔄 Attempting asyncio.run() method...")
        # asyncio.run() creates a new event loop, runs the coroutine, and closes it
        result = asyncio.run(
            pipeline.run_full_pipeline(
                input_image,
                edit_request,
                enable_upscale=enable_upscale,
                vl_params=vl_params,
                edit_params=edit_params,
                upscale_params=upscale_params,
            )
        )
        logger.info("✅ asyncio.run() succeeded")
        return result
    except RuntimeError as e:
        logger.warning(f"⚠️  asyncio.run() failed with RuntimeError: {e}")
        # If there's already a running event loop, use run_in_executor
        if "cannot be called from a running event loop" in str(e):
            logger.info("🔄 Trying fallback: new event loop with run_until_complete...")
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    pipeline.run_full_pipeline(
                        input_image,
                        edit_request,
                        enable_upscale=enable_upscale,
                        vl_params=vl_params,
                        edit_params=edit_params,
                        upscale_params=upscale_params,
                    )
                )
                logger.info("✅ Fallback method succeeded")
                return result
            except Exception as inner_e:
                logger.error(f"❌ Fallback method failed: {inner_e}")
                raise
            finally:
                logger.info("🔒 Closing event loop...")
                loop.close()
                logger.info("✅ Event loop closed")
        else:
            logger.error(f"❌ Unexpected RuntimeError: {e}")
            raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in run_full_pipeline_sync: {e}")
        logger.error(f"   Error type: {type(e)}")
        import traceback
        logger.error(f"   Traceback:\n{traceback.format_exc()}")
        raise


def export_metrics_sync(filename: str) -> str:
    """Export metrics to file."""
    global pipeline
    
    if pipeline is None:
        return "❌ No metrics to export"
    
    try:
        output_path = f"outputs/metrics/{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pipeline.metrics.export_logs(output_path)
        return f"✅ Metrics exported to: {output_path}"
    except Exception as e:
        return f"❌ Export failed: {e}"


def create_interface(config_path: str) -> gr.Blocks:
    """Create the production Gradio interface."""
    
    with gr.Blocks(title="Production Image Editing Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🚀 Production Image Editing Pipeline
            
            **Distributed system with:**
            - Qwen2.5-VL for instruction generation (vLLM, 2 GPUs)
            - Qwen-Image-Edit-Plus for editing (Ray Serve, 6 GPUs)
            - Tiled super-resolution for 2× upscaling
            """
        )
        
        # Initialization
        with gr.Row():
            init_btn = gr.Button("🔄 Initialize Pipeline", variant="primary", size="lg")
            init_status = gr.Textbox(label="Initialization Status", lines=3, interactive=False)
        
        init_btn.click(
            fn=lambda: initialize_pipeline(config_path),
            outputs=[init_status]
        )
        
        gr.Markdown("---")
        
        # Main pipeline interface
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")
                input_image = gr.Image(label="Input Image", type="pil", height=400)
                edit_request = gr.Textbox(
                    label="Edit Request",
                    lines=3,
                    placeholder="Describe what you want to do with the image...",
                    value="Transform this into a cinematic scene with dramatic lighting"
                )
                
                with gr.Accordion("⚙️ Generation Parameters", open=False):
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            label="Inference Steps", minimum=1, maximum=50, value=20, step=1
                        )
                        guidance_scale = gr.Slider(
                            label="Guidance Scale", minimum=1.0, maximum=10.0, value=4.0, step=0.1
                        )
                    with gr.Row():
                        width = gr.Slider(label="Width", minimum=256, maximum=1024, value=768, step=64)
                        height = gr.Slider(label="Height", minimum=256, maximum=1024, value=768, step=64)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, value=-1, step=1)
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="blurry, low quality, distorted, artifacts",
                        lines=2
                    )
                
                with gr.Accordion("🔍 VL Model Parameters", open=False):
                    temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.2, step=0.05)
                    max_tokens = gr.Slider(label="Max Tokens", minimum=64, maximum=2048, value=512, step=32)
                
                with gr.Accordion("📈 Upscaling", open=False):
                    enable_upscale = gr.Checkbox(label="Enable Upscaling", value=True)
                    upscale_factor = gr.Slider(label="Upscale Factor", minimum=1.0, maximum=4.0, value=2.0, step=0.1)
                
                run_btn = gr.Button("✨ Run Full Pipeline", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")
                
                with gr.Tabs():
                    with gr.Tab("Edited Image"):
                        edited_image = gr.Image(label="Edited Result", type="pil", height=400)
                    with gr.Tab("Upscaled Image"):
                        upscaled_image = gr.Image(label="Upscaled Result", type="pil", height=400)
                    with gr.Tab("Comparison"):
                        comparison = gr.Image(label="Before → After", type="pil", height=400)
                
                generated_instructions = gr.Textbox(
                    label="Generated Instructions",
                    lines=6,
                    interactive=False
                )
        
        # Status and metrics
        pipeline_status = gr.Textbox(label="Pipeline Status", lines=4, interactive=False)
        
        with gr.Accordion("📊 Metrics Dashboard", open=True):
            metrics_df = gr.Dataframe(
                label="Execution Metrics",
                headers=["timestamp", "stage", "duration_seconds"],
                interactive=False
            )
            
            with gr.Row():
                export_filename = gr.Textbox(label="Export Filename", value="pipeline_metrics")
                export_btn = gr.Button("💾 Export Metrics")
                export_status = gr.Textbox(label="Export Status", interactive=False)
        
        # Wire up the pipeline
        run_btn.click(
            fn=run_full_pipeline_sync,
            inputs=[
                input_image,
                edit_request,
                enable_upscale,
                num_inference_steps,
                guidance_scale,
                seed,
                width,
                height,
                negative_prompt,
                upscale_factor,
                temperature,
                max_tokens,
            ],
            outputs=[
                generated_instructions,
                edited_image,
                upscaled_image,
                pipeline_status,
                metrics_df
            ]
        )
        
        export_btn.click(
            fn=export_metrics_sync,
            inputs=[export_filename],
            outputs=[export_status]
        )
        
        gr.Markdown(
            """
            ---
            ### 📝 Usage Tips
            
            1. **Initialize** the pipeline first (connects to services)
            2. **Upload** an input image
            3. **Describe** your desired edit
            4. **Run** the full pipeline
            5. **Compare** results in different tabs
            6. **Export** metrics for analysis
            
            ### 🔧 System Info
            
            - **VL Service**: 2 GPUs (tensor parallel)
            - **Edit Service**: 6 GPUs (load balanced)
            - **Max Concurrent**: 12-24 requests
            
            For configuration changes, edit `configs/production_pipeline.yaml` and restart services.
            """
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="Production Gradio Frontend")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/production_pipeline.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load config to get frontend settings
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    frontend_config = config['frontend']
    
    # Create interface
    demo = create_interface(args.config)
    
    # Launch
    demo.queue(max_size=frontend_config.get('max_concurrent_requests', 24))
    
    logger.info(f"Launching Gradio frontend on {frontend_config['host']}:{frontend_config['port']}")
    logger.info(f"Share enabled: {frontend_config.get('share', False)}")
    
    # Launch and capture the share link
    app, local_url, share_url = demo.launch(
        server_name=frontend_config['host'],
        server_port=frontend_config['port'],
        share=frontend_config.get('share', False),
        show_error=True,
        prevent_thread_lock=False,  # Block to keep server running
    )
    
    # Print URLs to logs
    logger.info(f"Local URL: {local_url}")
    if share_url:
        logger.info(f"Public URL: {share_url}")
        print(f"GRADIO_SHARE_LINK: {share_url}")  # Special marker for extraction


if __name__ == "__main__":
    main()

