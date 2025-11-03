"""
Ray Serve deployment for Qwen-Image-Edit-Plus.

Deploys the diffusion model across multiple GPUs with load balancing,
allowing high-throughput image editing at scale.
"""

import asyncio
import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import sys

import torch
from PIL import Image
from pydantic import BaseModel
from ray import serve
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from diffusers import QwenImageEditPlusPipeline

logger = logging.getLogger(__name__)


class ImageEditRequest(BaseModel):
    """Request model for image editing."""
    image: Union[str, List[str]]  # Base64 encoded image(s) - can be single or list
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 4.0
    seed: int = -1
    width: int = 512
    height: int = 512


class ImageEditResponse(BaseModel):
    """Response model for image editing."""
    image: str  # Base64 encoded result
    processing_time: float
    gpu_id: int
    replica_id: str


def encode_image_base64(image: Image.Image) -> str:
    """Encode PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def decode_image_base64(image_b64: str) -> Image.Image:
    """Decode base64 string to PIL image."""
    image_data = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_data))


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=2,
    # Autoscaling config is overridden by launch_ray_serve.py based on YAML config
)
class QwenImageEditDeployment:
    """
    Ray Serve deployment for Qwen-Image-Edit-Plus model.
    
    Each replica runs on a single GPU and can handle multiple
    concurrent requests with automatic load balancing.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize the deployment with configuration.
        
        Args:
            config_dict: Configuration containing model settings
        """
        self.config_dict = config_dict
        self.trainer = None
        self.replica_id = os.getenv("RAY_SERVE_REPLICA_CONTEXT", "unknown")
        self.gpu_id = -1
        
        # Get model configuration
        self.model_config = config_dict.get("models", {}).get("qwen_image_edit", {})
        self.model_name = self.model_config.get("model_name", "Qwen/Qwen-Image-Edit-Plus")
        self.lora_path = self.model_config.get("lora_path")
        self.dtype = self.model_config.get("dtype", "bfloat16")
        
        logger.info(f"Initializing QwenImageEditDeployment replica {self.replica_id}")
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen-Image-Edit model using diffusers pipeline with multi-GPU sharding."""
        try:
            # Check GPU availability
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, loading on CPU")
                device_map = None
                device = "cpu"
            else:
                # Ray Serve allocates multiple GPUs per replica
                num_gpus = torch.cuda.device_count()
                logger.info(f"Ray assigned {num_gpus} GPU(s) to this replica")
                
                # Use device_map="balanced" to shard model across all available GPUs
                # QwenImageEditPlusPipeline supports "balanced" and "cuda", not "auto"
                device_map = "balanced"
                device = None  # device_map handles placement
                logger.info(f"Using device_map='balanced' to shard model across {num_gpus} GPUs")
            
            # Parse dtype
            if self.dtype == "float16" or self.dtype == "fp16":
                torch_dtype = torch.float16
            elif self.dtype == "bfloat16" or self.dtype == "bf16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            
            # Load pipeline directly from diffusers with low memory mode
            logger.info(f"Loading QwenImageEditPlusPipeline from: {self.model_name}")
            logger.info(f"Device: {device}, dtype: {torch_dtype}")
            
            # Clear any existing CUDA cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Load with maximum memory efficiency and sharding
            logger.info(f"Loading with device_map='{device_map}' for multi-GPU sharding...")
            
            # Try loading without safety checker (might cause black images)
            try:
                self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,  # Load model efficiently
                    device_map=device_map,   # Balanced sharding across GPUs
                    safety_checker=None,     # Disable safety checker (can cause black images)
                    # Note: variant parameter removed - not all models have separate fp16/bf16 files
                )
                logger.info("Loaded with safety_checker=None")
            except TypeError:
                # Model doesn't have safety_checker parameter
                self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map=device_map,
                )
                logger.info("Loaded without safety_checker parameter (not supported)")
            
            # Disable safety checker if it exists (can cause black images)
            if hasattr(self.pipe, 'safety_checker'):
                logger.info("Disabling safety_checker to prevent black images")
                self.pipe.safety_checker = None
            if hasattr(self.pipe, 'feature_extractor'):
                self.pipe.feature_extractor = None
            
            # Only move to device if device_map is not used
            if device_map is None and device:
                logger.info(f"Moving model to {device}...")
                self.pipe = self.pipe.to(device)
            else:
                logger.info("Model sharded across GPUs via device_map")
            
            # Aggressive memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
                
            # Log memory usage for each GPU
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(f"GPU {i} - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
            
            # Load LoRA if specified
            if self.lora_path:
                logger.info(f"Loading LoRA weights from: {self.lora_path}")
                self.pipe.load_lora_weights(self.lora_path)
            
            # Enable memory optimizations for production
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            
            logger.info(f"Model loaded successfully on GPU {self.gpu_id}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def __call__(self, request) -> Dict[str, Any]:
        """
        Handle image editing request.
        
        Args:
            request: HTTP Request object (from Ray Serve/FastAPI)
            
        Returns:
            Image editing response with result
        """
        start_time = time.time()
        
        try:
            # Parse JSON body from HTTP request
            if hasattr(request, 'json'):
                # FastAPI/Starlette Request object
                request_data = await request.json()
            elif isinstance(request, dict):
                # Already parsed JSON
                request_data = request
            elif hasattr(request, 'image'):
                # Pydantic model (backwards compatibility)
                request_data = request.dict()
            else:
                raise ValueError(f"Unsupported request type: {type(request)}")
            
            # Parse into Pydantic model for validation
            parsed_request = ImageEditRequest(**request_data)
            
            # Decode input image(s)
            if isinstance(parsed_request.image, list):
                # Multiple images (for upscaling with context)
                input_images = [decode_image_base64(img_b64) for img_b64 in parsed_request.image]
                logger.info(f"📥 Received {len(input_images)} images: {[img.size for img in input_images]}")
                # Use first image as primary for size calculation
                input_image = input_images[0]
            else:
                # Single image (normal editing)
                input_image = decode_image_base64(parsed_request.image)
                input_images = [input_image]
                logger.info(f"📥 Received single image: {input_image.size}")
            
            # Ensure all images are RGB mode
            for i, img in enumerate(input_images):
                if img.mode != 'RGB':
                    logger.warning(f"Converting image {i} from {img.mode} to RGB")
                    input_images[i] = img.convert('RGB')
            
            # Smart resize: maintain aspect ratio, longest side = 1328 (for first image)
            original_width, original_height = input_image.size
            max_size = 1328
            
            # Resize first image (primary image) if needed
            if max(original_width, original_height) > max_size:
                if original_width > original_height:
                    new_width = max_size
                    new_height = int(original_height * (max_size / original_width))
                else:
                    new_height = max_size
                    new_width = int(original_width * (max_size / original_height))
                
                new_width = (new_width // 8) * 8
                new_height = (new_height // 8) * 8
                
                logger.info(f"Resizing primary image from {input_image.size} to ({new_width}, {new_height})")
                input_images[0] = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                input_image = input_images[0]
            else:
                # Ensure multiples of 8
                new_width = (original_width // 8) * 8
                new_height = (original_height // 8) * 8
                if (new_width, new_height) != (original_width, original_height):
                    logger.info(f"Adjusting to multiple of 8: {input_image.size} -> ({new_width}, {new_height})")
                    input_images[0] = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    input_image = input_images[0]
            
            # Override parsed_request dimensions to match resized primary image
            parsed_request.width = input_image.size[0]
            parsed_request.height = input_image.size[1]
            logger.info(f"Final image size for processing: {parsed_request.width}x{parsed_request.height}")
            logger.info(f"📸 Total images to process: {len(input_images)}")
            
            # Set seed if specified
            if parsed_request.seed >= 0:
                torch.manual_seed(parsed_request.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(parsed_request.seed)
            
            # Run inference using pipeline directly (production-optimized)
            loop = asyncio.get_event_loop()
            
            def run_pipeline():
                # Use the pipeline for inference
                # Note: Qwen-Image-Edit models don't use guidance_scale
                # Generator device should be 'cpu' or first device for sharded models
                generator = None
                if parsed_request.seed >= 0:
                    # Use CPU generator for compatibility with multi-GPU sharding
                    generator = torch.Generator(device='cpu').manual_seed(parsed_request.seed)
                
                logger.info(f"Running pipeline: prompt_len={len(parsed_request.prompt)}, steps={parsed_request.num_inference_steps}")
                logger.info(f"   Images: {[img.size for img in input_images]}")
                logger.info(f"   Output: {parsed_request.width}x{parsed_request.height}")
                
                # Callback for progress monitoring (like ComfyUI K-sampler)
                step_times = []
                last_time = [time.time()]  # Use list for mutable reference
                
                def progress_callback(pipe, step, timestep, callback_kwargs):
                    current_time = time.time()
                    step_duration = current_time - last_time[0]
                    step_times.append(step_duration)
                    last_time[0] = current_time
                    
                    # Log progress every 5 steps
                    if step % 5 == 0 or step == 1:
                        avg_time = sum(step_times) / len(step_times) if step_times else 0
                        remaining_steps = parsed_request.num_inference_steps - step
                        eta = avg_time * remaining_steps
                        logger.info(f"   Step {step}/{parsed_request.num_inference_steps} | "
                                  f"{step_duration:.2f}s/step | ETA: {eta:.0f}s")
                    
                    # Note: Preview image saving is disabled due to FrozenDict config issues
                    # The latent decoding would require complex handling of the VAE config
                    # Instead, we rely on the progress logging with ETA
                    
                    return callback_kwargs
                
                # Pass images to pipeline
                # Note: parameter is 'image' (singular), but pipeline handles lists internally
                output = self.pipe(
                    prompt=parsed_request.prompt,
                    image=input_images,  # Changed from images= to image=
                    num_inference_steps=parsed_request.num_inference_steps,
                    height=parsed_request.height,
                    width=parsed_request.width,
                    generator=generator,
                    output_type='pil',
                    callback_on_step_end=progress_callback,
                )
                
                avg_step_time = sum(step_times) / len(step_times) if step_times else 0
                logger.info(f"Pipeline completed: avg={avg_step_time:.2f}s/step")
                logger.info(f"Pipeline returned: {type(output)}, images={len(output.images) if hasattr(output, 'images') else 'N/A'}")
                return output.images[0]
            
            result_image = await loop.run_in_executor(None, run_pipeline)
            
            if result_image is None:
                raise RuntimeError("No image generated")
            
            # Debug: Check image statistics
            import numpy as np
            img_array = np.array(result_image)
            logger.info(f"Generated image stats: shape={img_array.shape}, dtype={img_array.dtype}")
            logger.info(f"   Values: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}")
            
            # Check for NaN or inf values
            has_nan = np.isnan(img_array).any()
            has_inf = np.isinf(img_array).any()
            if has_nan or has_inf:
                logger.error(f"❌ Image has invalid values: NaN={has_nan}, Inf={has_inf}")
                logger.error(f"   This suggests a model output issue")
                # Try to salvage by clipping
                if has_nan:
                    img_array = np.nan_to_num(img_array, nan=0.0)
                if has_inf:
                    img_array = np.clip(img_array, 0, 255)
                result_image = Image.fromarray(img_array.astype(np.uint8))
                logger.warning("   Attempted to fix by clipping values")
            
            result_b64 = encode_image_base64(result_image)
            logger.info(f"Encoded image size: {len(result_b64)} chars")
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Processed image in {processing_time:.2f}s "
                f"(GPU {self.gpu_id}, replica {self.replica_id})"
            )
            
            response = ImageEditResponse(
                image=result_b64,
                processing_time=processing_time,
                gpu_id=self.gpu_id,
                replica_id=self.replica_id,
            )
            
            # Return as dict for JSON serialization
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "replica_id": self.replica_id,
            "gpu_id": self.gpu_id,
            "model_loaded": self.trainer is not None,
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


async def build_app(config_path: str):
    """
    Build and deploy the Ray Serve application.
    
    Args:
        config_path: Path to production_pipeline.yaml
    """
    # Load configuration
    config = load_config(config_path)
    
    # Get Ray Serve configuration
    ray_config = config.get("ray_serve", {})
    model_config = config.get("models", {}).get("qwen_image_edit", {})
    num_replicas = model_config.get("num_replicas", 6)
    
    # Update deployment configuration
    deployment = QwenImageEditDeployment.options(
        num_replicas=num_replicas,
        max_ongoing_requests=model_config.get("max_concurrent_queries_per_replica", 2),
    )
    
    # Deploy
    serve.run(deployment.bind(config), route_prefix="/edit", name="qwen_image_edit")
    
    logger.info(f"Deployed Qwen-Image-Edit service with {num_replicas} replicas")
    
    return deployment


# For direct execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Qwen-Image-Edit-Plus with Ray Serve")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/production_pipeline.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind to"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the application"
    )
    
    args = parser.parse_args()
    
    # Initialize Ray Serve
    serve.start(
        http_options={"host": args.host, "port": args.port},
        share=args.share
    )
    
    # Deploy application
    asyncio.run(build_app(args.config))
    
    logger.info(f"Ray Serve running at http://{args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop")
    
    # Keep running
    try:
        import signal
        signal.pause()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        serve.shutdown()

