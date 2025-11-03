"""
HTTP client implementations for distributed pipeline components.

Provides clients for:
- vLLM-served vision-language models
- Ray Serve image editing service
- Tiled upscaling using image editor
"""

import asyncio
import base64
import io
import logging
import time
from typing import Optional, Tuple, Dict, Any, Union, List
from PIL import Image
import aiohttp
from openai import AsyncOpenAI

from src.pipeline.base import VisionGenerator, ImageEditor, Upscaler
from src.pipeline.tiling import TiledProcessor
from src.pipeline.registry import register_vision_generator, register_image_editor, register_upscaler

logger = logging.getLogger(__name__)


def encode_image_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def decode_image_base64(image_b64: str) -> Image.Image:
    """Decode base64 string to PIL image."""
    image_data = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_data))


@register_vision_generator("vllm_client")
class VLLMVisionClient(VisionGenerator):
    """
    Client for vLLM-served Qwen2.5-VL vision-language model.
    
    Connects to a vLLM server running the vision model and generates
    editing instructions from images and user prompts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "http://localhost:8000/v1")
        self.api_key = config.get("api_key", "EMPTY")
        self.model = config.get("model", "Qwen/Qwen2.5-VL-72B-Instruct")
        self.system_prompt = config.get(
            "system_prompt",
            "You are a senior photo editor. Given a control image and the user's intent, "
            "produce precise, actionable editing instructions that a diffusion model can follow."
        )
        self.client = None
        self._loaded = False
    
    async def load(self, **config) -> str:
        """Initialize the OpenAI client."""
        self.config.update(config)
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        self._loaded = True
        return f"✅ VLLMVisionClient connected to {self.base_url}"
    
    async def generate(
        self,
        image: Union[Image.Image, List[Image.Image]],
        prompt: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate editing instructions from image(s) and prompt.
        
        Args:
            image: Input image to analyze (single Image or list of Images for comparison)
            prompt: User's editing intent
            **kwargs: max_tokens, temperature, system_prompt, etc.
        """
        if not self._loaded:
            await self.load()
        
        start_time = time.time()
        
        # Encode image(s) for API
        max_edge = kwargs.get("max_edge", 1024)
        jpeg_quality = kwargs.get("jpeg_quality", 90)
        
        # Handle single image or list of images
        images_to_encode = [image] if not isinstance(image, list) else image
        
        # Encode all images to data URLs
        image_urls = []
        for img in images_to_encode:
            # Resize image if needed
            width, height = img.size
            if max(width, height) > max_edge:
                scale = max_edge / max(width, height)
                img = img.resize(
                    (int(width * scale), int(height * scale)),
                    Image.Resampling.LANCZOS
                )
            
            # Encode to data URL
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality)
            image_url = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
            image_urls.append(image_url)
        
        # Build user content with text and image(s)
        user_content = [{"type": "text", "text": prompt}]
        for image_url in image_urls:
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        
        # Use custom system prompt if provided, otherwise use default
        system_prompt = kwargs.get("system_prompt", self.system_prompt)
        
        # Call VL model
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.2),
            )
            
            instructions = response.choices[0].message.content.strip()
            inference_time = time.time() - start_time
            
            metrics = {
                "inference_time": inference_time,
                "model": self.model,
                "tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0,
            }
            
            return instructions, metrics
            
        except Exception as e:
            raise RuntimeError(f"VLLMVisionClient generation failed: {e}")
    
    @property
    def name(self) -> str:
        return "vLLM Vision Client"
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


@register_image_editor("ray_serve_client")
class RayServeImageClient(ImageEditor):
    """
    Client for Ray Serve-deployed Qwen-Image-Edit-Plus service.
    
    Sends HTTP requests to a Ray Serve deployment running the diffusion
    model across multiple GPUs with load balancing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "http://localhost:8001")
        self.timeout_seconds = config.get("timeout_seconds", 900)  # Default: 15 minutes
        self.session: Optional[aiohttp.ClientSession] = None
        self._loaded = False
    
    async def load(self, **config) -> str:
        """Initialize the HTTP session."""
        self.config.update(config)
        # Update timeout if provided
        if "timeout_seconds" in config:
            self.timeout_seconds = config["timeout_seconds"]
        # Don't create session here - it will be created per-request
        # to avoid event loop binding issues
        self._loaded = True
        return f"✅ RayServeImageClient connected to {self.base_url}"
    
    async def edit(
        self,
        image: Union[Image.Image, List[Image.Image]],  # Can be single or list
        prompt: str,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """
        Edit an image using the Ray Serve deployment.
        
        Args:
            image: Input image(s) to edit - can be single Image or list of Images
            prompt: Editing instructions
            negative_prompt: Things to avoid
            **kwargs: num_inference_steps, guidance_scale, seed, width, height
        """
        logger.info("🔧 RayServeImageClient.edit called")
        logger.info(f"   Base URL: {self.base_url}")
        logger.info(f"   Prompt: {prompt[:100]}...")
        
        # Handle single image or list
        if isinstance(image, list):
            logger.info(f"   📥 Multiple images: {[img.size for img in image]}")
            images_list = image
        else:
            logger.info(f"   📥 Single image: {image.size}")
            images_list = [image]
        
        if not self._loaded:
            logger.info("⚙️  Client not loaded, loading now...")
            await self.load()
        
        start_time = time.time()
        
        # Prepare request - encode all images
        logger.info("📦 Encoding image(s) to base64...")
        image_b64_list = [encode_image_base64(img) for img in images_list]
        logger.info(f"   Encoded {len(image_b64_list)} image(s)")
        
        payload = {
            "image": image_b64_list if len(image_b64_list) > 1 else image_b64_list[0],  # Send list if multiple, single if one
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "num_inference_steps": kwargs.get("num_inference_steps", 20),
            "guidance_scale": kwargs.get("guidance_scale", 4.0),
            "seed": kwargs.get("seed", -1),
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
        }
        logger.info(f"📋 Payload prepared: steps={payload['num_inference_steps']}, guidance={payload['guidance_scale']}")
        logger.info(f"   Image field type: {type(payload['image'])} (list={isinstance(payload['image'], list)})")
        
        try:
            # Get timeout from kwargs or use configured default
            request_timeout = kwargs.get("timeout_seconds", self.timeout_seconds)
            logger.info(f"⏱️  Request timeout: {request_timeout}s")
            
            # Create a fresh session for this request to avoid event loop binding issues
            # asyncio.run() creates a new loop per request, so we need a new session each time
            logger.info("🔗 Creating fresh aiohttp session for this request...")
            async with aiohttp.ClientSession() as session:
                logger.info(f"📡 Sending POST request to {self.base_url}/edit...")
                async with session.post(
                    f"{self.base_url}/edit",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=request_timeout)
                ) as response:
                    logger.info(f"📥 Received response: status={response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"❌ Server error: {error_text}")
                        raise RuntimeError(f"Server returned {response.status}: {error_text}")
                    
                    logger.info("📄 Parsing JSON response...")
                    result = await response.json()
                    logger.info(f"✅ JSON parsed, keys: {list(result.keys())}")
                    
                    # Decode result image
                    logger.info("🖼️  Decoding result image...")
                    edited_image = decode_image_base64(result["image"])
                    logger.info(f"✅ Image decoded: size={edited_image.size}, mode={edited_image.mode}")
                    
                    inference_time = time.time() - start_time
                    
                    metrics = {
                        "inference_time": inference_time,
                        "server_time": result.get("processing_time", 0),
                        "gpu_id": result.get("gpu_id", -1),
                    }
                    logger.info(f"📊 Metrics: inference={inference_time:.2f}s, server={metrics['server_time']:.2f}s, GPU={metrics['gpu_id']}")
                    
                    return edited_image, metrics
                
        except Exception as e:
            logger.error(f"❌ RayServeImageClient.edit failed: {e}")
            import traceback
            logger.error(f"   Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"RayServeImageClient edit failed: {e}")
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    @property
    def name(self) -> str:
        return "Ray Serve Image Editor"
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


@register_upscaler("tiled")
class TiledUpscaler(Upscaler):
    """
    Tiled super-resolution using an image editor.
    
    Splits the image into tiles, upscales each tile using the image editor,
    then stitches them back with seam blending.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_editor: Optional[ImageEditor] = None
        self.processor: Optional[TiledProcessor] = None
        self._loaded = False
    
    async def load(self, **config) -> str:
        """Initialize the tiled processor and image editor."""
        self.config.update(config)
        
        # Get tiling configuration
        tile_config = self.config.get("tiling", {})
        self.processor = TiledProcessor(
            tile_width=tile_config.get("tile_width", 512),
            tile_height=tile_config.get("tile_height", 512),
            overlap=tile_config.get("overlap", 64),
            mask_blur=tile_config.get("mask_blur", 8),
        )
        
        # Get image editor instance (should be pre-loaded)
        editor_config = self.config.get("image_editor_config", {})
        if "image_editor_instance" in self.config:
            self.image_editor = self.config["image_editor_instance"]
        else:
            # Create new instance if not provided
            from src.pipeline.registry import get_registry
            registry = get_registry()
            self.image_editor = registry.get_image_editor(
                self.config.get("image_editor_type", "ray_serve_client"),
                editor_config
            )
            await self.image_editor.load()
        
        self._loaded = True
        return "✅ TiledUpscaler initialized with tiled processing"
    
    async def upscale(
        self,
        image: Image.Image,
        factor: float = 2.0,
        prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Upscale image using tiled processing with control image support.
        
        Args:
            image: Input image to upscale
            factor: Upscaling factor
            prompt: Prompt to guide upscaling (e.g., "enhance quality, add detail")
            **kwargs: Additional parameters for the image editor
        """
        if not self._loaded:
            await self.load()
        
        start_time = time.time()
        tiling_config = self.config.get("tiling", {})
        
        # Use default upscaling prompt if not provided
        if prompt is None:
            prompt = tiling_config.get(
                "prompt_template",
                "enhance quality, add detail, sharpen, ultra high resolution"
            )
        
        # First upscale image to target size (this becomes the control base)
        target_width = int(image.width * factor)
        target_height = int(image.height * factor)
        upscaled_base = image.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS
        )
        
        # Check if we should use control images
        use_control = tiling_config.get("use_control_image", True)
        control_max_size = tiling_config.get("control_max_size", 1328)  # New: max size for whole image control
        
        # Prepare full image control (resized to max_size on longer side)
        control_image_full = None
        if use_control:
            # Resize original image to max_size on longer side (maintain aspect ratio)
            original_width, original_height = image.size
            max_size = control_max_size
            
            if max(original_width, original_height) > max_size:
                if original_width > original_height:
                    new_width = max_size
                    new_height = int(original_height * (max_size / original_width))
                else:
                    new_height = max_size
                    new_width = int(original_width * (max_size / original_height))
                
                # Ensure dimensions are multiples of 8 (required by diffusion models)
                new_width = (new_width // 8) * 8
                new_height = (new_height // 8) * 8
                
                control_image_full = image.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS
                )
                logger.info(f"📐 Control image prepared: {image.size} → {control_image_full.size} (max {max_size}px)")
            else:
                # Already under max_size, just ensure multiples of 8
                new_width = (original_width // 8) * 8
                new_height = (original_height // 8) * 8
                if (new_width, new_height) != (original_width, original_height):
                    control_image_full = image.resize(
                        (new_width, new_height),
                        Image.Resampling.LANCZOS
                    )
                else:
                    control_image_full = image.copy()
        
        # Define tile processor function with full image control
        async def process_tile(
            tile: Image.Image,
            tile_info = None,
            **tile_kwargs
        ) -> Image.Image:
            """Process a single tile through the image editor with full image control."""
            
            # Prepare two images: tile crop (to upscale) + full context image
            if use_control and control_image_full is not None:
                # Pass both: [tile crop, full context image]
                images_to_edit = [tile, control_image_full]
                logger.info(f"🔲 Processing tile: {tile.size}")
                logger.info(f"   🎯 Images: tile={tile.size}, full_context={control_image_full.size}")
                
                # Update prompt to instruct upscaling the first image with context from second
                upscale_prompt = f"Upscale and enhance the first image (the crop region) using the style, colors, and context from the second image (the full image). {prompt}"
            else:
                # No control, just use tile
                images_to_edit = [tile]
                upscale_prompt = prompt
                logger.info(f"🔲 Processing tile: {tile.size} (no control image)")
            
            edited, _ = await self.image_editor.edit(
                images_to_edit,  # Pass as list: [tile, full_context] or [tile]
                upscale_prompt,
                negative_prompt=kwargs.get("negative_prompt", ""),
                num_inference_steps=kwargs.get("num_inference_steps", 15),
                guidance_scale=kwargs.get("guidance_scale", 3.5),
                seed=kwargs.get("seed", -1),
                width=tile.width,
                height=tile.height,
            )
            return edited
        
        # Process in tiles with control images
        result = await self.processor.process_tiled_with_control(
            upscaled_base,
            process_tile,
            upscale_factor=1.0,  # Already upscaled, just enhance
            progress_callback=kwargs.get("progress_callback"),
        )
        
        inference_time = time.time() - start_time
        
        metrics = {
            "inference_time": inference_time,
            "upscale_factor": factor,
            "original_size": image.size,
            "output_size": result.size,
            "tiles_processed": len(self.processor.split_into_tiles(upscaled_base)),
            "used_control_images": use_control,
        }
        
        return result, metrics
    
    # def _extract_control_region(
    #     self,
    #     image: Image.Image,
    #     tile_info,
    #     crop_factor: float
    # ) -> Image.Image:
    #     """
    #     Extract a larger region around a tile to use as control image.
    #     DEPRECATED: Now using full image resized to 1328px instead.
    #     """
    #     # Calculate expanded region
    #     tile_center_x = tile_info.x + tile_info.width // 2
    #     tile_center_y = tile_info.y + tile_info.height // 2
        
    #     expanded_width = int(tile_info.width * crop_factor)
    #     expanded_height = int(tile_info.height * crop_factor)
        
    #     # Calculate crop bounds (centered on tile)
    #     crop_x1 = max(0, tile_center_x - expanded_width // 2)
    #     crop_y1 = max(0, tile_center_y - expanded_height // 2)
    #     crop_x2 = min(image.width, crop_x1 + expanded_width)
    #     crop_y2 = min(image.height, crop_y1 + expanded_height)
        
    #     # Extract and resize to tile size
    #     control_region = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    #     control_region = control_region.resize(
    #         (tile_info.width, tile_info.height),
    #         Image.Resampling.LANCZOS
    #     )
        
    #     return control_region
    
    @property
    def name(self) -> str:
        return "Tiled Upscaler"
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


# For backward compatibility with existing code
class LocalImageEditor(ImageEditor):
    """
    Local image editor using QwenImageEditApp directly.
    
    This is for running the model locally without Ray Serve.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = None
        self._loaded = False
    
    async def load(self, **config) -> str:
        """Load the model using QwenImageEditApp."""
        self.config.update(config)
        
        # Import locally to avoid circular dependencies
        from script.gradio_image_edit_app import QwenImageEditApp
        
        self.app = QwenImageEditApp(
            model_name=self.config.get("model_name", "Qwen/Qwen-Image-Edit-Plus"),
            lora_path=self.config.get("lora_path"),
            device=self.config.get("device", "cuda"),
            dtype=self.config.get("dtype", "bfloat16"),
        )
        
        status = self.app.load_model()
        self._loaded = self.app.is_loaded
        return status
    
    async def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Edit image using local model."""
        if not self._loaded:
            return None, {"error": "Model not loaded"}
        
        start_time = time.time()
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result_image, status = await loop.run_in_executor(
            None,
            self.app.generate_image,
            image,
            prompt,
            negative_prompt or "",
            kwargs.get("num_inference_steps", 20),
            kwargs.get("guidance_scale", 4.0),
            kwargs.get("seed", -1),
            kwargs.get("width", 512),
            kwargs.get("height", 512),
        )
        
        inference_time = time.time() - start_time
        metrics = {"inference_time": inference_time}
        
        return result_image, metrics
    
    @property
    def name(self) -> str:
        return "Local Qwen Image Editor"
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded

