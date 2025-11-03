"""
Abstract base classes for pipeline components.

Defines the interface that all vision generators, image editors,
and upscalers must implement for plug-and-play modularity.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from PIL import Image


class VisionGenerator(ABC):
    """
    Base class for vision-language instruction generators.
    
    These components take an image and user intent, then generate
    detailed editing instructions that can be used by image editors.
    """
    
    @abstractmethod
    async def generate(
        self,
        image: Image.Image,
        prompt: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate editing instructions from image and prompt.
        
        Args:
            image: Input image to analyze
            prompt: User's editing intent/request
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Tuple of (generated_instructions, metrics_dict)
            metrics_dict contains: {"inference_time": float, ...}
        """
        pass
    
    @abstractmethod
    async def load(self, **config) -> str:
        """
        Load/initialize the model.
        
        Args:
            **config: Configuration parameters for the model
            
        Returns:
            Status message string
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the generator."""
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return False


class ImageEditor(ABC):
    """
    Base class for image editing backends.
    
    These components take an image and editing instructions, then
    generate the edited result using diffusion or other models.
    """
    
    @abstractmethod
    async def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """
        Edit an image according to the prompt.
        
        Args:
            image: Input image to edit
            prompt: Editing instructions
            negative_prompt: Things to avoid in the output
            **kwargs: Additional parameters (steps, guidance_scale, seed, etc.)
            
        Returns:
            Tuple of (edited_image, metrics_dict)
            metrics_dict contains: {"inference_time": float, "gpu_id": int, ...}
        """
        pass
    
    @abstractmethod
    async def load(self, **config) -> str:
        """
        Load/initialize the model.
        
        Args:
            **config: Configuration parameters for the model
            
        Returns:
            Status message string
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the editor."""
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return False


class Upscaler(ABC):
    """
    Base class for super-resolution/upscaling backends.
    
    These components enhance image resolution while preserving
    or improving quality using various techniques.
    """
    
    @abstractmethod
    async def upscale(
        self,
        image: Image.Image,
        factor: float = 2.0,
        prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Upscale an image by the specified factor.
        
        Args:
            image: Input image to upscale
            factor: Upscaling factor (2.0 = 2x resolution)
            prompt: Optional prompt to guide upscaling
            **kwargs: Additional parameters (tile_size, strength, etc.)
            
        Returns:
            Tuple of (upscaled_image, metrics_dict)
            metrics_dict contains: {"inference_time": float, "tiles_processed": int, ...}
        """
        pass
    
    @abstractmethod
    async def load(self, **config) -> str:
        """
        Load/initialize the upscaling model.
        
        Args:
            **config: Configuration parameters
            
        Returns:
            Status message string
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the upscaler."""
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return False


class PipelineComponent(ABC):
    """
    Base class for any pipeline component with common utilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def _update_config(self, **kwargs):
        """Update configuration with new parameters."""
        self.config.update(kwargs)

