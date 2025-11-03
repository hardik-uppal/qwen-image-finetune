"""
Production pipeline module for distributed image editing.

This module provides abstract base classes and implementations for:
- Vision-language instruction generation (VL models)
- Image editing (diffusion models)
- Super-resolution upscaling (tiled processing)
"""

from src.pipeline.base import VisionGenerator, ImageEditor, Upscaler
from src.pipeline.registry import ComponentRegistry, get_registry

__all__ = [
    "VisionGenerator",
    "ImageEditor",
    "Upscaler",
    "ComponentRegistry",
    "get_registry",
]

