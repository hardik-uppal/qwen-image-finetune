"""
Component registry and factory for pipeline components.

Provides a centralized registry for all vision generators, image editors,
and upscalers, allowing dynamic component instantiation and swapping.
"""

from typing import Dict, Type, List, Optional, Any
from src.pipeline.base import VisionGenerator, ImageEditor, Upscaler


class ComponentRegistry:
    """
    Registry for pipeline components.
    
    Allows registration and retrieval of vision generators, image editors,
    and upscalers by name.
    """
    
    def __init__(self):
        self._vision_generators: Dict[str, Type[VisionGenerator]] = {}
        self._image_editors: Dict[str, Type[ImageEditor]] = {}
        self._upscalers: Dict[str, Type[Upscaler]] = {}
    
    def register_vision_generator(
        self,
        name: str,
        cls: Type[VisionGenerator]
    ) -> None:
        """Register a vision generator class."""
        if not issubclass(cls, VisionGenerator):
            raise TypeError(f"{cls} must be a subclass of VisionGenerator")
        self._vision_generators[name] = cls
    
    def register_image_editor(
        self,
        name: str,
        cls: Type[ImageEditor]
    ) -> None:
        """Register an image editor class."""
        if not issubclass(cls, ImageEditor):
            raise TypeError(f"{cls} must be a subclass of ImageEditor")
        self._image_editors[name] = cls
    
    def register_upscaler(
        self,
        name: str,
        cls: Type[Upscaler]
    ) -> None:
        """Register an upscaler class."""
        if not issubclass(cls, Upscaler):
            raise TypeError(f"{cls} must be a subclass of Upscaler")
        self._upscalers[name] = cls
    
    def get_vision_generator(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> VisionGenerator:
        """
        Get an instance of a vision generator by name.
        
        Args:
            name: Registered name of the generator
            config: Configuration dict to pass to constructor
            
        Returns:
            Instance of the vision generator
        """
        if name not in self._vision_generators:
            raise ValueError(
                f"Vision generator '{name}' not found. "
                f"Available: {list(self._vision_generators.keys())}"
            )
        cls = self._vision_generators[name]
        return cls(config or {})
    
    def get_image_editor(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> ImageEditor:
        """
        Get an instance of an image editor by name.
        
        Args:
            name: Registered name of the editor
            config: Configuration dict to pass to constructor
            
        Returns:
            Instance of the image editor
        """
        if name not in self._image_editors:
            raise ValueError(
                f"Image editor '{name}' not found. "
                f"Available: {list(self._image_editors.keys())}"
            )
        cls = self._image_editors[name]
        return cls(config or {})
    
    def get_upscaler(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Upscaler:
        """
        Get an instance of an upscaler by name.
        
        Args:
            name: Registered name of the upscaler
            config: Configuration dict to pass to constructor
            
        Returns:
            Instance of the upscaler
        """
        if name not in self._upscalers:
            raise ValueError(
                f"Upscaler '{name}' not found. "
                f"Available: {list(self._upscalers.keys())}"
            )
        cls = self._upscalers[name]
        return cls(config or {})
    
    def list_available(self) -> Dict[str, List[str]]:
        """
        List all available components.
        
        Returns:
            Dict with keys: 'vision_generators', 'image_editors', 'upscalers'
        """
        return {
            "vision_generators": list(self._vision_generators.keys()),
            "image_editors": list(self._image_editors.keys()),
            "upscalers": list(self._upscalers.keys()),
        }


# Global registry instance
_registry = ComponentRegistry()


def get_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _registry


def register_vision_generator(name: str):
    """Decorator to register a vision generator class."""
    def decorator(cls: Type[VisionGenerator]):
        _registry.register_vision_generator(name, cls)
        return cls
    return decorator


def register_image_editor(name: str):
    """Decorator to register an image editor class."""
    def decorator(cls: Type[ImageEditor]):
        _registry.register_image_editor(name, cls)
        return cls
    return decorator


def register_upscaler(name: str):
    """Decorator to register an upscaler class."""
    def decorator(cls: Type[Upscaler]):
        _registry.register_upscaler(name, cls)
        return cls
    return decorator

