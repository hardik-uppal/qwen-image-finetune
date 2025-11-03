"""
Tiled image processing utilities for handling large images.

Based on ComfyUI Ultimate SD Upscale approach with seam blending
to avoid visible artifacts at tile boundaries.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Any
from PIL import Image, ImageFilter, ImageDraw
import numpy as np


@dataclass
class TileInfo:
    """Information about a tile's position and size."""
    x: int  # X position in the source image
    y: int  # Y position in the source image
    width: int  # Tile width
    height: int  # Tile height
    row: int  # Row index
    col: int  # Column index
    total_rows: int  # Total number of rows
    total_cols: int  # Total number of columns


class TiledProcessor:
    """
    Tile-based image processing with seam blending.
    
    Splits large images into overlapping tiles, processes them individually,
    then stitches them back together with smooth blending at seams.
    """
    
    def __init__(
        self,
        tile_width: int = 512,
        tile_height: int = 512,
        overlap: int = 64,
        mask_blur: int = 8,
    ):
        """
        Initialize the tiled processor.
        
        Args:
            tile_width: Width of each tile
            tile_height: Height of each tile
            overlap: Overlap between adjacent tiles (in pixels)
            mask_blur: Blur radius for seam blending masks
        """
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.overlap = overlap
        self.mask_blur = mask_blur
    
    def split_into_tiles(
        self,
        image: Image.Image,
        force_uniform: bool = True
    ) -> List[Tuple[Image.Image, TileInfo]]:
        """
        Split an image into overlapping tiles.
        
        Args:
            image: Input image to split
            force_uniform: If True, ensure all tiles have uniform size
            
        Returns:
            List of (tile_image, tile_info) tuples
        """
        img_width, img_height = image.size
        
        # Calculate number of tiles needed
        cols = math.ceil((img_width - self.overlap) / (self.tile_width - self.overlap))
        rows = math.ceil((img_height - self.overlap) / (self.tile_height - self.overlap))
        
        tiles = []
        
        for row in range(rows):
            for col in range(cols):
                # Calculate tile position
                x = col * (self.tile_width - self.overlap)
                y = row * (self.tile_height - self.overlap)
                
                # Calculate tile size
                if force_uniform:
                    # Expand tile to maintain uniform size
                    tile_w = self.tile_width
                    tile_h = self.tile_height
                    
                    # Adjust position if tile would go beyond image bounds
                    if x + tile_w > img_width:
                        x = max(0, img_width - tile_w)
                    if y + tile_h > img_height:
                        y = max(0, img_height - tile_h)
                    
                    # Clamp tile size to image bounds
                    tile_w = min(tile_w, img_width - x)
                    tile_h = min(tile_h, img_height - y)
                else:
                    # Use minimal tile size at edges
                    tile_w = min(self.tile_width, img_width - x)
                    tile_h = min(self.tile_height, img_height - y)
                
                # Extract tile
                tile_img = image.crop((x, y, x + tile_w, y + tile_h))
                
                # If tile is smaller than expected (edge case), pad it
                if tile_img.size != (self.tile_width, self.tile_height) and force_uniform:
                    padded = Image.new(image.mode, (self.tile_width, self.tile_height))
                    padded.paste(tile_img, (0, 0))
                    tile_img = padded
                
                tile_info = TileInfo(
                    x=x,
                    y=y,
                    width=tile_w,
                    height=tile_h,
                    row=row,
                    col=col,
                    total_rows=rows,
                    total_cols=cols,
                )
                
                tiles.append((tile_img, tile_info))
        
        return tiles
    
    def create_blend_mask(
        self,
        tile_info: TileInfo,
        processed_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Create a blending mask for a tile.
        
        The mask is white in the center and fades to black at the edges
        where tiles overlap. This creates smooth blending at tile boundaries.
        
        Args:
            tile_info: Information about the tile
            processed_size: Size of the processed tile (may differ from input if upscaled)
            
        Returns:
            Grayscale mask image
        """
        width, height = processed_size
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        
        # Calculate fade regions based on overlap
        scale_x = width / tile_info.width
        scale_y = height / tile_info.height
        fade_x = int(self.overlap * scale_x)
        fade_y = int(self.overlap * scale_y)
        
        # Create gradient masks at edges where tiles overlap
        mask_array = np.array(mask, dtype=np.float32)
        
        # Left edge fade (if not first column)
        if tile_info.col > 0:
            for x in range(min(fade_x, width)):
                alpha = x / fade_x
                mask_array[:, x] *= alpha
        
        # Right edge fade (if not last column)
        if tile_info.col < tile_info.total_cols - 1:
            for x in range(max(0, width - fade_x), width):
                alpha = (width - x) / fade_x
                mask_array[:, x] *= alpha
        
        # Top edge fade (if not first row)
        if tile_info.row > 0:
            for y in range(min(fade_y, height)):
                alpha = y / fade_y
                mask_array[y, :] *= alpha
        
        # Bottom edge fade (if not last row)
        if tile_info.row < tile_info.total_rows - 1:
            for y in range(max(0, height - fade_y), height):
                alpha = (height - y) / fade_y
                mask_array[y, :] *= alpha
        
        mask = Image.fromarray(mask_array.astype(np.uint8), mode='L')
        
        # Apply Gaussian blur for smoother blending
        if self.mask_blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=self.mask_blur))
        
        return mask
    
    def stitch_tiles(
        self,
        tiles: List[Tuple[Image.Image, TileInfo]],
        original_size: Tuple[int, int],
        upscale_factor: float = 1.0
    ) -> Image.Image:
        """
        Stitch processed tiles back into a single image with seam blending.
        
        Args:
            tiles: List of (processed_tile_image, tile_info) tuples
            original_size: Size of the original input image
            upscale_factor: Factor by which tiles were upscaled
            
        Returns:
            Stitched image
        """
        output_width = int(original_size[0] * upscale_factor)
        output_height = int(original_size[1] * upscale_factor)
        
        # Create output image and weight map for blending
        if tiles[0][0].mode == 'RGB':
            output = Image.new('RGB', (output_width, output_height), (0, 0, 0))
            output_array = np.zeros((output_height, output_width, 3), dtype=np.float32)
        else:
            output = Image.new('RGBA', (output_width, output_height), (0, 0, 0, 0))
            output_array = np.zeros((output_height, output_width, 4), dtype=np.float32)
        
        weight_map = np.zeros((output_height, output_width), dtype=np.float32)
        
        for tile_img, tile_info in tiles:
            # Calculate position in output image
            out_x = int(tile_info.x * upscale_factor)
            out_y = int(tile_info.y * upscale_factor)
            out_w = int(tile_info.width * upscale_factor)
            out_h = int(tile_info.height * upscale_factor)
            
            # Resize tile if needed (handle padding case)
            if tile_img.size != (out_w, out_h):
                tile_img = tile_img.resize((out_w, out_h), Image.Resampling.LANCZOS)
            
            # Create blend mask for this tile
            mask = self.create_blend_mask(tile_info, (out_w, out_h))
            mask_array = np.array(mask, dtype=np.float32) / 255.0
            
            # Convert tile to array
            tile_array = np.array(tile_img, dtype=np.float32)
            
            # Ensure arrays don't go out of bounds
            actual_h = min(out_h, output_height - out_y)
            actual_w = min(out_w, output_width - out_x)
            
            # Apply weighted blending
            if tile_array.ndim == 3:
                for c in range(tile_array.shape[2]):
                    output_array[out_y:out_y+actual_h, out_x:out_x+actual_w, c] += (
                        tile_array[:actual_h, :actual_w, c] * mask_array[:actual_h, :actual_w]
                    )
            else:
                output_array[out_y:out_y+actual_h, out_x:out_x+actual_w] += (
                    tile_array[:actual_h, :actual_w] * mask_array[:actual_h, :actual_w]
                )
            
            weight_map[out_y:out_y+actual_h, out_x:out_x+actual_w] += mask_array[:actual_h, :actual_w]
        
        # Normalize by weight map to get final blended result
        weight_map = np.maximum(weight_map, 1e-6)  # Avoid division by zero
        
        if output_array.ndim == 3:
            for c in range(output_array.shape[2]):
                output_array[:, :, c] /= weight_map
        else:
            output_array /= weight_map
        
        # Convert back to image
        output_array = np.clip(output_array, 0, 255).astype(np.uint8)
        output = Image.fromarray(output_array, mode=tiles[0][0].mode)
        
        return output
    
    async def process_tiled(
        self,
        image: Image.Image,
        processor_fn: Callable,
        upscale_factor: float = 1.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **processor_kwargs
    ) -> Image.Image:
        """
        Process an image in tiles using the provided processor function.
        
        Args:
            image: Input image to process
            processor_fn: Async function that processes a single tile
                         Should have signature: async fn(tile_image, **kwargs) -> processed_image
            upscale_factor: Factor by which tiles will be upscaled
            progress_callback: Optional callback(current, total) for progress tracking
            **processor_kwargs: Additional arguments to pass to processor_fn
            
        Returns:
            Fully processed and stitched image
        """
        # Split image into tiles
        tiles = self.split_into_tiles(image)
        processed_tiles = []
        
        total_tiles = len(tiles)
        
        for idx, (tile_img, tile_info) in enumerate(tiles):
            # Process the tile
            processed_tile = await processor_fn(tile_img, **processor_kwargs)
            processed_tiles.append((processed_tile, tile_info))
            
            # Report progress
            if progress_callback:
                progress_callback(idx + 1, total_tiles)
        
        # Stitch tiles back together
        result = self.stitch_tiles(
            processed_tiles,
            image.size,
            upscale_factor=upscale_factor
        )
        
        return result
    
    async def process_tiled_with_control(
        self,
        image: Image.Image,
        processor_fn: Callable,
        upscale_factor: float = 1.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **processor_kwargs
    ) -> Image.Image:
        """
        Process an image in tiles with control image support.
        
        This variant passes tile_info to the processor function, enabling
        context-aware processing where each tile can be enhanced with
        knowledge of its position and surrounding context.
        
        Args:
            image: Input image to process
            processor_fn: Async function that processes a single tile
                         Should have signature: async fn(tile_image, tile_info=None, **kwargs) -> processed_image
            upscale_factor: Factor by which tiles will be upscaled
            progress_callback: Optional callback(current, total) for progress tracking
            **processor_kwargs: Additional arguments to pass to processor_fn
            
        Returns:
            Fully processed and stitched image
        """
        # Split image into tiles
        tiles = self.split_into_tiles(image)
        processed_tiles = []
        
        total_tiles = len(tiles)
        
        for idx, (tile_img, tile_info) in enumerate(tiles):
            # Process the tile with tile_info for control extraction
            processed_tile = await processor_fn(
                tile_img,
                tile_info=tile_info,
                **processor_kwargs
            )
            processed_tiles.append((processed_tile, tile_info))
            
            # Report progress
            if progress_callback:
                progress_callback(idx + 1, total_tiles)
        
        # Stitch tiles back together
        result = self.stitch_tiles(
            processed_tiles,
            image.size,
            upscale_factor=upscale_factor
        )
        
        return result


def calculate_optimal_tile_size(
    image_size: Tuple[int, int],
    max_tile_size: int = 512,
    min_tiles: int = 4
) -> Tuple[int, int]:
    """
    Calculate optimal tile size for an image.
    
    Args:
        image_size: (width, height) of the image
        max_tile_size: Maximum tile dimension
        min_tiles: Minimum number of tiles to create
        
    Returns:
        (tile_width, tile_height)
    """
    width, height = image_size
    
    # Calculate minimum tiles needed in each dimension
    tiles_x = max(min_tiles, math.ceil(width / max_tile_size))
    tiles_y = max(min_tiles, math.ceil(height / max_tile_size))
    
    # Calculate tile size
    tile_width = min(max_tile_size, math.ceil(width / tiles_x))
    tile_height = min(max_tile_size, math.ceil(height / tiles_y))
    
    # Round to multiples of 8 for better model performance
    tile_width = ((tile_width + 7) // 8) * 8
    tile_height = ((tile_height + 7) // 8) * 8
    
    return tile_width, tile_height

