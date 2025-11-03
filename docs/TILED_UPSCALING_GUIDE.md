# Context-Aware Tiled Upscaling Guide

## Overview

The production pipeline includes a **context-aware tiled upscaling** system that generates high-resolution images while maintaining global coherence. Unlike naive tiling approaches that process each tile independently, our system provides surrounding context to each tile generation.

## How It Works

### Problem with Naive Tiling

Standard tiled upscaling:
```
Input (1024×1024) → Split into 4 tiles (512×512)
For each tile:
    Generate(tile) → Enhanced tile
Stitch tiles → Output (2048×2048)
```

**Issues**:
- Tiles lack context from neighboring regions
- Can produce inconsistent lighting/colors
- May have visible seams
- No awareness of overall composition

### Our Context-Aware Approach

```
Input (1024×1024) → Upscale to 2048×2048 (LANCZOS)
For each tile position:
    1. Extract tile (512×512)
    2. Extract control region (768×768 centered on tile)
    3. Resize control to tile size
    4. Generate(control_image=control, prompt="enhance")
       → Model sees surrounding context!
Blend tiles with Gaussian masks → Output
```

**Benefits**:
- ✅ Each tile generation has context from surroundings
- ✅ Consistent lighting and colors across tiles
- ✅ Seamless blending with weighted masks
- ✅ Maintains global composition

## Configuration

### Enable/Disable Control Images

```yaml
pipeline:
  tiling:
    use_control_image: true  # Enable context-aware upscaling
    control_crop_factor: 1.5  # Control region size (1.5 = 1.5× tile size)
```

### Control Crop Factor

The `control_crop_factor` determines how much surrounding context to provide:

- `1.0`: Control = tile size (no extra context)
- `1.5`: Control = 1.5× tile size (recommended)
- `2.0`: Control = 2× tile size (maximum context, slower)

**Example** with 512×512 tiles:
- Factor 1.0 → 512×512 control
- Factor 1.5 → 768×768 control
- Factor 2.0 → 1024×1024 control

### Full Configuration

```yaml
pipeline:
  tiling:
    tile_width: 512              # Tile dimensions
    tile_height: 512
    overlap: 64                  # Overlap for blending
    mask_blur: 8                 # Gaussian blur radius
    upscale_factor: 2.0          # Output scale
    num_inference_steps: 15      # Generation steps
    guidance_scale: 3.5          # CFG scale
    use_control_image: true      # Enable context
    control_crop_factor: 1.5     # Context size
    prompt_template: "enhance quality, add detail, sharpen, ultra high resolution"
```

## Examples

### Standard Upscaling (No Control)

```yaml
tiling:
  use_control_image: false
```

Each tile processed independently. Faster but may have inconsistencies.

**Use case**: Draft previews, non-critical upscaling

### Context-Aware (Recommended)

```yaml
tiling:
  use_control_image: true
  control_crop_factor: 1.5
```

Balanced context and performance.

**Use case**: Production upscaling, maintaining coherence

### Maximum Context

```yaml
tiling:
  use_control_image: true
  control_crop_factor: 2.0
```

Maximum surrounding context. Slower but highest quality.

**Use case**: Critical images, complex compositions

## Visual Example

```
┌─────────────────────────────────┐
│       Full Image (2048×2048)     │
│                                  │
│    ┌────────────┐                │
│    │ Control    │  ← 768×768     │
│    │  ┌──────┐  │                │
│    │  │ Tile │  │  ← 512×512     │
│    │  └──────┘  │                │
│    └────────────┘                │
│                                  │
└─────────────────────────────────┘

The model generates the tile while "seeing"
the control region, maintaining consistency
with surrounding areas.
```

## Performance Impact

| Control Factor | Quality        | Speed       | Memory   |
|----------------|----------------|-------------|----------|
| Disabled (1.0) | Good           | Fastest     | Lowest   |
| 1.5 (default)  | Excellent      | Moderate    | Moderate |
| 2.0            | Best           | Slower      | Higher   |

**Recommendation**: Use 1.5 for best quality/speed balance.

## Advanced: Custom Control Extraction

The control region extraction is automatic, but you can customize it:

```python
from src.pipeline.clients import TiledUpscaler

class CustomTiledUpscaler(TiledUpscaler):
    def _extract_control_region(self, image, tile_info, crop_factor):
        # Custom logic here
        # e.g., always use full image as control
        return image.resize((tile_info.width, tile_info.height))
```

## Integration with Qwen-Image-Edit-Plus

The Qwen-Image-Edit-Plus model expects a control image (`prompt_image` parameter). Our tiled upscaler:

1. **Upscales** input to target resolution (LANCZOS)
2. For each tile position:
   - **Extracts** control region (crop_factor × tile_size)
   - **Resizes** control to tile size
   - **Generates** using: `edit(control_image, "enhance", width=512, height=512)`
3. **Blends** results with Gaussian masks

The model uses the control image to:
- Understand local context
- Maintain consistent style
- Preserve fine details
- Ensure coherent colors

## Troubleshooting

### Visible Seams

Increase overlap and blur:
```yaml
tiling:
  overlap: 128  # More overlap
  mask_blur: 16  # More blur
```

### Inconsistent Colors

Increase control region:
```yaml
tiling:
  control_crop_factor: 2.0  # More context
```

### Too Slow

Reduce context or tile count:
```yaml
tiling:
  control_crop_factor: 1.2  # Less context
  tile_width: 768           # Larger tiles = fewer tiles
```

### Out of Memory

Reduce tile size or disable control:
```yaml
tiling:
  tile_width: 384            # Smaller tiles
  use_control_image: false   # Or disable control
```

## Comparison with Alternatives

| Method                  | Context | Quality | Speed | Memory |
|-------------------------|---------|---------|-------|--------|
| Naive tiling            | ❌      | ⭐⭐    | ⚡⚡⚡ | 💾     |
| Full image generation   | ✅      | ⭐⭐⭐  | 🐌    | 💾💾💾 |
| Our context-aware tiling| ✅      | ⭐⭐⭐  | ⚡⚡   | 💾💾   |

Our approach provides **near full-image quality** at **tiled speed/memory**.

## Future Enhancements

Planned improvements:
- [ ] Parallel tile processing (process multiple tiles concurrently)
- [ ] Adaptive tile sizes based on image complexity
- [ ] Multi-scale processing (coarse to fine)
- [ ] ControlNet integration for additional guidance

---

For more details on the tiling algorithm, see [PRODUCTION_ARCHITECTURE.md](PRODUCTION_ARCHITECTURE.md#tiling-system).

