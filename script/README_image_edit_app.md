# Qwen-Image-Edit-Plus (2509) Gradio App

This Gradio application provides an intuitive web interface for the Qwen-Image-Edit-Plus (2509) image generation and editing model. Upload an input image, provide editing instructions, and generate high-quality edited results.

## Overview

**What is Qwen-Image-Edit-Plus?**

Qwen-Image-Edit-Plus is a diffusion-based image editing model that can:
- Transform images based on text prompts
- Perform various editing tasks (style transfer, segmentation, composition, etc.)
- Support LoRA fine-tuning for specialized tasks
- Generate high-quality edited images

**Key Differences from Vision-Language Models:**
- **Input**: Image + Text Prompt
- **Output**: Edited/Generated Image (not text)
- **Use Cases**: Image editing, style transfer, object manipulation, etc.

## Quick Start

### Prerequisites

1. **Python 3.12+** with CUDA support
2. **GPU Requirements**: 
   - Minimum: 16GB VRAM (base model in bfloat16)
   - Recommended: 24GB+ VRAM for comfortable usage
   - Can use quantization (fp4) for lower memory

3. **Dependencies**: All required packages should already be installed if you followed the main setup:
   ```bash
   pip install -r requirements.txt
   ```

### Launch the App

```bash
cd /workspace/hardik/test_repos/qwen-image-finetune
python script/gradio_image_edit_app.py
```

**With specific GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 python script/gradio_image_edit_app.py
```

**Access the interface:**
```
http://localhost:7860
```

## Usage Guide

### Step 1: Load the Model

1. **Configure Model Settings** in the "Model Configuration" section:
   - **Model Name**: Keep default `Qwen/Qwen-Image-Edit-Plus` or use a local path
   - **Device**: Select your GPU (cuda, cuda:0, cuda:1) or CPU
   - **Data Type**: `bfloat16` (recommended), `float16`, or `float32`
   - **LoRA Path** (Optional): Path to fine-tuned LoRA weights

2. **Click "Load Model"** - This will:
   - Download the model if needed (first time only)
   - Load the model into memory
   - Load LoRA weights if specified
   - Takes 1-3 minutes depending on your hardware

### Step 2: Generate Images

1. **Upload Input Image**: Click the image upload area and select your image
2. **Enter Prompt**: Describe the edit you want to make
3. **Configure Parameters** (optional):
   - **Inference Steps**: 20-30 for good quality (more = better but slower)
   - **Guidance Scale**: 3-5 for balanced results
   - **Resolution**: Adjust width/height (512x512 recommended)
   - **Seed**: Set for reproducible results

4. **Click "Generate Image"**: Wait for generation (5-30 seconds depending on steps and hardware)
5. **View Result**: The edited image appears on the right

### Step 3: Iterate and Refine

- Adjust the prompt for different edits
- Change guidance scale for more/less prompt adherence
- Try different seeds for variations
- Adjust inference steps for quality/speed tradeoff

## Example Use Cases

### 1. Face Segmentation

**Use Case**: Generate face segmentation masks for training or analysis

**Setup:**
```
Model: Qwen/Qwen-Image-Edit-Plus
LoRA: TsienDragon/qwen-image-edit-plus-lora-face-seg
```

**Prompt:**
```
change the image from the face to the face segmentation mask
```

**Result**: Segmented face mask showing different facial regions

### 2. Style Transfer

**Setup:**
```
Model: Qwen/Qwen-Image-Edit-Plus
LoRA: (none or custom style LoRA)
```

**Prompts:**
- "convert this photo to an oil painting style"
- "make this image look like a watercolor painting"
- "transform to anime style"

### 3. Character Composition

**Setup:**
```
Model: Qwen/Qwen-Image-Edit-Plus
LoRA: (multi-control trained LoRA)
```

**Prompt:**
```
compose the characters from multiple images into a single scene
```

**Note**: For multi-control, you may need a modified interface or use the trainer directly.

### 4. Image Enhancement

**Prompts:**
- "enhance the image quality and sharpness"
- "remove noise and blur"
- "upscale and improve details"

### 5. Color Adjustments

**Prompts:**
- "make the image black and white"
- "enhance colors to be more vibrant"
- "adjust lighting to golden hour"
- "change to warm tone color grading"

## Using LoRA Models

### Available Pre-trained LoRA Models

1. **Face Segmentation**
   ```
   TsienDragon/qwen-image-edit-plus-lora-face-seg
   ```
   - Task: Face segmentation mask generation
   - Use case: Dataset preparation, face analysis

2. **Character Composition**
   ```
   TsienDragon/character-composition
   ```
   - Task: Multi-character composition
   - Use case: Combining characters from different images

### Using Your Own Fine-tuned LoRA

If you've trained a custom LoRA using this framework:

1. **Local Path**: Point to your checkpoint directory
   ```
   /path/to/your/checkpoint/lora_weights.safetensors
   ```

2. **HuggingFace**: Upload your LoRA and use the repo name
   ```
   your-username/your-lora-model
   ```

## Configuration Options

### Model Configuration

| Parameter | Options | Description |
|-----------|---------|-------------|
| Model Name | HF repo or path | Base model to use |
| Device | cuda, cuda:0, cuda:1, cpu | Compute device |
| Data Type | bfloat16, float16, float32 | Model precision |
| LoRA Path | Path or HF repo | Optional fine-tuned weights |

### Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Inference Steps | 1-50 | 20 | Quality vs speed (higher = better) |
| Guidance Scale | 1.0-10.0 | 4.0 | Prompt adherence strength |
| Width | 256-1024 | 512 | Output image width (px) |
| Height | 256-1024 | 512 | Output image height (px) |
| Seed | -1 or 0-2147483647 | -1 | Random seed (-1 = random) |

### Parameter Tuning Tips

**Inference Steps:**
- 10-15: Fast preview (lower quality)
- 20-30: Balanced (recommended)
- 30-50: High quality (slower)

**Guidance Scale:**
- 1-2: Minimal guidance, more creative
- 3-5: Balanced (recommended)
- 6-10: Strong guidance, very literal

**Resolution:**
- 512x512: Fast, standard quality
- 768x768: Better detail, slower
- 1024x1024: Highest quality, much slower

## Performance and Optimization

### GPU Memory Usage

| Configuration | VRAM Usage | Speed (512x512, 20 steps) |
|---------------|------------|---------------------------|
| bfloat16 base | ~16GB | ~10 seconds |
| float16 base | ~14GB | ~10 seconds |
| bfloat16 + LoRA | ~17GB | ~10 seconds |

### Speed Optimization

1. **Lower Resolution**: Use 512x512 instead of 1024x1024
2. **Fewer Steps**: Use 15-20 steps instead of 30+
3. **Batch Size**: Process one image at a time
4. **Model Quantization**: Use fp4 quantization (requires code modification)

### Memory Optimization

If you encounter out-of-memory errors:

1. **Use float16 or quantization**:
   ```python
   # In the code, modify dtype parameter
   dtype = "float16"  # or implement fp4 loading
   ```

2. **Lower Resolution**:
   ```
   Width: 384
   Height: 384
   ```

3. **Enable CPU Offloading** (requires code modification):
   ```python
   # Add to trainer setup
   enable_model_cpu_offload=True
   ```

## Troubleshooting

### Model Loading Issues

**Problem**: `CUDA out of memory` during model loading
```bash
# Solution 1: Use smaller precision
dtype = "float16"

# Solution 2: Use specific GPU
CUDA_VISIBLE_DEVICES=1 python script/gradio_image_edit_app.py

# Solution 3: Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

**Problem**: Model download is slow
```bash
# Solution: Set HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
```

**Problem**: LoRA weights not found
```
# Ensure the path is correct
# For local: /full/path/to/lora_weights.safetensors
# For HF: username/repo-name (no https://)
```

### Generation Issues

**Problem**: Generated image looks bad
- Increase inference steps to 30+
- Adjust guidance scale to 3-5
- Try different seeds
- Improve prompt clarity

**Problem**: Generation is too slow
- Reduce inference steps to 15-20
- Lower resolution to 512x512
- Use fewer guidance steps

**Problem**: Image doesn't match prompt
- Increase guidance scale to 5-7
- Make prompt more specific and detailed
- Try different seeds
- Check if LoRA is appropriate for task

### Common Errors

**Error**: `Model not loaded`
- Click "Load Model" button first
- Wait for loading to complete
- Check status message

**Error**: `ImportError: No module named 'src'`
```bash
# Ensure you're in the project root
cd /workspace/hardik/test_repos/qwen-image-finetune
python script/gradio_image_edit_app.py
```

**Error**: `Trainer not found`
```bash
# Install project in development mode
pip install -e .
```

## Advanced Usage

### Using Programmatically

You can use the QwenImageEditApp class directly:

```python
from script.gradio_image_edit_app import QwenImageEditApp
from PIL import Image

# Initialize app
app = QwenImageEditApp(
    model_name="Qwen/Qwen-Image-Edit-Plus",
    lora_path="TsienDragon/qwen-image-edit-plus-lora-face-seg",
    device="cuda",
    dtype="bfloat16",
)

# Load model
status = app.load_model()
print(status)

# Generate image
input_img = Image.open("input.jpg")
output_img, status = app.generate_image(
    input_image=input_img,
    prompt="change the image from the face to the face segmentation mask",
    num_inference_steps=20,
    guidance_scale=4.0,
)

# Save result
output_img.save("output.jpg")
```

### Using Trainer Directly

For more control, use the trainer API directly:

```python
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
from src.data.config import load_config_from_yaml
from PIL import Image

# Load config
config = load_config_from_yaml("configs/your_config.yaml")

# Initialize trainer
trainer = QwenImageEditPlusTrainer(config)
trainer.setup_predict()

# Load LoRA if needed
trainer.load_lora_weights("path/to/lora")

# Generate
input_img = Image.open("input.jpg")
result = trainer.predict(
    prompt_image=input_img,
    prompt="your editing instruction",
    num_inference_steps=20,
    true_cfg_scale=4.0,
    height=512,
    width=512,
)

result[0].save("output.jpg")
```

### Batch Processing

For processing multiple images, create a script:

```python
from pathlib import Path
from script.gradio_image_edit_app import QwenImageEditApp
from PIL import Image

# Initialize once
app = QwenImageEditApp(model_name="Qwen/Qwen-Image-Edit-Plus")
app.load_model()

# Process directory
input_dir = Path("input_images")
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*.jpg"):
    input_img = Image.open(img_path)
    output_img, status = app.generate_image(
        input_image=input_img,
        prompt="your prompt here",
        num_inference_steps=20,
    )
    if output_img:
        output_img.save(output_dir / img_path.name)
        print(f"Processed: {img_path.name}")
```

## Comparison with Other Apps

### vs gradio_prompt_generator.py

| Feature | image_edit_app.py | prompt_generator.py |
|---------|-------------------|---------------------|
| Purpose | Image editing/generation | Prompt generation from image pairs |
| Input | 1 image + text | 2 images (before/after) |
| Output | Edited image | Text description |
| Model | Qwen-Image-Edit-Plus | Qwen2.5-VL (vision-language) |
| Backend | Local (this framework) | vLLM server |

### vs gradio_qwen_image_app.py

| Feature | image_edit_app.py | qwen_image_app.py |
|---------|-------------------|-------------------|
| Purpose | Image editing | Image analysis |
| Input | Image + edit prompt | Image + question |
| Output | Edited image | Text answer |
| Model | Image Edit (diffusion) | Vision-Language |
| Task | Generate/edit images | Understand/describe images |

## Tips and Best Practices

1. **Start with Base Model**: Test with base model first, then add LoRA
2. **Iterate Prompts**: Refine your prompt based on results
3. **Use Seeds**: Set seed for reproducible results
4. **Save Good Results**: Note down prompt, seed, and parameters for good results
5. **Batch Similar**: Process similar images with same settings for consistency
6. **Monitor Memory**: Watch GPU usage, especially with high resolution
7. **Experiment**: Try different guidance scales and steps for your use case

## Related Resources

- [Main Project README](../README.md)
- [Training Guide](../docs/training.md)
- [Model Architecture](../docs/architecture/qwen_image_edit_plus.md)
- [Inference Guide](../docs/inference.md)
- [Example Notebooks](../tests/trainer/)

## Support

For issues or questions:
- Check the main [project documentation](../README.md)
- Review the [troubleshooting section](#troubleshooting)
- Open an issue on the repository
- Check existing [examples and notebooks](../tests/trainer/)



