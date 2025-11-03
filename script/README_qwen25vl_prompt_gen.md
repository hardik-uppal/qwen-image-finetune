# Qwen2.5-VL Prompt Generation

Train a vision-language model to automatically generate image editing prompts from photographs.

## What This Does

Given a control image, the model generates detailed editing instructions like:

> "Increase the overall brightness of the scene, particularly in the areas surrounding the bed. Enhance the natural light coming through the windows to create a more inviting atmosphere. Adjust the color temperature slightly warmer to give the room a cozier feel. Sharpen details on the bedding and furniture to improve clarity..."

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

**First, validate images and format the dataset:**

```bash
# Automatic (recommended)
./script/prepare_qwen25vl_dataset.sh

# Manual
python script/prepare_qwen25vl_dataset.py --config configs/qwen25vl_prompt_gen.yaml

# Debug mode (only 100 samples)
python script/prepare_qwen25vl_dataset.py --config configs/qwen25vl_prompt_gen.yaml --debug
```

This will:
- Validate all image paths exist and are readable
- Filter out broken/missing images
- Format data into chat template
- Save to `workspace/prepared_data/train.jsonl` and `eval.jsonl`

**Note:** Training uses **streaming mode** - images are loaded on-demand during training, not all loaded into memory at once. This enables training on arbitrarily large datasets without memory issues.

### 3. Train the Model

**Automatic (recommended):**
```bash
./script/train_qwen25vl_prompt_gen.sh
```

**Manual:**
```bash
# Single GPU
python script/train_qwen25vl_prompt_generation.py --config configs/qwen25vl_prompt_gen.yaml

# Multi-GPU (4xA100)
accelerate launch --num_processes=4 \
    script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml
```

### 4. Generate Prompts

**Single Image:**
```bash
python script/infer_qwen25vl_prompt_gen.py \
    --adapter_path workspace/qwen25vl-prompt-gen-lora \
    --image /path/to/your/image.jpg
```

**Batch Processing:**
```bash
python script/infer_qwen25vl_prompt_gen.py \
    --adapter_path workspace/qwen25vl-prompt-gen-lora \
    --test_csv workspace/metadata-v12-recaptioned-long/val.csv \
    --output_csv results/generated_prompts.csv \
    --num_samples 100 \
    --compute_metrics
```

## Files

| File | Purpose |
|------|---------|
| `prepare_qwen25vl_dataset.py` | Dataset validation and formatting |
| `prepare_qwen25vl_dataset.sh` | Quick start dataset preparation |
| `train_qwen25vl_prompt_generation.py` | Main training script |
| `train_qwen25vl_prompt_gen.sh` | Quick start training script |
| `infer_qwen25vl_prompt_gen.py` | Inference and evaluation |
| `example_qwen25vl_usage.py` | Example usage code |
| `../configs/qwen25vl_prompt_gen.yaml` | Training configuration |
| `../docs/qwen25vl_prompt_generation_trl.md` | Full documentation |

## Configuration

Edit `configs/qwen25vl_prompt_gen.yaml`:

```yaml
# Key settings
model:
  quantization: 4bit  # 4bit=faster, null=better quality

lora:
  r: 64  # Increase for larger datasets

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4  # Adjust for your GPU
  learning_rate: 2.0e-5
```

## Training Time

| Hardware | Dataset Prep | Training (3 epochs, 47k samples) |
|----------|--------------|-----------------------------------|
| CPU | ~5-10 min | N/A |
| 1x A100 80GB | ~5-10 min | ~8-12 hours |
| 4x A100 80GB | ~5-10 min | ~2-3 hours |
| 8x A100 80GB | ~5-10 min | ~1-2 hours |

Note: Dataset preparation is a one-time step that validates images and can be reused across multiple training runs.

## Features

- ✅ **Streaming dataset loading** - Handle arbitrarily large datasets without RAM issues
- ✅ **Image validation** - Separate preparation step filters out broken images before training
- ✅ **Multi-GPU support** - Efficient distributed training with Accelerate
- ✅ **QLoRA (4-bit)** - Train on consumer GPUs with quantization
- ✅ **Checkpointing** - Resume training and save best models automatically

## Use Cases

1. **Automatic prompt generation** for image editing models
2. **Dataset augmentation** - generate variations of editing instructions
3. **Photo editing assistant** - suggest improvements for photos
4. **Batch processing** - analyze large collections of images
5. **Large-scale training** - streaming enables training on massive datasets (100k+ images)

## Integration Example

```python
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model.load_adapter("workspace/qwen25vl-prompt-gen-lora")
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Generate prompt
from PIL import Image
image = Image.open("photo.jpg")
messages = [...]  # See example_qwen25vl_usage.py
prompt = generate(model, processor, messages)

# Use with image editing model
from diffusers import QwenImageEditPipeline
pipe = QwenImageEditPipeline.from_pretrained(...)
edited = pipe(image=image, prompt=prompt).images[0]
```

## Advanced: DPO Fine-tuning

After SFT training, improve quality with preference optimization:

1. Generate prompts with current model
2. Collect human preferences (A vs B comparisons)
3. Train with TRL's DPOTrainer
4. Get better, more aligned prompts

See [TRL DPO docs](https://huggingface.co/docs/trl/dpo_trainer) for details.

## Troubleshooting

**Missing images / File not found errors?**
- Run `prepare_qwen25vl_dataset.py` first - it validates all images
- Check the `dataset_metadata.json` to see how many samples were skipped
- Update your CSV with valid image paths

**Out of Memory?**
- Reduce `per_device_train_batch_size: 2` (or even 1)
- Increase `gradient_accumulation_steps: 8` (or 16)
- Reduce `dataloader_num_workers: 0`
- Use 4-bit quantization (already enabled by default)

**Training too slow?**
- Use multiple GPUs with `accelerate launch`
- Increase batch size if VRAM allows
- Increase `dataloader_num_workers` if CPU is underutilized

**Poor quality outputs?**
- Train longer (more epochs)
- Increase LoRA rank (`r: 128`)
- Check validation loss (may be overfitting)
- Ensure high-quality training data

## Full Documentation

See [`docs/qwen25vl_prompt_generation_trl.md`](../docs/qwen25vl_prompt_generation_trl.md) for:
- Detailed configuration guide
- Memory requirements
- Evaluation metrics
- Advanced usage
- Best practices

## References

- [HuggingFace TRL VLM Tutorial](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)
- [Qwen2.5-VL Model](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [TRL Documentation](https://huggingface.co/docs/trl)

