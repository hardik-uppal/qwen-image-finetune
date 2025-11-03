# Qwen2.5-VL Prompt Generation Training Guide

This guide explains how to fine-tune Qwen2.5-VL-3B to generate image editing prompts from control images using TRL (Transformer Reinforcement Learning) and LoRA.

## Overview

**Task**: Image-to-text generation (vision-language captioning)
**Model**: Qwen2.5-VL-3B-Instruct  
**Method**: Supervised Fine-Tuning (SFT) with QLoRA  
**Framework**: HuggingFace TRL + PEFT  
**Hardware**: Optimized for 8xA100 GPUs (works on 1+ GPUs)

## Table of Contents

1. [Quick Start](#quick-start)
2. [Run Tracking and Naming](#run-tracking-and-naming)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Evaluation](#evaluation)

## Quick Start

### 1. Install Dependencies

```bash
# Install TRL and Qwen VL utils
pip install trl>=0.22.0 qwen-vl-utils>=0.0.11

# Install evaluation metrics (optional)
pip install evaluate rouge-score bert-score
```

### 2. Prepare Dataset

Your CSV should have these columns:
- `path_control`: Path to control image
- `prompt`: Target text prompt describing edits

Example:
```csv
path_control,prompt
/path/to/image1.jpg,"Increase brightness and enhance contrast. Adjust color temperature to warmer tones..."
/path/to/image2.jpg,"Remove background distractions. Sharpen details and improve clarity..."
```

### 3. Configure Training

Edit `configs/qwen25vl_prompt_gen.yaml`:

```yaml
data:
  train_csv: workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv
  train_split: 0.9  # 90% train, 10% validation

training:
  output_dir: workspace/qwen25vl-prompt-gen-lora
  num_train_epochs: 3
  per_device_train_batch_size: 4
```

### 4. Train the Model

**Single GPU:**
```bash
python script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml
```

**Multi-GPU (8xA100):**
```bash
accelerate launch --num_processes=8 \
    script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml
```

**Debug Mode (100 samples):**
```bash
python script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml \
    --debug
```

### 5. Run Inference

**Single Image:**
```bash
python script/infer_qwen25vl_prompt_gen.py \
    --adapter_path workspace/qwen25vl-prompt-gen-lora \
    --image /path/to/test_image.jpg
```

**Batch Inference:**
```bash
python script/infer_qwen25vl_prompt_gen.py \
    --adapter_path workspace/qwen25vl-prompt-gen-lora \
    --test_csv workspace/metadata-v12-recaptioned-long/val.csv \
    --output_csv results/generated_prompts.csv \
    --num_samples 100
```

**With Metrics:**
```bash
python script/infer_qwen25vl_prompt_gen.py \
    --adapter_path workspace/qwen25vl-prompt-gen-lora \
    --test_csv workspace/metadata-v12-recaptioned-long/val.csv \
    --output_csv results/generated_prompts.csv \
    --compute_metrics
```

## Run Tracking and Naming

**NEW**: The training system now includes comprehensive run tracking to correlate checkpoints with wandb runs!

### Why This Matters

Previously, it was difficult to:
- Find which checkpoint corresponds to which wandb run
- Track experiment results across multiple runs
- Clean up incomplete or failed training runs

### How It Works

Every training run now:
1. Uses a **consistent run name** for both output directory and wandb
2. Saves **metadata** linking the checkpoint to wandb run URL
3. Creates an **organized directory structure**

### Usage

#### Option 1: Auto-generated Run Name (Default)

```bash
python script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml
```

Creates: `qwen25vl_prompt_gen_20251031_143022/` (config name + timestamp)

#### Option 2: Custom Run Name (Recommended)

```bash
python script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml \
    --run-name experiment_v1_high_lr_r64
```

Creates: `experiment_v1_high_lr_r64/`

**Benefits**:
- Semantic names for easy reference
- Better team collaboration
- Simple experiment tracking

#### Option 3: Run Name in Config

Edit your config:
```yaml
training:
  run_name: baseline_experiment
```

### Inspecting Runs

Use the inspection utility to see all your runs:

```bash
# Scan for all training runs
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora

# Filter by name
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
    --filter experiment_v1

# Show only incomplete runs
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
    --incomplete-only

# Verbose output with all details
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
    --verbose
```

**Example Output**:
```
[1] experiment_v1_high_lr
    Status: ✓ Complete
    Directory: /skynas/.../experiment_v1_high_lr
    Created: 2025-10-31T14:30:22
    Wandb Run:
      • URL: https://wandb.ai/your-entity/project/runs/abc123
      • Project: your-entity/qwen25vl-prompt-gen
    Checkpoints: 5 found
```

### Run Metadata File

Each run directory contains `run_metadata.json` with:
- Direct link to wandb run URL
- Config summary
- Timestamp and hostname
- Complete traceability

Example:
```json
{
  "run_name": "experiment_v1_high_lr",
  "wandb": {
    "run_url": "https://wandb.ai/your-entity/project/runs/abc123",
    "run_id": "abc123",
    "project": "qwen25vl-prompt-gen"
  },
  "config_summary": {
    "learning_rate": 2e-05,
    "lora_r": 32
  }
}
```

📖 **For complete documentation**, see [docs/run_tracking.md](run_tracking.md)

## Configuration Guide

### Model Settings

```yaml
model:
  model_name: Qwen/Qwen2.5-VL-3B-Instruct
  quantization: 4bit  # Options: null, 4bit, 8bit
```

- **No quantization** (`null`): Best quality, requires ~40GB VRAM
- **4-bit** (`4bit`): QLoRA, ~18-22GB VRAM, minimal quality loss
- **8-bit** (`8bit`): ~25-30GB VRAM, better than 4-bit

### LoRA Settings

```yaml
lora:
  r: 64              # Rank: higher = more capacity (16-128)
  lora_alpha: 64     # Scaling: typically equals r
  lora_dropout: 0.05 # Regularization (0.0-0.1)
```

**Recommendations:**
- Small dataset (<10k): r=32
- Medium dataset (10k-100k): r=64
- Large dataset (>100k): r=128

### Training Settings

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 2.0e-5
```

**Effective Batch Size** = `per_device_batch_size × num_gpus × gradient_accumulation_steps`

Example with 8 GPUs: 4 × 8 × 2 = 64

### Data Settings

```yaml
data:
  system_prompt: "You are an expert photo editor..."
  user_prompt: "Describe the edits needed for this image."
```

Customize these to change the model's behavior.

## Training Time Estimates

| GPUs | Batch Size | Samples | Epochs | Time |
|------|------------|---------|--------|------|
| 1x A100 | 8 | 47k | 3 | ~8-12h |
| 4x A100 | 32 | 47k | 3 | ~2-3h |
| 8x A100 | 64 | 47k | 3 | ~1-2h |

*With 4-bit quantization and gradient checkpointing*

## Memory Requirements

| Configuration | VRAM per GPU |
|---------------|--------------|
| 4-bit QLoRA | ~18-22 GB |
| 8-bit | ~25-30 GB |
| Full precision | ~40-50 GB |

## Monitoring Training

### Weights & Biases

```bash
# Login to W&B
wandb login

# Training will automatically log to W&B
# View at: https://wandb.ai/your-username/qwen25vl-prompt-gen-lora
```

### TensorBoard

```yaml
training:
  report_to: tensorboard
```

```bash
tensorboard --logdir workspace/qwen25vl-prompt-gen-lora/runs
```

## Evaluation Metrics

The inference script computes:

- **BLEU-4**: N-gram overlap (0-1, higher better)
- **ROUGE-L**: Longest common subsequence (0-1, higher better)
- **BERTScore**: Semantic similarity (0-1, higher better)

Example output:
```
Evaluation Metrics:
================================================================================
bleu: 0.3542
rouge-1: 0.5821
rouge-2: 0.4123
rouge-l: 0.5234
bertscore_f1: 0.8912
================================================================================
```

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size: `per_device_train_batch_size: 2`
2. Increase gradient accumulation: `gradient_accumulation_steps: 4`
3. Enable 4-bit quantization: `quantization: 4bit`
4. Reduce LoRA rank: `r: 32`

### Training Too Slow

**Solutions:**
1. Increase batch size (if VRAM allows)
2. Reduce `dataloader_num_workers` if CPU bottleneck
3. Use multiple GPUs with `accelerate launch`
4. Disable gradient checkpointing (uses more VRAM)

### Poor Quality Outputs

**Solutions:**
1. Train longer: Increase `num_train_epochs`
2. Increase LoRA rank: `r: 128`
3. Lower learning rate: `learning_rate: 1e-5`
4. Check validation loss - may be overfitting
5. Improve dataset quality (better prompts)

### Model Overfitting

**Symptoms:** Training loss decreases, validation loss increases

**Solutions:**
1. Reduce epochs: `num_train_epochs: 2`
2. Increase dropout: `lora_dropout: 0.1`
3. Increase weight decay: `weight_decay: 0.05`
4. Use more data or data augmentation
5. Reduce LoRA rank: `r: 32`

## Advanced Usage

### Resume Training

```bash
python script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml \
    --resume_from_checkpoint workspace/qwen25vl-prompt-gen-lora/checkpoint-1000
```

### Custom Prompts

Edit `configs/qwen25vl_prompt_gen.yaml`:

```yaml
data:
  system_prompt: "You are a professional photographer analyzing images for post-processing."
  user_prompt: "What Lightroom adjustments would you recommend?"
```

### Export for Deployment

```python
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

# Load base model + adapter
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model.load_adapter("workspace/qwen25vl-prompt-gen-lora")

# Merge adapter into base model (optional, for faster inference)
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("workspace/qwen25vl-merged")
processor.save_pretrained("workspace/qwen25vl-merged")
```

## Next Steps: DPO Training

After SFT, you can further improve quality with Direct Preference Optimization (DPO):

1. Generate prompts using SFT model
2. Collect human preferences (which prompt is better?)
3. Create preference dataset with chosen/rejected pairs
4. Train with TRL's `DPOTrainer`

See [TRL DPO documentation](https://huggingface.co/docs/trl/dpo_trainer) for details.

## References

- [HuggingFace TRL VLM Tutorial](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)
- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## Support

For issues or questions:
1. Check this documentation
2. Review training logs in `output_dir`
3. Check W&B dashboard for training metrics
4. Examine sample outputs with inference script

