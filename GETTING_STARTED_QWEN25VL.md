# Getting Started with Qwen2.5-VL Prompt Generation

## 🎯 What This Does

Trains Qwen2.5-VL-3B to automatically generate detailed image editing prompts from control images.

**Input:** A photograph  
**Output:** Detailed editing instructions

Example:
```
Input: [Interior room photo]
Output: "Increase the overall brightness of the scene, particularly 
focusing on the areas surrounding the furniture. Enhance the natural 
light coming through the windows to create a more inviting atmosphere. 
Adjust the color temperature slightly warmer..."
```

## ⚡ Quick Start (5 Minutes)

### Step 1: Validate Setup

```bash
python script/validate_qwen25vl_setup.py --config configs/qwen25vl_prompt_gen.yaml
```

This checks:
- ✅ Required packages installed
- ✅ GPUs available
- ✅ Dataset accessible
- ✅ Config valid

### Step 2: Train (1-2 hours on 8xA100)

```bash
./script/train_qwen25vl_prompt_gen.sh configs/qwen25vl_prompt_gen.yaml
```

Or manually:
```bash
accelerate launch --num_processes=8 \
    script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml
```

### Step 3: Generate Prompts

```bash
# Single image
python script/infer_qwen25vl_prompt_gen.py \
    --adapter_path workspace/qwen25vl-prompt-gen-lora \
    --image /path/to/image.jpg

# Batch processing
python script/infer_qwen25vl_prompt_gen.py \
    --adapter_path workspace/qwen25vl-prompt-gen-lora \
    --test_csv workspace/metadata-v12-recaptioned-long/val.csv \
    --output_csv results/prompts.csv \
    --compute_metrics
```

## 📁 What Was Created

### Core Scripts
- **`train_qwen25vl_prompt_generation.py`** - Training script with TRL
- **`infer_qwen25vl_prompt_gen.py`** - Inference and evaluation
- **`train_qwen25vl_prompt_gen.sh`** - Quick start launcher
- **`validate_qwen25vl_setup.py`** - Environment validator
- **`example_qwen25vl_usage.py`** - Integration example

### Configuration
- **`configs/qwen25vl_prompt_gen.yaml`** - Training config (8xA100 optimized)

### Documentation
- **`docs/qwen25vl_prompt_generation_trl.md`** - Full guide (~500 lines)
- **`script/README_qwen25vl_prompt_gen.md`** - Quick reference
- **`QWEN25VL_IMPLEMENTATION_SUMMARY.md`** - Implementation details

## 🔧 Configuration Tuning

### For Your Hardware

**8x A100 (80GB)** - Default config works perfectly
```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
quantization: 4bit
```

**4x A100 (80GB)** - Adjust batch size
```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 4  # Increase to maintain effective batch size
```

**1x A100 (80GB)** - Single GPU
```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 16  # Larger accumulation
```

**Lower VRAM (24GB)** - Smaller batch
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
quantization: 4bit  # Essential!
```

### For Your Dataset Size

**Current: 47k samples** - Default config is optimal
```yaml
lora:
  r: 64
training:
  num_train_epochs: 3
```

**Smaller (<10k samples)** - Prevent overfitting
```yaml
lora:
  r: 32  # Lower rank
  lora_dropout: 0.1  # More dropout
training:
  num_train_epochs: 5
  weight_decay: 0.05  # More regularization
```

**Larger (>100k samples)** - More capacity
```yaml
lora:
  r: 128  # Higher rank
  lora_dropout: 0.05
training:
  num_train_epochs: 2  # Fewer epochs needed
```

## 📊 Expected Results

### Training Time (47k samples, 8xA100)
- **Setup & data loading:** ~5 minutes
- **Training (3 epochs):** ~1-2 hours
- **Total:** ~2 hours

### Performance Metrics
| Metric | Expected Range | Meaning |
|--------|----------------|---------|
| BLEU-4 | 0.30-0.40 | N-gram overlap with ground truth |
| ROUGE-L | 0.50-0.60 | Sequence similarity |
| BERTScore | 0.85-0.90 | Semantic similarity |

### VRAM Usage
- **Per GPU:** ~18-22 GB (with 4-bit quantization)
- **Peak:** ~25 GB (during optimizer step)
- **A100 80GB:** Plenty of headroom ✅

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Problem:** `CUDA out of memory` error

**Solutions:**
1. Reduce batch size:
   ```yaml
   per_device_train_batch_size: 2  # or even 1
   ```

2. Increase gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 4  # or 8
   ```

3. Enable 4-bit quantization:
   ```yaml
   quantization: 4bit
   ```

4. Enable gradient checkpointing (already on):
   ```yaml
   gradient_checkpointing: true
   ```

### Training Too Slow

**Problem:** Training takes >4 hours on 8xA100

**Solutions:**
1. Check GPU utilization:
   ```bash
   nvidia-smi dmon -s u
   # Should show ~80-100% GPU usage
   ```

2. Increase batch size (if VRAM allows):
   ```yaml
   per_device_train_batch_size: 8
   ```

3. Reduce data loading bottleneck:
   ```yaml
   dataloader_num_workers: 8  # Increase if CPU has capacity
   ```

4. Check if running in debug mode:
   ```bash
   # Should NOT see --debug flag
   ps aux | grep train_qwen25vl
   ```

### Poor Quality Outputs

**Problem:** Generated prompts are vague or incorrect

**Solutions:**
1. Train longer:
   ```yaml
   num_train_epochs: 5
   ```

2. Increase model capacity:
   ```yaml
   lora:
     r: 128  # More parameters
   ```

3. Lower learning rate:
   ```yaml
   learning_rate: 1.0e-5  # More careful updates
   ```

4. Check validation loss:
   ```bash
   # Should decrease steadily
   tensorboard --logdir workspace/qwen25vl-prompt-gen-lora
   ```

5. Review your dataset quality:
   - Are prompts detailed enough?
   - Are images diverse?
   - Any corrupted samples?

### Model Not Loading

**Problem:** Cannot load Qwen2.5-VL model

**Solutions:**
1. Login to HuggingFace:
   ```bash
   huggingface-cli login
   # Enter your token
   ```

2. Check internet connection:
   ```bash
   ping huggingface.co
   ```

3. Try manual download:
   ```python
   from transformers import Qwen2VLProcessor
   processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
   ```

## 🚀 Next Steps After Training

### 1. Evaluate Quality

```bash
python script/infer_qwen25vl_prompt_gen.py \
    --adapter_path workspace/qwen25vl-prompt-gen-lora \
    --test_csv workspace/metadata-v12-recaptioned-long/val.csv \
    --output_csv results/eval.csv \
    --num_samples 500 \
    --compute_metrics
```

Review `results/eval.csv` to compare generated vs ground truth prompts.

### 2. Integrate with Image Editing

```python
from transformers import Qwen2VLForConditionalGeneration
from diffusers import QwenImageEditPipeline

# Load prompt generator
prompt_model = Qwen2VLForConditionalGeneration.from_pretrained(...)
prompt_model.load_adapter("workspace/qwen25vl-prompt-gen-lora")

# Generate prompt from image
prompt = generate_prompt(control_image, prompt_model, processor)

# Use with image editing model
edit_pipe = QwenImageEditPipeline.from_pretrained(...)
edited = edit_pipe(image=control_image, prompt=prompt).images[0]
```

### 3. Deploy as Service (Optional)

**Option A: vLLM Server** (Fast inference)
```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --enable-lora \
    --lora-modules qwen25vl-prompt=workspace/qwen25vl-prompt-gen-lora
```

**Option B: Gradio Interface** (User-friendly)
```python
import gradio as gr

def generate_prompt_gradio(image):
    return generate_prompt(image, model, processor)

gr.Interface(
    fn=generate_prompt_gradio,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox()
).launch()
```

**Option C: REST API** (Production)
```python
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/generate-prompt")
async def generate(image: UploadFile):
    img = Image.open(image.file)
    prompt = generate_prompt(img, model, processor)
    return {"prompt": prompt}
```

### 4. Improve with DPO (Advanced)

After SFT, use Direct Preference Optimization:

1. **Collect preferences:**
   - Generate multiple prompts per image
   - Have humans rate them
   - Create chosen/rejected pairs

2. **Train with DPO:**
   ```python
   from trl import DPOTrainer
   
   trainer = DPOTrainer(
       model=model,
       train_dataset=preference_dataset,
       ...
   )
   trainer.train()
   ```

3. **Result:** More natural, preferred prompts

## 📚 Documentation

### Quick Reference
- **Quick start:** `script/README_qwen25vl_prompt_gen.md`
- **Examples:** `script/example_qwen25vl_usage.py`

### Comprehensive Guides
- **Full guide:** `docs/qwen25vl_prompt_generation_trl.md`
- **Implementation:** `QWEN25VL_IMPLEMENTATION_SUMMARY.md`

### External Resources
- [HuggingFace TRL Tutorial](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)
- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [TRL Documentation](https://huggingface.co/docs/trl)

## 💡 Tips & Best Practices

### Dataset Preparation
- ✅ Use diverse, high-quality images
- ✅ Ensure prompts are detailed and actionable
- ✅ Remove corrupted or duplicate samples
- ✅ Balance prompt length distribution

### Training
- ✅ Monitor validation loss (check for overfitting)
- ✅ Use W&B or TensorBoard for visualization
- ✅ Save intermediate checkpoints
- ✅ Test on validation set regularly

### Inference
- ✅ Adjust temperature (0.7 default, higher=more creative)
- ✅ Use batch processing for efficiency
- ✅ Cache generated prompts to avoid regeneration
- ✅ Implement fallback for error cases

## 🎓 Learning Resources

### Understanding the Code
1. Start with `script/example_qwen25vl_usage.py` - Simple example
2. Review `configs/qwen25vl_prompt_gen.yaml` - Configuration
3. Read `script/train_qwen25vl_prompt_generation.py` - Full trainer

### Understanding the Method
1. Read [TRL SFT Tutorial](https://huggingface.co/docs/trl/sft_trainer)
2. Learn about [LoRA/QLoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)
3. Explore [Qwen2.5-VL architecture](https://qwenlm.github.io/blog/qwen2.5-vl/)

## ✅ Pre-Flight Checklist

Before starting training:

- [ ] Run `python script/validate_qwen25vl_setup.py`
- [ ] Check GPU availability: `nvidia-smi`
- [ ] Login to HuggingFace: `huggingface-cli login`
- [ ] Review config: `configs/qwen25vl_prompt_gen.yaml`
- [ ] Verify dataset path in config
- [ ] Check output directory is writeable
- [ ] (Optional) Setup W&B: `wandb login`

## 🎉 Ready to Train!

```bash
# Validate setup
python script/validate_qwen25vl_setup.py

# Start training (8xA100)
./script/train_qwen25vl_prompt_gen.sh configs/qwen25vl_prompt_gen.yaml

# Monitor progress
# - Watch terminal output
# - Check W&B dashboard (if enabled)
# - Monitor GPUs: watch -n 1 nvidia-smi

# After training
python script/infer_qwen25vl_prompt_gen.py \
    --adapter_path workspace/qwen25vl-prompt-gen-lora \
    --image test.jpg
```

Good luck with your training! 🚀

