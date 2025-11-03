# Production Pipeline Deployment Guide

Complete guide for deploying the distributed image editing pipeline on an 8-GPU server.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment](#deployment)
5. [Using the System](#using-the-system)
6. [Monitoring](#monitoring)
7. [LoRA Management](#lora-management)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tuning](#performance-tuning)

---

## System Requirements

### Hardware

- **GPUs**: 8× NVIDIA A600 (48GB each) or similar
  - 2 GPUs for Qwen2.5-VL (vLLM)
  - 6 GPUs for Qwen-Image-Edit-Plus (Ray Serve)
- **RAM**: 128GB+ recommended
- **Storage**: 500GB+ for models and outputs

### Software

- Ubuntu 20.04+ or similar Linux
- CUDA 11.8+ with compatible drivers
- Python 3.10+
- Docker (optional, for containerized deployment)

---

## Installation

### 1. Clone Repository

```bash
cd /workspace/hardik/test_repos
git clone <repo-url> qwen-image-finetune
cd qwen-image-finetune
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt

# Install additional production dependencies
pip install vllm ray[serve] aiohttp pydantic rich
```

### 4. Download Models

Models will be auto-downloaded on first run, but you can pre-download:

```bash
# Qwen2.5-VL (vision-language model)
huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct

# Qwen-Image-Edit-Plus (diffusion model)
huggingface-cli download Qwen/Qwen-Image-Edit-2509
```

### 5. Create Log Directory

```bash
mkdir -p logs outputs/metrics
```

---

## Configuration

### Master Configuration File

Edit `configs/production_pipeline.yaml`:

```yaml
# GPU allocation (adjust based on your setup)
gpu_allocation:
  vl_service: [0, 1]                    # 2 GPUs for VL
  image_edit_service: [2, 3, 4, 5, 6, 7]  # 6 GPUs for editing

# Service endpoints (change if running on different machines)
services:
  vl_endpoint: "http://localhost:8000/v1"
  image_edit_endpoint: "http://localhost:8001"

# Model settings
models:
  qwen_vl:
    model_name: "Qwen/Qwen2.5-VL-72B-Instruct"
    tensor_parallel_size: 2
    max_model_len: 8192
    gpu_memory_utilization: 0.9
  
  qwen_image_edit:
    model_name: "Qwen/Qwen-Image-Edit-2509"
    lora_path: null  # Set to LoRA path if needed
    dtype: "bfloat16"
    num_replicas: 6

# Frontend settings
frontend:
  host: "0.0.0.0"
  port: 7860
  max_concurrent_requests: 24
```

### Multiple Configurations

Create variants for different use cases:

```bash
# Base model
cp configs/production_pipeline.yaml configs/base_model.yaml

# With LoRA A
cp configs/production_pipeline_lora_example.yaml configs/lora_face_seg.yaml
# Edit and set: lora_path: "/path/to/face_seg_lora.safetensors"

# With LoRA B
cp configs/production_pipeline_lora_example.yaml configs/lora_style.yaml
# Edit and set: lora_path: "/path/to/style_lora.safetensors"
```

---

## Deployment

### Option 1: Start All Services at Once (Recommended)

```bash
# Start everything with one command
python script/launch_production_services.py \
  --config configs/production_pipeline.yaml
```

This launches:
1. vLLM service (Qwen2.5-VL)
2. Ray Serve deployment (Qwen-Image-Edit-Plus)
3. Gradio frontend

**Logs** are saved to `logs/` directory.

### Option 2: Start Services Individually

Useful for debugging or running services on different machines.

#### Step 1: Start vLLM Service

```bash
bash script/launch_vllm_service.sh
```

Or with custom config:

```bash
CONFIG_FILE=configs/production_pipeline.yaml bash script/launch_vllm_service.sh
```

**Wait** until you see "Application startup complete" in logs.

#### Step 2: Start Ray Serve

```bash
python script/launch_ray_serve.py \
  --config configs/production_pipeline.yaml
```

**Wait** until all 6 replicas are deployed (check logs).

#### Step 3: Start Frontend

```bash
python script/gradio_production_frontend.py \
  --config configs/production_pipeline.yaml
```

### Verify Deployment

```bash
# Check service health
python script/health_check.py --config configs/production_pipeline.yaml

# Continuous monitoring
python script/health_check.py --monitor --interval 5
```

---

## Using the System

### Access the Frontend

Open your browser:

```
http://your-server-ip:7860
```

If running locally:

```
http://localhost:7860
```

### Basic Workflow

1. **Initialize Pipeline**
   - Click "🔄 Initialize Pipeline"
   - Wait for confirmation

2. **Upload Image**
   - Drag & drop or click to upload

3. **Enter Edit Request**
   - Example: "Transform this into a cinematic scene with dramatic lighting"

4. **Adjust Parameters** (optional)
   - Inference steps (more = better quality, slower)
   - Guidance scale (higher = more prompt adherence)
   - Enable/disable upscaling

5. **Run Pipeline**
   - Click "✨ Run Full Pipeline"
   - View results in Output tabs

6. **Compare Results**
   - Switch between Edited, Upscaled, and Comparison tabs

7. **Export Metrics**
   - Enter filename and click "💾 Export Metrics"

### API Usage (Advanced)

#### Vision Generation (vLLM)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Describe editing steps for: make this cyberpunk"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]}
    ]
)
```

#### Image Editing (Ray Serve)

```python
import aiohttp
import base64

async def edit_image(image_path, prompt):
    with open(image_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    payload = {
        "image": image_b64,
        "prompt": prompt,
        "negative_prompt": "blurry, low quality",
        "num_inference_steps": 20,
        "guidance_scale": 4.0,
        "seed": -1,
        "width": 768,
        "height": 768,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8001/edit", json=payload) as resp:
            result = await resp.json()
            return result["image"]  # Base64 encoded
```

---

## Monitoring

### Real-time Health Check

```bash
# One-time check
python script/health_check.py

# Continuous monitoring (updates every 5s)
python script/health_check.py --monitor

# Custom interval
python script/health_check.py --monitor --interval 10
```

Shows:
- Service status (vLLM, Ray Serve)
- GPU utilization per device
- Memory usage
- Temperature

### View Logs

```bash
# vLLM service
tail -f logs/vllm_service.log

# Ray Serve
tail -f logs/ray_serve.log

# Frontend
tail -f logs/frontend.log
```

### GPU Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nvidia-smi
```

### Metrics Export

From the frontend:
1. Enter filename in "Export Filename"
2. Click "💾 Export Metrics"
3. Check `outputs/metrics/`

Metrics include:
- Per-stage timing
- GPU assignment
- Throughput statistics

---

## LoRA Management

### Using Image Editing LoRA

1. **Update Configuration**

Edit `configs/production_pipeline.yaml`:

```yaml
models:
  qwen_image_edit:
    lora_path: "/path/to/your_editing_lora.safetensors"
    # OR HuggingFace repo:
    # lora_path: "username/repo-name"
```

2. **Restart Ray Serve**

```bash
# Stop existing deployment
ray stop

# Restart with new config
python script/launch_ray_serve.py --config configs/production_pipeline.yaml
```

### Using Vision-Language LoRA

Fine-tuned VL models can generate better instructions for specific domains.

1. **Update Configuration**

Edit `configs/production_pipeline.yaml`:

```yaml
models:
  qwen_vl:
    lora_path: "/path/to/your_vl_lora"
    # OR HuggingFace repo:
    # lora_path: "username/vl-lora-repo"
    lora_name: "my_vl_lora"  # Adapter name (can be anything)
```

2. **Restart vLLM**

```bash
# Stop existing service
pkill -f vllm

# Restart with LoRA
bash script/launch_vllm_service.sh
```

**Note**: vLLM automatically enables LoRA support when `lora_path` is set. The LoRA adapter will be loaded at startup.

### Using Both LoRAs Simultaneously

You can use both VL and Image Editing LoRAs at the same time:

```yaml
models:
  qwen_vl:
    lora_path: "/path/to/vl_lora"
    lora_name: "vl_adapter"
  
  qwen_image_edit:
    lora_path: "/path/to/editing_lora.safetensors"
```

Restart both services to apply.

### Switching Between LoRAs

Create separate configs for each LoRA:

```bash
# Start with LoRA A
python script/launch_production_services.py \
  --config configs/lora_face_seg.yaml

# Stop services (Ctrl+C)

# Start with LoRA B
python script/launch_production_services.py \
  --config configs/lora_style.yaml
```

### LoRA Training

See main README for training new LoRAs:

```bash
# Train a custom LoRA
python src/train.py --config configs/qwen_image_edit_plus_custom.yaml
```

---

## Troubleshooting

### Service Won't Start

#### vLLM Issues

**Problem**: "CUDA out of memory"

```bash
# Reduce GPU memory utilization
# Edit config: gpu_memory_utilization: 0.85
```

**Problem**: "Model not found"

```bash
# Pre-download model
huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct
```

#### Ray Serve Issues

**Problem**: "Failed to allocate GPU"

```bash
# Check GPU availability
nvidia-smi

# Ensure CUDA_VISIBLE_DEVICES matches config
echo $CUDA_VISIBLE_DEVICES
```

**Problem**: "Replica startup failed"

```bash
# Check Ray Serve logs
tail -f logs/ray_serve.log

# Check Ray status
ray status
```

### Frontend Issues

**Problem**: "Pipeline not initialized"

Solution: Click "🔄 Initialize Pipeline" before running inference.

**Problem**: "Connection refused"

```bash
# Verify services are running
python script/health_check.py

# Check endpoints in config match running services
```

### Performance Issues

**Problem**: "Slow inference"

Solutions:
- Reduce `num_inference_steps` (try 15-20)
- Disable upscaling for faster results
- Check GPU utilization (should be >80%)

**Problem**: "High memory usage"

Solutions:
- Reduce `max_model_len` for vLLM
- Lower `gpu_memory_utilization`
- Reduce image resolution (width/height)

### Network Issues

**Problem**: "Cannot access from remote machine"

```bash
# Ensure firewall allows ports
sudo ufw allow 7860  # Frontend
sudo ufw allow 8000  # vLLM
sudo ufw allow 8001  # Ray Serve

# Check if services bind to 0.0.0.0
netstat -tulpn | grep -E '7860|8000|8001'
```

---

## Performance Tuning

### Throughput Optimization

1. **Increase Replicas**

```yaml
models:
  qwen_image_edit:
    num_replicas: 8  # Use more GPUs
    max_concurrent_queries_per_replica: 3  # Handle more concurrent
```

2. **Tune Autoscaling**

```yaml
ray_serve:
  autoscaling_config:
    target_num_ongoing_requests_per_replica: 2
    min_replicas: 4
    max_replicas: 8
```

3. **Reduce Inference Steps**

For draft/preview mode:

```yaml
pipeline:
  edit_defaults:
    num_inference_steps: 15  # Faster, slightly lower quality
```

### Latency Optimization

1. **Keep Models Warm**

```yaml
ray_serve:
  autoscaling_config:
    downscale_delay_s: 3600  # Keep replicas alive longer
```

2. **Optimize Tiling**

```yaml
pipeline:
  tiling:
    tile_width: 768  # Larger tiles = fewer tiles = faster
    overlap: 32      # Less overlap = faster stitching
```

3. **Use FP16**

```yaml
models:
  qwen_image_edit:
    dtype: "float16"  # Faster than bfloat16 on some GPUs
```

### Memory Optimization

For limited VRAM:

```yaml
models:
  qwen_vl:
    gpu_memory_utilization: 0.85  # Leave more headroom
    max_model_len: 4096           # Shorter context

  qwen_image_edit:
    num_replicas: 4               # Fewer replicas
```

---

## Advanced Topics

### Running on Multiple Machines

Split services across servers:

**Machine 1** (VL Server):
```yaml
# Config: vl_only.yaml
services:
  vl_endpoint: "http://machine1:8000/v1"
```

```bash
bash script/launch_vllm_service.sh
```

**Machine 2** (Edit Server):
```yaml
# Config: edit_only.yaml
services:
  vl_endpoint: "http://machine1:8000/v1"
  image_edit_endpoint: "http://machine2:8001"
```

```bash
python script/launch_ray_serve.py --config edit_only.yaml
```

**Machine 3** (Frontend):
```bash
python script/gradio_production_frontend.py --config frontend_only.yaml
```

### Containerized Deployment

Coming soon: Docker Compose setup for easy deployment.

### Load Balancing

For high-traffic scenarios, add nginx in front:

```nginx
upstream ray_serve {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://ray_serve;
    }
}
```

---

## Support

For issues or questions:

1. Check logs in `logs/` directory
2. Run health check: `python script/health_check.py`
3. Review configuration: `configs/production_pipeline.yaml`
4. See main project README for training/fine-tuning

---

## Quick Reference

### Common Commands

```bash
# Start everything
python script/launch_production_services.py

# Health check
python script/health_check.py --monitor

# Stop all services
# Press Ctrl+C in launch terminal, or:
pkill -f launch_production_services
ray stop

# Restart with new config
ray stop
python script/launch_production_services.py --config new_config.yaml

# View logs
tail -f logs/*.log

# GPU usage
watch -n 1 nvidia-smi
```

### Port Reference

- `7860`: Gradio frontend
- `8000`: vLLM service (Qwen2.5-VL)
- `8001`: Ray Serve (Qwen-Image-Edit-Plus)
- `9090`: Metrics (optional)

### Config File Locations

- `configs/production_pipeline.yaml` - Main config
- `configs/production_pipeline_lora_example.yaml` - LoRA template
- `logs/` - Service logs
- `outputs/metrics/` - Exported metrics

