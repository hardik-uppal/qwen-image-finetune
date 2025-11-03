# Production Pipeline - Quick Start Guide

Get the distributed image editing pipeline running in 5 minutes.

## Prerequisites

- **Hardware**: 8× NVIDIA GPUs (A600 or similar, 48GB VRAM)
- **Software**: Ubuntu 20.04+, CUDA 11.8+, Python 3.10+

## Installation (2 minutes)

```bash
# 1. Install dependencies
pip install vllm ray[serve] aiohttp pydantic rich gradio openai

# 2. Create logs directory
mkdir -p logs outputs/metrics

# 3. Optional: Pre-download models (or they'll auto-download)
huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct
huggingface-cli download Qwen/Qwen-Image-Edit-2509
```

## Configuration (1 minute)

The default config at `configs/production_pipeline.yaml` works out of the box:

- GPUs 0-1: Qwen2.5-VL (vision-language)
- GPUs 2-7: Qwen-Image-Edit-2509 (image editing, 6 replicas)

**Optional**: Edit GPU allocation if needed:

```yaml
gpu_allocation:
  vl_service: [0, 1]                    # Change these if needed
  image_edit_service: [2, 3, 4, 5, 6, 7]
```

## Launch (1 minute)

```bash
# Start all services with one command
python script/launch_production_services.py \
  --config configs/production_pipeline.yaml
```

Wait for:
- "vLLM started" (~30-60s)
- "Ray Serve deployed with 6 replicas" (~30-60s)
- "Gradio frontend running" (~5s)

## Use (1 minute)

### Option 1: Web Interface

1. Open browser: `http://localhost:7860`
2. Click **"🔄 Initialize Pipeline"**
3. Upload an image
4. Enter edit request: *"Transform this into a cinematic scene"*
5. Click **"✨ Run Full Pipeline"**

### Option 2: API

```python
from src.pipeline.clients import VLLMVisionClient, RayServeImageClient

# Connect to services
vl = VLLMVisionClient({"base_url": "http://localhost:8000/v1"})
editor = RayServeImageClient({"base_url": "http://localhost:8001"})

# Generate instructions
instructions, _ = await vl.generate(image, "Make it cyberpunk")

# Edit image (distributed across 6 GPUs automatically)
edited, metrics = await editor.edit(image, instructions)
print(f"Done in {metrics['inference_time']:.2f}s on GPU {metrics['gpu_id']}")
```

## Health Check

```bash
# Check all services
python script/health_check.py

# Continuous monitoring
python script/health_check.py --monitor
```

Should show:
- ✅ vLLM (Qwen2.5-VL): Healthy
- ✅ Ray Serve (Image Edit): Healthy
- GPU utilization for all 8 GPUs

## Troubleshooting

**Services won't start?**
```bash
# Check GPU availability
nvidia-smi

# View logs
tail -f logs/*.log
```

**"CUDA out of memory"?**
```yaml
# Edit config: reduce memory usage
models:
  qwen_vl:
    gpu_memory_utilization: 0.85  # Default: 0.9
```

**Can't access from remote machine?**
```bash
# Check firewall
sudo ufw allow 7860
```

## Using LoRA

### Image Editing LoRA

To use a fine-tuned LoRA for image editing:

1. **Edit config:**
```yaml
models:
  qwen_image_edit:
    lora_path: "/path/to/your_editing_lora.safetensors"
```

2. **Restart Ray Serve:**
```bash
ray stop
python script/launch_ray_serve.py --config configs/production_pipeline.yaml
```

### Vision-Language LoRA

To use a fine-tuned LoRA for instruction generation:

1. **Edit config:**
```yaml
models:
  qwen_vl:
    lora_path: "/path/to/your_vl_lora"
    lora_name: "my_vl_lora"  # Adapter name
```

2. **Restart vLLM:**
```bash
pkill -f vllm
bash script/launch_vllm_service.sh
```

### Using Both LoRAs

You can use both simultaneously! Just set both paths in the config and restart both services.

## Next Steps

- **[Full Deployment Guide](docs/deployment_guide.md)**: Advanced configuration
- **[Production README](docs/production_pipeline_README.md)**: Architecture details
- **[Training Guide](README.md)**: Train custom LoRAs

## Common Commands

```bash
# Start everything
python script/launch_production_services.py

# Stop everything
# Press Ctrl+C, or:
pkill -f launch_production_services
ray stop

# Health check
python script/health_check.py

# View logs
tail -f logs/vllm_service.log
tail -f logs/ray_serve.log
tail -f logs/frontend.log

# GPU monitoring
watch -n 1 nvidia-smi
```

## Performance Expectations

With the default setup (2 + 6 GPUs):

- **Concurrent Requests**: 12-24 images
- **Throughput**: 30-60 images/minute
- **Latency**: 
  - VL generation: 2-5s
  - Image editing: 5-15s
  - Upscaling (2×): 15-30s

## Configuration Examples

### More Replicas (Higher Throughput)

```yaml
models:
  qwen_image_edit:
    num_replicas: 8  # Use all 8 GPUs for editing
```

### Faster Inference (Lower Quality)

```yaml
pipeline:
  edit_defaults:
    num_inference_steps: 15  # Default: 20
```

### Different Model

```yaml
models:
  qwen_vl:
    model_name: "Qwen/Qwen2.5-VL-32B-Instruct"  # Smaller model
```

## Support

1. Check [Deployment Guide](docs/deployment_guide.md)
2. Run health check: `python script/health_check.py`
3. Review logs: `tail -f logs/*.log`
4. Check GPU: `nvidia-smi`

---

**That's it!** You now have a production-ready distributed image editing pipeline running on 8 GPUs with automatic load balancing.

For more advanced topics (multi-server deployment, Docker, monitoring), see the [Full Deployment Guide](docs/deployment_guide.md).

