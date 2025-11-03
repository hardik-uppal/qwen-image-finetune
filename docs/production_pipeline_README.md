# Production Image Editing Pipeline

A distributed, scalable image editing pipeline built on **Qwen2.5-VL** and **Qwen-Image-Edit-Plus** with automatic load balancing and tiled super-resolution.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Production Pipeline                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────┐ │
│  │   Frontend   │───▶│  VL Service   │───▶│ Editing │ │
│  │   (Gradio)   │◀───│   (vLLM)      │◀───│ Service │ │
│  │   Port 7860  │    │   Port 8000   │    │(Ray)8001│ │
│  └──────────────┘    └───────────────┘    └─────────┘ │
│                                                │         │
│                                                ▼         │
│                                          ┌─────────────┐│
│                                          │   Tiled     ││
│                                          │ Upscaling   ││
│                                          └─────────────┘│
└─────────────────────────────────────────────────────────┘

GPU Allocation (8× A600):
├─ GPUs 0-1: Qwen2.5-VL (vLLM, tensor parallel)
└─ GPUs 2-7: Qwen-Image-Edit-Plus (Ray Serve, 6 replicas)
```

## Key Features

### ✅ Distributed Architecture
- **vLLM**: Efficient serving of Qwen2.5-VL across 2 GPUs with tensor parallelism
- **Ray Serve**: Auto-scaling image editing across 6 GPUs with load balancing
- **Horizontal Scaling**: Add more GPUs by adjusting configuration

### ✅ Modular Design
- **Abstract Base Classes**: Swap components without changing code
- **Registry Pattern**: Dynamic component instantiation
- **Config-Driven**: All settings in YAML, no code changes needed

### ✅ Production-Ready
- **Health Monitoring**: Real-time service and GPU monitoring
- **Metrics Tracking**: Per-stage timing, throughput, resource usage
- **Error Handling**: Graceful degradation and detailed error messages
- **Logging**: Comprehensive logs for debugging

### ✅ Advanced Features
- **Tiled Super-Resolution**: Memory-efficient 2-4× upscaling with seam blending
- **LoRA Support**: Server-level LoRA loading for specialized tasks
- **Batch Processing**: Handle multiple requests concurrently
- **A/B Testing**: Compare different models/configs side-by-side

## Quick Start

### 1. Install Dependencies

```bash
pip install vllm ray[serve] aiohttp pydantic rich gradio
```

### 2. Configure Pipeline

Edit `configs/production_pipeline.yaml`:

```yaml
gpu_allocation:
  vl_service: [0, 1]                    # 2 GPUs
  image_edit_service: [2, 3, 4, 5, 6, 7]  # 6 GPUs

models:
  qwen_vl:
    model_name: "Qwen/Qwen2.5-VL-72B-Instruct"
    tensor_parallel_size: 2
  
  qwen_image_edit:
    model_name: "Qwen/Qwen-Image-Edit-2509"
    lora_path: null  # Optional LoRA
    num_replicas: 6
```

### 3. Launch Services

```bash
# Start everything at once
python script/launch_production_services.py \
  --config configs/production_pipeline.yaml
```

### 4. Access Frontend

Open browser: `http://localhost:7860`

**That's it!** The system handles:
- Model loading
- GPU allocation
- Load balancing
- Health checks

## Components

### 1. Pipeline Module (`src/pipeline/`)

**Base Classes** (`base.py`):
- `VisionGenerator`: Vision-language instruction generation
- `ImageEditor`: Image editing/generation
- `Upscaler`: Super-resolution

**Clients** (`clients.py`):
- `VLLMVisionClient`: Connects to vLLM service
- `RayServeImageClient`: Connects to Ray Serve deployment
- `TiledUpscaler`: Tiled processing for large images

**Tiling** (`tiling.py`):
- Tile splitting with overlap
- Gaussian seam blending
- Stitching with weight maps

**Registry** (`registry.py`):
- Component registration
- Factory pattern for instantiation
- Dynamic loading

### 2. Serving Module (`src/serving/`)

**Ray Serve Deployment** (`image_edit_server.py`):
- Automatic GPU allocation (1 per replica)
- Load balancing across replicas
- Autoscaling based on traffic
- LoRA loading at startup

### 3. Scripts

**Deployment**:
- `launch_vllm_service.sh`: Start vLLM
- `launch_ray_serve.py`: Deploy Ray Serve
- `launch_production_services.py`: Start all services

**Frontend**:
- `gradio_production_frontend.py`: Web UI with metrics

**Monitoring**:
- `health_check.py`: Service and GPU monitoring

### 4. Configuration

**Main Config** (`configs/production_pipeline.yaml`):
- GPU allocation
- Model settings
- Service endpoints
- Pipeline parameters

**LoRA Variants**:
- `production_pipeline_lora_example.yaml`: Template for LoRA configs

## Usage Examples

### Basic Workflow

```python
# 1. User uploads image and describes edit
# 2. Qwen2.5-VL generates detailed instructions
# 3. Qwen-Image-Edit-Plus applies edits (distributed across 6 GPUs)
# 4. Optional: Tiled upscaling to 2× resolution
# 5. Results displayed with metrics
```

### API Usage

```python
from src.pipeline.clients import VLLMVisionClient, RayServeImageClient

# Initialize clients
vl_client = VLLMVisionClient({"base_url": "http://localhost:8000/v1"})
edit_client = RayServeImageClient({"base_url": "http://localhost:8001"})

# Generate instructions
instructions, _ = await vl_client.generate(
    image,
    "Make this image cyberpunk style"
)

# Edit image
edited, metrics = await edit_client.edit(
    image,
    instructions,
    num_inference_steps=20,
    guidance_scale=4.0
)

print(f"Processed on GPU {metrics['gpu_id']} in {metrics['inference_time']:.2f}s")
```

### Using LoRAs

```yaml
# Edit config
models:
  qwen_image_edit:
    lora_path: "/path/to/your_lora.safetensors"
```

```bash
# Restart Ray Serve
ray stop
python script/launch_ray_serve.py --config configs/production_pipeline.yaml
```

## Performance

### Throughput

With 6 replicas and 2 concurrent queries per replica:

- **Max Concurrent**: 12-24 requests
- **Throughput**: 30-60 images/min (depends on steps)

### Latency

Typical timings:

- **VL Generation**: 2-5s
- **Image Editing**: 5-15s (20-30 steps)
- **Tiled Upscale (2×)**: 15-30s (depends on tile count)

### Optimization Tips

1. **Increase throughput**: Add more replicas
2. **Reduce latency**: Use fewer inference steps
3. **Save memory**: Reduce image resolution
4. **Faster upscaling**: Larger tiles, less overlap

See [Deployment Guide](deployment_guide.md) for details.

## Monitoring

### Health Check

```bash
# One-time check
python script/health_check.py

# Continuous monitoring
python script/health_check.py --monitor
```

Shows:
- Service status (✅ healthy / ❌ down)
- GPU utilization, memory, temperature
- Request queue depth

### Metrics Export

Frontend exports:
- Per-stage timing
- GPU assignment
- Throughput statistics

Format: JSON (`outputs/metrics/`)

## Deployment Scenarios

### Single Server (8 GPUs)

Use default config:

```bash
python script/launch_production_services.py
```

### Multi-Server

Split services across machines:

**Server 1** (VL):
```bash
# Only vLLM
python script/launch_production_services.py --skip-ray --skip-frontend
```

**Server 2** (Editing):
```bash
# Only Ray Serve
python script/launch_production_services.py --skip-vllm --skip-frontend
```

**Server 3** (Frontend):
```bash
# Only Gradio
python script/launch_production_services.py --skip-vllm --skip-ray
```

Update endpoints in config accordingly.

### Docker (Coming Soon)

Containerized deployment with docker-compose.

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
→ Reduce `gpu_memory_utilization` in config

**"Service not responding"**
→ Check health: `python script/health_check.py`

**"Pipeline not initialized"**
→ Click "Initialize Pipeline" in frontend

**"Slow inference"**
→ Check GPU utilization with `nvidia-smi`

See [Deployment Guide](deployment_guide.md#troubleshooting) for more.

## Extending the System

### Add New Component

```python
from src.pipeline.base import Upscaler
from src.pipeline.registry import register_upscaler

@register_upscaler("my_upscaler")
class MyCustomUpscaler(Upscaler):
    async def upscale(self, image, factor, **kwargs):
        # Your implementation
        return upscaled_image, metrics
```

Now usable via config:

```yaml
pipeline:
  upscaler: "my_upscaler"
```

### Add New Endpoint

Modify `src/serving/image_edit_server.py` to add routes:

```python
@serve.ingress(app)
class QwenImageEditDeployment:
    @app.post("/custom_endpoint")
    async def custom_handler(self, request):
        # Custom logic
        return response
```

## Documentation

- **[Deployment Guide](deployment_guide.md)**: Complete setup and configuration
- **[API Reference](../README.md)**: Training and fine-tuning
- **[Troubleshooting](deployment_guide.md#troubleshooting)**: Common issues

## Architecture Decisions

### Why vLLM for VL Model?

- Optimized for LLM inference
- Tensor parallelism across 2 GPUs
- Continuous batching for throughput
- OpenAI-compatible API

### Why Ray Serve for Diffusion Model?

- Automatic load balancing
- GPU isolation (1 per replica)
- Graceful autoscaling
- Production-grade reliability

### Why Tiled Processing?

- Handle arbitrary image sizes
- Memory-efficient (one tile at a time)
- Seam blending avoids artifacts
- Parallel processing potential

## Roadmap

- [ ] Docker Compose deployment
- [ ] Prometheus metrics export
- [ ] Multi-LoRA dynamic loading
- [ ] Batch processing API
- [ ] Result caching layer
- [ ] A/B testing framework

## License

See main project LICENSE.

## Support

For issues:
1. Check [Deployment Guide](deployment_guide.md)
2. Run health check
3. Review logs in `logs/`
4. Open GitHub issue

---

**Built with** Qwen2.5-VL, Qwen-Image-Edit-Plus, vLLM, Ray Serve, and Gradio.

