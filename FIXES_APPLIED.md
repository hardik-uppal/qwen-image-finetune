# Fixes Applied for Production Deployment

## Issues Found & Fixed

### 1. **CUDA Out of Memory During Startup**
**Problem**: All 8 replicas trying to load simultaneously caused GPU 0 to run out of memory.

**Fixes Applied**:
- Added `low_cpu_mem_usage=True` to pipeline loading for efficient memory usage
- Added `torch.cuda.empty_cache()` after model loading to clear fragmented memory
- Reduced initial replicas from 8 to 4 to prevent simultaneous loading OOM

### 2. **Model Loading Optimizations**
**Changes in** `src/serving/image_edit_server.py`:
```python
# Before:
self.pipe = QwenImageEditPlusPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)

# After:
self.pipe = QwenImageEditPlusPipeline.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,  # Memory efficient loading
    device_map=None,         # Manual device placement
)
torch.cuda.empty_cache()     # Clear after loading
```

### 3. **Configuration Updates**
**File**: `configs/production_pipeline.yaml`

- **vLLM Endpoint**: Set to external server `http://192.168.20:8000/v1`
- **GPU Allocation**: All 8 GPUs available for Ray Serve (will use 4)
- **Replicas**: Reduced to 4 (prevents OOM during startup)
- **Max Concurrent**: Adjusted to 16 (4 replicas × 2 × 2)

## Current Setup

```yaml
# External vLLM server (on another machine)
vl_endpoint: "http://192.168.20:8000/v1"

# Local Ray Serve (4 replicas on 4 GPUs)
image_edit_service: [0, 1, 2, 3, 4, 5, 6, 7]  # 8 GPUs available
num_replicas: 4  # Use 4 to start safely

# Local Gradio frontend
port: 7860
share: true  # Gradio share link enabled
```

## Launch Command

```bash
cd /workspace/hardik/test_repos/qwen-image-finetune

# Activate environment
conda activate myenv

# Clear any existing Ray instances
source ~/.bashrc && conda activate myenv && python -c "import ray; ray.shutdown()" 2>/dev/null || true

# Launch (skip vLLM since it's external)
python script/launch_production_services.py \
    --config configs/production_pipeline.yaml \
    --skip-vllm
```

## Expected Behavior

1. **Ray Serve Startup** (~2-3 minutes):
   - Loads 4 replicas sequentially
   - Each replica on separate GPU (GPUs 0-3 will be used)
   - Memory per GPU: ~20-24GB

2. **Gradio Frontend** (~10 seconds):
   - Local: http://0.0.0.0:7860
   - Public: https://xxxxx.gradio.live (auto-displayed in logs)

3. **Resource Usage**:
   - GPUs 0-3: Ray Serve replicas (~20-24GB each)
   - GPUs 4-7: Available for scaling up later
   - vLLM: External server (192.168.20:8000)

## Monitoring

```bash
# Watch logs
tail -f logs/ray_serve.log      # Model loading progress
tail -f logs/frontend.log        # Gradio startup & share link

# Check GPU usage
watch -n 1 nvidia-smi

# Check Ray Serve health
curl http://localhost:8001/-/healthz
```

## Scaling Up

Once initial 4 replicas are stable, you can scale to 8:

```bash
# Edit config
nano configs/production_pipeline.yaml
# Change: num_replicas: 4 → num_replicas: 8

# Restart Ray Serve only
source ~/.bashrc && conda activate myenv && python -c "import ray; ray.shutdown()"
python script/launch_ray_serve.py \
    --config configs/production_pipeline.yaml \
    --detached
```

## Troubleshooting

### Still getting OOM?
- Try `num_replicas: 2` for even safer startup
- Check no other processes using GPUs: `nvidia-smi`

### Can't connect to vLLM?
```bash
# Test from this server
curl http://192.168.20:8000/v1/models
# Should return model info
```

### Ray Serve not starting?
```bash
# Clean restart
source ~/.bashrc && conda activate myenv
python -c "import ray; ray.shutdown()"
pkill -f ray
sleep 2
# Then relaunch
```
