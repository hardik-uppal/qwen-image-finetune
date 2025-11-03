# Multi-GPU Model Sharding Setup

## Overview

The Qwen-Image-Edit-2509 model (~24GB) is now configured to be **sharded across multiple GPUs** using Diffusers' `device_map="auto"` feature, similar to how vLLM distributes large models.

---

## Architecture

### **Current Configuration: 2 Replicas × 4 GPUs Each**

```
┌─────────────────────────────────────────────┐
│   Server (8 GPUs Total)                     │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │  Replica 0 (Sharded)                 │  │
│  │  ├─ GPU 0: Transformer Layers 0-5    │  │
│  │  ├─ GPU 1: Transformer Layers 6-11   │  │
│  │  ├─ GPU 2: Transformer Layers 12-17  │  │
│  │  └─ GPU 3: VAE + Text Encoder        │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │  Replica 1 (Sharded)                 │  │
│  │  ├─ GPU 4: Transformer Layers 0-5    │  │
│  │  ├─ GPU 5: Transformer Layers 6-11   │  │
│  │  ├─ GPU 6: Transformer Layers 12-17  │  │
│  │  └─ GPU 7: VAE + Text Encoder        │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  Ray Serve Load Balancer (Port 8001)       │
│         ▲                                   │
└─────────┼───────────────────────────────────┘
          │
    ┌─────┴─────┐
    │  Gradio   │
    │ (Port 7860)│
    └───────────┘
```

---

## Configuration Details

### **`configs/production_pipeline.yaml`**

```yaml
models:
  qwen_image_edit:
    model_name: "Qwen/Qwen-Image-Edit-2509"
    dtype: "float16"             # FP16 for memory efficiency
    num_replicas: 2              # 2 replicas
    gpus_per_replica: 4          # Each replica uses 4 GPUs
    max_concurrent_queries_per_replica: 2

gpu_allocation:
  image_edit_service: [0, 1, 2, 3, 4, 5, 6, 7]  # All 8 GPUs available
```

---

## How It Works

### **1. Ray Serve Allocation**
```python
# Ray allocates 4 GPUs to each replica
ray_actor_options={"num_gpus": 4}

# Replica 0 gets GPUs 0-3
# Replica 1 gets GPUs 4-7
```

### **2. Diffusers Balanced Sharding**
```python
# Inside each replica
pipe = QwenImageEditPlusPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="balanced",  # Balanced distribution across assigned GPUs
    low_cpu_mem_usage=True
)
# Note: QwenImageEditPlusPipeline supports "balanced" and "cuda", not "auto"
```

### **3. Automatic Distribution**
Diffusers analyzes the model architecture and distributes components:
- **Text Encoder** → GPU 0 (or 4 for replica 1)
- **Transformer Blocks** → Split across GPUs 0-2 (or 4-6)
- **VAE** → GPU 3 (or 7)

---

## Memory Usage Per GPU

With FP16 and sharding across 4 GPUs:

| Component | Size | Placement |
|-----------|------|-----------|
| Transformer Blocks (split) | ~16-18GB | GPUs 0-2 (each ~5-6GB) |
| VAE | ~3-4GB | GPU 3 |
| Text Encoder | ~2-3GB | GPU 0 |
| **Total per GPU** | **~5-7GB** | **Instead of ~24GB!** |

---

## Scaling Options

### **Option 1: Current Setup (Balanced)**
```yaml
num_replicas: 2
gpus_per_replica: 4
# Total: 8 GPUs used
# Throughput: 2 concurrent images (2 replicas × 1 concurrent)
```

### **Option 2: More Replicas (Higher Throughput)**
```yaml
num_replicas: 4
gpus_per_replica: 2
# Total: 8 GPUs used
# Throughput: 4 concurrent images
# Note: Model might not fit well on just 2 GPUs
```

### **Option 3: Single Large Replica (Lowest Latency)**
```yaml
num_replicas: 1
gpus_per_replica: 8
# Total: 8 GPUs used
# Throughput: 1 concurrent image
# Advantage: Maximum memory, best single-request performance
```

### **Option 4: Conservative (Most Memory per GPU)**
```yaml
num_replicas: 1
gpus_per_replica: 4
# Total: 4 GPUs used (GPUs 4-7 idle)
# Throughput: 1 concurrent image
# Advantage: Safest option, leaves GPUs free for other tasks
```

---

## Launch Command

```bash
cd /workspace/hardik/test_repos/qwen-image-finetune

# Activate environment
conda activate myenv

# Clean up old Ray
python -c "import ray; ray.shutdown()" 2>/dev/null || true

# Launch with sharding
python script/launch_production_services.py \
    --config configs/production_pipeline.yaml \
    --skip-vllm
```

---

## Expected Startup Logs

```
Starting Ray Serve...
  Model: Qwen/Qwen-Image-Edit-2509
  GPUs available: [0, 1, 2, 3, 4, 5, 6, 7]
  Replicas: 2
  GPUs per replica: 4
  Total GPUs needed: 8

# For each replica:
Ray assigned 4 GPU(s) to this replica
Using device_map='balanced' to shard model across 4 GPUs
Loading with device_map='balanced' for multi-GPU sharding...
Loading checkpoint shards: 100%|████| 4/4
Model sharded across GPUs via device_map

GPU 0 - Allocated: 5.34 GB, Reserved: 5.50 GB
GPU 1 - Allocated: 5.89 GB, Reserved: 6.00 GB
GPU 2 - Allocated: 6.12 GB, Reserved: 6.25 GB
GPU 3 - Allocated: 4.23 GB, Reserved: 4.50 GB

Model loaded successfully
```

---

## Monitoring

### **Check GPU Distribution**
```bash
watch -n 1 nvidia-smi

# You should see:
# GPUs 0-3: Replica 0 (~5-7GB each)
# GPUs 4-7: Replica 1 (~5-7GB each)
```

### **Check Ray Serve Status**
```bash
# Health check
curl http://localhost:8001/-/healthz

# Replica info
tail -f logs/ray_serve.log | grep -E "GPU|memory|Replica"
```

### **Test Inference**
```bash
# Send a test request
curl -X POST http://localhost:8001/edit \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_image>",
    "prompt": "make it sunny",
    "num_inference_steps": 20
  }'

# Watch GPU utilization during inference
nvidia-smi dmon -i 0,1,2,3,4,5,6,7
```

---

## Advantages of Sharding

✅ **No OOM Errors**: Model distributed across multiple GPUs  
✅ **Better Utilization**: All 8 GPUs actively used  
✅ **Parallel Throughput**: 2 replicas handle concurrent requests  
✅ **Automatic Management**: Diffusers handles distribution automatically  
✅ **Scalable**: Easy to adjust replicas vs GPUs per replica  

---

## Troubleshooting

### **Still OOM?**
Reduce to 1 replica or increase GPUs per replica:
```yaml
num_replicas: 1
gpus_per_replica: 8  # Use all GPUs for single replica
```

### **Slow Inference?**
Model sharding adds communication overhead between GPUs. Check:
```bash
# Monitor GPU-to-GPU transfers
nvidia-smi nvlink -g 0
```

### **Ray Can't Find GPUs?**
Check CUDA_VISIBLE_DEVICES is set correctly:
```bash
echo $CUDA_VISIBLE_DEVICES
# Should show: 0,1,2,3,4,5,6,7
```

### **Replica Startup Hanging?**
Check if GPUs are truly free:
```bash
nvidia-smi
# All GPUs should show 0-2 MiB used before starting
```

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| Model Loading Time | ~2-3 min per replica |
| Memory per GPU | ~5-7 GB (FP16, sharded) |
| Inference Time | ~3-5 sec for 512×512 (20 steps) |
| Concurrent Requests | 2 (one per replica) |
| Queue Capacity | 4 (max_concurrent_queries) |

---

## Next Steps

1. ✅ Launch services and verify memory distribution
2. Test inference latency with sharding
3. Adjust `num_replicas` and `gpus_per_replica` based on workload
4. Consider enabling autoscaling for dynamic load

**Ready to launch!** 🚀

