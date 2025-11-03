# Production Pipeline Architecture

Technical deep-dive into the distributed image editing pipeline architecture.

## System Overview

The production pipeline is a **3-tier distributed system**:

1. **Vision-Language Tier**: Instruction generation (Qwen2.5-VL via vLLM)
2. **Diffusion Tier**: Image editing (Qwen-Image-Edit-Plus via Ray Serve)
3. **Presentation Tier**: Web interface (Gradio)

All tiers communicate via HTTP/REST APIs for loose coupling and horizontal scalability.

## Component Architecture

### 1. Pipeline Module (`src/pipeline/`)

#### Base Classes (`base.py`)

```
┌─────────────────────────────────────────┐
│         Pipeline Components             │
├─────────────────────────────────────────┤
│                                         │
│  VisionGenerator (Abstract)             │
│  ├─ generate(image, prompt) → str      │
│  └─ load(config) → status               │
│                                         │
│  ImageEditor (Abstract)                 │
│  ├─ edit(image, prompt) → Image        │
│  └─ load(config) → status               │
│                                         │
│  Upscaler (Abstract)                    │
│  ├─ upscale(image, factor) → Image     │
│  └─ load(config) → status               │
│                                         │
└─────────────────────────────────────────┘
```

**Design Principles:**
- **Interface segregation**: Each component has a single responsibility
- **Dependency inversion**: Depend on abstractions, not implementations
- **Open/closed**: Open for extension (new components), closed for modification

#### Client Implementations (`clients.py`)

**VLLMVisionClient**:
- Wraps OpenAI-compatible API
- Handles image encoding (JPEG, base64)
- Async/await for non-blocking I/O
- Automatic retry logic (TODO)

**RayServeImageClient**:
- HTTP client for Ray Serve endpoint
- Batches requests internally
- Connection pooling via aiohttp
- Load balancing handled by Ray

**TiledUpscaler**:
- Uses ImageEditor for per-tile processing
- Delegates to tiling module for splitting/stitching
- Memory-efficient streaming (processes one tile at a time)

#### Tiling System (`tiling.py`)

```
┌───────────────────────────────────┐
│      Tiled Processing Flow        │
├───────────────────────────────────┤
│                                   │
│  1. Split image into tiles        │
│     ├─ Calculate grid (rows×cols) │
│     ├─ Add overlap between tiles  │
│     └─ Handle edge tiles          │
│                                   │
│  2. Process each tile             │
│     └─ Call processor_fn(tile)    │
│                                   │
│  3. Create blend masks            │
│     ├─ Gaussian fade at edges     │
│     └─ Weight map for overlaps    │
│                                   │
│  4. Stitch tiles                  │
│     ├─ Weighted blending          │
│     ├─ Normalize by weight map    │
│     └─ Clip to original bounds    │
│                                   │
└───────────────────────────────────┘
```

**Key Algorithms:**

*Tile Calculation*:
```python
cols = ceil((width - overlap) / (tile_width - overlap))
rows = ceil((height - overlap) / (tile_height - overlap))
```

*Blend Mask*:
- Linear fade over overlap region
- Gaussian blur for smooth transitions
- Weight normalization to avoid brightness changes

*Stitching*:
```python
output[y:y+h, x:x+w] += tile * mask
weights[y:y+h, x:x+w] += mask
output /= weights  # Normalize
```

#### Registry (`registry.py`)

**Pattern**: Factory + Registry
- Components self-register via decorators
- Factory creates instances by name
- Config-driven instantiation

```python
@register_image_editor("ray_serve_client")
class RayServeImageClient(ImageEditor):
    ...

# Later:
editor = registry.get_image_editor("ray_serve_client", config)
```

### 2. Serving Module (`src/serving/`)

#### Ray Serve Architecture

```
┌─────────────────────────────────────────────┐
│            Ray Serve Cluster                │
├─────────────────────────────────────────────┤
│                                             │
│  HTTP Server (Port 8001)                    │
│         │                                   │
│         ├──▶ Router (Load Balancer)         │
│              │                              │
│              ├──▶ Replica 1 (GPU 2)         │
│              ├──▶ Replica 2 (GPU 3)         │
│              ├──▶ Replica 3 (GPU 4)         │
│              ├──▶ Replica 4 (GPU 5)         │
│              ├──▶ Replica 5 (GPU 6)         │
│              └──▶ Replica 6 (GPU 7)         │
│                                             │
│  Autoscaling Controller                     │
│  ├─ Monitor queue depth                     │
│  ├─ Scale up if > target                    │
│  └─ Scale down if < target                  │
│                                             │
└─────────────────────────────────────────────┘
```

**Deployment Configuration**:
```python
@serve.deployment(
    num_replicas=6,                   # Start with 6
    max_concurrent_queries=2,         # Per replica
    ray_actor_options={"num_gpus": 1}, # GPU isolation
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 1,
        "min_replicas": 3,
        "max_replicas": 6,
    }
)
```

**Request Flow**:
1. Client sends HTTP POST to `/edit`
2. Ray ingress receives request
3. Router selects replica (least loaded)
4. Replica processes on assigned GPU
5. Result returned with metadata (GPU ID, timing)

**GPU Isolation**:
- Each replica pins to 1 GPU
- Set via `ray_actor_options={"num_gpus": 1}`
- CUDA_VISIBLE_DEVICES managed by Ray
- No cross-GPU interference

### 3. vLLM Integration

#### vLLM Architecture

```
┌───────────────────────────────────────┐
│         vLLM Service                  │
├───────────────────────────────────────┤
│                                       │
│  API Server (OpenAI compatible)       │
│         │                             │
│         ├──▶ Request Queue            │
│              │                        │
│              ├──▶ Continuous Batching │
│              │    ├─ Batch requests   │
│              │    └─ Fill GPU capacity│
│              │                        │
│              └──▶ Model Engine        │
│                   ├─ GPU 0 (shard 1)  │
│                   └─ GPU 1 (shard 2)  │
│                   (Tensor Parallel)   │
│                                       │
└───────────────────────────────────────┘
```

**Key Features**:
- **Tensor Parallelism**: Model split across 2 GPUs
- **Continuous Batching**: Batch requests as they arrive
- **PagedAttention**: Efficient KV cache management
- **OpenAI API**: Drop-in replacement for OpenAI client

**Performance Optimizations**:
- `--gpu-memory-utilization 0.9`: Use 90% of VRAM
- `--max-model-len 8192`: Context window limit
- `--tensor-parallel-size 2`: Split across 2 GPUs

### 4. Frontend (Gradio)

#### Component Structure

```
Frontend
├─ Initialization
│  └─ Pipeline connection
├─ Input Section
│  ├─ Image upload
│  ├─ Edit request
│  └─ Parameters
├─ Pipeline Execution
│  ├─ Async calls to services
│  ├─ Progress tracking
│  └─ Error handling
├─ Output Section
│  ├─ Edited image
│  ├─ Upscaled image
│  └─ Comparison view
└─ Metrics Dashboard
   ├─ Per-stage timing
   ├─ GPU assignment
   └─ Export functionality
```

**Async Architecture**:
- Gradio uses synchronous interface
- We wrap async pipeline with `asyncio.run()`
- Maintains responsiveness during long operations

**Metrics Collection**:
```python
class MetricsCollector:
    def track_request(stage, duration, **kwargs):
        # Store in list
    
    def get_summary() → DataFrame:
        # Aggregate statistics
    
    def export_logs(path):
        # Save to JSON
```

## Data Flow

### End-to-End Request

```
User Upload
    │
    ▼
┌─────────────────┐
│  Gradio Frontend│
│  (Port 7860)    │
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────┐
│  VLLMVisionClient│
│  → vLLM (8000)  │
└────────┬────────┘
         │ instructions
         ▼
┌─────────────────┐
│ RayServeClient  │
│ → Ray (8001)    │
│   └─ GPU N      │
└────────┬────────┘
         │ edited_image
         ▼
┌─────────────────┐
│ TiledUpscaler   │
│ → Ray (8001)    │
│   └─ GPU M×N    │
└────────┬────────┘
         │ upscaled_image
         ▼
┌─────────────────┐
│  Gradio Frontend│
│  Display Result │
└─────────────────┘
```

**Latency Breakdown**:
- Frontend → vLLM: ~1ms (local)
- vLLM processing: 2-5s
- vLLM → Frontend: ~1ms
- Frontend → Ray: ~1ms
- Ray processing: 5-15s (depends on steps)
- Ray → Frontend: ~1ms
- Frontend → Ray (upscale): ~1ms
- Ray tiled processing: 15-30s
- Ray → Frontend: ~1ms

**Total**: 22-51s for full pipeline

## Scalability

### Horizontal Scaling

**Add More GPUs**:
```yaml
# Scale from 6 to 12 replicas
models:
  qwen_image_edit:
    num_replicas: 12
gpu_allocation:
  image_edit_service: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
```

**Multi-Machine**:
- Run vLLM on Machine A
- Run Ray Serve on Machine B
- Run Frontend on Machine C
- Update endpoints in config

### Autoscaling

Ray Serve automatically scales replicas:
- **Scale up**: If queue depth > target
- **Scale down**: If idle for downscale_delay_s

```yaml
autoscaling_config:
  target_num_ongoing_requests_per_replica: 1
  min_replicas: 3   # Always keep 3 warm
  max_replicas: 6   # Never exceed 6
  upscale_delay_s: 30    # Scale up quickly
  downscale_delay_s: 600 # Scale down slowly
```

### Load Balancing

**Ray Serve Strategy**:
- Round-robin by default
- Can configure: `deployment_strategy="LEAST_QUEUE"`
- Monitors per-replica queue depth
- Routes to replica with fewest requests

**vLLM Strategy**:
- Continuous batching
- Fills GPU capacity with multiple requests
- No manual load balancing needed

## Reliability

### Fault Tolerance

**Ray Serve**:
- Replica failure → automatic restart
- Health checks every 10s
- Failed replica replaced within 30s

**vLLM**:
- Process crash → restart via systemd/supervisor
- No built-in HA (use multiple instances + load balancer)

**Frontend**:
- Stateless → easy to restart
- Metrics stored in-memory (lost on restart)
- TODO: Persistent metrics store

### Error Handling

**Network Errors**:
- Timeout after 300s (configurable)
- Return partial results if possible
- Display user-friendly error message

**GPU OOM**:
- Caught at replica level
- Return error to client
- Replica remains healthy for next request

**Model Loading Errors**:
- Fail fast during initialization
- Log detailed error message
- Prevent accepting requests

## Monitoring

### Health Checks

**Service Level**:
- `/health` endpoint on each service
- Returns 200 OK if healthy
- Includes replica metadata

**GPU Level**:
- Query `nvidia-smi` every 5s
- Track utilization, memory, temperature
- Alert if > 85°C or > 95% memory

### Metrics

**Collected**:
- Request count (per stage)
- Latency (p50, p95, p99)
- Throughput (requests/min)
- GPU utilization (%)
- Queue depth

**Export**:
- JSON format
- Schema: `{timestamp, stage, duration, gpu_id, ...}`
- Stored in `outputs/metrics/`

**Future**:
- Prometheus exporter
- Grafana dashboards
- Real-time alerting

## Security

**Current**:
- No authentication on services
- Services bind to 0.0.0.0 (all interfaces)
- Suitable for internal/trusted networks

**TODO**:
- API key authentication
- TLS/HTTPS support
- Rate limiting
- Input validation

## Performance Tuning

### GPU Utilization

**Target**: 80-95% utilization on all GPUs

**If low utilization**:
- Increase `max_concurrent_queries`
- Reduce `num_replicas` (more load per replica)
- Increase inference steps (longer processing)

**If high utilization + slow**:
- Add more replicas
- Reduce inference steps
- Lower image resolution

### Memory Optimization

**vLLM**:
- `gpu_memory_utilization`: 0.85-0.95
- `max_model_len`: Reduce if OOM

**Ray Serve**:
- Use FP16 instead of BF16 (saves ~25% memory)
- Reduce batch size (not exposed currently)

### Latency Reduction

**Fastest setup**:
```yaml
pipeline:
  edit_defaults:
    num_inference_steps: 10  # Minimum quality
  tiling:
    tile_width: 1024  # Fewer tiles
    overlap: 32       # Less blending
```

**Quality setup**:
```yaml
pipeline:
  edit_defaults:
    num_inference_steps: 30  # Better quality
  tiling:
    tile_width: 512   # More tiles
    overlap: 64       # Smoother seams
```

## Future Enhancements

### Short-term
- [ ] Prometheus metrics export
- [ ] Docker Compose deployment
- [ ] Multi-LoRA dynamic loading
- [ ] Result caching (Redis)

### Medium-term
- [ ] Batch processing API
- [ ] A/B testing framework
- [ ] Model quantization (INT8)
- [ ] CPU fallback mode

### Long-term
- [ ] Multi-region deployment
- [ ] Federated learning
- [ ] Model distillation
- [ ] Edge deployment

## References

- **Ray Serve**: https://docs.ray.io/en/latest/serve/
- **vLLM**: https://docs.vllm.ai/
- **Qwen Models**: https://huggingface.co/Qwen
- **Gradio**: https://gradio.app/docs/

---

**Architecture Version**: 1.0.0  
**Last Updated**: 2025-01-01

