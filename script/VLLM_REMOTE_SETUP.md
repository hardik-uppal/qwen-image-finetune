# vLLM Remote Server Setup Guide

Instructions for running vLLM on a remote machine (192.168.0.20) to serve vision models.

## On Remote Machine (192.168.0.20)

### 1. Install vLLM

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install vLLM with CUDA support
pip install vllm

# Or for specific CUDA version (e.g., CUDA 12.1)
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

### 2. Start vLLM Server

#### Option A: Qwen2.5-VL-7B (Recommended for most GPUs)

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096 \
  --max-num-seqs 16
```

**GPU Requirements:**
- VRAM: ~16GB
- Recommended: RTX 3090, RTX 4090, A5000, or better

#### Option B: Qwen2.5-VL-72B (For high-end GPUs)

```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --max-num-seqs 8
```

**GPU Requirements:**
- VRAM: ~80GB+ (2x A100 40GB or similar)
- Multi-GPU setup recommended

#### Option C: Quantized Version (For GPUs with less VRAM)

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype half \
  --quantization awq \
  --max-model-len 2048 \
  --max-num-seqs 8
```

**GPU Requirements:**
- VRAM: ~8GB
- Works on: RTX 3060, RTX 4060, etc.

### 3. Run in Background (Optional)

```bash
# Using nohup
nohup vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096 > vllm.log 2>&1 &

# Or using screen
screen -S vllm
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000 --dtype auto
# Press Ctrl+A then D to detach
```

### 4. Verify Server is Running

```bash
# Check if the server is responding
curl http://localhost:8000/v1/models

# Should return something like:
# {"object":"list","data":[{"id":"Qwen/Qwen2.5-VL-7B-Instruct",...}]}
```

## Network Configuration

### Allow Port 8000 Through Firewall

#### On Ubuntu/Debian:
```bash
sudo ufw allow 8000/tcp
sudo ufw reload
```

#### On CentOS/RHEL:
```bash
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

### Test from Local Machine

From your local machine, test the connection:

```bash
# Check if the endpoint is reachable
curl http://192.168.0.20:8000/v1/models

# Test with a simple completion (text-only)
curl http://192.168.0.20:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Run the Gradio App

On your local machine:

```bash
# The script is already configured to use http://192.168.0.20:8000/v1
python script/gradio_prompt_generator.py

# Open browser at http://localhost:7860
```

## Troubleshooting

### Connection Refused
- Ensure vLLM is running: `ps aux | grep vllm`
- Check port is listening: `netstat -tuln | grep 8000`
- Verify firewall allows connections: `sudo ufw status`

### Out of Memory
- Reduce `--max-model-len` (e.g., 2048 instead of 4096)
- Reduce `--max-num-seqs` (e.g., 4 instead of 16)
- Use a smaller model (7B instead of 72B)
- Enable quantization with `--quantization awq`

### Slow Performance
- Increase `--max-num-seqs` for better batching
- Use `--dtype half` or `--dtype bfloat16`
- Ensure CUDA is properly installed: `nvidia-smi`

### Model Download Issues
```bash
# Pre-download the model
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct

# Or set HuggingFace cache directory
export HF_HOME=/path/to/large/disk
```

## Performance Tips

### Optimal Settings for RTX 4090 (24GB VRAM):
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.95
```

### Optimal Settings for A100 (80GB VRAM):
```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90
```

## Monitoring

### Check vLLM logs:
```bash
# If running in screen
screen -r vllm

# If running with nohup
tail -f vllm.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Performance metrics:
```bash
# vLLM provides metrics at
curl http://192.168.0.20:8000/metrics
```

## Security Notes

- vLLM doesn't have built-in authentication
- Only expose on trusted networks
- Consider using a reverse proxy (nginx) with authentication if needed
- Use VPN for access over untrusted networks

## Alternative: Docker Deployment

```bash
# Pull vLLM Docker image
docker pull vllm/vllm-openai:latest

# Run container
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
```

