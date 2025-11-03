# External vLLM Setup Guide

This guide shows how to use an external vLLM server (on another machine) with Ray Serve for image editing on this server.

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│   vLLM Server       │         │   This Server        │
│   (Another Machine) │◄────────┤   (Image Editing)    │
│                     │  HTTP   │                      │
│  Qwen2.5-VL-72B     │         │  Ray Serve (8 GPUs)  │
│  Port: 8000         │         │  + Gradio Frontend   │
└─────────────────────┘         └──────────────────────┘
```

## Step 1: Configure vLLM Server URL

Edit `configs/production_pipeline.yaml`:

```yaml
services:
  # Replace YOUR_VLLM_SERVER_IP with actual IP address
  vl_endpoint: "http://192.168.1.100:8000/v1"  # Example
  image_edit_endpoint: "http://localhost:8001"
```

## Step 2: Open Firewall Port (on vLLM Server)

On the **vLLM server** machine, ensure port 8000 is accessible:

### Check if vLLM is listening:
```bash
sudo netstat -tlnp | grep 8000
# or
sudo ss -tlnp | grep 8000
```

### Option A: Using UFW (Ubuntu/Debian)
```bash
sudo ufw allow 8000/tcp
sudo ufw status
```

### Option B: Using firewalld (CentOS/RHEL)
```bash
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
sudo firewall-cmd --list-ports
```

### Option C: Using iptables (Generic Linux)
```bash
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables-save
```

### Test connectivity from this server:
```bash
curl http://YOUR_VLLM_SERVER_IP:8000/v1/models
# Should return list of models
```

## Step 3: Launch Services (Skip vLLM)

On **this server**, launch only Ray Serve and Gradio:

```bash
# Activate environment
conda activate myenv

# Launch with --skip-vllm flag
python script/launch_production_services.py \
    --config configs/production_pipeline.yaml \
    --skip-vllm
```

This will:
- ✅ Start Ray Serve with 8 replicas (one per GPU)
- ✅ Start Gradio frontend
- ❌ Skip vLLM (uses external server instead)

## Step 4: Verify Services

### Check logs:
```bash
# Ray Serve
tail -f logs/ray_serve.log

# Gradio Frontend
tail -f logs/frontend.log
```

### Test endpoints:
```bash
# External vLLM (from this server)
curl http://YOUR_VLLM_SERVER_IP:8000/v1/models

# Local Ray Serve
curl http://localhost:8001/-/healthz
```

### Check GPU usage:
```bash
watch -n 1 nvidia-smi
# Should see all 8 GPUs in use by Ray Serve replicas
```

## Troubleshooting

### Connection Refused
- Check firewall on vLLM server
- Verify vLLM is listening on `0.0.0.0`, not `127.0.0.1`
- Test with: `telnet YOUR_VLLM_SERVER_IP 8000`

### Ray Serve OOM
- Check that all 8 GPUs are visible: `nvidia-smi`
- Verify `CUDA_VISIBLE_DEVICES` not set in your shell
- Each replica should use ~6-8GB VRAM

### vLLM Endpoint Not Found
- Verify URL in config matches vLLM server
- Check vLLM is running: `curl http://YOUR_VLLM_SERVER_IP:8000/health`
- Look for connection errors in `logs/frontend.log`

## Example vLLM Launch Command (for reference)

On the **vLLM server**, you might be running:

```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code
```

Note the `--host 0.0.0.0` to accept external connections!

