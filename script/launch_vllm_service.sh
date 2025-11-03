#!/bin/bash
#
# Launch vLLM service for Qwen2.5-VL
# Reads configuration from production_pipeline.yaml
#

set -e

# Default values
CONFIG_FILE="${CONFIG_FILE:-configs/production_pipeline.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "======================================"
echo "  vLLM Service Launcher"
echo "======================================"
echo "Config file: $CONFIG_FILE"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse config using Python
read -r MODEL GPUS TP_SIZE MAX_LEN GPU_MEM PORT LORA_PATH LORA_NAME <<< $(python3 - <<'EOF'
import sys
import yaml

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

vl_config = config['models']['qwen_vl']
gpu_alloc = config['gpu_allocation']['vl_service']
endpoint = config['services']['vl_endpoint']

# Extract port from endpoint
port = endpoint.split(':')[-1].split('/')[0]

# Format GPUs as comma-separated string
gpus = ','.join(str(g) for g in gpu_alloc)

# Get LoRA info
lora_path = vl_config.get('lora_path', 'null')
lora_name = vl_config.get('lora_name', 'vl_lora')

print(
    vl_config['model_name'],
    gpus,
    vl_config['tensor_parallel_size'],
    vl_config['max_model_len'],
    vl_config['gpu_memory_utilization'],
    port,
    lora_path if lora_path else 'null',
    lora_name
)
EOF
echo "$CONFIG_FILE")

echo "Model: $MODEL"
echo "GPUs: $GPUS"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Max Model Length: $MAX_LEN"
echo "GPU Memory Utilization: $GPU_MEM"
echo "Port: $PORT"
echo "LoRA Path: $LORA_PATH"
if [ "$LORA_PATH" != "null" ]; then
    echo "LoRA Name: $LORA_NAME"
fi
echo ""

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "Starting vLLM server..."
echo "Access at: http://localhost:$PORT/v1"
echo ""

# Build vLLM command
VLLM_CMD="vllm serve \"$MODEL\" \
    --tensor-parallel-size \"$TP_SIZE\" \
    --max-model-len \"$MAX_LEN\" \
    --gpu-memory-utilization \"$GPU_MEM\" \
    --port \"$PORT\" \
    --host \"0.0.0.0\" \
    --served-model-name \"$(basename $MODEL)\" \
    --trust-remote-code"

# Add LoRA support if path is provided
if [ "$LORA_PATH" != "null" ] && [ -n "$LORA_PATH" ]; then
    echo "Enabling LoRA support with adapter: $LORA_NAME=$LORA_PATH"
    VLLM_CMD="$VLLM_CMD --enable-lora --lora-modules ${LORA_NAME}=${LORA_PATH}"
fi

# Launch vLLM
eval "$VLLM_CMD" 2>&1 | tee logs/vllm_service.log

