#!/bin/bash
# Script to start vLLM server for Qwen Vision-Language models

set -e

# Default values
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL                      Model name (default: Qwen/Qwen2.5-VL-7B-Instruct)"
            echo "  --port PORT                        Port number (default: 8000)"
            echo "  --host HOST                        Host address (default: 0.0.0.0)"
            echo "  --tensor-parallel-size SIZE        Number of GPUs for tensor parallelism (default: 1)"
            echo "  --gpu-memory-utilization RATIO     GPU memory utilization ratio (default: 0.9)"
            echo ""
            echo "Environment variables:"
            echo "  CUDA_VISIBLE_DEVICES              Specify which GPUs to use"
            echo ""
            echo "Examples:"
            echo "  # Start with 7B model on single GPU"
            echo "  $0"
            echo ""
            echo "  # Start with 72B model on 4 GPUs"
            echo "  $0 --model Qwen/Qwen2.5-VL-72B-Instruct --tensor-parallel-size 4"
            echo ""
            echo "  # Start on specific GPU"
            echo "  CUDA_VISIBLE_DEVICES=0 $0"
            echo ""
            echo "  # Start with custom port"
            echo "  $0 --port 8080"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if vllm is installed
if ! command -v vllm &> /dev/null; then
    echo "❌ Error: vllm is not installed"
    echo "Install it with: pip install vllm"
    exit 1
fi

# Display configuration
echo "🚀 Starting vLLM Server"
echo "======================="
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
fi
echo "======================="
echo ""

# Start vLLM server
echo "Starting server... (Press Ctrl+C to stop)"
echo ""

vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"


