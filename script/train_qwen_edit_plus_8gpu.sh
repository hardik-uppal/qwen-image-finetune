#!/bin/bash
# Train Qwen Image Edit Plus on 8 GPUs
# Effective batch size: 2 (per GPU) × 4 (grad accum) × 8 (GPUs) = 64

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Qwen Image Edit Plus Training${NC}"
echo -e "${BLUE}8 GPU Multi-Node Setup${NC}"
echo -e "${BLUE}================================${NC}\n"

# Activate conda environment
echo -e "${GREEN}Activating conda environment: myenv${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate myenv

# Check if config file exists
CONFIG_FILE="${1:-configs/qwen_image_edit_plus_custom.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}Error: Config file not found: $CONFIG_FILE${NC}"
    echo "Usage: ./script/train_qwen_edit_plus_8gpu.sh [config_file]"
    exit 1
fi

echo -e "${GREEN}Using config: $CONFIG_FILE${NC}\n"

# Set CUDA devices - all 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo -e "${GREEN}Using GPUs: 0,1,2,3,4,5,6,7${NC}\n"

# Display GPU info
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU Info:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU may not be available.${NC}"
fi

# Training info
echo -e "${BLUE}Training Configuration:${NC}"
echo -e "  GPUs: 8"
echo -e "  Batch size per GPU: 2"
echo -e "  Gradient accumulation: 4"
echo -e "  ${GREEN}Effective batch size: 64${NC}"
echo -e "  Mixed precision: bf16"
echo -e "  LPIPS metrics: enabled (VGG)\n"

# Launch training with accelerate
echo -e "${BLUE}Launching distributed training...${NC}\n"

accelerate launch --config_file accelerate_config.yaml \
    -m src.main --config "$CONFIG_FILE"

# Training complete
echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}================================${NC}"

# Show output directory
OUTPUT_DIR=$(grep "output_dir:" "$CONFIG_FILE" | awk '{print $2}')
echo -e "\nCheckpoints saved to: ${BLUE}$OUTPUT_DIR${NC}"
echo -e "\nTo monitor training:"
echo -e "  ${BLUE}# Watch GPU usage${NC}"
echo -e "  ${BLUE}gpuwatch${NC}  # or: watch -n 1 nvidia-smi"
echo -e ""
echo -e "  ${BLUE}# View wandb logs${NC}"
echo -e "  ${BLUE}wandb online${NC}  # Check project: qwen_edit_plus_custom"

