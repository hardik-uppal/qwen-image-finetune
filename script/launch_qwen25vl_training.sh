#!/bin/bash
# Launch Qwen2.5-VL training with torchrun for multi-GPU support

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Qwen2.5-VL Prompt Generation Training${NC}"
echo -e "${BLUE}================================${NC}\n"

# Activate conda environment
echo -e "${GREEN}Activating conda environment: myenv${NC}"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Check if config file exists
CONFIG_FILE="${1:-configs/qwen25vl_prompt_gen.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}Error: Config file not found: $CONFIG_FILE${NC}"
    echo "Usage: ./script/launch_qwen25vl_training.sh [config_file]"
    exit 1
fi

echo -e "${GREEN}Using config: $CONFIG_FILE${NC}\n"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo -e "${GREEN}Using GPUs: 0,1,2,3${NC}\n"

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU may not be available.${NC}"
else
    echo -e "${GREEN}GPU Info:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Launch training with torchrun for 4 GPUs
echo -e "${GREEN}Launching training with torchrun (4 GPUs)...${NC}"
echo -e "${BLUE}Starting training...${NC}\n"

torchrun --nproc_per_node=4 \
    --master_port=29500 \
    script/train_qwen25vl_simple.py \
    --config "$CONFIG_FILE"

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}================================${NC}"

# Show output directory
OUTPUT_DIR=$(grep "output_dir:" "$CONFIG_FILE" | awk '{print $2}')
echo -e "\nModel saved to: ${BLUE}$OUTPUT_DIR${NC}"
echo -e "\nTo run inference:"
echo -e "  ${BLUE}python script/infer_qwen25vl_prompt_gen.py \\\\${NC}"
echo -e "    ${BLUE}--adapter_path $OUTPUT_DIR \\\\${NC}"
echo -e "    ${BLUE}--image /path/to/image.jpg${NC}"