#!/bin/bash
# Run multi-prompt dataset generation for DPO and SFT training
# This script generates multiple prompt variations for ALL control variants in the CSV

# Configuration
INPUT_CSV="workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv"
OUTPUT_DIR="/skynas/01/hardik/qwen-dataset/multi-prompt-train-filtered-2k"
NUM_VARIATIONS=3
BASE_URL="http://192.168.0.20:8000/v1"
MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
CONCURRENCY=3
TEMP_MIN=0.1
TEMP_MAX=0.5

# Parse command line arguments
RESUME_FLAG=""
START_IDX=""
END_IDX=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --start_idx)
            START_IDX="--start_idx $2"
            shift 2
            ;;
        --end_idx)
            END_IDX="--end_idx $2"
            shift 2
            ;;
        --num_variations)
            NUM_VARIATIONS="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--resume] [--start_idx N] [--end_idx N] [--num_variations N] [--concurrency N]"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=================================================="
echo "Multi-Prompt Dataset Generation"
echo "=================================================="
echo "Input CSV: $INPUT_CSV"
echo "Output Dir: $OUTPUT_DIR"
echo "Variations per control: $NUM_VARIATIONS"
echo "Temperature range: $TEMP_MIN - $TEMP_MAX"
echo "Concurrency: $CONCURRENCY"
echo "Model: $MODEL"
echo "=================================================="

# Run the script
python script/generate_multi_prompt_dataset.py \
    --input_csv "$INPUT_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --num_variations $NUM_VARIATIONS \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --temp_min $TEMP_MIN \
    --temp_max $TEMP_MAX \
    --concurrency $CONCURRENCY \
    $RESUME_FLAG \
    $START_IDX \
    $END_IDX

echo ""
echo "=================================================="
echo "Generation complete!"
echo "Check outputs at: $OUTPUT_DIR"
echo "  - sft_dataset.csv: For supervised fine-tuning"
echo "  - dpo_dataset.csv: For preference optimization"
echo "=================================================="

