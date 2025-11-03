#!/bin/bash
# Prepare and validate dataset for Qwen2.5-VL prompt generation training
# This script validates all images and creates JSONL files ready for training

set -e  # Exit on error

# Configuration
CONFIG_FILE="configs/qwen25vl_prompt_gen.yaml"
OUTPUT_DIR="workspace/prepared_data"

# Parse command line arguments
DEBUG_MODE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE="--debug"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--debug] [--config CONFIG_FILE] [--output-dir OUTPUT_DIR]"
            exit 1
            ;;
    esac
done

# Display configuration
echo "================================"
echo "Dataset Preparation Configuration"
echo "================================"
echo "Config file:    $CONFIG_FILE"
echo "Output dir:     $OUTPUT_DIR"
echo "Debug mode:     ${DEBUG_MODE:-false}"
echo "================================"
echo ""

# Run preparation script
python script/prepare_qwen25vl_dataset.py \
    --config "$CONFIG_FILE" \
    ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"} \
    $DEBUG_MODE

echo ""
echo "================================"
echo "Dataset preparation complete!"
echo "================================"
echo "Train JSONL: $OUTPUT_DIR/train.jsonl"
echo "Eval JSONL:  $OUTPUT_DIR/eval.jsonl"
echo ""
echo "Next step: Run training with:"
echo "  bash script/train_qwen25vl_prompt_gen.sh"
echo "================================"

