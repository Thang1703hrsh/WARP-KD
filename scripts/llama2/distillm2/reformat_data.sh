#!/bin/bash

# ================================================
# Reformat DistiLLM-2 Data for LLaMA-2 (13B → 7B)
# ================================================
# This script reformats separately generated teacher and student responses
# into the contrastive paired format needed for DistiLLM-2 training

BASE_PATH=${1-"."}

# Input directories (raw generated outputs)
INPUT_DIR="${BASE_PATH}/data/distillm2/llama2"

# Output directory (formatted contrastive pairs)
OUTPUT_DIR="${BASE_PATH}/data/distillm2/llama2/formatted"

echo "================================================"
echo "Reformatting DistiLLM-2 Data: LLaMA-2 13B → 7B"
echo "================================================"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "================================================"

# Run reformatting script
python reformat_distillm2_data.py \
    --input-dir ${INPUT_DIR} \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "================================================"
echo "Reformatting complete!"
echo "================================================"
echo "Formatted data saved to: ${OUTPUT_DIR}"
echo "Files created:"
echo "  - train.json (contrastive pairs)"
echo "  - dev.json (contrastive pairs)"
echo "  - train/ (HuggingFace Arrow format)"
echo "  - test/ (HuggingFace Arrow format)"
echo "================================================"
echo ""
echo "Next step: Run training with:"
echo "  bash scripts/llama2/distillm2/train.sh"
echo "================================================"
