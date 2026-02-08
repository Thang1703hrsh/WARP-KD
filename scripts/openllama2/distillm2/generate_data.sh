#!/bin/bash
# ==============================================
# DistiLLM-2 Data Generation Script for OpenLLaMA
# Generates teacher-student response pairs
# ==============================================

set -e

BASE_PATH="."

# Model configuration
# Option 1: Use SFT checkpoints (recommended after fine-tuning)
TEACHER_MODEL="${BASE_PATH}/results/openllama2/train/sft/openllama2-7B/"
STUDENT_MODEL="${BASE_PATH}/results/openllama2/train/init/openllama2-3B/"

# Option 2: Use HuggingFace models directly (uncomment to use)
# TEACHER_MODEL="openlm-research/open_llama_7b"
# STUDENT_MODEL="openlm-research/open_llama_3b"

DATA_DIR="${BASE_PATH}/data/dolly"
OUTPUT_DIR="${BASE_PATH}/data/distillm2/openllama2"

mkdir -p ${OUTPUT_DIR}

echo "================================================"
echo "DistiLLM-2 Data Generation for OpenLLaMA"
echo "Teacher: ${TEACHER_MODEL}"
echo "Student: ${STUDENT_MODEL}"
echo "================================================"

# Generate teacher and student responses for train and dev splits
echo ""
echo "Generating teacher and student responses..."

for SPLIT in train dev; do
    echo "  Processing ${SPLIT} split..."
    
    python generate_distillm2_data.py \
        --teacher-model ${TEACHER_MODEL} \
        --student-model ${STUDENT_MODEL} \
        --data-path ${DATA_DIR}/${SPLIT}.jsonl \
        --output-dir ${OUTPUT_DIR} \
        --split-type ${SPLIT} \
        --temperature 1.0 \
        --top-p 0.95 \
        --max-tokens 512 \
        --use-vllm
done

echo ""
echo "================================================"
echo "Reformatting data for training..."
echo "================================================"

python reformat_distillm2_data.py \
    --input-dir ${OUTPUT_DIR} \
    --output-dir ${OUTPUT_DIR}/formatted

echo ""
echo "================================================"
echo "Data generation complete!"
echo "Output directory: ${OUTPUT_DIR}/formatted"
echo "  - train.json (reformatted paired data)"
echo "================================================"
echo "Raw outputs saved in: ${OUTPUT_DIR}"
echo "  - generated_train_teacher.jsonl"
echo "  - generated_train_student.jsonl"
echo "  - generated_dev_teacher.jsonl"
echo "  - generated_dev_student.jsonl"
echo "================================================"


