#!/bin/bash

# DistiLLM-2 Data Generation Script for Contra-KD
# This script generates paired teacher-student responses for DistiLLM-2 training

# Configuration
TEACHER_MODEL="meta-llama/Llama-2-7b-hf"
STUDENT_MODEL="meta-llama/Llama-2-1.3b-hf"
DATA_PATH="HuggingFaceH4/ultrachat_200k"
OUTPUT_DIR="data/distillm2_paired"
SPLIT="train_sft"
NUM_SAMPLES=50000  # Adjust as needed

# Generation parameters
TEMPERATURE=0.8
TOP_P=0.95
MAX_TOKENS=1024
SEED=42
TENSOR_PARALLEL_SIZE=2  # Use 2 GPUs for larger models

# Step 1: Generate train data
echo "================================================"
echo "Generating TRAIN data..."
echo "================================================"

python generate_distillm2_data.py \
    --teacher-model ${TEACHER_MODEL} \
    --student-model ${STUDENT_MODEL} \
    --data-path ${DATA_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --split-type train \
    --num-samples ${NUM_SAMPLES} \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P} \
    --max-tokens ${MAX_TOKENS} \
    --seed ${SEED} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --use-vllm

# Step 2: Generate dev data
echo "================================================"
echo "Generating DEV data..."
echo "================================================"

python generate_distillm2_data.py \
    --teacher-model ${TEACHER_MODEL} \
    --student-model ${STUDENT_MODEL} \
    --data-path ${DATA_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --split "test_sft" \
    --split-type dev \
    --num-samples 1000 \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P} \
    --max-tokens ${MAX_TOKENS} \
    --seed ${SEED} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --use-vllm

echo "================================================"
echo "Data generation complete!"
echo "Output directory: ${OUTPUT_DIR}"
echo "================================================"
