#!/bin/bash

# DistiLLM-2 + Contra-KD Training Script
# This script runs the integrated DistiLLM-2 training with Contra-KD framework

# Model configuration
STUDENT_MODEL="meta-llama/Llama-2-1.3b-hf"
TEACHER_MODEL="meta-llama/Llama-2-7b-hf"

# Data configuration
DISTILLM2_DATA_DIR="data/distillm2_formatted"

# Training configuration
EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-4
MAX_LENGTH=512
MAX_PROMPT_LENGTH=256

# DistiLLM-2 specific
LOSS_TYPE="distillm_v2"  # or "distillm_v1"
GRADUAL_BETA=""  # Add "--gradual-beta" to enable
BASE_ALPHA_1=0.1
BASE_ALPHA_2=0.1

# Output configuration
SAVE_DIR="outputs/distillm2_contra_llama2_1.3b"
CKPT_NAME="llama2-1.3b"
TEACHER_CKPT_NAME="llama2-7b"

# DeepSpeed configuration
DEEPSPEED_CONFIG="configs/deepspeed/ds_config_zero2.json"

# Number of GPUs
N_GPU=4
N_NODES=1

echo "================================================"
echo "Starting DistiLLM-2 + Contra-KD Training"
echo "================================================"
echo "Student Model: ${STUDENT_MODEL}"
echo "Teacher Model: ${TEACHER_MODEL}"
echo "Loss Type: ${LOSS_TYPE}"
echo "Data: ${DISTILLM2_DATA_DIR}"
echo "Output: ${SAVE_DIR}"
echo "================================================"

deepspeed --num_gpus=${N_GPU} train_distillm2_contra.py \
    --type distillm2-v2 \
    --model-path ${STUDENT_MODEL} \
    --teacher-model-path ${TEACHER_MODEL} \
    --ckpt-name ${CKPT_NAME} \
    --teacher-ckpt-name ${TEACHER_CKPT_NAME} \
    --distillm2-data-dir ${DISTILLM2_DATA_DIR} \
    --distillm2-loss-type ${LOSS_TYPE} \
    --base-alpha-1 ${BASE_ALPHA_1} \
    --base-alpha-2 ${BASE_ALPHA_2} \
    ${GRADUAL_BETA} \
    --save ${SAVE_DIR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr ${LR} \
    --max-length ${MAX_LENGTH} \
    --max-prompt-length ${MAX_PROMPT_LENGTH} \
    --save-interval 500 \
    --log-interval 10 \
    --deepspeed_config ${DEEPSPEED_CONFIG} \
    --n-gpu ${N_GPU} \
    --n-nodes ${N_NODES} \
    --seed 42

echo "================================================"
echo "Training complete!"
echo "Model saved to: ${SAVE_DIR}"
echo "================================================"
