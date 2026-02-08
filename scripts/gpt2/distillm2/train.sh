#!/bin/bash

# DistiLLM-2 Training for GPT-2 (1.5B → 0.1B)

BASE_PATH=${1-"/home/MiniLLM"}
MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
GPUS_PER_NODE=${3-1}

# Model configuration
STUDENT_MODEL="${BASE_PATH}/results/gpt2/train/init/gpt2-base"
TEACHER_MODEL="${BASE_PATH}/results/gpt2/train/sft/gpt2-xlarge/"
CKPT_NAME="gpt2-base"
TEACHER_CKPT_NAME="gpt2-xlarge"

# Data configuration
DISTILLM2_DATA_DIR="${BASE_PATH}/data/distillm2/gpt2/formatted"
PROMPT_DATA_DIR="${BASE_PATH}/data/dolly"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/22.87K/"

# Training configuration
EPOCHS=20
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
LR=5e-4
MAX_LENGTH=512
MAX_PROMPT_LENGTH=256

# DistiLLM-2 specific
LOSS_TYPE="distillm_v2"
BASE_ALPHA_1=0.1
BASE_ALPHA_2=0.1

# Output configuration
SAVE_DIR="${BASE_PATH}/results/gpt2/train/distillm2/0.1B_1.5B"

# DeepSpeed configuration
DEEPSPEED_CONFIG="configs/deepspeed/ds_config_zero2.json"

# Seed
SEED=10

echo "================================================"
echo "Starting DistiLLM-2 Training: GPT-2 1.5B → 0.1B"
echo "================================================"
echo "Student Model: ${STUDENT_MODEL}"
echo "Teacher Model: ${TEACHER_MODEL}"
echo "Loss Type: ${LOSS_TYPE}"
echo "Data: ${DISTILLM2_DATA_DIR}"
echo "Output: ${SAVE_DIR}"
echo "================================================"

deepspeed --num_gpus=${GPUS_PER_NODE} finetune.py \
    --do-train \
    --type distillm2-v2 \
    --model-path ${STUDENT_MODEL} \
    --teacher-model-path ${TEACHER_MODEL} \
    --ckpt-name ${CKPT_NAME} \
    --teacher-ckpt-name ${TEACHER_CKPT_NAME} \
    --data-dir ${DISTILLM2_DATA_DIR} \
    --lm-data-dir ${LM_DATA_DIR} \
    --distillm2-loss-type ${LOSS_TYPE} \
    --base-alpha-1 ${BASE_ALPHA_1} \
    --base-alpha-2 ${BASE_ALPHA_2} \
    --save ${SAVE_DIR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr ${LR} \
    --lr-decay-style cosine \
    --warmup-iters 100 \
    --max-length ${MAX_LENGTH} \
    --max-prompt-length ${MAX_PROMPT_LENGTH} \
    --save-interval 1000 \
    --eval-interval 1000 \
    --log-interval 10 \
    --eval-gen \
    --eval-batch-size 16 \
    --deepspeed_config ${DEEPSPEED_CONFIG} \
    --n-gpu ${GPUS_PER_NODE} \
    --n-nodes 1 \
    --seed ${SEED} \
    --model-type gpt2 \
    --kd-ratio 0.5 \
    --gradient-checkpointing

echo "================================================"
echo "Training complete!"
echo "Model saved to: ${SAVE_DIR}"
echo "================================================"
