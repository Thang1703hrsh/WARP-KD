#!/bin/bash

# DistiLLM-2 Training for LLaMA-2 (13B → 7B)

BASE_PATH=${1-"/home/MiniLLM"}
MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
GPUS_PER_NODE=${3-1}

# Model configuration
STUDENT_MODEL="meta-llama/Llama-2-7b-hf"
TEACHER_MODEL="meta-llama/Llama-2-13b-hf"
CKPT_NAME="llama2-7B"
TEACHER_CKPT_NAME="llama2-13B"
PEFT_CKPT="${BASE_PATH}/results/llama2/train/init/${CKPT_NAME}/"
TEACHER_PEFT_CKPT="${BASE_PATH}/results/llama2/train/sft/${TEACHER_CKPT_NAME}/"

# Data configuration
DISTILLM2_DATA_DIR="${BASE_PATH}/data/distillm2/llama2/formatted"
PROMPT_DATA_DIR="${BASE_PATH}/data/dolly"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/llama2/512/22.87K/"

# Training configuration
EPOCHS=10
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
SAVE_DIR="${BASE_PATH}/results/llama2/train/distillm2/7B_13B"

# DeepSpeed configuration
DEEPSPEED_CONFIG="configs/deepspeed/ds_config_zero2.json"

# Seed
SEED=1031

echo "================================================"
echo "Starting DistiLLM-2 Training: LLaMA-2 13B → 7B"
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
    --peft lora \
    --peft-path ${PEFT_CKPT} \
    --teacher-peft-path ${TEACHER_PEFT_CKPT} \
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
    --model-type llama \
    --kd-ratio 0.5 \
    --teacher-model-fp16 \
    --gradient-checkpointing

echo "================================================"
echo "Training complete!"
echo "Model saved to: ${SAVE_DIR}"
echo "================================================"
