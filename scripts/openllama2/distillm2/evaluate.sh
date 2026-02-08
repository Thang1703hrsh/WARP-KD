#!/bin/bash

# Evaluate DistiLLM-2 trained OpenLLaMA model

BASE_PATH=${1-"/home/MiniLLM"}
MASTER_PORT=${2-2012}
GPUS_PER_NODE=${3-8}

# Model configuration
MODEL_PATH="openlm-research/open_llama_3b_v2"
PEFT_PATH="${BASE_PATH}/results/openllama2/train/distillm2/3B_7B"  # Trained LoRA
CKPT_NAME="openllama2-3B-distillm2"

# Data configuration
DATA_DIR="${BASE_PATH}/contra-kd/data/dolly"
DATA_NAMES="dolly"
DEV_NUM=1000

# Evaluation configuration
EVAL_BATCH_SIZE=8
MAX_LENGTH=512
MAX_PROMPT_LENGTH=256

# Generation parameters
TEMPERATURE=0.7
TOP_P=1.0
TOP_K=0
DO_SAMPLE=true
NO_REPEAT_NGRAM_SIZE=0
REPETITION_PENALTY=1.0

# Output configuration
EVAL_SAVE_DIR="${BASE_PATH}/results/openllama2/eval/distillm2"

# DeepSpeed configuration
DEEPSPEED_CONFIG="configs/deepspeed/ds_config_zero2.json"

echo "================================================"
echo "DistiLLM-2 Evaluation: OpenLLaMA 3B (Distilled)"
echo "================================================"
echo "Base Model: ${MODEL_PATH}"
echo "LoRA: ${PEFT_PATH}"
echo "Data: ${DATA_DIR}"
echo "Output: ${EVAL_SAVE_DIR}"
echo "================================================"

mkdir -p ${EVAL_SAVE_DIR}

deepspeed --num_gpus=${GPUS_PER_NODE} evaluate_distillm2.py \
    --type eval_distillm2 \
    --model-path ${MODEL_PATH} \
    --peft lora \
    --peft-path ${PEFT_PATH} \
    --ckpt-name ${CKPT_NAME} \
    --data-dir ${DATA_DIR} \
    --data-names ${DATA_NAMES} \
    --dev-num ${DEV_NUM} \
    --eval-batch-size ${EVAL_BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --max-prompt-length ${MAX_PROMPT_LENGTH} \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P} \
    --top-k ${TOP_K} \
    --do-sample \
    --no-repeat-ngram-size ${NO_REPEAT_NGRAM_SIZE} \
    --repetition-penalty ${REPETITION_PENALTY} \
    --save ${EVAL_SAVE_DIR} \
    --deepspeed_config ${DEEPSPEED_CONFIG} \
    --n-gpu ${GPUS_PER_NODE} \
    --model-type llama \
    --num-workers 1

echo "================================================"
echo "Evaluation complete!"
echo "Results saved to: ${EVAL_SAVE_DIR}"
echo "  - metrics.json: ROUGE-L and metrics"
echo "  - answers.jsonl: Generated answers"
echo "  - preds.txt: Prompt-response pairs"
echo "================================================"

# Display results
if [ -f "${EVAL_SAVE_DIR}/metrics.json" ]; then
    echo ""
    echo "Metrics:"
    cat ${EVAL_SAVE_DIR}/metrics.json
fi
