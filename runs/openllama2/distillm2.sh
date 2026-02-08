#!/bin/bash

# DistiLLM-2 complete pipeline for OpenLLaMA (7B → 3B)
# Usage: bash runs/openllama2/distillm2.sh [BASE_PATH] [MASTER_PORT] [NUM_GPUS]

BASE_PATH=${1-"."}
MASTER_PORT=${2-2012}
NUM_GPUS=${3-16}

echo "========================================"
echo "DistiLLM-2 Pipeline: OpenLLaMA (7B → 3B)"
echo "========================================"
echo "BASE_PATH: ${BASE_PATH}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "========================================"

# One time run, can disable for subsequent runs
# bash ${BASE_PATH}/install.sh

# Process OpenWebText dataset (one time)
# python3 tools/get_openwebtext.py
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
#     --data-dir ${BASE_PATH}/data/openwebtext \
#     --processed-data-dir ${BASE_PATH}/processed_data/openwebtext/openllama2/512/ \
#     --model-path openlm-research/open_llama_3b_v2 \
#     --max-length 512 \
#     --train-num 22870 \
#     --data-process-workers 32 \
#     --dev-num 1000 \

# Process Dolly dataset (one time)
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/dolly/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
#     --model-path openlm-research/open_llama_3b_v2 \
#     --data-process-workers 32 \
#     --max-prompt-length 256 \
#     --dev-num 1000 \
#     --model-type openllama2

# Base checkpoints training (one time)
# bash ${BASE_PATH}/scripts/openllama2/sft/sft_7B_lora.sh ${BASE_PATH} ${MASTER_PORT} 1
# bash ${BASE_PATH}/scripts/openllama2/init/init_3B_lora.sh ${BASE_PATH} ${MASTER_PORT} 1

# Step 1: Generate DistiLLM-2 data (teacher-student pairs)
echo "Step 1/3: Generating DistiLLM-2 data..."
bash ${BASE_PATH}/scripts/openllama2/distillm2/generate_data.sh ${BASE_PATH}

# Step 2: Train with DistiLLM-2
echo "Step 2/2: Training with DistiLLM-2..."
bash ${BASE_PATH}/scripts/openllama2/distillm2/distillm2_3B_7B_lora.sh ${BASE_PATH} ${MASTER_PORT} ${NUM_GPUS}

# Run evaluation on multiple benchmarks with different seeds
EVAL_PORT=2040
DEVICE=0
for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_${benchmark}_lora.sh ./ ${EVAL_PORT} 1 openllama2-3B distillm2/3B_7B openlm-research/open_llama_3b_v2 --seed $seed --eval-batch-size 32
    done
done

echo "========================================"
echo "DistiLLM-2 Pipeline Complete!"
echo "Model: ${BASE_PATH}/results/openllama2/train/distillm2/3B_7B"
echo "Metrics: ${BASE_PATH}/results/openllama2/eval/distillm2/metrics.json"
echo "========================================"
