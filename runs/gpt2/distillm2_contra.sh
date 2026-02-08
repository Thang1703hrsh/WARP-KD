#!/bin/bash

# DistiLLM-2 + Contra complete pipeline for GPT-2 (1.5B → 0.1B)
# Usage: bash runs/gpt2/distillm2_contra.sh [BASE_PATH] [MASTER_PORT] [NUM_GPUS]

BASE_PATH=${1-"."}
MASTER_PORT=${2-2012}
NUM_GPUS=${3-16}

echo "========================================"
echo "DistiLLM-2 + Contra Pipeline: GPT-2 (1.5B → 0.1B)"
echo "========================================"
echo "BASE_PATH: ${BASE_PATH}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "========================================"

# One time run, can disable for subsequent runs
# bash ${BASE_PATH}/install.sh

# Process Dolly dataset (one time)
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/dolly/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
#     --model-path gpt2 \
#     --data-process-workers 32 \
#     --max-prompt-length 256 \
#     --dev-num 1000 \
#     --model-type gpt2

# Base checkpoints training (one time)
# bash ${BASE_PATH}/scripts/gpt2/sft/sft_xlarge.sh ${BASE_PATH} ${MASTER_PORT} 1
# bash ${BASE_PATH}/scripts/gpt2/init/init_base.sh ${BASE_PATH} ${MASTER_PORT} 1

# Step 1: Generate DistiLLM-2 data (teacher-student pairs)
echo "Step 1/3: Generating DistiLLM-2 data..."
bash ${BASE_PATH}/scripts/gpt2/distillm2/generate_data.sh ${BASE_PATH}

bash ${BASE_PATH}/scripts/gpt2/distillm2/train_velocity_field_distillm2.sh ${BASE_PATH} ${MASTER_PORT} ${NUM_GPUS}

# Step 2: Train with DistiLLM-2 + Contra
echo "Step 2/2: Training with DistiLLM-2 + Contra..."
bash ${BASE_PATH}/scripts/gpt2/distillm2/contra_0.1B_1.5B.sh ${BASE_PATH} ${MASTER_PORT} ${NUM_GPUS}

# Run evaluation on multiple benchmarks with different seeds
EVAL_PORT=2040
DEVICE=0
for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_${benchmark}.sh ./ ${EVAL_PORT} 1 contra/distillm2/0.1B_1.5B --seed $seed --eval-batch-size 64
    done
done

echo "========================================"
echo "DistiLLM-2 + Contra Pipeline Complete!"
echo "Model: ${BASE_PATH}/results/gpt2/train/contra/distillm2/0.1B_1.5B"
echo "Metrics: ${BASE_PATH}/results/gpt2/eval/contra/distillm2/metrics.json"
echo "========================================"
