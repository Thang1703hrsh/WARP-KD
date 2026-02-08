#!/bin/bash

MASTER_PORT=2040
DEVICE=${1}
ckpt=${2}

# dolly eval
for seed in 10 20 30 40 50
do
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/llama2/eval/eval_main_dolly_lora.sh ./ ${MASTER_PORT} 1 llama2-7B ${ckpt} meta-llama/Llama-2-7b-hf --seed $seed  --eval-batch-size 64
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/llama2/eval/eval_main_self_inst_lora.sh ./ ${MASTER_PORT} 1 llama2-7B ${ckpt} meta-llama/Llama-2-7b-hf --seed $seed  --eval-batch-size 64
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/llama2/eval/eval_main_vicuna_lora.sh ./ ${MASTER_PORT} 1 llama2-7B ${ckpt} meta-llama/Llama-2-7b-hf --seed $seed  --eval-batch-size 64
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/llama2/eval/eval_main_sinst_lora.sh ./ ${MASTER_PORT} 1 llama2-7B ${ckpt} meta-llama/Llama-2-7b-hf --seed $seed  --eval-batch-size 64
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/llama2/eval/eval_main_uinst_lora.sh ./ ${MASTER_PORT} 1 llama2-7B ${ckpt} meta-llama/Llama-2-7b-hf --seed $seed  --eval-batch-size 64
done