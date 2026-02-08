BASE_PATH=${1-"."}

# one time run, can disable for subsequent runs
# bash ${BASE_PATH}/install.sh

# load openwebtext dataset
# python3 tools/get_openwebtext.py
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/openwebtext \
    --processed-data-dir ${BASE_PATH}/processed_data/openwebtext/llama2/512/ \
    --model-path meta-llama/Llama-2-7b-hf \
    --max-length 512 \
    --train-num 22870 \
    --data-process-workers 32 \
    --dev-num 1000 \

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path meta-llama/Llama-2-7b-hf \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type llama2

# base ckpts training
# bash ${BASE_PATH}/scripts/llama2/sft/sft_13B_lora.sh ${BASE_PATH} 2012 1
# bash ${BASE_PATH}/scripts/llama2/init/init_7B_lora.sh ${BASE_PATH} 2012 1

# FDD
bash ${BASE_PATH}/scripts/llama2/fdd/fdd_7B_13B_teacher_lora.sh ${BASE_PATH} 2012 1

MASTER_PORT=2040
DEVICE=0
for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/llama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 llama2-7B fdd/7B_13B meta-llama/Llama-2-7b-hf --seed $seed --eval-batch-size 32
    done
done