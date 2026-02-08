BASE_PATH=${1-"."}

# one time run, can disable for subsequent runs
# bash ${BASE_PATH}/install.sh

# load openwebtext dataset
# python3 tools/get_openwebtext.py
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/openwebtext \
    --processed-data-dir ${BASE_PATH}/processed_data/openwebtext/openllama2/512/ \
    --model-path openlm-research/open_llama_3b_v2 \
    --max-length 512 \
    --train-num 22870 \
    --data-process-workers 32 \
    --dev-num 1000 \

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path openlm-research/open_llama_3b_v2 \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type openllama2

# base ckpts training
# bash ${BASE_PATH}/scripts/openllama2/sft/sft_7B_lora.sh ${BASE_PATH} 2012 1
# bash ${BASE_PATH}/scripts/openllama2/init/init_3B_lora.sh ${BASE_PATH} 2012 1

# FDD
bash ${BASE_PATH}/scripts/openllama2/fdd/fdd_3B_7B_teacher_lora.sh ${BASE_PATH} 2012 1

MASTER_PORT=2040
DEVICE=0
for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B fdd/3B_7B openlm-research/open_llama_3b_v2 --seed $seed --eval-batch-size 32
    done
done