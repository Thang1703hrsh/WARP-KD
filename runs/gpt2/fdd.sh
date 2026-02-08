BASE_PATH=${1-"."}

# one time run, can disable for subsequent runs
# bash ${BASE_PATH}/install.sh

# load openwebtext dataset
# python3 tools/get_openwebtext.py
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
#     --data-dir ${BASE_PATH}/data/openwebtext \
#     --processed-data-dir ${BASE_PATH}/processed_data/openwebtext/gpt2/512/ \
#     --model-path gpt2 \
#     --max-length 512 \
#     --train-num 22870 \
#     --data-process-workers 32 \
#     --dev-num 1000 \

# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/data/dolly/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
#     --model-path gpt2 \
#     --data-process-workers 32 \
#     --max-prompt-length 256 \
#     --dev-num 1000 \
#     --model-type gpt2

# base ckpts training
# bash ${BASE_PATH}/scripts/gpt2/sft/sft_xlarge.sh ${BASE_PATH} 2012 1
# bash ${BASE_PATH}/scripts/gpt2/init/init_base.sh ${BASE_PATH} 2012 1

# FDD
bash ${BASE_PATH}/scripts/gpt2/fdd/fdd_base.sh ${BASE_PATH} 2012 1

MASTER_PORT=2040
DEVICE=0
for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_${benchmark}.sh ./ ${MASTER_PORT} 1 fdd/base --seed $seed --eval-batch-size 64
    done
done

# FDD + Contra
# stage 1
bash ${BASE_PATH}/scripts/gpt2/train_velocity_field.sh ${BASE_PATH} 2012 1
# stage 2
bash ${BASE_PATH}/scripts/gpt2/fdd/contra_base.sh ${BASE_PATH} 2012 1

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_${benchmark}.sh ./ ${MASTER_PORT} 1 contra/fdd/base --seed $seed --eval-batch-size 64
    done
done