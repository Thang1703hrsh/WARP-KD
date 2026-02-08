BASE_PATH=${1-"."}

# FDD + Contra
# stage 1
# bash ${BASE_PATH}/scripts/llama2/train_velocity_field.sh ${BASE_PATH} 2012 1
# stage 2
bash ${BASE_PATH}/scripts/llama2/fdd/contra_7B_13B_teacher_lora.sh ${BASE_PATH} 2012 1

MASTER_PORT=2040
DEVICE=0
for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/llama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 llama2-7B contra/fdd/7B_13B meta-llama/Llama-2-7b-hf --seed $seed --eval-batch-size 32
    done
done