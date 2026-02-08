BASE_PATH=${1-"."}

# FDD + Contra
# stage 1
# bash ${BASE_PATH}/scripts/openllama2/train_velocity_field.sh ${BASE_PATH} 2012 1
# stage 2
bash ${BASE_PATH}/scripts/openllama2/fdd/contra_3B_7B_teacher_lora.sh ${BASE_PATH} 2012 1

MASTER_PORT=2040
DEVICE=0
for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B contra/fdd/3B_7B openlm-research/open_llama_3b_v2 --seed $seed --eval-batch-size 32
    done
done