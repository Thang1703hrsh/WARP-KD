BASE_PATH=${1-"."}

# one time run, can disable for subsequent runs
# bash ${BASE_PATH}/install.sh

bash ${BASE_PATH}/scripts/llama2/tools/process_data_dolly.sh ${BASE_PATH} 2012 1
bash ${BASE_PATH}/scripts/llama2/tools/process_data_pretrain.sh ${BASE_PATH} 2012 1

bash ${BASE_PATH}/scripts/llama2/sft/sft_13B_lora.sh ${BASE_PATH} 2012 1
bash ${BASE_PATH}/scripts/llama2/init/init_7B_lora.sh ${BASE_PATH} 2012 1

# stage 1
bash ${BASE_PATH}/scripts/llama2/train_velocity_field.sh ${BASE_PATH} 2012 1

# stage 2
bash ${BASE_PATH}/scripts/llama2/contra_7B_13B_lora.sh ${BASE_PATH} 2012 1

# evaluation
# bash ${BASE_PATH}/scripts/llama2/eval/run_eval.sh 0 contra_7B_13B
MASTER_PORT=2040
DEVICE=0
ckpt="contra_7B_13B"

# dolly eval
for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/llama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 llama2-7B ${ckpt} meta-llama/Llama-2-7b-hf --seed $seed --eval-batch-size 32
    done
done