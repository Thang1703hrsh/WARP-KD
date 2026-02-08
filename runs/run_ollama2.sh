BASE_PATH=${1-"."}

# one time run, can disable for subsequent runs
# bash ${BASE_PATH}/install.sh
# pip install -U transformers==4.43.4 peft==0.11.1 accelerate
# pip install pyyaml

bash ${BASE_PATH}/scripts/openllama2/tools/process_data_dolly.sh ${BASE_PATH} 2012 1
bash ${BASE_PATH}/scripts/openllama2/tools/process_data_pretrain.sh ${BASE_PATH} 2012 1
bash ${BASE_PATH}/scripts/openllama2/sft/sft_7B_lora.sh ${BASE_PATH} 2012 1
bash ${BASE_PATH}/scripts/openllama2/init/init_3B_lora.sh ${BASE_PATH} 2012 1

# stage 1
bash ${BASE_PATH}/scripts/openllama2/train_velocity_field.sh ${BASE_PATH} 2012 1

# stage 2
bash ${BASE_PATH}/scripts/openllama2/contra_3B_7B_lora.sh ${BASE_PATH} 2012 1

# evaluation
# bash ${BASE_PATH}/scripts/openllama2/eval/run_eval.sh 0 contra_3B_7B
MASTER_PORT=2040
DEVICE=0
ckpt="contra_3B_7B"

# dolly eval
for seed in 10 20 30 40 50
do
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_dolly_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} openlm-research/open_llama_3b_v2 --seed $seed  --eval-batch-size 32
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_self_inst_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} openlm-research/open_llama_3b_v2 --seed $seed  --eval-batch-size 32
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_vicuna_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} openlm-research/open_llama_3b_v2 --seed $seed  --eval-batch-size 32
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_sinst_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} openlm-research/open_llama_3b_v2 --seed $seed  --eval-batch-size 32
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_uinst_lora.sh ./ ${MASTER_PORT} 1 openllama2-3B ${ckpt} openlm-research/open_llama_3b_v2 --seed $seed  --eval-batch-size 32
done