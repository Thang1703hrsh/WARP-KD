BASE_PATH=${1-"."}

# one time run, can disable for subsequent runs
# bash ${BASE_PATH}/install.sh
# pip install -U transformers==4.43.4 peft==0.11.1 accelerate
# pip install pyyaml

# load openwebtext dataset
# python3 tools/get_openwebtext.py
# bash ${BASE_PATH}/scripts/gpt2/tools/process_data_pretrain.sh ${BASE_PATH} 2012 1

# bash ${BASE_PATH}/scripts/gpt2/tools/process_data_dolly.sh ${BASE_PATH} 2012 1
# bash ${BASE_PATH}/scripts/gpt2/sft/sft_xlarge.sh ${BASE_PATH} 2012 1
# bash ${BASE_PATH}/scripts/gpt2/init/init_base.sh ${BASE_PATH} 2012 1

bash ${BASE_PATH}/scripts/gpt2/distillm/train_0.1B_1.5B.sh ${BASE_PATH} 2012 1

# evaluation
# bash ${BASE_PATH}/scripts/gpt2/eval/run_eval.sh 0 distill_0.1B_1.5B_final2
MASTER_PORT=2040
DEVICE=0
ckpt="distillm/0.1B_1.5B"

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_${benchmark}.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 64
    done
done