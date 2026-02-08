BASE_PATH="."
num_layers=${1-6}
velocity_save_path="${BASE_PATH}/results/gpt2/train/velocity_field/num_layers/${num_layers}"
model_save_path="${BASE_PATH}/results/gpt2/train/ablation/num_layers/${num_layers}"

mkdir -p ${velocity_save_path}
mkdir -p ${model_save_path}
velocity_field_path="${velocity_save_path}/velocity_field.pth"
projector_path="${velocity_save_path}/projector.pth"
# one time run, can disable for subsequent runs
# bash ${BASE_PATH}/install.sh

# load openwebtext dataset
# python3 tools/get_openwebtext.py
# bash ${BASE_PATH}/scripts/gpt2/tools/process_data_pretrain.sh ${BASE_PATH} 2012 1
# bash ${BASE_PATH}/scripts/gpt2/tools/process_data_dolly.sh ${BASE_PATH} 2012 1

# stage 1
bash ${BASE_PATH}/scripts/gpt2/train_velocity_field.sh ${BASE_PATH} 2012 1 --num-distill-layers ${num_layers} --save ${velocity_save_path}

# stage 2
bash ${BASE_PATH}/scripts/gpt2/distillm/contra_0.1B_1.5B.sh ${BASE_PATH} 2012 1 --velocity-field-path ${velocity_field_path} --projector-path ${projector_path} --num-distill-layers ${num_layers} --save ${model_save_path}

# evaluation
MASTER_PORT=2040
DEVICE=0
ckpt="ablation/num_layers/${num_layers}"

for benchmark in dolly self_inst vicuna sinst uinst
do
    for seed in 10 20 30 40 50
    do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_${benchmark}.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 64
    done
done