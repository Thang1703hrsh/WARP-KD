# eval teacher ckpt

BASE_PATH="."

export TF_CPP_MIN_LOG_LEVEL=3

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path openlm-research/open_llama_3b_v2 \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type openllama2

bash ./scripts/openllama2/sft/sft_7B_lora.sh . 2012 1

MASTER_PORT=2040
DEVICE=0
ckpt="sft/openllama2-7B"
TEACHER_EVAL_BATCH_SIZE=32

for benchmark in dolly self_inst vicuna sinst uinst
do
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 openllama2-7B ${ckpt} openlm-research/open_llama_7b_v2 --seed 10  --eval-batch-size ${TEACHER_EVAL_BATCH_SIZE}
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 openllama2-7B ${ckpt} openlm-research/open_llama_7b_v2 --seed 20  --eval-batch-size ${TEACHER_EVAL_BATCH_SIZE}
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 openllama2-7B ${ckpt} openlm-research/open_llama_7b_v2 --seed 30  --eval-batch-size ${TEACHER_EVAL_BATCH_SIZE}
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 openllama2-7B ${ckpt} openlm-research/open_llama_7b_v2 --seed 40  --eval-batch-size ${TEACHER_EVAL_BATCH_SIZE}
    CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openllama2/eval/eval_main_${benchmark}_lora.sh ./ ${MASTER_PORT} 1 openllama2-7B ${ckpt} openlm-research/open_llama_7b_v2 --seed 50  --eval-batch-size ${TEACHER_EVAL_BATCH_SIZE}
done

# # rerun contra-kd (with velocity field warmup)
# BASE_PATH=${1-"."}
# # stage 1
# bash ${BASE_PATH}/scripts/gpt2/train_velocity_field.sh ${BASE_PATH} 2012 1

# # stage 2
# bash ${BASE_PATH}/scripts/gpt2/contra_0.1B_1.5B.sh ${BASE_PATH} 2012 1

# # evaluation
# # bash ${BASE_PATH}/scripts/gpt2/eval/run_eval.sh 0 contra_0.1B_1.5B_final2
# # MASTER_PORT=2040
# # DEVICE=0
# ckpt="contra_0.1B_1.5B_final2"
# STUDENT_EVAL_BATCH_SIZE=64

# for seed in 10 20 30 40 50
# do
#     CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_dolly.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size ${STUDENT_EVAL_BATCH_SIZE}
#     CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_self_inst.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size ${STUDENT_EVAL_BATCH_SIZE}
#     CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_vicuna.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size ${STUDENT_EVAL_BATCH_SIZE}
#     CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_sinst.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size ${STUDENT_EVAL_BATCH_SIZE}
#     CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/gpt2/eval/eval_main_uinst.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size ${STUDENT_EVAL_BATCH_SIZE}
# done