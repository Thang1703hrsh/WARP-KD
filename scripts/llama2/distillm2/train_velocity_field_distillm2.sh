#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-16}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/MiniLLM"}
CKPT_NAME="llama2-7B"
# CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
CKPT="meta-llama/Llama-2-7b-hf"
PEFT_CKPT_NAME="llama2-7B"
# PEFT_CKPT="${BASE_PATH}/results/llama2/train/init/${PEFT_CKPT_NAME}/"
PEFT_CKPT="bachthetrollface/llama2-7B-init-dolly-lora"
TEACHER_CKPT_NAME="llama2-13B"
# TEACHER_CKPT="${BASE_PATH}/checkpoints/${TEACHER_CKPT_NAME}/"
TEACHER_CKPT="meta-llama/Llama-2-13b-hf"
TEACHER_PEFT_CKPT_NAME="llama2-13B"
# TEACHER_PEFT_CKPT="${BASE_PATH}/results/llama2/train/sft/${TEACHER_PEFT_CKPT_NAME}/"
TEACHER_PEFT_CKPT="bachthetrollface/llama2-13B-teacher-dolly-lora"
# data
DATA_DIR="${BASE_PATH}/data/distillm2/llama2/formatted/"
# hp
BATCH_SIZE=16
LR=0.0005
GRAD_ACC=1
EVAL_BATCH_SIZE=8
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/llama2/train/velocity_field/distillm2"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 3"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# lora
OPTS+=" --peft lora"
OPTS+=" --peft-name ${PEFT_CKPT_NAME}"
OPTS+=" --peft-path ${PEFT_CKPT}"
OPTS+=" --teacher-peft-name ${TEACHER_PEFT_CKPT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_CKPT}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type lm-distillm2"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
#contra-kd config
OPTS+=" --velocity-n-layers 4"
OPTS+=" --velocity-d-model 3200"
OPTS+=" --d-teacher 5120"
OPTS+=" --d-student 4096"
OPTS+=" --num-distill-layers 6"
OPTS+=" --num-teacher-layers 40"
OPTS+=" --num-student-layers 32"
OPTS+=" --teacher-device 0"
OPTS+=" --student-device 0"


export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_velocity_field.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
