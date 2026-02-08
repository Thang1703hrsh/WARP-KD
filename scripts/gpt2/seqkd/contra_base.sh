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
CKPT_NAME="gpt2-base"
CKPT="${BASE_PATH}/results/gpt2/train/init/${CKPT_NAME}"
# CKPT="gpt2"
TEACHER_CKPT_NAME="xlarge-sft"
TEACHER_CKPT="${BASE_PATH}/results/gpt2/train/sft/gpt2-xlarge/"
VELOCITY_FIELD_CKPT="${BASE_PATH}/results/gpt2/train/velocity_field/seqkd/velocity_field.pth"
PROJECTOR_CKPT="${BASE_PATH}/results/gpt2/train/velocity_field/seqkd/projector.pth"
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/pseudo/gpt2/"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/22.87K/"
# hp
BATCH_SIZE=16
LR=0.0005
GRAD_ACC=1
EVAL_BATCH_SIZE=64
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/gpt2/train/contra/seqkd/base"
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
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 4"
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
OPTS+=" --epochs 20"
OPTS+=" --kd-ratio 1.0"
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
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type fkl-contra"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
#contra-kd config
OPTS+=" --velocity-n-layers 4"
OPTS+=" --velocity-d-model 1024"
OPTS+=" --d-teacher 1600"
OPTS+=" --d-student 768"
OPTS+=" --num-distill-layers 6"
OPTS+=" --num-teacher-layers 48"
OPTS+=" --num-student-layers 12"
OPTS+=" --teacher-device 0"
OPTS+=" --student-device 0"
OPTS+=" --velocity-field-path ${VELOCITY_FIELD_CKPT}"
OPTS+=" --projector-path ${PROJECTOR_CKPT}"
OPTS+=" --velocity-epochs 1"
OPTS+=" --velocity-update-interval 1"


export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
