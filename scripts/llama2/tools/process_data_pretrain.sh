BASE_PATH=${1}

MAX_LENGTH=512

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/openwebtext \
    --processed-data-dir ${BASE_PATH}/processed_data/openwebtext/llama2/${MAX_LENGTH}/ \
    --model-path meta-llama/Llama-2-7b-hf \
    --max-length ${MAX_LENGTH} \
    --train-num 22870 \
    --data-process-workers 32 \
    --dev-num 1000 \
