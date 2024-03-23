#!/bin/bash

gpu_list=$(nvidia-smi --list-gpus | awk '{print NR-1}' | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$gpu_list
IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.6-mistral-7b"
SPLIT="train-00000-of-00738-80a58552f2fb3344-small"
ANSDIR="./playground/data/eval/websight/answers"
WORKSPACE="/home/ray/image2code-mar22"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_websight \
        --model-path liuhaotian/llava-v1.6-mistral-7b \
        --question-file /home/ray/image2code-mar22/smart-consumer-dev/img2code-test/short-questions.jsonl \
        --image-folder /efs/shared_storage/img2code/WebSight/processed/image \
        --answers-file $ANSDIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode mistral_instruct &
done

wait

output_file=$ANSDIR/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ANSDIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_websight_for_eval.py --ansdir $ANSDIR --split $SPLIT --ckpt $CKPT --workspace $WORKSPACE