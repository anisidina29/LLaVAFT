#!/bin/bash

gpu_list=$(nvidia-smi --list-gpus | awk '{print NR-1}' | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$gpu_list
IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.6-vicuna-13b"
SPLIT="train-00000-of-00738-80a58552f2fb3344"
WORKSPACE="/home/ray/image2code-mar22"
ANSDIR="${WORKSPACE}/data/predictions/raw"
LOCAL_HTMLDIR="${WORKSPACE}/data/predictions/processed"
SHARED_HTMLDIR="/efs/shared_storage/img2code/predictions"
RESDIR="${WORKSPACE}/data/results"
REFDIR="/efs/shared_storage/img2code/eval-d2c"
current_ts=$(date '+%Y-%m-%d-%H-%M-%S')


# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_websight \
#         --model-path liuhaotian/llava-v1.6-vicuna-13b \
#         --question-file ${WORKSPACE}/data/eval-queries/questions-10.jsonl \
#         --image-folder /efs/shared_storage/img2code/eval-d2c \
#         --answers-file ${ANSDIR}/${CKPT}/${SPLIT}/${current_ts}/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --max_new_tokens 2048 \
#         --load-4bit \
#         --conv-mode llava_v1 &
# done

# wait

# output_file=$ANSDIR/$CKPT/$SPLIT/${current_ts}/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat $ANSDIR/$CKPT/$SPLIT/${current_ts}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_websight_for_eval.py --ansdir $ANSDIR --split $SPLIT --ckpt $CKPT --htmldir $LOCAL_HTMLDIR --ts $current_ts
# To evaluate, need to:
# 1. Copy the generated HTML to your desktop (so that it can open a chromimum)
# 2. Generate screenshots
# 3. Move the images to the same folder where generated HTML files are stored

# TODO: change current_ts to the last value
current_ts='2024-03-25-23-06-24'
mv ${LOCAL_HTMLDIR}/${CKPT}/${SPLIT}/${current_ts}/image/*  ${LOCAL_HTMLDIR}/${CKPT}/${SPLIT}/${current_ts}
python llava/eval/eval_d2c.py --resdir $RESDIR --refdir $REFDIR --split $SPLIT --ckpt $CKPT --htmldir $LOCAL_HTMLDIR --ts $current_ts --corrupt-threshold 0.1 --bad-threshold 0.6