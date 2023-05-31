deepspeed --include=localhost:4,5,6,7 --master_port 26003 \
    llava/train/train_mem.py \
    --deepspeed deepspeed.json \
    --model_name_or_path ./checkpoints/vicuna-7b-v0 \
    --data_path ./playground/data/llava_instruct/conv_reason_no_overlap_80k.json \
    --image_folder /Data/haotian/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/llava-vicuna-7b-v0-pretrain-cc_595k-1epoch.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/deepspeed_dev_lora_finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to tensorboard
