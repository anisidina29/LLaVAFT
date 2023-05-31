deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 25001 \
    llava/train/train.py \
    --deepspeed deepspeed.json \
    --model_name_or_path ./checkpoints/llama_hf/llama_65b \
    --version v1 \
    --data_path /Data/haotian/cc3m/cc3m_np_top100_skip1_595k_fastchat.json \
    --image_folder /Data/haotian/cc3m/cc3m_np_top100_skip1_595k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/deepspeed_dev_pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4800 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
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