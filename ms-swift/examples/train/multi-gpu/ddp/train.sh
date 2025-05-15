# 27.5GiB * 2
nproc_per_node=8

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /home/zhoujk/LLM/Qwen_7b\
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset /home/zhoujk/LLM/SFT/value.json\
    --model_type qwen2_5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 3 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
