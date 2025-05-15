export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((12345 + RANDOM % 1000))  # 避免端口冲突

# Define model paths and datasets
models=(
    # "/home/zhoujk/LLM/Qwen_7b"
    # "/home/zhoujk/LLM/llama3.2_3B_Instruct"
    "/home/zhoujk/LLM/llama3.2_1B_noi"
)

datasets=(
# "/home/zhoujk/LLM/DPO_RM/honest_dpo.json"
# "/home/zhoujk/LLM/DPO_RM/culture_dpo.json"
# "/home/zhoujk/LLM/SFT/SFT/value/helpful_sft.json"
# "/home/zhoujk/LLM/SFT/SFT/value/honest_sft.json"
# "/home/zhoujk/LLM/SFT/SFT/culture/culture_output.json"
# "/home/zhoujk/LLM/SFT/SFT/abs_100000.json"
"/home/zhoujk/LLM/DPO_RM/harmless_dpo_message_cleaned.jsonl"
"/home/zhoujk/LLM/DPO_RM/helpful_dpo_message_cleaned.jsonl"
"/home/zhoujk/LLM/DPO_RM/p-soups.json"
)

# Set training parameters
nproc_per_node=9

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export NPROC_PER_NODE=$nproc_per_node

# Loop through models and datasets
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        output_dir="/home/zhoujk/LLM/MODEL/benchmark_model/rm_model/llama3_2/$(basename "$dataset")"
        mkdir -p "$output_dir"
        
        swift rlhf \
            --rlhf_type rm \
            --model "$model"\
            --model_type llama3_2\
            --train_type lora \
            --dataset "$dataset" \
            --torch_dtype bfloat16 \
            --num_train_epochs 1 \
            --per_device_train_batch_size 3 \
            --per_device_eval_batch_size 1 \
            --learning_rate 1e-4 \
            --lora_rank 4 \
            --lora_alpha 32 \
            --target_modules all-linear \
            --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
            --eval_steps 1000 \
            --save_steps 20000 \
            --save_total_limit 5 \
            --logging_steps 5 \
            --max_length 1024 \
            --output_dir "$output_dir"\
            --warmup_ratio 0.05 \
            --dataloader_num_workers 1 \
            --deepspeed zero2 \
            --dataset_num_proc 4\
            --model_author swift \
            --model_name swift-robot \
            --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    done
done