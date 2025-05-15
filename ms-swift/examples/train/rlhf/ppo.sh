# Currently, it only supports the case where the model and reward_model use the same template/tokenizer.
# Currently, multimodal model PPO is not supported.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((12345 + RANDOM % 1000))  # 避免端口冲突



# Define model paths and datasets
models=(
    # "/home/zhoujk/LLM/Qwen_7b"
    # "/home/zhoujk/LLM/llama3.1_8B-Instruct"
    "/home/zhoujk/LLM/llama3.2_3B_Instruct"
)

datasets=(
# "/home/zhoujk/LLM/DPO_RM/honest_dpo.json"
# "/home/zhoujk/LLM/DPO_RM/culture_dpo.json"
# "/home/zhoujk/LLM/SFT/SFT/value/helpful_sft.json"
# "/home/zhoujk/LLM/SFT/SFT/value/honest_sft.json"
# "/home/zhoujk/LLM/SFT/SFT/culture/culture_output.json"
# "/home/zhoujk/LLM/SFT/SFT/abs_100000.json"
# "/home/zhoujk/LLM/DPO_RM/harmless_dpo_message_cleaned.jsonl"
# "/home/zhoujk/LLM/DPO_RM/helpful_dpo_message_cleaned.jsonl"
# "/home/zhoujk/LLM/DPO_RM/p-soups.json"
# "/home/zhoujk/LLM/PPO/harmless_ppo_message_cleaned.jsonl"
"/home/zhoujk/LLM/PPO/helpful_ppo_message_cleaned.jsonl"
"/home/zhoujk/LLM/PPO/ppo_p-soups.json"
)

declare -A reward_models
reward_models["/home/zhoujk/LLM/PPO/harmless_ppo_message_cleaned.jsonl"]="/home/zhoujk/LLM/MODEL/benchmark_model/rm_model/llama3_2/harmless_dpo_message_cleaned.jsonl/v0-20250416-124604/checkpoint-1528-merged"
reward_models["/home/zhoujk/LLM/PPO/helpful_ppo_message_cleaned.jsonl"]="/home/zhoujk/LLM/MODEL/benchmark_model/rm_model/llama3_2/helpful_dpo_message_cleaned.jsonl/v0-20250416-131703/checkpoint-1585-merged"
reward_models["/home/zhoujk/LLM/PPO/ppo_p-soups.json"]="/home/zhoujk/LLM/MODEL/benchmark_model/rm_model/llama3_2/helpful_dpo_message_cleaned.jsonl/v0-20250416-131703/checkpoint-1585-merged"

# 添加更多数据集映射...

# 添加更多数据集映射...

# Set training parameters
nproc_per_node=9
# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export NPROC_PER_NODE=$nproc_per_node

# Loop through models and datasets
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        reward_model_path="${reward_models[$dataset]}"
        
        if [[ -z "$reward_model_path" ]]; then
            echo "❌ No reward model found for dataset: $dataset"
            continue
        fi

        output_dir="/home/zhoujk/LLM/MODEL/benchmark_model/ppo_model/llama3b/$(basename "$dataset")"
        mkdir -p "$output_dir"
        
        swift rlhf \
            --rlhf_type ppo \
            --model "$model"\
            --model_type llama3_2\
            --reward_model "$reward_model_path" \
            --train_type lora \
            --dataset "$dataset" \
            --torch_dtype bfloat16 \
            --num_train_epochs 2 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --learning_rate 1e-8 \
            --lora_rank 4 \
            --lora_alpha 32 \
            --target_modules all-linear \
            --gradient_accumulation_steps 1\
            --eval_steps 1000 \
            --save_steps 2000 \
            --save_total_limit 5 \
            --logging_steps 5 \
            --max_length 50 \
            --output_dir "$output_dir"\
            --warmup_ratio 0.05 \
            --dataloader_num_workers 1 \
            --deepspeed zero2 \
            --dataset_num_proc 3\
            --model_author swift \
            --gradient_checkpointing \
            --model_name swift-robot \
            --use_liger_kernel True \
            --cliprange 0.4\
            --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    done
done
