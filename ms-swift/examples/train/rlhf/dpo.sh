
# Define model paths and datasets
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((12345 + RANDOM % 1000))  # 避免端口冲突


models=(
    # "/home/zhoujk/LLM/Qwen_7b"
    "/home/zhoujk/LLM/llama3.2_3B_Instruct"
    # "/home/zhoujk/LLM/llama3.1_8B-Instruct"
    # "/home/zhoujk/LLM/llama3.2_1B_Instruct"
)

datasets=(
# "/home/zhoujk/LLM/DPO_RM/value/harmless/harmless_dpo_message_cleaned.jsonl"
# "/home/zhoujk/LLM/DPO_RM/value/helpful/helpful_dpo.jsonl"
# "/home/zhoujk/LLM/DPO_RM/value/honesty/honest_dpo.jsonl"
"/home/zhoujk/LLM/DPO_RM/Creat/message_sp-soups.json"
# "/home/zhoujk/LLM/SFT/SFT/value/helpful_sft.json"
# "/home/zhoujk/LLM/SFT/SFT/value/honest_sft.json"
# "/home/zhoujk/LLM/SFT/SFT/culture/culture_output.json"
# "/home/zhoujk/LLM/SFT/SFT/abs_100000.json"
)

# Set training parameters
nproc_per_node=6

# Set environment variables
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
export NPROC_PER_NODE=$nproc_per_node

# Loop through models and datasets
# Loop through models and datasets
for model_path in "${models[@]}"; do
    # 自动识别模型类型
    if [[ "$model_path" == *"wen"* ]]; then
        model_type="qwen2_5"
    elif [[ "$model_path" == *"llama"* ]]; then
        model_type="llama3_2"
    else
        echo "无法识别模型类型: $model_path"
        continue
    fi

    for dataset in "${datasets[@]}"; do
        # 用模型名自动命名 output_dir
        model_name=$(basename "$model_path")
        dataset_name=$(basename "$dataset")
        output_dir="/home/zhoujk/LLM/MODEL/benchmark_model/dpo/${model_name}/lora/${dataset_name}"
        mkdir -p "$output_dir"

        swift rlhf \
            --rlhf_type dpo \
            --model "$model_path" \
            --model_type "$model_type" \
            --train_type lora \
            --dataset "$dataset" \
            --torch_dtype bfloat16 \
            --num_train_epochs 2 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --learning_rate 1e-6 \
            --lora_rank 8 \
            --lora_alpha 32 \
            --target_modules all-linear \
            --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
            --eval_steps 100 \
            --save_steps 20000 \
            --save_total_limit 5 \
            --logging_steps 5 \
            --max_length 512 \
            --output_dir "$output_dir" \
            --warmup_ratio 0.05 \
            --dataloader_num_workers 1 \
            --use_liger_kernel True \
            --cliprange 0.4 \
            --deepspeed zero2 \
            --dataset_num_proc 1 \
            --model_author swift \
            --model_name swift-robot \
            --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    done
done
