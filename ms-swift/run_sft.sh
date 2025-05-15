#!/bin/bash
export MASTER_PORT=29501

# Define model paths and datasets
models=(
    "/home/zhoujk/LLM/Qwen_7b"
    # "/home/zhoujk/LLM/llama3.1_8B-Instruct"
)

datasets=(
    "/home/zhoujk/LLM/SFT/SFT/Creat/p-soups/sft_psoups.jsonl"
    "/home/zhoujk/LLM/SFT/SFT/Creat/abs_100000.json"
    "/home/zhoujk/LLM/SFT/SFT/culture/culture_sft.jsonl"
    "/home/zhoujk/LLM/SFT/SFT/humor/humorous.jsonl"
    "/home/zhoujk/LLM/SFT/SFT/value/harmless/sft_harmless.jsonl"
    "/home/zhoujk/LLM/SFT/SFT/value/helpful/sft_helpful.jsonl"
    "/home/zhoujk/LLM/SFT/SFT/value/honest/sft_honest.jsonl"
# "/home/zhoujk/LLM/SFT/SFT/value/harmless_sft.json"
# "/home/zhoujk/LLM/SFT/SFT/value/helpful_sft.json"
# "/home/zhoujk/LLM/SFT/SFT/value/honest_sft.json"
# "/home/zhoujk/LLM/SFT/SFT/culture/culture_output.json"
# "/home/zhoujk/LLM/SFT/SFT/abs_100000.json"
# "/home/zhoujk/LLM/SFT/SFT/culture/WVQ_v1_gpt-4_1_sft.json"
)



# Set training parameters
nproc_per_node=9
min_free_mem=23000  # 单位：MiB

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export NPROC_PER_NODE=$nproc_per_node

# 检查 GPU 空闲内存函数
function wait_for_gpus() {
    while true; do
        echo "⏳ 正在检查所有 GPU 是否每块空闲内存 ≥ ${min_free_mem} MiB..."
        ready=true
        for id in $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '); do
            free_mem=$(nvidia-smi --id=$id --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
            echo "GPU $id 空闲内存：${free_mem} MiB"
            if [ "$free_mem" -lt "$min_free_mem" ]; then
                ready=false
                break
            fi
        done

        if [ "$ready" = true ]; then
            echo "✅ 所有 GPU 都满足条件，开始训练！"
            break
        else
            echo "❌ 不满足条件，30 秒后重试..."
            sleep 30
        fi
    done
}

# 启动前等待 GPU
wait_for_gpus

# Loop through models and datasets
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        output_dir="/home/zhoujk/LLM/MODEL/benchmark_model/sft/qwen7b/$(basename "$dataset")"
        mkdir -p "$output_dir"
        
        swift sft \
            --model "$model" \
            --train_type lora \
            --torch_dtype bfloat16 \
            --dataset "$dataset" \
            --model_type qwen2_5 \
            --num_train_epochs 1 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --learning_rate 1e-4 \
            --lora_rank 8 \
            --lora_alpha 32 \
            --target_modules all-linear \
            --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
            --eval_steps 100000 \
            --save_steps 100000 \
            --save_total_limit 2 \
            --logging_steps 5 \
            --max_length 800 \
            --output_dir "$output_dir" \
            --system 'You are a assistant.' \
            --warmup_ratio 0.05 \
            --dataloader_num_workers 4 \
            --model_author swift \
            --model_name swift-robot \
            --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    done
done