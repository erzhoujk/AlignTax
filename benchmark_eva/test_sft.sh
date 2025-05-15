
#!/bin/bash

# ======================== 参数配置 ========================


#诚实  triviaqa 毒性 toxigen realtoxicityprompts 偏见：bbq、crows_pairs、simple_cooccurrence_bias、winogender

#!/bin/bash

# DEVICES=( "cuda:1" "cuda:2" "cuda:3" "cuda:4" "cuda:5" "cuda:6" "cuda:7" "cuda:8")
DEVICES=("cuda:6" "cuda:7" "cuda:8")
OUTPUT_BASE_PATH="/home/zhoujk/LLM/Benchmark_eva/evaluation/benchmark/large_main/table"
export HF_ALLOW_CODE_EVAL=1
MIN_MEMORY=14000

declare -A MODEL_PATHS=(


    ["qwen7b"]="/home/zhoujk/LLM/Qwen_7b"
    # ["llama_1b"]="/home/zhoujk/LLM/llama3.2_1B_Instruct"
    # ["llama_harmless_1b"]="/home/zhoujk/LLM/MODEL/benchmark_model/dpo/llama3.2_1B_Instruct/lora/harmless_dpo_message_cleaned.jsonl/v0-20250428-165912/checkpoint-2495-merged"

    # ["qwen_7b_psoups"]="/home/zhoujk/LLM/MODEL/benchmark_model/dpo/Qwen_7b/lora/message_sp-soups.json/v0-20250425-005727/checkpoint-3394-merged"
    # ["llama3.2_8b_psoups"]="/home/zhoujk/LLM/MODEL/benchmark_model/dpo/llama3.1_8B-Instruct/lora/message_sp-soups.json/v0-20250425-044814/checkpoint-3382-merged"
    # ["qwen7b_culture_sft"]="/home/zhoujk/LLM/MODEL/benchmark_model/sft/qwen7b/culture_sft.jsonl/v0-20250422-032719/checkpoint-170-merged"
    # ["qwen7b_harmless_sft"]="/home/zhoujk/LLM/MODEL/benchmark_model/sft/qwen7b/sft_harmless.jsonl/v0-20250422-033755/checkpoint-4573-merged"
    # ["qwen7b_helpful_sft"]="/home/zhoujk/LLM/MODEL/benchmark_model/sft/qwen7b/sft_helpful.jsonl/v0-20250422-061738/checkpoint-10176-merged"
    # ["qwen7b_dpo_harmless"]="/home/zhoujk/LLM/MODEL/benchmark_model/dpo/qwen7b/lora/harmless_dpo_message_cleaned.jsonl/v2-20250421-100316/checkpoint-4442-merged"
    # ["qwen7b_dpo_helpful"]="/home/zhoujk/LLM/MODEL/benchmark_model/dpo/qwen7b/lora/helpful_dpo.jsonl/v1-20250421-124252/checkpoint-8027-merged"
    # ["qwen7b_dpo_honest"]="/home/zhoujk/LLM/MODEL/benchmark_model/dpo/qwen7b/lora/honest_dpo.jsonl/v0-20250421-181359/checkpoint-3366-merged"
    # ["qwen1.5b"]="/home/zhoujk/LLM/Qwen_1.5b"
#     ["qwen7b_sft_culture"]="/home/zhoujk/LLM/MODEL/benchmark_model/sftt/culture_output.json/v0-20250322-093003/checkpoint-1323-merged"
#     ["qwen7b_sft_abs"]="/home/zhoujk/LLM/MODEL/benchmark_model/sftt/abs_100000.json/v0-20250322-110255/checkpoint-9282-merged"
#     ["qwen7b_sft_harmless"]="/home/zhoujk/LLM/MODEL/benchmark_model/sftt/harmless_sft.json/v1-20250321-152812/checkpoint-3948-merged"
#     ["qwen7b_sft_helpful"]="/home/zhoujk/LLM/MODEL/benchmark_model/sftt/helpful_sft.json/v0-20250321-203212/checkpoint-4068-merged"
#     ["qwen7b_sft_honest"]="/home/zhoujk/LLM/MODEL/benchmark_model/sftt/honest_sft.json/v0-20250322-014844/checkpoint-4878-merged"
    # ["qwen7b_dpo_honest"]="/home/zhoujk/LLM/MODEL/benchmark_model/dpo/qwen7b/honest_dpo.json/v0-20250325-165614/checkpoint-1234-merged"
    # ["qwen7b_dpo_culture"]="/home/zhoujk/LLM/MODEL/benchmark_model/dpo/qwen7b/culture_dpo.json/v4-20250325-153804/checkpoint-1234-merged"

    # ["llama3.18b"]="/home/zhoujk/LLM/llama3.1_8B-Instruct"
    # ["llama3.18b_sft_harmless"]="/home/zhoujk/LLM/MODEL/benchmark_model/sftt/llama3.1_8b/harmless_sft.json/v0-20250325-221514/checkpoint-3948-merged"
    # ["llama3.18b_sft_helpful"]="/home/zhoujk/LLM/MODEL/benchmark_model/sftt/llama3.1_8b/helpful_sft.json/v0-20250326-090447/checkpoint-4068-merged"
    # ["llama3.18b_sft_honest"]="/home/zhoujk/LLM/MODEL/benchmark_model/sftt/llama3.1_8b/honest_sft.json/v0-20250326-201652/checkpoint-4863-merged"
    # ["llama3.18b_sft_culture"]="/home/zhoujk/LLM/MODEL/benchmark_model/sftt/llama3.1_8b/culture_output.json/v0-20250327-121009/checkpoint-1323-merged"
    # ["llamam3.18b_sft_abs"]="/home/zhoujk/LLM/MODEL/benchmark_model/sftt/llama3.1_8b/abs_100000.json/v0-20250327-153501/checkpoint-9279-merged"
    # ["llama3_23b_dpo_harmless_full"]="/home/zhoujk/LLM/MODEL/benchmark_model/dpo/llama3b/full/harmless_dpo_message_cleaned.jsonl/v8-20250418-012755/checkpoint-2793"
    # ["llama3_23b_dpo_psoups_full"]="/home/zhoujk/LLM/MODEL/benchmark_model/dpo/llama3b/full/p-soups.json/v0-20250417-214633/checkpoint-879"
)

# TASKS=("arc_challenge" "gsm8k" "glue" "drop" "hellaswag" "mbpp" "humaneval" "mmlu_pro" "hendrycks_math" "mgsm_direct_en" "race" "blimp" "commonsense_qa" "truthfulqa" "triviaqa" "toxigen" "realtoxicityprompts" "bbq" "crows_pairs" "simple_cooccurrence_bias" "winogender")
# TASK_QUEUE=("triviaqa" "toxigen" "realtoxicityprompts" "bbq" "crows_pairs" "simple_cooccurrence_bias" "winogender")
# TASKS=( "arc_challenge" "gsm8k" "glue" "drop" "hellaswag" "mbpp" "humaneval" "mmlu_pro" "hendrycks_math" "mgsm_direct_en" "race" "blimp" "commonsense_qa" "truthfulqa" "triviaqa"  )
TASKS=( "mbpp" "humaneval")



# 初始化任务队列：所有模型 × 所有任务
TASK_QUEUE=()
for MODEL_NAME in "${!MODEL_PATHS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        TASK_QUEUE+=("$MODEL_NAME|$TASK")
    done
done

# 检查某个 GPU 是否满足最小内存要求
check_gpu_memory() {
    local device=$1
    local index=${device/cuda:/}
    local free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$index" | tr -d ' ')
    [[ "$free_mem" -ge "$MIN_MEMORY" ]]
}

# 运行任务并放入 tmux 会话
# 
run_task_direct() {
    local MODEL_NAME=$1
    local TASK=$2
    local GPU_DEVICE=$3
    local MODEL_PATH=${MODEL_PATHS[$MODEL_NAME]}
    local OUTPUT_DIR="${OUTPUT_BASE_PATH}/${MODEL_NAME}/$(echo $TASK | tr ',' '_')"
    mkdir -p "$OUTPUT_DIR"

    echo "[$(date)] Launching: $MODEL_NAME | $TASK on $GPU_DEVICE"

    lm_eval --model hf --model_args pretrained="$MODEL_PATH" \
            --tasks "$TASK" --num_fewshot 3 --device "$GPU_DEVICE" \
            --output "$OUTPUT_DIR" --batch_size 4 &\


    echo "[$(date)] $MODEL_NAME | $TASK launched on $GPU_DEVICE, logging to $OUTPUT_DIR/stdout.log"
}


# 主调度循环
monitor_and_allocate() {
    while [ ${#TASK_QUEUE[@]} -gt 0 ]; do
        AVAILABLE_GPUS=()

        for GPU_DEVICE in "${DEVICES[@]}"; do
            if check_gpu_memory "$GPU_DEVICE"; then
                # 若该 GPU 未在 tmux 中运行任务，则加入可用列表
                RUNNING=$(tmux ls 2>/dev/null | grep -c "$GPU_DEVICE")
                if [[ "$RUNNING" -eq 0 ]]; then
                    AVAILABLE_GPUS+=("$GPU_DEVICE")
                fi
            fi
        done

        if [ "${#AVAILABLE_GPUS[@]}" -eq 0 ]; then
            echo "[$(date)] 所有 GPU 均繁忙，等待空闲中..."
            sleep 10
            continue
        fi

        for GPU in "${AVAILABLE_GPUS[@]}"; do
            if [ ${#TASK_QUEUE[@]} -eq 0 ]; then
                break
            fi

            JOB="${TASK_QUEUE[0]}"
            IFS="|" read -r MODEL TASK <<< "$JOB"
            TASK_QUEUE=("${TASK_QUEUE[@]:1}")

            if [ ! -d "${MODEL_PATHS[$MODEL]}" ]; then
                echo "模型路径无效：${MODEL_PATHS[$MODEL]}，跳过..."
                continue
            fi

            # run_task_tmux "$MODEL" "$TASK" "$GPU"
            run_task_direct "$MODEL" "$TASK" "$GPU"
            sleep 120  # 避免短时间并发造成 tmux/session 冲突
        done

        sleep 35  # 等待下一轮资源检查
    done

    echo "所有任务分配完成，脚本退出。"
}

trap 'echo "脚本终止，清理中..."' EXIT
monitor_and_allocate
