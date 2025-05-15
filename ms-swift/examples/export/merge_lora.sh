# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
export CUDA_VISIBLE_DEVICES=2,3,4
swift export \
    --adapters /home/zhoujk/LLM/MODEL/benchmark_model/dpo/llama3.2_1B_Instruct/lora/harmless_dpo_message_cleaned.jsonl/v0-20250428-165912/checkpoint-2495\
    --merge_lora true
