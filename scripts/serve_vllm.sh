#!/bin/bash
# vLLM 服务启动脚本
# Usage: bash scripts/serve_vllm.sh [--port 8000] [--max-model-len 4096]

PORT=8000
MAX_MODEL_LEN=4096
MODEL_PATH="outputs/qwen_merged"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        *) echo "Unknown: $1"; shift ;;
    esac
done

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: $MODEL_PATH 不存在，请先运行 scripts/export.py"
    exit 1
fi

echo "Starting vLLM: $MODEL_PATH on port $PORT"
VLLM_USE_FLASHINFER_SAMPLER=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --dtype auto \
    --enforce-eager
