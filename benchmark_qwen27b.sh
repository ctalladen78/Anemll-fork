#!/bin/bash

# Configuration
LLAMA_BENCH="/Users/ctalladen/.llama_cpp/build/bin/llama-bench"
MODEL_PATH="/Users/ctalladen/.huggingface/Jackrong-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf"

# Check if binaries and model exist
if [ ! -f "$LLAMA_BENCH" ]; then
    echo "Error: llama-bench not found at $LLAMA_BENCH"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

echo "Benchmarking Qwen3.5-27B..."
echo "--------------------------------"

# Run benchmark
# -p 512: prompt size
# -n 128: tokens to generate
# -t 8: use 8 threads (default)
"$LLAMA_BENCH" -m "$MODEL_PATH" -p 512 -n 128
