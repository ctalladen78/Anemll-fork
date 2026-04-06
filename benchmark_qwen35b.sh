#!/bin/bash

# Configuration
LLAMA_BENCH="/Users/ctalladen/.llama_cpp/build/bin/llama-bench"
MODEL_PATH="/Users/ctalladen/.huggingface/unsloth-Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"

# Check if binaries and model exist
if [ ! -f "$LLAMA_BENCH" ]; then
    echo "Error: llama-bench not found at $LLAMA_BENCH"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

echo "Benchmarking unsloth-Qwen3.5-35B-A3B..."
echo "--------------------------------"

# Run benchmark
# -p 512: prompt size
# -n 128: tokens to generate
# -t 8: use 8 threads (default)
"$LLAMA_BENCH" -m "$MODEL_PATH" -p 512 -n 128
