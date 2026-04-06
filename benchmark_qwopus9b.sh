#!/bin/bash

# Script to benchmark Qwopus3.5-9B-v3-GGUF using llama.cpp llama-bench
# Usage: ./benchmark_qwopus.sh

set -e  # Exit on error

echo "=========================================="
echo "Benchmarking Qwopus3.5-9B-v3 (Q6_K GGUF)"
echo "=========================================="

# Define model path
MODEL_PATH="$HOME/.huggingface/Jackrong-Qwopus3.5-9B-v3-GGUF-Q6_K/Qwen3.5-9B.Q6_K.gguf"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please download the model first:"
    echo "  huggingface-cli download Jackrong/Qwopus3.5-9B-v3-GGUF Qwen3.5-9B.Q6_K.gguf"
    exit 1
fi

echo "Model found at $MODEL_PATH"

# Check file size
SIZE_GB=$(du -h "$MODEL_PATH" | cut -f1)
echo "Model size: $SIZE_GB"

# Check for llama-bench (benchmarking tool)
LLAMA_BENCH="$HOME/.llama_cpp/build/bin/llama-bench"
if [ ! -f "$LLAMA_BENCH" ]; then
    LLAMA_BENCH=$(find "$HOME/.llama_cpp" -name "llama-bench" 2>/dev/null | head -1)
fi

if [ -z "$LLAMA_BENCH" ]; then
    echo "Error: llama-bench not found"
    echo "Please build llama.cpp from source in ~/.llama_cpp"
    echo "  cd ~/.llama_cpp && mkdir build && cd build && cmake .. && make -j llama-bench"
    exit 1
fi

echo "Using llama-bench at: $LLAMA_BENCH"

# Run benchmark with llama-bench
# Default llama-bench options: -m <model> -ngl <gpu_layers> -c <context> -n <tokens>
echo ""
echo "Running benchmark tests..."
echo ""

$LLAMA_BENCH \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -p 512 \
    -n 128 \
    -b 2048 \
    -t 8

echo "=========================================="
echo "Benchmark completed"
echo "=========================================="