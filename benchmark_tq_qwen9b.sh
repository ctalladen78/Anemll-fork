#!/bin/bash

# Benchmark TurboQuant settings on Qwen3.5-9B
# Usage: ./benchmark_tq_qwen9b.sh

set -e

echo "=========================================="
echo "TurboQuant Benchmark: Qwen3.5-9B (Q6_K GGUF)"
echo "=========================================="

MODEL_PATH="$HOME/.huggingface/Jackrong-Qwopus3.5-9B-v3-GGUF-Q6_K/Qwen3.5-9B.Q6_K.gguf"
BENCH_BIN="/Users/ctalladen/Documents/Antigravity-Claude/output-Apr3/llama-cpp-turboquant/build/bin/llama-bench"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

echo "Model: $MODEL_PATH"
SIZE_GB=$(du -h "$MODEL_PATH" | cut -f1)
echo "Size: $SIZE_GB"
echo ""

echo "TurboQuant Settings:"
echo "  -ctk q8_0  (K-cache: Q8_0)"
echo "  -ctv turbo4  (V-cache: TurboQuant 4-bit)"
echo "  -fa 1  (Flash Attention: on)"
echo ""

echo "Running benchmark..."
echo ""

$BENCH_BIN \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -ctk q8_0 \
    -ctv turbo4 \
    -fa 1 \
    -p 512 \
    -n 128 \
    -b 2048 \
    -t 8 \
    -r 3 \
    --no-warmup

echo ""
echo "=========================================="
echo "Benchmark completed"
echo "=========================================="