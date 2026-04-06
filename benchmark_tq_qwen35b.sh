#!/bin/bash

# Configuration
MODEL_PATH="/Users/ctalladen/.huggingface/unsloth-Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
BENCH_BIN="/Users/ctalladen/Documents/Antigravity-Claude/output-Apr3/llama-cpp-turboquant/build/bin/llama-bench"
LOG_FILE="benchmark.log"

echo "--- Starting TurboQuant Benchmark: Qwen3.5-35B-A3B ---" | tee -a "$LOG_FILE"
echo "Binary: $BENCH_BIN" | tee -a "$LOG_FILE"
echo "Model: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Optimizations 1: -ctk q8_0 -ctv turbo4 (Asymmetric KV)" | tee -a "$LOG_FILE"
echo "Optimizations 2: -ctk turbo4 -ctv turbo4 (Symmetric TQ4)" | tee -a "$LOG_FILE"
echo "--------------------------------------------------------" | tee -a "$LOG_FILE"

# Run benchmark with Asymmetric TQ optimizations (8k, 16k, 32k context)
echo "Running Asymmetric TQ (Q8_0 / Turbo4)..." | tee -a "$LOG_FILE"
"$BENCH_BIN" -m "$MODEL_PATH" -p 8192,16384,32768 -n 1 -ctk q8_0 -ctv turbo4 2>&1 | tee -a "$LOG_FILE"

# Run benchmark with Symmetric TQ4 optimizations (8k, 16k, 32k context)
echo "Running Symmetric TQ4 (Turbo4 / Turbo4)..." | tee -a "$LOG_FILE"
"$BENCH_BIN" -m "$MODEL_PATH" -p 8192,16384,32768 -n 1 -ctk turbo4 -ctv turbo4 2>&1 | tee -a "$LOG_FILE"

echo "--- Benchmark Complete ---" | tee -a "$LOG_FILE"
