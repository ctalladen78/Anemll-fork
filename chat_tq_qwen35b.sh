#!/bin/bash

# Configuration
MODEL_PATH="/Users/ctalladen/.huggingface/unsloth-Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
CLI_BIN="/Users/ctalladen/Documents/Antigravity-Claude/output-Apr3/llama-cpp-turboquant/build/bin/llama-cli"

# Optimal TurboQuant Chat Settings:
# -ctk q8_0    : K-cache Q8_0 (High accuracy)
# -ctv turbo4  : V-cache Turbo4 (PolarQuant 4-bit, 4x memory savings)
# --flash-attn : Required for TQ kernels
# -c 16384     : 16k Balanced Context window
# -cnv         : Chat conversational mode
# --color      : Pretty console output

echo "💬 Starting Interactive TurboQuant Chat (Qwen3.5-35B-A3B)..."
echo "Model: $MODEL_PATH"
echo "Optimizations: Asymmetric TQ (Q8/TQ4) | Context: 16k"
echo "--------------------------------------------------------"

"$CLI_BIN" \
  -m "$MODEL_PATH" \
  -ctk q8_0 \
  -ctv turbo4 \
  --flash-attn on \
  -c 16384 \
  -cnv \
  --color on \
  -ngl 99 \
  --reasoning off \
  -p "You are Qwen, a helpful and efficient AI assistant powered by TurboQuant. Answer concisely and accurately." \
  "$@"
