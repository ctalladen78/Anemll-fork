#!/bin/bash

# Configuration
MODEL_PATH="/Users/ctalladen/.huggingface/unsloth-Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
SERVER_BIN="/Users/ctalladen/Documents/Antigravity-Claude/output-Apr3/llama-cpp-turboquant/build/bin/llama-server"

# Optimal TurboQuant Flags:
# -ctk q8_0    : K-cache Q8_0 (High accuracy)
# -ctv turbo4  : V-cache Turbo4 (PolarQuant 4-bit, 4x memory savings)
# --flash-attn : Required for TQ kernels
# -c 32768     : 32k Context Window (Costs only 171 MiB of VRAM with TQ)

echo "🚀 Starting Optimal TurboQuant Server for Qwen3.5-35B-A3B..."
echo "Model: $MODEL_PATH"
echo "Context: 32,768 tokens (KV Cache Footprint: ~171 MiB)"

"$SERVER_BIN" \
  -m "$MODEL_PATH" \
  -ctk q8_0 \
  -ctv turbo4 \
  --flash-attn on \
  -c 16384 \
  --port 8080 \
  --host 0.0.0.0 \
  --chat-template-kwargs "{\"enable_thinking\": false}"
