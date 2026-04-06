#!/bin/bash

# Configuration 
# Context LIMITS 16384 32768 (sweet spot) 65536
MODEL_PATH="/Users/ctalladen/.huggingface/unsloth-Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
SERVER_BIN="/Users/ctalladen/Documents/Antigravity-Claude/output-Apr3/llama-cpp-turboquant/build/bin/llama-server"

PORT=8080

echo "🚀 Starting TurboQuant WebUI Server for Qwen3.5-27B..."
echo "Model: $MODEL_PATH"
echo "WebUI: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"

"$SERVER_BIN" \
  -m "$MODEL_PATH" \
  -ctk q8_0 \
  -ctv turbo4 \
  --flash-attn on \
  -c 65536 \
  --port $PORT \
  --host 0.0.0.0 \
  --webui \
  --chat-template-kwargs "{\"enable_thinking\": false}"
