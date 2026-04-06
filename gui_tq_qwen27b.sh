#!/bin/bash

# Configuration
MODEL_PATH="/Users/ctalladen/.huggingface/Jackrong-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf"
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
  -c 4096 \
  --port $PORT \
  --host 0.0.0.0 \
  --webui \
  --chat-template-kwargs "{\"enable_thinking\": false}"
