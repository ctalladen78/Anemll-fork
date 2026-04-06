#!/bin/bash

SERVER_BIN="/Users/ctalladen/.llama_cpp/build/bin/llama-server"
MODEL_PATH="/Users/ctalladen/.huggingface/Jackrong-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf"

"$SERVER_BIN" \
  -m "$MODEL_PATH" \
  --jinja \
  -c 0 \
  --host 127.0.0.1 \
  --port 8033
