#!/bin/bash

# Configuration
MODEL_PATH="/Users/ctalladen/.huggingface/Jackrong-Qwopus3.5-9B-v3-GGUF-Q6_K/Qwen3.5-9B.Q6_K.gguf"

SERVER_BIN="/Users/ctalladen/Documents/Antigravity-Claude/output-Apr3/llama-cpp-turboquant/build/bin/llama-server"

# ctx | 16384 | 32768 | 131072 |

$SERVER_BIN \
    -m "$MODEL_PATH" \
    -ctk q8_0 \
    -ctv turbo4 \
    --flash-attn on \
    -ngl 99 \
    -c 32768 \
    -n -1 \
    --temp 0.7 \
    --repeat-penalty 1.1 \
    -t 8 \
    --port 8080 \
    --chat-template-kwargs "{\"enable_thinking\": false}"

