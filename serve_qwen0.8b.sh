


SERVER_BIN="/Users/ctalladen/.llama_cpp/build/bin/llama-server"

MODEL_PATH="/Users/ctalladen/.huggingface/unsloth-Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf"

# System message to inject (passed via API after server starts)
SYSTEM_MESSAGE="You are a helpful AI assistant with tools available. Use tools when appropriate to help the user."

"$SERVER_BIN" \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -c 131072 \
    -n -1 \
    --temp 0.7 \
    --repeat-penalty 1.1 \
    -t 8 \
    --no-mmap \
    --chat-template-kwargs "{\"enable_thinking\": false}" \
    --props \
    --port 8080 \
