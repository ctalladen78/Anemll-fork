#!/usr/bin/env python3
"""Simple web GUI for Anemll chat.py"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import threading

app = Flask(__name__)

MODEL_DIR = "."
TOKENIZER = None
MODELS = {}
METADATA = {}
TOKENIZER_OBJ = None
STOP_TOKEN_IDS = set()

conversations = {}
conversation_lock = threading.Lock()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Anemll Chat</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; display: flex; flex-direction: column; }
        #header { background: #16213e; padding: 1rem; display: flex; align-items: center; gap: 1rem; border-bottom: 1px solid #0f3460; }
        #header h1 { color: #00d9ff; font-size: 1.5rem; }
        #settings-btn { margin-left: auto; background: #0f3460; border: none; color: #fff; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; }
        #settings-btn:hover { background: #1a4a7a; }
        #chat-container { flex: 1; overflow-y: auto; padding: 1rem; max-width: 800px; margin: 0 auto; width: 100%; }
        .message { margin-bottom: 1rem; padding: 1rem; border-radius: 8px; max-width: 90%; }
        .user-message { background: #0f3460; margin-left: auto; }
        .assistant-message { background: #16213e; }
        .thinking { color: #888; font-style: italic; }
        .thought { color: #f59e0b; }
        .response { color: #00d9ff; }
        .metadata { font-size: 0.75rem; color: #666; margin-top: 0.5rem; }
        #input-container { background: #16213e; padding: 1rem; border-top: 1px solid #0f3460; }
        #input-form { max-width: 800px; margin: 0 auto; display: flex; gap: 0.5rem; }
        #user-input { flex: 1; background: #0f3460; border: 1px solid #1a4a7a; color: #fff; padding: 0.75rem; border-radius: 4px; resize: none; font-family: inherit; }
        #user-input:focus { outline: none; border-color: #00d9ff; }
        #send-btn { background: #00d9ff; border: none; color: #1a1a2e; padding: 0.75rem 1.5rem; border-radius: 4px; cursor: pointer; font-weight: bold; }
        #send-btn:hover { background: #00b8d9; }
        #send-btn:disabled { background: #666; cursor: not-allowed; }
        #loading { display: none; text-align: center; padding: 1rem; color: #00d9ff; }
        #error { display: none; background: #ff4444; color: #fff; padding: 1rem; margin: 1rem; border-radius: 4px; }
        .settings-panel { display: none; position: fixed; top: 0; right: 0; width: 300px; height: 100%; background: #16213e; padding: 1rem; border-left: 1px solid #0f3460; overflow-y: auto; z-index: 1000; }
        .settings-panel.open { display: block; }
        .settings-panel h2 { color: #00d9ff; margin-bottom: 1rem; }
        .setting-group { margin-bottom: 1rem; }
        .setting-group label { display: block; margin-bottom: 0.25rem; color: #888; }
        .setting-group input, .setting-group textarea { width: 100%; background: #0f3460; border: 1px solid #1a4a7a; color: #fff; padding: 0.5rem; border-radius: 4px; }
    </style>
</head>
<body>
    <div id="header">
        <h1>Anemll Chat</h1>
        <button id="settings-btn">Settings</button>
    </div>
    <div id="error"></div>
    <div id="chat-container"></div>
    <div id="loading">Generating response...</div>
    <div id="input-container">
        <form id="input-form">
            <textarea id="user-input" rows="2" placeholder="Enter your message..."></textarea>
            <button type="submit" id="send-btn">Send</button>
        </form>
    </div>
    <div class="settings-panel" id="settings-panel">
        <h2>Settings</h2>
        <div class="setting-group">
            <label>Model Directory</label>
            <input type="text" id="model-dir" value="Anemll">
        </div>
        <div class="setting-group">
            <label>System Prompt</label>
            <textarea id="system-prompt" rows="3">You are a helpful AI assistant.</textarea>
        </div>
        <div class="setting-group">
            <label>Max Tokens</label>
            <input type="number" id="max-tokens" value="512">
        </div>
        <div class="setting-group">
            <label>Temperature</label>
            <input type="number" id="temperature" value="0.0" step="0.1" min="0" max="2">
        </div>
    </div>
    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const loading = document.getElementById('loading');
        const errorDiv = document.getElementById('error');
        const settingsBtn = document.getElementById('settings-btn');
        const settingsPanel = document.getElementById('settings-panel');
        
        let conversationId = Date.now().toString();
        
        settingsBtn.addEventListener('click', () => {
            settingsPanel.classList.toggle('open');
        });
        
        function showError(msg) {
            errorDiv.textContent = msg;
            errorDiv.style.display = 'block';
            setTimeout(() => errorDiv.style.display = 'none', 5000);
        }
        
        function addMessage(role, content, metadata = '') {
            const div = document.createElement('div');
            div.className = `message ${role}-message`;
            
            let formattedContent = content;
            if (role === 'assistant') {
                // Handle thinking blocks
                if (content.includes('<think>')) {
                    const parts = content.split('');
                    formattedContent = '';
                    parts.forEach((part, i) => {
                        if (i === parts.length - 1) return;
                        formattedContent += `<span class="thought">${part}</span>`;
                    });
                    if (parts[parts.length - 1]) {
                        formattedContent += `<span class="response">${parts[parts.length - 1]}</span>`;
                    }
                } else {
                    formattedContent = `<span class="response">${content}</span>`;
                }
            }
            
            div.innerHTML = formattedContent + (metadata ? `<div class="metadata">${metadata}</div>` : '');
            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        document.getElementById('input-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const message = userInput.value.trim();
            if (!message) return;
            
            userInput.value = '';
            addMessage('user', message);
            sendBtn.disabled = true;
            loading.style.display = 'block';
            errorDiv.style.display = 'none';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message,
                        conversation_id: conversationId,
                        model_dir: document.getElementById('model-dir').value,
                        system_prompt: document.getElementById('system-prompt').value,
                        max_tokens: parseInt(document.getElementById('max-tokens').value),
                        temperature: parseFloat(document.getElementById('temperature').value)
                    })
                });
                
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.error || 'Request failed');
                }
                
                const data = await response.json();
                addMessage('assistant', data.response, data.metadata);
                
            } catch (err) {
                showError(err.message);
            } finally {
                sendBtn.disabled = false;
                loading.style.display = 'none';
            }
        });
        
        // Send on Ctrl+Enter
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                document.getElementById('input-form').dispatchEvent(new Event('submit'));
            }
        });
    </script>
</body>
</html>
"""


def init_models(model_dir):
    """Initialize models on demand"""
    global MODELS, METADATA, TOKENIZER_OBJ, TOKENIZER, STOP_TOKEN_IDS

    if MODELS:
        return

    sys.path.insert(0, str(Path(__file__).parent))
    import chat as anemll_chat

    model_path = Path(model_dir)
    meta_file = model_path / "meta.yaml"

    args = argparse.Namespace(
        d=str(model_path),
        embed=None,
        ffn=None,
        lmhead=None,
        pf=None,
        tokenizer=str(model_path),
        meta=str(meta_file) if meta_file.exists() else None,
        context_length=None,
        batch_size=None,
        num_chunks=None,
        split_lm_head=None,
        argmax_in_model=False,
        debug=False,
        debug_argmax=False,
        sliding_window=512,
        eval=False,
    )

    if meta_file.exists():
        import yaml

        with open(meta_file) as f:
            yaml_data = yaml.safe_load(f)
            params = yaml_data.get("model_info", {}).get("parameters", {})

            prefix = params.get("model_prefix", "model")
            lut = params.get("lut_ffn", 0)
            num_chunks = params.get("num_chunks", 1)

            args.embed = (
                params.get("embeddings", f"{prefix}_embeddings")
                .replace(".mlmodelc", "")
                .replace(".mlpackage", "")
            )
            args.lmhead = (
                params.get("lm_head", f"{prefix}_lm_head_lut{lut}")
                .replace(".mlmodelc", "")
                .replace(".mlpackage", "")
            )
            args.ffn = (
                params.get(
                    "ffn", f"{prefix}_FFN_PF_lut{lut}_chunk_01of{num_chunks:02d}"
                )
                .replace(".mlmodelc", "")
                .replace(".mlpackage", "")
            )
            args.context_length = params.get("context_length", 512)
            args.batch_size = params.get("batch_size", 64)
            args.split_lm_head = params.get("split_lm_head", 8)

    print(f"Loading models from: {model_path}")
    print(f"Meta file: {meta_file}, exists: {meta_file.exists()}")

    # Prepend model_dir to relative paths (like chat.py does at line 2363)
    if args.embed and not Path(args.embed).is_absolute():
        args.embed = str(model_path / args.embed)
    if args.lmhead and not Path(args.lmhead).is_absolute():
        args.lmhead = str(model_path / args.lmhead)
    if args.ffn and not Path(args.ffn).is_absolute():
        args.ffn = str(model_path / args.ffn)

    print(f"Embed: {args.embed}, FFN: {args.ffn}, LMHead: {args.lmhead}")

    METADATA = {}
    embed_model, ffn_models, lmhead_model, metadata = anemll_chat.load_models(
        args, METADATA
    )
    TOKENIZER_OBJ = anemll_chat.initialize_tokenizer(args.tokenizer, eval_mode=False)
    STOP_TOKEN_IDS = anemll_chat.build_stop_token_ids(TOKENIZER_OBJ)

    metadata["context_length"] = args.context_length or 512
    metadata["state_length"] = metadata["context_length"]
    metadata["batch_size"] = args.batch_size or 64
    metadata["split_lm_head"] = args.split_lm_head or 8
    metadata["debug"] = False

    MODELS = {"embed": embed_model, "ffn": ffn_models, "lmhead": lmhead_model}
    MODELS["_model_dir"] = model_dir  # Track which model dir is loaded
    METADATA.update(metadata)

    print(f"Models loaded. Context length: {METADATA['context_length']}")

    # Create state(s) - handle both unified state and per-chunk states
    ffn0 = ffn_models[0]
    if isinstance(ffn0, dict):
        # Chunked model with dicts - need to handle both unified and per-chunk states
        if metadata.get("has_global_cache", False) or metadata.get("cache_type") in (
            "unified",
            "split",
        ):
            # Use unified state from first chunk
            model_for_state = ffn0.get("prefill") or ffn0.get("infer")
            state = model_for_state.make_state()
            print("Created unified transformer state")
        else:
            # Create per-chunk states
            states = []
            for fm in ffn_models:
                model = fm.get("prefill") or fm.get("infer")
                states.append(model.make_state())
            state = states
            print(f"Created {len(states)} per-chunk states")
    else:
        state = ffn_models[0].make_state()
        print("Created single model state")

    MODELS["state"] = state

    causal_mask = anemll_chat.make_causal_mask(METADATA["context_length"], 0)
    MODELS["causal_mask"] = anemll_chat.initialize_causal_mask(
        METADATA["state_length"], False
    )

    MODELS["chat_module"] = anemll_chat


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    conversation_id = data.get("conversation_id", str(time.time()))
    model_dir = data.get("model_dir", "Anemll")
    system_prompt = data.get("system_prompt", "You are a helpful AI assistant.")
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.0)

    global MODELS, CURRENT_MODEL_DIR

    print(f"Chat request: model_dir={model_dir}, MODELS loaded: {bool(MODELS)}")

    # Check if we need to reinitialize with a different model dir
    if MODELS and getattr(MODELS, "_model_dir", None) != model_dir:
        print(f"Model dir changed, reinitializing...")
        MODELS.clear()
        globals().pop("METADATA", None)
        globals().pop("TOKENIZER_OBJ", None)
        globals().pop("STOP_TOKEN_IDS", None)

    if not MODELS:
        try:
            init_models(model_dir)
        except Exception as e:
            return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

    chat_module = MODELS["chat_module"]
    tokenizer = TOKENIZER_OBJ

    # Build conversation
    messages = [{"role": "system", "content": system_prompt}]

    with conversation_lock:
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        messages.extend(conversations[conversation_id])
        messages.append({"role": "user", "content": message})

    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = message

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = input_ids[:, :-1]  # Remove eos

    context_pos = input_ids.shape[1]

    # Convert to torch tensor (like chat.py does)
    import torch

    input_tensor = torch.from_numpy(input_ids)

    # Run prefill
    state = MODELS["state"]
    causal_mask = MODELS["causal_mask"]

    try:
        context_pos_tensor = chat_module.run_prefill(
            MODELS["embed"],
            MODELS["ffn"],
            input_tensor,
            context_pos,
            METADATA["context_length"],
            batch_size=METADATA["batch_size"],
            state=state,
            causal_mask=causal_mask,
        )
    except Exception as e:
        import traceback

        return jsonify(
            {"error": f"Prefill failed: {str(e)}\n{traceback.format_exc()}"}
        ), 500

    # Generate tokens
    response_tokens = []
    start_time = time.time()

    # Keep input_tensor as torch tensor for generation loop
    input_tensor = torch.from_numpy(input_ids)

    for _ in range(max_tokens):
        next_token = chat_module.generate_next_token(
            MODELS["embed"],
            MODELS["ffn"],
            MODELS["lmhead"],
            input_tensor,
            context_pos,
            METADATA["context_length"],
            METADATA,
            state=state,
            causal_mask=causal_mask,
            temperature=temperature,
        )

        if next_token in STOP_TOKEN_IDS:
            break

        response_tokens.append(next_token)

        # Append to input_tensor for next iteration
        next_token_str = tokenizer.decode([next_token])
        new_text = prompt + tokenizer.decode(response_tokens)
        new_input = tokenizer(new_text, return_tensors="np")["input_ids"]
        input_tensor = torch.from_numpy(new_input)
        context_pos = input_tensor.shape[1]

        # Yield to allow streaming (optional, simplified here)

    response = tokenizer.decode(response_tokens)

    # Save to conversation
    with conversation_lock:
        conversations[conversation_id].append({"role": "user", "content": message})
        conversations[conversation_id].append(
            {"role": "assistant", "content": response}
        )

    elapsed = time.time() - start_time
    tokens_per_sec = len(response_tokens) / elapsed if elapsed > 0 else 0

    return jsonify(
        {
            "response": response,
            "metadata": f"{len(response_tokens)} tokens, {tokens_per_sec:.1f} t/s",
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--model-dir", type=str, default=".")
    args = parser.parse_args()

    print(f"Starting Anemll WebGUI on http://{args.host}:{args.port}")
    print(f"Model directory: {args.model_dir}")
    app.run(port=args.port, host=args.host)
