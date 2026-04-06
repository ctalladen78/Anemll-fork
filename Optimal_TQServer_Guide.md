# Optimal TurboQuant Server Guide (Qwen3.5-35B-A3B)

This guide provides instructions for running the `llama-server` using the most efficient TurboQuant configuration identified during benchmarking.

## 1. Optimal Configuration
Based on our analysis, the **Asymmetric TurboQuant** mode provides the best throughput while keeping the KV cache footprint extremely low.

- **K-Cache Quant**: `q8_0` (High precision for keys)
- **V-Cache Quant**: `turbo4` (4-bit PolarQuant for values)
- **Context Length**: Supports up to 32k+ in ~171 MiB of KV VRAM.

## 2. Server Command
Run the server from this workspace using the following command:

```bash
# From the workspace root (/Users/ctalladen/Documents/Antigravity-Claude/output-Apr3)
./llama-cpp-turboquant/build/bin/llama-server \
  -m ~/.huggingface/unsloth-Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ctk q8_0 \
  -ctv turbo4 \
  --flash-attn \
  -c 32768 \
  --port 8080
```

## 3. Flags Breakdown
| Flag | Description | Recommendation |
| :--- | :--- | :--- |
| `-ctk q8_0` | K-cache quantization | Use `q8_0` for accuracy. |
| `-ctv turbo4` | V-cache quantization | Use `turbo4` for 4x memory savings. |
| `--flash-attn` | Required for TQ | Must be enabled for TurboQuant kernels. |
| `-c 32768` | Context window | 32k context costs only 171MiB with TQ. |

## 4. Automation Script
A convenience script `run_tq_optimized.sh` has been created in this directory. 
You can start the server by running:
```bash
./run_tq_optimized.sh
```

---
*Guide created by Antigravity.*
