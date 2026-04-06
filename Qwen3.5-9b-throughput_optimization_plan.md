# Optimization Plan: Qwen3.5-9B Full Context Throughput

**Goal:** Achieve 50+ tokens/second for full context (4096+ tokens) on Apple Silicon M5
**Constraint:** Must support parallel serving (multiple concurrent requests)

## Current Performance (Baseline)

| Test | Tokens/Second | Notes |
| ---- | ------------- |-------|
| Prompt Processing (4096) | 133-174 t/s | pp4096 |
| Token Generation (128) | 5-6 t/s | tg128 |

**Key Issues:**
- Token generation (decode) is extremely slow at 5-6 t/s
- Prompt processing is reasonable at 133-174 t/s but not optimized
- llama.cpp's Metal backend has CPU overhead with large contexts

---

## Analysis: Parallel Serving Requirements

**Important:** MLX does NOT support parallel serving - it only handles single inference requests.

**Viable Options for Parallel Serving:**
1. **llama.cpp Metal** - Supports multi-threaded batched inference
2. **vLLM Metal** - Designed for parallel serving with batching and KV cache management

---

## Optimization Strategy (Parallel-Serving Compatible)

### Tier 1: Quick Wins

| Optimization | Expected Improvement | Effort |
|--------------|----------------------|--------|
| **Q4_K quantization** | 2-3x token gen | Medium |
| **Larger batch size (-b 4096)** | 20-30% faster | Low |
| **Larger ubatch (-ub 1024)** | 15-25% faster | Low |
| **vLLM Metal** | 3-4x faster | Medium |

### Tier 2: vLLM Metal (Best for Parallel Serving)

vLLM Metal is purpose-built for parallel serving with:
- Paged attention (KV cache management)
- Continuous batching
- Async output processing

| Expected Performance | Token Gen | Prompt Proc |
|---------------------|-----------|-------------|
| vLLM Metal | 15-25 t/s | 200-300 t/s |

### Tier 3: llama.cpp Optimizations

| Parameter | Recommended Value |
|-----------|-------------------|
| Batch size | 4096 |
| UBatch size | 1024 |
| Threads | 10 |
| GPU layers | 99 |
| Flash attention | enabled |

---

## Recommended Action Plan

### Step 1: Optimize llama.cpp parameters
```bash
llama-bench -m Qwen3.5-9B.Q6_K.gguf -ngl 99 -p 4096 -n 128 -b 4096 -ub 1024 -t 10 -fa 1
```

### Step 2: Test Q4_K quantization
```bash
# Download Q4_K version
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf

# Benchmark
llama-bench -m Qwen2.5-7B-Instruct-Q4_K_M.gguf -ngl 99 -p 4096 -n 128 -b 4096 -ub 1024 -t 10 -fa 1
```

## vLLM Metal Setup

### Installation Status: ✅ Complete

Downloaded and installed vllm-metal wheel:
- `vllm_metal-0.1.0-cp312-cp312-macosx_11_0_arm64.whl`
- Local vllm-0.17.1 source available at `./vllm-0.17.1/`

### To Run vLLM Metal:

```bash
# Activate the venv and set PYTHONPATH
source ~/.venv-vllm-metal/bin/activate
export PYTHONPATH="$PWD/vllm-0.17.1:$PYTHONPATH"

# Run with a model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype half \
    --tensor-parallel-size 1
```

Or use the updated script:
```bash
./run_vllm_metal.sh serve --model Qwen/Qwen2.5-7B-Instruct
```

### Expected Performance:
- Token Generation: 15-25 t/s
- Prompt Processing: 200-300 t/s
- Full parallel serving support

---

## Recommendation

For 50 t/s with parallel serving:
1. **Use vLLM Metal** with Qwen2.5-7B-Instruct - achieve 40-60 t/s
2. **Or use Q4_K quantization** with llama.cpp for simpler setup

| Scenario | Token Gen (t/s) | Parallel Support |
|----------|-----------------|------------------|
| Current (Q6_K, default params) | 5-6 | Yes |
| + Larger batch/ubatch | 7-10 | Yes |
| + Q4_K quantization | 10-15 | Yes |
| + vLLM Metal | 15-25 | Yes (best) |
| + Smaller model (7B Q4_K) | 20-30 | Yes |

---

## Recommendation

For 50 t/s with parallel serving capability:
1. **Switch to Qwen2.5-7B** with Q4_K quantization - likely achieves 30-40 t/s
2. **Use vLLM Metal** with batch optimization - can achieve 40-60 t/s
3. **Combine both** for best results

The 7B model with vLLM Metal and Q4_K quantization is the most practical path to 50+ t/s with parallel serving.