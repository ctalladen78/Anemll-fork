# Llama-CPP Baseline Benchmarks

These benchmarks represent the baseline performance of the identified GGUF models on this system using `llama-bench`.

## System Info
- **GPU**: MTL0 (Apple Metal support)
- **Unified Memory**: Enabled
- **Recommended Max Working Set**: 26,800.60 MB

## Performance Results

### [Qwen3.5-27B](file:///Users/ctalladen/.huggingface/Jackrong-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q4_K_M.gguf)
- **Type**: Dense
- **Size**: 15.39 GiB
- **Params**: 26.90 B
- **Quantization**: Q4_K_M

| Test | Tokens/Second (t/s) |
| :--- | :--- |
| **Prompt Processing (pp512)** | 118.95 ± 0.49 |
| **Token Generation (tg128)** | 7.01 ± 0.10 |
| **Peak RAM (32k Context)** | **~20.4 GiB** |

---

### [Qwen3.5-35B-A3B](file:///Users/ctalladen/.huggingface/unsloth-Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf)
- **Type**: Mixture of Experts (MoE)
- **Size**: 20.49 GiB
- **Params**: 34.66 B
- **Quantization**: Q4_K_M

| Test | Tokens/Second (t/s) |
| :--- | :--- |
| **Prompt Processing (pp512)** | 710.38 ± 4.68 |
| **Token Generation (tg128)** | 31.46 ± 1.64 |
| **Peak RAM (32k Context)** | **~22.7 GiB** |

## Comparison

| Model | Size | pp512 (t/s) | tg128 (t/s) | RAM (32k) |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3.5-27B** | 15.39 GiB | 118.95 | 7.01 | ~20.4 GiB |
| **Qwen3.5-35B-A3B** | 20.49 GiB | **710.38** | **31.46** | ~22.7 GiB |

---

## Technical Note: Omnicoder
The Omnicoder model was identified in `~/.huggingface/Omnicoder` in Safetensors format. However, its benchmark was skipped because it uses the **`mxfp8`** (Microscaling FP8) quantization format, which is not currently compatible with the GGUF conversion tools available in your `llama-cpp` installation.

---

## Memory Footprint Calculation (32k Context)

The estimated peak RAM footprint includes the model weights (GGUF) and the KV cache (FP16) for a 32,768 token context window.

### Qwen3.5-27B
- **Weights**: 15.39 GiB
- **KV Cache**: `2 (K/V) * 64 (layers) * 4 (heads) * 128 (dim) * 32768 (ctx) * 2 (bytes)` ≈ **4.0 GiB**
- **Overhead**: ~1.0 GiB
- **Total**: **~20.4 GiB**

### Qwen3.5-35B-A3B
- **Weights**: 20.49 GiB
- **KV Cache**: `2 (K/V) * 40 (layers) * 2 (heads) * 128 (dim) * 32768 (ctx) * 2 (bytes)` ≈ **1.25 GiB**
- **Overhead**: ~1.0 GiB
- **Total**: **~22.7 GiB**

> [!NOTE]
> Despite having more parameters and a larger model file, the Qwen3.5-35B-A3B (MoE) model has a more efficient KV cache footprint due to its architecture (fewer layers and KV heads compared to the 27B model's dense configuration), yet its total memory requirement is higher because of the larger base weight size.

# Benchmark Results: Qwopus3.5-9B-v3-GGUF (Q6_K)

**Date:** 2026-04-04
**Model:** Qwen3.5-9B.Q6_K.gguf
**Size:** 6.84 GiB (7.3B params)
**Backend:** Metal (MTL) with BLAS
**Build:** 08f21453a (8589)

## Results

| Test | Tokens/Second | Notes |
| ---- | ------------- |-------|
| Prompt Processing (512 tokens) | 392.18 ± 2.07 | pp512 |
| Token Generation (128 tokens) | 16.12 ± 0.85 | tg128 |

## Configuration
- GPU Layers: 99
- Threads: 8
- Batch Size: 2048
- UBatch Size: 512
- Prompt Tokens: 512
- Generation Tokens: 128

## System Info
- CPU: Apple M5 (Accelerate)
- GPU: Apple M5
- Backend: MTL, BLAS
