# Qwen 3.5-0.8B ANE Conversion: Master Strategy & Implementation Plan

This document serves as the single source of truth for the **ANE (Apple Neural Engine)** conversion project, targeting the **M4 NPU (38 TOPS)** for full-time background subagent inference with near-zero CPU/GPU utilization.

## 1. Project Objective
Convert the hybrid **Qwen 3.5-0.8B** architecture (Linear Attention/SSM + Full Attention) into an optimized CoreML `.mlpackage` that stays 100% resident on the ANE hardware.

---

## 2. Technical Blueprint

### Framework Strategy
- **Primary Architecture**: Interleaved 18x Linear Attention (SSM-like) and 6x Full Attention layers (GQA).
- **Source Environment**: `mlx-lm` for Apple Silicon-native tensor layout handling.
- **ANE residency Trick**: Use `nn.Conv2d` with 1x1 kernels for dense layer mapping to force ANE residency (via **Anemll** patterns).
- **Optimization**: **4-bit Weight Palettization** to ensure the model and context fit within the ANE's high-speed local SRAM (approx. 32-128MB working set).

### Precision & Shape Constraints
- **Precision**: Source **BF16** must be cast to **FP16** for ANE hardware compatibility.
- **Context Length**: Target **4096 Enumerated Shapes** (Buckets: 512, 1024, 2048, 4096) to avoid re-compilation latency on the M4.
- **Caching**: Utilize **Stateful Inputs** (macOS 15 native) to keep the KV cache resident on the ANE controller across generation steps.

---

## 3. Implementation Roadmap

### [ ] Phase I: Environment Setup
- Create a dedicated python environment for `coremltools`, `mlx-lm`, and `torch`.
- Clone `https://github.com/Anemll/Anemll` for Qwen conversion logic.
- Clone `https://github.com/mechramc/Orion` for ANE hardware profiling tools.

### [ ] Phase II: Kernel Development (Hybrid Architecture)
- Deep dive into the `linear_attention` (SSM) layer math.
- Implement the `QwenLinearAttention` module in the `Anemll` model template.
- Verify that the `argmax` logic is mapped to the NPU to minimize CPU-side overhead.

### [ ] Phase III: Weight Extraction & Conversion
- Load `Qwen3.5-0.8B-Base` from `~/.huggingface/` using `mlx-lm`.
- Run the conversion script with **Enumerated Shapes** and **4-bit Palettization**.
- Target `minimum_deployment_target=ct.target.macOS15`.

### [ ] Phase IV: Validation & Benchmarking
- Profile 100% NPU residency using Xcode Instruments.
- Verify 0% CPU/GPU usage during a background inference loop.
- Perform logic validation (e.g., "Car Wash Test").

---

## 4. Why this matters for the M4
The M4's ANE is optimized for longer context and complex LLM ops that previously "kicked" back to the GPU. For an autonomous subagent, this setup allows the agent to think silently in the background while you develop on the main CPU/GPU without any performance contention.

---
*Strategy consolidated by Antigravity. Standing by for rest period.*
