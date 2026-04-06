# Progress: Qwen ANE Optimization (M5 Hardware)

## 1. Project Objectives (PRD)
*   [x] Achieve 100% ANE residency for Qwen 0.6B (Strategy 24).
*   [x] Achieve 100% ANE residency for Qwen 4B (Strategy 36).
*   [ ] Achieve >10 TPS for Qwen 4B on M5.
*   [ ] Deploy interactive inference server (Swift).

## 2. Chronological Log

### 2026-04-05
*   **18:30**: **Strategy 36 Initialization**. Refactored `QwenForCausalLM` to support 36 layers with decoupled biases.
*   **18:50**: **Conversion Success**. 36 segments generated for Qwen2.5-3B-Instruct (4B).
*   **19:00**: **Benchmark Run (S36)**. Achieved **3.29 TPS**. Identified "Dispatch Churn" (36 dispatches per token).
*   **19:05**: **Stress Test Verification**. Confirmed **7-8% NPU utilization** on M5. Validated 100% Neural Engine residency.
*   **19:09**: **Strategy 36 Completed**. Transitioning to Strategy 37 (18-Segment Fusion).

## 3. Active Tasks
- [x] Strategy 36: SRAM-Native (1x36) Residency (4B).
- [ ] Strategy 37: SRAM-Native Fusion (2x18) (4B).
- [ ] Strategy 46: Qwen-1.5B (4096-ctx) 4-Chunk LUT8.
    - [x] Download Qwen2.5-1.5B-Instruct source weights.
    - [/] Trace and palettize 4 segments (7 layers each).
    - [ ] Update Swift Driver (h=1536, nh=12, hd=128).
    - [ ] Benchmark TPS (Target > 10.0 TPS).

## 4. Bottlenecks
*   **Dispatch Overhead**: 36 dispatches (4B) vs 4 dispatches (1.5B).
*   **SRAM Hard Limit**: Streaming weights from DRAM (400MB segments) vs SRAM-native residency.
*   **Context Pressure**: 4096-context KV-cache management on M5.
