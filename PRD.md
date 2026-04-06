# PRD: Qwen ANE Optimization (M5 Hardware)

## 1. Project Overview
The objective is to achieve high-performance, Neural Engine-native inference for Qwen-class models (0.6B to 4B) on Apple M5 hardware, ensuring 100% NPU residency and hitting >33 TPS for 0.6B and >10 TPS for 4B.

## 2. Technical Requirements
- **Hardware**: Apple M5 (Neural Engine).
- **Runtime**: Swift 6.0 / CoreML 9.0 (macOS 15+).
- **Residency**: 100% ANE Residency (Zero CPU/GPU fallback).
- **SRAM Bound**: Segments must stay within the 128MB SRAM-native threshold to avoid DRAM weight-streaming latency.
- **Precision**: Atomic FP16 + LUT4 Block-wise Palettization.

## 3. Goals
- **G1**: 100% NPU Residency for 4B model (Strategy 36: Achieved).
- **G2**: Reduce Dispatch Overhead (Churn) via Segment Fusion (Strategy 37: Pending).
- **G3**: Reach >10 Tokens Per Second (TPS) for Qwen-4B on M5.

## 4. Acceptance Criteria
- [x] Strategy 24 (0.6B): >100 TPS (Achieved).
- [ ] Strategy 37 (4B): >10 TPS (Targeted).
- [x] Zero-Copy Context Persistence via persistent MultiArrays.
