# Strategy 36: SRAM-Native 4B Residency Map

This map illustrates the memory layout of the Qwen-4B (Qwen2.5-3B) model optimized for **100% Neural Engine Residency** on M5 hardware.

## Architecture

| Component | Working Set (Weights) | Working Set (KV-Cache) | Total SRAM Req | Residency Plan |
| :--- | :--- | :--- | :--- | :--- |
| **Embeddings** | 389 MB | N/A | 389 MB | Disk-Paged (1-time) |
| **Transformer (L0)** | 60 MB | 10 MB | 70 MB | **SRAM-Native** (<128MB) |
| **Transformer (L1)** | 60 MB | 10 MB | 70 MB | **SRAM-Native** (<128MB) |
| ... | ... | ... | ... | ... |
| **Transformer (L27)** | 60 MB | 10 MB | 70 MB | **SRAM-Native** (<128MB) |
| **LM-Head (C0)** | 97 MB | N/A | 97 MB | **SRAM-Native** (<128MB) |
| **LM-Head (C1)** | 97 MB | N/A | 97 MB | **SRAM-Native** (<128MB) |
| **LM-Head (C2)** | 97 MB | N/A | 97 MB | **SRAM-Native** (<128MB) |
| **LM-Head (C3)** | 97 MB | N/A | 97 MB | **SRAM-Native** (<128MB) |

---

## Data Flow Diagram

```mermaid
graph TD
    subgraph "SRAM-Native Boundary (128MB)"
    A[L0 Segment: 70MB] --> B[L1 Segment: 70MB]
    B --> C[L2 Segment: 70MB]
    C --> D[...]
    D --> E[L27 Segment: 70MB]
    end

    subgraph "Memory-Mapped Weights"
    W[Weight Flash/DRAM] -.->|Burst Load| A
    W -.->|Burst Load| B
    W -.->|Burst Load| E
    end

    subgraph "Zero-Copy State"
    K[KV-Cache Buffer] <-->|Link| A
    K <-->|Link| B
    K <-->|Link| E
    end

    E --> L1[LM-Head C0: 97MB]
    E --> L2[LM-Head C1: 97MB]
    E --> L3[LM-Head C2: 97MB]
    E --> L4[LM-Head C3: 97MB]

    L1 & L2 & L3 & L4 --> T[Argmax Token Selection]
```

## Performance Targets
* **Residency**: 100% (No DRAM Weight Streaming during dispatch)
* **Energy**: ~3.5W (Full NPU)
* **Target TPS**: 10-15 (M5 High-Efficiency Architecture)

