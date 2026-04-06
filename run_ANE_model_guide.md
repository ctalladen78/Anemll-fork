The **ANEMLL** project is specifically designed to bypass the GPU and run LLMs directly on the **Apple Neural Engine (ANE)**. This particular Qwen 1.7B model is highly optimized because it has been converted into the `.mlmodelc` format (CoreML), which is the only way to ensure 100% NPU utilization on a MacBook.

Since you are running an M4, this model will be exceptionally fast because it leverages the improved neural accelerators on your chip.

### 1. Download and Installation
To get this working, you need to clone the repository and extract the compiled CoreML files.

**Step 1: Clone the Repository**
```bash
# Ensure you have git-lfs installed (brew install git-lfs)
git lfs install
git clone https://huggingface.co/anemll/anemll-Qwen-Qwen3-1.7B-ctx2048_0.3.5
cd anemll-Qwen-Qwen3-1.7B-ctx2048_0.3.5
```

**Step 2: Unzip the Model Files**
ANEMLL models are often distributed as zipped `.mlmodelc` folders to save space on HuggingFace. You must unzip them for the Python script to find them.


**Step 3: Setup Dependencies**
You will need `coremltools` and `transformers` to run the inference scripts provided in the repo.
```bash
pip install coremltools transformers huggingface_hub
```

---

### 2. Running the Model on ANE
The repository includes a `chat.py` script. This script is hardcoded to load the model and map it to the ANE.

```bash
# Run the basic chat interface
python chat.py --meta ./meta.yaml
```

**Note:** The first time you run this, macOS will perform a "first load" optimization. It might take 10–30 seconds to load into memory. Every subsequent run will be near-instant.

---

### 3. Verify 100% NPU Utilization
To confirm that the GPU is idle and the NPU (ANE) is doing the work, use the `powermetrics` tool.

1.  Open a **new** Terminal window.
2.  Run the following command:
    ```bash
    sudo powermetrics --samplers ane,gpu
    ```
3.  **The Test:** Start chatting with the model in your first terminal.
4.  **The Result:** Watch the `ANE Power` metric. 
    * If you see **ANE Power** jumping to **1000mW - 4000mW** while the **GPU Power** stays near **0mW**, you have successfully confirmed the model is running 100% on the NPU.

### Why this is a "Proof of Concept" for your projects
For your work on **Clodbot** and **Picobot**, this model structure is the blueprint for "rugged," local-first agents. By running on the ANE:
* **Zero GPU Tax:** Your GPU remains 100% free for UI rendering, game engines (like Godot), or video processing.
* **Thermal Efficiency:** The NPU is significantly more power-efficient than the GPU, meaning your MacBook won't throttle during long coding sessions with your agents.
* **Context:** The 2048 context in this 1.7B model is specifically tuned to fit within the ANE's memory cache constraints, which is why it's a "proof" that larger contexts are viable on the NPU without falling back to the slower CPU.


# Analysis: ANE Model Loading & Utilization Guide

This analysis examines the `run_ANE_model_guide.md` file and provides a structured walkthrough of the technical mechanisms used to load and run Large Language Models (LLMs) on the Apple Neural Engine (ANE).

## 1. Summary of `run_ANE_model_guide.md`

The guide serves as a proof-of-concept for running LLMs (specifically Qwen 1.7B) with **100% NPU utilization** on Apple Silicon hardware.

### Key Takeaways:
- **Architecture Priority**: Bypassing the GPU entirely to save resources for UI and other tasks.
- **Model Format**: Heavy emphasis on `.mlmodelc` (Compiled CoreML).
- **Optimization Strategy**: 
    - **Quantization**: Uses LUT-6 (6-bit Look-Up Table) quantization.
    - **Segmentation**: Large model layers are chunked (e.g., `chunk_01of02.mlmodelc`) to fit within ANE's specific hardware cache constraints.
- **Verification Method**: Standard validation using `sudo powermetrics` to isolate ANE vs. GPU power consumption.

---

## 2. Walkthrough: Loading ML Models onto ANE NPU

Loading models onto the ANE requires specific API calls and model structures to prevent the system from falling back to the CPU or GPU.

### Phase 1: Model Organization
Ensure your model is split into functional components. The ANE has limited buffer sizes (e.g., ~400MB–800MB depending on the chip), so monolithic 7B+ models must be modularized:
- **Embedding Layer**: Maps token IDs to vectors.
- **Transformer Chunks**: Sequences of FFN (Feed-Forward Network) layers.
- **LM Head**: Final prediction layer.

### Phase 2: Loading Logic (Python)
The `ANEMLL` framework uses `coremltools` to bind models to the hardware.

```python
import coremltools as ct
from pathlib import Path

def load_to_npu(model_path: str):
    """
    Loads a CoreML model specifically targeting the Neural Engine.
    """
    # 1. Select Compute Unit
    # CPU_AND_NE ensures the model runs on NPU whenever possible.
    # CPU_ONLY is used for debugging.
    # GPU is avoided in this architecture.
    compute_unit = ct.ComputeUnit.CPU_AND_NE

    # 2. Handle Compiled (.mlmodelc) vs Source (.mlpackage)
    path = Path(model_path)
    if path.suffix == '.mlmodelc':
        # Compiled models skip the runtime optimization phase
        model = ct.models.CompiledMLModel(str(path), compute_unit)
    else:
        # Standard packages allow for inspection/modification
        model = ct.models.MLModel(str(path), compute_units=compute_unit)
        
    return model
```

### Phase 3: Function Dispatch
If your model is an "ML Program" (CoreML 5+), it might contain multiple entry points (e.g., `prefill` for context processing and `infer` for token generation).

```python
# Loading a specific function from a multi-function model
prefill_fn = ct.models.CompiledMLModel(
    str(path), 
    compute_unit, 
    function_name='prefill'
)
```

### Phase 4: State Awareness (CoreML 9+)
For high-performance zero-copy inference, use `make_state()` to manage KV-caches directly on the NPU:

```python
# Create a persistent state buffer on the NPU
model_state = model.make_state()

# Execute with state
outputs = model.predict(inputs, state=model_state)
```

---

## 3. Benefits of ANE Residency
1. **Efficiency**: ANE consumes ~1/10th the power of the GPU for similar throughput.
2. **Availability**: Keeps the GPU free for 60fps UI rendering or heavy graphical applications.
3. **Thermal Stability**: Reduced power draw means less heat and lower chance of thermal throttling during long inference sessions.

> [!TIP]
> **Verification**: Always run `sudo powermetrics --samplers ane,gpu` while running your model. If **ANE Power** stays at 0mW while **GPU Power** jumps, the model has fallen back to the GPU due to incompatible operators or buffer sizes.

```bash
source .venv/bin/activate
cd Anemll
python chat.py --meta ./meta.yaml

# run webgui
$ .venv/bin/python Anemll/webgui.py --model-dir Anemll --port 5000
```