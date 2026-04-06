import os
import torch
import torch.nn as nn
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig
from anemll.ane_converter.qwen_converter import QwenConverter

def start_strategy_36():
    # 1. Setup Config (3B-Instruct confirmed specs)
    config = QwenConfig(
        hidden_size=2048,
        intermediate_size=11008,
        num_attention_heads=16,
        num_hidden_layers=36,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=151936,
        state_length=1024,
        batch_size=8
    )

    # 2. Load Model
    print("Initializing Qwen-4B Architecture (Strategy 36)...")
    model = QwenForCausalLM(config)
    model.half() # Initial cast to ensure params are float16
    
    # Weights are in Qwen2.5-3B-Instruct-Metadata/ (Sharded)
    # The load_pretrained_weights method handles sharded file check.
    weight_path = "Qwen2.5-3B-Instruct-Metadata"
    print(f"Loading Weights from {weight_path}...")
    model.load_pretrained_weights(weight_path)
    model.half() # Second pass to catch any loaded float32 and cast to half
    model.eval()

    # 3. Convert Strategy 36 (SRAM-Native 1-layer segments)
    converter = QwenConverter(model, context_length=1024, lut_bits=4)
    print("Starting SRAM-Native Partitioning (36 Segments)...")
    paths = converter.convert_segmented(layers_per_segment=1, context_length=1024, batch_size=8, lut_bits=4)

    print("\nStrategy 36 Conversion Finished.")
    print(f"Total Segments Generated: {len(paths)}")
    for p in paths:
        print(f"  -> {p}")

if __name__ == "__main__":
    start_strategy_36()
