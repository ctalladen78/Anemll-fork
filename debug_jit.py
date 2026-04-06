import torch
import torch.nn as nn
from anemll.models.qwen_model import QwenForCausalLM, QwenConfig, MODEL_DTYPE
from transformers import AutoConfig
import os
import sys

# Mock settings
TEST_DEVICE = "cpu"

def debug_jit():
    model_id = "Qwen/Qwen3-0.6B"
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    rope_params = getattr(hf_config, "rope_parameters", {})
    rope_theta = rope_params.get("rope_theta", 1000000.0)
    
    # Create Anemll QwenConfig
    config = QwenConfig(
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=1, # Small for debug
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        vocab_size=hf_config.vocab_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=rope_theta,
        head_dim=hf_config.head_dim,
        context_length=512,
        state_length=512
    )
    
    model = QwenForCausalLM(config, enable_coreml=True).to(TEST_DEVICE).eval()
    
    # Simple inputs
    sample_input_ids = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)
    sample_update_mask = torch.zeros((1, 1, 1, 512), dtype=torch.float16, device=TEST_DEVICE)
    sample_position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    sample_causal_mask = torch.zeros((1, 1, 1, 512), dtype=torch.float16, device=TEST_DEVICE)
    sample_current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

    print("Tracing model...")
    try:
        # Note: We need to pass arguments exactly as QwenForCausalLM.forward expects
        # def forward(self, input_ids, update_mask, position_ids, causal_mask, current_pos, IN_PREFILL=False, fixed_pos=None)
        traced = torch.jit.trace(
            model,
            (
                sample_input_ids,
                sample_update_mask,
                sample_position_ids,
                sample_causal_mask,
                sample_current_pos,
                # False, # IN_PREFILL - trace doesn't handle non-tensor bools well if passed as positional
                # 0      # fixed_pos - same
            )
        )
        print("Tracing successful!")
        
        # Search for aten::Int in the graph
        graph_str = str(traced.graph)
        if "aten::Int" in graph_str:
            print("\nFound aten::Int in graph!")
            # Find the nodes
            for line in graph_str.split("\n"):
                if "aten::Int" in line:
                    print(f"  {line.strip()}")
        else:
            print("\nNo aten::Int found in graph.")
            
    except Exception as e:
        print(f"Tracing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_jit()
