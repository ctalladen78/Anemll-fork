import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros((1, 10), dtype=torch.float16))
        self.linear = nn.Linear(10, 10).to(torch.float16)

    def forward(self, x):
        # Pattern 1: [:] = 
        new_state = self.state + x
        self.state[:] = new_state
        return self.linear(self.state)

model = MiniModel().eval()
sample_input = torch.zeros((1, 10), dtype=torch.float16)
traced = torch.jit.trace(model, sample_input)

states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1, 10), dtype=np.float16),
        name="state"
    )
]

print("Starting conversion with [:] = pattern...")
try:
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 10), dtype=np.float16)],
        states=states,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram"
    )
    print("Success with [:] = !")
except Exception as e:
    print(f"Failed with [:] = : {e}")

class MiniModelSlice(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros((1, 10, 5), dtype=torch.float16))

    def forward(self, x):
        # Pattern 2: slice assignment
        self.cache[:, :, 0:1] = x.unsqueeze(-1)
        return self.cache.sum()

model2 = MiniModelSlice().eval()
sample_input2 = torch.zeros((1, 10), dtype=torch.float16)
traced2 = torch.jit.trace(model2, sample_input2)
states2 = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1, 10, 5), dtype=np.float16),
        name="cache"
    )
]

print("\nStarting conversion with slice assignment pattern...")
try:
    mlmodel2 = ct.convert(
        traced2,
        inputs=[ct.TensorType(name="x", shape=(1, 10), dtype=np.float16)],
        states=states2,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram"
    )
    print("Success with slice assignment!")
except Exception as e:
    print(f"Failed with slice assignment: {e}")


class MiniModelFunctional(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros((1, 10), dtype=torch.float32))
        self.linear = nn.Linear(10, 10)

    def forward(self, x, state_in=None):
        if state_in is None:
            state_in = self.state
        new_state = state_in + x
        out = self.linear(new_state)
        return out, new_state

model_func = MiniModelFunctional().eval()
dummy_state = torch.zeros((1, 10), dtype=torch.float32)
sample_input3 = torch.zeros((1, 10))
# Trace with BOTH input AND state
traced_func = torch.jit.trace(model_func, (sample_input3, dummy_state))

states_func = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=(1, 10), dtype=np.float32),
        name="state"
    )
]

print("\nStarting conversion with Functional State pattern...")
try:
    mlmodel_func = ct.convert(
        traced_func,
        inputs=[
            ct.TensorType(name="x", shape=(1, 10)),
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, 10), dtype=np.float32), name="state")
        ],
        outputs=[
            ct.TensorType(name="output", shape=(1, 10)),
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, 10), dtype=np.float32), name="state")
        ],
        states=states_func,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram"
    )
    print("Success with Functional State!")
except Exception as e:
    print(f"Failed with Functional State: {e}")
