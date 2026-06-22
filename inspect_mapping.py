import torch
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

print("Layer 0 MLP Experts inspection:")
mlp = model.model.layers[0].mlp
for name, module in mlp.named_modules():
    if "expert" in name:
        print(f"Module: {name} | Type: {type(module)}")

print("\nLayer 0 MLP Experts parameters:")
for name, param in mlp.named_parameters():
    print(f"Param: {name} | Shape: {list(param.shape)}")
