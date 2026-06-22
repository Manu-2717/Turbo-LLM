import torch
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

print("=" * 60)
print("PHASE 1 — META MODEL PROBE")
print("=" * 60)

print("\nLoading config only...")
config = AutoConfig.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

print("Architecture:", config.architectures)
print("Hidden size:", config.hidden_size)
print("Layers:", config.num_hidden_layers)

print("\nCreating empty model (NO WEIGHTS)...")

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

print("✓ Model instantiated")

# -----------------------------
# Count tensors
# -----------------------------

total_params = 0
meta_params = 0

for p in model.parameters():
    total_params += p.numel()
    if p.device.type == "meta":
        meta_params += p.numel()

print("\nParameter Summary")
print("------------------")
print("Total:", f"{total_params:,}")
print("Meta :", f"{meta_params:,}")

# -----------------------------
# Inspect layers
# -----------------------------

print("\nInspecting Layers")
print("------------------")

layers = model.model.layers

print("Layer count:", len(layers))

# -----------------------------
# Detect MoE
# -----------------------------

print("\nSearching for MoE blocks...")
print("------------------")

found = 0

for i, layer in enumerate(layers):
    names = []
    for name, module in layer.named_modules():
        if any(
            k in name.lower()
            for k in [
                "expert",
                "router",
                "gate"
            ]
        ):
            names.append(name)

    if names:
        found += 1
        if i == 0:
            print(f"\nLayer {i} (Detailed Structure)")
            print("---------------------------")
            mlp = layer.mlp
            print("MLP Gate module:", type(mlp.gate), mlp.gate)
            print("MLP Experts module:", type(mlp.experts), len(mlp.experts) if hasattr(mlp.experts, '__len__') else "unknown")
            print("MLP Experts details:")
            for name, param in mlp.named_parameters():
                print(f"  Param: {name} | Shape: {list(param.shape)} | Dtype: {param.dtype}")
        else:
            if found <= 5 or found == 48:
                print(f"Layer {i} (MoE blocks: {', '.join(names)})")
            elif found == 6:
                print("...")

print("\nMoE layers found:", found)

# -----------------------------
# Verify memory
# -----------------------------

if torch.cuda.is_available():
    print("\nGPU Memory")
    print(
        torch.cuda.memory_allocated()
        / 1024**2,
        "MB"
    )

print("\nSUCCESS")
print("No weights loaded.")
print("Ready for Phase 2.")
