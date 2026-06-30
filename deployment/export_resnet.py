"""
export_resnet.py
Exports ResNet50 to:
  - ONNX  → models/resnet50_onnx/1/model.onnx
  - TorchScript → models/resnet50_notonnx/1/model.pt

Tensor names match the Triton config.pbtxt files:
  resnet50_onnx    : input="input",    output="output"
  resnet50_notonnx : input="input__0", output="output__0"  (implicit for TorchScript)
"""
import os
import torch
import torchvision.models as models

# ── paths ────────────────────────────────────────────────────────────────────
ONNX_DIR  = os.path.join("models", "resnet50_onnx",    "1")
PT_DIR    = os.path.join("models", "resnet50_notonnx", "1")
ONNX_PATH = os.path.join(ONNX_DIR, "model.onnx")
PT_PATH   = os.path.join(PT_DIR,   "model.pt")

os.makedirs(ONNX_DIR, exist_ok=True)
os.makedirs(PT_DIR,   exist_ok=True)

# ── load pre-trained ResNet50 ────────────────────────────────────────────────
print("Loading ResNet50 (pretrained) ...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

# ── export ONNX ──────────────────────────────────────────────────────────────
print(f"Exporting ONNX  → {ONNX_PATH}")
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],          # must match config.pbtxt
    output_names=["output"],        # must match config.pbtxt
    dynamic_axes={
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
# Downgrade IR version if needed for older Triton servers (e.g. Triton 24.01-py3)
import onnx
onnx_model = onnx.load(ONNX_PATH)
if onnx_model.ir_version > 9:
    print(f"Downgrading model IR version from {onnx_model.ir_version} to 9 for compatibility...")
    onnx_model.ir_version = 9
    onnx.save(onnx_model, ONNX_PATH)
print(f"  saved {os.path.getsize(ONNX_PATH) / 1e6:.1f} MB")

# ── export TorchScript ───────────────────────────────────────────────────────
print(f"Exporting TorchScript → {PT_PATH}")
with torch.no_grad():
    scripted = torch.jit.trace(model, dummy)
    scripted.save(PT_PATH)
print(f"  saved {os.path.getsize(PT_PATH) / 1e6:.1f} MB")

print("\nDone! Both ResNet50 models exported successfully.")
