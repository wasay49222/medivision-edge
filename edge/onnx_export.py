import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
import onnx

# Configuration
MODEL_PATH = "models/federated/global_model_secagg.pth"
ONNX_PATH = "models/federated/global_model.onnx"
DEVICE = torch.device("cpu")

def main():
    print("🔄 Loading global PyTorch model...")
    model = resnet18(weights=None, num_classes=2)
    # Must match the 1-channel architecture used in training
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    print("🔄 Exporting to ONNX format...")
    # Create a dummy input matching our 1-channel, 224x224 image shape
    dummy_input = torch.randn(1, 1, 224, 224, device=DEVICE)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Verify the ONNX model
    print("🔍 Verifying ONNX model...")
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    
    # Get file size
    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"✅ ONNX Export Successful!")
    print(f"📦 Model saved to: {ONNX_PATH}")
    print(f"📏 File size: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()