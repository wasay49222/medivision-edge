import os
from onnxruntime.quantization import quantize_dynamic, QuantType

# Configuration
ONNX_PATH = "models/federated/global_model.onnx"
QUANTIZED_PATH = "models/federated/global_model_quantized.onnx"

def main():
    print("🔄 Starting INT8 Dynamic Quantization...")
    print("   This may take a minute as it analyzes the model graph.")
    
    # Perform dynamic quantization (FP32 -> INT8)
    quantize_dynamic(
        model_input=ONNX_PATH,
        model_output=QUANTIZED_PATH,
        weight_type=QuantType.QUInt8
    )
    
    # Get file sizes for comparison
    original_size = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    quantized_size = os.path.getsize(QUANTIZED_PATH) / (1024 * 1024)
    reduction = ((original_size - quantized_size) / original_size) * 100
    
    print("\n" + "="*50)
    print("📊 EDGE OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Original ONNX (FP32):  {original_size:.2f} MB")
    print(f"Quantized ONNX (INT8): {quantized_size:.2f} MB")
    print(f"Size Reduction:        {reduction:.1f}%")
    print("="*50)
    print(f"✅ Quantization Complete! Model saved to: {QUANTIZED_PATH}")

if __name__ == "__main__":
    main()