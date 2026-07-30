import os
import time
import numpy as np
import onnxruntime as ort

# Configuration
FP32_MODEL_PATH = "models/federated/global_model.onnx"
INT8_MODEL_PATH = "models/federated/global_model_quantized.onnx"
NUM_RUNS = 50  # Number of inference runs for accurate averaging

def create_dummy_input():
    """Creates a dummy 1-channel, 224x224 float32 tensor matching our model's input."""
    return np.random.rand(1, 1, 224, 224).astype(np.float32)

def benchmark_model(model_path, model_name):
    print(f"\n🚀 Benchmarking {model_name}...")
    
    # Initialize ONNX Runtime session (CPU only to simulate edge hardware)
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    dummy_input = create_dummy_input()
    input_feed = {input_name: dummy_input}
    
    # Warmup runs (to avoid cold-start CPU overhead)
    for _ in range(5):
        session.run(None, input_feed)
    
    # Benchmark runs
    times = []
    for _ in range(NUM_RUNS):
        start_time = time.perf_counter()
        session.run(None, input_feed)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000) # Convert to milliseconds
        
    avg_time = np.mean(times)
    fps = 1000 / avg_time
    
    print(f"   Average Inference Time: {avg_time:.2f} ms")
    print(f"   Estimated FPS: {fps:.2f}")
    return avg_time, fps

def main():
    print("="*50)
    print("📊 EDGE INFERENCE BENCHMARKING")
    print("="*50)
    
    if not os.path.exists(FP32_MODEL_PATH):
        print(f"❌ Error: {FP32_MODEL_PATH} not found.")
        return
        
    if not os.path.exists(INT8_MODEL_PATH):
        print(f"❌ Error: {INT8_MODEL_PATH} not found.")
        return

    fp32_time, fp32_fps = benchmark_model(FP32_MODEL_PATH, "FP32 Model")
    int8_time, int8_fps = benchmark_model(INT8_MODEL_PATH, "INT8 Quantized Model")
    
    speedup = fp32_time / int8_time
    
    print("\n" + "="*50)
    print("🏆 BENCHMARK RESULTS")
    print("="*50)
    print(f"FP32 Average Latency:  {fp32_time:.2f} ms ({fp32_fps:.2f} FPS)")
    print(f"INT8 Average Latency:  {int8_time:.2f} ms ({int8_fps:.2f} FPS)")
    print(f"Speedup Factor:        {speedup:.2f}x faster")
    print("="*50)
    print("\n✅ Level 9 Complete! Edge benchmarking finished.")

if __name__ == "__main__":
    main()