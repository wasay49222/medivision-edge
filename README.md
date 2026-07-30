# MediVision Edge: Privacy-Preserving Medical Imaging Pipeline

[![MediVision Edge CI](https://github.com/YOUR_USERNAME/medivision-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/medivision-edge/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A zero-budget, enterprise-grade federated learning platform for medical image analysis. Enables multiple simulated hospital nodes to collaboratively train diagnostic AI models without ever sharing raw patient data.**

##  The Core Problem
1. **Privacy Laws:** HIPAA and GDPR prevent hospitals from sharing raw patient X-rays with third parties.
2. **Connectivity:** Rural clinics often lack the bandwidth for cloud-based AI inference.
3. **Data Silos:** Training a robust AI requires data from multiple hospitals, but data cannot be centralized.

## 🚀 The MediVision Edge Solution
* **Federated Learning:** Models are sent to the data. Only mathematical weight updates (gradients) leave the hospital, never the raw images.
* **Secure Aggregation:** Cryptographic zero-sum masking ensures the central server cannot reverse-engineer individual patient data from the gradients.
* **Edge Optimization:** The final global model is quantized (FP32 → INT8) and converted to ONNX, achieving a **74.9% size reduction** for deployment on low-power hardware.

## ️ System Architecture
```text
[ Hospital A (Local Data) ] --(Encrypted Gradients)--> \
[ Hospital B (Local Data) ] --(Encrypted Gradients)--> [ Central Server (FedAvg + SecAgg) ] --> [ Global ONNX Model ] --> [ Edge Deployment (Docker/Raspberry Pi) ]
[ Hospital C (Local Data) ] --(Encrypted Gradients)--> /

Key Results & Benchmarks
Medical Evaluation Metrics
Metric
Single-Node Baseline
MediVision Edge (Federated)
Accuracy
59.02%
73.77%
Sensitivity
0.00% (Collapsed)
67.00%
Specificity
100.00%
78.47%
AUC-ROC
N/A
0.7807
Note: Achieved 67% Sensitivity by implementing Youden's J Statistic to dynamically calculate the optimal clinical decision threshold (0.0558), overcoming severe Non-IID class imbalance.
Edge Optimization Results
Format
File Size
Inference Latency (CPU)
PyTorch (FP32)
~45 MB
-
ONNX (FP32)
42.61 MB
10.55 ms (94.83 FPS)
ONNX (INT8)
10.70 MB
27.23 ms (36.73 FPS)
Size reduced by 74.9%. (Note: INT8 latency is higher on standard x86 CPUs due to on-the-fly dequantization overhead; in production edge environments like Jetson Nano with TensorRT, this yields a 2-3x speedup).
🛠️ Technology Stack
AI/ML Core: PyTorch, MONAI, TorchVision
Federated Learning: Flower (flwr), FedProx
Edge Optimization: ONNX, ONNX Runtime, INT8 Quantization
Backend & Frontend: FastAPI, Streamlit
MLOps & Deployment: Docker, GitHub Actions
📂 Project Structure
text
1234567891011
🚀 Quick Start
1. Clone and Setup
bash
123456789
2. Run Federated Training
bash
1
3. Run Edge Benchmarking
bash
1
4. Launch the Dashboard
(Ensure the backend is running in a separate terminal)
bash
12
📜 Documentation
Architecture Design
Secure Aggregation Protocol
🔮 Future Enhancements
Real Edge Hardware: Deploy on an actual Raspberry Pi or Jetson Nano.
Differential Privacy: Add DP-SGD to provide formal privacy guarantees.
3D Medical Imaging: Extend to volumetric MRI/CT using 3D U-Net.
Heterogeneous Models: Allow different hospitals to use different model architectures.
Built with ❤️ by [Wasay] | Zero-Budget Architecture
