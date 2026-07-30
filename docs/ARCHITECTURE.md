# MediVision Edge Architecture

## 1. System Overview
MediVision Edge is a distributed machine learning system designed to train medical imaging models across isolated data silos (hospitals) without centralizing raw patient data. The system is composed of three main layers: the Federated Training Layer, the Edge Optimization Layer, and the Application Layer.

## 2. Component Breakdown

### 2.1 Federated Training Layer (The Core)
* **Client Nodes (Hospitals):** Each node runs a local PyTorch training loop using MONAI for medical image preprocessing. Data never leaves this node.
* **Orchestrator (Flower Server):** Manages the federated rounds. It uses a custom weighted-averaging strategy (FedAvg/FedProx) to handle Non-IID data distributions across hospitals.
* **Security Layer (Secure Aggregation):** Implements additive zero-sum masking. Before a client sends its weights, it adds a random cryptographic mask. The server generates masks such that the sum of all masks equals zero, preventing gradient inversion attacks.

### 2.2 Edge Optimization Layer
* **ONNX Export:** The aggregated global PyTorch model is exported to the Open Neural Network Exchange (ONNX) format to decouple it from the PyTorch runtime.
* **INT8 Quantization:** The FP32 ONNX model is dynamically quantized to INT8. This reduces the model size by ~75% and optimizes it for low-power CPU inference.

### 2.3 Application Layer
* **FastAPI Backend:** Serves the optimized ONNX model via a REST API. It handles image preprocessing, inference, and applies the clinically optimized decision threshold (Youden's J Statistic).
* **Streamlit Frontend:** A web dashboard that visualizes training metrics, edge benchmarks, and provides a live demo interface for uploading X-rays.
* **Docker Edge Simulation:** The entire backend and model are containerized. Resource constraints (2 CPU cores, 2GB RAM) are applied via Docker Compose to simulate Raspberry Pi 4 hardware limits.

## 3. Data Flow
1. Server initializes global weights $W_g$.
2. Server sends $W_g$ to Clients A, B, and C.
3. Clients train locally for $E$ epochs using FedProx, producing local weights $W_{local}$.
4. Clients apply cryptographic mask $M_i$ and send $W_{local} + M_i$ to the server.
5. Server aggregates: $\sum (W_{local} + M_i) = \sum W_{local} + \sum M_i = \sum W_{local} + 0$.
6. Server updates $W_g$ and repeats the cycle.
7. Final $W_g$ is quantized and deployed to the FastAPI/Streamlit stack.