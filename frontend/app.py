import streamlit as st
import requests
import json
import os
from PIL import Image
import io

# Configuration
API_URL = "http://127.0.0.1:8000"
OPTIMAL_THRESHOLD = 0.0558

st.set_page_config(
    page_title="MediVision Edge Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header"> MediVision Edge</h1>', unsafe_allow_html=True)
st.markdown("### Privacy-Preserving Medical Imaging Pipeline")
st.markdown("**Federated Learning · Edge AI · Secure Aggregation**")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Live Demo", "📊 Model Metrics", " Edge Benchmarks"]
)

# Home Page
if page == "🏠 Home":
    st.header("Welcome to MediVision Edge")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Federated Hospitals", "3")
    with col2:
        st.metric("Model Accuracy", "73.77%")
    with col3:
        st.metric("Sensitivity", "67.00%")
    
    st.markdown("### 🎯 Project Overview")
    st.markdown("""
    MediVision Edge is a privacy-preserving medical imaging platform that enables multiple 
    hospitals to collaboratively train AI models **without sharing raw patient data**.
    
    #### Key Features:
    - 🔒 **Federated Learning**: Train across 3 hospital nodes without data leaving local environments
    - 🛡️ **Secure Aggregation**: Cryptographic protection against gradient inversion attacks
    - ⚡ **Edge Optimization**: 74.9% size reduction via ONNX quantization (42.61 MB → 10.70 MB)
    -  **Clinical Threshold**: Optimized threshold (0.0558) for maximum sensitivity
    
    #### Technology Stack:
    - **AI Framework**: PyTorch + MONAI
    - **Federated Learning**: Flower (flwr)
    - **Edge Optimization**: ONNX + INT8 Quantization
    - **Backend**: FastAPI
    - **Deployment**: Docker with Raspberry Pi simulation
    """)
    
    st.markdown("### 📈 Architecture Diagram")
    st.info("🏥 Hospital A (Local Training) → 🔐 Encrypted Gradients → 🌐 Central Server (FedAvg) →  Global Model → 📱 Edge Deployment")

# Live Demo Page
elif page == "🔍 Live Demo":
    st.header("🔍 Live Pneumonia Detection Demo")
    st.markdown("Upload a chest X-ray image to see the model's prediction in real-time.")
    
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded X-ray", width=500)
        
        with col2:
            if st.button("🔬 Analyze Image", type="primary"):
                with st.spinner("Running inference..."):
                    # Send to API
                    try:
                        response = requests.post(
                            f"{API_URL}/predict",
                            files={"file": uploaded_file.getvalue()}
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.success("✅ Analysis Complete!")
                            
                            # Display metrics
                            st.metric("Prediction", result["prediction"])
                            
                            # Confidence bars
                            st.markdown("#### Confidence Scores:")
                            st.progress(result["confidence_pneumonia"])
                            st.write(f"**Pneumonia:** {result['confidence_pneumonia']*100:.2f}%")
                            
                            st.progress(result["confidence_normal"])
                            st.write(f"**Normal:** {result['confidence_normal']*100:.2f}%")
                            
                            st.write(f"**Threshold Used:** {result['threshold_used']}")
                            st.write(f"**Message:** {result['message']}")
                            
                            # Clinical interpretation
                            if result["prediction"] == "Pneumonia":
                                st.warning("⚠️ **Clinical Recommendation:** Review by radiologist recommended")
                            else:
                                st.success("✅ **Clinical Recommendation:** No signs of pneumonia detected")
                            
                        else:
                            st.error(f"API Error: {response.status_code}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to API. Make sure the backend is running on port 8000.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# Model Metrics Page
# Model Metrics Page
elif page == "📊 Model Metrics":
    st.header("📊 Model Evaluation Metrics")
    
    st.markdown("### 🎯 Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Federated Model (Our Approach)**")
        st.metric("Accuracy", "73.77%")
        st.metric("Sensitivity", "67.00%")
        st.metric("Specificity", "78.47%")
        st.metric("AUC-ROC", "0.7807")
    
    with col2:
        st.warning("**Single-Node Baseline**")
        st.metric("Accuracy", "59.02%")
        st.metric("Sensitivity", "0% (Collapsed)")
        st.metric("Specificity", "100%")
        st.metric("Note", "Model predicted only 'Normal'")
    
    st.markdown("---")
    
    st.markdown("###  Confusion Matrix Results")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True Positives", "67")
    col2.metric("False Negatives", "33")
    col3.metric("True Negatives", "113")
    col4.metric("False Positives", "31")
    
    st.markdown("---")
    
    st.success(f"""
    **Optimal Threshold: {OPTIMAL_THRESHOLD}**
    
    Using Youden's J Statistic, we calculated the optimal threshold to maximize 
    clinical sensitivity while maintaining acceptable specificity.
    """)

# Edge Benchmarks Page
elif page == " Edge Benchmarks":
    st.header(" Edge Optimization Results")
    
    st.markdown("### 📦 Model Size Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PyTorch (.pth)", "~45 MB")
    
    with col2:
        st.metric("ONNX (FP32)", "42.61 MB")
    
    with col3:
        st.metric("ONNX (INT8)", "10.70 MB")
        st.success("✅ 74.9% reduction!")
    
    st.markdown("---")
    
    st.markdown("### ️ Inference Latency")
    
    st.json({
        "FP32 Model": {
            "Latency": "10.55 ms",
            "FPS": "94.83"
        },
        "INT8 Quantized": {
            "Latency": "27.23 ms", 
            "FPS": "36.73",
            "Note": "Slower due to CPU dequantization overhead"
        }
    })
    
    st.warning("""
    **Note on INT8 Performance:**
    
    On this CPU, INT8 quantization actually increased latency due to on-the-fly dequantization.
    In production edge environments (Intel NUC, Jetson Nano), using optimized execution providers 
    (OpenVINO, TensorRT) would achieve true 2-3x speedups.
    """)
    
    st.markdown("### 🐳 Docker Edge Simulation")
    st.code("""
    Resources:
    - CPU: 2.0 cores
    - Memory: 2 GB RAM
    - Simulates: Raspberry Pi 4 constraints
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>MediVision Edge v1.0.0 | Built with PyTorch, MONAI, Flower, ONNX, FastAPI & Streamlit</p>
    <p>Privacy-Preserving Medical Imaging Pipeline</p>
</div>
""", unsafe_allow_html=True)