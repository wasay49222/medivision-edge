import io
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import torch.nn.functional as F

app = FastAPI(
    title="MediVision Edge API",
    description="Privacy-preserving medical imaging inference endpoint",
    version="1.0.0"
)

# Configuration: Use absolute path to prevent "file not found" errors
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "federated", "global_model.onnx")
OPTIMAL_THRESHOLD = 0.0558 # Discovered in Level 7
TARGET_SIZE = (224, 224)

# Global variable to hold the ONNX session
onnx_session = None

@app.on_event("startup")
async def load_model():
    """Load the ONNX model into memory once at startup."""
    global onnx_session
    print(f"🔄 Loading ONNX model from {MODEL_PATH}...")
    # Use CPU provider. In production edge, this would be 'OpenVINOExecutionProvider'
    onnx_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("✅ Model loaded successfully!")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Replicates the exact MONAI training transforms:
    1. Convert to Grayscale (1 channel)
    2. Resize to 224x224
    3. Convert to float32 and normalize to [0, 1]
    4. Add batch dimension (1, 1, 224, 224)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L") # 'L' mode = 1-channel grayscale
    image = image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    
    img_array = np.array(image, dtype=np.float32)
    
    # Safe normalization (prevents division by zero)
    min_val = img_array.min()
    max_val = img_array.max()
    if (max_val - min_val) > 0:
        img_array = (img_array - min_val) / (max_val - min_val)
    else:
        img_array = np.zeros_like(img_array)
        
    # Add batch and channel dimensions: (224, 224) -> (1, 1, 224, 224)
    img_array = np.expand_dims(img_array, axis=(0, 1))
    return img_array

@app.post("/predict", summary="Predict pneumonia from chest X-ray")
async def predict(file: UploadFile = File(...)):
    # SENIOR FIX: Robustly handle content_type variations from frontend uploads
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    
    # Allow standard image types, octet-stream (common in browser uploads), or check extension
    is_valid_type = content_type.startswith("image/") or content_type == "application/octet-stream" or content_type == ""
    is_valid_ext = filename.endswith(('.png', '.jpg', '.jpeg'))
    
    if not (is_valid_type or is_valid_ext):
        print(f"DEBUG: Rejected upload. Content-Type: '{content_type}', Filename: '{filename}'")
        raise HTTPException(status_code=400, detail="File must be a valid image (PNG, JPG, JPEG).")
    
    # 1. Read and preprocess the image
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    
    # 2. Run inference
    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: input_tensor})
    
    # 3. Post-process outputs
    logits = outputs[0][0] # Shape: (2,)
    probs = F.softmax(torch.tensor(logits), dim=0).numpy()
    
    prob_pneumonia = float(probs[1])
    prob_normal = float(probs[0])
    
    # 4. Apply the optimal clinical threshold
    is_pneumonia = prob_pneumonia >= OPTIMAL_THRESHOLD
    
    return {
        "filename": file.filename,
        "prediction": "Pneumonia" if is_pneumonia else "Normal",
        "confidence_pneumonia": round(prob_pneumonia, 4),
        "confidence_normal": round(prob_normal, 4),
        "threshold_used": OPTIMAL_THRESHOLD,
        "message": "Review recommended by radiologist" if is_pneumonia else "No signs of pneumonia detected."
    }

@app.get("/health", summary="Health check endpoint")
async def health_check():
    return {"status": "healthy", "model_loaded": onnx_session is not None}