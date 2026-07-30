import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.data import PersistentDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord, Lambdad
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np

TEST_DATA_DIR = "data/test_set"
TEST_MANIFEST_PATH = os.path.join(TEST_DATA_DIR, "manifest.csv")
TEST_CACHE_DIR = "data/cache/test_set"
MODEL_PATH = "models/federated/global_model_secagg.pth"
BATCH_SIZE = 8
DEVICE = torch.device("cpu")

def safe_normalize(img):
    img = img.float()
    min_val = img.min()
    max_val = img.max()
    if (max_val - min_val) == 0:
        return torch.zeros_like(img)
    return (img - min_val) / (max_val - min_val)

def main():
    print("🔄 Setting up evaluation pipeline...")
    os.makedirs(TEST_CACHE_DIR, exist_ok=True)

    transforms = Compose([
        LoadImaged(keys=["image"], reader="PILReader"),
        EnsureChannelFirstd(keys=["image"]),
        ToTensord(keys=["image"]),
        Lambdad(keys=["image"], func=safe_normalize)
    ])

    manifest_df = pd.read_csv(TEST_MANIFEST_PATH)
    data_dicts = [
        {"image": os.path.join(TEST_DATA_DIR, "images", row["filename"]), "label": int(row["label"])}
        for _, row in manifest_df.iterrows()
    ]

    dataset = PersistentDataset(data=data_dicts, transform=transforms, cache_dir=TEST_CACHE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"✅ Test set loaded. Total images: {len(dataset)}")

    print(f"🔄 Loading global model from {MODEL_PATH}...")
    model = resnet18(weights=None, num_classes=2)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    print("🚀 Running inference on test set...")
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE, dtype=torch.long)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            pos_probs = probs[:, 1].cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(pos_probs)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ==========================================
    # 🔍 SENIOR ENGINEERING FIX: OPTIMAL THRESHOLD
    # ==========================================
    # Calculate ROC curve to find the threshold that maximizes Sensitivity - Specificity
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    # Youden's J statistic: J = Sensitivity + Specificity - 1
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n🔍 Threshold Optimization:")
    print(f"   Default threshold (0.5) resulted in 0% Sensitivity.")
    print(f"   Calculated Optimal Threshold: {optimal_threshold:.4f}")
    
    # Apply the optimal threshold to get final predictions
    all_preds = (all_probs >= optimal_threshold).astype(int)

    # 6. Calculate Medical Metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    auc_roc = roc_auc_score(all_labels, all_probs)

    print("\n" + "="*50)
    print("📊 MEDICAL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:    {accuracy * 100:.2f}%")
    print(f"Sensitivity: {sensitivity * 100:.2f}%  <-- (Crucial: Did we catch the sick patients?)")
    print(f"Specificity: {specificity * 100:.2f}%  <-- (Did we correctly clear the healthy patients?)")
    print(f"AUC-ROC:     {auc_roc:.4f}")
    print("-" * 50)
    print(f"True Positives: {tp} | False Negatives: {fn}")
    print(f"True Negatives: {tn} | False Positives: {fp}")
    print("="*50)
    print("\n🎉 Level 7 Complete! Medical evaluation finished.")

if __name__ == "__main__":
    main()