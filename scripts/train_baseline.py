import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.data import PersistentDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord
)
from torchvision.models import resnet18
import pandas as pd

# Configuration
HOSPITAL_DATA_DIR = "data/hospital_a"
MANIFEST_PATH = os.path.join(HOSPITAL_DATA_DIR, "manifest.csv")
CACHE_DIR = "data/cache/hospital_a"
BATCH_SIZE = 8
EPOCHS = 5  # Keep it low for fast CPU baseline testing
LEARNING_RATE = 1e-3
DEVICE = torch.device("cpu") # Force CPU usage

def main():
    print("🔄 Setting up MONAI PersistentDataset pipeline...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Define MONAI Transforms (Dictionary-based, note the 'd' at the end)
    transforms = Compose([
        LoadImaged(keys=["image"], reader="PILReader"),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]), # Normalizes to [0, 1]
        ToTensord(keys=["image"])
    ])

    # 2. Load Manifest
    manifest_df = pd.read_csv(MANIFEST_PATH)
    # Create list of dictionaries as required by MONAI Dataset
    data_dicts = [
        {
            "image": os.path.join(HOSPITAL_DATA_DIR, "images", row["filename"]), 
            "label": int(row["label"])
        }
        for _, row in manifest_df.iterrows()
    ]

    # 3. Create PersistentDataset
    # This caches the transformed tensors to disk, saving RAM
    dataset = PersistentDataset(data=data_dicts, transform=transforms, cache_dir=CACHE_DIR)
    
    # 4. Create DataLoader (num_workers=0 is critical for Windows 8GB RAM stability)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"✅ Dataset ready. Total images: {len(dataset)}")

    # 5. Initialize Model, Loss, and Optimizer
    print("🔄 Initializing ResNet-18 for CPU (1-channel grayscale)...")
    model = resnet18(weights=None, num_classes=2) # 2 classes: Normal (0), Pneumonia (1)
    
    # SENIOR ENGINEERING TWEAK: 
    # Replace the first conv layer to accept 1-channel (grayscale) X-rays 
    # instead of the default 3-channel (RGB). This saves compute and is standard in medical AI.
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    print(f"🚀 Starting training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            # Extract image and label from the dictionary batch
            images = batch["image"].to(DEVICE)
            # PyTorch DataLoader automatically collates the 'label' list into a tensor. 
            # We explicitly cast to torch.long for CrossEntropyLoss.
            labels = batch["label"].to(DEVICE, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        print(f"   Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # 7. Save the Baseline Model
    model_path = "models/resnet/baseline_hospital_a.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Baseline model saved to {model_path}")
    print("🎉 Level 3 Complete! You now have a single-node baseline.")

if __name__ == "__main__":
    main()