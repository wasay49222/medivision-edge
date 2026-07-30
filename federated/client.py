import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.data import PersistentDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord
from torchvision.models import resnet18
import pandas as pd
import flwr as fl
import numpy as np

# Configuration
HOSPITALS = ["hospital_a", "hospital_b", "hospital_c"]
DATA_BASE_DIR = "data"
CACHE_BASE_DIR = "data/cache"
BATCH_SIZE = 16
LOCAL_EPOCHS = 1  # Minimum epochs to prevent overfitting/exploding
LEARNING_RATE = 1e-4  # ULTRA-STABLE: 10x smaller than before
DEVICE = torch.device("cpu")
MU = 0.0  # DISABLED: Temporarily turn off FedProx to isolate the issue

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

class PneumoniaClient(fl.client.NumPyClient):
    def __init__(self, hospital_name):
        self.hospital_name = hospital_name
        self.model = self._load_model()
        self.dataloader = self._load_data()

    def _load_model(self):
        model = resnet18(weights=None, num_classes=2)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model.to(DEVICE)

    def _load_data(self):
        hospital_dir = os.path.join(DATA_BASE_DIR, self.hospital_name)
        manifest_path = os.path.join(hospital_dir, "manifest.csv")
        cache_dir = os.path.join(CACHE_BASE_DIR, self.hospital_name)
        os.makedirs(cache_dir, exist_ok=True)

        transforms = Compose([
            LoadImaged(keys=["image"], reader="PILReader"),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image"])
        ])

        manifest_df = pd.read_csv(manifest_path)
        data_dicts = [
            {"image": os.path.join(hospital_dir, "images", row["filename"]), "label": int(row["label"])}
            for _, row in manifest_df.iterrows()
        ]

        dataset = PersistentDataset(data=data_dicts, transform=transforms, cache_dir=cache_dir)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        global_params = [p.clone().detach() for p in self.model.parameters()]
        
        # ULTRA-STABLE: Mild 2x penalty for missing Pneumonia
        class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            for batch in self.dataloader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE, dtype=torch.long)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # FedProx is disabled (MU=0.0), so this term is 0
                proximal_term = 0.0
                if MU > 0:
                    for local_weights, global_weights in zip(self.model.parameters(), global_params):
                        proximal_term += (local_weights - global_weights).norm(2).pow(2)
                loss += (MU / 2) * proximal_term
                
                loss.backward()
                
                # Gradient clipping to absolutely prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
        updated_params = get_parameters(self.model)
        if "mask" in config:
            mask = config["mask"]
            updated_params = [p + m for p, m in zip(updated_params, mask)]
            
        return updated_params, len(self.dataloader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {"loss": 0.0}

def client_fn(cid: str) -> fl.client.Client:
    hospital_name = HOSPITALS[int(cid)]
    return PneumoniaClient(hospital_name).to_client()