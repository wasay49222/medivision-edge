import numpy as np
import torch
import os
from torchvision.models import resnet18
import torch.nn as nn
from client import PneumoniaClient, get_parameters, set_parameters, HOSPITALS

def get_initial_parameters():
    """Returns the initial global model parameters as NumPy arrays."""
    model = resnet18(weights=None, num_classes=2)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def aggregate_weights(client_weights, client_sizes):
    """Standard Federated Averaging (FedAvg) math for the server."""
    total_size = sum(client_sizes)
    aggregated = []
    
    for i in range(len(client_weights[0])):
        layer_sum = np.zeros_like(client_weights[0][i])
        for weights, size in zip(client_weights, client_sizes):
            layer_sum += weights[i] * size
        aggregated.append(layer_sum / total_size)
        
    return aggregated

def main():
    print("🚀 Starting Level 6: Federated Learning with FedProx (SecAgg Disabled for Debugging)...")
    
    global_params = get_initial_parameters()
    num_rounds = 2
    
    print("\n Initializing Hospital Clients...")
    clients = [PneumoniaClient(hosp) for hosp in HOSPITALS]
    print("✅ Clients initialized.")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Global Round {round_num}/{num_rounds} ---")
        client_weights = []
        client_sizes = []
        
        for idx, client in enumerate(clients):
            print(f"   Training {HOSPITALS[idx]}...")
            # SECAGG DISABLED: Passing empty config {} so no mask is applied
            params, num_examples, _ = client.fit(global_params, {})
            client_weights.append(params)
            client_sizes.append(num_examples)
            
        print("   Aggregating global weights...")
        global_params = aggregate_weights(client_weights, client_sizes)
        print("✅ Global model updated.")

    model_path = "models/federated/global_model_secagg.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model = resnet18(weights=None, num_classes=2)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    params_dict = zip(model.state_dict().keys(), global_params)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    torch.save(model.state_dict(), model_path)
    
    print(f"\n🎉 Level 6 Complete! Final model saved to {model_path}")

if __name__ == "__main__":
    main()