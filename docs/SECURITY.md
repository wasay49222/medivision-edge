# Security and Privacy Considerations

## Threat Model
1. **Data Leakage:** Raw patient X-rays never leave the local hospital node. Only 32-bit floating-point weight tensors are transmitted.
2. **Gradient Inversion Attacks:** A malicious central server could theoretically attempt to reconstruct input images from the gradients. 
   * **Mitigation:** Secure Aggregation (Additive Masking) ensures the server only ever sees the sum of the masked gradients, making isolation of a single client's update mathematically impossible.
3. **Model Poisoning:** A malicious client could send bad weights to corrupt the global model.
   * **Mitigation:** FedProx proximal term restricts how far a local model can drift from the global model, naturally dampening the effect of poisoned updates.

## Compliance
* **HIPAA (US):** Compliant by design. No Protected Health Information (PHI) is transmitted or stored centrally.
* **GDPR (EU):** Compliant. Data minimization and purpose limitation are enforced by the federated architecture.