---
name: federated-learning
description: "Build federated learning systems with Flower and PySyft including privacy-preserving training, differential privacy, secure aggregation, cross-silo/cross-device FL, and communication efficiency. Use when training models across distributed data sources, implementing privacy-preserving ML, or designing federated architectures."
---

# Federated Learning

## Expert Agent

For ML engineering, model training pipelines, and production ML systems, delegate to:

- **`ml-expert`**: Expert in classical ML, MLOps pipelines, and data engineering.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`

This skill covers Flower framework, FedAvg and variants, differential privacy, secure aggregation, and communication-efficient FL.

## Flower Framework

### Server Setup

```python
import flwr as fl
from flwr.server.strategy import FedAvg

strategy = FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.2,
    min_fit_clients=2,
    min_available_clients=3,
)
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

### Client Implementation

```python
import flwr as fl
import torch
from collections import OrderedDict

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_one_epoch(self.model, self.trainloader)
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate_model(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}
```

## FedAvg Variants

| Algorithm | Key Idea | Best For |
|-----------|----------|----------|
| FedAvg | Weighted average of client models | IID data |
| FedProx | Proximal term limits client drift | Non-IID data |
| FedBN | Keep BatchNorm local per client | Feature shift |
| SCAFFOLD | Variance reduction with control variates | Non-IID + many clients |

### FedProx: Add Proximal Term to Client Loss

```python
def train_fedprox(model, trainloader, global_params, mu=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    global_tensors = [torch.tensor(p) for p in global_params]

    for images, labels in trainloader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        prox = sum(((lp - gp) ** 2).sum() for lp, gp in zip(model.parameters(), global_tensors))
        (loss + (mu / 2) * prox).backward()
        optimizer.step()
```

## Differential Privacy with Opacus

```python
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=trainloader,
    epochs=10,
    target_epsilon=5.0,
    target_delta=1e-5,
    max_grad_norm=1.0,
)
```

| Privacy Level | Epsilon | Delta | Use Case |
|--------------|---------|-------|----------|
| Strong | 1.0 | 1e-5 | Medical, financial |
| Moderate | 5.0 | 1e-5 | Internal analytics |
| Weak | 10.0 | 1e-4 | Non-sensitive benchmarks |

## Communication Efficiency

```python
import numpy as np

def top_k_sparsify(gradients: np.ndarray, k_frac: float = 0.01) -> np.ndarray:
    """Keep only top-k percent of gradient values by magnitude."""
    flat = gradients.flatten()
    k = max(1, int(len(flat) * k_frac))
    top_idx = np.argpartition(np.abs(flat), -k)[-k:]
    sparse = np.zeros_like(flat)
    sparse[top_idx] = flat[top_idx]
    return sparse.reshape(gradients.shape)
```

## Non-IID Data Partitioning

```python
def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int = 42):
    """Partition indices using Dirichlet distribution (alpha controls skew)."""
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        proportions = rng.dirichlet([alpha] * num_clients)
        splits = (proportions * len(idx)).astype(int)
        splits[-1] = len(idx) - splits[:-1].sum()
        pos = 0
        for cid, count in enumerate(splits):
            client_indices[cid].extend(idx[pos:pos + count])
            pos += count
    return [np.array(ci) for ci in client_indices]
```

## Production Checklist

- [ ] Define topology: cross-silo (few reliable) vs. cross-device (many unreliable)
- [ ] Set minimum client participation threshold per round
- [ ] Implement client timeout and dropout handling
- [ ] Choose aggregation strategy based on data heterogeneity
- [ ] Configure differential privacy budget per compliance requirements
- [ ] Enable gradient compression for bandwidth-constrained clients
- [ ] Monitor per-client convergence to detect stragglers or poisoning
- [ ] Version model checkpoints per round for rollback capability
- [ ] Test with simulated non-IID partitions before deployment
- [ ] Log communication overhead, round time, and convergence metrics
