---
name: devops-ml-infrastructure
description: DevOps for ML with GitHub Actions pipelines, Terraform IaC, Docker/Kubernetes, and cloud ML platforms (SageMaker, Azure ML, Vertex AI). Use when automating training, deploying models, or provisioning ML infrastructure.
---

# DevOps and ML Infrastructure

ML-specific infrastructure patterns for training, deployment, and experiment tracking.

## Expert Agent

- **`ml-expert`**: Unified specialist for MLOps and Infrastructure.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`

For general CI/CD, Terraform IaC, and Kubernetes patterns, see `dev-suite` skills.

## ML Platform Selection

| Platform | Use Case |
|----------|----------|
| SageMaker | AWS-native training & serving |
| Vertex AI | GCP-native ML pipelines |
| MLflow | Experiment tracking & registry |
| W&B | Experiment dashboards |

## Cloud ML Platforms

### AWS SageMaker
```python
from sagemaker.pytorch import PyTorchModel
model = PyTorchModel(model_data="s3://bucket/model.tar.gz", role=role)
predictor = model.deploy(instance_count=2, instance_type="ml.m5.xlarge")
```

### GCP Vertex AI
```python
from google.cloud import aiplatform
model = aiplatform.Model.upload(artifact_uri="gs://bucket/model/")
endpoint = model.deploy(machine_type="n1-standard-4")
```

## Experiment Tracking

```python
import mlflow

mlflow.set_experiment("model-v2")
with mlflow.start_run():
    mlflow.log_params({"lr": 0.001, "epochs": 50})
    mlflow.log_metrics({"accuracy": 0.92, "loss": 0.08})
    mlflow.pytorch.log_model(model, "model")
```

## Training Pipeline Pattern

```python
# Data validation → Train → Evaluate → Register → Deploy
stages = {
    "validate": "python scripts/validate_data.py --schema configs/schema.yaml",
    "train":    "python scripts/train.py --config configs/model.yaml",
    "evaluate": "python scripts/evaluate.py --min-accuracy 0.85",
    "register": "python scripts/register_model.py --registry mlflow",
}
```

## Inference Optimization

| Technique | Impact |
|-----------|--------|
| Batching | Throughput: 5-10x |
| torch.compile | Latency: 2-3x |
| Quantization | Memory: 2-4x reduction |
| Connection pooling | Concurrency improvement |

## ML Best Practices

| Area | Practice |
|------|----------|
| **Data** | Validate schema, detect drift, version datasets |
| **Training** | Performance gates, reproducible seeds, checkpointing |
| **Serving** | Health probes, canary deploys, automated rollback |
| **Tracking** | Log all hyperparams, metrics, and artifacts |
