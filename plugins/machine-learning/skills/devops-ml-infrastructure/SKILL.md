---
name: devops-ml-infrastructure
version: "1.0.7"
maturity: "5-Expert"
specialization: CI/CD & Infrastructure for ML Systems
description: DevOps for ML with GitHub Actions pipelines, Terraform IaC, Docker/Kubernetes, and cloud ML platforms (SageMaker, Azure ML, Vertex AI). Use when automating training, deploying models, or provisioning ML infrastructure.
---

# DevOps and ML Infrastructure

CI/CD pipelines, infrastructure automation, and deployment orchestration for production ML systems.

---

## Pipeline Selection

| Stage | Tool | Use Case |
|-------|------|----------|
| CI/CD | GitHub Actions | Workflow automation |
| IaC | Terraform | Cloud provisioning |
| Container | Docker | Model packaging |
| Orchestration | Kubernetes | Scalable serving |
| Registry | ECR/GCR/ACR | Image storage |
| MLOps | MLflow/W&B | Experiment tracking |

---

## GitHub Actions ML Pipeline

```yaml
name: ML Training Pipeline
on:
  push: { branches: [main] }
  schedule: [{ cron: '0 0 * * 0' }]

jobs:
  train-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install -r requirements.txt
      - run: python scripts/train.py --config configs/model.yaml
      - run: python scripts/evaluate.py --min-accuracy 0.85
      - uses: actions/upload-artifact@v3
        with: { name: model, path: models/ }

  build-container:
    needs: train-model
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ${{ secrets.ECR_REGISTRY }}/ml-model:${{ github.sha }}

  deploy:
    needs: build-container
    environment: production
    runs-on: ubuntu-latest
    steps:
      - run: kubectl set image deployment/ml-model ml-model=${{ secrets.ECR_REGISTRY }}/ml-model:${{ github.sha }}
```

---

## Terraform ML Infrastructure

```hcl
# EKS Cluster
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = "ml-cluster"
  cluster_version = "1.28"

  eks_managed_node_groups = {
    cpu_nodes = { instance_types = ["m5.2xlarge"], min_size = 2, max_size = 10 }
    gpu_nodes = { instance_types = ["p3.2xlarge"], min_size = 0, max_size = 5 }
  }
}

# S3 for artifacts
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "ml-artifacts-${var.environment}"
}

# ECR for images
resource "aws_ecr_repository" "ml_models" {
  name = "ml-models"
  image_scanning_configuration { scan_on_push = true }
}
```

---

## Dockerfile for ML

```dockerfile
FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY models/ ./models/
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Kubernetes ML Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: ml-model
        image: registry/ml-model:latest
        resources:
          requests: { memory: "2Gi", cpu: "1" }
          limits: { memory: "4Gi", cpu: "2" }
        livenessProbe:
          httpGet: { path: /health, port: 8000 }
        readinessProbe:
          httpGet: { path: /health, port: 8000 }
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource: { name: cpu, target: { averageUtilization: 70 } }
```

---

## Optimized Inference Service

```python
from fastapi import FastAPI
from dataclasses import dataclass
from collections import deque
import asyncio

@dataclass
class BatchConfig:
    max_batch_size: int = 32
    max_wait_time: float = 0.01

class BatchProcessor:
    def __init__(self, model, config: BatchConfig):
        self.model = model
        self.config = config
        self.queue = deque()

    async def add_request(self, data):
        future = asyncio.Future()
        self.queue.append((data, future))
        if not self.processing:
            asyncio.create_task(self._process_batch())
        return await future

    async def _process_batch(self):
        await asyncio.sleep(self.config.max_wait_time)
        batch = [self.queue.popleft() for _ in range(min(len(self.queue), self.config.max_batch_size))]
        outputs = self.model(torch.stack([x[0] for x in batch]))
        for i, (_, future) in enumerate(batch):
            future.set_result(outputs[i])
```

---

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

---

## Best Practices

| Area | Practice |
|------|----------|
| **CI/CD** | Automate data validation, performance gates, canary deploys |
| **IaC** | Module-based Terraform, state locking, lifecycle policies |
| **Deployment** | Batching, caching, torch.compile, connection pooling |
| **Monitoring** | Health checks, automated rollback, cost tracking |

---

## Commands Reference

```bash
# Terraform
terraform init && terraform plan -var-file="prod.tfvars"
terraform apply -auto-approve

# GitHub Actions
gh workflow run ml-pipeline.yml
gh run list --workflow=ml-pipeline.yml

# Kubernetes
kubectl rollout status deployment/ml-model
kubectl rollout undo deployment/ml-model
```

---

## Infrastructure Checklist

- [ ] CI/CD pipeline with data validation
- [ ] Model performance thresholds
- [ ] Containerized with multi-stage build
- [ ] Kubernetes deployment with autoscaling
- [ ] Health/readiness probes
- [ ] Terraform state management
- [ ] Canary deployment strategy
- [ ] Rollback procedures documented

---

**Version**: 1.0.5
