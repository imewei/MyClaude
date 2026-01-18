---
name: model-deployment-serving
version: "1.0.7"
maturity: "5-Expert"
specialization: Production ML Model Serving & Deployment
description: Deploy ML models with FastAPI, TorchServe, BentoML, Docker, Kubernetes, and cloud platforms. Implement monitoring, A/B testing, and drift detection. Use when building model serving APIs, containerizing models, or setting up production ML infrastructure.
---

# Model Deployment and Serving

Production ML model deployment from local serving to cloud-scale with monitoring, versioning, and drift detection.

---

## Framework Selection

| Framework | Use Case | Complexity |
|-----------|----------|------------|
| FastAPI | Custom REST APIs | Low |
| TorchServe | PyTorch models | Medium |
| BentoML | Multi-framework | Medium |
| KServe | Kubernetes-native | High |
| SageMaker | AWS managed | Medium |
| Vertex AI | GCP managed | Medium |

---

## FastAPI Model Serving

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# allow-torch
import torch

app = FastAPI(title="ML Model API")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    with torch.no_grad():
        x = torch.tensor(request.features).unsqueeze(0)
        output = model(x)
        return PredictionResponse(
            prediction=output.item(),
            confidence=torch.sigmoid(output).item()
        )
```

---

## Docker Containerization

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

## Kubernetes Deployment

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
        image: registry/ml-model:v1.0.0
        ports: [{containerPort: 8000}]
        resources:
          requests: {memory: "2Gi", cpu: "1"}
          limits: {memory: "4Gi", cpu: "2"}
        livenessProbe:
          httpGet: {path: /health, port: 8000}
          initialDelaySeconds: 30
        readinessProbe:
          httpGet: {path: /health, port: 8000}
          initialDelaySeconds: 10
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource: {name: cpu, target: {type: Utilization, averageUtilization: 70}}
```

---

## Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

prediction_counter = Counter('model_predictions_total', 'Total predictions', ['status'])
prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
async def predict(request: PredictionRequest):
    with prediction_latency.time():
        result = model.predict(request.features)
        prediction_counter.labels(status='success').inc()
        return result

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

---

## Drift Detection

```python
from scipy import stats
import numpy as np

class DriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        self.reference = reference_data
        self.threshold = threshold

    def detect(self, production_data):
        results = {}
        for i in range(production_data.shape[1]):
            ks_stat, p_value = stats.ks_2samp(
                self.reference[:, i], production_data[:, i]
            )
            results[f'feature_{i}'] = {
                'drift_detected': p_value < self.threshold,
                'p_value': p_value
            }
        return results
```

---

## A/B Testing

```python
from dataclasses import dataclass

@dataclass
class Variant:
    name: str
    model_version: str
    traffic_weight: float

class ABTest:
    def __init__(self, variants: list[Variant]):
        self.variants = variants

    def select_variant(self, user_id: str) -> Variant:
        hash_value = hash(user_id) % 10000 / 10000
        cumulative = 0
        for variant in self.variants:
            cumulative += variant.traffic_weight
            if hash_value < cumulative:
                return variant
        return self.variants[-1]
```

---

## Cloud Deployment

### AWS SageMaker
```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data="s3://bucket/model.tar.gz",
    role=role,
    framework_version="2.0",
    entry_point="inference.py"
)
predictor = model.deploy(
    initial_instance_count=2,
    instance_type="ml.m5.xlarge"
)
```

### GCP Vertex AI
```python
from google.cloud import aiplatform

model = aiplatform.Model.upload(
    display_name="ml-model",
    artifact_uri="gs://bucket/model/",
    serving_container_image_uri="gcr.io/project/image:latest"
)
endpoint = model.deploy(machine_type="n1-standard-4")
```

---

## Best Practices

| Area | Practice |
|------|----------|
| **Deployment** | Containerize, health checks, resource limits, graceful shutdown |
| **Monitoring** | Latency/throughput/errors, confidence distribution, drift detection |
| **Performance** | Batch predictions, caching, quantization, GPU acceleration |
| **Reliability** | Retries with backoff, circuit breakers, multiple versions, rollback |

---

## Common Commands

```bash
# Docker
docker build -t ml-model:latest .
docker run -p 8000:8000 ml-model:latest

# Kubernetes
kubectl apply -f deployment.yaml
kubectl scale deployment ml-model --replicas=5
kubectl rollout undo deployment/ml-model

# Testing
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [1,2,3]}'
```

---

## Deployment Checklist

- [ ] Model containerized with multi-stage build
- [ ] Health/readiness probes configured
- [ ] Resource limits set
- [ ] Prometheus metrics exposed
- [ ] Drift detection implemented
- [ ] A/B testing framework ready
- [ ] Autoscaling configured
- [ ] Rollback procedure documented

---

**Version**: 1.0.5
