---
name: model-deployment-serving
description: Deploy and serve ML models in production using FastAPI, TorchServe, BentoML, Docker, Kubernetes, and cloud platforms (AWS SageMaker, GCP Vertex AI, Azure ML) with monitoring, A/B testing, and drift detection. Use when writing or editing FastAPI serving code (`.py`), Dockerfiles, Kubernetes manifests (`.yaml`), TorchServe handlers, or cloud deployment scripts. Apply this skill when building FastAPI endpoints for model serving, implementing TorchServe custom handlers, containerizing models with Docker, deploying to Kubernetes with KServe or Seldon, deploying to AWS SageMaker endpoints, deploying to GCP Vertex AI or Azure ML, implementing batch prediction APIs, adding Prometheus metrics for model monitoring, implementing data and prediction drift detection, building A/B testing frameworks for model comparison, managing model versions and registries, implementing canary or blue-green deployments, optimizing inference latency and throughput, or setting up autoscaling for model serving infrastructure.
---

# Model Deployment and Serving

Expert guidance on deploying and serving ML models in production environments. Use when building model serving APIs, containerizing models, implementing cloud deployments, or setting up monitoring for production ML systems.

## When to Use This Skill

- Writing or editing FastAPI serving code (`.py`) for model inference APIs
- Writing or editing TorchServe custom handlers for PyTorch model serving
- Writing or editing BentoML service definitions for multi-framework serving
- Creating or modifying Dockerfiles for ML model containerization
- Writing Kubernetes deployment manifests (`.yaml`) for model serving
- Building REST API endpoints with FastAPI for real-time predictions
- Implementing batch prediction APIs for offline inference
- Creating TorchServe model archives (MAR files) with custom preprocessing
- Containerizing ML models and dependencies with Docker multi-stage builds
- Deploying models to Kubernetes with KServe, Seldon Core, or custom deployments
- Deploying to AWS SageMaker endpoints with auto-scaling
- Deploying to GCP Vertex AI prediction services
- Deploying to Azure ML managed endpoints
- Implementing Prometheus metrics for model performance monitoring
- Adding health check and readiness probe endpoints
- Implementing data drift detection with statistical tests (KS test, PSI)
- Building A/B testing frameworks for comparing model versions
- Managing model versions and promoting models through registries (MLflow, custom)
- Implementing deployment strategies (canary, blue-green, shadow mode)
- Optimizing inference latency with batching, caching, or quantization
- Setting up horizontal pod autoscaling (HPA) for Kubernetes deployments

## Overview

This skill covers end-to-end model deployment strategies, from local serving to cloud-scale production deployments with monitoring, versioning, and drift detection.

## Core Topics

### 1. Model Serving Frameworks

#### FastAPI for Model Serving

**Complete ML API with FastAPI**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: List[float] = Field(
        ...,
        min_items=10,
        max_items=10,
        description="Input features (must be length 10)"
    )
    model_version: Optional[str] = Field(
        default="latest",
        description="Model version to use"
    )

    @validator('features')
    def validate_features(cls, v):
        """Validate feature values."""
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features contain NaN or Inf values")
        return v

class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    prediction: float
    probability: Optional[float] = None
    model_version: str
    confidence: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="Production ML model serving API",
    version="1.0.0"
)

# Global model instance
class ModelService:
    """Singleton model service."""

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.version: str = "unknown"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, path: str) -> None:
        """Load model from checkpoint."""
        logger.info(f"Loading model from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        # Reconstruct model (assumes model architecture is known)
        self.model = YourModelClass(**checkpoint['config'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.version = checkpoint.get('version', 'unknown')
        logger.info(f"Model loaded successfully (version: {self.version})")

    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Generate prediction."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Prepare input
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(x)
            prediction = output.item()

            # For binary classification
            probability = torch.sigmoid(output).item()
            confidence = max(probability, 1 - probability)

        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'model_version': self.version
        }

# Initialize service
model_service = ModelService()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = Path("models/production_model.pth")
    if model_path.exists():
        model_service.load_model(str(model_path))
    else:
        logger.warning("No model found at startup")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_service.model is not None else "unhealthy",
        model_loaded=model_service.model is not None,
        model_version=model_service.version
    )

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate prediction for input features.

    Args:
        request: Prediction request with features

    Returns:
        Prediction response with result and metadata

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    try:
        result = model_service.predict(request.features)
        return PredictionResponse(**result)
    except RuntimeError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=503, detail="Model not available")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """
    Generate predictions for multiple inputs.

    Args:
        requests: List of prediction requests

    Returns:
        List of predictions
    """
    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds limit of 100"
        )

    results = []
    for req in requests:
        try:
            result = model_service.predict(req.features)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            results.append({'error': str(e)})

    return results

# Model reload endpoint
@app.post("/model/reload")
async def reload_model(path: str):
    """
    Reload model from specified path.

    Args:
        path: Path to model checkpoint

    Returns:
        Success message
    """
    try:
        model_service.load_model(path)
        return {"message": "Model reloaded successfully", "version": model_service.version}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Testing the API**
```python
import requests
import json

# Test health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Test prediction
prediction_request = {
    "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "model_version": "latest"
}
response = requests.post(
    "http://localhost:8000/predict",
    json=prediction_request
)
print(response.json())

# Test batch prediction
batch_request = [
    {"features": [1.0] * 10},
    {"features": [2.0] * 10},
    {"features": [3.0] * 10}
]
response = requests.post(
    "http://localhost:8000/predict/batch",
    json=batch_request
)
print(response.json())
```

#### TorchServe for PyTorch Models

**Custom Handler**
```python
# custom_handler.py
from ts.torch_handler.base_handler import BaseHandler
import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CustomModelHandler(BaseHandler):
    """
    Custom TorchServe handler for ML model.
    """

    def initialize(self, context):
        """
        Initialize model for serving.

        Args:
            context: TorchServe context with model artifacts
        """
        # Call parent initialization
        super().initialize(context)

        # Load custom preprocessing config
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Custom initialization
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model initialized successfully")

    def preprocess(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Preprocess input data.

        Args:
            data: List of input dictionaries

        Returns:
            Preprocessed tensor
        """
        # Extract features from request
        features_list = []
        for item in data:
            body = item.get("data") or item.get("body")
            if isinstance(body, (bytes, bytearray)):
                body = body.decode('utf-8')

            # Parse JSON
            import json
            features = json.loads(body)['features']
            features_list.append(features)

        # Convert to tensor
        tensor = torch.tensor(features_list, dtype=torch.float32)
        return tensor.to(self.device)

    def inference(self, data: torch.Tensor) -> torch.Tensor:
        """
        Run inference.

        Args:
            data: Preprocessed input tensor

        Returns:
            Model predictions
        """
        with torch.no_grad():
            predictions = self.model(data)

        return predictions

    def postprocess(self, data: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Postprocess predictions.

        Args:
            data: Model predictions

        Returns:
            List of prediction dictionaries
        """
        # Apply softmax for probabilities
        probabilities = F.softmax(data, dim=1)

        # Convert to numpy
        predictions = data.argmax(dim=1).cpu().numpy()
        probs = probabilities.cpu().numpy()

        # Format response
        results = []
        for pred, prob in zip(predictions, probs):
            results.append({
                'prediction': int(pred),
                'probabilities': prob.tolist(),
                'confidence': float(prob.max())
            })

        return results

# Package model
# torch-model-archiver --model-name my_model \
#   --version 1.0 \
#   --serialized-file model.pth \
#   --handler custom_handler.py \
#   --export-path model_store/

# Start TorchServe
# torchserve --start --model-store model_store --models my_model=my_model.mar
```

#### BentoML for Multi-Framework Serving

```python
import bentoml
from bentoml.io import JSON, NumpyNdarray
import numpy as np
import torch

# Save model to BentoML
model = YourModel()
bentoml.pytorch.save_model("my_model", model)

# Create service
@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10}
)
class MyModelService:
    """BentoML service for model serving."""

    model_ref = bentoml.models.get("my_model:latest")

    def __init__(self):
        self.model = bentoml.pytorch.load_model(self.model_ref)
        self.model.eval()

    @bentoml.api(
        input=NumpyNdarray(shape=(-1, 10), dtype=np.float32),
        output=JSON()
    )
    def predict(self, input_array: np.ndarray) -> dict:
        """Generate predictions."""
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_array)
            predictions = self.model(input_tensor)

        return {
            'predictions': predictions.numpy().tolist(),
            'model_version': str(self.model_ref.tag)
        }

    @bentoml.api(
        input=JSON(),
        output=JSON()
    )
    def predict_json(self, input_data: dict) -> dict:
        """Generate predictions from JSON input."""
        features = np.array(input_data['features'], dtype=np.float32)
        return self.predict(features.reshape(1, -1))

# Build Bento
# bentoml build

# Serve locally
# bentoml serve service:MyModelService

# Containerize
# bentoml containerize my_model_service:latest
```

### 2. Containerization and Orchestration

#### Docker for ML Models

**Dockerfile for PyTorch Model**
```dockerfile
# Multi-stage build for smaller image
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY main.py .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**
```yaml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/production_model.pth
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/models:ro
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
```

**Build and Run**
```bash
# Build image
docker build -t ml-model:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/models:ro \
  -e MODEL_PATH=/models/production_model.pth \
  ml-model:latest

# Or use docker-compose
docker-compose up -d

# View logs
docker-compose logs -f ml-api

# Scale service
docker-compose up -d --scale ml-api=3
```

#### Kubernetes Deployment

**Kubernetes Manifests**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model
        image: your-registry/ml-model:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/production_model.pth"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: model-volume
          mountPath: /models
          readOnly: true
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

---
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
```

**Deploy to Kubernetes**
```bash
# Apply manifests
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml
kubectl apply -f pvc.yaml

# Check status
kubectl get deployments
kubectl get pods
kubectl get services
kubectl get hpa

# View logs
kubectl logs -f deployment/ml-model-deployment

# Port forward for testing
kubectl port-forward service/ml-model-service 8000:80

# Update deployment (rolling update)
kubectl set image deployment/ml-model-deployment \
  ml-model=your-registry/ml-model:v1.1.0

# Rollback if needed
kubectl rollout undo deployment/ml-model-deployment
```

### 3. Cloud Platform Deployment

#### AWS SageMaker

**Deploy PyTorch Model to SageMaker**
```python
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::ACCOUNT_ID:role/SageMakerRole"

# Package model
model = PyTorchModel(
    model_data="s3://bucket/model.tar.gz",
    role=role,
    framework_version="2.0",
    py_version="py310",
    entry_point="inference.py",
    source_dir="code/",
    sagemaker_session=sagemaker_session
)

# Deploy model
predictor = model.deploy(
    initial_instance_count=2,
    instance_type="ml.m5.xlarge",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
    endpoint_name="ml-model-endpoint"
)

# Make predictions
response = predictor.predict({
    'features': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
})
print(response)

# Auto-scaling configuration
client = boto3.client('application-autoscaling')

# Register scalable target
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{predictor.endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)

# Configure scaling policy
client.put_scaling_policy(
    PolicyName='target-tracking-policy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{predictor.endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60
    }
)

# Clean up
# predictor.delete_endpoint()
```

**SageMaker Inference Script**
```python
# inference.py
import torch
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def model_fn(model_dir: str) -> torch.nn.Module:
    """Load model from directory."""
    model_path = f"{model_dir}/model.pth"
    model = YourModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def input_fn(request_body: str, content_type: str = 'application/json') -> torch.Tensor:
    """Parse input data."""
    if content_type == 'application/json':
        data = json.loads(request_body)
        features = torch.tensor(data['features'], dtype=torch.float32)
        return features.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(data: torch.Tensor, model: torch.nn.Module) -> Dict[str, Any]:
    """Generate predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)

    with torch.no_grad():
        output = model(data)
        prediction = output.item()
        probability = torch.sigmoid(output).item()

    return {
        'prediction': prediction,
        'probability': probability,
        'confidence': max(probability, 1 - probability)
    }

def output_fn(prediction: Dict[str, Any], accept: str = 'application/json') -> str:
    """Format output."""
    if accept == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
```

#### Google Cloud Vertex AI

```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project="your-project", location="us-central1")

# Upload model
model = aiplatform.Model.upload(
    display_name="ml-model",
    artifact_uri="gs://bucket/model/",
    serving_container_image_uri="gcr.io/your-project/serving-image:latest"
)

# Deploy to endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=5,
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1
)

# Make predictions
prediction = endpoint.predict(instances=[{
    'features': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}])
print(prediction)
```

### 4. Model Monitoring and Observability

#### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI
from fastapi.responses import Response
import time

app = FastAPI()

# Define metrics
prediction_counter = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_version', 'status']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_version']
)

model_confidence = Histogram(
    'model_confidence',
    'Model confidence distribution',
    ['model_version'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

active_requests = Gauge(
    'model_active_requests',
    'Number of active prediction requests'
)

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Generate prediction with metrics."""
    active_requests.inc()

    try:
        start_time = time.time()

        # Generate prediction
        result = model_service.predict(request.features)

        # Record metrics
        latency = time.time() - start_time
        prediction_latency.labels(
            model_version=result['model_version']
        ).observe(latency)

        model_confidence.labels(
            model_version=result['model_version']
        ).observe(result['confidence'])

        prediction_counter.labels(
            model_version=result['model_version'],
            status='success'
        ).inc()

        return PredictionResponse(**result)

    except Exception as e:
        prediction_counter.labels(
            model_version='unknown',
            status='error'
        ).inc()
        raise

    finally:
        active_requests.dec()

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

**Prometheus Configuration**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml-model'
    static_configs:
      - targets: ['ml-api:8000']
    metrics_path: '/metrics'
```

#### Drift Detection

```python
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    """Detect data and prediction drift in production."""

    def __init__(
        self,
        reference_data: np.ndarray,
        reference_predictions: np.ndarray,
        threshold: float = 0.05
    ):
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.threshold = threshold

        # Compute reference statistics
        self.ref_mean = reference_data.mean(axis=0)
        self.ref_std = reference_data.std(axis=0)
        self.ref_pred_dist = reference_predictions

    def detect_data_drift(
        self,
        production_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect drift in input data using KS test.

        Args:
            production_data: Recent production data

        Returns:
            Dictionary with drift results per feature
        """
        drift_results = {}

        for feature_idx in range(production_data.shape[1]):
            ref_feature = self.reference_data[:, feature_idx]
            prod_feature = production_data[:, feature_idx]

            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(ref_feature, prod_feature)

            drift_detected = p_value < self.threshold

            drift_results[f'feature_{feature_idx}'] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'drift_detected': drift_detected,
                'mean_shift': float(prod_feature.mean() - ref_feature.mean()),
                'std_shift': float(prod_feature.std() - ref_feature.std())
            }

            if drift_detected:
                logger.warning(
                    f"Drift detected in feature {feature_idx}: "
                    f"p-value={p_value:.4f}"
                )

        return drift_results

    def detect_prediction_drift(
        self,
        production_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect drift in model predictions.

        Args:
            production_predictions: Recent production predictions

        Returns:
            Dictionary with prediction drift results
        """
        # KS test for prediction distribution
        ks_statistic, p_value = stats.ks_2samp(
            self.ref_pred_dist,
            production_predictions
        )

        drift_detected = p_value < self.threshold

        # Compare distribution statistics
        ref_mean = self.ref_pred_dist.mean()
        prod_mean = production_predictions.mean()
        mean_shift = prod_mean - ref_mean

        result = {
            'ks_statistic': float(ks_statistic),
            'p_value': float(p_value),
            'drift_detected': drift_detected,
            'reference_mean': float(ref_mean),
            'production_mean': float(prod_mean),
            'mean_shift': float(mean_shift)
        }

        if drift_detected:
            logger.warning(
                f"Prediction drift detected: p-value={p_value:.4f}, "
                f"mean_shift={mean_shift:.4f}"
            )

        return result

    def monitor(
        self,
        production_data: np.ndarray,
        production_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run full drift monitoring.

        Args:
            production_data: Recent production data
            production_predictions: Recent production predictions

        Returns:
            Complete monitoring report
        """
        data_drift = self.detect_data_drift(production_data)
        pred_drift = self.detect_prediction_drift(production_predictions)

        # Count features with drift
        n_drifted_features = sum(
            1 for result in data_drift.values()
            if result['drift_detected']
        )

        report = {
            'data_drift': data_drift,
            'prediction_drift': pred_drift,
            'n_drifted_features': n_drifted_features,
            'total_features': len(data_drift),
            'overall_drift_detected': (
                pred_drift['drift_detected'] or n_drifted_features > 0
            )
        }

        return report

# Usage
detector = DriftDetector(
    reference_data=train_data,
    reference_predictions=train_predictions,
    threshold=0.05
)

# Monitor production data
report = detector.monitor(
    production_data=recent_data,
    production_predictions=recent_predictions
)

if report['overall_drift_detected']:
    logger.warning("Drift detected! Consider retraining model.")
    # Trigger alert or retraining pipeline
```

### 5. Model Versioning and A/B Testing

#### Model Registry

```python
from typing import Dict, Any, Optional
from pathlib import Path
import json
import torch
from datetime import datetime

class ModelRegistry:
    """Manage model versions and metadata."""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_path / "registry.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load registry metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {'models': {}, 'aliases': {}}

    def _save_metadata(self) -> None:
        """Save registry metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def register_model(
        self,
        name: str,
        version: str,
        model_path: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new model version.

        Args:
            name: Model name
            version: Version string (e.g., "1.0.0")
            model_path: Path to model checkpoint
            metrics: Performance metrics
            metadata: Additional metadata
        """
        model_id = f"{name}:{version}"

        # Copy model to registry
        dest_path = self.registry_path / name / version / "model.pth"
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy(model_path, dest_path)

        # Store metadata
        self.metadata['models'][model_id] = {
            'name': name,
            'version': version,
            'path': str(dest_path),
            'metrics': metrics,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat()
        }

        self._save_metadata()
        logger.info(f"Registered model: {model_id}")

    def set_alias(self, name: str, alias: str, version: str) -> None:
        """
        Set an alias for a model version.

        Args:
            name: Model name
            alias: Alias name (e.g., "production", "staging")
            version: Version to alias
        """
        model_id = f"{name}:{version}"

        if model_id not in self.metadata['models']:
            raise ValueError(f"Model {model_id} not found")

        alias_key = f"{name}:{alias}"
        self.metadata['aliases'][alias_key] = version

        self._save_metadata()
        logger.info(f"Set alias {alias} -> {version} for {name}")

    def get_model(self, name: str, version: str = "production") -> Dict[str, Any]:
        """
        Get model information.

        Args:
            name: Model name
            version: Version or alias

        Returns:
            Model metadata
        """
        # Check if version is an alias
        alias_key = f"{name}:{version}"
        if alias_key in self.metadata['aliases']:
            version = self.metadata['aliases'][alias_key]

        model_id = f"{name}:{version}"

        if model_id not in self.metadata['models']:
            raise ValueError(f"Model {model_id} not found")

        return self.metadata['models'][model_id]

    def list_models(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        models = [
            info for model_id, info in self.metadata['models'].items()
            if info['name'] == name
        ]
        return sorted(models, key=lambda x: x['registered_at'], reverse=True)

# Usage
registry = ModelRegistry("/path/to/registry")

# Register new model
registry.register_model(
    name="sentiment_classifier",
    version="1.0.0",
    model_path="model.pth",
    metrics={'accuracy': 0.95, 'f1': 0.94},
    metadata={'framework': 'pytorch', 'dataset': 'imdb'}
)

# Set production alias
registry.set_alias("sentiment_classifier", "production", "1.0.0")

# Load production model
model_info = registry.get_model("sentiment_classifier", "production")
model = torch.load(model_info['path'])
```

#### A/B Testing Framework

```python
import random
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

@dataclass
class Variant:
    """A/B test variant."""
    name: str
    model_version: str
    traffic_weight: float

class ABTestingFramework:
    """Manage A/B tests for model deployments."""

    def __init__(self):
        self.active_tests: Dict[str, List[Variant]] = {}
        self.metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def create_test(
        self,
        test_name: str,
        variants: List[Variant]
    ) -> None:
        """
        Create a new A/B test.

        Args:
            test_name: Name of the test
            variants: List of variants to test
        """
        # Validate weights sum to 1
        total_weight = sum(v.traffic_weight for v in variants)
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        self.active_tests[test_name] = variants
        logger.info(f"Created A/B test: {test_name} with {len(variants)} variants")

    def select_variant(self, test_name: str, user_id: Optional[str] = None) -> Variant:
        """
        Select variant for a request.

        Args:
            test_name: Name of the test
            user_id: Optional user ID for consistent assignment

        Returns:
            Selected variant
        """
        if test_name not in self.active_tests:
            raise ValueError(f"Test {test_name} not found")

        variants = self.active_tests[test_name]

        if user_id:
            # Consistent assignment based on user_id
            hash_value = hash(user_id) % 10000 / 10000
        else:
            # Random assignment
            hash_value = random.random()

        # Select variant based on cumulative weights
        cumulative = 0
        for variant in variants:
            cumulative += variant.traffic_weight
            if hash_value < cumulative:
                return variant

        return variants[-1]

    def record_metric(
        self,
        test_name: str,
        variant_name: str,
        metric_name: str,
        value: float
    ) -> None:
        """Record a metric for a variant."""
        self.metrics[test_name][f"{variant_name}:{metric_name}"].append(value)

    def get_results(self, test_name: str) -> Dict[str, Dict[str, float]]:
        """
        Get test results with statistical analysis.

        Args:
            test_name: Name of the test

        Returns:
            Results for each variant
        """
        if test_name not in self.active_tests:
            raise ValueError(f"Test {test_name} not found")

        results = {}

        for variant in self.active_tests[test_name]:
            variant_results = {}

            for metric_key, values in self.metrics[test_name].items():
                if metric_key.startswith(f"{variant.name}:"):
                    metric_name = metric_key.split(':', 1)[1]
                    variant_results[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'count': len(values)
                    }

            results[variant.name] = variant_results

        return results

    def compare_variants(
        self,
        test_name: str,
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Statistical comparison between variants.

        Args:
            test_name: Name of the test
            metric_name: Metric to compare

        Returns:
            Comparison results with statistical tests
        """
        variants = self.active_tests[test_name]
        variant_values = []

        for variant in variants:
            key = f"{variant.name}:{metric_name}"
            if key in self.metrics[test_name]:
                variant_values.append((
                    variant.name,
                    self.metrics[test_name][key]
                ))

        if len(variant_values) < 2:
            return {'error': 'Not enough data for comparison'}

        # Perform t-test between first two variants
        from scipy import stats as scipy_stats
        name1, values1 = variant_values[0]
        name2, values2 = variant_values[1]

        t_stat, p_value = scipy_stats.ttest_ind(values1, values2)

        return {
            'variant_a': name1,
            'variant_b': name2,
            'mean_a': float(np.mean(values1)),
            'mean_b': float(np.mean(values2)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'winner': name1 if np.mean(values1) > np.mean(values2) else name2
        }

# Usage in API
ab_testing = ABTestingFramework()

# Create test
ab_testing.create_test(
    test_name="model_comparison",
    variants=[
        Variant("control", "v1.0.0", 0.5),
        Variant("treatment", "v1.1.0", 0.5)
    ]
)

@app.post("/predict_ab")
async def predict_ab(request: PredictionRequest, user_id: str):
    """Prediction with A/B testing."""
    # Select variant
    variant = ab_testing.select_variant("model_comparison", user_id)

    # Load appropriate model version
    model = load_model(variant.model_version)

    # Generate prediction
    result = model.predict(request.features)

    # Record metrics
    ab_testing.record_metric(
        "model_comparison",
        variant.name,
        "confidence",
        result['confidence']
    )

    return result

# Get results
results = ab_testing.get_results("model_comparison")
comparison = ab_testing.compare_variants("model_comparison", "confidence")
```

## Best Practices Summary

### Deployment
1. Use containerization for consistency across environments
2. Implement health checks and readiness probes
3. Use horizontal scaling with load balancing
4. Set resource limits and requests
5. Implement graceful shutdown

### Monitoring
1. Track prediction latency, throughput, and errors
2. Monitor model confidence distributions
3. Implement data and prediction drift detection
4. Use structured logging
5. Set up alerts for anomalies

### Performance
1. Use batch prediction where possible
2. Implement caching for repeated requests
3. Use async processing for long-running tasks
4. Optimize model size (quantization, pruning)
5. Use GPU acceleration when available

### Reliability
1. Implement retries with exponential backoff
2. Use circuit breakers for dependencies
3. Maintain multiple model versions
4. Implement A/B testing for new models
5. Have rollback procedures

## Quick Reference

### Docker Commands
```bash
# Build
docker build -t ml-model:latest .

# Run
docker run -p 8000:8000 ml-model:latest

# Push to registry
docker tag ml-model:latest registry/ml-model:v1.0.0
docker push registry/ml-model:v1.0.0
```

### Kubernetes Commands
```bash
# Deploy
kubectl apply -f deployment.yaml

# Scale
kubectl scale deployment ml-model --replicas=5

# Update
kubectl set image deployment/ml-model ml-model=registry/ml-model:v1.1.0

# Rollback
kubectl rollout undo deployment/ml-model
```

### Testing Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1,2,3,4,5,6,7,8,9,10]}'

# Metrics
curl http://localhost:8000/metrics
```

## When to Use This Skill

Use this skill when you need to:
- Deploy ML models as production APIs
- Containerize models with Docker
- Orchestrate deployments with Kubernetes
- Deploy to cloud platforms (AWS/GCP/Azure)
- Implement monitoring and observability
- Detect data and prediction drift
- Set up A/B testing for models
- Manage model versions
- Scale model serving infrastructure
- Optimize inference latency

This skill provides complete guidance for taking ML models from development to production at scale.
