# ML Pipeline Phases - Detailed Implementation Guide

Comprehensive phase-by-phase guide for implementing production ML pipelines with checklists and templates.

## Phase 1: Data Infrastructure & Requirements

### Data Infrastructure (Enterprise Mode - data-engineer)

**Data Source Audit Checklist**:
- [ ] Identify all data sources (databases, APIs, files, streams)
- [ ] Document data schemas and formats
- [ ] Assess data volume and growth rate
- [ ] Determine update frequency (real-time, batch, micro-batch)
- [ ] Evaluate data quality and completeness
- [ ] Check access permissions and authentication
- [ ] Measure current latency and throughput

**Schema Validation Setup**:
```python
# Great Expectations setup
import great_expectations as ge

# Create expectation suite
suite = context.create_expectation_suite("data_quality_suite")

# Add expectations
suite.add_expectation(ge.expect_column_values_to_not_be_null(column="customer_id"))
suite.add_expectation(ge.expect_column_values_to_be_in_set(column="status", value_set=["active", "churned"]))
suite.add_expectation(ge.expect_column_values_to_be_between(column="age", min_value=18, max_value=100))
```

**Data Versioning (DVC)**:
```bash
# Initialize DVC
dvc init
dvc remote add -d storage s3://ml-data-versioning

# Track dataset
dvc add data/raw/customer_data.csv
git add data/raw/customer_data.csv.dvc .dvc/config
git commit -m "Track customer data v1.0"
dvc push
```

**Storage Architecture**:
```
Bronze Layer (Raw Data)
  ↓ Minimal transformation
Silver Layer (Cleaned Data)
  ↓ Feature engineering
Gold Layer (Feature Store)
  ↓ Model training/serving
```

---

### Requirements Analysis (All Modes - data-scientist)

**Feature Engineering Specification**:
```python
# Feature transformation specs
feature_specs = {
    'customer_tenure_days': {
        'type': 'derived',
        'calculation': 'days_between(current_date, signup_date)',
        'validation': 'value >= 0'
    },
    'total_spend_30d': {
        'type': 'aggregation',
        'calculation': 'sum(transaction_amount) over last 30 days',
        'window': '30 days',
        'validation': 'value >= 0'
    },
    'avg_session_duration': {
        'type': 'aggregation',
        'calculation': 'avg(session_duration) over last 7 days',
        'window': '7 days',
        'validation': '0 <= value <= 86400'
    }
}
```

**Model Requirements Template**:
```yaml
model_requirements:
  problem_type: binary_classification
  target_variable: churned
  baseline_metrics:
    accuracy: 0.75
    precision: 0.70
    recall: 0.65
    f1_score: 0.67

  business_metrics:
    cost_of_false_negative: $500  # Lost customer lifetime value
    cost_of_false_positive: $50   # Cost of retention offer

  performance_requirements:
    training_time: < 2 hours
    inference_latency_p99: < 100ms
    throughput: > 1000 predictions/sec

  data_requirements:
    minimum_training_samples: 100000
    minimum_positive_class_ratio: 0.15
    feature_count: 80-120
```

---

## Phase 2: Model Development & Training

### Training Pipeline Implementation (ml-engineer)

**Modular Training Code Structure**:
```python
# train.py
import mlflow
from src.data import load_data, preprocess
from src.features import engineer_features
from src.models import XGBoostClassifier
from src.evaluation import evaluate_model

def train_pipeline(config):
    """End-to-end training pipeline"""

    # Load and preprocess
    data = load_data(config['data_path'])
    X_train, X_test, y_train, y_test = preprocess(data, config)

    # Feature engineering
    X_train_features = engineer_features(X_train)
    X_test_features = engineer_features(X_test)

    # Train with hyperparameter optimization
    best_model = optimize_hyperparameters(
        X_train_features, y_train,
        n_trials=config['n_trials']
    )

    # Evaluate
    metrics = evaluate_model(best_model, X_test_features, y_test)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")

    return best_model, metrics
```

**Hyperparameter Optimization (Optuna)**:
```python
import optuna

def optimize_hyperparameters(X_train, y_train, n_trials=50):
    """Optuna hyperparameter search"""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }

        model = XGBClassifier(**params)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return XGBClassifier(**study.best_params)
```

---

### Code Optimization & Testing (Standard/Enterprise - python-pro)

**Production Code Quality Checklist**:
- [ ] Replace hardcoded values with configuration
- [ ] Add comprehensive error handling with specific exceptions
- [ ] Implement structured logging (JSON format)
- [ ] Create reusable utility functions
- [ ] Add type hints for all functions
- [ ] Write docstrings with examples
- [ ] Profile and optimize bottlenecks
- [ ] Implement caching where appropriate

**Testing Framework**:
```python
# tests/test_features.py
import pytest
from src.features import engineer_features, calculate_customer_tenure

def test_customer_tenure_calculation():
    """Test tenure calculation logic"""
    signup_date = "2023-01-01"
    current_date = "2024-01-01"

    tenure = calculate_customer_tenure(signup_date, current_date)

    assert tenure == 365, f"Expected 365 days, got {tenure}"

def test_feature_engineering_output_shape():
    """Test feature engineering maintains sample count"""
    X_input = pd.DataFrame({'customer_id': range(1000)})

    X_features = engineer_features(X_input)

    assert len(X_features) == 1000, "Sample count mismatch"
    assert X_features.shape[1] >= 80, f"Expected >=80 features, got {X_features.shape[1]}"

# tests/test_model.py
def test_model_invariance():
    """Test model predictions are invariant to feature order"""
    model = load_model('model.pkl')
    X_test = load_test_data()

    predictions_1 = model.predict(X_test)
    predictions_2 = model.predict(X_test[X_test.columns[::-1]])  # Reverse column order

    assert (predictions_1 == predictions_2).all(), "Model not invariant to feature order"
```

---

## Phase 3: Production Deployment

### Model Serving Infrastructure (mlops-engineer)

**FastAPI Model Serving**:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI()

# Load model
model = mlflow.pyfunc.load_model("models:/CustomerChurn/Production")

class PredictionRequest(BaseModel):
    customer_id: str
    features: dict

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: str
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Extract features
        X = pd.DataFrame([request.features])

        # Predict
        proba = model.predict_proba(X)[0][1]
        prediction = "churn" if proba > 0.5 else "retain"

        return PredictionResponse(
            customer_id=request.customer_id,
            churn_probability=float(proba),
            prediction=prediction,
            model_version="v2.1.3"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
```

**CI/CD Pipeline (GitHub Actions)**:
```yaml
# .github/workflows/deploy-model.yml
name: Deploy ML Model

on:
  push:
    tags:
      - 'model-v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: |
          docker build -t ml-model:${{ github.ref_name }} .
          docker tag ml-model:${{ github.ref_name }} ml-model:latest

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push ml-model:${{ github.ref_name }}
          docker push ml-model:latest

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/ml-model-serving ml-model=ml-model:${{ github.ref_name }}
          kubectl rollout status deployment/ml-model-serving
```

---

### Kubernetes Orchestration (Enterprise - kubernetes-architect)

**GPU Training Job**:
```yaml
# k8s/training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ml-training-job
spec:
  template:
    spec:
      containers:
      - name: training
        image: ml-training:latest
        resources:
          limits:
            nvidia.com/gpu: 2  # Request 2 GPUs
            memory: "32Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 2
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: training-data
          mountPath: /data
      volumes:
      - name: training-data
        persistentVolumeClaim:
          claimName: training-data-pvc
      restartPolicy: Never
  backoffLimit: 3
```

**Model Serving with Autoscaling**:
```yaml
# k8s/model-serving.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-serving
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
      - name: model-server
        image: ml-model:latest
        ports:
        - containerPort: 8000
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
---
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
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-serving
  minReplicas: 3
  maxReplicas: 20
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
```

---

## Phase 4: Monitoring & Continuous Improvement

### Monitoring Setup (Standard/Enterprise - observability-engineer)

**Prometheus Metrics**:
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
prediction_counter = Counter('ml_predictions_total', 'Total predictions made')
prediction_latency = Histogram('ml_prediction_latency_seconds', 'Prediction latency')
model_accuracy_gauge = Gauge('ml_model_accuracy', 'Current model accuracy')

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()

    # Make prediction
    result = model.predict(request.features)

    # Record metrics
    prediction_counter.inc()
    prediction_latency.observe(time.time() - start_time)

    return result
```

**Drift Detection**:
```python
from scipy.stats import ks_2samp
import pandas as pd

def detect_feature_drift(reference_data, current_data, threshold=0.05):
    """Detect statistical drift using Kolmogorov-Smirnov test"""

    drift_report = []

    for column in reference_data.columns:
        statistic, p_value = ks_2samp(
            reference_data[column],
            current_data[column]
        )

        drifted = p_value < threshold

        drift_report.append({
            'feature': column,
            'ks_statistic': statistic,
            'p_value': p_value,
            'drifted': drifted
        })

    return pd.DataFrame(drift_report)

# Run daily drift check
reference_data = load_training_data()
current_data = load_production_data(last_7_days=True)
drift_report = detect_feature_drift(reference_data, current_data)

if drift_report['drifted'].any():
    alert_data_science_team(drift_report)
```

**Grafana Dashboard Configuration**:
```json
{
  "dashboard": {
    "title": "ML Model Performance",
    "panels": [
      {
        "title": "Prediction Latency (p50, p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, ml_prediction_latency_seconds_bucket)"
          },
          {
            "expr": "histogram_quantile(0.95, ml_prediction_latency_seconds_bucket)"
          },
          {
            "expr": "histogram_quantile(0.99, ml_prediction_latency_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Prediction Throughput",
        "targets": [
          {
            "expr": "rate(ml_predictions_total[5m])"
          }
        ]
      },
      {
        "title": "Model Accuracy (Last 24h)",
        "targets": [
          {
            "expr": "ml_model_accuracy"
          }
        ]
      }
    ]
  }
}
```

---

## Phase Transition Checklist

### Phase 1 → Phase 2 Transition
- [ ] Data sources documented and accessible
- [ ] Data quality validation passing >99%
- [ ] Feature store configured
- [ ] Feature engineering specs approved
- [ ] Model requirements defined with baselines

### Phase 2 → Phase 3 Transition
- [ ] Model training pipeline automated
- [ ] Model meets performance baselines
- [ ] Model registered in MLflow registry
- [ ] All tests passing (unit, integration, model quality)
- [ ] Documentation complete (model card, architecture)

### Phase 3 → Phase 4 Transition
- [ ] Model deployed to production
- [ ] CI/CD pipeline operational
- [ ] Rollback procedure tested
- [ ] Load testing completed successfully
- [ ] Infrastructure as Code committed to git

### Phase 4 → Continuous Improvement
- [ ] Monitoring dashboards live
- [ ] Drift detection running
- [ ] Automated alerts configured
- [ ] Retraining pipeline defined
- [ ] Incident response runbook created
