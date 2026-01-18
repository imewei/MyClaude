# MLOps Methodology Guide

Comprehensive guide to MLOps principles, maturity models, and best practices for production machine learning systems.

## Table of Contents
- [MLOps Maturity Model](#mlops-maturity-model)
- [CI/CD for Machine Learning](#cicd-for-machine-learning)
- [Experiment Tracking](#experiment-tracking-best-practices)
- [Model Governance](#model-governance--compliance)
- [Cost Optimization](#cost-optimization-strategies)
- [Team Collaboration](#team-collaboration-patterns)

---

## MLOps Maturity Model

### Level 0: Manual Process
**Characteristics**:
- Manual model training and deployment
- No experiment tracking or versioning
- Ad-hoc monitoring and debugging
- Jupyter notebooks as primary tooling
- No automated testing or validation

**Limitations**:
- Not reproducible
- Slow iteration cycles
- High risk of errors
- Difficult to scale
- No audit trail

**Typical for**: Proof-of-concepts, research projects

---

### Level 1: DevOps for ML (ML Pipeline Automation)
**Characteristics**:
- Automated training pipelines
- Experiment tracking (MLflow, W&B)
- Model registry with versioning
- Basic CI/CD for model deployment
- Automated data validation

**Capabilities**:
- Reproducible training runs
- Version control for code, data, and models
- Automated model serving
- Basic monitoring dashboards
- Rollback capabilities

**Tools**: MLflow, Airflow, Jenkins, Docker

**Typical for**: Production ML systems, single team

---

### Level 2: Automated ML Pipeline (MLOps)
**Characteristics**:
- End-to-end automated pipelines
- Continuous training with new data
- A/B testing infrastructure
- Comprehensive monitoring and drift detection
- Feature stores for feature reuse
- Automated retraining triggers

**Capabilities**:
- Continuous integration and deployment
- Automated model validation before deployment
- Feature engineering pipelines
- Advanced monitoring (data drift, concept drift)
- Cost tracking and optimization
- Multi-model deployment

**Tools**: Kubeflow, MLflow, Feast, KServe, Prometheus, Grafana

**Typical for**: Enterprise ML platforms, multiple teams

---

### Level 3: Full CI/CD/CT Automation
**Characteristics**:
- Continuous training (CT) with data pipelines
- Automated feature engineering
- Multi-cloud deployment strategies
- Real-time monitoring and auto-remediation
- Federated learning capabilities
- ML platform as a service

**Capabilities**:
- Fully automated end-to-end workflows
- Self-healing systems
- Advanced experimentation (multi-armed bandits)
- Automated compliance checking
- Cross-functional ML platform
- Model performance SLAs

**Tools**: Kubeflow, Vertex AI, SageMaker, Databricks ML

**Typical for**: ML-native companies, large-scale platforms

---

## CI/CD for Machine Learning

### Model Versioning

**Git for Code**:
```bash
# Version control for training code
git tag v1.2.3-model-training
git push --tags
```

**DVC for Data**:
```bash
# Version large datasets
dvc add data/training_data.csv
dvc push

# Checkout specific data version
dvc checkout data/training_data.csv.dvc
```

**MLflow for Models**:
```python
import mlflow

# Log model with version
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_params({"n_estimators": 100, "max_depth": 5})
    mlflow.log_metrics({"accuracy": 0.95, "f1": 0.93})
```

---

### CI Pipeline for ML

**GitHub Actions Workflow**:
```yaml
# .github/workflows/ml-ci.yml
name: ML CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-data-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          uv uv pip install -r requirements.txt
          uv uv pip install pytest great-expectations

      - name: Run data validation
        run: |
          great_expectations checkpoint run data_quality_checkpoint

      - name: Run unit tests
        run: |
          pytest tests/data_pipeline/ -v

  test-model-training:
    runs-on: ubuntu-latest
    needs: test-data-pipeline
    steps:
      - uses: actions/checkout@v3
      - name: Train model on sample data
        run: |
          python train.py --config config/test_config.yaml

      - name: Run model tests
        run: |
          pytest tests/model/ -v

      - name: Check model performance
        run: |
          python scripts/validate_model.py --min_accuracy 0.80

  build-docker-image:
    runs-on: ubuntu-latest
    needs: [test-data-pipeline, test-model-training]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Build and push Docker image
        run: |
          docker build -t ml-model:${{ github.sha }} .
          docker push ml-model:${{ github.sha }}
```

---

### CD Pipeline for Models

**Automated Deployment with ArgoCD**:
```yaml
# argocd/ml-model-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-model-serving
spec:
  project: ml-platform
  source:
    repoURL: https://github.com/company/ml-models
    targetRevision: HEAD
    path: k8s/model-serving
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

**Progressive Rollout**:
```python
# Canary deployment with gradual traffic shift
def deploy_canary(model_version, traffic_percentage=10):
    """Deploy new model version to canary with gradual traffic increase"""

    # Deploy canary
    deploy_model(model_version, "canary")

    # Shift traffic gradually: 10% → 25% → 50% → 100%
    for traffic in [10, 25, 50, 100]:
        set_traffic_split(stable=100-traffic, canary=traffic)

        # Monitor for 1 hour
        metrics = monitor_canary(duration_minutes=60)

        if metrics['error_rate'] > 0.01 or metrics['latency_p99'] > 500:
            # Rollback if quality degrades
            rollback_canary()
            raise Exception(f"Canary failed at {traffic}% traffic")

    # Promote canary to stable
    promote_canary_to_stable(model_version)
```

---

## Experiment Tracking Best Practices

### MLflow Organization

**Experiment Structure**:
```
project-name/
├── experiments/
│   ├── baseline-models/
│   ├── feature-engineering-v1/
│   ├── hyperparameter-tuning/
│   └── production-candidates/
```

**Comprehensive Logging**:
```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Set experiment
mlflow.set_experiment("customer-churn-v2")

with mlflow.start_run(run_name="xgboost-tuned"):
    # Log parameters
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8
    }
    mlflow.log_params(params)

    # Train model
    model = train_xgboost(params)

    # Log metrics
    metrics = evaluate_model(model, X_test, y_test)
    mlflow.log_metrics({
        "accuracy": metrics['accuracy'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1_score": metrics['f1'],
        "roc_auc": metrics['auc']
    })

    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.sklearn.log_model(model, "model")

    # Log dataset info
    mlflow.log_param("training_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("feature_count", X_train.shape[1])

    # Add tags for filtering
    mlflow.set_tags({
        "model_type": "xgboost",
        "dataset_version": "v2.1",
        "engineer": "data-science-team",
        "production_candidate": "true"
    })
```

---

### Weights & Biases Integration

```python
import wandb

# Initialize W&B
wandb.init(
    project="customer-churn",
    config={
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32
    }
)

# Log during training
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

# Log model
wandb.save("model.pth")

# Create table for predictions
wandb.log({"predictions": wandb.Table(dataframe=predictions_df)})
```

---

## Model Governance & Compliance

### Model Registry with Promotion Workflow

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = "runs:/<run_id>/model"
model_details = mlflow.register_model(model_uri, "CustomerChurnModel")

# Transition to staging
client.transition_model_version_stage(
    name="CustomerChurnModel",
    version=model_details.version,
    stage="Staging"
)

# Add model description and metadata
client.update_model_version(
    name="CustomerChurnModel",
    version=model_details.version,
    description="XGBoost model with feature engineering v2.1, 94% accuracy"
)

# Set model tags
client.set_model_version_tag(
    name="CustomerChurnModel",
    version=model_details.version,
    key="validation_status",
    value="passed"
)

# After validation, promote to production
client.transition_model_version_stage(
    name="CustomerChurnModel",
    version=model_details.version,
    stage="Production",
    archive_existing_versions=True  # Archive old production model
)
```

---

### Audit Logging

```python
import logging
from datetime import datetime

def audit_log_model_deployment(model_name, version, deployed_by, environment):
    """Log all model deployments for compliance"""

    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "MODEL_DEPLOYMENT",
        "model_name": model_name,
        "model_version": version,
        "environment": environment,
        "deployed_by": deployed_by,
        "approval_status": get_approval_status(model_name, version),
        "compliance_check": run_compliance_check(model_name)
    }

    # Log to audit database
    audit_db.insert(audit_entry)

    # Notify stakeholders
    notify_deployment(audit_entry)

    logging.info(f"Audit log: {audit_entry}")
```

---

## Cost Optimization Strategies

### Resource Right-Sizing

**Training Optimization**:
```python
# Use spot instances for training (70% cost savings)
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri='training-image',
    role='SageMakerRole',
    instance_count=4,
    instance_type='ml.p3.8xlarge',
    use_spot_instances=True,  # Enable spot instances
    max_wait=7200,  # Maximum wait time
    max_run=3600,   # Maximum run time
    checkpoint_s3_uri='s3://bucket/checkpoints/'  # Save checkpoints
)
```

**Inference Optimization**:
```python
# Auto-scaling based on load
from kubernetes import client, config

autoscaling = client.V1HorizontalPodAutoscaler(
    metadata=client.V1ObjectMeta(name="ml-model-hpa"),
    spec=client.V2HorizontalPodAutoscalerSpec(
        scale_target_ref=client.V2CrossVersionObjectReference(
            api_version="apps/v1",
            kind="Deployment",
            name="ml-model-serving"
        ),
        min_replicas=2,
        max_replicas=10,
        metrics=[
            client.V2MetricSpec(
                type="Resource",
                resource=client.V2ResourceMetricSource(
                    name="cpu",
                    target=client.V2MetricTarget(
                        type="Utilization",
                        average_utilization=70
                    )
                )
            )
        ]
    )
)
```

---

### Storage Lifecycle Policies

```python
# S3 lifecycle policy for ML artifacts
import boto3

s3 = boto3.client('s3')

lifecycle_policy = {
    'Rules': [
        {
            'ID': 'Archive old experiments',
            'Status': 'Enabled',
            'Prefix': 'mlflow/experiments/',
            'Transitions': [
                {
                    'Days': 90,
                    'StorageClass': 'INTELLIGENT_TIERING'
                },
                {
                    'Days': 365,
                    'StorageClass': 'GLACIER'
                }
            ]
        },
        {
            'ID': 'Delete temporary artifacts',
            'Status': 'Enabled',
            'Prefix': 'temp/',
            'Expiration': {
                'Days': 7
            }
        }
    ]
}

s3.put_bucket_lifecycle_configuration(
    Bucket='ml-artifacts',
    LifecycleConfiguration=lifecycle_policy
)
```

---

## Team Collaboration Patterns

### Cross-Functional Workflows

```
Data Engineers         Data Scientists           ML Engineers            MLOps Engineers
      ↓                       ↓                        ↓                       ↓
Build data pipelines → Design features    → Build training    → Deploy to production
Monitor data quality → Run experiments    → Optimize models   → Monitor performance
Maintain feature     → Validate models    → Set up registry   → Manage infrastructure
  store                                      Test pipelines       Scale serving
```

### Code Review Standards

**ML-Specific Review Checklist**:
- [ ] Data validation tests included?
- [ ] Model performance meets baseline?
- [ ] Experiment tracked in MLflow/W&B?
- [ ] Model explainability considered?
- [ ] Resource requirements documented?
- [ ] Deployment rollback plan defined?
- [ ] Monitoring alerts configured?
- [ ] Cost impact analyzed?

---

### Documentation Standards

**Model Card Template**:
```markdown
# Model Card: Customer Churn Prediction

## Model Details
- **Model Type**: XGBoost Classifier
- **Version**: 2.1.3
- **Training Date**: 2025-01-15
- **Owner**: Data Science Team
- **Contact**: ds-team@company.com

## Intended Use
- **Primary Use**: Predict customer churn risk for retention campaigns
- **Out-of-Scope**: Not for regulatory/legal decisions

## Training Data
- **Source**: Customer database (2022-2024)
- **Size**: 500K customers, 120 features
- **Preprocessing**: StandardScaler, OneHotEncoder

## Performance
- **Validation Accuracy**: 94.2%
- **Precision**: 0.91
- **Recall**: 0.89
- **AUC-ROC**: 0.96

## Limitations
- Performance degrades for customers with <3 months history
- May underperform for newly launched products

## Ethical Considerations
- Fairness audit conducted across demographic groups
- No protected attributes used as features
```

---

## Best Practices Summary

1. **Start Simple, Automate Gradually**: Don't try to reach maturity level 3 immediately
2. **Version Everything**: Code, data, models, and infrastructure
3. **Monitor Continuously**: Data quality, model performance, system health, costs
4. **Fail Fast**: Comprehensive validation gates before production
5. **Document Thoroughly**: Model cards, runbooks, architecture decisions
6. **Optimize Costs**: Spot instances, lifecycle policies, right-sizing
7. **Collaborate Effectively**: Clear handoffs, code reviews, shared repositories
8. **Measure Impact**: Track business KPIs alongside ML metrics
