---
version: 1.0.5
command: /ml-pipeline
description: Design and implement production-ready ML pipelines with multi-agent MLOps orchestration
execution_modes:
  quick: "2-3 days"
  standard: "1-2 weeks"
  enterprise: "3-4 weeks"
external_docs:
  methodology: "../docs/ml-pipeline/mlops-methodology.md"
  phases: "../docs/ml-pipeline/pipeline-phases.md"
  deployment: "../docs/ml-pipeline/deployment-strategies.md"
  monitoring: "../docs/ml-pipeline/monitoring-frameworks.md"
  practices: "../docs/ml-pipeline/best-practices.md"
  criteria: "../docs/ml-pipeline/success-criteria.md"
---

# ML Pipeline - Multi-Agent MLOps Orchestration

Build production ML pipeline for: $ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope | Agents |
|------|----------|-------|--------|
| Quick | 2-3 days | MVP: data pipeline, training, simple deployment | data-scientist, ml-engineer, mlops-engineer |
| Standard (default) | 1-2 weeks | Full: + monitoring, testing, optimization | + python-pro, observability-engineer |
| Enterprise | 3-4 weeks | Complete: + K8s, distributed training, governance | + kubernetes-architect, data-engineer |

---

## Agent Responsibilities

| Agent | Role | Key Tasks |
|-------|------|-----------|
| data-engineer | Data Infrastructure | Ingestion, quality, versioning, storage |
| data-scientist | Feature & Model Design | Feature engineering, experiments, evaluation |
| ml-engineer | Training Infrastructure | Pipelines, tuning, distributed training, registry |
| python-pro | Code Optimization | Refactoring, testing, performance |
| mlops-engineer | Deployment & Serving | Model serving, CI/CD, IaC |
| kubernetes-architect | K8s Orchestration | Scheduling, autoscaling, service mesh |
| observability-engineer | Monitoring | Drift detection, alerting, cost tracking |

---

## Phase 1: Data & Requirements Analysis

### Data Infrastructure (Enterprise)
- Data ingestion and CDC strategies
- Schema validation (Pydantic, Great Expectations)
- Data versioning (DVC, lakeFS)
- Storage architecture (Bronze/Silver/Gold)

### Feature & Model Design (All Modes)
- Feature engineering and store schema (Feast/Tecton)
- Model requirements and algorithm selection
- Performance baselines and evaluation criteria
- Experiment design and A/B methodology

**Reference:** [pipeline-phases.md#phase-1](../docs/ml-pipeline/pipeline-phases.md)

---

## Phase 2: Model Development & Training

### Training Pipeline
| Component | Tools |
|-----------|-------|
| Hyperparameter tuning | Optuna, Ray Tune |
| Distributed training | Horovod, PyTorch DDP |
| Experiment tracking | MLflow, W&B |
| Model registry | MLflow, custom |

### Code Optimization (Standard+)
- Production standards and error handling
- Performance profiling and optimization
- Comprehensive test coverage

**Reference:** [pipeline-phases.md#phase-2](../docs/ml-pipeline/pipeline-phases.md)

---

## Phase 3: Production Deployment

### Model Serving
| Type | Tools |
|------|-------|
| REST/gRPC APIs | FastAPI, TorchServe |
| Batch pipelines | Airflow, Kubeflow |
| Streaming | Kafka, Kinesis |

### Deployment Strategies
| Strategy | Use Case |
|----------|----------|
| Blue-green | Safe rollout |
| Canary | Gradual rollout |
| Shadow | Testing with production traffic |
| A/B testing | Comparative validation |

### K8s Infrastructure (Enterprise)
- GPU scheduling and autoscaling
- KEDA event-driven scaling
- Service mesh (Istio)

**Reference:** [deployment-strategies.md](../docs/ml-pipeline/deployment-strategies.md)

---

## Phase 4: Monitoring & Improvement

### Observability Components
| Component | Purpose |
|-----------|---------|
| Model performance | Accuracy, latency, throughput |
| Drift detection | KS test, PSI, feature distributions |
| System metrics | Prometheus, Grafana |
| Alerting | PagerDuty, retraining triggers |
| Cost tracking | Resource utilization, optimization |

**Reference:** [monitoring-frameworks.md](../docs/ml-pipeline/monitoring-frameworks.md)

---

## Technology Stack Options

| Component | Options |
|-----------|---------|
| Experiment Tracking | MLflow, W&B, Neptune, ClearML |
| Feature Store | Feast, Tecton, Databricks |
| Serving Platform | KServe, Seldon, TorchServe, Triton, BentoML |
| Orchestration | Kubeflow, Airflow, Prefect, Dagster |
| Cloud Provider | AWS, Azure, GCP |
| Deployment Mode | Realtime, batch, streaming, hybrid |
| Monitoring Stack | Prometheus, Datadog, New Relic |

---

## Success Criteria

| Category | Target |
|----------|--------|
| Data Quality | <0.1% issues, <1s feature latency |
| Model Performance | Meet baselines, <5% degradation before retrain |
| Operational | 99.9% uptime, <200ms p99, <5min rollback |
| Velocity | <1h commit-to-prod, reproducible runs |
| Cost | <20% waste, >60% spot utilization |

**Full metrics:** [success-criteria.md](../docs/ml-pipeline/success-criteria.md)

---

## Deliverables

| Deliverable | Content |
|-------------|---------|
| ML Pipeline | End-to-end with automation |
| Infrastructure | Terraform/Helm IaC |
| CI/CD | Continuous deployment pipelines |
| Monitoring | Dashboards and alerting |
| Documentation | Operational runbooks |
| Cost Optimization | Scaling strategies |
| DR | Rollback procedures |

---

## External Documentation

| Document | Purpose |
|----------|---------|
| [mlops-methodology.md](../docs/ml-pipeline/mlops-methodology.md) | Core principles and methodology |
| [pipeline-phases.md](../docs/ml-pipeline/pipeline-phases.md) | Detailed phase specifications |
| [deployment-strategies.md](../docs/ml-pipeline/deployment-strategies.md) | Deployment patterns |
| [monitoring-frameworks.md](../docs/ml-pipeline/monitoring-frameworks.md) | Observability setup |
| [best-practices.md](../docs/ml-pipeline/best-practices.md) | Production readiness |
| [success-criteria.md](../docs/ml-pipeline/success-criteria.md) | Metrics and KPIs |
