---
version: "1.0.7"
command: /ml-pipeline
description: Design and implement production-ready ML pipelines with multi-agent MLOps orchestration
execution_modes:
  quick: "2-3d: MVP data pipeline, training, simple deployment"
  standard: "1-2w: + Monitoring, testing, optimization"
  enterprise: "3-4w: + K8s, distributed training, governance"
external_docs:
  methodology: "../docs/ml-pipeline/mlops-methodology.md"
  phases: "../docs/ml-pipeline/pipeline-phases.md"
  deployment: "../docs/ml-pipeline/deployment-strategies.md"
  monitoring: "../docs/ml-pipeline/monitoring-frameworks.md"
  practices: "../docs/ml-pipeline/best-practices.md"
  criteria: "../docs/ml-pipeline/success-criteria.md"
---

# ML Pipeline

$ARGUMENTS

## Phase 1: Data & Requirements

**Data Infrastructure (Enterprise):** Ingestion, CDC, schema validation (Pydantic, Great Expectations), versioning (DVC, lakeFS), storage (Bronze/Silver/Gold)

**Feature & Model Design (All):** Feature engineering + store schema (Feast/Tecton), model requirements + algorithm selection, performance baselines + evaluation, experiment design + A/B methodology

**Ref:** [pipeline-phases.md#phase-1](../docs/ml-pipeline/pipeline-phases.md)

## Phase 2: Development & Training

**Training:** Hyperparameter tuning (Optuna, Ray Tune), Distributed training (Horovod, PyTorch DDP), Experiment tracking (MLflow, W&B), Model registry (MLflow, custom)

**Code Optimization (Standard+):** Production standards, error handling, performance profiling, comprehensive testing

**Ref:** [pipeline-phases.md#phase-2](../docs/ml-pipeline/pipeline-phases.md)

## Phase 3: Deployment

**Model Serving:**
- REST/gRPC: FastAPI, TorchServe
- Batch: Airflow, Kubeflow
- Streaming: Kafka, Kinesis

**Strategies:** Blue-green (safe rollout), Canary (gradual), Shadow (testing with production traffic), A/B testing (comparative)

**K8s (Enterprise):** GPU scheduling + autoscaling, KEDA event-driven scaling, Service mesh (Istio)

**Ref:** [deployment-strategies.md](../docs/ml-pipeline/deployment-strategies.md)

## Phase 4: Monitoring & Improvement

**Observability:** Model performance (accuracy, latency, throughput), Drift detection (KS test, PSI, feature distributions), System metrics (Prometheus, Grafana), Alerting (PagerDuty, retraining triggers), Cost tracking (resource utilization, optimization)

**Ref:** [monitoring-frameworks.md](../docs/ml-pipeline/monitoring-frameworks.md)

## Tech Stack

- Experiment Tracking: MLflow, W&B, Neptune, ClearML
- Feature Store: Feast, Tecton, Databricks
- Serving: KServe, Seldon, TorchServe, Triton, BentoML
- Orchestration: Kubeflow, Airflow, Prefect, Dagster
- Cloud: AWS, Azure, GCP
- Deployment: Realtime, batch, streaming, hybrid
- Monitoring: Prometheus, Datadog, New Relic

## Success Criteria

- Data Quality: <0.1% issues, <1s feature latency
- Model Performance: Meet baselines, <5% degradation before retrain
- Operational: 99.9% uptime, <200ms p99, <5min rollback
- Velocity: <1h commit-to-prod, reproducible runs
- Cost: <20% waste, >60% spot utilization

**Full metrics:** [success-criteria.md](../docs/ml-pipeline/success-criteria.md)

## Deliverables

End-to-end ML pipeline with automation, Infrastructure (Terraform/Helm IaC), CI/CD pipelines, Monitoring dashboards + alerting, Operational runbooks, Cost optimization strategies, Rollback procedures

## External Docs

[mlops-methodology.md](../docs/ml-pipeline/mlops-methodology.md), [pipeline-phases.md](../docs/ml-pipeline/pipeline-phases.md), [deployment-strategies.md](../docs/ml-pipeline/deployment-strategies.md), [monitoring-frameworks.md](../docs/ml-pipeline/monitoring-frameworks.md), [best-practices.md](../docs/ml-pipeline/best-practices.md), [success-criteria.md](../docs/ml-pipeline/success-criteria.md)
