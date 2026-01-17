---
description: Design and implement production-ready ML pipelines with multi-agent MLOps
  orchestration
triggers:
- /ml-pipeline
- design and implement production ready
allowed-tools: [Bash, Read, Write, Task]
version: 1.0.0
---



# ML Pipeline

$ARGUMENTS

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

## Phase 1: Data & Requirements (Parallel Execution)

**Data Infrastructure (Enterprise):** Ingestion, CDC, schema validation (Pydantic, Great Expectations), versioning (DVC, lakeFS), storage (Bronze/Silver/Gold)

**Feature & Model Design (All):** Feature engineering + store schema (Feast/Tecton), model requirements + algorithm selection, performance baselines + evaluation, experiment design + A/B methodology

**Ref:** [pipeline-phases.md#phase-1](../../plugins/machine-learning/docs/ml-pipeline/pipeline-phases.md)

## Phase 2: Development & Training (Iterative/Parallel)

**Training:** Hyperparameter tuning (Optuna, Ray Tune), Distributed training (Horovod, PyTorch DDP), Experiment tracking (MLflow, W&B), Model registry (MLflow, custom)

**Code Optimization (Standard+):** Production standards, error handling, performance profiling, comprehensive testing

**Ref:** [pipeline-phases.md#phase-2](../../plugins/machine-learning/docs/ml-pipeline/pipeline-phases.md)

## Phase 3: Deployment (Sequential)

**Model Serving:**
- REST/gRPC: FastAPI, TorchServe
- Batch: Airflow, Kubeflow
- Streaming: Kafka, Kinesis

**Strategies:** Blue-green (safe rollout), Canary (gradual), Shadow (testing with production traffic), A/B testing (comparative)

**K8s (Enterprise):** GPU scheduling + autoscaling, KEDA event-driven scaling, Service mesh (Istio)

**Ref:** [deployment-strategies.md](../../plugins/machine-learning/docs/ml-pipeline/deployment-strategies.md)

## Phase 4: Monitoring & Improvement (Continuous)

**Observability:** Model performance (accuracy, latency, throughput), Drift detection (KS test, PSI, feature distributions), System metrics (Prometheus, Grafana), Alerting (PagerDuty, retraining triggers), Cost tracking (resource utilization, optimization)

**Ref:** [monitoring-frameworks.md](../../plugins/machine-learning/docs/ml-pipeline/monitoring-frameworks.md)

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

**Full metrics:** [success-criteria.md](../../plugins/machine-learning/docs/ml-pipeline/success-criteria.md)

## Deliverables

End-to-end ML pipeline with automation, Infrastructure (Terraform/Helm IaC), CI/CD pipelines, Monitoring dashboards + alerting, Operational runbooks, Cost optimization strategies, Rollback procedures

## External Docs

[mlops-methodology.md](../../plugins/machine-learning/docs/ml-pipeline/mlops-methodology.md), [pipeline-phases.md](../../plugins/machine-learning/docs/ml-pipeline/pipeline-phases.md), [deployment-strategies.md](../../plugins/machine-learning/docs/ml-pipeline/deployment-strategies.md), [monitoring-frameworks.md](../../plugins/machine-learning/docs/ml-pipeline/monitoring-frameworks.md), [best-practices.md](../../plugins/machine-learning/docs/ml-pipeline/best-practices.md), [success-criteria.md](../../plugins/machine-learning/docs/ml-pipeline/success-criteria.md)
