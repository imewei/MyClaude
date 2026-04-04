---
name: ml-deployment
description: Meta-orchestrator for ML model deployment and operations. Routes to model serving, optimization, production engineering, pipelines, DevOps infrastructure, and federated learning skills. Use when deploying ML models with FastAPI/TorchServe, optimizing with quantization/pruning, building production ML systems, building MLOps pipelines, or implementing federated learning.
---

# ML Deployment

Orchestrator for ML model deployment and MLOps. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`ml-expert`**: Specialist for ML deployment, serving infrastructure, and production ML operations.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`
  - *Capabilities*: Model serving, optimization for inference, CI/CD for ML, monitoring, and federated learning.

## Core Skills

### [Model Deployment Serving](../model-deployment-serving/SKILL.md)
Serving infrastructure: FastAPI, TorchServe, Triton, BentoML, and REST/gRPC endpoints.

### [Model Optimization Deployment](../model-optimization-deployment/SKILL.md)
Post-training inference optimization: quantization, pruning, ONNX export, TensorRT, and mobile deployment. For training-time scaling (distributed training, mixed precision), see the `deep-learning-hub`.

### [ML Engineering Production](../ml-engineering-production/SKILL.md)
Production ML engineering: type-safe code, testing, data pipelines, monitoring, and drift detection for maintainable ML systems.

### [ML Pipeline Workflow](../ml-pipeline-workflow/SKILL.md)
End-to-end ML pipelines: Airflow, Prefect, Metaflow, and automated retraining workflows.

### [DevOps ML Infrastructure](../devops-ml-infrastructure/SKILL.md)
ML infrastructure: Docker, Kubernetes, GPU provisioning, and cloud ML platforms (AWS/GCP/Azure).

### [Federated Learning](../federated-learning/SKILL.md)
Privacy-preserving ML: federated averaging, differential privacy, secure aggregation, and PySyft.

## Routing Decision Tree

```
What is the ML deployment task?
|
+-- Expose a model via an API endpoint?
|   --> model-deployment-serving
|
+-- Reduce model size / latency for inference?
|   --> model-optimization-deployment
|
+-- Production ML code quality / monitoring / drift?
|   --> ml-engineering-production
|
+-- Automate training / retraining pipelines?
|   --> ml-pipeline-workflow
|
+-- Container / Kubernetes / cloud infrastructure?
|   --> devops-ml-infrastructure
|
+-- Privacy-preserving / distributed training?
    --> federated-learning
```

## Skill Selection Table

| Task | Skill |
|------|-------|
| REST/gRPC serving, Triton | `model-deployment-serving` |
| Quantization, ONNX, TensorRT | `model-optimization-deployment` |
| ML code quality, drift, monitoring | `ml-engineering-production` |
| Airflow, Prefect, retraining | `ml-pipeline-workflow` |
| Docker, K8s, GPU cloud | `devops-ml-infrastructure` |
| FedAvg, DP, secure aggregation | `federated-learning` |

## Checklist

- [ ] Validate model accuracy after optimization (quantization/pruning) matches baseline
- [ ] Load-test serving endpoint before production release
- [ ] Implement model versioning and rollback capability in serving layer
- [ ] Set up data drift and prediction drift monitoring post-deployment
- [ ] Use containerized environments to ensure dev/prod parity
- [ ] Define SLOs (latency p99, throughput) before choosing serving stack
- [ ] Automate retraining trigger conditions in pipeline workflow
- [ ] Audit federated learning implementations for privacy guarantees before use
