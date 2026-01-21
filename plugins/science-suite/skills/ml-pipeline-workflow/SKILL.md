---
name: ml-pipeline-workflow
version: "1.0.7"
maturity: "5-Expert"
specialization: MLOps Pipeline Orchestration
description: Build end-to-end MLOps pipelines with Airflow, Dagster, Kubeflow, or Prefect for data preparation, training, validation, and deployment. Use when creating DAG definitions, workflow configs, or orchestrating ML lifecycle stages.
---

# ML Pipeline Workflow

End-to-end MLOps pipeline orchestration from data to deployment.

---

## Orchestration Tools

| Tool | Best For | Key Feature |
|------|----------|-------------|
| Apache Airflow | DAG-based workflows | Mature, widely adopted |
| Dagster | Asset-based pipelines | Software-defined assets |
| Kubeflow Pipelines | K8s-native ML | Component reusability |
| Prefect | Modern dataflow | Dynamic workflows |

---

## Pipeline Stages

| Stage | Tasks | Key Outputs |
|-------|-------|-------------|
| Data Ingestion | Extract, validate sources | Raw datasets |
| Data Preparation | Clean, transform, feature eng | Processed features |
| Model Training | Train, tune hyperparameters | Model artifacts |
| Model Validation | Test, compare, approve | Validation report |
| Deployment | Package, serve, monitor | Production endpoint |

---

## DAG Pattern

```yaml
stages:
  - name: data_preparation
    dependencies: []
  - name: model_training
    dependencies: [data_preparation]
  - name: model_evaluation
    dependencies: [model_training]
  - name: model_deployment
    dependencies: [model_evaluation]
    condition: evaluation_passed
```

---

## Integration Points

| Category | Tools |
|----------|-------|
| Experiment Tracking | MLflow, Weights & Biases, TensorBoard |
| Data Versioning | DVC, Delta Lake, LakeFS |
| Data Validation | Great Expectations, TFX Data Validation |
| Model Registry | MLflow Registry, Vertex AI, SageMaker |
| Serving | TorchServe, TF Serving, Triton, KServe |

---

## Deployment Strategies

| Strategy | Use Case | Risk |
|----------|----------|------|
| Shadow | Test in production traffic | None |
| Canary | Gradual rollout (5% → 100%) | Low |
| Blue-Green | Instant switchover | Medium |
| A/B Testing | Compare model versions | Low |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Modularity | Each stage independently testable |
| Idempotency | Re-running stages is safe |
| Observability | Log metrics at every stage |
| Versioning | Track data, code, and model versions |
| Failure Handling | Retry logic and alerting |
| Validation Gates | Block deployment on failures |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Monolithic pipelines | Break into modular stages |
| No data versioning | Use DVC or Delta Lake |
| Missing validation | Add quality gates before deploy |
| Manual deployments | Automate with CI/CD triggers |
| No rollback plan | Implement automated rollback |

---

## Troubleshooting

| Issue | Debug Steps |
|-------|-------------|
| Pipeline failures | Check dependencies, data availability |
| Training instability | Review hyperparams, data quality |
| Deployment issues | Validate artifacts, serving config |
| Performance degradation | Monitor drift, retrain triggers |

---

## Checklist

- [ ] Pipeline stages defined with clear dependencies
- [ ] Data validation at ingestion
- [ ] Experiment tracking configured
- [ ] Model versioning enabled
- [ ] Validation gates before deployment
- [ ] Deployment strategy selected
- [ ] Monitoring and alerting configured
- [ ] Rollback mechanism tested

---

**Version**: 1.0.5
