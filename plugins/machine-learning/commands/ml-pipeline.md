---
version: 1.0.3
command: /ml-pipeline
description: Design and implement production-ready ML pipelines with multi-agent MLOps orchestration
execution_modes:
  quick:
    duration: "2-3 days"
    description: "MVP ML pipeline with core components"
    scope: "Basic data pipeline, model training, simple deployment"
    agents: ["data-scientist", "ml-engineer", "mlops-engineer"]
    phases: ["Data Analysis & Features", "Model Development", "Basic Deployment"]

  standard:
    duration: "1-2 weeks"
    description: "Full production pipeline with monitoring and testing"
    scope: "Complete data pipeline, optimized training, production serving, monitoring"
    agents: ["data-scientist", "ml-engineer", "python-pro", "mlops-engineer", "observability-engineer"]
    phases: ["Data & Requirements", "Model Development & Testing", "Production Deployment", "Monitoring"]

  enterprise:
    duration: "3-4 weeks"
    description: "Complete MLOps platform with Kubernetes, advanced monitoring, and governance"
    scope: "Enterprise data infrastructure, distributed training, K8s orchestration, comprehensive observability"
    agents: ["data-engineer", "data-scientist", "ml-engineer", "python-pro", "mlops-engineer", "kubernetes-architect", "observability-engineer"]
    phases: ["Data Infrastructure", "Model Development & Optimization", "K8s Production Deployment", "Advanced Monitoring & Governance"]

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

## Thinking

This workflow orchestrates specialized agents to create end-to-end ML pipelines following modern MLOps best practices:

**Core Principles**:
- **Phase-based coordination**: Clear handoffs between agents, each phase builds on previous outputs
- **Modern MLOps stack**: MLflow/W&B experiments, Feast/Tecton features, KServe/Seldon serving
- **Production-first**: Every component designed for scale, monitoring, and reliability from day one
- **Reproducibility**: Full versioning for data, models, code, and infrastructure
- **Continuous improvement**: Automated retraining, A/B testing, drift detection

**Multi-Agent Specialization**:
- **data-engineer**: Data infrastructure, ingestion, quality, and storage architecture
- **data-scientist**: Feature engineering, model design, experiments, and evaluation
- **ml-engineer**: Training pipelines, hyperparameter tuning, distributed training, model registry
- **python-pro**: Code optimization, testing frameworks, production refactoring (optional: python-development plugin)
- **mlops-engineer**: Model serving, CI/CD pipelines, deployment automation, IaC
- **kubernetes-architect**: K8s orchestration, GPU scheduling, autoscaling (optional: cicd-automation plugin)
- **observability-engineer**: Monitoring, drift detection, alerting, cost tracking (optional: observability-monitoring plugin)

**For detailed methodology, see**: [MLOps Methodology](../docs/ml-pipeline/mlops-methodology.md)

---

## Execution Mode Selection

<AskUserQuestion>
questions:
  - question: "Which execution mode best fits your project timeline and requirements?"
    header: "Execution Mode"
    multiSelect: false
    options:
      - label: "Quick (2-3 days)"
        description: "MVP pipeline with core ML components: basic data pipeline, model training, simple deployment. Best for prototypes and proof-of-concepts."

      - label: "Standard (1-2 weeks)"
        description: "Full production pipeline with monitoring: complete data processing, optimized training, production serving, basic monitoring. Best for production systems."

      - label: "Enterprise (3-4 weeks)"
        description: "Complete MLOps platform: enterprise data infrastructure, Kubernetes orchestration, distributed training, comprehensive monitoring and governance. Best for large-scale production ML."
</AskUserQuestion>

---

## Agent Reference Table

| Agent | Plugin | Role | Key Responsibilities | Execution Modes |
|-------|--------|------|---------------------|-----------------|
| **data-engineer** | machine-learning | Data Infrastructure | Ingestion pipelines, data quality, versioning, storage optimization | Enterprise |
| **data-scientist** | machine-learning | Feature & Model Design | Feature engineering, model selection, experiments, evaluation metrics | All modes |
| **ml-engineer** | machine-learning | Training Infrastructure | Training pipelines, hyperparameter tuning, distributed training, model registry | All modes |
| **python-pro** | python-development* | Code Optimization | Production refactoring, testing, performance optimization, code quality | Standard, Enterprise |
| **mlops-engineer** | machine-learning | Deployment & Serving | Model serving, CI/CD, deployment automation, infrastructure as code | All modes |
| **kubernetes-architect** | cicd-automation* | K8s Orchestration | Cluster design, GPU scheduling, autoscaling, service mesh | Enterprise |
| **observability-engineer** | observability-monitoring* | Monitoring & Alerts | Drift detection, performance monitoring, alerting, cost tracking | Standard, Enterprise |

\* *Optional cross-plugin agents - graceful degradation if not available*

---

## Phase 1: Data & Requirements Analysis

**Goal**: Establish data infrastructure and feature requirements

**Quick Mode**: data-scientist handles basic data analysis and feature design

**Standard/Enterprise Mode**: data-engineer sets up comprehensive data infrastructure

<Task>
subagent_type: data-engineer
prompt: |
  [Enterprise mode only - skip if agent not available]

  Design scalable data infrastructure for ML system: $ARGUMENTS

  Deliverables:
  1. **Data Ingestion & Quality**: Source audit, schema validation (Pydantic/Great Expectations), CDC strategies, data versioning (DVC/lakeFS)
  2. **Storage Architecture**: Bronze/Silver/Gold layers, partitioning strategy, retention policies, cost optimization
  3. **Data Quality Framework**: Profiling, anomaly detection, lineage tracking, quality gates

  **See detailed specs**: [Pipeline Phases Guide](../docs/ml-pipeline/pipeline-phases.md#phase-1-data-infrastructure)

  Provide implementation code for ingestion pipelines, quality checks, and storage configuration.
</Task>

<Task>
subagent_type: data-scientist
prompt: |
  Design features and model requirements for: $ARGUMENTS

  [Enterprise mode: Build on data architecture from data-engineer]
  [Quick/Standard: Include basic data analysis]

  Deliverables:
  1. **Feature Engineering**: Transformation specs, feature store schema (Feast/Tecton), validation rules
  2. **Model Requirements**: Algorithm selection, performance baselines, evaluation criteria
  3. **Experiment Design**: Hypothesis, success metrics, A/B testing methodology

  **See detailed specs**: [Pipeline Phases Guide](../docs/ml-pipeline/pipeline-phases.md#phase-1-requirements-analysis)

  Include feature transformation code and statistical validation logic.
</Task>

**External Documentation**: [Pipeline Phases Guide - Phase 1](../docs/ml-pipeline/pipeline-phases.md)

---

## Phase 2: Model Development & Training

**Goal**: Implement production-ready training pipeline

<Task>
subagent_type: ml-engineer
prompt: |
  Implement training pipeline based on: {phase1.data-scientist.output}
  [Enterprise mode: Using data pipeline from {phase1.data-engineer.output}]

  Build comprehensive training system:
  1. **Training Pipeline**: Modular code, hyperparameter optimization (Optuna/Ray Tune), distributed training (Horovod/PyTorch DDP)
  2. **Experiment Tracking**: MLflow/W&B integration, metrics, artifact management
  3. **Model Registry**: Versioning, tagging, promotion workflows (dev→staging→prod), rollback procedures

  **See detailed specs**: [Pipeline Phases Guide](../docs/ml-pipeline/pipeline-phases.md#phase-2-model-development)

  Provide complete training code with configuration management.
</Task>

<Task>
subagent_type: python-pro
prompt: |
  [Standard/Enterprise modes - skip if python-development plugin not available]

  Optimize and productionize ML code from: {phase2.ml-engineer.output}

  Focus areas:
  1. **Code Quality**: Production standards, error handling, structured logging, reusable components
  2. **Performance**: Profile and optimize bottlenecks, caching, data loading optimization
  3. **Testing**: Unit tests for transforms, integration tests for pipeline, model quality tests

  **See detailed specs**: [Best Practices - Code Quality](../docs/ml-pipeline/best-practices.md#code-quality)

  Deliver production-ready code with comprehensive test coverage.
</Task>

**External Documentation**: [Pipeline Phases Guide - Phase 2](../docs/ml-pipeline/pipeline-phases.md)

---

## Phase 3: Production Deployment & Serving

**Goal**: Deploy models to production with CI/CD automation

<Task>
subagent_type: mlops-engineer
prompt: |
  Design production deployment for models from: {phase2.ml-engineer.output}
  [Standard/Enterprise: With optimized code from {phase2.python-pro.output}]

  Implementation:
  1. **Model Serving**: REST/gRPC APIs (FastAPI/TorchServe), batch pipelines (Airflow/Kubeflow), streaming (Kafka/Kinesis)
  2. **Deployment Strategies**: Blue-green, canary releases, shadow deployments, A/B testing infrastructure
  3. **CI/CD Pipeline**: GitHub Actions workflows, automated testing gates, model validation, ArgoCD GitOps
  4. **Infrastructure as Code**: Terraform modules, Helm charts, Docker multi-stage builds, secret management

  **See detailed specs**:
  - [Deployment Strategies Guide](../docs/ml-pipeline/deployment-strategies.md)
  - [Pipeline Phases - Phase 3](../docs/ml-pipeline/pipeline-phases.md#phase-3-deployment)

  Provide deployment configuration and automation scripts.
</Task>

<Task>
subagent_type: kubernetes-architect
prompt: |
  [Enterprise mode only - skip if cicd-automation plugin not available]

  Design Kubernetes infrastructure for ML workloads from: {phase3.mlops-engineer.output}

  Requirements:
  1. **Workload Orchestration**: Training job scheduling (Kubeflow), GPU allocation, spot instances, resource quotas
  2. **Serving Infrastructure**: HPA/VPA autoscaling, KEDA event-driven scaling, Istio service mesh
  3. **Storage & Data Access**: PVC for training data, model artifact storage, distributed feature stores, cache layers

  **See detailed specs**: [Pipeline Phases - K8s Infrastructure](../docs/ml-pipeline/pipeline-phases.md#phase-3-kubernetes)

  Provide Kubernetes manifests and Helm charts.
</Task>

**External Documentation**: [Deployment Strategies Guide](../docs/ml-pipeline/deployment-strategies.md)

---

## Phase 4: Monitoring & Continuous Improvement

**Goal**: Comprehensive observability and automated improvement

<Task>
subagent_type: observability-engineer
prompt: |
  [Standard/Enterprise modes - skip if observability-monitoring plugin not available]

  Implement monitoring for ML system deployed in: {phase3.mlops-engineer.output}
  [Enterprise: Using K8s infrastructure from {phase3.kubernetes-architect.output}]

  Monitoring framework:
  1. **Model Performance**: Accuracy tracking, latency/throughput metrics, feature importance shifts, business KPI correlation
  2. **Drift Detection**: Statistical drift (KS test, PSI), concept drift, feature distribution tracking, automated alerts
  3. **System Observability**: Prometheus metrics, Grafana dashboards, distributed tracing (Jaeger/Zipkin), log aggregation
  4. **Alerting & Automation**: PagerDuty/Opsgenie integration, retraining triggers, incident runbooks
  5. **Cost Tracking**: Resource utilization, cost allocation per model, optimization recommendations

  **See detailed specs**:
  - [Monitoring Frameworks Guide](../docs/ml-pipeline/monitoring-frameworks.md)
  - [Pipeline Phases - Phase 4](../docs/ml-pipeline/pipeline-phases.md#phase-4-monitoring)

  Deliver monitoring configuration, dashboards, and alert rules.
</Task>

**External Documentation**: [Monitoring Frameworks Guide](../docs/ml-pipeline/monitoring-frameworks.md)

---

## Configuration Options

**Execution Mode** (via AskUserQuestion):
- `quick`: 2-3 days MVP with core agents only
- `standard`: 1-2 weeks full pipeline with monitoring
- `enterprise`: 3-4 weeks complete MLOps platform with K8s

**Technology Stack** (customize per project):
- **Experiment Tracking**: mlflow | wandb | neptune | clearml
- **Feature Store**: feast | tecton | databricks | custom
- **Serving Platform**: kserve | seldon | torchserve | triton | bentoml
- **Orchestration**: kubeflow | airflow | prefect | dagster
- **Cloud Provider**: aws | azure | gcp | multi-cloud
- **Deployment Mode**: realtime | batch | streaming | hybrid
- **Monitoring Stack**: prometheus | datadog | newrelic | custom

**Library-Specific Documentation** (optional):
- Use `--docs-url=<library-docs-url>` to integrate library-specific ML framework documentation
- Example: `/ml-pipeline "customer churn prediction" --docs-url=https://scikit-learn.org/stable/`

---

## Success Criteria

**See comprehensive metrics**: [Success Criteria Guide](../docs/ml-pipeline/success-criteria.md)

**Summary**:
1. **Data Pipeline**: <0.1% quality issues, <1s feature latency, 99.9% validation pass rate
2. **Model Performance**: Meet baselines, <5% degradation before retraining, statistical A/B test significance
3. **Operational Excellence**: 99.9% uptime, <200ms p99 latency, <5 min automated rollback, <1 min alert time
4. **Development Velocity**: <1 hour commit-to-prod, parallel experiments, reproducible runs, self-service deployment
5. **Cost Efficiency**: <20% infrastructure waste, optimized resource allocation, >60% spot instance utilization

---

## Final Deliverables

Upon completion, receive:
- ✅ End-to-end ML pipeline with full automation
- ✅ Production-ready infrastructure as code (Terraform/Helm)
- ✅ CI/CD pipelines for continuous deployment
- ✅ Comprehensive monitoring and alerting system
- ✅ Complete documentation and operational runbooks
- ✅ Cost optimization and scaling strategies
- ✅ Disaster recovery and rollback procedures

**For best practices**: [Production Readiness Guide](../docs/ml-pipeline/best-practices.md)

**For methodology deep-dive**: [MLOps Methodology Guide](../docs/ml-pipeline/mlops-methodology.md)
