# Agent Selection Guide

**Quick reference for choosing the right agent for your task**

---

## 🎯 Quick Decision Tree

### Start Here: What are you working on?

1. **Machine Learning / AI** → [ML & AI Section](#ml--ai-agents)
2. **Scientific Computing / Physics** → [Scientific Computing Section](#scientific-computing-agents)
3. **Data Engineering / Analytics** → [Data Section](#data-agents)
4. **Software Engineering** → [Software Engineering Section](#software-engineering-agents)
5. **Infrastructure / DevOps** → [Infrastructure Section](#infrastructure-agents)
6. **Documentation / Quality** → [Documentation & Quality Section](#documentation--quality-agents)

---

## ML & AI Agents

### Decision Tree

```
Working with ML/AI?
│
├─ Is it about JAX specifically?
│  ├─ YES → Is it physics simulation?
│  │  ├─ YES → jax-scientific-domains
│  │  └─ NO → Is it core JAX programming (jit/vmap/pmap)?
│  │     ├─ YES → jax-pro
│  │     └─ NO → neural-architecture-engineer (then delegates to jax-pro)
│  │
│  └─ NO → Is it neural network architecture design?
│     ├─ YES → neural-architecture-engineer
│     └─ NO → Is it MLOps/deployment?
│        ├─ YES → ml-pipeline-coordinator
│        └─ NO → Is it classical ML (sklearn, XGBoost)?
│           ├─ YES → ml-pipeline-coordinator
│           └─ NO → Is it AI system architecture (LLMs, MCP)?
│              └─ YES → ai-systems-architect
```

### Agent Descriptions

**ml-pipeline-coordinator** (formerly ai-ml-specialist)
- **Use for:** Classical ML, MLOps, experiment tracking, deployment
- **Keywords:** scikit-learn, XGBoost, MLflow, W&B, Docker, Kubernetes
- **Delegates:** JAX → jax-pro, architectures → neural-architecture-engineer

**neural-architecture-engineer** (formerly neural-networks)
- **Use for:** Neural network design, training strategies, debugging
- **Keywords:** transformers, CNNs, RNNs, BERT, GPT, architecture
- **Delegates:** JAX optimization → jax-pro, deployment → ml-pipeline-coordinator

**jax-pro**
- **Use for:** Core JAX programming, Flax NNX, Optax, NumPyro
- **Keywords:** jit, vmap, pmap, pytrees, functional programming
- **Delegates:** Architecture → neural-architecture-engineer, physics → jax-scientific-domains

**jax-scientific-domains**
- **Use for:** JAX for physics (quantum, CFD, MD, PINNs)
- **Keywords:** JAX-MD, JAX-CFD, physics-informed ML, differentiable physics
- **Delegates:** Core JAX → jax-pro, traditional MD → simulation-expert

**ai-systems-architect**
- **Use for:** LLM architecture, MCP integration, multi-model orchestration
- **Keywords:** LLM serving, prompt engineering, AI platform design
- **Different from ml-pipeline-coordinator:** Infrastructure design vs model training

---

## Scientific Computing Agents

### Decision Tree

```
Working on scientific computing?
│
├─ Is it JAX-based physics?
│  └─ YES → jax-scientific-domains
│
├─ Is it molecular dynamics?
│  ├─ With JAX → jax-scientific-domains
│  └─ With LAMMPS/GROMACS → simulation-expert
│
├─ Is it statistical physics / correlation functions?
│  └─ YES → correlation-function-expert
│
├─ Is it legacy code modernization?
│  └─ YES → scientific-code-adoptor
│
└─ Is it general HPC/numerical methods?
   └─ YES → hpc-numerical-coordinator
```

### Agent Descriptions

**hpc-numerical-coordinator** (formerly scientific-computing)
- **Use for:** General numerical methods, HPC strategy, Python vs Julia
- **Keywords:** ODE/PDE solvers, parallel computing, GPU acceleration
- **Delegates:** MD → simulation-expert, correlation → correlation-function-expert

**simulation-expert**
- **Use for:** Molecular dynamics, atomistic modeling, materials prediction
- **Keywords:** LAMMPS, GROMACS, HOOMD-blue, DPD, force fields
- **Specialized:** Traditional MD tools, not JAX-based

**correlation-function-expert**
- **Use for:** Statistical physics, correlation analysis
- **Keywords:** FFT analysis, pair correlations, structure factors
- **Specialized:** Very specific niche, not for general physics

**jax-scientific-domains**
- **Use for:** Physics simulations with JAX
- **Keywords:** Quantum computing, CFD, JAX-MD, PINNs
- **Different from hpc-numerical-coordinator:** JAX-specific vs general

**scientific-code-adoptor**
- **Use for:** Legacy code migration
- **Keywords:** Fortran → Python, MATLAB → JAX, numerical accuracy
- **Specialized:** Cross-language modernization only

---

## Data Agents

### Decision Tree

```
Working with data?
│
├─ Is it machine learning?
│  └─ YES → ml-pipeline-coordinator
│
├─ Is it PostgreSQL-specific or Airflow DAGs?
│  └─ YES → database-workflow-engineer
│
└─ Is it ETL/analytics/visualization?
   └─ YES → data-engineering-coordinator
```

### Agent Descriptions

**data-engineering-coordinator** (formerly data-professional)
- **Use for:** ETL/ELT, analytics, dashboards, data quality
- **Keywords:** Spark, Kafka, Tableau, Power BI, statistical analysis
- **Delegates:** PostgreSQL/Airflow → database-workflow-engineer, ML → ml-pipeline-coordinator

**database-workflow-engineer**
- **Use for:** PostgreSQL optimization, Airflow workflows, dbt
- **Keywords:** Database architecture, SQL tuning, workflow orchestration
- **Specialized:** PostgreSQL + Airflow focus

---

## Software Engineering Agents

### Decision Tree

```
Software engineering task?
│
├─ Is it CLI tools?
│  └─ YES → command-systems-engineer
│
├─ Is it testing/QA?
│  └─ YES → code-quality-master
│
├─ Is it full-stack web development?
│  └─ YES → fullstack-developer
│
└─ Is it high-level architecture?
   └─ YES → systems-architect
```

### Agent Descriptions

**fullstack-developer**
- **Use for:** End-to-end web features, database to UI
- **Keywords:** Backend, frontend, authentication, deployment
- **Generic:** No specific tech stack

**systems-architect**
- **Use for:** Architecture design, API strategy, scalability
- **Keywords:** Patterns, technology evaluation, system design
- **High-level:** Strategy vs implementation

**command-systems-engineer**
- **Use for:** CLI tool development, developer automation
- **Keywords:** Command-line, interactive prompts, workflow tools
- **Specialized:** Command development focus

**code-quality-master**
- **Use for:** Testing, QA, accessibility
- **Keywords:** pytest, Jest, test strategies, quality engineering
- **Focus:** Quality assurance, not development

---

## Infrastructure Agents

### Agent Descriptions

**devops-security-engineer**
- **Use for:** CI/CD, Kubernetes, Terraform, security, compliance
- **Keywords:** DevSecOps, infrastructure automation, resilience
- **Delegates:** ML infrastructure → ml-pipeline-coordinator

---

## Documentation & Quality Agents

### Agent Descriptions

**documentation-architect**
- **Use for:** Technical writing, API docs, tutorials
- **Keywords:** Sphinx, MkDocs, knowledge management, accessibility
- **Focus:** Documentation creation, not code

**visualization-interface**
- **Use for:** Scientific visualization, data viz, UX design
- **Keywords:** D3.js, Plotly, AR/VR, visual narratives
- **Specialized:** Visualization focus

**research-intelligence**
- **Use for:** Research methodology, literature analysis, insights
- **Keywords:** Evidence-based research, trend forecasting
- **Specialized:** Research focus, not implementation

---

## Coordination Agents

### Agent Descriptions

**multi-agent-orchestrator**
- **Use for:** Coordinating multiple agents, complex workflows
- **Keywords:** Team assembly, task allocation, distributed systems
- **Meta-role:** Manages other agents

---

## Common Scenarios

### Scenario: "Build a neural network in JAX for image classification"

**Path:** neural-architecture-engineer → jax-pro
1. **neural-architecture-engineer** designs the CNN architecture
2. **jax-pro** implements it efficiently in JAX with jit/vmap
3. **ml-pipeline-coordinator** handles deployment

### Scenario: "Optimize my scientific Python code for GPU"

**Question:** Is it ML code or numerical code?
- **ML code** → ml-pipeline-coordinator → jax-pro
- **Numerical code** → hpc-numerical-coordinator

### Scenario: "Implement a physics-informed neural network"

**Path:** jax-scientific-domains (leads)
- Most specific for PINNs
- Delegates JAX optimization to jax-pro if needed
- Delegates architecture design to neural-architecture-engineer if needed

### Scenario: "Set up MLflow experiment tracking"

**Agent:** ml-pipeline-coordinator
- MLOps is core expertise
- Can integrate with data-engineering-coordinator for data pipelines

### Scenario: "Build REST API for ML model serving"

**Path:** ml-pipeline-coordinator → fullstack-developer (if frontend needed)
- ml-pipeline-coordinator handles ML model serving
- fullstack-developer handles web interface if needed

### Scenario: "Migrate legacy Fortran code to Python"

**Agent:** scientific-code-adoptor
- Specialized in cross-language migration
- Preserves numerical accuracy

### Scenario: "Analyze correlation functions from MD simulation"

**Agent:** correlation-function-expert
- Very specific niche
- Uses FFT methods, statistical physics

### Scenario: "Build data pipeline with Kafka and Spark"

**Agent:** data-engineering-coordinator
- ETL/ELT focus
- Delegates ML to ml-pipeline-coordinator if needed

### Scenario: "Optimize PostgreSQL queries for Airflow DAGs"

**Agent:** database-workflow-engineer
- PostgreSQL + Airflow specialist
- More focused than data-engineering-coordinator

---

## Renamed Agents Cheat Sheet

| Old Name | New Name | Why |
|----------|----------|-----|
| `ai-ml-specialist` | `ml-pipeline-coordinator` | Clearer coordination role |
| `neural-networks` | `neural-architecture-engineer` | Role-based name, not technology |
| `scientific-computing` | `hpc-numerical-coordinator` | More specific, coordinator role |
| `data-professional` | `data-engineering-coordinator` | Clearer focus, coordination role |

---

## Key Principles

### 1. Coordinators Delegate

Coordinator agents (with "coordinator" in the name) route to specialists:
- `ml-pipeline-coordinator` → `jax-pro`, `neural-architecture-engineer`
- `hpc-numerical-coordinator` → `simulation-expert`, `correlation-function-expert`
- `data-engineering-coordinator` → `database-workflow-engineer`, `ml-pipeline-coordinator`

### 2. Specialists Don't Overlap

Each specialist has a specific niche:
- `jax-pro` = JAX programming, not architecture
- `neural-architecture-engineer` = Architecture, not JAX optimization
- `simulation-expert` = Traditional MD, not JAX
- `jax-scientific-domains` = JAX + physics, not general JAX

### 3. When in Doubt

- **Complex task spanning multiple domains** → Start with coordinator agent
- **Very specific technical task** → Start with specialist
- **Unclear boundaries** → Use `multi-agent-orchestrator`

---

## Quick Reference Table

| Task Type | Agent | Second Choice |
|-----------|-------|---------------|
| Classical ML | ml-pipeline-coordinator | - |
| Neural architecture | neural-architecture-engineer | - |
| JAX programming | jax-pro | - |
| JAX physics | jax-scientific-domains | jax-pro |
| Traditional MD | simulation-expert | - |
| Correlation functions | correlation-function-expert | - |
| HPC strategy | hpc-numerical-coordinator | - |
| ETL pipelines | data-engineering-coordinator | - |
| PostgreSQL/Airflow | database-workflow-engineer | data-engineering-coordinator |
| MLOps | ml-pipeline-coordinator | - |
| Testing/QA | code-quality-master | - |
| Documentation | documentation-architect | - |
| CLI tools | command-systems-engineer | - |
| Visualization | visualization-interface | - |
| DevOps/Security | devops-security-engineer | - |
| Research | research-intelligence | - |
| Multi-agent tasks | multi-agent-orchestrator | - |

---

## Related Documentation

- **`AGENT_SYSTEM.md`** - Complete technical reference for automatic command-based agent triggering system

**Note:** This guide helps you **manually select agents** for any task, while AGENT_SYSTEM.md documents the **automatic agent triggering** system used by slash commands. Use this guide when working outside the command system or when you need to understand which agent to delegate to.

---

**Last Updated:** 2025-10-05 (Phase 2 complete)
