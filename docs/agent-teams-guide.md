# Agent Teams Guide for MyClaude Plugin Suites

> 10 ready-to-use team configurations with 20 variants, leveraging 25 MyClaude agents + 18 official plugin agents across 4 suites.
>
> **v3.3.0:** Consolidated from 27 teams to 10 teams with a variant system (`--var MODE=x`). Zero function loss — every capability from every absorbed team is reachable via a variant. 20 aliases provide backward compatibility.

## Prerequisites

Enable agent teams (experimental) in your settings:

```json
// ~/.claude/settings.json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

## Agent Inventory

| Suite | Agent | `subagent_type` | Specialization |
|-------|-------|-----------------|----------------|
| **agent-core** | orchestrator | `agent-core:orchestrator` | Workflow coordination, team assembly |
| | reasoning-engine | `agent-core:reasoning-engine` | Chain-of-Thought, prompt design |
| | context-specialist | `agent-core:context-specialist` | Memory, context management |
| **dev** | software-architect | `dev-suite:software-architect` | Backend, API, microservices |
| | app-developer | `dev-suite:app-developer` | Frontend, mobile, React/Next.js |
| | systems-engineer | `dev-suite:systems-engineer` | C/C++/Rust, CLI tools, low-level |
| | devops-architect | `dev-suite:devops-architect` | Cloud, K8s, Terraform |
| | sre-expert | `dev-suite:sre-expert` | Observability, SLOs, incidents |
| | automation-engineer | `dev-suite:automation-engineer` | CI/CD, GitHub Actions, Git |
| | debugger-pro | `dev-suite:debugger-pro` | Root cause analysis, log correlation |
| | documentation-expert | `dev-suite:documentation-expert` | Tech writing, API docs |
| | quality-specialist | `dev-suite:quality-specialist` | Code review, security audit, testing |
| **science** | jax-pro | `science-suite:jax-pro` | JAX, NumPyro, Bayesian inference |
| | neural-network-master | `science-suite:neural-network-master` | Deep learning, Transformers, Flax |
| | ml-expert | `science-suite:ml-expert` | Scikit-learn, MLOps, XGBoost |
| | ai-engineer | `science-suite:ai-engineer` | LLM apps, RAG, agents |
| | python-pro | `science-suite:python-pro` | Python systems, packaging |
| | research-expert | `science-suite:research-expert` | Scientific methodology, papers |
| | simulation-expert | `science-suite:simulation-expert` | Physics simulation, MD |
| | statistical-physicist | `science-suite:statistical-physicist` | Stat mech, stochastic dynamics |
| | nonlinear-dynamics-expert | `science-suite:nonlinear-dynamics-expert` | Bifurcation, chaos, SINDy/UDE |
| | julia-pro | `science-suite:julia-pro` | Julia HPC, SciML |
| | julia-ml-hpc | `science-suite:julia-ml-hpc` | Julia ML/DL/HPC, Lux.jl, CUDA.jl |
| | prompt-engineer | `science-suite:prompt-engineer` | LLM optimization, eval |

### Official Plugin Agents

| Plugin | Agent | `subagent_type` | Specialization |
|--------|-------|-----------------|----------------|
| **pr-review-toolkit** | code-reviewer | `pr-review-toolkit:code-reviewer` | Style, guidelines, best practices |
| | silent-failure-hunter | `pr-review-toolkit:silent-failure-hunter` | Error swallowing, bad fallbacks |
| | code-simplifier | `pr-review-toolkit:code-simplifier` | Clarity, maintainability |
| | comment-analyzer | `pr-review-toolkit:comment-analyzer` | Comment accuracy, rot detection |
| | pr-test-analyzer | `pr-review-toolkit:pr-test-analyzer` | Test coverage gaps |
| | type-design-analyzer | `pr-review-toolkit:type-design-analyzer` | Type invariants, encapsulation |
| **feature-dev** | code-explorer | `feature-dev:code-explorer` | Execution path tracing |
| | code-architect | `feature-dev:code-architect` | Feature architecture blueprints |
| | code-reviewer | `feature-dev:code-reviewer` | Bug, logic, security review |
| **code-simplifier** | code-simplifier | `code-simplifier:code-simplifier` | Code clarity and refinement |
| **agent-sdk-dev** | agent-sdk-verifier-ts | `agent-sdk-dev:agent-sdk-verifier-ts` | TS Agent SDK validation |
| | agent-sdk-verifier-py | `agent-sdk-dev:agent-sdk-verifier-py` | Python Agent SDK validation |
| **plugin-dev** | agent-creator | `plugin-dev:agent-creator` | Claude Code agent generation |
| | skill-reviewer | `plugin-dev:skill-reviewer` | Skill quality review |
| | plugin-validator | `plugin-dev:plugin-validator` | Plugin structure validation |
| **superpowers** | code-reviewer | `superpowers:code-reviewer` | Plan adherence review |
| **hookify** | conversation-analyzer | `hookify:conversation-analyzer` | Behavior analysis for hooks |
| **product-tracking-skills** | tracking-watchdog | `product-tracking-skills:tracking-watchdog` | Proactive tracking coverage monitor |

---

## Quick Reference

| # | Team | Variants | Best For | Agents |
|---|------|----------|----------|--------|
| 1 | feature-dev | -- | Feature build + review | 4 |
| 2 | debug | triage, gui, numerical, schema, incident | All debugging + incident response | 2-4 |
| 3 | quality-gate | security, full | PR review + security audit | 4 |
| 4 | api-infra | infra, config | APIs + cloud + CI/CD + config | 3-4 |
| 5 | sci-compute | bayesian, julia-sciml, julia-ml, dynamics, md-sim, desktop, reproduce | All scientific computing | 4 |
| 6 | modernize | -- | Legacy migration + refactoring | 4 |
| 7 | ai-engineering | multi-agent | LLM apps + RAG + multi-agent | 4 |
| 8 | ml-deploy | data, perf | Model deploy + data + performance | 4 |
| 9 | docs-publish | research | Documentation + reproducibility | 4 |
| 10 | plugin-forge | -- | Claude Code extensions | 4 |

---

## Team 1: Feature Development

Build new features end-to-end with a design-first pipeline and automated review gate.

### Agents

| Role | Agent | Suite | Responsibility |
|------|-------|-------|----------------|
| architect | `feature-dev:code-architect` | feature-dev | Analyze codebase, produce implementation blueprint |
| builder | `dev-suite:app-developer` | dev-suite | Implement frontend components |
| backend | `dev-suite:software-architect` | dev-suite | Implement backend services and APIs |
| reviewer | `pr-review-toolkit:code-reviewer` | pr-review-toolkit | Review all changes (read-only) |

### Workflow

`architect` analyzes the codebase and produces an implementation blueprint for approval. Then `builder` (frontend) and `backend` work in parallel following the blueprint. Finally `reviewer` checks all changes for adherence to the blueprint and best practices.

### Placeholders

`FEATURE_NAME`, `FRONTEND_STACK`, `BACKEND_STACK`, `PROJECT`

### When to Use

Any feature build -- full-stack, backend-only, or frontend-only -- where you want a design-first approach with automated review.

---

## Team 2: Debug

Consolidated debugging team covering all bug categories: general triage, GUI threading, numerical/JAX, schema drift, and production incidents.

### Agents (by variant)

| Variant | MODE | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
|---------|------|---------|---------|---------|---------|
| *default* | -- | `feature-dev:code-explorer` | `dev-suite:debugger-pro` | `science-suite:python-pro` | (auto-selected specialist) |
| triage | `triage` | `feature-dev:code-explorer` | `dev-suite:debugger-pro` | -- | -- |
| gui | `gui` | `feature-dev:code-explorer` | `dev-suite:debugger-pro` | `science-suite:python-pro` | `dev-suite:sre-expert` |
| numerical | `numerical` | `feature-dev:code-explorer` | `dev-suite:debugger-pro` | `science-suite:python-pro` | `science-suite:jax-pro` |
| schema | `schema` | `feature-dev:code-explorer` | `dev-suite:debugger-pro` | `science-suite:python-pro` | `pr-review-toolkit:type-design-analyzer` |
| incident | `incident` | `dev-suite:debugger-pro` | `dev-suite:sre-expert` | `dev-suite:devops-architect` | -- |

### Variant Details

**Default (auto-detect):** The explorer maps the architecture first, then debugger-pro and python-pro investigate in parallel. A fourth specialist is auto-selected based on symptoms: SRE for threading issues, jax-pro for numerical bugs, type-design-analyzer for schema drift.

**Triage (`--var MODE=triage`):** Lightweight 2-agent team for quick initial investigation (2-5 min). Explorer maps the execution path, then debugger-pro assesses severity (P0/P1/P2) and recommends whether to escalate to a full variant.

**GUI (`--var MODE=gui`):** Targets Qt threading bugs -- signal safety, shiboken crashes, singleton races, event loop issues. SRE investigates GIL contention, QThread lifecycle, and cross-thread signal/slot safety. Python-pro checks attribute mismatches and Protocol compliance across abstraction boundaries.

**Numerical (`--var MODE=numerical`):** Targets JAX/numerical bugs -- NaN gradients, ODE solver divergence, JIT tracing errors, shape mismatches. Jax-pro investigates XLA compilation failures, custom VJP correctness, non-JIT-safe operations, and host-device transfer overhead.

**Schema (`--var MODE=schema`):** Targets schema/type drift -- incompatible data classes, field name mismatches, serialization errors. Type-design-analyzer rates types 1-5 and recommends canonical definitions. Note: do NOT run type-analyzer and quality-specialist simultaneously (they overlap on interface contract checking).

**Incident (`--var MODE=incident`):** 3-agent parallel-hypothesis investigation for production issues. Debugger examines application code, SRE checks observability data (metrics, logs, traces), and devops-architect investigates infrastructure. Agents share findings and challenge each other's theories.

### Workflow

All variants except incident: `explorer` first (architecture mapping) then remaining agents in parallel, then `debugger-pro` synthesizes findings into a prioritized fix list.

Incident variant: all 3 agents investigate simultaneously, then synthesize into a root cause report.

**Cross-variant escalation:** If root cause points to a different domain, switch variant (e.g., "if root cause is GUI -> `--var MODE=gui`").

### Placeholders

`SYMPTOMS`, `AFFECTED_MODULES`

### Aliases

`debug-triage` -> `debug --var MODE=triage`, `debug-gui` -> `debug --var MODE=gui`, `debug-numerical` -> `debug --var MODE=numerical`, `debug-schema` -> `debug --var MODE=schema`, `incident` -> `debug --var MODE=incident`

### Signals

Required: user-provided `SYMPTOMS` (all variants) or production incident context (incident variant). Debug team requires explicit symptoms -- never auto-recommended without them.

---

## Team 3: Quality Gate

Comprehensive code review and security audit with PR-specific analyzers and codebase-wide auditing.

### Agents (by variant)

| Variant | MODE | Agents |
|---------|------|--------|
| *default* | -- | `pr-review-toolkit:silent-failure-hunter` + `pr-review-toolkit:pr-test-analyzer` + `pr-review-toolkit:type-design-analyzer` + `pr-review-toolkit:code-reviewer` |
| security | `security` | `dev-suite:software-architect` + `dev-suite:quality-specialist` + `dev-suite:sre-expert` + `dev-suite:debugger-pro` |
| full | `full` | Run default + security sequentially |

### Variant Details

**Default:** PR-focused review using the quality-gate-toolkit's 4 specialized analyzers. Each reviewer works independently on the same diff: code-reviewer checks style/guidelines/bugs, silent-failure-hunter flags swallowed errors, pr-test-analyzer identifies coverage gaps, type-design-analyzer reviews type quality. Lead collects all findings sorted by severity.

**Security (`--var MODE=security`):** Codebase-wide security and architecture audit. Software-architect assesses design patterns, SOLID, and complexity. Quality-specialist scans for OWASP Top 10 vulnerabilities. SRE reviews operational security. Debugger-pro investigates runtime security concerns. Produces a prioritized remediation plan with CVSS severity.

**Full (`--var MODE=full`):** Runs both the default PR review pass and the security audit pass sequentially.

### Placeholders

`PR_OR_BRANCH` (default), `PROJECT_PATH` (security)

### Aliases

`pr-review` -> `quality-gate`, `security` -> `quality-gate --var MODE=security`

### Signals

Required: git repo. Strong: open PR context. Auto-variant: security if missing security CI.

---

## Team 4: API & Infrastructure

API design, cloud infrastructure, CI/CD, and configuration management.

### Agents (by variant)

| Variant | MODE | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
|---------|------|---------|---------|---------|---------|
| *default* (api) | -- | `dev-suite:software-architect` | `dev-suite:app-developer` | `dev-suite:quality-specialist` | `dev-suite:sre-expert` |
| infra | `infra` | `dev-suite:devops-architect` | `dev-suite:automation-engineer` | `dev-suite:sre-expert` | -- |
| config | `config` | `dev-suite:software-architect` | `dev-suite:automation-engineer` | `dev-suite:sre-expert` | `science-suite:python-pro` |

### Variant Details

**Default (API):** Full API design-build-test-observe pipeline. Architect designs the API spec (REST/GraphQL/gRPC), app-developer implements endpoints with auth and rate limiting, quality-specialist writes contract/integration/security tests, SRE adds observability. Workflow: architect defines spec -> implementer + tester in parallel -> SRE instruments.

**Infra (`--var MODE=infra`):** Cloud infrastructure and CI/CD from scratch. Devops-architect provisions IaC (Terraform/Pulumi) with zero-trust networking. Automation-engineer builds GitHub Actions pipelines for staged deployments. SRE sets up Prometheus, Grafana, OpenTelemetry, and alerting. Workflow: devops-architect first -> automation-engineer -> SRE.

**Config (`--var MODE=config`):** Configuration management, caching, and job scheduling. Architect designs config hierarchy and cache invalidation. Automation-engineer builds GitOps deployment pipelines. SRE monitors config propagation and cache hit rates. Python-pro implements typed config models and CLI tools.

### Placeholders

`SERVICE_NAME`, `API_PROTOCOL` (api) | `PROJECT_NAME`, `CLOUD_PROVIDER` (infra) | `PROJECT_NAME` (config)

### Aliases

`api-design` -> `api-infra`, `infra-setup` -> `api-infra --var MODE=infra`

### Signals

Required (api): python | typescript | go | rust. Strong (api): `src/api/`, `routes/`, `openapi.yaml`. Required (infra): any. Strong (infra): `terraform/`, `k8s/`, `Dockerfile`.

---

## Team 5: Scientific Computing

Consolidated team for all scientific computing: JAX/ML/DL pipelines, Bayesian inference, Julia SciML, nonlinear dynamics, MD simulation, scientific desktop apps, and paper reproduction.

### Agents (by variant)

| Variant | MODE | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
|---------|------|---------|---------|---------|---------|
| *default* (jax/ml) | -- | `science-suite:jax-pro` | `science-suite:neural-network-master` | `science-suite:ml-expert` | `science-suite:research-expert` |
| bayesian | `bayesian` | `science-suite:jax-pro` | `science-suite:statistical-physicist` | `science-suite:ml-expert` | `science-suite:research-expert` |
| julia-sciml | `julia-sciml` | `science-suite:julia-pro` | `science-suite:simulation-expert` | `science-suite:jax-pro` | `science-suite:research-expert` |
| julia-ml | `julia-ml` | `science-suite:julia-ml-hpc` | `science-suite:neural-network-master` | `science-suite:ml-expert` | `science-suite:research-expert` |
| dynamics | `dynamics` | `science-suite:nonlinear-dynamics-expert` | `science-suite:jax-pro` | `science-suite:julia-pro` | `science-suite:research-expert` |
| md-sim | `md-sim` | `science-suite:simulation-expert` | `science-suite:jax-pro` | `science-suite:ml-expert` | `science-suite:research-expert` |
| desktop | `desktop` | `dev-suite:app-developer` | `science-suite:jax-pro` | `science-suite:python-pro` | `science-suite:research-expert` |
| reproduce | `reproduce` | `science-suite:research-expert` | `science-suite:python-pro` | `science-suite:jax-pro` | `science-suite:ml-expert` |

### Variant Details

**Default (JAX/ML/DL):** Build ML and deep learning pipelines with JAX. Jax-pro implements JIT-compiled kernels with vmap/pmap. Neural-network-master designs architectures with gradient flow analysis. ML-expert handles experiment tracking (W&B/MLflow), hyperparameter optimization, and model versioning. Research-expert validates methodology and reproducibility. JAX-first: minimize host-device transfers, use interpax for interpolation.

**Bayesian (`--var MODE=bayesian`):** Rigorous Bayesian inference with NumPyro and MCMC diagnostics. Statistical-physicist implements NumPyro models with NUTS sampler, warm-start from NLSQ. Jax-pro ensures convergence diagnostics: R-hat (<1.01), ESS (>400/chain), BFMI (>0.3). ML-expert runs model comparison (WAIC, LOO-CV via ArviZ). Mandatory ArviZ diagnostics and explicit seeds.

**Julia SciML (`--var MODE=julia-sciml`):** Julia's SciML ecosystem (DifferentialEquations.jl, ModelingToolkit.jl). Julia-pro implements solvers with proper algorithm selection (Tsit5, TRBDF2, SOSRI for SDEs). Simulation-expert defines physical systems, conservation laws, and boundary conditions. Jax-pro handles Python-Julia interop for data exchange. Research-expert validates against analytical solutions.

**Julia ML (`--var MODE=julia-ml`):** Julia ML/DL/HPC with Lux.jl, CUDA.jl, MPI.jl. Julia-ml-hpc implements models with explicit parameter management and distributed training. Neural-network-master designs AD-friendly architectures. ML-expert sets up experiment tracking and benchmarks. Research-expert validates reproducibility.

**Dynamics (`--var MODE=dynamics`):** Bifurcation analysis, chaos, coupled oscillators, and equation discovery. Nonlinear-dynamics-expert classifies the system and designs the analysis protocol (phase portrait, Lyapunov exponents, SINDy/UDE). Jax-pro implements GPU-accelerated parameter sweeps. Julia-pro implements analysis via DynamicalSystems.jl and ChaosTools.jl. Research-expert validates against published benchmarks.

**MD Simulation (`--var MODE=md-sim`):** Molecular dynamics and ML force fields. Simulation-expert designs the simulation protocol: force field selection, ensemble settings, equilibration, and for ML workflows, DFT training data curation. Jax-pro implements JAX-MD kernels, neighbor lists, and enhanced sampling. ML-expert handles training loops with force matching loss. Research-expert validates (energy MAE, phonon dispersion, elastic constants). Force field validation before production runs.

**Desktop (`--var MODE=desktop`):** PyQt/PySide6 scientific desktop applications with JAX backends. App-developer builds the GUI with docking panels, PyQtGraph plots, and light/dark theming. Jax-pro implements the computation backend as a clean API (GUI never imports JAX directly). Python-pro wires GUI to backend: worker threads, signal/slot, config management, packaging. Research-expert validates numerical outputs.

**Reproduce (`--var MODE=reproduce`):** Research paper reproduction. Research-expert leads: decomposes the paper into implementable components, validates methodology, and designs the reproduction plan. Python-pro handles systems infrastructure and packaging. Jax-pro implements numerical kernels. ML-expert sets up experiment tracking and evaluation metrics. Goal: `uv sync && uv run reproduce-all`.

### Auto-Variant Selection

When no MODE is specified and the recommender runs, signal detection determines the variant:
- numpyro/pymc + arviz -> bayesian
- julia + DifferentialEquations/ModelingToolkit -> julia-sciml
- julia + Lux/Flux/MLJ + CUDA -> julia-ml
- diffrax + DynamicalSystems/BifurcationKit -> dynamics
- jax-md/openmm + trajectories -> md-sim
- PyQt6/PySide6 + jax/numpy -> desktop
- arxiv IDs in README -> reproduce
- otherwise -> default

### Placeholders

`PROBLEM`, `REFERENCE_PAPERS` (default) | `DATA_TYPE`, `MODEL_CLASS` (bayesian) | `SYSTEM_DESCRIPTION` (dynamics) | `SYSTEM`, `PROPERTY`, `FORCE_FIELD` (md-sim) | `APP_NAME`, `GUI_FRAMEWORK`, `DOMAIN` (desktop) | `PAPER_TITLE`, `PAPER_REF` (reproduce)

### Aliases

`bayesian` -> `sci-compute --var MODE=bayesian`, `julia-sciml` -> `sci-compute --var MODE=julia-sciml`, `julia-ml` -> `sci-compute --var MODE=julia-ml`, `nonlinear-dynamics` -> `sci-compute --var MODE=dynamics`, `md-simulation` -> `sci-compute --var MODE=md-sim`, `paper-implement` -> `sci-compute --var MODE=reproduce`, `sci-desktop` -> `sci-compute --var MODE=desktop`

### Signals

Required: python + (jax | equinox | optax) OR julia + (DifferentialEquations | Lux | Flux). Strong: `experiments/`, `notebooks/`, interpax, arviz, numpyro, DynamicalSystems, PyQt6, PySide6. Counter: react/next dominant (blocks science recommendation).

---

## Team 6: Modernize

Migrate a legacy codebase to modern architecture using the Strangler Fig pattern.

### Agents

| Role | Agent | Suite | Responsibility |
|------|-------|-------|----------------|
| legacy-analyst | `dev-suite:software-architect` | dev-suite | Map legacy architecture, identify strangler boundaries, design target architecture with ADRs |
| migration-engineer | `dev-suite:systems-engineer` | dev-suite | Execute module-by-module migration with adapter layers for backward compatibility |
| quality-gate | `dev-suite:quality-specialist` | dev-suite | Write characterization tests BEFORE migration, run continuously to catch regressions |
| test-engineer | `dev-suite:debugger-pro` | dev-suite | Build migration test harness, validate feature parity |

### Workflow

`legacy-analyst` maps the existing codebase and designs the target architecture. `quality-gate` writes characterization tests for existing behavior. Then `migration-engineer` migrates module by module (characterization tests must pass before each module). `test-engineer` validates feature parity throughout.

**Critical rule:** QA must have characterization tests passing before migration-engineer begins each module.

### Placeholders

`LEGACY_SYSTEM`, `OLD_STACK`, `NEW_STACK`

### When to Use

Legacy migration, technology modernization, or major refactoring where the existing system must remain operational during transition.

---

## Team 7: AI Engineering

Build production AI applications -- RAG systems, LLM-powered apps, multi-agent orchestration, and prompt R&D.

### Agents (by variant)

| Variant | MODE | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
|---------|------|---------|---------|---------|---------|
| *default* (llm-app) | -- | `science-suite:ai-engineer` | `science-suite:prompt-engineer` | `dev-suite:software-architect` | `science-suite:python-pro` |
| multi-agent | `multi-agent` | `agent-core:orchestrator` | `agent-core:reasoning-engine` | `agent-core:context-specialist` | `science-suite:ai-engineer` |

### Variant Details

**Default (LLM app):** Build RAG systems, LLM-powered apps, and tool-use agents. AI-engineer designs the core pipeline (ingestion, chunking, embedding, retrieval, LLM orchestration, guardrails). Prompt-engineer designs system prompts and builds evaluation frameworks (LLM-as-judge, A/B testing). Software-architect builds streaming API endpoints, caching, and observability. Python-pro handles Python systems integration.

**Multi-Agent (`--var MODE=multi-agent`):** Build multi-agent orchestration systems with 2+ coordinated agents. Orchestrator designs the agent topology (hub-spoke, pipeline, blackboard), task decomposition, and coordination protocol. Reasoning-engine reviews agent prompts for chain-of-thought quality and error recovery. Context-specialist implements shared memory, context passing, and knowledge persistence. AI-engineer builds the agent runtime with tool definitions, state machines, and evaluation.

### Placeholders

`USE_CASE`

### Aliases

`llm-app` -> `ai-engineering`, `multi-agent` -> `ai-engineering --var MODE=multi-agent`

### Signals

Required: python + llm-libs. Strong: `prompts/`, `rag/`, vector DB. Auto-variant: multi-agent if `agents/` + `tools/` + langgraph.

---

## Team 8: ML Deploy

Model deployment, data pipeline engineering, and performance optimization.

### Agents (by variant)

| Variant | MODE | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
|---------|------|---------|---------|---------|---------|
| *default* (deploy) | -- | `science-suite:ml-expert` | `dev-suite:devops-architect` | `dev-suite:sre-expert` | `science-suite:jax-pro` |
| data | `data` | `science-suite:ml-expert` | `science-suite:python-pro` | `dev-suite:automation-engineer` | `science-suite:research-expert` |
| perf | `perf` | `dev-suite:debugger-pro` | `science-suite:python-pro` | `science-suite:jax-pro` | `dev-suite:systems-engineer` |

### Variant Details

**Default (deploy):** Deploy ML models to production. ML-expert handles model export (ONNX/SavedModel, quantization), model cards, and validation datasets. Devops-architect builds serving infrastructure (FastAPI/TorchServe/Triton), container packaging, autoscaling, and canary deployments. SRE sets up prediction drift detection, latency SLOs, and automated rollback. Jax-pro optimizes inference latency (XLA AOT, batch scheduling, model sharding).

**Data (`--var MODE=data`):** ETL pipelines, feature stores, and MLOps data infrastructure. ML-expert architects the ETL/ELT pipeline with schema validation (pandera) and incremental processing. Python-pro implements the feature store with online/offline serving and drift detection. Automation-engineer sets up orchestration (Airflow/Dagster) and data lineage. Research-expert implements data quality checks (Great Expectations) and anomaly detection. Key constraint: all transformations must be idempotent.

**Perf (`--var MODE=perf`):** CPU/GPU profiling and performance optimization. Debugger-pro investigates algorithmic bottlenecks, GIL contention, and I/O issues. Python-pro applies Cython/mypyc compilation, asyncio, and Rust extensions via PyO3. Jax-pro converts sequential loops to vmap, optimizes XLA compilation, and minimizes host-device transfers. Systems-engineer profiles CPU/memory/cache with perf, flamegraphs, and tracemalloc. Protocol: profile -> identify top 3 bottlenecks -> optimize one at a time -> re-profile.

### Placeholders

`MODEL_TYPE`, `SERVING_FRAMEWORK` (deploy) | `DATA_SOURCE`, `ML_TARGET` (data) | `TARGET_CODE`, `SPEEDUP_TARGET` (perf)

### Aliases

`data-pipeline` -> `ml-deploy --var MODE=data`, `perf-optimize` -> `ml-deploy --var MODE=perf`

### Signals

Required: python + ml-libs. Strong: `models/`, `serving/`, `deploy/`. Auto-variant: data if `dags/` + airflow; perf if `benchmarks/` + `profiling/`.

---

## Team 9: Docs & Publishing

Documentation overhaul and reproducible research infrastructure.

### Agents (by variant)

| Variant | MODE | Agent 1 | Agent 2 | Agent 3 | Agent 4 |
|---------|------|---------|---------|---------|---------|
| *default* (docs) | -- | `dev-suite:documentation-expert` | `dev-suite:software-architect` | `science-suite:research-expert` | `science-suite:python-pro` |
| research | `research` | `science-suite:research-expert` | `agent-core:context-specialist` | `science-suite:python-pro` | `dev-suite:automation-engineer` |

### Variant Details

**Default (docs):** Comprehensive documentation overhaul. Documentation-expert designs the information architecture following Diataxis (tutorials, how-tos, reference, explanation) and sets up Sphinx/MkDocs. Software-architect reviews technical accuracy by cross-referencing source code. Research-expert creates interactive tutorials, Jupyter notebooks, and Sphinx gallery examples. Python-pro structures the project as an installable package with proper CI. Goal: every public API must have docstring + reference page + example.

**Research (`--var MODE=research`):** Reproducible research infrastructure bridging science-suite and agent-core. Research-expert defines reproducibility requirements: experiment tracking, artifact versioning, data provenance. Context-specialist implements research knowledge graphs, paper reference management, and cross-project context sharing. Python-pro builds experiment runners, artifact storage (HDF5/Arrow), and CLI tools. Automation-engineer wires automated experiment scheduling, notebook execution, and reproducibility CI.

### Placeholders

`PROJECT_NAME` (docs) | `PROJECT_NAME`, `RESEARCH_GOAL` (research)

### Signals

Required: docs dir present. Strong: `tutorials/`, sphinx-gallery. Auto-variant: research if `experiments/` + `notebooks/` + references.bib.

---

## Team 10: Plugin Forge

Build Claude Code extensions -- plugins, hooks, agents, commands, skills, and SDK integrations.

### Agents

| Role | Agent | Suite | Responsibility |
|------|-------|-------|----------------|
| creator | `plugin-dev:agent-creator` | plugin-dev | Generate plugin structure: manifest, agents, commands, skills |
| hook-designer | `hookify:conversation-analyzer` | hookify | Analyze conversation patterns, design PreToolUse/SessionStart hooks |
| quality | `dev-suite:quality-specialist` | dev-suite | Write tests, set up CI for metadata validation and context budget |
| validator | `plugin-dev:plugin-validator` | plugin-dev | Validate complete plugin structure (read-only) |

### Workflow

`creator` + `hook-designer` work in parallel to generate plugin components and hook rules. Then `quality` writes tests and CI workflows. Finally `validator` checks the complete plugin structure: manifest schema, file references, frontmatter, and context budget.

### Placeholders

`PLUGIN_NAME`, `PLUGIN_DESCRIPTION`

### When to Use

Building any Claude Code extension: new plugins, custom hooks, agent definitions, slash commands, or skill files.

---

## Long-Running Workflow Protocol

All teams follow this protocol for multi-session work, based on Anthropic's [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents):

1. **Session Init** -- First agent reads `PROGRESS.md` + recent git log. If no progress file exists, create one from the task prompt.
2. **Task Tracking** -- Maintain `PROGRESS.md` as a structured JSON checklist. Each task maps to one agent's deliverable.
3. **Incremental Progress** -- Complete one task fully before starting the next. No parallel edits to the same file.
4. **Clean State** -- Git commit after each completed task with descriptive message. Update progress file before commit.
5. **Session Resume** -- On resume, read progress + git log + git diff. Skip completed tasks.
6. **Verification** -- Run team-appropriate verification after each task (tests for dev teams, numerical validation for science teams).
7. **QA Gate** -- Designated reviewer agent runs last, checks all completed tasks against the original spec.

### Team-Specific Verification

| Team | Env Check | Verification |
|------|-----------|-------------|
| feature-dev | Tests pass, linter clean | Test suite + manual feature test |
| debug | Symptoms reproduced | Original failure no longer triggers |
| quality-gate | PR branch checked out | All review comments addressable |
| api-infra | API server starts / terraform init | Contract tests / `terraform plan` |
| sci-compute | JAX/Julia/GPU detected | Numerical validation + convergence |
| modernize | Legacy system accessible | Feature parity tests |
| ai-engineering | API keys valid, MCP reachable | E2E agent execution |
| ml-deploy | Model loadable, infra available | Inference latency within SLO |
| docs-publish | Sphinx/MkDocs builds | No broken links, coverage > threshold |
| plugin-forge | Plugin structure valid | `metadata_validator.py` passes |

---

## Quality Gate Enhancers

Any team can be enhanced by adding official plugin agents as quality gates. Append an enhancer to any team's configuration for automated review after implementation.

| Enhancer | Agent Type | Best With | What It Catches |
|----------|-----------|-----------|-----------------|
| Code Review | `pr-review-toolkit:code-reviewer` | feature-dev, api-infra, modernize | Style, bugs, guidelines |
| Silent Failures | `pr-review-toolkit:silent-failure-hunter` | debug, api-infra:infra, docs-publish | Swallowed errors |
| Test Gaps | `pr-review-toolkit:pr-test-analyzer` | quality-gate, modernize, api-infra | Missing test coverage |
| Type Quality | `pr-review-toolkit:type-design-analyzer` | ml-deploy:data, docs-publish, sci-compute | Weak type invariants |
| Code Simplicity | `code-simplifier:code-simplifier` | modernize, ml-deploy:data, quality-gate | Unnecessary complexity |
| Plan Adherence | `superpowers:code-reviewer` | feature-dev, modernize, ml-deploy:perf | Drift from plan |

To add an enhancer, append to any team prompt:

```
Additionally, spawn a "reviewer" teammate
(pr-review-toolkit:code-reviewer) that reviews all changes after the
implementation teammates finish their work. This reviewer is read-only
and reports issues sorted by severity. The team should not be considered
done until the reviewer's critical issues are addressed.
```

---

## Alias Table

All 20 aliases for backward compatibility with previous team names:

| # | Alias | Resolves To |
|---|-------|-------------|
| 1 | `pr-review` | `quality-gate` |
| 2 | `security` | `quality-gate --var MODE=security` |
| 3 | `api-design` | `api-infra` |
| 4 | `infra-setup` | `api-infra --var MODE=infra` |
| 5 | `bayesian` | `sci-compute --var MODE=bayesian` |
| 6 | `julia-sciml` | `sci-compute --var MODE=julia-sciml` |
| 7 | `julia-ml` | `sci-compute --var MODE=julia-ml` |
| 8 | `nonlinear-dynamics` | `sci-compute --var MODE=dynamics` |
| 9 | `md-simulation` | `sci-compute --var MODE=md-sim` |
| 10 | `paper-implement` | `sci-compute --var MODE=reproduce` |
| 11 | `sci-desktop` | `sci-compute --var MODE=desktop` |
| 12 | `incident` | `debug --var MODE=incident` |
| 13 | `debug-triage` | `debug --var MODE=triage` |
| 14 | `debug-gui` | `debug --var MODE=gui` |
| 15 | `debug-numerical` | `debug --var MODE=numerical` |
| 16 | `debug-schema` | `debug --var MODE=schema` |
| 17 | `llm-app` | `ai-engineering` |
| 18 | `multi-agent` | `ai-engineering --var MODE=multi-agent` |
| 19 | `data-pipeline` | `ml-deploy --var MODE=data` |
| 20 | `perf-optimize` | `ml-deploy --var MODE=perf` |

---

## Usage Tips

1. **Replace placeholders** (`[BRACKETS]`) with your project specifics before pasting
2. **Start with quality-gate** (default variant) if new to agent teams -- read-only, low-risk
3. **Use delegate mode** (`Shift+Tab`) to prevent the lead from implementing tasks itself
4. **Monitor progress** with `Shift+Up/Down` (in-process) or click panes (tmux)
5. **`Ctrl+T`** toggles the shared task list view
6. **Prefer Sonnet** for most teammates (cost-effective); use Opus for architecture/design decisions
7. **Avoid file conflicts** -- ensure each teammate owns distinct directories
8. **Use variants** (`--var MODE=x`) to specialize a team without remembering separate team names
9. **Use aliases** when you remember the old name -- `/team-assemble bayesian` resolves to `sci-compute --var MODE=bayesian` automatically
10. **Default variants** (no MODE) cover 80% of use cases -- variants are optional specializations

## References

- [Official Agent Teams Documentation](https://code.claude.com/docs/en/agent-teams)
- [Claude Code Subagents](https://code.claude.com/docs/en/sub-agents)
- [Claude Code Agent Teams Blog](https://addyosmani.com/blog/claude-code-agent-teams/)
- [Integration Map](integration-map.rst) -- Suite dependencies, MCP server roles, and skill coverage
- [Glossary](glossary.rst) -- Key terms including hub skills, sub-skills, and routing trees
