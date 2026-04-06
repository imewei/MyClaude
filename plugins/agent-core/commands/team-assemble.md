---
name: team-assemble
description: Generate ready-to-use agent team configurations from 21 pre-built templates for MyClaude v3.1.1
argument-hint: <team-type> [--var KEY=VALUE]
category: agent-core
execution-modes:
  quick: "1-2 minutes"
  standard: "2-5 minutes"
allowed-tools: [Read, Glob, Grep, Bash, Task, Write, Edit]
tags: [agent-teams, orchestration, multi-agent, collaboration, parallel]
---

# Agent Team Assembly

$ARGUMENTS

You are a team assembly specialist. Your job is to generate a ready-to-use agent team prompt from the pre-built templates below, customized with the user's project details.

## Actions

| Action | Description |
|--------|-------------|
| `list` | Show all 21 available team configurations |
| `<team-type>` | Generate the prompt for the specified team |
| `<team-type> --var KEY=VALUE` | Generate with placeholder substitution |

**Examples:**
```bash
/team-assemble list
/team-assemble feature-dev
/team-assemble incident-response --var SYMPTOMS="API returning 500 errors on /auth endpoint"
/team-assemble sci-compute --var PROBLEM="Bayesian parameter estimation for SAXS data"
/team-assemble pr-review --var PR_OR_BRANCH=142
```

---

## Step 1: Parse the Command

1. If the argument is `list`, display the team catalog table and stop.
2. Otherwise, match the argument to one of the 21 team types below.
3. If `--var` flags are provided, substitute `[PLACEHOLDER]` values in the template.
4. Output the final prompt in a fenced code block, ready to paste.

---

## Step 2: Team Catalog

When `list` is invoked, display this table:

```
Agent Team Catalog (MyClaude v3.1.1) — 21 Teams
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEVELOPMENT & OPERATIONS
 #  Type                Teammates  Suites Used              Best For
 1  feature-dev         4          dev + feature-dev plugin  Feature build + review
 2  incident-response   3          dev-suite                 Live production incident (SRE/infra)
 3  pr-review           4          pr-review-toolkit         Comprehensive PR review
 4  quality-security    4          dev-suite                 Quality + security audit
 5  api-design          4          dev-suite                 API design (REST/GraphQL/gRPC)
 6  infra-setup         3          dev-suite                 Cloud + CI/CD setup
 7  modernization       4          dev-suite                 Legacy migration

SCIENTIFIC COMPUTING
 8  sci-compute         4          science                   JAX/ML/DL pipelines
 9  bayesian-pipeline   4          science                   NumPyro / MCMC inference
10  julia-sciml         4          science                   Julia SciML / DiffEq
11  md-simulation       4          science                   Molecular dynamics + ML FF
12  paper-implement     4          science                   Reproduce research papers

CROSS-CUTTING
13  ai-engineering      4          science + dev + core      AI/LLM apps + agents
14  perf-optimize       4          dev + science             Performance profiling
15  data-pipeline       4          science + dev             ETL, feature engineering
16  docs-publish        4          dev + science             Documentation + reproducibility

PLUGIN DEVELOPMENT
17  plugin-forge        4          plugin-dev + hookify      Claude Code extensions

DEBUGGING
18  debug-triage        2          dev + feature             Quick bug triage (lightweight)
19  debug-gui           4          dev + feature + science   GUI threading, signal safety
20  debug-numerical     4          dev + feature + science   JAX/NaN, ODE solver, tracing
21  debug-schema        4          dev + feature + pr-review Schema/type drift, contracts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage: /team-assemble <type> [--var KEY=VALUE ...]
Docs:  docs/agent-teams-guide.md
```

---

## Step 3: Team Templates

### feature-dev

**Placeholders:** `FEATURE_NAME`, `PROJECT`, `FRONTEND_STACK`, `BACKEND_STACK`

```
Create an agent team called "feature-dev" to design, build, and review
[FEATURE_NAME] for [PROJECT].

Spawn 4 specialist teammates:

1. "architect" (feature-dev:code-architect) - Analyze the existing
   codebase patterns and conventions, then produce a comprehensive
   implementation blueprint: files to create/modify, component designs,
   data flows, and build sequence. Present the blueprint for approval
   before any code is written. Owns docs/design/.

2. "builder" (dev-suite:app-developer) - Implement the frontend
   components following the architect's blueprint. Build with
   [FRONTEND_STACK]. Focus on performance, accessibility, and
   offline-first patterns. Owns src/components/, src/pages/, src/hooks/.

3. "backend" (dev-suite:software-architect) - Implement the backend
   services following the architect's blueprint. Build with
   [BACKEND_STACK]. Design scalable APIs with proper error handling
   and validation. Owns src/api/, src/services/, src/models/.

4. "reviewer" (pr-review-toolkit:code-reviewer) - After builder and
   backend complete their work, review all changes for adherence to
   the architect's blueprint, project guidelines, and best practices.
   Report only high-priority issues. Read-only.

Workflow: architect → (builder + backend in parallel) → reviewer.
```

### incident-response

**Placeholders:** `SYMPTOMS`, `AFFECTED_MODULES`

```
Create an agent team called "incident-response" to investigate a production issue:
[SYMPTOMS].

Spawn 3 teammates to investigate different hypotheses in parallel:

1. "debugger" (dev-suite:debugger-pro) - Root cause analyst. Examine the
   application code for bugs, race conditions, or logic errors. Focus on
   [AFFECTED_MODULES]. Analyze stack traces, reproduce the issue locally,
   and form hypotheses. Challenge the other teammates' findings.

2. "sre" (dev-suite:sre-expert) - Reliability investigator. Check
   observability data: metrics, logs, distributed traces. Look for patterns
   in error rates, latency spikes, resource exhaustion. Correlate timing
   with deployments or config changes.

3. "infra" (dev-suite:devops-architect) - Infrastructure analyst. Investigate
   the deployment environment: container health, network connectivity,
   database performance, resource limits. Check if infrastructure changes
   correlate with the issue.

Have teammates share findings with each other and challenge each other's
theories. Synthesize into a root cause report with: confirmed root cause,
evidence, recommended fix, and prevention measures.
```

### pr-review

**Placeholders:** `PR_OR_BRANCH`

```
Create an agent team called "pr-review" to perform a comprehensive
review of [PR_OR_BRANCH].

Spawn 4 specialist teammates:

1. "code-reviewer" (pr-review-toolkit:code-reviewer) - Review all changed
   files for adherence to project guidelines, style violations, logic
   errors, security vulnerabilities, and comment accuracy. Use
   confidence-based filtering to report only high-priority issues.

2. "failure-hunter" (pr-review-toolkit:silent-failure-hunter) - Examine
   every catch block, fallback path, and error handler in the diff. Flag
   silent failures, swallowed exceptions, and inappropriate default
   values. Rate severity: critical/warning/info.

3. "test-analyzer" (pr-review-toolkit:pr-test-analyzer) - Analyze test
   coverage for all new functionality. Identify critical gaps: untested
   edge cases, missing error path tests, and uncovered branches. Suggest
   specific test cases to add.

4. "type-analyzer" (pr-review-toolkit:type-design-analyzer) - Review all
   new or modified types for encapsulation quality, invariant expression,
   and enforcement. Rate each type on a 1-5 scale. Flag any types that
   leak implementation details or fail to express their invariants.

Coordination: Each reviewer works independently on the same diff. Lead
collects all findings and produces a unified review with issues sorted
by severity. No file ownership conflicts since all agents are read-only.
```

### quality-security

**Placeholders:** `PROJECT_PATH`
**Aliases:** `quality-audit`, `security-harden`, `code-health`

```
Create an agent team called "quality-security" to perform a comprehensive
code quality, architecture, and security audit of [PROJECT_PATH].

Spawn 4 reviewers, each with a distinct lens:

1. "security" (dev-suite:sre-expert) - Security and reliability auditor.
   Scan for OWASP Top 10 vulnerabilities: injection, broken auth, data
   exposure, XSS, CSRF, insecure deserialization. Review authentication
   flows, input validation, secret handling, and dependency vulnerabilities.
   Check container security, TLS configuration, network segmentation, and
   IAM policies. Rate each finding: Critical/High/Medium/Low with CVSS scores.

2. "architecture" (dev-suite:software-architect) - Architecture reviewer.
   Assess design patterns, SOLID principles, coupling/cohesion, cyclomatic
   complexity, and code duplication. Identify architectural anti-patterns,
   tech debt hotspots, and missing module boundaries. Produce Architecture
   Decision Records for major concerns.

3. "testing" (dev-suite:quality-specialist) - Test coverage analyst. Map
   untested code paths, identify missing edge cases, check for flaky tests,
   and assess the testing pyramid (unit/integration/e2e ratio). Verify
   error path coverage and contract tests. Recommend specific tests to add.

4. "secops" (dev-suite:automation-engineer) - Security automation engineer.
   Build or verify security CI/CD: SAST (Semgrep/CodeQL), DAST (OWASP ZAP),
   dependency scanning (Dependabot/Snyk), container scanning (Trivy), and
   secret detection (TruffleHog). Set up pre-commit hooks for security
   checks. Owns .github/workflows/security.yml, scripts/security/.

Each reviewer works independently, then shares findings. Synthesize into
a prioritized remediation plan with effort estimates and CVSS severity.
```

### api-design

**Placeholders:** `API_PROTOCOL` (REST/GraphQL/gRPC), `SERVICE_NAME`

```
Create an agent team called "api-design" to design and implement a
[API_PROTOCOL] API for [SERVICE_NAME].

Spawn 4 teammates:

1. "api-designer" (dev-suite:software-architect) - API architect. Design
   the API specification following REST best practices: resource naming,
   HTTP methods, status codes, pagination, filtering, versioning strategy,
   and error response format. Create OpenAPI/Swagger spec. Require plan
   approval before implementation. Owns api/, specs/.

2. "implementer" (dev-suite:app-developer) - API developer. Implement the
   endpoints following the approved spec. Handle authentication (JWT/OAuth2),
   rate limiting, input validation, and error handling. Implement database
   queries with proper indexing. Owns src/routes/, src/middleware/,
   src/controllers/.

3. "tester" (dev-suite:quality-specialist) - API test engineer. Write
   contract tests (Pact), integration tests, load tests, and security
   tests (auth bypass, injection, rate limit circumvention). Validate all
   error paths. Owns tests/.

4. "docs-writer" (dev-suite:documentation-expert) - API documentation
   specialist. Generate comprehensive API docs with: endpoint reference,
   authentication guide, code examples in multiple languages, error
   handling guide, and migration guide from previous versions.
   Owns docs/api/.

Dependency: api-designer defines spec -> implementer + tester work in
parallel -> docs-writer documents the final API.
```

### infra-setup

**Placeholders:** `PROJECT_NAME`, `CLOUD_PROVIDER`

```
Create an agent team called "infra-setup" to build the infrastructure for
[PROJECT_NAME] on [CLOUD_PROVIDER].

Spawn 3 infrastructure specialists:

1. "cloud-architect" (dev-suite:devops-architect) - Platform engineer. Design
   and implement Infrastructure as Code using Terraform/Pulumi. Set up:
   VPC/networking, compute (EKS/ECS/Lambda), database (RDS/DynamoDB),
   storage (S3), and IAM policies. Follow zero-trust networking and
   least-privilege principles. Owns infra/, terraform/.

2. "cicd-engineer" (dev-suite:automation-engineer) - Pipeline architect.
   Build GitHub Actions workflows for: lint/test/build, container image
   builds, staged deployments (dev->staging->prod), security scanning
   (SAST/DAST), and release automation. Implement caching and artifact
   promotion. Owns .github/workflows/, scripts/ci/.

3. "sre-lead" (dev-suite:sre-expert) - Observability architect. Set up
   Prometheus metrics collection, Grafana dashboards, distributed tracing
   (OpenTelemetry), structured logging, and alerting rules. Define
   SLIs/SLOs for key services. Implement health checks and readiness
   probes. Owns monitoring/, dashboards/.

Dependencies: cloud-architect defines infrastructure first, then
cicd-engineer configures deployment targets, then sre-lead instruments
the services.
```

### modernization

**Placeholders:** `LEGACY_SYSTEM`, `OLD_STACK`, `NEW_STACK`

```
Create an agent team called "modernization" to migrate [LEGACY_SYSTEM]
from [OLD_STACK] to [NEW_STACK].

Spawn 4 teammates:

1. "architect" (dev-suite:software-architect) - Target architecture designer.
   Analyze the existing codebase, identify migration boundaries (Strangler
   Fig pattern), design the target architecture with clean module boundaries.
   Create Architecture Decision Records for key choices. Require plan
   approval before implementation.

2. "implementer" (dev-suite:systems-engineer) - Migration developer. Execute
   the migration following the architect's plan. Implement adapter layers
   for backward compatibility during transition. Refactor module by module,
   ensuring each module works independently before moving to the next.
   Owns src/new/, src/adapters/.

3. "qa-lead" (dev-suite:quality-specialist) - Regression guardian. Write
   comprehensive tests for existing behavior BEFORE migration begins
   (characterization tests). Run tests continuously during migration to
   catch regressions. Owns tests/.

4. "docs-lead" (dev-suite:documentation-expert) - Migration documenter.
   Document the migration plan, track progress, write runbooks for rollback
   procedures, and update API documentation as interfaces change.
   Owns docs/migration/.

Critical rule: QA must have characterization tests passing before
implementer begins each module migration.
```

### sci-compute

**Placeholders:** `PROBLEM`, `REFERENCE_PAPERS`
**Aliases:** `sci-pipeline`, `dl-research`

```
Create an agent team called "sci-compute" to build a scientific computing
or deep learning pipeline for [PROBLEM].

Spawn 4 specialist teammates:

1. "jax-engineer" (science-suite:jax-pro) - JAX implementation specialist.
   Implement the core computational kernels using JAX with JIT compilation,
   vmap for batching, pmap for multi-device parallelism, and custom VJPs
   where needed. Handle GPU memory management, efficient batching, and
   mixed precision training. For neural networks, implement training loops
   with gradient clipping. Owns src/core/, src/kernels/, src/training/.

2. "architect" (science-suite:neural-network-master) - Model and architecture
   designer. For deep learning: design neural architectures considering
   attention mechanisms, normalization, activation functions, and parameter
   efficiency. Analyze gradient flow and provide theoretical justification.
   For non-DL pipelines: design the computational graph, algorithm selection,
   and numerical stability strategy. Reference [REFERENCE_PAPERS].
   Owns src/models/.

3. "ml-engineer" (science-suite:ml-expert) - ML pipeline architect. Set up
   experiment tracking (W&B/MLflow), hyperparameter optimization (Optuna),
   data loading pipelines, model versioning, and checkpoint management.
   Configure data augmentation and preprocessing. Owns configs/, scripts/,
   src/data/.

4. "researcher" (science-suite:research-expert) - Research methodology
   validator. Review the computational approach for scientific correctness,
   reproducibility (explicit seeds, deterministic ops), and statistical
   validity. Implement evaluation metrics, ablation studies, and training
   diagnostics. Validate against [REFERENCE_PAPERS]. Owns docs/, notebooks/,
   evaluation/.

Ensure JAX-first architecture: minimize host-device transfers, use
interpax for interpolation, mandatory ArviZ diagnostics for Bayesian work.
```

### bayesian-pipeline

**Placeholders:** `DATA_TYPE`, `MODEL_CLASS`

```
Create an agent team called "bayesian-pipeline" to build a Bayesian
inference pipeline for [DATA_TYPE] using [MODEL_CLASS].

Spawn 4 specialist teammates:

1. "bayesian-engineer" (science-suite:statistical-physicist) - NumPyro/JAX specialist. Implement the
   probabilistic model in NumPyro. Set up NUTS sampler with
   appropriate warmup, target accept probability, and mass matrix
   adaptation. Implement warm-start from NLSQ point estimates.
   Handle GPU memory for large datasets. Owns src/models/, src/inference/.

2. "statistician" (science-suite:research-expert) - Prior and model structure expert. Design informative
   vs weakly informative priors with physical justification. Implement
   hierarchical model structure if needed. Design posterior predictive
   checks and prior predictive simulations. Handle model reparametrization
   for sampling efficiency (non-centered parameterization). Owns
   src/priors/, src/diagnostics/.

3. "ml-validator" (science-suite:ml-expert) - Model comparison and validation. Implement model
   comparison metrics: WAIC, LOO-CV (using ArviZ), Bayes factors.
   Design cross-validation strategies. Build predictive performance
   benchmarks against frequentist baselines (MLE, MAP). Owns
   src/comparison/, src/validation/.

4. "convergence-auditor" (science-suite:jax-pro) - MCMC diagnostics specialist. Ensure convergence
   diagnostics are comprehensive: R-hat (<1.01), ESS (>400/chain),
   BFMI (>0.3), divergence checks, trace plots. Document all modeling
   choices and sensitivity analyses. Owns docs/, notebooks/.

Mandatory: ArviZ for all diagnostics. NLSQ warm-start before NUTS.
Explicit seeds for reproducibility.
```

### julia-sciml

**Placeholders:** `PROBLEM`, `REFERENCE_PAPERS`

```
Create an agent team called "julia-sciml" to build a Julia SciML pipeline
for [PROBLEM].

Spawn 4 specialist teammates:

1. "julia-engineer" (science-suite:julia-pro) - Julia SciML specialist. Implement the core solvers
   using DifferentialEquations.jl with appropriate algorithm selection
   (Tsit5, TRBDF2, SOSRI for SDEs). Use ModelingToolkit.jl for symbolic
   model definition and automatic Jacobian generation. Set up Turing.jl
   for Bayesian parameter estimation if needed. Owns src/, Project.toml.

2. "simulation-architect" (science-suite:simulation-expert) - Physics model designer. Define the physical
   system, conservation laws, boundary conditions, and validation
   benchmarks. Ensure numerical stability (CFL conditions, adaptive
   stepping). Design parameter studies and sensitivity analyses.
   Owns models/, benchmarks/.

3. "methodology" (science-suite:research-expert) - Research validator. Verify the mathematical formulation
   against [REFERENCE_PAPERS]. Set up convergence tests, error analysis,
   and comparison with analytical solutions where available. Ensure
   reproducibility with fixed seeds and version pinning. Owns docs/,
   notebooks/, test/.

4. "python-bridge" (science-suite:python-pro) - Interoperability engineer. Build Python-Julia bridges
   using PythonCall.jl or PyJulia for data exchange. Set up data ingestion
   pipelines, results export (HDF5/Arrow), and visualization (Makie.jl
   for interactive, Plots.jl for publication). Owns scripts/, viz/.

Use Julia 1.10+ with strict type annotations at module boundaries.
```

### md-simulation

**Placeholders:** `SYSTEM`, `PROPERTY`, `FORCE_FIELD`
**Aliases:** `md-campaign`, `ml-forcefield`

```
Create an agent team called "md-simulation" to run a molecular dynamics
campaign for [SYSTEM] studying [PROPERTY].

Spawn 4 specialist teammates:

1. "simulation-architect" (science-suite:simulation-expert) - Simulation setup
   and data specialist. Design the simulation protocol: system construction
   (particle placement, box geometry), force field selection ([FORCE_FIELD]),
   ensemble (NVT/NPT/NVE), thermostat/barostat settings, integration
   timestep, and cutoff schemes. Handle equilibration protocol with staged
   heating/cooling if needed. For ML force field workflows: curate DFT
   training data with active learning, design the training distribution to
   cover relevant PES regions. Owns simulations/, configs/, data/.

2. "gpu-engine" (science-suite:jax-pro) - JAX-MD implementation and training
   engineer. Implement the simulation engine using JAX-MD or custom JAX
   kernels. Optimize neighbor list updates, force computation (JIT-compiled),
   and trajectory output. Handle multi-GPU scaling with pmap. For ML force
   fields: implement training loop with per-atom energy loss + force matching
   loss, learning rate scheduling, gradient clipping, and EMA weights.
   Implement enhanced sampling methods (metadynamics, replica exchange) if
   needed. Owns src/engine/, src/sampling/, src/training/.

3. "analyst" (science-suite:statistical-physicist) - Thermodynamic and
   structural analysis. Compute observables: radial distribution function
   g(r), structure factor S(q), mean-square displacement (diffusion),
   velocity autocorrelation, pressure tensor, free energy profiles.
   Implement block averaging for error estimation. For ML force fields:
   benchmark against DFT reference (energy MAE, force MAE/RMSE, phonon
   dispersion, elastic constants). Owns src/analysis/, results/, evaluation/.

4. "researcher" (science-suite:research-expert) - Scientific validation and
   workflow automation. Build the campaign workflow: parameter sweep
   management, job scheduling, trajectory storage (HDF5/MDAnalysis),
   checkpoint/restart logic, and automated convergence checking. Validate
   results against known benchmarks. For ML force fields: run stability
   tests (NVE energy drift, melting point prediction). Owns scripts/,
   workflows/, notebooks/.

Ensure: proper equilibration verification, production run length
justified by autocorrelation analysis, explicit seeds. For ML force
fields: ensure physical symmetries are built into architecture, not learned.
```

### paper-implement

**Placeholders:** `PAPER_TITLE`, `PAPER_REF` (arXiv ID, DOI, or URL)

```
Create an agent team called "paper-implement" to reproduce results from
[PAPER_TITLE] ([PAPER_REF]).

Spawn 4 specialist teammates:

1. "paper-analyst" (science-suite:research-expert) - Research methodology expert. Read and decompose the
   paper: extract the core algorithm, mathematical formulation, key
   equations, hyperparameters, dataset descriptions, and evaluation
   metrics. Identify ambiguities or missing details that need resolution.
   Create a structured implementation specification. Owns docs/spec/.

2. "python-engineer" (science-suite:python-pro) - Clean implementation. Build the codebase with
   proper structure: typed interfaces, configuration management (hydra
   or dataclasses), CLI entry points, and comprehensive logging. Handle
   data loading, preprocessing, and results serialization. Owns src/,
   pyproject.toml.

3. "numerical-engineer" (science-suite:jax-pro) - Core algorithm implementation. Implement the
   mathematical core in JAX: numerical kernels, optimization routines,
   custom gradients if needed. Ensure numerical stability (log-space
   computation, gradient clipping). Match the paper's convergence
   criteria exactly. Owns src/core/, src/optim/.

4. "reproducer" (science-suite:ml-expert) - Results reproduction. Run the experiments from the
   paper with identical hyperparameters. Compare outputs: tables,
   figures, metrics. Document any discrepancies and their likely causes.
   Prepare reproduction report with side-by-side comparison.
   Owns experiments/, results/, notebooks/.

Goal: exact reproduction within reported error bars. Document ALL
deviations from the paper.
```

### ai-engineering

**Placeholders:** `USE_CASE`
**Aliases:** `llm-app`, `ai-agent-dev`, `prompt-lab`

```
Create an agent team called "ai-engineering" to build a production LLM
application for [USE_CASE].

Spawn 4 specialists:

1. "ai-engineer" (science-suite:ai-engineer) - LLM application architect.
   Design and implement the core AI pipeline: document ingestion, chunking
   strategy, embedding generation, vector store, retrieval logic, and LLM
   orchestration. For agent systems: design tool definitions, state
   management, memory systems, and planning strategies. Implement
   guardrails, content filtering, and hallucination detection.
   Owns src/ai/, src/retrieval/, src/agents/.

2. "prompt-engineer" (science-suite:prompt-engineer) - Prompt design and
   evaluation specialist. Design system prompts using chain-of-thought and
   constitutional AI patterns. Build evaluation framework: LLM-as-judge
   scoring, A/B testing, regression testing for prompt changes. Optimize
   for cost/latency/quality trade-offs. Owns prompts/, evaluation/.

3. "backend-architect" (dev-suite:software-architect) - API and serving
   infrastructure. Build streaming API endpoints, authentication, rate
   limiting, semantic caching, session management, and observability.
   Design for horizontal scaling. Owns src/api/, src/middleware/, infra/.

4. "reasoning-architect" (agent-core:reasoning-engine) - Cognitive
   architecture designer. Design reasoning scaffolds, self-reflection
   checkpoints, confidence calibration, and error correction loops.
   Owns src/reasoning/.

For LLM-only apps (no agents): omit reasoning-architect, keep 3 teammates.
```

### perf-optimize

**Placeholders:** `TARGET_CODE`, `SPEEDUP_TARGET`

```
Create an agent team called "perf-optimize" to profile and optimize
[TARGET_CODE].

Spawn 4 specialist teammates:

1. "systems-profiler" (dev-suite:systems-engineer) - Low-level performance analyst. Profile CPU usage
   (perf, py-spy), memory allocation (tracemalloc, memray), I/O patterns,
   and cache behavior. Identify hot functions, memory leaks, and
   unnecessary copies. Generate flamegraphs. Owns profiling/, reports/.

2. "jax-optimizer" (science-suite:jax-pro) - GPU/vectorization specialist. Convert sequential
   loops to vmap, identify JIT compilation opportunities, optimize
   XLA compilation (avoid recompilation), minimize host-device transfers,
   and implement efficient batching strategies. Profile with JAX's
   built-in profiler. Owns src/optimized/.

3. "bottleneck-hunter" (dev-suite:debugger-pro) - Root cause analyst. Investigate why specific
   operations are slow: algorithmic complexity (O(n^2) to O(n log n)),
   unnecessary recomputation, inefficient data structures, GIL
   contention, or I/O bottlenecks. Propose and validate fixes with
   micro-benchmarks. Owns benchmarks/.

4. "python-optimizer" (science-suite:python-pro) - Python-level optimization. Apply: Cython/mypyc
   compilation for hot paths, asyncio for I/O-bound code, multiprocessing
   for CPU-bound parallelism, efficient data structures (numpy structured
   arrays, pandas optimizations), and Rust extensions via PyO3 if needed.
   Owns src/extensions/.

Protocol: Profile first (measure) then Identify top 3 bottlenecks then
Optimize one at a time then Re-profile then Repeat.
Target: [SPEEDUP_TARGET] (e.g., 10x throughput improvement).
```

### data-pipeline

**Placeholders:** `DATA_SOURCE`, `ML_TARGET`

```
Create an agent team called "data-pipeline" to build a data pipeline
for [DATA_SOURCE] feeding [ML_TARGET].

Spawn 4 specialist teammates:

1. "data-engineer" (science-suite:python-pro) - Pipeline architect. Build the ETL/ELT pipeline:
   data ingestion (batch/streaming), transformation logic (pandas/polars/
   dask), schema validation (pandera), and output sinks (parquet/Delta
   Lake). Handle incremental processing, idempotency, and type-safe
   pipeline configuration. Owns src/pipeline/, src/transforms/.

2. "feature-engineer" (science-suite:ml-expert) - ML feature specialist. Design and implement
   the feature store: feature definitions, computation logic, online/
   offline serving, feature versioning, and point-in-time correctness.
   Implement feature monitoring for drift detection. Owns src/features/,
   feature_store/.

3. "infra-engineer" (dev-suite:devops-architect) - Data infrastructure. Set up storage (S3/GCS),
   orchestration (Airflow/Dagster), compute (Spark/Dask cluster),
   and metadata management (data catalog). Configure data lineage
   tracking and access controls. Owns infra/, dags/.

4. "quality-engineer" (dev-suite:quality-specialist) - Data quality guardian. Implement data quality
   checks: schema validation, statistical tests (Great Expectations),
   freshness monitoring, completeness checks, and anomaly detection.
   Build data quality dashboards. Owns tests/, quality_checks/.

Key constraint: all transformations must be idempotent and testable
with synthetic data.
```

### docs-publish

**Placeholders:** `PROJECT_NAME`
**Aliases:** `docs-sprint`, `reproducible-research`

```
Create an agent team called "docs-publish" to create comprehensive
documentation and ensure full reproducibility for [PROJECT_NAME].

Spawn 4 specialist teammates:

1. "docs-architect" (dev-suite:documentation-expert) - Documentation structure
   designer. Design the information architecture following the Diataxis
   framework: getting started guide, tutorials (learning-oriented), how-to
   guides (task-oriented), reference (API docs), and explanation
   (understanding-oriented). Set up Sphinx/MkDocs with proper theme and
   navigation. Audit for reproducibility gaps: hardcoded paths, missing
   seeds, undocumented parameters. Owns docs/.

2. "accuracy-validator" (dev-suite:quality-specialist) - Technical accuracy
   checker. Review all documentation for technical accuracy by
   cross-referencing with source code. Ensure code examples compile and
   run, CLI flags match actual behavior, and configuration options are
   complete. Fix stale or misleading content. Verify all experiments can
   be re-run from a single command. Owns docs/reference/.

3. "tutorial-builder" (science-suite:research-expert) - Interactive examples
   and methodology. Create tutorials with runnable code examples, Jupyter
   notebooks for interactive exploration, and a cookbook of common patterns.
   Build a docs testing harness that validates all code snippets. Create
   a "reproducing our results" guide. Convert key notebooks to Sphinx
   gallery examples. Owns docs/tutorials/, notebooks/.

4. "ci-packager" (dev-suite:automation-engineer) - Automation and packaging
   specialist. Structure the project as an installable package with
   pyproject.toml, proper dependency pinning (uv.lock), and entry points.
   Build GitHub Actions workflows: automated testing, notebook execution
   verification, figure regeneration, dependency scanning, and release
   automation. Set up pre-commit hooks. Owns .github/workflows/,
   pyproject.toml, .pre-commit-config.yaml.

Goal: anyone should be able to clone, install, and reproduce all results
with: uv sync && uv run reproduce-all
Standard: every public API must have docstring + reference page + example.
```

### plugin-forge

**Placeholders:** `PLUGIN_NAME`, `PLUGIN_DESCRIPTION`

```
Create an agent team called "plugin-forge" to build a Claude Code
extension: [PLUGIN_NAME] — [PLUGIN_DESCRIPTION].

Spawn 4 specialist teammates:

1. "creator" (plugin-dev:agent-creator) - Generate the plugin structure:
   plugin.json manifest, agent definitions (.md files with proper
   frontmatter: name, version, color, description, model, memory),
   command definitions with argument hints and allowed-tools, and skill
   files. Follow MyClaude plugin conventions for file paths and
   metadata. Owns agents/, commands/, skills/, plugin.json.

2. "hook-designer" (hookify:conversation-analyzer) - Analyze conversation
   patterns to identify behaviors that should be prevented or enhanced
   with hooks. Design PreToolUse and SessionStart hooks that improve
   the extension's reliability. Create hook rules with clear trigger
   conditions. Owns hooks/.

3. "quality" (dev-suite:quality-specialist) - Write comprehensive
   tests for the plugin: manifest validation, agent prompt testing,
   command argument parsing. Set up GitHub Actions for automated
   validation: lint checks, metadata validation, context budget
   checking, and test runs on PR. Owns tests/, .github/workflows/.

4. "validator" (plugin-dev:plugin-validator) - After all components are
   created, validate the complete plugin structure: check plugin.json
   schema, verify all referenced files exist, validate agent/command
   frontmatter, and confirm skill sizes are within context budget.
   Read-only.

Workflow: creator + hook-designer (parallel) → quality → validator.
```

### debug-triage

**Placeholders:** `SYMPTOMS`, `AFFECTED_MODULES`

```
Create an agent team called "debug-triage" to quickly triage a bug:
[SYMPTOMS].

Spawn 2 lightweight teammates for fast initial investigation:

1. "explorer" (feature-dev:code-explorer) - Run FIRST. Rapidly map the
   execution path through [AFFECTED_MODULES]: trace from entry point to
   failure site, identify the architectural layers involved, and document
   key dependencies. Produce a focused component map of the affected area.

2. "debugger" (dev-suite:debugger-pro) - After explorer provides the
   architecture map, perform targeted root cause analysis: examine the
   specific failure path, check for obvious issues (null/undefined access,
   off-by-one, missing error handling, type mismatches), and produce an
   initial severity assessment (P0/P1/P2). Recommend whether the bug
   needs escalation to a full debug team (debug-gui, debug-numerical,
   or debug-schema).

Workflow: explorer → debugger (sequential, not parallel).
Use this team for: initial bug investigation, severity assessment, and
routing to the appropriate specialist team. Typical runtime: 2-5 minutes.
Escalation guide:
- GUI threading bugs → /team-assemble debug-gui
- JAX/numerical bugs → /team-assemble debug-numerical
- Schema/type drift  → /team-assemble debug-schema
```

### debug-gui

**Placeholders:** `SYMPTOMS`, `AFFECTED_MODULES`

```
Create an agent team called "debug-gui" to investigate a GUI/threading bug:
[SYMPTOMS].

Spawn 4 specialist teammates using the proven Debugging Core Trio + SRE pattern:

1. "explorer" (feature-dev:code-explorer) - Run FIRST. Map the architecture:
   trace signal flows (e.g., Worker.signals.completed → Pool._on_worker_completed
   → store reducer), identify Qt thread boundaries, and document the execution
   path through [AFFECTED_MODULES]. Produce a component map before other agents
   begin targeted investigation.

2. "debugger" (dev-suite:debugger-pro) - ANCHOR agent. After explorer maps
   the architecture, perform root cause analysis: correlate logs, analyze stack
   traces, reproduce the issue. Synthesize all findings from other agents into a
   prioritized fix list (P0/P1/P2). Focus on signal safety, shiboken lifecycle,
   and singleton race conditions.

3. "python-pro" (science-suite:python-pro) - Type and contract verification.
   Check for attribute mismatches across abstraction boundaries (e.g., unit vs
   units, cancel() vs cancel_token.cancel()). Verify Protocol compliance,
   thread-safety of shared state, and API contract consistency between layers.

4. "sre" (dev-suite:sre-expert) - Threading and reliability specialist.
   Investigate Qt event loop interactions, GIL contention with background workers,
   QThread lifecycle management, and cross-thread signal/slot safety. Check for
   resource leaks, deadlocks, and race conditions in the threading model.

Workflow: explorer first → (debugger + python-pro + sre in parallel) → debugger synthesizes.
Parallelism cap: 3-4 agents max. More causes duplicate findings.
Cross-ref: if root cause is numerical/JAX → escalate to debug-numerical;
if root cause is schema/type drift → escalate to debug-schema.
```

### debug-numerical

**Placeholders:** `SYMPTOMS`, `AFFECTED_MODULES`

```
Create an agent team called "debug-numerical" to investigate a numerical/JAX bug:
[SYMPTOMS].

Spawn 4 specialist teammates using the proven Debugging Core Trio + JAX Pro pattern:

1. "explorer" (feature-dev:code-explorer) - Run FIRST. Map the computational
   pipeline: trace data flow from input through transformations to output,
   identify JIT compilation boundaries, vmap/pmap usage, and host-device
   transfer points in [AFFECTED_MODULES]. Document the numerical pipeline
   architecture before other agents begin investigation.

2. "debugger" (dev-suite:debugger-pro) - ANCHOR agent. After explorer maps
   the pipeline, perform root cause analysis: correlate NaN propagation paths,
   analyze gradient flow, and trace convergence failures. Synthesize all findings
   from other agents into a prioritized fix list (P0/P1/P2).

3. "python-pro" (science-suite:python-pro) - Type and contract verification.
   Check dtype mismatches, shape errors across function boundaries, incorrect
   array broadcasting, and API contract violations between numerical modules.
   Verify that JIT-traced functions receive consistent static arguments.

4. "jax-pro" (science-suite:jax-pro) - JAX/numerical specialist. Investigate
   JIT tracing errors, XLA compilation failures, NaN gradients, ODE solver
   divergence, custom VJP correctness, and host-device transfer overhead.
   Check for non-JIT-safe operations (e.g., Python control flow inside traced
   functions, non-interpax interpolation). Verify vmap/pmap sharding.

Workflow: explorer first → (debugger + python-pro + jax-pro in parallel) → debugger synthesizes.
Parallelism cap: 3-4 agents max. More causes duplicate findings.
Cross-ref: if root cause is GUI/threading → escalate to debug-gui;
if root cause is schema/type drift → escalate to debug-schema.
```

### debug-schema

**Placeholders:** `SYMPTOMS`, `AFFECTED_MODULES`

```
Create an agent team called "debug-schema" to investigate a schema/type drift bug:
[SYMPTOMS].

Spawn 4 specialist teammates using the proven Debugging Core Trio + Type Analyzer pattern:

1. "explorer" (feature-dev:code-explorer) - Run FIRST. Map the data flow:
   trace how data structures (dataclasses, TypedDicts, Pydantic models) flow
   across layer boundaries in [AFFECTED_MODULES]. Identify all definitions of
   the same logical type (e.g., 3 incompatible BayesianResult classes across
   worker, service, and store layers). Document the schema dependency graph.

2. "debugger" (dev-suite:debugger-pro) - ANCHOR agent. After explorer maps
   the schema landscape, perform root cause analysis: identify where schemas
   diverged, which layer introduced the incompatibility, and whether the drift
   is in field names, types, optionality, or serialization. Synthesize all
   findings into a prioritized fix list (P0/P1/P2).

3. "python-pro" (science-suite:python-pro) - Type and contract verification.
   Use Protocol analysis to check structural compatibility between type
   definitions that should be identical. Verify serialization/deserialization
   round-trips, check for missing fields, type narrowing errors, and Optional
   vs required field mismatches across abstraction boundaries.

4. "type-analyzer" (pr-review-toolkit:type-design-analyzer) - Type design
   specialist. Analyze all types involved in the drift for encapsulation
   quality, invariant expression, and enforcement. Rate each type 1-5. Flag
   types that leak implementation details, have weak invariants, or fail to
   enforce their contracts. Recommend canonical type definitions. Read-only.

Workflow: explorer first → (debugger + python-pro + type-analyzer in parallel) → debugger synthesizes.
Do NOT run type-analyzer and quality-specialist simultaneously — they overlap on interface contract checking.
Cross-ref: if root cause is GUI/threading → escalate to debug-gui;
if root cause is numerical/JAX → escalate to debug-numerical.
```

---

## Step 4: Output Format

After matching the team type and substituting variables, output:

1. A brief summary: team name, number of teammates, suites involved
2. The complete prompt in a fenced code block
3. Any remaining `[PLACEHOLDER]` values that still need user input
4. The tip: "Paste this prompt into Claude Code to create the team. Enable agent teams first: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`"

---

## Step 5: Alias Resolution

Some teams have aliases from their pre-consolidation names. Map these automatically:

| Alias | Resolves To |
|-------|-------------|
| `quality-audit` | `quality-security` |
| `security-harden` | `quality-security` |
| `code-health` | `quality-security` |
| `sci-pipeline` | `sci-compute` |
| `dl-research` | `sci-compute` |
| `md-campaign` | `md-simulation` |
| `ml-forcefield` | `md-simulation` |
| `docs-sprint` | `docs-publish` |
| `reproducible-research` | `docs-publish` |
| `full-pr-review` | `pr-review` |
| `llm-app` | `ai-engineering` |
| `ai-agent-dev` | `ai-engineering` |
| `prompt-lab` | `ai-engineering` |

When an alias is used, resolve it to the canonical team name and note the alias in the output.

---

## Error Handling

- If the team type doesn't match any template or alias: show the catalog and suggest the closest match
- If `--var` keys don't match template placeholders: warn and show available placeholders for that template
- If no arguments provided: show the catalog
