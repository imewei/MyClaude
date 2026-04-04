---
name: team-assemble
description: Generate ready-to-use agent team configurations from 34 pre-built templates for MyClaude v3.1.0
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
| `list` | Show all 34 available team configurations |
| `<team-type>` | Generate the prompt for the specified team |
| `<team-type> --var KEY=VALUE` | Generate with placeholder substitution |

**Examples:**
```bash
/team-assemble list
/team-assemble feature-dev
/team-assemble incident-response --var SYMPTOMS="API returning 500 errors on /auth endpoint"
/team-assemble sci-pipeline --var PROBLEM="Bayesian parameter estimation for SAXS data"
/team-assemble pr-review --var PR_NUMBER=142
```

---

## Step 1: Parse the Command

1. If the argument is `list`, display the team catalog table and stop.
2. Otherwise, match the argument to one of the 34 team types below.
3. If `--var` flags are provided, substitute `[PLACEHOLDER]` values in the template.
4. Output the final prompt in a fenced code block, ready to paste.

---

## Step 2: Team Catalog

When `list` is invoked, display this table:

```
Agent Team Catalog (MyClaude v3.1.0) — 34 Teams
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEVELOPMENT & OPERATIONS
 #  Type                Teammates  Suites Used              Best For
 1  feature-dev         4          dev + feature-dev plugin  Feature build + review
 2  incident-response   3          dev-suite                 Production debugging
 3  quality-audit       4          dev-suite                 Pre-release review
 4  sci-pipeline        4          science                   ML/JAX workflows
 5  infra-setup         3          dev-suite                 Cloud + CI/CD setup
 6  modernization       4          dev-suite                 Legacy migration
 7  dl-research         4          science                   Neural network dev
 8  api-design          4          dev-suite                 API development
 9  pr-review           3          dev-suite                 Critical PR review
10  ai-engineering      4          science + dev + core      AI/LLM apps + agents

SCIENTIFIC COMPUTING
11  julia-sciml         4          science                   Julia SciML / DiffEq
12  stat-phys           4          science                   Phase transitions, dynamics
13  bayesian-pipeline   4          science                   NumPyro / MCMC inference
14  md-campaign         4          science                   Molecular dynamics
15  ml-forcefield       4          science                   ML potentials (NequIP/MACE)
16  paper-implement     4          science                   Reproduce research papers

CROSS-SUITE SPECIALIZED
17  perf-optimize       4          dev + science             Performance profiling
18  hpc-interop         4          science + dev             Cross-language HPC
19  reproducible-research 4        science + dev             Open science, CI/CD
20  data-pipeline       4          science + dev             ETL, feature engineering
21  security-harden     4          dev-suite                 Security hardening
22  docs-sprint         4          dev + science             Documentation overhaul
23  monorepo-refactor   4          dev-suite                 Monorepo restructuring

OFFICIAL PLUGIN INTEGRATION
24  full-pr-review      4          pr-review-toolkit         Maximum PR scrutiny
25  feature-dev         4          feature-dev + dev-suite   Feature + review pipeline
26  agent-sdk-build     4          agent-sdk-dev + science   Agent SDK applications
27  plugin-forge        4          plugin-dev + hookify      Claude Code extensions
28  codebase-archaeology 3         feature-dev + dev-suite   Codebase understanding
29  code-health         4          pr-review + simplifier    Code quality + type safety
30  hf-ml-publish       4          huggingface + science     HuggingFace model publish

DEBUGGING
31  debug-gui           4          dev + feature + science   GUI threading, signal safety
32  debug-numerical     4          dev + feature + science   JAX/NaN, ODE solver, tracing
33  debug-schema        4          dev + feature + pr-review Schema/type drift, contracts
34  debug-triage        2          dev + feature             Quick bug triage (lightweight)

ADVANCED DEBUGGING (use debug-triage first to route)
 *  debug-full-audit    6          dev + feat + sci + pr     Comprehensive multi-phase audit

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

### quality-audit

**Placeholders:** `PROJECT_PATH`

```
Create an agent team called "quality-audit" to perform a comprehensive
code quality and security audit of [PROJECT_PATH].

Spawn 4 reviewers, each with a distinct lens:

1. "security" (dev-suite:quality-specialist) - Security auditor. Scan for
   OWASP Top 10 vulnerabilities: injection, broken auth, data exposure,
   XSS, CSRF. Review authentication flows, input validation, secret
   handling, and dependency vulnerabilities. Rate each finding:
   Critical/High/Medium/Low.

2. "architecture" (dev-suite:software-architect) - Architecture reviewer.
   Assess design patterns, SOLID principles, coupling/cohesion, cyclomatic
   complexity, and code duplication. Identify architectural anti-patterns
   and tech debt. Produce an Architecture Decision Record for any major
   concerns.

3. "testing" (dev-suite:quality-specialist) - Test coverage analyst. Map
   untested code paths, identify missing edge cases, check for flaky tests,
   and assess the testing pyramid (unit/integration/e2e ratio). Recommend
   specific tests to add.

4. "docs" (dev-suite:documentation-expert) - Documentation reviewer. Audit
   API documentation completeness, README accuracy, inline comment quality,
   and operational runbooks. Identify undocumented public APIs and missing
   error documentation.

Each reviewer works independently, then shares findings. Synthesize into
a prioritized remediation plan with effort estimates.
```

### sci-pipeline

**Placeholders:** `PROBLEM`, `REFERENCE_PAPERS`

```
Create an agent team called "sci-pipeline" to build a scientific computing
pipeline for [PROBLEM].

Spawn 4 specialist teammates:

1. "jax-engineer" (science-suite:jax-pro) - JAX implementation specialist. Implement the core
   computational kernels using JAX with JIT compilation, vmap for
   batching, and pmap for multi-device parallelism. Handle GPU memory
   management and custom VJPs. Owns src/core/, src/kernels/.

2. "ml-engineer" (science-suite:ml-expert) - ML pipeline architect. Set up experiment tracking
   (W&B/MLflow), hyperparameter optimization (Optuna), data loading
   pipelines, and model versioning. Owns src/training/, src/data/,
   configs/.

3. "python-architect" (science-suite:python-pro) - Systems integration. Design the package structure
   with proper typing (Protocols, Generics), CLI interface, configuration
   management, and test infrastructure. Owns src/__init__.py, setup files,
   src/cli/, pyproject.toml.

4. "methodology" (science-suite:research-expert) - Research methodology validator. Review the computational
   approach for scientific correctness, reproducibility (explicit seeds,
   deterministic ops), and statistical validity. Validate against
   [REFERENCE_PAPERS]. Owns docs/, notebooks/.

Ensure JAX-first architecture: minimize host-device transfers, use
interpax for interpolation, mandatory ArviZ diagnostics for Bayesian work.
Use Sonnet for all teammates.
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

### dl-research

**Placeholders:** `RESEARCH_TASK`, `BASELINE_PAPER`

```
Create an agent team called "dl-research" to develop a neural network
for [RESEARCH_TASK].

Spawn 4 deep learning specialists:

1. "nn-architect" (science-suite:neural-network-master) - Neural architecture designer. Design the model
   architecture considering: attention mechanisms, normalization strategies,
   activation functions, and parameter efficiency. Analyze gradient flow
   and provide theoretical justification. Reference [BASELINE_PAPER].
   Owns src/models/.

2. "jax-implementer" (science-suite:jax-pro) - JAX/Flax implementation engineer. Implement the
   architecture using Flax Linen (or Equinox if functional style preferred).
   Ensure JIT compatibility, efficient memory usage, and multi-device
   training with pmap/sharding. Implement custom training loops with
   gradient clipping and mixed precision. Owns src/training/, src/utils/.

3. "mlops-engineer" (science-suite:ml-expert) - Training infrastructure. Set up experiment tracking
   (W&B), hyperparameter sweeps, checkpoint management, and model
   evaluation pipelines. Configure data loading with proper prefetching
   and augmentation. Owns configs/, scripts/, src/data/.

4. "researcher" (science-suite:research-expert) - Scientific validation. Implement evaluation metrics
   from the reference paper, set up ablation studies, analyze training
   diagnostics (loss landscapes, gradient norms, attention patterns).
   Ensure reproducibility with explicit seeds. Owns notebooks/,
   evaluation/.

Use mandatory ArviZ diagnostics for any Bayesian components.
```

### api-design

**Placeholders:** `API_TYPE`, `SERVICE_NAME`

```
Create an agent team called "api-design" to design and implement a
[API_TYPE] API for [SERVICE_NAME].

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

### pr-review

**Placeholders:** `PR_NUMBER`

```
Create an agent team called "pr-review" to review PR #[PR_NUMBER] from
multiple perspectives.

Spawn 3 specialized reviewers:

1. "security-reviewer" (dev-suite:quality-specialist) - Security-focused
   review. Check for: injection vulnerabilities, broken authentication,
   sensitive data exposure, missing input validation, insecure defaults,
   and dependency vulnerabilities. Rate each finding: Critical/High/Medium/Low.

2. "performance-reviewer" (dev-suite:systems-engineer) - Performance-focused
   review. Check for: O(n^2) algorithms, N+1 queries, unnecessary memory
   allocations, missing caching opportunities, blocking operations, and
   resource leaks. Profile hot paths and suggest optimizations.

3. "correctness-reviewer" (dev-suite:debugger-pro) - Logic and correctness
   review. Check for: off-by-one errors, race conditions, null/undefined
   handling, edge cases, error propagation, and breaking changes to public
   APIs. Verify test coverage for all new code paths.

Each reviewer must:
- Run `gh pr diff [PR_NUMBER]` to get the changes
- Focus ONLY on changed files and their immediate dependencies
- Provide specific line references for each finding
- Suggest concrete fixes, not just problem descriptions

Have reviewers share findings with each other and debate disagreements.
Synthesize into a unified review with prioritized action items.
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

### stat-phys

**Placeholders:** `PHYSICAL_SYSTEM`, `PHENOMENON`

```
Create an agent team called "stat-phys" to investigate [PHYSICAL_SYSTEM]
with focus on [PHENOMENON] (e.g., phase transitions, correlation functions,
non-equilibrium dynamics).

Spawn 4 specialist teammates:

1. "theorist" (science-suite:statistical-physicist) - Derive the theoretical framework:
   partition function, order parameters, critical exponents, scaling
   relations. Formulate the Langevin/Fokker-Planck equations if
   non-equilibrium. Identify universality class and relevant symmetries.
   Predict expected behavior to validate simulations. Owns theory/, docs/.

2. "gpu-compute" (science-suite:jax-pro) - JAX computation specialist. Implement GPU-accelerated
   Monte Carlo or molecular dynamics using JAX. Use vmap for ensemble
   averaging, pmap for multi-GPU scaling. Implement efficient correlation
   function computation (FFT-based), histogram reweighting, and
   finite-size scaling analysis. Owns src/compute/, src/analysis/.

3. "simulator" (science-suite:simulation-expert) - Simulation architect. Design the simulation protocol:
   equilibration criteria, production run lengths, sampling strategies
   (replica exchange, Wang-Landau, umbrella sampling). Implement
   observables: structure factor S(q), radial distribution g(r),
   mean-square displacement, susceptibility. Owns src/simulation/.

4. "researcher" (science-suite:research-expert) - Methodology and literature. Survey existing results
   for [PHYSICAL_SYSTEM], identify open questions, validate simulation
   results against known benchmarks. Prepare publication-quality figures
   using scientific visualization best practices. Owns papers/, figures/.

Ensure: explicit random seeds, ArviZ diagnostics for any Bayesian
components, proper error estimation (jackknife/bootstrap).
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

4. "methodology" (science-suite:jax-pro) - Research methodology. Ensure MCMC convergence
   diagnostics are comprehensive: R-hat (<1.01), ESS (>400/chain),
   BFMI (>0.3), divergence checks, trace plots. Document all modeling
   choices and sensitivity analyses. Owns docs/, notebooks/.

Mandatory: ArviZ for all diagnostics. NLSQ warm-start before NUTS.
Explicit seeds for reproducibility.
```

### md-campaign

**Placeholders:** `SYSTEM`, `PROPERTY`, `FORCE_FIELD`

```
Create an agent team called "md-campaign" to run a molecular dynamics
campaign for [SYSTEM] studying [PROPERTY].

Spawn 4 specialist teammates:

1. "md-architect" (science-suite:simulation-expert) - Simulation setup specialist. Design the simulation
   protocol: system construction (particle placement, box geometry),
   force field selection ([FORCE_FIELD]), ensemble (NVT/NPT/NVE),
   thermostat/barostat settings, integration timestep, and cutoff
   schemes. Handle equilibration protocol with staged heating/cooling
   if needed. Owns simulations/, configs/.

2. "gpu-engine" (science-suite:jax-pro) - JAX-MD implementation. Implement the simulation engine
   using JAX-MD or custom JAX kernels. Optimize neighbor list updates,
   force computation (JIT-compiled), and trajectory output. Handle
   multi-GPU scaling with pmap. Implement enhanced sampling methods
   (metadynamics, replica exchange) if needed. Owns src/engine/, src/sampling/.

3. "analyst" (science-suite:statistical-physicist) - Thermodynamic and structural analysis. Compute observables:
   radial distribution function g(r), structure factor S(q), mean-square
   displacement (diffusion), velocity autocorrelation, pressure tensor,
   free energy profiles. Implement block averaging for error estimation.
   Owns src/analysis/, results/.

4. "workflow-engineer" (science-suite:python-pro) - Pipeline automation. Build the campaign workflow:
   parameter sweep management, job scheduling, trajectory storage
   (HDF5/MDAnalysis), checkpoint/restart logic, and automated convergence
   checking. Owns scripts/, workflows/, src/io/.

Ensure: proper equilibration verification, production run length
justified by autocorrelation analysis, explicit seeds.
```

### ml-forcefield

**Placeholders:** `CHEMICAL_SYSTEM`

```
Create an agent team called "ml-forcefield" to develop a machine learning
force field for [CHEMICAL_SYSTEM].

Spawn 4 specialist teammates:

1. "nn-designer" (science-suite:neural-network-master) - Neural network architect. Design the ML potential
   architecture: equivariant message passing (E(3)-equivariant),
   interaction layers, radial basis functions, and output heads
   (energy, forces, stress). Choose between NequIP-style vs MACE-style
   architectures based on accuracy/speed trade-offs. Owns src/models/.

2. "data-engineer" (science-suite:simulation-expert) - Training data specialist. Curate DFT training data:
   structure selection (active learning, FPS), energy/force/stress labels,
   train/val/test splits ensuring chemical diversity. Implement data
   augmentation (rotation, translation). Design the training distribution
   to cover relevant PES regions. Owns data/, src/datasets/.

3. "jax-trainer" (science-suite:jax-pro) - Training engineer. Implement the training loop in
   JAX/Flax with: per-atom energy loss + force matching loss (weighted),
   learning rate scheduling (cosine with warmup), gradient clipping,
   EMA weights. Multi-GPU training with sharding. Checkpoint management
   and early stopping. Owns src/training/, configs/.

4. "validator" (science-suite:research-expert) - Scientific validation. Benchmark against DFT reference:
   energy MAE, force MAE/RMSE, equation of state, phonon dispersion,
   elastic constants. Run stability tests: NVE energy drift, radial
   distribution function comparison, melting point prediction.
   Owns evaluation/, benchmarks/, notebooks/.

Ensure physical symmetries are built into architecture, not learned.
```

### paper-implement

**Placeholders:** `PAPER_TITLE`, `PAPER_URL`

```
Create an agent team called "paper-implement" to reproduce results from
[PAPER_TITLE] ([PAPER_URL]).

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

### hpc-interop

**Placeholders:** `COMPUTATIONAL_TASK`

```
Create an agent team called "hpc-interop" to build a cross-language HPC
pipeline for [COMPUTATIONAL_TASK].

Spawn 4 specialist teammates:

1. "julia-lead" (science-suite:julia-pro) - Julia performance engineer. Implement core numerical
   algorithms in Julia for maximum performance: type-stable code,
   @inbounds/@simd annotations, pre-allocation, and LoopVectorization.jl.
   Build Julia package with proper Project.toml and test suite.
   Owns julia/, Project.toml.

2. "python-integrator" (science-suite:python-pro) - Python ecosystem bridge. Build Python wrappers
   using PythonCall.jl (Julia to Python) or juliacall (Python to Julia).
   Handle data serialization (Arrow/HDF5 for zero-copy), memory
   management across language boundaries, and error propagation.
   Owns python/, src/bridge/.

3. "systems-engineer" (dev-suite:systems-engineer) - FFI and compiled extensions. Build C/Rust
   extensions for performance-critical inner loops using PyO3 (Rust)
   or cffi (C). Handle memory layout compatibility (row vs column major),
   thread safety, and SIMD intrinsics. Owns extensions/, src/native/.

4. "gpu-architect" (science-suite:jax-pro) - GPU compute layer. Implement GPU kernels in JAX
   for embarrassingly parallel operations. Handle CPU-GPU data
   transfer optimization, kernel fusion, and multi-GPU distribution.
   Ensure numerical equivalence with CPU reference implementation.
   Owns src/gpu/, benchmarks/.

Key constraint: zero-copy data transfer between languages where possible.
Benchmark each language boundary to quantify overhead.
```

### reproducible-research

**Placeholders:** `PROJECT_NAME`

```
Create an agent team called "reproducible-research" to make [PROJECT_NAME]
fully reproducible and publishable.

Spawn 4 specialist teammates:

1. "methodology" (science-suite:research-expert) - Research reproducibility expert. Audit the current
   codebase for reproducibility gaps: hardcoded paths, missing seeds,
   undocumented parameters, version-dependent behavior. Create a
   reproducibility checklist and ensure all experiments can be re-run
   from a single command. Owns docs/methodology/.

2. "packager" (science-suite:python-pro) - Python packaging specialist. Structure the project as
   an installable package with pyproject.toml, proper dependency
   pinning (uv.lock), entry points for all scripts, and typed
   configuration objects. Set up dev/test/docs dependency groups.
   Owns pyproject.toml, src/__init__.py, configs/.

3. "ci-engineer" (dev-suite:automation-engineer) - Automation specialist. Build GitHub Actions workflows:
   automated testing on every push, notebook execution verification,
   figure regeneration, dependency security scanning, and automated
   release with changelog generation. Set up pre-commit hooks. Owns
   .github/workflows/, .pre-commit-config.yaml.

4. "doc-writer" (dev-suite:documentation-expert) - Research documentation. Create comprehensive docs:
   installation guide, quickstart tutorial, API reference (autodoc),
   mathematical derivation appendix, and a "reproducing our results"
   guide. Convert key notebooks to Sphinx gallery examples. Owns
   docs/, notebooks/.

Goal: anyone should be able to clone, install, and reproduce all results
with: uv sync && uv run reproduce-all
```

### data-pipeline

**Placeholders:** `DATA_SOURCE`, `ML_TARGET`

```
Create an agent team called "data-pipeline" to build a data pipeline
for [DATA_SOURCE] feeding [ML_TARGET].

Spawn 4 specialist teammates:

1. "data-engineer" (science-suite:ml-expert) - Pipeline architect. Build the ETL/ELT pipeline:
   data ingestion (batch/streaming), transformation logic (pandas/polars/
   dask), schema validation (pandera), and output sinks (parquet/Delta
   Lake). Handle incremental processing and idempotency. Owns
   src/pipeline/, src/transforms/.

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

### security-harden

**Placeholders:** `PROJECT_OR_INFRASTRUCTURE`

```
Create an agent team called "security-harden" to perform security
hardening of [PROJECT_OR_INFRASTRUCTURE].

Spawn 4 security specialists:

1. "appsec" (dev-suite:quality-specialist) - Application security expert. Audit for OWASP Top 10:
   injection, broken auth, sensitive data exposure, XXE, broken access
   control, security misconfiguration, XSS, insecure deserialization,
   vulnerable components, insufficient logging. Implement fixes with
   defense-in-depth. Owns src/security/, src/middleware/.

2. "infra-sec" (dev-suite:devops-architect) - Infrastructure security. Harden: network segmentation,
   firewall rules, TLS configuration, secrets management (Vault/SOPS),
   IAM policies (least privilege), container security (distroless images,
   read-only filesystem), and Kubernetes security policies (PSP/OPA).
   Owns infra/security/, k8s/policies/.

3. "systems-sec" (dev-suite:systems-engineer) - Systems-level security. Review: memory safety (buffer
   overflows, use-after-free), privilege escalation vectors, file
   permission hardening, syscall filtering (seccomp), and binary
   protections (ASLR, stack canaries, PIE). Owns security/system/.

4. "secops" (dev-suite:automation-engineer) - Security automation. Build security CI/CD: SAST (Semgrep/
   CodeQL), DAST (OWASP ZAP), dependency scanning (Dependabot/Snyk),
   container scanning (Trivy), secret detection (TruffleHog), and
   compliance checks. Set up security alerts and incident runbooks.
   Owns .github/workflows/security.yml, scripts/security/.

Output: prioritized finding report with CVSS scores, remediation
steps, and verification procedures.
```

### docs-sprint

**Placeholders:** `PROJECT_NAME`

```
Create an agent team called "docs-sprint" to create comprehensive
documentation for [PROJECT_NAME].

Spawn 4 specialist teammates:

1. "docs-architect" (dev-suite:documentation-expert) - Documentation structure designer. Design the
   information architecture: getting started guide, tutorials (learning-
   oriented), how-to guides (task-oriented), reference (API docs), and
   explanation (understanding-oriented) following Diataxis framework.
   Set up Sphinx/MkDocs with proper theme and navigation. Owns docs/.

2. "technical-writer" (dev-suite:quality-specialist) - Content accuracy validator. Review all
   documentation for technical accuracy by cross-referencing with source
   code. Ensure code examples compile and run, CLI flags match actual
   behavior, and configuration options are complete. Fix any stale or
   misleading content. Owns docs/reference/.

3. "tutorial-builder" (science-suite:research-expert) - Interactive examples. Create tutorials with
   runnable code examples, Jupyter notebooks for interactive exploration,
   and a cookbook of common patterns. Build a docs testing harness that
   validates all code snippets. Owns docs/tutorials/, notebooks/.

4. "architecture-writer" (dev-suite:software-architect) - System documentation. Create architecture
   decision records (ADRs), system diagrams (Mermaid), component
   interaction documentation, deployment guides, and operational
   runbooks. Document all configuration options with defaults and
   examples. Owns docs/architecture/, docs/operations/.

Standard: every public API must have docstring + reference page +
at least one usage example.
```

### monorepo-refactor

**Placeholders:** `REPOSITORY`

```
Create an agent team called "monorepo-refactor" to restructure
[REPOSITORY] into a well-organized monorepo (or split it into separate
packages).

Spawn 4 specialist teammates:

1. "architect" (dev-suite:software-architect) - Module boundary designer. Analyze dependency graphs,
   identify circular dependencies, define clean module boundaries with
   explicit public APIs, and design the target directory structure.
   Create a migration plan with atomic, reversible steps. Require plan
   approval before implementation. Owns docs/architecture/.

2. "build-engineer" (dev-suite:automation-engineer) - CI/CD and build system. Configure the build tool
   (Turborepo/Nx/Pants/Bazel): task pipelines, caching, affected
   detection (only build/test changed packages), and dependency
   graph-based ordering. Optimize CI for monorepo: path filtering,
   incremental builds, shared caches. Owns build configs, .github/workflows/.

3. "quality-guard" (dev-suite:quality-specialist) - Testing and standards. Ensure test isolation (each
   package testable independently), shared lint/format configurations,
   consistent versioning strategy (independent or lockstep), and
   cross-package integration tests. Set up git hooks for pre-commit
   validation. Owns tests/, .eslintrc, .prettierrc.

4. "systems-optimizer" (dev-suite:systems-engineer) - Build performance. Profile build times, optimize
   dependency resolution, implement workspace caching strategies, and
   minimize cold build times. Handle shared dependencies (hoisting vs
   isolation trade-off). Owns performance benchmarks.

Key constraint: migration must be atomic - the repo must build and pass
tests at every intermediate step.
```

### full-pr-review

**Placeholders:** `PR_OR_BRANCH`

```
Create an agent team called "full-pr-review" to perform an exhaustive
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

### agent-sdk-build

**Placeholders:** `AGENT_DESCRIPTION`, `LANGUAGE`

```
Create an agent team called "agent-sdk-build" to build a production
Agent SDK application: [AGENT_DESCRIPTION].

Target language: [LANGUAGE] (typescript or python)

Spawn 4 specialist teammates:

1. "ai-engineer" (science-suite:ai-engineer) - Design the agent
   architecture: tool definitions, agent orchestration patterns,
   memory/context management, and multi-agent coordination if needed.
   Implement the core agent logic using Claude Agent SDK patterns.
   Owns src/agents/, src/tools/.

2. "prompt-designer" (science-suite:prompt-engineer) - Craft production
   system prompts using chain-of-thought, constitutional AI principles,
   and few-shot examples. Optimize for accuracy and safety. Create
   evaluation test cases. Owns prompts/, system instructions.

3. "api-architect" (dev-suite:software-architect) - Build the
   API layer and infrastructure: REST/WebSocket endpoints, streaming
   support, authentication, rate limiting, and deployment config.
   Owns src/api/, infrastructure/.

4. "verifier" (agent-sdk-dev:agent-sdk-verifier-[LANGUAGE]) - After
   implementation is complete, verify the application follows SDK best
   practices, proper tool use patterns, and documentation
   recommendations. Flag any misconfigurations. Read-only.

Workflow: ai-engineer + prompt-designer + api-architect (parallel) → verifier.
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
Optional: For extensions with SDK components, add a 5th teammate
"sdk-verifier" (agent-sdk-dev:agent-sdk-verifier-[LANGUAGE]) for
SDK best practices validation.
```

### codebase-archaeology

**Placeholders:** `REPOSITORY`, `FOCUS_AREAS`

```
Create an agent team called "codebase-archaeology" to deeply analyze
and document [REPOSITORY].

Focus areas: [FOCUS_AREAS]

Spawn 3 specialist teammates:

1. "explorer" (feature-dev:code-explorer) - Trace execution paths
   through the codebase starting from entry points. Map the architecture
   layers, identify patterns and abstractions, document dependencies
   between modules. Produce a component map with data flow diagrams.

2. "documenter" (dev-suite:documentation-expert) - Transform the
   explorer's findings into clear, comprehensive architecture
   documentation. Create: system overview, component catalog, API
   reference, dependency map, and onboarding guide. Write for a
   developer joining the team.

3. "researcher" (science-suite:research-expert) - Synthesize findings
   into actionable insights: identify technical debt hotspots,
   architectural anti-patterns, and modernization opportunities.
   Produce a prioritized recommendations report with effort estimates.

Workflow: explorer → documenter + researcher (parallel).
Output: docs/architecture/ with overview, components, and recommendations.
```

### code-health

**Placeholders:** `TARGET_PATH`, `LANGUAGE`

```
Create an agent team called "code-health" to improve code quality in
[TARGET_PATH] using [LANGUAGE] (Python/TypeScript).

Spawn 4 specialist teammates:

1. "simplifier" (code-simplifier:code-simplifier) - Review all files in
   the target path. Simplify overly complex functions, remove dead code,
   consolidate duplicate logic, and improve naming. Preserve all
   functionality — every change must be behavior-preserving. Focus on
   recently modified files first.

2. "type-engineer" (science-suite:python-pro) - Add or improve type
   annotations throughout the target path. For Python: use Protocols,
   Generics, and strict typing patterns. For TypeScript: use branded
   types, discriminated unions, and const assertions. Convert Any types
   to specific types. Ensure all public APIs have complete type
   signatures. Run mypy/tsc in strict mode.

3. "type-reviewer" (pr-review-toolkit:type-design-analyzer) - Analyze
   all type definitions (new and existing) for encapsulation quality,
   invariant expression, and enforcement. Rate each type 1-5. Flag
   types that leak implementation details or fail to express their
   invariants. Read-only.

4. "enforcer" (dev-suite:quality-specialist) - After simplifier and
   type-engineer complete, run the full test suite and verify no
   regressions. Review all changes for security implications. Add tests
   for any uncovered edge cases discovered during the sprint.

Workflow: (simplifier + type-engineer) in parallel → type-reviewer → enforcer.
Constraint: all existing tests must pass after every change.
Success criteria: zero type errors in strict mode, all tests pass.
```

### hf-ml-publish

**Placeholders:** `MODEL_TYPE`, `TASK_DESCRIPTION`, `FRAMEWORK`

```
Create an agent team called "hf-ml-publish" to train and publish
[MODEL_TYPE] for [TASK_DESCRIPTION] to HuggingFace Hub.

Spawn 4 specialist teammates:

1. "ml-engineer" (science-suite:ml-expert) - Design the training
   pipeline: model selection, hyperparameter configuration, data
   preprocessing, training loop with proper logging, and checkpoint
   management. Use [FRAMEWORK] (scikit-learn/XGBoost/PyTorch).
   Owns src/training/, configs/.

2. "coder" (science-suite:python-pro) - Build the data pipeline and
   utilities: data loading, feature engineering, preprocessing
   transforms, and evaluation metrics. Ensure type safety, proper
   packaging with pyproject.toml, and clean module structure.
   Owns src/data/, src/utils/.

3. "publisher" (huggingface-skills:AGENTS) - After training completes,
   publish to HuggingFace Hub: create model card with performance
   metrics, upload model weights, create dataset card if applicable,
   and set up inference endpoint. Handle versioning and metadata.

4. "evaluator" (science-suite:research-expert) - Design and run
   evaluation: benchmark selection, statistical significance testing,
   comparison with baselines, and visualization of results. Produce
   a reproducibility report. Owns evaluation/, benchmarks/.

Workflow: coder → ml-engineer → (publisher + evaluator in parallel).
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

### debug-full-audit

**Placeholders:** `SYMPTOMS`, `AFFECTED_MODULES`

```
Create an agent team called "debug-full-audit" to perform a comprehensive
multi-phase debugging audit: [SYMPTOMS].

Spawn 6 specialist teammates — the full Debugging Core Trio plus all 3
specialist angles. Strict phased workflow keeps active parallelism at 3-4
agents per phase (the proven sweet spot).

CORE TRIO (active in all phases):

1. "explorer" (feature-dev:code-explorer) - Run in Phase 1 (solo). Map the
   full architecture: execution paths, thread boundaries, data flow, schema
   dependency graph, and JIT compilation boundaries across [AFFECTED_MODULES].
   Produce a comprehensive component map that all other agents will reference.

2. "debugger" (dev-suite:debugger-pro) - ANCHOR agent. Active from Phase 2
   onward. Coordinates investigation, cross-references findings from all
   specialists, and produces the final synthesized report. Owns the
   prioritized fix list (P0/P1/P2) with evidence chains.

3. "python-pro" (science-suite:python-pro) - Active from Phase 2 onward.
   Type and contract verification across ALL boundaries: attribute mismatches,
   Protocol compliance, dtype/shape errors, serialization round-trips, and
   API contract consistency between layers.

ROTATING SPECIALISTS (one per phase, sequential):

4. "sre" (dev-suite:sre-expert) - Phase 2: Threading & reliability.
   Qt event loop interactions, GIL contention, QThread lifecycle, cross-thread
   signal/slot safety, resource leaks, deadlocks, and race conditions.

5. "jax-pro" (science-suite:jax-pro) - Phase 3: Numerical & JAX. JIT tracing
   errors, XLA compilation failures, NaN gradients, ODE solver divergence,
   custom VJP correctness, host-device transfer overhead, non-JIT-safe
   operations, and vmap/pmap sharding issues.

6. "type-analyzer" (pr-review-toolkit:type-design-analyzer) - Phase 4: Schema
   & type design. Analyze all types for encapsulation quality, invariant
   expression, and enforcement. Rate each type 1-5. Flag types that leak
   implementation details or fail to enforce contracts. Read-only.

WORKFLOW (strict phasing):
  Phase 1: explorer maps architecture (solo)
  Phase 2: debugger + python-pro + sre investigate threading/reliability
  Phase 3: debugger + python-pro + jax-pro investigate numerical/JAX
  Phase 4: type-analyzer reviews all types identified in Phases 2-3 (read-only)
  Phase 5: debugger produces final synthesis — unified P0/P1/P2 fix list

CRITICAL RULES:
- Never run more than 4 agents in parallel (diminishing returns above that)
- type-analyzer runs AFTER code investigation phases (it needs their findings)
- debugger is the single source of truth for the final report
- Each specialist focuses on their domain — do not duplicate another's work
```

---

## Step 4: Output Format

After matching the team type and substituting variables, output:

1. A brief summary: team name, number of teammates, suites involved
2. The complete prompt in a fenced code block
3. Any remaining `[PLACEHOLDER]` values that still need user input
4. The tip: "Paste this prompt into Claude Code to create the team. Enable agent teams first: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`"

---

## Error Handling

- If the team type doesn't match any template: show the catalog and suggest the closest match
- If `--var` keys don't match template placeholders: warn and show available placeholders for that template
- If no arguments provided: show the catalog
