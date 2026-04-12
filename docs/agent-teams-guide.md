# Agent Teams Guide for MyClaude Plugin Suites

> 25 ready-to-use team configurations leveraging 24 MyClaude agents + 20 official plugin agents across 3 suites.
>
> **v3.1.0:** Skills use a hub-skill routing architecture (26 hubs → 180 sub-skills as of v3.2.0). Agent teams reference agents directly and are unaffected by skill consolidation.
>
> **v3.1.4:** `/team-assemble` is now a codebase-aware recommender — run it with no arguments in a project root to get team suggestions. Four new teams added: `nonlinear-dynamics`, `julia-ml`, `multi-agent-systems`, `sci-desktop`.

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

These agents come from [claude-plugins-official](https://github.com/anthropics/claude-plugins-official) and complement MyClaude domain experts with quality gates and specialized workflows.

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
| **coderabbit** | code-reviewer | `coderabbit:code-reviewer` | AI-powered code review |
| **code-simplifier** | code-simplifier | `code-simplifier:code-simplifier` | Code clarity and refinement |
| **agent-sdk-dev** | agent-sdk-verifier-ts | `agent-sdk-dev:agent-sdk-verifier-ts` | TS Agent SDK validation |
| | agent-sdk-verifier-py | `agent-sdk-dev:agent-sdk-verifier-py` | Python Agent SDK validation |
| **plugin-dev** | agent-creator | `plugin-dev:agent-creator` | Claude Code agent generation |
| | skill-reviewer | `plugin-dev:skill-reviewer` | Skill quality review |
| | plugin-validator | `plugin-dev:plugin-validator` | Plugin structure validation |
| **superpowers** | code-reviewer | `superpowers:code-reviewer` | Plan adherence review |
| **hookify** | conversation-analyzer | `hookify:conversation-analyzer` | Behavior analysis for hooks |

---

## Quick Reference

### Development & Operations (Teams 1-7)

| # | Team | Best For | Agents Used | Teammates |
|---|------|----------|-------------|-----------|
| 1 | feature-dev | Feature build + review | code-architect + app-dev + architect + reviewer | 4 |
| 2 | incident-response | Production debugging | debugger + sre + devops | 3 |
| 3 | pr-review | Comprehensive PR review | 4 pr-review-toolkit analyzers | 4 |
| 4 | quality-security | Quality + security audit | quality + architect + quality + automation | 4 |
| 5 | api-design | API development | architect + app-dev + quality + docs | 4 |
| 6 | infra-setup | Cloud + CI/CD setup | devops + automation + sre | 3 |
| 7 | modernization | Legacy migration | architect + systems + quality + docs | 4 |

### Scientific Computing (Teams 8-12)

| # | Team | Best For | Agents Used | Teammates |
|---|------|----------|-------------|-----------|
| 8 | sci-compute | JAX/ML/DL pipelines | jax + nn-master + ml + research | 4 |
| 9 | bayesian-pipeline | NumPyro / MCMC inference | stat-physicist + jax + ml + research | 4 |
| 10 | julia-sciml | Julia SciML / DiffEq | julia + simulation + research + julia-ml-hpc | 4 |
| 11 | md-simulation | Molecular dynamics + ML FF | simulation + jax + stat-physicist + research | 4 |
| 12 | paper-implement | Reproduce research papers | research + python + jax + ml | 4 |

### Cross-Cutting (Teams 13-16)

| # | Team | Best For | Agents Used | Teammates |
|---|------|----------|-------------|-----------|
| 13 | ai-engineering | AI/LLM apps + agents | ai-eng + prompt-eng + architect + reasoning | 4 |
| 14 | perf-optimize | Performance profiling | systems + jax + debugger + python | 4 |
| 15 | data-pipeline | ETL, feature engineering | ml + ml + devops + quality | 4 |
| 16 | docs-publish | Documentation + reproducibility | docs + quality + research + automation | 4 |

### Plugin Development (Team 17)

| # | Team | Best For | Agents Used | Teammates |
|---|------|----------|-------------|-----------|
| 17 | plugin-forge | Claude Code extensions | plugin-dev + hookify + quality + validator | 4 |

### Debugging (Teams 18-21)

| # | Team | Best For | Agents Used | Teammates |
|---|------|----------|-------------|-----------|
| 18 | debug-triage | Quick bug triage (lightweight) | debugger + explorer | 2 |
| 19 | debug-gui | GUI threading, signal safety | explorer + debugger + python-pro + sre | 4 |
| 20 | debug-numerical | JAX/NaN, ODE solver, tracing | explorer + debugger + python-pro + jax-pro | 4 |
| 21 | debug-schema | Schema/type drift, contracts | explorer + debugger + python-pro + type-analyzer | 4 |

---

## Team 1: Feature Development

**When:** Building new features — full-stack, backend-only, or frontend-only — with a design-first pipeline and automated review gate.
**Suites:** dev-suite, feature-dev plugin, pr-review-toolkit

| Role | Agent Type | File Ownership |
|------|-----------|----------------|
| architect | `feature-dev:code-architect` | docs/design/, architecture decisions |
| builder | `dev-suite:app-developer` | src/components/, src/pages/, src/hooks/ |
| backend | `dev-suite:software-architect` | src/api/, src/services/, src/models/ |
| reviewer | `pr-review-toolkit:code-reviewer` | Read-only review |

### Prompt

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

---

## Team 2: Production Incident Response

**When:** Debugging a production issue with unknown root cause.
**Suites:** dev-suite

| Role | Agent Type | Focus Area |
|------|-----------|------------|
| debugger | `dev-suite:debugger-pro` | Code-level root cause analysis |
| sre | `dev-suite:sre-expert` | Metrics, logs, traces |
| infra | `dev-suite:devops-architect` | Infrastructure investigation |

### Prompt

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

---

## Team 3: PR Review

**When:** Critical PRs needing comprehensive, multi-perspective review — security-sensitive changes, major refactors, or pre-release reviews.
**Suites:** pr-review-toolkit
**Aliases:** `full-pr-review`

| Role | Agent Type | Focus Area |
|------|-----------|------------|
| code-reviewer | `pr-review-toolkit:code-reviewer` | Style, guidelines, bugs, comments |
| failure-hunter | `pr-review-toolkit:silent-failure-hunter` | Error handling, fallbacks |
| test-analyzer | `pr-review-toolkit:pr-test-analyzer` | Test coverage gaps |
| type-analyzer | `pr-review-toolkit:type-design-analyzer` | Type invariants |

### Prompt

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

---

## Team 4: Quality & Security Audit

**When:** Comprehensive codebase review before a release, compliance audit, or security hardening sprint.
**Suites:** dev-suite
**Aliases:** `quality-audit`, `security-harden`, `code-health`

| Role | Agent Type | Review Lens |
|------|-----------|-------------|
| security | `dev-suite:quality-specialist` | OWASP, vulnerabilities, auth |
| architecture | `dev-suite:software-architect` | Design patterns, SOLID, complexity |
| testing | `dev-suite:quality-specialist` | Test coverage, edge cases |
| secops | `dev-suite:automation-engineer` | Security CI/CD automation |

### Prompt

```
Create an agent team called "quality-security" to perform a comprehensive
code quality, architecture, and security audit of [PROJECT_PATH].

Spawn 4 reviewers, each with a distinct lens:

1. "security" (dev-suite:quality-specialist) - Application security auditor.
   Scan for OWASP Top 10 vulnerabilities: injection, broken auth, data
   exposure, XSS, CSRF, insecure deserialization. Review authentication
   flows, input validation, secret handling, and dependency vulnerabilities.
   Check container security, TLS configuration, and IAM policies if
   applicable. Rate each finding: Critical/High/Medium/Low with CVSS scores.

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

---

## Team 5: API Design & Integration

**When:** Designing, building, and documenting a public or internal API.
**Suites:** dev-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| api-designer | `dev-suite:software-architect` | API specification |
| implementer | `dev-suite:app-developer` | Endpoint implementation |
| tester | `dev-suite:quality-specialist` | Contract testing |
| docs-writer | `dev-suite:documentation-expert` | API documentation |

### Prompt

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

---

## Team 6: Infrastructure & DevOps Setup

**When:** Setting up cloud infrastructure, CI/CD, and observability from scratch.
**Suites:** dev-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| cloud-architect | `dev-suite:devops-architect` | IaC, cloud resources |
| cicd-engineer | `dev-suite:automation-engineer` | Pipelines, Git workflows |
| sre-lead | `dev-suite:sre-expert` | Observability stack |

### Prompt

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

---

## Team 7: Legacy Modernization

**When:** Migrating a legacy codebase to modern architecture.
**Suites:** dev-suite

| Role | Agent Type | Focus |
|------|-----------|-------|
| architect | `dev-suite:software-architect` | Target architecture |
| implementer | `dev-suite:systems-engineer` | Migration code |
| qa-lead | `dev-suite:quality-specialist` | Regression testing |
| docs-lead | `dev-suite:documentation-expert` | Migration documentation |

### Prompt

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

---

## Team 8: Scientific Computing Pipeline

**When:** Building ML/JAX/deep learning scientific computing workflows.
**Suites:** science-suite
**Aliases:** `sci-pipeline`, `dl-research`

| Role | Agent Type | Domain |
|------|-----------|--------|
| jax-engineer | `science-suite:jax-pro` | JAX core, GPU acceleration |
| architect | `science-suite:neural-network-master` | Model/algorithm design |
| ml-engineer | `science-suite:ml-expert` | Experiment tracking, MLOps |
| researcher | `science-suite:research-expert` | Methodology validation |

### Prompt

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

---

## Team 9: Bayesian Inference Pipeline

**When:** Building rigorous Bayesian analysis with NumPyro, MCMC diagnostics, and model comparison.
**Suites:** science-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| bayesian-engineer | `science-suite:statistical-physicist` | NumPyro, NUTS, GPU acceleration |
| statistician | `science-suite:research-expert` | Prior selection, model structure |
| ml-validator | `science-suite:ml-expert` | Model comparison, validation |
| methodology | `science-suite:jax-pro` | MCMC diagnostics, reproducibility |

### Prompt

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

---

## Team 10: Julia SciML Pipeline

**When:** Building scientific computing workflows with Julia's SciML ecosystem (DifferentialEquations.jl, ModelingToolkit.jl, Turing.jl).
**Suites:** science-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| julia-engineer | `science-suite:julia-pro` | Julia packages, DiffEq, SciML |
| simulation-architect | `science-suite:simulation-expert` | Physics models, numerical methods |
| methodology | `science-suite:research-expert` | Methodology, validation |
| python-bridge | `science-suite:python-pro` | Interoperability, visualization |

### Prompt

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

---

## Team 11: Molecular Dynamics & ML Force Fields

**When:** Running MD simulation campaigns or developing machine learning interatomic potentials.
**Suites:** science-suite
**Aliases:** `md-campaign`, `ml-forcefield`

| Role | Agent Type | Domain |
|------|-----------|--------|
| simulation-architect | `science-suite:simulation-expert` | MD setup, force fields, training data |
| gpu-engine | `science-suite:jax-pro` | JAX-MD, GPU acceleration, ML FF training |
| analyst | `science-suite:statistical-physicist` | Thermodynamics, structural analysis |
| researcher | `science-suite:research-expert` | Validation, workflow automation |

### Prompt

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

---

## Team 12: Scientific Paper Implementation

**When:** Reproducing results from a research paper or implementing a published algorithm.
**Suites:** science-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| paper-analyst | `science-suite:research-expert` | Paper analysis, methodology |
| python-engineer | `science-suite:python-pro` | Clean implementation |
| numerical-engineer | `science-suite:jax-pro` | Numerical implementation |
| reproducer | `science-suite:ml-expert` | Results reproduction |

### Prompt

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

---

## Team 13: AI Engineering

**When:** Building production AI applications — RAG systems, LLM-powered apps, multi-agent systems, or prompt R&D.
**Suites:** science-suite, dev-suite, agent-core
**Aliases:** `llm-app`, `ai-agent-dev`, `prompt-lab`

| Role | Agent Type | Domain |
|------|-----------|--------|
| ai-engineer | `science-suite:ai-engineer` | RAG, vector search, agent framework |
| prompt-engineer | `science-suite:prompt-engineer` | Prompt design, evaluation |
| backend-architect | `dev-suite:software-architect` | API, infrastructure |
| reasoning-architect | `agent-core:reasoning-engine` | Cognitive architecture (agent systems) |

### Prompt

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

---

## Team 14: Performance Optimization

**When:** Profiling and optimizing computational code (Python, JAX, or systems-level).
**Suites:** dev-suite, science-suite

| Role | Agent Type | Focus |
|------|-----------|-------|
| systems-profiler | `dev-suite:systems-engineer` | CPU, memory, I/O profiling |
| jax-optimizer | `science-suite:jax-pro` | JIT, vectorization, GPU |
| bottleneck-hunter | `dev-suite:debugger-pro` | Root cause, algorithmic complexity |
| python-optimizer | `science-suite:python-pro` | Python-level optimization |

### Prompt

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

---

## Team 15: Data Pipeline & Feature Engineering

**When:** Building ETL pipelines, feature stores, or MLOps data infrastructure.
**Suites:** science-suite, dev-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| data-engineer | `science-suite:ml-expert` | ETL/ELT pipeline architect |
| feature-engineer | `science-suite:ml-expert` | Feature store, drift detection |
| infra-engineer | `dev-suite:devops-architect` | Storage, orchestration |
| quality-engineer | `dev-suite:quality-specialist` | Data quality, testing |

### Prompt

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

---

## Team 16: Documentation & Reproducibility

**When:** Comprehensive documentation overhaul, open science packaging, or ensuring full research reproducibility.
**Suites:** dev-suite, science-suite
**Aliases:** `docs-sprint`, `reproducible-research`

| Role | Agent Type | Domain |
|------|-----------|--------|
| docs-architect | `dev-suite:documentation-expert` | Information architecture, Diataxis |
| content-writer | `dev-suite:quality-specialist` | Technical accuracy validation |
| tutorial-builder | `science-suite:research-expert` | Interactive examples, notebooks |
| ci-packager | `dev-suite:automation-engineer` | Automation, packaging, CI/CD |

### Prompt

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

2. "content-writer" (dev-suite:quality-specialist) - Content accuracy
   validator. Review all documentation for technical accuracy by
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

---

## Team 17: Plugin Forge

**When:** Building Claude Code extensions — plugins, hooks, agents, commands, skills, and SDK integrations.
**Suites:** plugin-dev, hookify, dev-suite

| Role | Agent Type | File Ownership |
|------|-----------|----------------|
| creator | `plugin-dev:agent-creator` | agents/, commands/, skills/, plugin.json |
| hook-designer | `hookify:conversation-analyzer` | hooks/ |
| quality | `dev-suite:quality-specialist` | tests/, .github/workflows/ |
| validator | `plugin-dev:plugin-validator` | Read-only validation |

### Prompt

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

---

## Team 18: Debug Triage

**When:** Quick initial investigation to assess severity and route to the right specialist team.
**Suites:** dev-suite, feature-dev
**Pattern:** Lightweight 2-agent triage
**Typical runtime:** 2-5 minutes

| Role | Agent Type | Focus |
|------|-----------|-------|
| explorer | `feature-dev:code-explorer` | Rapid architecture mapping (runs FIRST) |
| debugger | `dev-suite:debugger-pro` | Severity assessment + team recommendation |

### Prompt

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

---

## Team 19: Debug GUI

**When:** GUI threading bugs — signal safety, shiboken crashes, singleton races, Qt event loop issues.
**Suites:** dev-suite, feature-dev, science-suite
**Pattern:** Debugging Core Trio + SRE Expert
**See also:** Team 18 for quick triage before committing to a full team.

| Role | Agent Type | Focus |
|------|-----------|-------|
| explorer | `feature-dev:code-explorer` | Architecture mapping (runs FIRST) |
| debugger | `dev-suite:debugger-pro` | Root cause synthesis (ANCHOR) |
| python-pro | `science-suite:python-pro` | Type/contract verification |
| sre | `dev-suite:sre-expert` | Threading, resource leaks, deadlocks |

### Prompt

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

---

## Team 20: Debug Numerical

**When:** JAX/numerical bugs — NaN gradients, ODE solver divergence, JIT tracing errors, shape mismatches.
**Suites:** dev-suite, feature-dev, science-suite
**Pattern:** Debugging Core Trio + JAX Pro

| Role | Agent Type | Focus |
|------|-----------|-------|
| explorer | `feature-dev:code-explorer` | Pipeline architecture mapping (runs FIRST) |
| debugger | `dev-suite:debugger-pro` | Root cause synthesis (ANCHOR) |
| python-pro | `science-suite:python-pro` | Type/shape/dtype verification |
| jax-pro | `science-suite:jax-pro` | JIT, XLA, gradient flow, vmap/pmap |

### Prompt

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

---

## Team 21: Debug Schema

**When:** Schema/type drift — incompatible data classes across layers, field name mismatches, serialization errors.
**Suites:** dev-suite, feature-dev, science-suite, pr-review-toolkit
**Pattern:** Debugging Core Trio + Type Analyzer
**Note:** Do NOT run type-analyzer and quality-specialist simultaneously — they overlap on interface contract checking.

| Role | Agent Type | Focus |
|------|-----------|-------|
| explorer | `feature-dev:code-explorer` | Schema dependency graph (runs FIRST) |
| debugger | `dev-suite:debugger-pro` | Root cause synthesis (ANCHOR) |
| python-pro | `science-suite:python-pro` | Protocol/structural compatibility |
| type-analyzer | `pr-review-toolkit:type-design-analyzer` | Type design quality (read-only) |

### Prompt

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

---

## Quality Gate Enhancers

Any team can be enhanced by adding official plugin agents as quality gates. Add these agents to any team's prompt to get automated review after implementation.

### Recommended Enhancers

| Enhancer | Agent Type | Add To Teams | What It Catches |
|----------|-----------|--------------|-----------------|
| Code Review | `pr-review-toolkit:code-reviewer` | 1, 5, 7 | Style, bugs, guidelines |
| Silent Failures | `pr-review-toolkit:silent-failure-hunter` | 2, 6, 15 | Swallowed errors |
| Test Gaps | `pr-review-toolkit:pr-test-analyzer` | 4, 7, 5 | Missing test coverage |
| Type Quality | `pr-review-toolkit:type-design-analyzer` | 14, 15, 8 | Weak type invariants |
| AI Review | `coderabbit:code-reviewer` | 1, 5, 3 | Second-opinion analysis |
| Code Simplicity | `code-simplifier:code-simplifier` | 7, 14, 4 | Unnecessary complexity |
| Plan Adherence | `superpowers:code-reviewer` | 1, 7, 13 | Drift from plan |

### How to Add an Enhancer

Append this to any team prompt:

```
Additionally, spawn a "reviewer" teammate
(pr-review-toolkit:code-reviewer) that reviews all changes after the
implementation teammates finish their work. This reviewer is read-only
and reports issues sorted by severity. The team should not be considered
done until the reviewer's critical issues are addressed.
```

---

## Usage Tips

1. **Replace placeholders** (`[BRACKETS]`) with your project specifics before pasting
2. **Start with Team 3** (PR Review) if new to agent teams — read-only, low-risk
3. **Use delegate mode** (`Shift+Tab`) to prevent the lead from implementing tasks itself
4. **Monitor progress** with `Shift+Up/Down` (in-process) or click panes (tmux)
5. **`Ctrl+T`** toggles the shared task list view
6. **Prefer Sonnet** for most teammates (cost-effective); use Opus for architecture/design decisions
7. **Avoid file conflicts** — ensure each teammate owns distinct directories
8. **Read-only reviewers** (Teams 3, 17) have zero conflict risk — they never edit files
9. **Use aliases** when you remember the old name — `/team-assemble quality-audit` resolves to `quality-security` automatically

## References

- [Official Agent Teams Documentation](https://code.claude.com/docs/en/agent-teams)
- [Claude Code Subagents](https://code.claude.com/docs/en/sub-agents)
- [Claude Code Agent Teams Blog](https://addyosmani.com/blog/claude-code-agent-teams/)
- [Integration Map](integration-map.rst) — Suite dependencies, MCP server roles, and skill coverage
- [Glossary](glossary.rst) — Key terms including hub skills, sub-skills, and routing trees
