# Agent Teams Guide for MyClaude Plugin Suites

> 33 ready-to-use team configurations leveraging 22 MyClaude agents + 20 official plugin agents across 5 suites.

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
| **engineering** | software-architect | `engineering-suite:software-architect` | Backend, API, microservices |
| | app-developer | `engineering-suite:app-developer` | Frontend, mobile, React/Next.js |
| | systems-engineer | `engineering-suite:systems-engineer` | C/C++/Rust, CLI tools, low-level |
| **infra** | devops-architect | `infrastructure-suite:devops-architect` | Cloud, K8s, Terraform |
| | sre-expert | `infrastructure-suite:sre-expert` | Observability, SLOs, incidents |
| | automation-engineer | `infrastructure-suite:automation-engineer` | CI/CD, GitHub Actions, Git |
| **quality** | debugger-pro | `quality-suite:debugger-pro` | Root cause analysis, log correlation |
| | documentation-expert | `quality-suite:documentation-expert` | Tech writing, API docs |
| | quality-specialist | `quality-suite:quality-specialist` | Code review, security audit, testing |
| **science** | jax-pro | `science-suite:jax-pro` | JAX, NumPyro, Bayesian inference |
| | neural-network-master | `science-suite:neural-network-master` | Deep learning, Transformers, Flax |
| | ml-expert | `science-suite:ml-expert` | Scikit-learn, MLOps, XGBoost |
| | ai-engineer | `science-suite:ai-engineer` | LLM apps, RAG, agents |
| | python-pro | `science-suite:python-pro` | Python systems, packaging |
| | research-expert | `science-suite:research-expert` | Scientific methodology, papers |
| | simulation-expert | `science-suite:simulation-expert` | Physics simulation, MD |
| | statistical-physicist | `science-suite:statistical-physicist` | Stat mech, stochastic dynamics |
| | julia-pro | `science-suite:julia-pro` | Julia HPC, SciML |
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
| **huggingface-skills** | AGENTS | `huggingface-skills:AGENTS` | HuggingFace Hub operations |

---

## Quick Reference

### Development & Operations (Teams 1-10)

| # | Team | Best For | Agents Used | Teammates |
|---|------|----------|-------------|-----------|
| 1 | feature-dev | New features | architect + app-dev + quality | 3 |
| 2 | incident-response | Prod debugging | debugger + sre + devops | 3 |
| 3 | quality-audit | Pre-release review | quality + architect + debugger + docs | 4 |
| 4 | sci-pipeline | ML/JAX workflows | jax + ml + python + research | 4 |
| 5 | infra-setup | Cloud + CI/CD | devops + automation + sre | 3 |
| 6 | modernization | Legacy migration | architect + app-dev + quality + docs | 4 |
| 7 | dl-research | Neural networks | nn-master + jax + ml + research | 4 |
| 8 | api-design | API development | architect + app-dev + quality + docs | 4 |
| 9 | pr-review | Critical PR review | quality + systems + debugger | 3 |
| 10 | llm-app | AI applications | ai-eng + prompt-eng + architect | 3 |

### Scientific Computing (Teams 11-16)

| # | Team | Best For | Agents Used | Teammates |
|---|------|----------|-------------|-----------|
| 11 | julia-sciml | Julia SciML / DiffEq | julia + simulation + research + python | 4 |
| 12 | stat-phys | Phase transitions, correlations | stat-phys + jax + simulation + research | 4 |
| 13 | bayesian-pipeline | NumPyro / MCMC inference | jax + stat-phys + ml + research | 4 |
| 14 | md-campaign | Molecular dynamics | simulation + jax + stat-phys + python | 4 |
| 15 | ml-forcefield | ML potentials (NequIP/MACE) | nn-master + simulation + jax + research | 4 |
| 16 | paper-implement | Reproduce research papers | research + python + jax + nn-master | 4 |

### Cross-Suite Specialized (Teams 17-25)

| # | Team | Best For | Agents Used | Teammates |
|---|------|----------|-------------|-----------|
| 17 | perf-optimize | Performance profiling | systems + jax + debugger + python | 4 |
| 18 | hpc-interop | Cross-language HPC | julia + python + systems + jax | 4 |
| 19 | reproducible-research | Open science, CI/CD for research | research + python + automation + docs | 4 |
| 20 | prompt-lab | Prompt R&D, evaluation | prompt-eng + reasoning + ai-eng + research | 4 |
| 21 | ai-agent-dev | Agent systems, multi-agent | ai-eng + prompt-eng + architect + reasoning | 4 |
| 22 | data-pipeline | ETL, feature engineering, MLOps | python + ml + devops + quality | 4 |
| 23 | security-harden | Security hardening | quality + devops + systems + automation | 4 |
| 24 | docs-sprint | Documentation overhaul | docs-expert + research + app-dev + architect | 4 |
| 25 | monorepo-refactor | Monorepo restructuring | architect + automation + quality + systems | 4 |

### Official Plugin Integration (Teams 26-33)

| # | Team | Best For | Agents Used | Teammates |
|---|------|----------|-------------|-----------|
| 26 | full-pr-review | Maximum PR scrutiny | 4 pr-review-toolkit analyzers | 4 |
| 27 | feature-ship | Feature + review pipeline | code-architect + architect + app-dev + reviewer | 4 |
| 28 | agent-sdk-build | Agent SDK applications | sdk-verifiers + ai-eng + prompt-eng | 4 |
| 29 | plugin-forge | Claude Code extensions | plugin-dev + hookify + quality + validator | 4 |
| 30 | codebase-archaeology | Codebase understanding | code-explorer + docs-expert + researcher | 3 |
| 31 | code-health | Code quality + type safety | simplifier + type-eng + type-reviewer + quality | 4 |
| 32 | hf-ml-publish | HuggingFace model publish | hf-agents + ml-expert + python-pro | 4 |
| 33 | frontend-excellence | Frontend with review gates | app-dev + pr-reviewer + coderabbit | 3 |

---

## Team 1: Full-Stack Feature Development

**When:** Building a new feature spanning frontend, backend, and tests.
**Suites:** engineering-suite, quality-suite
**See also:** Team 27 adds a design-first pipeline with review gates using official plugins.

| Role | Agent Type | File Ownership |
|------|-----------|----------------|
| Lead | `agent-core:orchestrator` | Coordination only |
| backend | `engineering-suite:software-architect` | `src/api/`, `src/models/`, `src/services/` |
| frontend | `engineering-suite:app-developer` | `src/components/`, `src/pages/`, `src/hooks/` |
| qa | `quality-suite:quality-specialist` | `tests/`, `*.test.*` |

### Prompt

```
Create an agent team called "feature-dev" with 3 teammates to build [FEATURE_NAME].

Teammates:
1. "backend" - Backend engineer focused on API endpoints, data models, and
   business logic. Owns src/api/, src/models/, src/services/. Tech stack:
   [YOUR_BACKEND_STACK]. Implement the server-side logic first so frontend
   can integrate.

2. "frontend" - Frontend engineer building the UI components and pages.
   Owns src/components/, src/pages/, src/hooks/. Tech stack: [YOUR_FRONTEND_STACK].
   Wait for backend to define API contracts before integrating.

3. "qa" - Quality engineer writing tests and reviewing code. Owns tests/.
   Write unit tests for backend logic, integration tests for API endpoints,
   and component tests for frontend. Review all code for security and
   maintainability.

Task breakdown:
- Backend: Design data model -> Implement API endpoints -> Add validation
- Frontend: Create component skeleton -> Implement UI logic -> Connect to API
- QA: Write unit tests -> Write integration tests -> Security review

Quality gates: QA must review all code before marking tasks complete.
```

---

## Team 2: Production Incident Response

**When:** Debugging a production issue with unknown root cause.
**Suites:** quality-suite, infrastructure-suite

| Role | Agent Type | Focus Area |
|------|-----------|------------|
| Lead | `agent-core:orchestrator` | Triage, synthesis |
| debugger | `quality-suite:debugger-pro` | Code-level root cause analysis |
| sre | `infrastructure-suite:sre-expert` | Metrics, logs, traces |
| infra | `infrastructure-suite:devops-architect` | Infrastructure investigation |

### Prompt

```
Create an agent team called "incident-response" to investigate a production issue:
[DESCRIBE_SYMPTOMS].

Spawn 3 teammates to investigate different hypotheses in parallel:

1. "debugger" - Root cause analyst. Examine the application code for bugs,
   race conditions, or logic errors. Focus on [AFFECTED_MODULES]. Analyze
   stack traces, reproduce the issue locally, and form hypotheses. Challenge
   the other teammates' findings.

2. "sre" - Reliability investigator. Check observability data: metrics,
   logs, distributed traces. Look for patterns in error rates, latency
   spikes, resource exhaustion. Correlate timing with deployments or
   config changes.

3. "infra" - Infrastructure analyst. Investigate the deployment environment:
   container health, network connectivity, database performance,
   resource limits. Check if infrastructure changes correlate with the issue.

Have teammates share findings with each other and challenge each other's
theories. Synthesize into a root cause report with: confirmed root cause,
evidence, recommended fix, and prevention measures.
```

---

## Team 3: Code Quality & Security Audit

**When:** Comprehensive codebase review before a release or compliance audit.
**Suites:** quality-suite, engineering-suite

| Role | Agent Type | Review Lens |
|------|-----------|-------------|
| Lead | `agent-core:orchestrator` | Synthesis, prioritization |
| security | `quality-suite:quality-specialist` | OWASP, vulnerabilities, auth |
| architecture | `engineering-suite:software-architect` | Design patterns, SOLID, complexity |
| testing | `quality-suite:debugger-pro` | Test coverage gaps, edge cases |
| docs | `quality-suite:documentation-expert` | API docs, runbooks, README |

### Prompt

```
Create an agent team called "quality-audit" to perform a comprehensive
code quality and security audit of [PROJECT_PATH].

Spawn 4 reviewers, each with a distinct lens:

1. "security" - Security auditor. Scan for OWASP Top 10 vulnerabilities:
   injection, broken auth, data exposure, XSS, CSRF. Review authentication
   flows, input validation, secret handling, and dependency vulnerabilities.
   Rate each finding: Critical/High/Medium/Low.

2. "architecture" - Architecture reviewer. Assess design patterns, SOLID
   principles, coupling/cohesion, cyclomatic complexity, and code
   duplication. Identify architectural anti-patterns and tech debt.
   Produce an Architecture Decision Record for any major concerns.

3. "testing" - Test coverage analyst. Map untested code paths, identify
   missing edge cases, check for flaky tests, and assess the testing
   pyramid (unit/integration/e2e ratio). Recommend specific tests to add.

4. "docs" - Documentation reviewer. Audit API documentation completeness,
   README accuracy, inline comment quality, and operational runbooks.
   Identify undocumented public APIs and missing error documentation.

Each reviewer works independently, then shares findings. Synthesize into
a prioritized remediation plan with effort estimates.
```

---

## Team 4: Scientific Computing Pipeline

**When:** Building ML/Bayesian/JAX-based scientific computing workflows.
**Suites:** science-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Pipeline coordination |
| jax | `science-suite:jax-pro` | JAX core, GPU acceleration |
| ml | `science-suite:ml-expert` | Model development, MLOps |
| python | `science-suite:python-pro` | Systems integration, packaging |
| research | `science-suite:research-expert` | Methodology validation |

### Prompt

```
Create an agent team called "sci-pipeline" to build a scientific computing
pipeline for [RESEARCH_PROBLEM].

Spawn 4 specialist teammates:

1. "jax-engineer" - JAX implementation specialist. Implement the core
   computational kernels using JAX with JIT compilation, vmap for
   batching, and pmap for multi-device parallelism. Handle GPU memory
   management and custom VJPs. Owns src/core/, src/kernels/.

2. "ml-engineer" - ML pipeline architect. Set up experiment tracking
   (W&B/MLflow), hyperparameter optimization (Optuna), data loading
   pipelines, and model versioning. Owns src/training/, src/data/,
   configs/.

3. "python-architect" - Systems integration. Design the package structure
   with proper typing (Protocols, Generics), CLI interface, configuration
   management, and test infrastructure. Owns src/__init__.py, setup files,
   src/cli/, pyproject.toml.

4. "methodology" - Research methodology validator. Review the computational
   approach for scientific correctness, reproducibility (explicit seeds,
   deterministic ops), and statistical validity. Validate against
   [REFERENCE_PAPERS]. Owns docs/, notebooks/.

Ensure JAX-first architecture: minimize host-device transfers, use
interpax for interpolation, mandatory ArviZ diagnostics for Bayesian work.
```

---

## Team 5: Infrastructure & DevOps Setup

**When:** Setting up cloud infrastructure, CI/CD, and observability from scratch.
**Suites:** infrastructure-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Architecture decisions |
| cloud | `infrastructure-suite:devops-architect` | IaC, cloud resources |
| cicd | `infrastructure-suite:automation-engineer` | Pipelines, Git workflows |
| monitoring | `infrastructure-suite:sre-expert` | Observability stack |

### Prompt

```
Create an agent team called "infra-setup" to build the infrastructure for
[PROJECT_NAME] on [CLOUD_PROVIDER].

Spawn 3 infrastructure specialists:

1. "cloud-architect" - Platform engineer. Design and implement Infrastructure
   as Code using Terraform/Pulumi. Set up: VPC/networking, compute
   (EKS/ECS/Lambda), database (RDS/DynamoDB), storage (S3), and IAM
   policies. Follow zero-trust networking and least-privilege principles.
   Owns infra/, terraform/.

2. "cicd-engineer" - Pipeline architect. Build GitHub Actions workflows
   for: lint/test/build, container image builds, staged deployments
   (dev->staging->prod), security scanning (SAST/DAST), and release
   automation. Implement caching and artifact promotion. Owns
   .github/workflows/, scripts/ci/.

3. "sre-lead" - Observability architect. Set up Prometheus metrics
   collection, Grafana dashboards, distributed tracing (OpenTelemetry),
   structured logging, and alerting rules. Define SLIs/SLOs for key
   services. Implement health checks and readiness probes. Owns
   monitoring/, dashboards/.

Dependencies: cloud-architect defines infrastructure first, then
cicd-engineer configures deployment targets, then sre-lead instruments
the services.
```

---

## Team 6: Legacy Modernization

**When:** Migrating a legacy codebase to modern architecture.
**Suites:** engineering-suite, quality-suite

| Role | Agent Type | Focus |
|------|-----------|-------|
| Lead | `agent-core:orchestrator` | Migration strategy |
| architect | `engineering-suite:software-architect` | Target architecture |
| implementer | `engineering-suite:app-developer` | Migration code |
| qa | `quality-suite:quality-specialist` | Regression testing |
| docs | `quality-suite:documentation-expert` | Migration documentation |

### Prompt

```
Create an agent team called "modernization" to migrate [LEGACY_SYSTEM]
from [OLD_STACK] to [NEW_STACK].

Spawn 4 teammates:

1. "architect" - Target architecture designer. Analyze the existing
   codebase, identify migration boundaries (Strangler Fig pattern),
   design the target architecture with clean module boundaries. Create
   Architecture Decision Records for key choices. Require plan approval
   before implementation.

2. "implementer" - Migration developer. Execute the migration following
   the architect's plan. Implement adapter layers for backward
   compatibility during transition. Refactor module by module, ensuring
   each module works independently before moving to the next. Owns
   src/new/, src/adapters/.

3. "qa-lead" - Regression guardian. Write comprehensive tests for existing
   behavior BEFORE migration begins (characterization tests). Run tests
   continuously during migration to catch regressions. Owns tests/.

4. "docs-lead" - Migration documenter. Document the migration plan,
   track progress, write runbooks for rollback procedures, and update
   API documentation as interfaces change. Owns docs/migration/.

Critical rule: QA must have characterization tests passing before
implementer begins each module migration.
```

---

## Team 7: Deep Learning Research

**When:** Developing and training neural network architectures.
**Suites:** science-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Experiment coordination |
| nn-architect | `science-suite:neural-network-master` | Architecture design |
| jax-impl | `science-suite:jax-pro` | JAX/Flax implementation |
| mlops | `science-suite:ml-expert` | Training pipeline, deployment |
| researcher | `science-suite:research-expert` | Paper implementation, validation |

### Prompt

```
Create an agent team called "dl-research" to develop a neural network
for [RESEARCH_TASK].

Spawn 4 deep learning specialists:

1. "nn-architect" - Neural architecture designer. Design the model
   architecture considering: attention mechanisms, normalization strategies,
   activation functions, and parameter efficiency. Analyze gradient flow
   and provide theoretical justification. Reference [BASELINE_PAPER].
   Owns src/models/.

2. "jax-implementer" - JAX/Flax implementation engineer. Implement the
   architecture using Flax Linen (or Equinox if functional style preferred).
   Ensure JIT compatibility, efficient memory usage, and multi-device
   training with pmap/sharding. Implement custom training loops with
   gradient clipping and mixed precision. Owns src/training/, src/utils/.

3. "mlops-engineer" - Training infrastructure. Set up experiment tracking
   (W&B), hyperparameter sweeps, checkpoint management, and model
   evaluation pipelines. Configure data loading with proper prefetching
   and augmentation. Owns configs/, scripts/, src/data/.

4. "researcher" - Scientific validation. Implement evaluation metrics
   from the reference paper, set up ablation studies, analyze training
   diagnostics (loss landscapes, gradient norms, attention patterns).
   Ensure reproducibility with explicit seeds. Owns notebooks/,
   evaluation/.

Use mandatory ArviZ diagnostics for any Bayesian components.
```

---

## Team 8: API Design & Integration

**When:** Designing, building, and documenting a public or internal API.
**Suites:** engineering-suite, quality-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | API strategy |
| designer | `engineering-suite:software-architect` | API specification |
| implementer | `engineering-suite:app-developer` | Client SDKs |
| tester | `quality-suite:quality-specialist` | Contract testing |
| docs | `quality-suite:documentation-expert` | API documentation |

### Prompt

```
Create an agent team called "api-design" to design and implement a
[REST/GraphQL/gRPC] API for [SERVICE_NAME].

Spawn 4 teammates:

1. "api-designer" - API architect. Design the API specification following
   REST best practices: resource naming, HTTP methods, status codes,
   pagination, filtering, versioning strategy, and error response format.
   Create OpenAPI/Swagger spec. Require plan approval before
   implementation. Owns api/, specs/.

2. "implementer" - API developer. Implement the endpoints following the
   approved spec. Handle authentication (JWT/OAuth2), rate limiting,
   input validation, and error handling. Implement database queries with
   proper indexing. Owns src/routes/, src/middleware/, src/controllers/.

3. "tester" - API test engineer. Write contract tests (Pact), integration
   tests, load tests, and security tests (auth bypass, injection, rate
   limit circumvention). Validate all error paths. Owns tests/.

4. "docs-writer" - API documentation specialist. Generate comprehensive
   API docs with: endpoint reference, authentication guide, code
   examples in multiple languages, error handling guide, and migration
   guide from previous versions. Owns docs/api/.

Dependency: api-designer defines spec -> implementer + tester work in
parallel -> docs-writer documents the final API.
```

---

## Team 9: PR Review Swarm

**When:** Thorough multi-perspective review of a critical pull request.
**Suites:** quality-suite, engineering-suite
**See also:** Team 26 uses pr-review-toolkit specialized analyzers for toolkit-driven review.

| Role | Agent Type | Review Lens |
|------|-----------|-------------|
| Lead | `agent-core:orchestrator` | Review synthesis |
| security | `quality-suite:quality-specialist` | Vulnerabilities, auth |
| performance | `engineering-suite:systems-engineer` | Perf, memory, complexity |
| correctness | `quality-suite:debugger-pro` | Bugs, edge cases, logic |

### Prompt

```
Create an agent team called "pr-review" to review PR #[PR_NUMBER] from
multiple perspectives.

Spawn 3 specialized reviewers:

1. "security-reviewer" - Security-focused review. Check for: injection
   vulnerabilities, broken authentication, sensitive data exposure,
   missing input validation, insecure defaults, and dependency
   vulnerabilities. Rate each finding: Critical/High/Medium/Low.

2. "performance-reviewer" - Performance-focused review. Check for:
   O(n^2) algorithms, N+1 queries, unnecessary memory allocations,
   missing caching opportunities, blocking operations, and resource
   leaks. Profile hot paths and suggest optimizations.

3. "correctness-reviewer" - Logic and correctness review. Check for:
   off-by-one errors, race conditions, null/undefined handling, edge
   cases, error propagation, and breaking changes to public APIs.
   Verify test coverage for all new code paths.

Each reviewer must:
- Run `gh pr diff [PR_NUMBER]` to get the changes
- Focus ONLY on changed files and their immediate dependencies
- Provide specific line references for each finding
- Suggest concrete fixes, not just problem descriptions

Have reviewers share findings with each other and debate disagreements.
Synthesize into a unified review with prioritized action items.
```

---

## Team 10: LLM Application Development

**When:** Building production LLM-powered applications (RAG, agents, chatbots).
**Suites:** science-suite, engineering-suite
**See also:** Team 21 for multi-agent systems, Team 28 for Agent SDK apps.

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Architecture decisions |
| ai-eng | `science-suite:ai-engineer` | RAG, vector search, agents |
| prompt-eng | `science-suite:prompt-engineer` | Prompt design, evaluation |
| backend | `engineering-suite:software-architect` | API, infrastructure |

### Prompt

```
Create an agent team called "llm-app" to build a production LLM
application for [USE_CASE].

Spawn 3 specialists:

1. "ai-engineer" - LLM application architect. Design and implement the
   core AI pipeline: document ingestion, chunking strategy, embedding
   generation, vector store (Pinecone/Weaviate/pgvector), retrieval
   logic, and LLM orchestration. Implement guardrails, content filtering,
   and hallucination detection. Owns src/ai/, src/retrieval/, src/agents/.

2. "prompt-engineer" - Prompt and evaluation specialist. Design system
   prompts, few-shot examples, and chain-of-thought templates. Build
   evaluation framework: automated scoring, human evaluation interface,
   regression testing for prompt changes. Optimize for cost/latency/quality
   trade-offs. Owns prompts/, evaluation/, benchmarks/.

3. "backend-architect" - API and infrastructure. Build the serving layer:
   streaming API endpoints, authentication, rate limiting, caching
   (semantic cache for LLM responses), observability (token usage,
   latency, error rates). Design for horizontal scaling. Owns src/api/,
   src/middleware/, infra/.

Key requirements: implement structured output parsing, retry logic with
exponential backoff, and cost tracking per request.
```

---

## Team 11: Julia SciML Pipeline

**When:** Building scientific computing workflows with Julia's SciML ecosystem (DifferentialEquations.jl, ModelingToolkit.jl, Turing.jl).
**Suites:** science-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Pipeline coordination |
| julia | `science-suite:julia-pro` | Julia packages, DiffEq, SciML |
| simulation | `science-suite:simulation-expert` | Physics models, numerical methods |
| research | `science-suite:research-expert` | Methodology, validation |
| python | `science-suite:python-pro` | Interop, data pipeline, packaging |

### Prompt

```
Create an agent team called "julia-sciml" to build a Julia SciML pipeline
for [PROBLEM].

Spawn 4 specialist teammates:

1. "julia-engineer" - Julia SciML specialist. Implement the core solvers
   using DifferentialEquations.jl with appropriate algorithm selection
   (Tsit5, TRBDF2, SOSRI for SDEs). Use ModelingToolkit.jl for symbolic
   model definition and automatic Jacobian generation. Set up Turing.jl
   for Bayesian parameter estimation if needed. Owns src/, Project.toml.

2. "simulation-architect" - Physics model designer. Define the physical
   system, conservation laws, boundary conditions, and validation
   benchmarks. Ensure numerical stability (CFL conditions, adaptive
   stepping). Design parameter studies and sensitivity analyses.
   Owns models/, benchmarks/.

3. "methodology" - Research validator. Verify the mathematical formulation
   against [REFERENCE_PAPERS]. Set up convergence tests, error analysis,
   and comparison with analytical solutions where available. Ensure
   reproducibility with fixed seeds and version pinning. Owns docs/,
   notebooks/, test/.

4. "python-bridge" - Interoperability engineer. Build Python-Julia bridges
   using PythonCall.jl or PyJulia for data exchange. Set up data ingestion
   pipelines, results export (HDF5/Arrow), and visualization (Makie.jl
   for interactive, Plots.jl for publication). Owns scripts/, viz/.

Use Julia 1.10+ with strict type annotations at module boundaries.
```

---

## Team 12: Statistical Physics Research

**When:** Studying phase transitions, correlation functions, non-equilibrium dynamics, or soft matter systems.
**Suites:** science-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Research coordination |
| physicist | `science-suite:statistical-physicist` | Theory, ensemble methods |
| jax | `science-suite:jax-pro` | GPU-accelerated computation |
| simulation | `science-suite:simulation-expert` | MD/MC simulations |
| research | `science-suite:research-expert` | Literature, methodology |

### Prompt

```
Create an agent team called "stat-phys" to investigate [PHYSICAL_SYSTEM]
with focus on [PHENOMENON] (e.g., phase transitions, correlation functions,
non-equilibrium dynamics).

Spawn 4 specialist teammates:

1. "theorist" - Statistical physicist. Derive the theoretical framework:
   partition function, order parameters, critical exponents, scaling
   relations. Formulate the Langevin/Fokker-Planck equations if
   non-equilibrium. Identify universality class and relevant symmetries.
   Predict expected behavior to validate simulations. Owns theory/, docs/.

2. "gpu-compute" - JAX computation specialist. Implement GPU-accelerated
   Monte Carlo or molecular dynamics using JAX. Use vmap for ensemble
   averaging, pmap for multi-GPU scaling. Implement efficient correlation
   function computation (FFT-based), histogram reweighting, and
   finite-size scaling analysis. Owns src/compute/, src/analysis/.

3. "simulator" - Simulation architect. Design the simulation protocol:
   equilibration criteria, production run lengths, sampling strategies
   (replica exchange, Wang-Landau, umbrella sampling). Implement
   observables: structure factor S(q), radial distribution g(r),
   mean-square displacement, susceptibility. Owns src/simulation/.

4. "researcher" - Methodology and literature. Survey existing results
   for [PHYSICAL_SYSTEM], identify open questions, validate simulation
   results against known benchmarks. Prepare publication-quality figures
   using scientific visualization best practices. Owns papers/, figures/.

Ensure: explicit random seeds, ArviZ diagnostics for any Bayesian
components, proper error estimation (jackknife/bootstrap).
```

---

## Team 13: Bayesian Inference Pipeline

**When:** Building rigorous Bayesian analysis with NumPyro, MCMC diagnostics, and model comparison.
**Suites:** science-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Pipeline coordination |
| jax | `science-suite:jax-pro` | NumPyro, NUTS, GPU acceleration |
| physicist | `science-suite:statistical-physicist` | Prior selection, model structure |
| ml | `science-suite:ml-expert` | Model comparison, validation |
| research | `science-suite:research-expert` | Methodology, reproducibility |

### Prompt

```
Create an agent team called "bayesian-pipeline" to build a Bayesian
inference pipeline for [DATA_TYPE] using [MODEL_CLASS].

Spawn 4 specialist teammates:

1. "bayesian-engineer" - NumPyro/JAX specialist. Implement the
   probabilistic model in NumPyro. Set up NUTS sampler with
   appropriate warmup, target accept probability, and mass matrix
   adaptation. Implement warm-start from NLSQ point estimates.
   Handle GPU memory for large datasets. Owns src/models/, src/inference/.

2. "statistician" - Prior and model structure expert. Design informative
   vs weakly informative priors with physical justification. Implement
   hierarchical model structure if needed. Design posterior predictive
   checks and prior predictive simulations. Handle model reparametrization
   for sampling efficiency (non-centered parameterization). Owns
   src/priors/, src/diagnostics/.

3. "ml-validator" - Model comparison and validation. Implement model
   comparison metrics: WAIC, LOO-CV (using ArviZ), Bayes factors.
   Design cross-validation strategies. Build predictive performance
   benchmarks against frequentist baselines (MLE, MAP). Owns
   src/comparison/, src/validation/.

4. "methodology" - Research methodology. Ensure MCMC convergence
   diagnostics are comprehensive: R-hat (<1.01), ESS (>400/chain),
   BFMI (>0.3), divergence checks, trace plots. Document all modeling
   choices and sensitivity analyses. Owns docs/, notebooks/.

Mandatory: ArviZ for all diagnostics. NLSQ warm-start before NUTS.
Explicit seeds for reproducibility.
```

---

## Team 14: Molecular Dynamics Campaign

**When:** Running MD simulation campaigns for soft matter, biomolecular, or materials science research.
**Suites:** science-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Campaign coordination |
| simulation | `science-suite:simulation-expert` | MD setup, force fields |
| jax | `science-suite:jax-pro` | JAX-MD, GPU acceleration |
| physicist | `science-suite:statistical-physicist` | Thermodynamics, analysis |
| python | `science-suite:python-pro` | Workflow automation, packaging |

### Prompt

```
Create an agent team called "md-campaign" to run a molecular dynamics
campaign for [SYSTEM] studying [PROPERTY].

Spawn 4 specialist teammates:

1. "md-architect" - Simulation setup specialist. Design the simulation
   protocol: system construction (particle placement, box geometry),
   force field selection ([FORCE_FIELD]), ensemble (NVT/NPT/NVE),
   thermostat/barostat settings, integration timestep, and cutoff
   schemes. Handle equilibration protocol with staged heating/cooling
   if needed. Owns simulations/, configs/.

2. "gpu-engine" - JAX-MD implementation. Implement the simulation engine
   using JAX-MD or custom JAX kernels. Optimize neighbor list updates,
   force computation (JIT-compiled), and trajectory output. Handle
   multi-GPU scaling with pmap. Implement enhanced sampling methods
   (metadynamics, replica exchange) if needed. Owns src/engine/, src/sampling/.

3. "analyst" - Thermodynamic and structural analysis. Compute observables:
   radial distribution function g(r), structure factor S(q), mean-square
   displacement (diffusion), velocity autocorrelation, pressure tensor,
   free energy profiles. Implement block averaging for error estimation.
   Owns src/analysis/, results/.

4. "workflow-engineer" - Pipeline automation. Build the campaign workflow:
   parameter sweep management, job scheduling, trajectory storage
   (HDF5/MDAnalysis), checkpoint/restart logic, and automated convergence
   checking. Owns scripts/, workflows/, src/io/.

Ensure: proper equilibration verification, production run length
justified by autocorrelation analysis, explicit seeds.
```

---

## Team 15: ML Force Field Development

**When:** Developing machine learning interatomic potentials (NequIP, MACE, SchNet).
**Suites:** science-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Development coordination |
| nn | `science-suite:neural-network-master` | Architecture design |
| simulation | `science-suite:simulation-expert` | Training data, validation |
| jax | `science-suite:jax-pro` | JAX/Flax implementation |
| research | `science-suite:research-expert` | Benchmarks, paper comparison |

### Prompt

```
Create an agent team called "ml-forcefield" to develop a machine learning
force field for [CHEMICAL_SYSTEM].

Spawn 4 specialist teammates:

1. "nn-designer" - Neural network architect. Design the ML potential
   architecture: equivariant message passing (E(3)-equivariant),
   interaction layers, radial basis functions, and output heads
   (energy, forces, stress). Choose between NequIP-style vs MACE-style
   architectures based on accuracy/speed trade-offs. Owns src/models/.

2. "data-engineer" - Training data specialist. Curate DFT training data:
   structure selection (active learning, FPS), energy/force/stress labels,
   train/val/test splits ensuring chemical diversity. Implement data
   augmentation (rotation, translation). Design the training distribution
   to cover relevant PES regions. Owns data/, src/datasets/.

3. "jax-trainer" - Training engineer. Implement the training loop in
   JAX/Flax with: per-atom energy loss + force matching loss (weighted),
   learning rate scheduling (cosine with warmup), gradient clipping,
   EMA weights. Multi-GPU training with sharding. Checkpoint management
   and early stopping. Owns src/training/, configs/.

4. "validator" - Scientific validation. Benchmark against DFT reference:
   energy MAE, force MAE/RMSE, equation of state, phonon dispersion,
   elastic constants. Run stability tests: NVE energy drift, radial
   distribution function comparison, melting point prediction.
   Owns evaluation/, benchmarks/, notebooks/.

Ensure physical symmetries are built into architecture, not learned.
```

---

## Team 16: Scientific Paper Implementation

**When:** Reproducing results from a research paper or implementing a published algorithm.
**Suites:** science-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Implementation coordination |
| research | `science-suite:research-expert` | Paper analysis, methodology |
| python | `science-suite:python-pro` | Clean implementation |
| jax | `science-suite:jax-pro` | Numerical implementation |
| nn | `science-suite:neural-network-master` | Neural architecture (if applicable) |

### Prompt

```
Create an agent team called "paper-implement" to reproduce results from
[PAPER_TITLE] ([PAPER_URL]).

Spawn 4 specialist teammates:

1. "paper-analyst" - Research methodology expert. Read and decompose the
   paper: extract the core algorithm, mathematical formulation, key
   equations, hyperparameters, dataset descriptions, and evaluation
   metrics. Identify ambiguities or missing details that need resolution.
   Create a structured implementation specification. Owns docs/spec/.

2. "python-engineer" - Clean implementation. Build the codebase with
   proper structure: typed interfaces, configuration management (hydra
   or dataclasses), CLI entry points, and comprehensive logging. Handle
   data loading, preprocessing, and results serialization. Owns src/,
   pyproject.toml.

3. "numerical-engineer" - Core algorithm implementation. Implement the
   mathematical core in JAX: numerical kernels, optimization routines,
   custom gradients if needed. Ensure numerical stability (log-space
   computation, gradient clipping). Match the paper's convergence
   criteria exactly. Owns src/core/, src/optim/.

4. "reproducer" - Results reproduction. Run the experiments from the
   paper with identical hyperparameters. Compare outputs: tables,
   figures, metrics. Document any discrepancies and their likely causes.
   Prepare reproduction report with side-by-side comparison.
   Owns experiments/, results/, notebooks/.

Goal: exact reproduction within reported error bars. Document ALL
deviations from the paper.
```

---

## Team 17: Performance Optimization Swarm

**When:** Profiling and optimizing computational code (Python, JAX, or systems-level).
**Suites:** engineering-suite, science-suite, quality-suite

| Role | Agent Type | Focus |
|------|-----------|-------|
| Lead | `agent-core:orchestrator` | Optimization strategy |
| systems | `engineering-suite:systems-engineer` | Systems profiling, memory |
| jax | `science-suite:jax-pro` | JIT, vectorization, GPU |
| debugger | `quality-suite:debugger-pro` | Bottleneck identification |
| python | `science-suite:python-pro` | Python optimization |

### Prompt

```
Create an agent team called "perf-optimize" to profile and optimize
[TARGET_CODE].

Spawn 4 specialist teammates:

1. "systems-profiler" - Low-level performance analyst. Profile CPU usage
   (perf, py-spy), memory allocation (tracemalloc, memray), I/O patterns,
   and cache behavior. Identify hot functions, memory leaks, and
   unnecessary copies. Generate flamegraphs. Owns profiling/, reports/.

2. "jax-optimizer" - GPU/vectorization specialist. Convert sequential
   loops to vmap, identify JIT compilation opportunities, optimize
   XLA compilation (avoid recompilation), minimize host-device transfers,
   and implement efficient batching strategies. Profile with JAX's
   built-in profiler. Owns src/optimized/.

3. "bottleneck-hunter" - Root cause analyst. Investigate why specific
   operations are slow: algorithmic complexity (O(n^2) → O(n log n)),
   unnecessary recomputation, inefficient data structures, GIL
   contention, or I/O bottlenecks. Propose and validate fixes with
   micro-benchmarks. Owns benchmarks/.

4. "python-optimizer" - Python-level optimization. Apply: Cython/mypyc
   compilation for hot paths, asyncio for I/O-bound code, multiprocessing
   for CPU-bound parallelism, efficient data structures (numpy structured
   arrays, pandas optimizations), and Rust extensions via PyO3 if needed.
   Owns src/extensions/.

Protocol: Profile first (measure) → Identify top 3 bottlenecks →
Optimize one at a time → Re-profile → Repeat.
Target: [SPEEDUP_TARGET] (e.g., 10x throughput improvement).
```

---

## Team 18: HPC & Cross-Language Interop

**When:** Building high-performance pipelines spanning Julia, Python, and compiled languages.
**Suites:** science-suite, engineering-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Architecture coordination |
| julia | `science-suite:julia-pro` | Julia performance, packages |
| python | `science-suite:python-pro` | Python integration |
| systems | `engineering-suite:systems-engineer` | C/Rust FFI, memory |
| jax | `science-suite:jax-pro` | GPU compute, JAX kernels |

### Prompt

```
Create an agent team called "hpc-interop" to build a cross-language HPC
pipeline for [COMPUTATIONAL_TASK].

Spawn 4 specialist teammates:

1. "julia-lead" - Julia performance engineer. Implement core numerical
   algorithms in Julia for maximum performance: type-stable code,
   @inbounds/@simd annotations, pre-allocation, and LoopVectorization.jl.
   Build Julia package with proper Project.toml and test suite.
   Owns julia/, Project.toml.

2. "python-integrator" - Python ecosystem bridge. Build Python wrappers
   using PythonCall.jl (Julia→Python) or juliacall (Python→Julia).
   Handle data serialization (Arrow/HDF5 for zero-copy), memory
   management across language boundaries, and error propagation.
   Owns python/, src/bridge/.

3. "systems-engineer" - FFI and compiled extensions. Build C/Rust
   extensions for performance-critical inner loops using PyO3 (Rust)
   or cffi (C). Handle memory layout compatibility (row vs column major),
   thread safety, and SIMD intrinsics. Owns extensions/, src/native/.

4. "gpu-architect" - GPU compute layer. Implement GPU kernels in JAX
   for embarrassingly parallel operations. Handle CPU↔GPU data
   transfer optimization, kernel fusion, and multi-GPU distribution.
   Ensure numerical equivalence with CPU reference implementation.
   Owns src/gpu/, benchmarks/.

Key constraint: zero-copy data transfer between languages where possible.
Benchmark each language boundary to quantify overhead.
```

---

## Team 19: Reproducible Research

**When:** Setting up fully reproducible computational research with CI/CD, packaging, and documentation.
**Suites:** science-suite, infrastructure-suite, quality-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Workflow design |
| research | `science-suite:research-expert` | Methodology, documentation |
| python | `science-suite:python-pro` | Package structure, environments |
| cicd | `infrastructure-suite:automation-engineer` | CI/CD, automation |
| docs | `quality-suite:documentation-expert` | Documentation, notebooks |

### Prompt

```
Create an agent team called "reproducible-research" to make [PROJECT_NAME]
fully reproducible and publishable.

Spawn 4 specialist teammates:

1. "methodology" - Research reproducibility expert. Audit the current
   codebase for reproducibility gaps: hardcoded paths, missing seeds,
   undocumented parameters, version-dependent behavior. Create a
   reproducibility checklist and ensure all experiments can be re-run
   from a single command. Owns docs/methodology/.

2. "packager" - Python packaging specialist. Structure the project as
   an installable package with pyproject.toml, proper dependency
   pinning (uv.lock), entry points for all scripts, and typed
   configuration objects. Set up dev/test/docs dependency groups.
   Owns pyproject.toml, src/__init__.py, configs/.

3. "ci-engineer" - Automation specialist. Build GitHub Actions workflows:
   automated testing on every push, notebook execution verification,
   figure regeneration, dependency security scanning, and automated
   release with changelog generation. Set up pre-commit hooks. Owns
   .github/workflows/, .pre-commit-config.yaml.

4. "doc-writer" - Research documentation. Create comprehensive docs:
   installation guide, quickstart tutorial, API reference (autodoc),
   mathematical derivation appendix, and a "reproducing our results"
   guide. Convert key notebooks to Sphinx gallery examples. Owns
   docs/, notebooks/.

Goal: anyone should be able to clone, install, and reproduce all results
with `uv sync && uv run reproduce-all`.
```

---

## Team 20: Prompt Engineering Lab

**When:** Systematic prompt R&D: designing, testing, and evaluating LLM prompts.
**Suites:** science-suite, agent-core

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Experiment coordination |
| prompt | `science-suite:prompt-engineer` | Prompt design, patterns |
| reasoning | `agent-core:reasoning-engine` | CoT, constitutional AI |
| ai | `science-suite:ai-engineer` | Integration, tooling |
| research | `science-suite:research-expert` | Evaluation methodology |

### Prompt

```
Create an agent team called "prompt-lab" to design and evaluate prompts
for [LLM_TASK].

Spawn 4 specialist teammates:

1. "prompt-designer" - Prompt engineering specialist. Design prompt
   variants using: zero-shot, few-shot, chain-of-thought, tree-of-thought,
   and constitutional AI patterns. Create a prompt library with versioning.
   Implement prompt templates with variable substitution. Owns
   prompts/, src/templates/.

2. "reasoning-architect" - Cognitive architecture designer. Design
   reasoning scaffolds: structured output formats, self-consistency
   checks, meta-cognitive reflection steps, and error correction loops.
   Implement the evaluation criteria for each reasoning pattern.
   Owns src/reasoning/, src/scaffolds/.

3. "eval-engineer" - Evaluation infrastructure. Build automated evaluation
   pipelines: LLM-as-judge scoring, reference-based metrics (BLEU/ROUGE
   for text, exact match for structured), human evaluation interfaces,
   and A/B testing frameworks. Track prompt performance over time.
   Owns evaluation/, benchmarks/, src/eval/.

4. "researcher" - Experimental methodology. Design statistically rigorous
   experiments: sample sizes, confidence intervals, ablation studies.
   Control for prompt ordering effects and model temperature sensitivity.
   Produce publication-quality analysis of prompt performance.
   Owns experiments/, analysis/, notebooks/.

Track all prompt versions and evaluation results. Never deploy without
regression testing against the existing prompt suite.
```

---

## Team 21: AI Agent Development

**When:** Building multi-agent AI systems, tool-using agents, or autonomous workflows.
**Suites:** science-suite, engineering-suite, agent-core

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Architecture coordination |
| ai | `science-suite:ai-engineer` | Agent framework, tools |
| prompt | `science-suite:prompt-engineer` | System prompts, behavior |
| architect | `engineering-suite:software-architect` | Backend, API design |
| reasoning | `agent-core:reasoning-engine` | Cognitive architecture |

### Prompt

```
Create an agent team called "ai-agent-dev" to build an AI agent system
for [AGENT_PURPOSE].

Spawn 4 specialist teammates:

1. "agent-architect" - AI agent framework engineer. Design the agent
   architecture: tool definitions, state management, memory systems
   (short-term/long-term), planning strategies (ReAct, Plan-and-Execute),
   and error recovery. Implement using Claude Agent SDK or LangGraph.
   Owns src/agents/, src/tools/.

2. "prompt-architect" - System prompt engineer. Design the agent's
   personality, capabilities description, tool usage instructions,
   safety guardrails, and output formatting. Implement constitutional
   AI principles for self-correction. Create few-shot examples for
   complex tool chains. Owns prompts/, src/guardrails/.

3. "backend-engineer" - Serving infrastructure. Build the API layer:
   streaming responses, session management, tool execution sandbox,
   rate limiting, cost tracking, and observability (latency, token
   usage, tool call frequency). Owns src/api/, src/middleware/.

4. "cognitive-designer" - Reasoning architecture. Design the agent's
   reasoning loop: when to use tools vs direct response, multi-step
   planning, self-reflection checkpoints, and confidence calibration.
   Implement evaluation harness for agent behavior. Owns src/reasoning/,
   evaluation/.

Key requirements: implement retry logic, graceful degradation when
tools fail, and comprehensive logging of agent decision traces.
```

---

## Team 22: Data Pipeline & Feature Engineering

**When:** Building ETL pipelines, feature stores, or MLOps data infrastructure.
**Suites:** science-suite, infrastructure-suite, quality-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Pipeline architecture |
| python | `science-suite:python-pro` | Data engineering |
| ml | `science-suite:ml-expert` | Feature engineering, MLOps |
| devops | `infrastructure-suite:devops-architect` | Infrastructure, storage |
| quality | `quality-suite:quality-specialist` | Data quality, testing |

### Prompt

```
Create an agent team called "data-pipeline" to build a data pipeline
for [DATA_SOURCE] feeding [ML_MODEL_OR_ANALYTICS].

Spawn 4 specialist teammates:

1. "data-engineer" - Pipeline architect. Build the ETL/ELT pipeline:
   data ingestion (batch/streaming), transformation logic (pandas/polars/
   dask), schema validation (pandera), and output sinks (parquet/Delta
   Lake). Handle incremental processing and idempotency. Owns
   src/pipeline/, src/transforms/.

2. "feature-engineer" - ML feature specialist. Design and implement
   the feature store: feature definitions, computation logic, online/
   offline serving, feature versioning, and point-in-time correctness.
   Implement feature monitoring for drift detection. Owns src/features/,
   feature_store/.

3. "infra-engineer" - Data infrastructure. Set up storage (S3/GCS),
   orchestration (Airflow/Dagster), compute (Spark/Dask cluster),
   and metadata management (data catalog). Configure data lineage
   tracking and access controls. Owns infra/, dags/.

4. "quality-engineer" - Data quality guardian. Implement data quality
   checks: schema validation, statistical tests (Great Expectations),
   freshness monitoring, completeness checks, and anomaly detection.
   Build data quality dashboards. Owns tests/, quality_checks/.

Key constraint: all transformations must be idempotent and testable
with synthetic data.
```

---

## Team 23: Security Hardening

**When:** Comprehensive security review and hardening of infrastructure and application code.
**Suites:** quality-suite, infrastructure-suite, engineering-suite

| Role | Agent Type | Focus |
|------|-----------|-------|
| Lead | `agent-core:orchestrator` | Security strategy |
| security | `quality-suite:quality-specialist` | AppSec, OWASP |
| infra | `infrastructure-suite:devops-architect` | Infrastructure security |
| systems | `engineering-suite:systems-engineer` | Binary security, memory |
| cicd | `infrastructure-suite:automation-engineer` | Security automation |

### Prompt

```
Create an agent team called "security-harden" to perform security
hardening of [PROJECT_OR_INFRASTRUCTURE].

Spawn 4 security specialists:

1. "appsec" - Application security expert. Audit for OWASP Top 10:
   injection, broken auth, sensitive data exposure, XXE, broken access
   control, security misconfiguration, XSS, insecure deserialization,
   vulnerable components, insufficient logging. Implement fixes with
   defense-in-depth. Owns src/security/, src/middleware/.

2. "infra-sec" - Infrastructure security. Harden: network segmentation,
   firewall rules, TLS configuration, secrets management (Vault/SOPS),
   IAM policies (least privilege), container security (distroless images,
   read-only filesystem), and Kubernetes security policies (PSP/OPA).
   Owns infra/security/, k8s/policies/.

3. "systems-sec" - Systems-level security. Review: memory safety (buffer
   overflows, use-after-free), privilege escalation vectors, file
   permission hardening, syscall filtering (seccomp), and binary
   protections (ASLR, stack canaries, PIE). Owns security/system/.

4. "secops" - Security automation. Build security CI/CD: SAST (Semgrep/
   CodeQL), DAST (OWASP ZAP), dependency scanning (Dependabot/Snyk),
   container scanning (Trivy), secret detection (TruffleHog), and
   compliance checks. Set up security alerts and incident runbooks.
   Owns .github/workflows/security.yml, scripts/security/.

Output: prioritized finding report with CVSS scores, remediation
steps, and verification procedures.
```

---

## Team 24: Documentation Sprint

**When:** Comprehensive documentation overhaul for a project or API.
**Suites:** quality-suite, science-suite, engineering-suite

| Role | Agent Type | Domain |
|------|-----------|--------|
| Lead | `agent-core:orchestrator` | Documentation strategy |
| docs | `quality-suite:documentation-expert` | Structure, standards |
| research | `science-suite:research-expert` | Technical accuracy |
| frontend | `engineering-suite:app-developer` | Interactive examples |
| architect | `engineering-suite:software-architect` | Architecture docs |

### Prompt

```
Create an agent team called "docs-sprint" to create comprehensive
documentation for [PROJECT_NAME].

Spawn 4 specialist teammates:

1. "docs-architect" - Documentation structure designer. Design the
   information architecture: getting started guide, tutorials (learning-
   oriented), how-to guides (task-oriented), reference (API docs), and
   explanation (understanding-oriented) following Diataxis framework.
   Set up Sphinx/MkDocs with proper theme and navigation. Owns docs/.

2. "technical-writer" - Content accuracy validator. Review all
   documentation for technical accuracy by cross-referencing with source
   code. Ensure code examples compile and run, CLI flags match actual
   behavior, and configuration options are complete. Fix any stale or
   misleading content. Owns docs/reference/.

3. "tutorial-builder" - Interactive examples. Create tutorials with
   runnable code examples, Jupyter notebooks for interactive exploration,
   and a cookbook of common patterns. Build a docs testing harness that
   validates all code snippets. Owns docs/tutorials/, notebooks/.

4. "architecture-writer" - System documentation. Create architecture
   decision records (ADRs), system diagrams (Mermaid), component
   interaction documentation, deployment guides, and operational
   runbooks. Document all configuration options with defaults and
   examples. Owns docs/architecture/, docs/operations/.

Standard: every public API must have docstring + reference page +
at least one usage example.
```

---

## Team 25: Monorepo Refactoring

**When:** Restructuring a monorepo, improving build systems, or splitting/merging repositories.
**Suites:** engineering-suite, infrastructure-suite, quality-suite

| Role | Agent Type | Focus |
|------|-----------|-------|
| Lead | `agent-core:orchestrator` | Refactoring strategy |
| architect | `engineering-suite:software-architect` | Module boundaries |
| cicd | `infrastructure-suite:automation-engineer` | Build system, CI |
| quality | `quality-suite:quality-specialist` | Testing, code quality |
| systems | `engineering-suite:systems-engineer` | Build performance |

### Prompt

```
Create an agent team called "monorepo-refactor" to restructure
[REPOSITORY] into a well-organized monorepo (or split it into separate
packages).

Spawn 4 specialist teammates:

1. "architect" - Module boundary designer. Analyze dependency graphs,
   identify circular dependencies, define clean module boundaries with
   explicit public APIs, and design the target directory structure.
   Create a migration plan with atomic, reversible steps. Require plan
   approval before implementation. Owns docs/architecture/.

2. "build-engineer" - CI/CD and build system. Configure the build tool
   (Turborepo/Nx/Pants/Bazel): task pipelines, caching, affected
   detection (only build/test changed packages), and dependency
   graph-based ordering. Optimize CI for monorepo: path filtering,
   incremental builds, shared caches. Owns build configs, .github/workflows/.

3. "quality-guard" - Testing and standards. Ensure test isolation (each
   package testable independently), shared lint/format configurations,
   consistent versioning strategy (independent or lockstep), and
   cross-package integration tests. Set up git hooks for pre-commit
   validation. Owns tests/, .eslintrc, .prettierrc.

4. "systems-optimizer" - Build performance. Profile build times, optimize
   dependency resolution, implement workspace caching strategies, and
   minimize cold build times. Handle shared dependencies (hoisting vs
   isolation trade-off). Owns performance benchmarks.

Key constraint: migration must be atomic - the repo must build and pass
tests at every intermediate step.
```

---

## Team 26: Full PR Review

**When:** Critical PRs needing maximum scrutiny — security-sensitive changes, major refactors, or pre-release reviews.
**Plugins:** pr-review-toolkit (4 specialized analyzers)
**See also:** Team 9 uses MyClaude domain experts (security, performance, correctness) for perspective-based review.

| Role | Agent Type | Focus Area |
|------|-----------|------------|
| Lead | `agent-core:orchestrator` | Coordination only |
| code-reviewer | `pr-review-toolkit:code-reviewer` | Style, guidelines, bugs, comments |
| failure-hunter | `pr-review-toolkit:silent-failure-hunter` | Error handling, fallbacks |
| test-analyzer | `pr-review-toolkit:pr-test-analyzer` | Test coverage gaps |
| type-analyzer | `pr-review-toolkit:type-design-analyzer` | Type invariants |

### Prompt

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

---

## Team 27: Feature Ship Pipeline

**When:** End-to-end feature development with architecture design, implementation, and automated review gates.
**Plugins:** feature-dev, pr-review-toolkit, engineering-suite

| Role | Agent Type | File Ownership |
|------|-----------|----------------|
| Lead | `agent-core:orchestrator` | Coordination only |
| architect | `feature-dev:code-architect` | docs/design/, architecture decisions |
| builder | `engineering-suite:app-developer` | src/, components, pages |
| backend | `engineering-suite:software-architect` | api/, services/, models/ |
| reviewer | `pr-review-toolkit:code-reviewer` | Read-only review |

### Prompt

```
Create an agent team called "feature-ship" to design, build, and review
[FEATURE_NAME] for [PROJECT].

Spawn 4 specialist teammates:

1. "architect" (feature-dev:code-architect) - Analyze the existing
   codebase patterns and conventions, then produce a comprehensive
   implementation blueprint: files to create/modify, component designs,
   data flows, and build sequence. Present the blueprint for approval
   before any code is written. Owns docs/design/.

2. "builder" (engineering-suite:app-developer) - Implement the frontend
   components following the architect's blueprint. Build with
   [FRONTEND_STACK]. Focus on performance, accessibility, and
   offline-first patterns. Owns src/components/, src/pages/, src/hooks/.

3. "backend" (engineering-suite:software-architect) - Implement the
   backend services following the architect's blueprint. Build with
   [BACKEND_STACK]. Design scalable APIs with proper error handling
   and validation. Owns src/api/, src/services/, src/models/.

4. "reviewer" (pr-review-toolkit:code-reviewer) - After builder and
   backend complete their work, review all changes for adherence to
   the architect's blueprint, project guidelines, and best practices.
   Report only high-priority issues. Read-only.

Workflow: architect → (builder + backend in parallel) → reviewer.
```

---

## Team 28: Agent SDK Build

**When:** Building AI agent applications using the Claude Agent SDK (TypeScript or Python).
**Plugins:** agent-sdk-dev, science-suite, engineering-suite

| Role | Agent Type | File Ownership |
|------|-----------|----------------|
| Lead | `agent-core:orchestrator` | Coordination only |
| ai-engineer | `science-suite:ai-engineer` | src/agents/, src/tools/ |
| prompt-designer | `science-suite:prompt-engineer` | prompts/, system instructions |
| api-architect | `engineering-suite:software-architect` | src/api/, infrastructure |
| verifier | `agent-sdk-dev:agent-sdk-verifier-[LANG]` | Read-only validation |

### Prompt

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

3. "api-architect" (engineering-suite:software-architect) - Build the
   API layer and infrastructure: REST/WebSocket endpoints, streaming
   support, authentication, rate limiting, and deployment config.
   Owns src/api/, infrastructure/.

4. "verifier" (agent-sdk-dev:agent-sdk-verifier-[LANGUAGE]) - After
   implementation is complete, verify the application follows SDK best
   practices, proper tool use patterns, and documentation
   recommendations. Flag any misconfigurations. Read-only.

Workflow: ai-engineer + prompt-designer + api-architect (parallel) → verifier.
```

---

## Team 29: Plugin Forge

**When:** Building Claude Code extensions — plugins, hooks, agents, commands, skills, and SDK integrations.
**Plugins:** plugin-dev, hookify, quality-suite, superpowers
**See also:** For Agent SDK apps (not plugins), use Team 28 (agent-sdk-build).

| Role | Agent Type | File Ownership |
|------|-----------|----------------|
| Lead | `agent-core:orchestrator` | Coordination only |
| creator | `plugin-dev:agent-creator` | agents/, commands/, skills/, plugin.json |
| hook-designer | `hookify:conversation-analyzer` | hooks/ |
| quality | `quality-suite:quality-specialist` | tests/, .github/workflows/ |
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

3. "quality" (quality-suite:quality-specialist) - Write comprehensive
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

---

## Team 30: Codebase Archaeology

**When:** Understanding unfamiliar codebases — onboarding, pre-refactor analysis, or architecture documentation.
**Plugins:** feature-dev, quality-suite, science-suite

| Role | Agent Type | Focus Area |
|------|-----------|------------|
| Lead | `agent-core:orchestrator` | Coordination only |
| explorer | `feature-dev:code-explorer` | Execution paths, patterns |
| documenter | `quality-suite:documentation-expert` | Architecture docs |
| researcher | `science-suite:research-expert` | Methodology, synthesis |

### Prompt

```
Create an agent team called "codebase-archaeology" to deeply analyze
and document [REPOSITORY].

Focus areas: [FOCUS_AREAS]

Spawn 3 specialist teammates:

1. "explorer" (feature-dev:code-explorer) - Trace execution paths
   through the codebase starting from entry points. Map the architecture
   layers, identify patterns and abstractions, document dependencies
   between modules. Produce a component map with data flow diagrams.

2. "documenter" (quality-suite:documentation-expert) - Transform the
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

---

## Team 31: Code Health Sprint

**When:** Systematic code quality improvement — reducing complexity, improving types, cleaning up dead code, or strengthening type safety.
**Plugins:** pr-review-toolkit, code-simplifier, quality-suite, science-suite
**See also:** For type-safety-only focus, omit the simplifier and use the type-engineer + type-reviewer + tester trio.

| Role | Agent Type | Focus Area |
|------|-----------|------------|
| Lead | `agent-core:orchestrator` | Coordination only |
| simplifier | `code-simplifier:code-simplifier` | Clarity, maintainability |
| type-engineer | `science-suite:python-pro` | Type annotations, strict typing |
| type-reviewer | `pr-review-toolkit:type-design-analyzer` | Type quality analysis |
| enforcer | `quality-suite:quality-specialist` | Standards, testing |

### Prompt

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

4. "enforcer" (quality-suite:quality-specialist) - After simplifier and
   type-engineer complete, run the full test suite and verify no
   regressions. Review all changes for security implications. Add tests
   for any uncovered edge cases discovered during the sprint.

Workflow: (simplifier + type-engineer) in parallel → type-reviewer → enforcer.
Constraint: all existing tests must pass after every change.
Success criteria: zero type errors in strict mode, all tests pass.
```

---

## Team 32: HuggingFace ML Publish

**When:** Training, evaluating, and publishing ML models to HuggingFace Hub.
**Plugins:** huggingface-skills, science-suite

| Role | Agent Type | File Ownership |
|------|-----------|----------------|
| Lead | `agent-core:orchestrator` | Coordination only |
| ml-engineer | `science-suite:ml-expert` | src/training/, configs/ |
| coder | `science-suite:python-pro` | src/data/, src/utils/ |
| publisher | `huggingface-skills:AGENTS` | model cards, Hub uploads |
| evaluator | `science-suite:research-expert` | evaluation/, benchmarks/ |

### Prompt

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

---

## Team 33: Frontend Excellence

**When:** Frontend development with multiple quality gates — code review, AI review, and standards enforcement.
**Plugins:** engineering-suite, pr-review-toolkit, coderabbit

| Role | Agent Type | File Ownership |
|------|-----------|----------------|
| Lead | `agent-core:orchestrator` | Coordination only |
| builder | `engineering-suite:app-developer` | src/components/, src/pages/ |
| pr-reviewer | `pr-review-toolkit:code-reviewer` | Read-only review |
| ai-reviewer | `coderabbit:code-reviewer` | Read-only review |

### Prompt

```
Create an agent team called "frontend-excellence" to build and
review [FEATURE_NAME] using [FRONTEND_STACK].

Spawn 3 specialist teammates:

1. "builder" (engineering-suite:app-developer) - Implement the frontend
   feature with focus on performance, accessibility (WCAG 2.1 AA),
   responsive design, and offline-first patterns. Write component
   tests and integration tests. Use [FRONTEND_STACK] patterns and
   conventions. Owns src/components/, src/pages/, src/hooks/, tests/.

2. "pr-reviewer" (pr-review-toolkit:code-reviewer) - After builder
   completes, review all changes for adherence to project guidelines,
   style violations, potential bugs, and accessibility issues. Use
   confidence-based filtering for high-priority issues only. Read-only.

3. "ai-reviewer" (coderabbit:code-reviewer) - Independently review the
   same changes using CodeRabbit AI analysis. Provide a second opinion
   on code quality, potential issues, and improvement suggestions.
   Read-only.

Workflow: builder → (pr-reviewer + ai-reviewer in parallel).
Output: implemented feature + two independent review reports.
```

---

## Quality Gate Enhancers

Any team can be enhanced by adding official plugin agents as quality gates. Add these agents to any team's prompt to get automated review after implementation.

### Recommended Enhancers

| Enhancer | Agent Type | Add To Teams | What It Catches |
|----------|-----------|--------------|-----------------|
| Code Review | `pr-review-toolkit:code-reviewer` | 1, 6, 8, 10 | Style, bugs, guidelines |
| Silent Failures | `pr-review-toolkit:silent-failure-hunter` | 2, 5, 22 | Swallowed errors |
| Test Gaps | `pr-review-toolkit:pr-test-analyzer` | 1, 3, 6, 8 | Missing test coverage |
| Type Quality | `pr-review-toolkit:type-design-analyzer` | 17, 22, 25 | Weak type invariants |
| AI Review | `coderabbit:code-reviewer` | 1, 6, 8, 9 | Second-opinion analysis |
| Code Simplicity | `code-simplifier:code-simplifier` | 6, 17, 25 | Unnecessary complexity |
| Plan Adherence | `superpowers:code-reviewer` | 1, 6, 21, 25 | Drift from plan |

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
2. **Start with Team 9** (PR Review) if new to agent teams - read-only, low-risk
3. **Use delegate mode** (`Shift+Tab`) to prevent the lead from implementing tasks itself
4. **Monitor progress** with `Shift+Up/Down` (in-process) or click panes (tmux)
5. **`Ctrl+T`** toggles the shared task list view
6. **Prefer Sonnet** for most teammates (cost-effective); use Opus for architecture/design decisions
7. **Avoid file conflicts** — ensure each teammate owns distinct directories
8. **Read-only reviewers** (Teams 26, 33) have zero conflict risk — they never edit files

## References

- [Official Agent Teams Documentation](https://code.claude.com/docs/en/agent-teams)
- [Claude Code Subagents](https://code.claude.com/docs/en/sub-agents)
- [Claude Code Agent Teams Blog](https://addyosmani.com/blog/claude-code-agent-teams/)
