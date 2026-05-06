---
name: dev-hub
description: Top-level router for all software development lifecycle topics. Use for: REST APIs/Node.js/Express/Fastify/FastAPI/asyncio/GraphQL/WebSockets/message queues; frontend accessibility/scientific GUIs/PyQt/cross-platform testing; system architecture/microservices/monorepo/containers/cloud/CLI tools/Terraform/K8s; test automation/TDD/E2E/coverage/code review/plugin validation; GitHub Actions/GitLab CI/deployment pipelines/security scanning/CI errors; Prometheus/Grafana/distributed tracing/SLOs/monitoring/observability/incident response; Python packaging/uv/ruff/mypy/performance profiling/error handling/legacy migration; database schema/SQL optimization/caching/search/authentication/secrets management; Git workflow/technical documentation/Airflow data pipelines/systematic debugging.
---

# Dev Suite (dev-hub)

Top-level entry point consolidating all dev-suite hub skills into a single router.

## Expert Agents

- **`app-developer`** — Full-stack feature implementation and product engineering
- **`automation-engineer`** — CI/CD pipelines, scripting, and workflow automation
- **`debugger-pro`** — Root-cause analysis and systematic bug resolution
- **`devops-architect`** — Infrastructure, Kubernetes, Terraform, and cloud design
- **`documentation-expert`** — API docs, READMEs, and technical writing
- **`quality-specialist`** — Test strategy, coverage, and code quality gates
- **`software-architect`** — System design, service decomposition, and contracts
- **`sre-expert`** — SLOs, incident response, and reliability engineering
- **`systems-engineer`** — Low-level systems, performance, and OS-level concerns

## Hub Skills

- [backend-patterns](../backend-patterns/SKILL.md) — Node.js, async Python, REST/GraphQL/WebSocket, message queues
- [frontend-and-mobile](../frontend-and-mobile/SKILL.md) — React/Vue/Svelte, React Native/Flutter, UI patterns
- [architecture-and-infra](../architecture-and-infra/SKILL.md) — System design, microservices, clean architecture, Terraform/K8s
- [testing-and-quality](../testing-and-quality/SKILL.md) — TDD, test automation, e2e testing, code quality
- [ci-cd-pipelines](../ci-cd-pipelines/SKILL.md) — GitHub Actions, GitLab CI, deployment pipelines
- [observability-and-sre](../observability-and-sre/SKILL.md) — Prometheus, Grafana, SLOs, distributed tracing, incident response
- [python-toolchain](../python-toolchain/SKILL.md) — uv/ruff/mypy, packaging, performance optimization, type hints
- [data-and-security](../data-and-security/SKILL.md) — Databases, SQL optimization, secrets management, auth patterns
- [dev-workflows](../dev-workflows/SKILL.md) — Git workflow, documentation, debugging, Airflow pipelines
- [three-brain](../three-brain/SKILL.md) — Internal multi-agent orchestration
- [ai-pair](../ai-pair/SKILL.md) — AI pair programming patterns

## Routing Decision Tree

```
What is the primary concern?
|
+-- Server-side API / async service / messaging?
|   --> backend-patterns
|
+-- UI components / mobile app / frontend framework?
|   --> frontend-and-mobile
|
+-- System design / infra / cloud / containers?
|   --> architecture-and-infra
|
+-- Tests / quality gates / coverage?
|   --> testing-and-quality
|
+-- Build pipelines / deployment automation?
|   --> ci-cd-pipelines
|
+-- Metrics / alerting / SLOs / incidents?
|   --> observability-and-sre
|
+-- Python packaging / linting / typing / perf?
|   --> python-toolchain
|
+-- Database / SQL / secrets / auth?
|   --> data-and-security
|
+-- Git workflow / docs / debugging / Airflow?
|   --> dev-workflows
|
+-- AI-assisted coding patterns?
|   --> ai-pair
|
+-- Complex multi-agent orchestration?
    --> three-brain
```

## Routing Table

| Trigger | Hub skill |
|---|---|
| REST, GraphQL, Node.js, FastAPI, queues | backend-patterns |
| React, Vue, Svelte, React Native, Flutter | frontend-and-mobile |
| Microservices, Terraform, K8s, clean arch | architecture-and-infra |
| TDD, pytest, Playwright, coverage, linting | testing-and-quality |
| GitHub Actions, GitLab CI, deploy pipeline | ci-cd-pipelines |
| Prometheus, Grafana, tracing, SLOs, oncall | observability-and-sre |
| uv, ruff, mypy, packaging, type hints | python-toolchain |
| SQL, database design, secrets, JWT, OAuth | data-and-security |
| Git branching, docs, debugging, Airflow | dev-workflows |
| LLM code assistance, pair programming | ai-pair |
| Multi-agent plan, internal orchestration | three-brain |

## Checklist

- [ ] Identify the domain (backend / frontend / infra / data / quality / ops) before routing
- [ ] Check if the task spans multiple hubs — if so, start with `architecture-and-infra` for design then delegate
- [ ] Confirm the target language/framework so the correct sub-skill is activated
- [ ] Verify security considerations are addressed (secrets, auth, input validation)
- [ ] Ensure tests are planned or updated whenever implementation changes
- [ ] Confirm observability hooks (logs, metrics, traces) are in scope for production changes
- [ ] Check CI/CD pipeline is updated when new build steps or env vars are introduced
- [ ] Use `three-brain` for tasks requiring coordinated parallel agent execution
