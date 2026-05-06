---
name: dev-hub
description: >-
  Top-level router for all software development lifecycle topics. Use for: REST APIs/Node.js/Express/Fastify/FastAPI/asyncio/GraphQL/WebSockets/message queues; frontend accessibility/scientific GUIs/PyQt/cross-platform testing; system architecture/microservices/monorepo/containers/cloud/CLI tools/Terraform/K8s; test automation/TDD/E2E/coverage/code review/plugin validation; GitHub Actions/GitLab CI/deployment pipelines/security scanning/CI errors; Prometheus/Grafana/distributed tracing/SLOs/monitoring/observability/incident response; Python packaging/uv/ruff/mypy/performance profiling/error handling/legacy migration; database schema/SQL optimization/caching/search/authentication/secrets management; Git workflow/technical documentation/Airflow data pipelines/systematic debugging; AI pair programming/multi-model dev team/Codex+Gemini review pipeline/content team/team-stop/ai-pair/start dev team/start content team/three-model collaboration/dual-model review/ongoing iterative review.
---

# Dev Suite (dev-hub)

Top-level entry point consolidating all dev-suite hub skills into a single router.

## Expert Agents

- **`app-developer`** — Scientific GUI and data-interface implementation
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
- [three-brain](../three-brain/SKILL.md) — Second-opinion review via Codex/Gemini, multimodal analysis, long-context scan (one-shot)
- [ai-pair](../ai-pair/SKILL.md) — AI pair programming patterns

## Routing Decision Tree

```
What is the primary concern?
|
+-- Server-side API / async service / messaging?
|   --> backend-patterns
|
+-- UI components / mobile app / frontend framework / TypeScript project / WCAG / accessibility?
|   --> frontend-and-mobile
|
+-- System design / infra / cloud / containers?
|   --> architecture-and-infra
|
+-- Tests / quality gates / coverage / PR review / schema validation?
|   --> testing-and-quality
|
+-- Build pipelines / deployment automation / SAST / security scanning / CI failure / flaky test?
|   --> ci-cd-pipelines
|
+-- Metrics / alerting / SLOs / incidents?
|   --> observability-and-sre
|
+-- Python packaging / uv / typing / profiling / error handling / legacy migration?
|   --> python-toolchain
|
+-- Database / SQL / caching / search / secrets / auth?
|   --> data-and-security
|
+-- Git workflow / docs / debugging / Airflow?
|   --> dev-workflows
|
+-- AI pair programming / multi-model team / Codex+Gemini review?
|   Triggers: /ai-pair, "start dev team", "start content team", "team-stop",
|   "pair with codex and gemini", "multi-model review", "three-model team",
|   "dual-model review", "ongoing review pipeline", "content team"
|   --> ai-pair
|
+-- Second opinion / ask Codex / ask Gemini / sanity check / multimodal / long-context scan (one-shot)?
|   --> three-brain
|
+-- None of the above / concern is ambiguous or spans multiple areas?
    --> Delegate to software-architect for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Routing Table

| Trigger | Hub skill |
|---|---|
| REST, GraphQL, Node.js, FastAPI, queues | backend-patterns |
| React, Vue, Svelte, React Native, Flutter, TypeScript, tsconfig, WCAG, ARIA, accessibility, mobile testing | frontend-and-mobile |
| Microservices, Terraform, K8s, clean arch | architecture-and-infra |
| TDD, pytest, Playwright, Cypress, coverage, PR review, code review, schema validation, Pydantic, Zod, plugin manifest | testing-and-quality |
| GitHub Actions, GitLab CI, deploy pipeline, SAST, Snyk, Trivy, SBOM, secrets scan, CI failure, flaky test, build error | ci-cd-pipelines |
| Prometheus, Grafana, tracing, SLOs, oncall | observability-and-sre |
| uv, ruff, mypy, packaging, type hints, pyproject.toml, PyPI, profiling, Cython, try/except, retry, Python migration | python-toolchain |
| SQL, database design, migrations, Redis, caching, Elasticsearch, vector search, secrets, JWT, OAuth, RBAC, Vault | data-and-security |
| Git branching, docs, debugging, Airflow | dev-workflows |
| /ai-pair, dev team, content team, team-stop, Codex+Gemini review, multi-model review, three-model team, dual reviewer, ongoing iterative review, pair programming with AI models | ai-pair |
| Second opinion, sanity check, ask Codex, ask Gemini, review your work, cross-check, multimodal analysis, long-context scan, repeated failure (one-shot) | three-brain |

## Checklist

- [ ] Identify the domain (backend / frontend / infra / data / quality / ops) before routing
- [ ] Check if the task spans multiple hubs — if so, start with `architecture-and-infra` for design then delegate
- [ ] Confirm the target language/framework so the correct sub-skill is activated
- [ ] Verify security considerations are addressed (secrets, auth, input validation)
- [ ] Ensure tests are planned or updated whenever implementation changes
- [ ] Confirm observability hooks (logs, metrics, traces) are in scope for production changes
- [ ] Check CI/CD pipeline is updated when new build steps or env vars are introduced
- [ ] Use `three-brain` for one-shot second opinions, high-risk path scrutiny, repeated failures, or multimodal/long-context review
- [ ] Use `ai-pair` (not `three-brain`) for sustained multi-task projects needing iterative creation + dual review
