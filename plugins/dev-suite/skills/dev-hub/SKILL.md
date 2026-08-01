---
name: dev-hub
description: >-
  Top-level router for scientific-computing software development lifecycle topics. Use for: FastAPI/asyncio scientific services; scientific GUIs (PyQt/PySide6); system architecture for numerical/ML/simulation systems, microservices/containers/cloud/CLI tools/Terraform/K8s; test automation/TDD/E2E/coverage/code review/plugin validation for scientific codebases; GitHub Actions/GitLab CI/deployment pipelines/security scanning/CI errors; Prometheus/Grafana/distributed tracing/SLOs/monitoring/observability/incident response for long-running scientific workloads; database schema/SQL optimization/caching/search/authentication/secrets management; Git workflow/technical documentation for numerical/ML codebases/Airflow data pipelines/systematic debugging for scientific computing; AI pair programming/multi-model dev team/Codex+Gemini review pipeline/content team/team-stop/ai-pair/start dev team/start content team/three-model collaboration/dual-model review/ongoing iterative review.
---

# Dev Suite (dev-hub)

Top-level entry point consolidating all dev-suite hub skills into a single router.

## Expert Agents

- **`app-developer`** — Full-stack application development: React/Next.js web, Flutter/React Native/Swift/Kotlin mobile, performance & accessibility
- **`automation-engineer`** — CI/CD pipeline architecture: GitHub Actions/GitLab CI, git workflow automation, release engineering, build optimization
- **`documentation-expert`** — Docs for numerical/ML/SciML codebases: API specs, Sphinx, notebook-to-doc pipelines
- **`quality-specialist`** — Scientific-computing validation: numerical precision, property-based invariants, reproducibility
- **`software-architect`** — System and backend architecture: service boundaries, REST/GraphQL/gRPC API strategy, technical governance
- **`sre-expert`** — Site reliability engineering: observability (metrics/logs/traces), SLOs/error budgets, incident response, capacity engineering

## Hub Skills

- [backend-patterns](../backend-patterns/SKILL.md) — Async Python/FastAPI services for scientific workflows, REST, message queues
- [architecture-and-infra](../architecture-and-infra/SKILL.md) — System design, microservices, clean architecture, Terraform/K8s
- [testing-and-quality](../testing-and-quality/SKILL.md) — TDD, test automation, e2e testing, code quality
- [ci-cd-pipelines](../ci-cd-pipelines/SKILL.md) — GitHub Actions, GitLab CI, deployment pipelines
- [observability-and-sre](../observability-and-sre/SKILL.md) — Prometheus, Grafana, SLOs, distributed tracing, incident response
- [data-and-security](../data-and-security/SKILL.md) — Databases, SQL optimization, secrets management, auth patterns
- [dev-workflows](../dev-workflows/SKILL.md) — Git workflow, documentation, debugging, Airflow pipelines
- [three-brain](../three-brain/SKILL.md) — Second-opinion review via Codex/Gemini, multimodal analysis, long-context scan (one-shot)
- [ai-pair](../ai-pair/SKILL.md) — AI pair programming patterns

## Routing Decision Tree

```
What is the primary concern?
|
+-- Server-side API / async service / messaging?
|   --> dev-suite:backend-patterns
|
+-- System design / infra / cloud / containers?
|   --> dev-suite:architecture-and-infra
|
+-- Tests / quality gates / coverage / PR review / schema validation?
|   --> dev-suite:testing-and-quality
|
+-- Build pipelines / deployment automation / SAST / security scanning / CI failure / flaky test?
|   --> dev-suite:ci-cd-pipelines
|
+-- Metrics / alerting / SLOs / incidents?
|   --> dev-suite:observability-and-sre
|
+-- Database / SQL / caching / search / secrets / auth?
|   --> dev-suite:data-and-security
|
+-- Git workflow / docs / debugging / Airflow?
|   --> dev-suite:dev-workflows
|
+-- AI pair programming / multi-model team / Codex+Gemini review?
|   Triggers: /ai-pair, "start dev team", "start content team", "team-stop",
|   "pair with codex and gemini", "multi-model review", "three-model team",
|   "dual-model review", "ongoing review pipeline", "content team"
|   --> dev-suite:ai-pair
|
+-- Second opinion / ask Codex / ask Gemini / sanity check / multimodal / long-context scan (one-shot)?
|   --> dev-suite:three-brain
|
+-- None of the above / concern is ambiguous or spans multiple areas?
    --> Delegate to software-architect for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Routing Table

| Trigger | Hub skill |
|---|---|
| REST APIs, FastAPI, async Python services, message queues | dev-suite:backend-patterns |
| Microservices, Terraform, K8s, clean arch | dev-suite:architecture-and-infra |
| TDD, pytest, Playwright, Cypress, coverage, PR review, code review, schema validation, Pydantic, Zod, plugin manifest | dev-suite:testing-and-quality |
| GitHub Actions, GitLab CI, deploy pipeline, SAST, Snyk, Trivy, SBOM, secrets scan, CI failure, flaky test, build error | dev-suite:ci-cd-pipelines |
| Prometheus, Grafana, tracing, SLOs, oncall | dev-suite:observability-and-sre |
| SQL, database design, migrations, Redis, caching, Elasticsearch, vector search, secrets, JWT, OAuth, RBAC, Vault | dev-suite:data-and-security |
| Git branching, docs, debugging, Airflow | dev-suite:dev-workflows |
| /ai-pair, dev team, content team, team-stop, Codex+Gemini review, multi-model review, three-model team, dual reviewer, ongoing iterative review, pair programming with AI models | dev-suite:ai-pair |
| Second opinion, sanity check, ask Codex, ask Gemini, review your work, cross-check, multimodal analysis, long-context scan, repeated failure (one-shot) | dev-suite:three-brain |

## Checklist

- [ ] Identify the domain (backend / frontend / infra / data / quality / ops) before routing
- [ ] Check if the task spans multiple hubs — if so, start with `architecture-and-infra` for design then delegate
- [ ] Confirm the target language/framework so the correct sub-skill is activated
- [ ] Verify security considerations are addressed (secrets, auth, input validation)
- [ ] Ensure tests are planned or updated whenever implementation changes
- [ ] Confirm observability hooks (logs, metrics, traces) are in scope for production changes
- [ ] Check CI/CD pipeline is updated when new build steps or env vars are introduced
- [ ] Use `dev-suite:three-brain` for one-shot second opinions, high-risk path scrutiny, repeated failures, or multimodal/long-context review
- [ ] Use `dev-suite:ai-pair` (not `dev-suite:three-brain`) for sustained multi-task projects needing iterative creation + dual review
