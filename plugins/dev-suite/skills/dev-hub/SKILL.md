---
name: dev-hub
description: >-
  Top-level router for software development lifecycle topics, with a dedicated track for scientific-computing code quality. Use for: FastAPI/asyncio backend services; full-stack web/mobile app development; system architecture (microservices/containers/cloud/CLI tools/Terraform/K8s); test automation/TDD/E2E/coverage/code review/plugin validation; numerical-precision and reproducibility validation for JAX/Julia scientific codebases; GitHub Actions/GitLab CI/deployment pipelines/security scanning/CI errors; Prometheus/Grafana/distributed tracing/SLOs/monitoring/observability/incident response; database schema/SQL optimization/caching/search/authentication/secrets management; Git workflow/technical documentation/Airflow data pipelines/systematic debugging; multi-model Codex+Agy review (one-shot second opinion or a persistent dev team/content team)/team-stop/ai pair programming/start dev team/start content team/three-model collaboration/dual-model review/ongoing iterative review.
---

# Dev Suite (dev-hub)

Top-level entry point consolidating all dev-suite hub skills into a single router.

## Expert Agents

- **`app-developer`** — Full-stack application development: React/Next.js web, Flutter/React Native/Swift/Kotlin mobile, performance & accessibility
- **`automation-engineer`** — CI/CD pipeline architecture: GitHub Actions/GitLab CI, git workflow automation, release engineering, build optimization
- **`documentation-expert`** — Technical documentation architecture: API references, ADRs, tutorials, docs-as-code (Diátaxis framework)
- **`quality-specialist`** — Code quality & scientific-computing validation: numerical precision/reproducibility audits (JAX/Julia), plus OWASP security review and test strategy design
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
- [three-brain](../three-brain/SKILL.md) — Multi-model Codex/Agy review: one-shot second opinion, multimodal/long-context scan (Route mode), or a persistent dev/content team (Team mode)

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
+-- Multi-model Codex+Agy review — one-shot or persistent team?
|   Triggers: second opinion, ask Codex, ask Agy, sanity check, multimodal,
|   long-context scan (Route mode) — OR /ai-pair, "start dev team",
|   "start content team", "team-stop", "pair with codex and agy",
|   "multi-model review", "three-model team", "content team" (Team mode)
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
| Second opinion, sanity check, ask Codex, ask Agy, review your work, cross-check, multimodal analysis, long-context scan, repeated failure (one-shot Route mode); dev team, content team, team-stop, Codex+Agy review, multi-model review, three-model team, dual reviewer, ongoing iterative review, pair programming with AI models (persistent Team mode) | dev-suite:three-brain |

## Checklist

- [ ] Identify the domain (backend / frontend / infra / data / quality / ops) before routing
- [ ] Check if the task spans multiple hubs — if so, start with `architecture-and-infra` for design then delegate
- [ ] Confirm the target language/framework so the correct sub-skill is activated
- [ ] Verify security considerations are addressed (secrets, auth, input validation)
- [ ] Ensure tests are planned or updated whenever implementation changes
- [ ] Confirm observability hooks (logs, metrics, traces) are in scope for production changes
- [ ] Check CI/CD pipeline is updated when new build steps or env vars are introduced
- [ ] Use `dev-suite:three-brain` Route mode for one-shot second opinions, high-risk path scrutiny, repeated failures, or multimodal/long-context review
- [ ] Use `dev-suite:three-brain` Team mode for sustained multi-task projects needing iterative creation + dual review
