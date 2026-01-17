---
description: Workflow for feature-development
triggers:
- /feature-development
- workflow for feature development
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



Orchestrate end-to-end feature development: $ARGUMENTS

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

| Phase | Step | Agent Type | Primary Role | Execution |
|-------|------|------------|--------------|-----------|
| 1 | 2 | comprehensive-review:architect-review | Technical architecture design | Sequential |
| 1 | 3 | comprehensive-review:security-auditor | Security & risk assessment | Sequential |
| 2 | 4 | backend-development:backend-architect | Backend services implementation | **Parallel** (depends: 2) |
| 2 | 5 | frontend-mobile-development:frontend-developer | Frontend UI implementation | **Parallel** (depends: 2) |
| 3 | 7 | unit-testing:test-automator | Test suite creation | **Parallel** (depends: 4,5) |
| 3 | 8 | comprehensive-review:security-auditor | Security validation | **Parallel** (depends: 4,5) |
| 3 | 9 | full-stack-orchestration:performance-engineer | Performance optimization | **Parallel** (depends: 4,5) |
| 4 | 10 | cicd-automation:deployment-engineer | CI/CD pipeline setup | Sequential |
| 4 | 11 | observability-monitoring:observability-engineer | Monitoring & alerting | **Parallel** (depends: 10) |
| 4 | 12 | code-documentation:docs-architect | Documentation generation | **Parallel** (depends: 10) |

## Configuration

**Methodology**: traditional (sequential) | tdd (test-first) | bdd (scenario-based) | ddd (domain-driven)
**Complexity**: simple (1-2d) | medium (3-5d) | complex (1-2w) | epic (2w+)
**Deployment**: direct | canary (5% rollout) | feature-flag | blue-green | a-b-test

## Phase 1: Discovery & Planning (Sequential)

1. **Business Analysis** - Requirements doc with user stories, success metrics, risk assessment
2. **Technical Architecture** (architect-review) - Service boundaries, API contracts, data models
3. **Security Assessment** (security-auditor) - Risk matrix, compliance checklist, mitigations

## Phase 2: Implementation (Parallel Execution)

> **Orchestration Note**: Execute Backend and Frontend streams concurrently. Ensure API contract (Step 2) is frozen first.

4. **Backend Services** (backend-architect) `(depends_on: Step 2)`
   - RESTful/GraphQL APIs, business logic, caching, feature flags
5. **Frontend** (frontend-developer) `(depends_on: Step 2)`
   - UI, state management, error handling, analytics
6. **Data Pipeline** `(depends_on: Step 2)`
   - ETL/ELT processes, validation, quality monitoring

## Phase 3: Testing & QA (Parallel Execution)

> **Orchestration Note**: Start these tasks once Phase 2 is functionally complete.

7. **Test Suite** (test-automator) `(depends_on: Step 4, 5)`
   - Unit, integration, E2E tests (80%+ coverage)
8. **Security Validation** (security-auditor) `(depends_on: Step 4, 5)`
   - OWASP checks, pen testing, dependency scanning
9. **Performance** (performance-engineer) `(depends_on: Step 4, 5)`
   - Query optimization, caching, bundle size reduction

## Phase 4: Deployment & Observability (Mixed)

10. **CI/CD Pipeline** (deployment-engineer) `(depends_on: Step 7, 8)`
    - Automated tests, feature flags, rollback procedures
11. **Monitoring** (observability-engineer) `(depends_on: Step 10)`
    - Distributed tracing, metrics, alerting, SLOs/SLIs
12. **Documentation** (docs-architect) `(depends_on: Step 10)`
    - API docs, user guides, deployment guides, runbooks

## Parameters

**Required**: `--feature`, `--methodology`, `--complexity`
**Optional**: `--mode` (quick|standard|enterprise), `--deployment-strategy`, `--test-coverage-min` (default 80%), `--performance-budget`, `--rollout-percentage` (default 5%), `--feature-flag-service`, `--analytics-platform`, `--monitoring-stack`

## Success Criteria

**Phase 1**: Requirements >90%, risk matrix, stakeholder sign-off
**Phase 2**: API contract 100%, feature flags configured, responsive UI
**Phase 3**: Coverage ≥80%, zero critical vulnerabilities, p95 <200ms
**Phase 4**: Deployment successful, monitoring live, docs published, rollback tested

**Overall**: All acceptance criteria met, security scan clean, performance budget met, feature flags configured, monitoring operational, analytics tracking

## Rollback

1. Feature flag disable (<1 min)
2. Blue-green traffic switch (<5 min)
3. Full deployment rollback (<15 min)
4. Database migration rollback (coordinate with data team)
5. Incident post-mortem before re-deployment
