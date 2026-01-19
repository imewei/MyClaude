---
version: "2.1.0"
description: Orchestrate end-to-end feature development with customizable methodologies (TDD, BDD, DDD) and deployment strategies
category: "backend-development"
command: "/feature-development"

execution-modes:
  quick: "1-2d: Steps 4,5,7,10 only - hot fixes, simple CRUD"
  standard: "3-14d: All 12 steps - full production workflow (default)"
  enterprise: "2-4w: All steps + compliance, legal, multi-region"

documentation:
  detailed-guides: "../docs/backend-development/methodology-guides.md"
  phase-templates: "../docs/backend-development/phase-templates.md"
  agent-patterns: "../docs/backend-development/agent-orchestration.md"
  deployment: "../docs/backend-development/deployment-strategies.md"
  best-practices: "../docs/backend-development/best-practices.md"
  metrics: "../docs/backend-development/success-metrics.md"
---

Orchestrate end-to-end feature development: $ARGUMENTS

| Phase | Step | Agent Type | Primary Role |
|-------|------|------------|--------------|
| 1 | 2 | comprehensive-review:architect-review | Technical architecture design |
| 1 | 3 | comprehensive-review:security-auditor | Security & risk assessment |
| 2 | 4 | backend-development:backend-architect | Backend services implementation |
| 2 | 5 | frontend-mobile-development:frontend-developer | Frontend UI implementation |
| 3 | 7 | unit-testing:test-automator | Test suite creation |
| 3 | 8 | comprehensive-review:security-auditor | Security validation |
| 3 | 9 | full-stack-orchestration:performance-engineer | Performance optimization |
| 4 | 10 | cicd-automation:deployment-engineer | CI/CD pipeline setup |
| 4 | 11 | observability-monitoring:observability-engineer | Monitoring & alerting |
| 4 | 12 | code-documentation:docs-architect | Documentation generation |

## Configuration

**Methodology**: traditional (sequential) | tdd (test-first) | bdd (scenario-based) | ddd (domain-driven)
**Complexity**: simple (1-2d) | medium (3-5d) | complex (1-2w) | epic (2w+)
**Deployment**: direct | canary (5% rollout) | feature-flag | blue-green | a-b-test

## Phase 1: Discovery & Planning

1. **Business Analysis** - Requirements doc with user stories, success metrics, risk assessment
2. **Technical Architecture** (architect-review) - Service boundaries, API contracts, data models
3. **Security Assessment** (security-auditor) - Risk matrix, compliance checklist, mitigations

## Phase 2: Implementation

4. **Backend Services** (backend-architect) - RESTful/GraphQL APIs, business logic, caching, feature flags
5. **Frontend** (frontend-developer) - UI, state management, error handling, analytics
6. **Data Pipeline** - ETL/ELT processes, validation, quality monitoring

## Phase 3: Testing & QA

7. **Test Suite** (test-automator) - Unit, integration, E2E tests (80%+ coverage)
8. **Security Validation** (security-auditor) - OWASP checks, pen testing, dependency scanning
9. **Performance** (performance-engineer) - Query optimization, caching, bundle size reduction

## Phase 4: Deployment & Observability

10. **CI/CD Pipeline** (deployment-engineer) - Automated tests, feature flags, rollback procedures
11. **Monitoring** (observability-engineer) - Distributed tracing, metrics, alerting, SLOs/SLIs
12. **Documentation** (docs-architect) - API docs, user guides, deployment guides, runbooks

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
