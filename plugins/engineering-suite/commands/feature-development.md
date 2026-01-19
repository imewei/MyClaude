---
version: "2.1.0"
description: Orchestrate end-to-end feature development with customizable methodologies (TDD, BDD, DDD) and deployment strategies
category: "engineering-suite"
command: "/feature-development"

execution-modes:
  quick: "1-2d: Steps 4,5,7,10 only - hot fixes, simple CRUD"
  standard: "3-14d: All 12 steps - full production workflow (default)"
  enterprise: "2-4w: All steps + compliance, legal, multi-region"

documentation:
  detailed-guides: "../docs/engineering-suite/methodology-guides.md"
  phase-templates: "../docs/engineering-suite/phase-templates.md"
  agent-patterns: "../docs/engineering-suite/agent-core.md"
  deployment: "../docs/engineering-suite/deployment-strategies.md"
  best-practices: "../docs/engineering-suite/best-practices.md"
  metrics: "../docs/engineering-suite/success-metrics.md"
---

Orchestrate end-to-end feature development: $ARGUMENTS

| Phase | Step | Agent Type | Primary Role |
|-------|------|------------|--------------|
| 1 | 2 | quality-suite:architect-review | Technical architecture design |
| 1 | 3 | quality-suite:quality-specialist | Security & risk assessment |
| 2 | 4 | engineering-suite:software-architect | Backend services implementation |
| 2 | 5 | engineering-suite:app-developer | Frontend UI implementation |
| 3 | 7 | quality-suite:quality-specialist | Test suite creation |
| 3 | 8 | quality-suite:quality-specialist | Security validation |
| 3 | 9 | agent-core:sre-expert | Performance optimization |
| 4 | 10 | infrastructure-suite:automation-engineer | CI/CD pipeline setup |
| 4 | 11 | infrastructure-suite:observability-engineer | Monitoring & alerting |
| 4 | 12 | quality-suite:documentation-expert | Documentation generation |

## Configuration

**Methodology**: traditional (sequential) | tdd (test-first) | bdd (scenario-based) | ddd (domain-driven)
**Complexity**: simple (1-2d) | medium (3-5d) | complex (1-2w) | epic (2w+)
**Deployment**: direct | canary (5% rollout) | feature-flag | blue-green | a-b-test

## Phase 1: Discovery & Planning

1. **Business Analysis** - Requirements doc with user stories, success metrics, risk assessment
2. **Technical Architecture** (architect-review) - Service boundaries, API contracts, data models
3. **Security Assessment** (quality-specialist) - Risk matrix, compliance checklist, mitigations

## Phase 2: Implementation

4. **Backend Services** (software-architect) - RESTful/GraphQL APIs, business logic, caching, feature flags
5. **Frontend** (app-developer) - UI, state management, error handling, analytics
6. **Data Pipeline** - ETL/ELT processes, validation, quality monitoring

## Phase 3: Testing & QA

7. **Test Suite** (quality-specialist) - Unit, integration, E2E tests (80%+ coverage)
8. **Security Validation** (quality-specialist) - OWASP checks, pen testing, dependency scanning
9. **Performance** (sre-expert) - Query optimization, caching, bundle size reduction

## Phase 4: Deployment & Observability

10. **CI/CD Pipeline** (automation-engineer) - Automated tests, feature flags, rollback procedures
11. **Monitoring** (observability-engineer) - Distributed tracing, metrics, alerting, SLOs/SLIs
12. **Documentation** (documentation-expert) - API docs, user guides, deployment guides, runbooks

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
