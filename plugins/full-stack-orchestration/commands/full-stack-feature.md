---
version: "1.0.5"
description: Orchestrate end-to-end full-stack feature development across frontend, backend, database, testing, security, and deployment
execution_time:
  quick: "30-60 minutes"
  standard: "3-6 hours"
  deep: "1-3 days"
external_docs:
  - architecture-patterns-library.md
  - testing-strategies.md
  - deployment-patterns.md
agents:
  primary:
    - observability-monitoring:database-optimizer
    - backend-development:backend-architect
    - frontend-mobile-development:frontend-developer
    - unit-testing:test-automator
    - full-stack-orchestration:deployment-engineer
  conditional: []
color: purple
tags: [full-stack, orchestration, api-first, e2e-workflow]
allowed-tools: [Read, Write, Edit, Bash, Glob, Grep, Task]
---

# Full-Stack Feature Development

Orchestrate systematic end-to-end feature development with API-first, contract-driven development.

## Feature

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Output |
|------|----------|--------|
| `--quick` | 30-60 min | Architecture & design only |
| standard (default) | 3-6 hours | Architecture + implementation |
| `--deep` | 1-3 days | Complete with testing, security, deployment |

**Options:** `--stack="React/FastAPI/PostgreSQL"`, `--api-style=rest|graphql`, `--deployment-target=aws|gcp`

---

## Phase 1: Architecture & Design

**Agents:** database-optimizer, backend-architect, frontend-developer

### Database Design
- Entity relationship model
- Table schemas with proper types
- Indexing strategy
- Migration scripts (zero-downtime)

### Backend Architecture
- API contracts (OpenAPI/GraphQL)
- Authentication/authorization flows
- Caching strategy
- Resilience patterns

### Frontend Architecture
- Component hierarchy
- State management approach
- Data fetching patterns
- Accessibility requirements

**Success:** Schema normalized, API documented, components follow SRP, ADRs written

🚨 **Quick Mode exits here**

---

## Phase 2: Parallel Implementation

**Agents:** fastapi-pro/django-pro, frontend-developer, database-optimizer

### Backend
- RESTful/GraphQL endpoints
- Business logic layer
- Input validation (Pydantic)
- Structured logging + observability
- Unit tests (80%+ coverage)

### Frontend
- React/Vue components
- State management
- API client with interceptors
- Form validation
- Component tests (70%+ coverage)

### Database
- Migration scripts
- Query optimization
- Connection pooling
- Indexes for query patterns

**Success:** All endpoints implemented, components rendering, migrations run, tests passing

---

## Phase 3: Integration & Testing

**Agents:** test-automator, security-auditor

### Contract Testing
- Pact provider/consumer tests
- OpenAPI validation
- Authentication flow tests
- Error response validation

### E2E Testing
- Playwright/Cypress for critical paths
- Cross-browser compatibility
- Mobile responsiveness
- Visual regression tests

### Security Audit
- Authentication review
- Authorization logic
- Input sanitization (XSS prevention)
- SQL injection validation
- OWASP Top 10 compliance

**Success:** 100% contract coverage, E2E passing, zero critical vulnerabilities

🚨 **Standard Mode exits here**

---

## Phase 4: Deployment & Operations

**Agents:** deployment-engineer, performance-engineer

### Infrastructure
- Dockerfiles for backend/frontend
- Kubernetes manifests
- CI/CD pipeline (GitHub Actions)
- Feature flags configuration
- Blue-green deployment

### Observability
- Distributed tracing (OpenTelemetry)
- Metrics (Prometheus)
- Centralized logging
- Dashboards + alerts
- SLIs/SLOs defined

### Performance
- Query optimization
- Multi-tier caching
- Bundle optimization
- Core Web Vitals validation

**Success:** CI/CD operational, observability capturing 100% traffic, rollback tested

🎯 **Deep Mode complete**

---

## Decision Trees

### Stack Selection
```
SSR needed? → Next.js + Django/FastAPI
Full type safety? → TypeScript + NestJS
Flexible schema? → MongoDB
Real-time? → NestJS or FastAPI + WebSockets
Team: Python → React/FastAPI | JS → Next.js/NestJS
```

### API Style
```
Flexible queries needed? → GraphQL
Simple CRUD? → REST
Real-time subscriptions? → GraphQL or WebSockets
Mobile with limited bandwidth? → GraphQL
```

---

## Agent Orchestration

**Phase 1** (Sequential):
1. database-optimizer → Schema design
2. backend-architect → API design (depends on schema)
3. frontend-developer → Component design (depends on API)

**Phase 2** (Parallel):
- Backend + Frontend + Database in parallel

**Phase 3** (Sequential with gates):
- Contract tests must pass before E2E
- E2E must pass before security audit

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Test Coverage | 80-90% backend, 70-80% frontend |
| Security | Zero critical/high vulnerabilities |
| Performance | p95 < 500ms, p99 < 1s |
| Availability | 99.9% SLO |
| Deploy Frequency | Daily with <5min rollback |

---

## Integration

- Before: `/code-migrate` if upgrading frameworks
- During: `/component-scaffold` for complex UI
- After: `/double-check` for validation
- Before deploy: `/run-all-tests`
