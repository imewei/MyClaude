---
version: "2.1.0"
description: Unified end-to-end feature development with customizable methodologies and deployment strategies
argument-hint: <action> <feature-name> [options]
category: engineering-suite
execution-time:
  quick: "30m-2d: Architecture + hot fixes"
  standard: "3-14d: Full production workflow"
  deep: "2-4w: Enterprise with compliance"
color: purple
allowed-tools: [Bash, Read, Write, Edit, Task, Glob, Grep, Bash(uv:*)]
external-docs:
  - architecture-patterns-library.md
  - methodology-guides.md
  - testing-strategies.md
  - deployment-patterns.md
tags: [full-stack, orchestration, api-first, feature-development]
---

# Engineering Feature Development

$ARGUMENTS

## Actions

| Action | Description |
|--------|-------------|
| `plan` | Architecture design and planning only |
| `build` | Full implementation with testing |
| `deploy` | Deployment and observability setup |
| `full` | Complete end-to-end workflow (default) |

**Examples:**
```bash
/eng-feature-dev plan user-authentication --methodology tdd
/eng-feature-dev build shopping-cart --stack "React/FastAPI/PostgreSQL"
/eng-feature-dev full payment-system --mode deep --deployment canary
```

## Options

**Core:**
- `--mode <depth>`: quick, standard (default), deep
- `--methodology <approach>`: traditional, tdd (test-first), bdd (scenario-based), ddd (domain-driven)
- `--complexity <level>`: simple (1-2d), medium (3-5d), complex (1-2w), epic (2w+)

**Stack:**
- `--stack <tech>`: "React/FastAPI/PostgreSQL", "Next.js/NestJS/MongoDB", etc.
- `--api-style <type>`: rest, graphql

**Deployment:**
- `--deployment <strategy>`: direct, canary (5% rollout), feature-flag, blue-green, a-b-test
- `--target <platform>`: aws, gcp, azure, k8s

---

## Phase 1: Discovery & Design

### Step 1: Business Analysis
- Requirements document with user stories
- Success metrics and acceptance criteria
- Risk assessment and stakeholder sign-off

### Step 2: Technical Architecture
**Agent:** quality-suite:architect-review

- Database: Entity relationship model, schemas, indexing, migrations (zero-downtime)
- Backend: API contracts (OpenAPI/GraphQL), auth flows, caching, resilience patterns
- Frontend: Component hierarchy, state management, data fetching, accessibility
- ADRs (Architecture Decision Records) written

### Step 3: Security Assessment
**Agent:** quality-suite:quality-specialist

- Threat modeling and risk matrix
- Compliance checklist (GDPR, SOC2, etc.)
- Security controls and mitigations

**Quick mode:** Exit after Phase 1

---

## Phase 2: Implementation

### Step 4: Backend Services
**Agent:** engineering-suite:software-architect

- RESTful/GraphQL endpoints with OpenAPI docs
- Business logic layer with validation (Pydantic/Zod)
- Structured logging and observability hooks
- Feature flags integration
- Unit tests (80%+ coverage)

### Step 5: Frontend Implementation
**Agent:** engineering-suite:app-developer

- React/Vue components following SRP
- State management (Redux/Zustand/Context)
- API client with interceptors and error handling
- Form validation and accessibility
- Component tests (70%+ coverage)

### Step 6: Database & Data Pipeline
- Migration scripts (reversible)
- Query optimization and connection pooling
- Indexes for query patterns
- ETL/ELT processes if needed
- Data quality monitoring

---

## Phase 3: Testing & QA

### Step 7: Test Suite
**Agent:** quality-suite:quality-specialist

- Unit, integration, E2E tests
- Contract testing (Pact provider/consumer)
- OpenAPI validation
- Cross-browser and mobile testing
- Visual regression tests

### Step 8: Security Validation
**Agent:** quality-suite:quality-specialist

- OWASP Top 10 compliance check
- Authentication/authorization review
- Input sanitization (XSS prevention)
- SQL injection validation
- Dependency vulnerability scanning

### Step 9: Performance Optimization
**Agent:** agent-core:sre-expert

- Query optimization and caching
- Bundle size reduction
- Core Web Vitals validation
- Load testing (p95 <500ms, p99 <1s)
- Memory leak detection

**Standard mode:** Exit after Phase 3

---

## Phase 4: Deployment & Operations

### Step 10: CI/CD Pipeline
**Agent:** infrastructure-suite:automation-engineer

- Dockerfiles for backend/frontend
- Kubernetes manifests or serverless configs
- GitHub Actions / GitLab CI pipeline
- Feature flags configuration
- Blue-green or canary deployment setup
- Rollback procedures tested

### Step 11: Observability
**Agent:** infrastructure-suite:observability-engineer

- Distributed tracing (OpenTelemetry)
- Metrics collection (Prometheus)
- Centralized logging
- Dashboards and alerts
- SLIs/SLOs defined

### Step 12: Documentation
**Agent:** quality-suite:documentation-expert

- API documentation
- User guides
- Deployment guides
- Runbooks for operations
- Architecture diagrams

---

## Stack Decision Trees

**Framework Selection:**
| Requirement | Recommended Stack |
|-------------|-------------------|
| SSR needed | Next.js + Django/FastAPI |
| Full type safety | TypeScript + NestJS |
| Flexible schema | MongoDB + GraphQL |
| Real-time | FastAPI/NestJS + WebSockets |
| Python team | React + FastAPI |
| JS team | Next.js + NestJS |

**API Style:**
| Requirement | Recommendation |
|-------------|----------------|
| Flexible queries | GraphQL |
| Simple CRUD | REST |
| Real-time subscriptions | GraphQL or WebSockets |
| Mobile (bandwidth) | GraphQL |

---

## Success Criteria

| Phase | Criteria |
|-------|----------|
| 1 | Requirements >90%, risk matrix complete, stakeholder sign-off |
| 2 | API contract 100%, feature flags configured, responsive UI |
| 3 | Coverage ≥80%, zero critical vulnerabilities, p95 <200ms |
| 4 | Deployment successful, monitoring live, docs published |

**Overall Metrics:**
- Test Coverage: 80-90% backend, 70-80% frontend
- Security: Zero critical/high vulnerabilities
- Performance: p95 <500ms, p99 <1s
- Availability: 99.9% SLO
- Deploy Frequency: Daily with <5min rollback

---

## Rollback Procedures

| Priority | Action | Time |
|----------|--------|------|
| 1 | Feature flag disable | <1 min |
| 2 | Blue-green traffic switch | <5 min |
| 3 | Full deployment rollback | <15 min |
| 4 | Database migration rollback | Coordinate with data team |

Always run incident post-mortem before re-deployment.
