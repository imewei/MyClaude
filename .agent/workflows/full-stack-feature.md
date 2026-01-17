---
description: Orchestrate end-to-end full-stack feature development across frontend,
  backend, database, testing, security, and deployment
triggers:
- /full-stack-feature
- orchestrate end to end full stack feature
allowed-tools: [Bash, Read, Write, Edit, Task, Glob, Grep]
version: 1.0.0
---



# Full-Stack Feature Development

$ARGUMENTS

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

**Options:** `--stack="React/FastAPI/PostgreSQL"`, `--api-style=rest|graphql`, `--deployment-target=aws|gcp`

## Phase 1: Architecture & Design (Sequential)

**Database:** Entity relationship model, table schemas with types, indexing strategy, migration scripts (zero-downtime)

**Backend:** API contracts (OpenAPI/GraphQL), authentication/authorization flows, caching strategy, resilience patterns

**Frontend:** Component hierarchy, state management approach, data fetching patterns, accessibility requirements

**Success:** Schema normalized, API documented, components follow SRP, ADRs written

🚨 **Quick mode:** Exit here

## Phase 2: Parallel Implementation (Parallel Execution)

> **Orchestration Note**: Execute Backend, Frontend, and Database work concurrently. Ensure contracts are frozen in Phase 1.

**Backend:** RESTful/GraphQL endpoints, business logic layer, input validation (Pydantic), structured logging + observability, unit tests (80%+ coverage)

**Frontend:** React/Vue components, state management, API client with interceptors, form validation, component tests (70%+ coverage)

**Database:** Migration scripts, query optimization, connection pooling, indexes for query patterns

**Success:** Endpoints implemented, components rendering, migrations run, tests passing

## Phase 3: Integration & Testing (Parallel Execution)

> **Orchestration Note**: Execute testing streams concurrently.

**Contract Testing:** Pact provider/consumer tests, OpenAPI validation, authentication flow tests, error response validation

**E2E Testing:** Playwright/Cypress for critical paths, cross-browser compatibility, mobile responsiveness, visual regression tests

**Security Audit:** Authentication review, authorization logic, input sanitization (XSS prevention), SQL injection validation, OWASP Top 10 compliance

**Success:** 100% contract coverage, E2E passing, zero critical vulnerabilities

🚨 **Standard mode complete**

## Phase 4: Deployment & Operations (Mixed)

**Infrastructure:** Dockerfiles for backend/frontend, Kubernetes manifests, CI/CD pipeline (GitHub Actions), feature flags configuration, blue-green deployment

**Observability:** Distributed tracing (OpenTelemetry), metrics (Prometheus), centralized logging, dashboards + alerts, SLIs/SLOs defined

**Performance:** Query optimization, multi-tier caching, bundle optimization, Core Web Vitals validation

**Success:** CI/CD operational, observability capturing 100% traffic, rollback tested

## Decision Trees

**Stack:**
- SSR needed? → Next.js + Django/FastAPI
- Full type safety? → TypeScript + NestJS
- Flexible schema? → MongoDB
- Real-time? → NestJS or FastAPI + WebSockets
- Team: Python → React/FastAPI | JS → Next.js/NestJS

**API Style:**
- Flexible queries? → GraphQL
- Simple CRUD? → REST
- Real-time subscriptions? → GraphQL or WebSockets
- Mobile with limited bandwidth? → GraphQL

## Success Metrics

- Test Coverage: 80-90% backend, 70-80% frontend
- Security: Zero critical/high vulnerabilities
- Performance: p95 <500ms, p99 <1s
- Availability: 99.9% SLO
- Deploy Frequency: Daily with <5min rollback
