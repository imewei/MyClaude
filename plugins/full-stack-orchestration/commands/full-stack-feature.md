---
version: 1.0.3
description: Orchestrate end-to-end full-stack feature development across frontend, backend, database, testing, security, and deployment with API-first approach
execution_time:
  quick: "30-60 minutes - Architecture & design planning only (Phase 1)"
  standard: "3-6 hours - Architecture + implementation (Phases 1-2)"
  deep: "1-3 days - Complete workflow with testing, security, deployment (All 4 phases)"
external_docs:
  - architecture-patterns-library.md
  - testing-strategies.md
  - deployment-patterns.md
  - technology-stack-guide.md
agents:
  primary:
    - observability-monitoring:database-optimizer
    - backend-development:backend-architect
    - frontend-mobile-development:frontend-developer
    - unit-testing:test-automator
    - full-stack-orchestration:deployment-engineer
    - full-stack-orchestration:performance-engineer
    - full-stack-orchestration:security-auditor
  conditional: []
color: purple
tags: [full-stack, orchestration, api-first, microservices, e2e-workflow, feature-development]
allowed-tools: [Read, Write, Edit, Bash, Glob, Grep, Task]
---

# Full-Stack Feature Development Orchestration

> **Purpose**: Orchestrate systematic end-to-end feature development from architecture through production deployment with API-first, contract-driven development and comprehensive quality gates

---

## Command Syntax

```bash
/full-stack-feature "<feature-description>" [options]

# Quick mode: Architecture & design planning only
/full-stack-feature "User authentication with OAuth2" --quick

# Standard mode: Architecture + implementation
/full-stack-feature "Product catalog with search" --stack="React/FastAPI/PostgreSQL"

# Deep mode: Complete workflow with full quality gates
/full-stack-feature "Real-time notifications system" --deep --deployment-target=aws --api-style=graphql
```

---

## Execution Modes

### Quick Mode (30-60 minutes)
**Purpose**: Architecture & design planning phase only

**Output**:
- Database schema design (ERD, tables, indexes, migrations)
- Backend service architecture (API contracts, service boundaries, authentication)
- Frontend component architecture (component tree, state management, routing)
- Technology stack recommendations
- Architecture Decision Records (ADRs)

**When to use**:
- Initial feature planning and scoping
- Architecture review and validation
- Technology selection decisions
- Team alignment on approach

### Standard Mode (3-6 hours)
**Purpose**: Architecture + parallel implementation

**Output**:
- Everything from Quick Mode, plus:
- Backend service implementation (endpoints, business logic, data access)
- Frontend application implementation (components, state, API integration)
- Database implementation (migrations, queries, optimization)
- Basic unit tests for critical paths
- API documentation

**When to use**:
- Feature development sprints
- MVP implementation
- Proof-of-concept validation
- Rapid prototyping with quality

### Deep Mode (1-3 days)
**Purpose**: Production-ready complete workflow

**Output**:
- Everything from Standard Mode, plus:
- Contract tests (Pact/Dredd for API contracts)
- E2E tests (Playwright/Cypress for user journeys)
- Security audit (OWASP Top 10, penetration testing)
- Performance optimization (caching, query optimization, bundle optimization)
- Infrastructure & CI/CD (Docker, Kubernetes, GitHub Actions)
- Observability stack (metrics, logs, traces, dashboards)
- Feature flags & progressive rollout configuration

**When to use**:
- Production feature releases
- Customer-facing functionality
- Mission-critical features
- Enterprise-grade quality requirements

---

## Phase 1: Architecture & Design Foundation

### Objective
Design scalable, maintainable architecture with clear separation of concerns and API-first contracts

### Key Activities

**1. Database Architecture Design**
- Design entity relationship model
- Define table schemas with proper data types
- Plan indexing strategy for query patterns
- Design migration strategy (zero-downtime)
- Specify data validation constraints
- Plan backup and disaster recovery

**2. Backend Service Architecture**
- Define service boundaries and responsibilities
- Design API contracts (OpenAPI/GraphQL schemas)
- Specify authentication & authorization flows
- Design inter-service communication patterns
- Plan caching strategy (multi-tier)
- Design resilience patterns (circuit breakers, retries, timeouts)

**3. Frontend Component Architecture**
- Design component hierarchy and composition
- Select state management approach (Redux/Zustand/Context)
- Plan routing structure and navigation
- Design data fetching patterns (React Query/SWR)
- Specify accessibility requirements (WCAG 2.1 AA)
- Plan responsive design strategy (mobile-first)

### Success Criteria
- ✅ Database schema normalized (3NF) with proper indexes
- ✅ API contracts documented in OpenAPI/GraphQL specification
- ✅ Component architecture follows single responsibility principle
- ✅ State management strategy clearly defined
- ✅ Security patterns identified (auth, authorization, input validation)
- ✅ All architectural decisions documented in ADRs

### Tools
- Use **Task** with `database-optimizer` for schema design
- Use **Task** with `backend-architect` for API design
- Use **Task** with `frontend-developer` for component architecture
- Use **Write** to document ADRs in `docs/architecture/`

**Reference**: See `architecture-patterns-library.md` for detailed patterns

---

## Phase 2: Parallel Implementation

### Objective
Implement frontend, backend, and database components in parallel following architecture specifications

### Key Activities

**4. Backend Service Implementation**
- Implement RESTful/GraphQL endpoints with proper HTTP methods
- Build business logic layer with service classes
- Create data access layer with repository pattern
- Implement authentication middleware (JWT/OAuth2)
- Add input validation with Pydantic/class-validator
- Implement structured logging with context
- Add observability (Prometheus metrics, OpenTelemetry traces)
- Handle errors gracefully with proper status codes
- Write unit tests for business logic (80%+ coverage)

**5. Frontend Implementation**
- Build React/Vue components following design system
- Implement state management with selected approach
- Create API client with interceptors (auth, error handling)
- Implement form validation with Zod/Yup
- Build responsive layouts (mobile-first, Tailwind CSS)
- Add accessibility attributes (ARIA, semantic HTML)
- Implement loading states and error boundaries
- Create Storybook stories for components
- Write component tests with Testing Library (70%+ coverage)

**6. Database Implementation & Optimization**
- Create migration scripts (Alembic/TypeORM/Prisma)
- Implement stored procedures if complex logic required
- Optimize queries identified during backend development
- Create composite indexes for common query patterns
- Add database-level constraints (foreign keys, unique, check)
- Implement soft deletes with timestamp tracking
- Set up database connection pooling
- Plan read replicas for read-heavy workloads

### Success Criteria
- ✅ All API endpoints implemented and documented
- ✅ Frontend components rendering correctly with proper styling
- ✅ Database migrations run without errors
- ✅ Unit tests passing with 70-80% coverage
- ✅ API client handling errors and loading states
- ✅ No SQL injection vulnerabilities (parameterized queries)
- ✅ Authentication working end-to-end

### Tools
- Use **Task** with stack-specific agents (python-pro/fastapi-pro, frontend-developer)
- Use **Task** with `database-optimizer` for query optimization
- Use **Bash** to run tests: `pytest backend/tests`, `npm test`
- Use **Read** to review generated code for quality

**Reference**: See `technology-stack-guide.md` for stack-specific guidance

---

## Phase 3: Integration & Testing

### Objective
Validate integration across all layers with comprehensive testing and security hardening

### Key Activities

**7. API Contract Testing**
- Implement Pact tests for provider (backend) contracts
- Implement Pact tests for consumer (frontend) contracts
- Validate API responses match OpenAPI specification (Dredd)
- Test authentication flows (login, logout, refresh, password reset)
- Validate error responses (400, 401, 403, 404, 500)
- Test CORS configuration for allowed origins
- Create load test scenarios (k6/Locust)
- Verify API rate limiting functionality

**8. End-to-End Testing**
- Create Playwright/Cypress tests for critical user journeys
- Test cross-browser compatibility (Chrome, Firefox, Safari)
- Validate mobile responsiveness on different devices
- Test error scenarios (network failures, API errors)
- Verify feature flag integration works correctly
- Test analytics tracking events
- Create visual regression tests (Percy/Chromatic)
- Measure and validate performance metrics (Core Web Vitals)

**9. Security Audit & Hardening**
- Review authentication implementation (secure token storage, HTTPS only)
- Check authorization logic (role-based, resource-based)
- Validate input sanitization prevents XSS
- Test for SQL injection vulnerabilities
- Review CSRF protection implementation
- Audit secrets management (no hardcoded credentials)
- Check for sensitive data exposure in logs
- Validate security headers (CSP, X-Frame-Options, HSTS)
- Run OWASP ZAP or Burp Suite penetration testing
- Document security findings and remediation

### Success Criteria
- ✅ All contract tests passing (100% API contract coverage)
- ✅ E2E tests covering critical paths (happy path + error scenarios)
- ✅ Security audit passed with no critical/high vulnerabilities
- ✅ Load tests meeting SLO targets (p95 < 500ms, p99 < 1s)
- ✅ Cross-browser compatibility validated
- ✅ Mobile responsive design verified on 3+ devices
- ✅ Zero SQL injection vulnerabilities
- ✅ Zero XSS vulnerabilities

### Tools
- Use **Task** with `test-automator` for contract and E2E tests
- Use **Task** with `security-auditor` for security review
- Use **Bash** to run tests: `npm run test:e2e`, `pytest tests/contract`
- Use **Write** to document security findings

**Reference**: See `testing-strategies.md` for comprehensive testing patterns

---

## Phase 4: Deployment & Operations

### Objective
Deploy to production with progressive rollout, comprehensive monitoring, and operational excellence

### Key Activities

**10. Infrastructure & CI/CD Setup**
- Create Dockerfiles for backend and frontend
- Build Kubernetes manifests (Deployments, Services, Ingress)
- Implement GitHub Actions/GitLab CI pipeline
- Add automated testing gates (unit, integration, E2E)
- Configure feature flags (LaunchDarkly/Unleash)
- Set up environment-specific configurations
- Implement blue-green deployment strategy
- Document rollback procedures
- Configure auto-scaling rules (HPA)
- Set up SSL/TLS certificates (Let's Encrypt/cert-manager)

**11. Observability & Monitoring**
- Set up distributed tracing (OpenTelemetry/Jaeger)
- Configure application metrics (Prometheus/DataDog)
- Implement centralized logging (ELK/Splunk/CloudWatch)
- Create dashboards for key metrics (Grafana)
- Define SLIs/SLOs (availability, latency, error rate)
- Set up alerting rules (PagerDuty/OpsGenie)
- Create runbooks for common incidents
- Implement synthetic monitoring (uptime checks)
- Set up Real User Monitoring (RUM)
- Configure log retention and archival

**12. Performance Optimization**
- Analyze and optimize database queries (EXPLAIN plans)
- Implement multi-tier caching (Redis L1, CDN L2)
- Optimize frontend bundle size (code splitting, tree shaking)
- Set up lazy loading for images and components
- Configure CDN for static assets (CloudFront/Cloudflare)
- Tune backend service performance (connection pooling, async I/O)
- Optimize Docker images (multi-stage builds, layer caching)
- Measure before/after performance metrics
- Validate Core Web Vitals improvements
- Load test optimizations with k6

### Success Criteria
- ✅ CI/CD pipeline with automated quality gates operational
- ✅ Feature flags configured for progressive rollout (5% → 25% → 50% → 100%)
- ✅ Observability stack capturing 100% of traffic
- ✅ SLIs/SLOs defined and dashboards created
- ✅ Performance improvements validated (>30% improvement)
- ✅ Rollback procedures tested and documented
- ✅ Auto-scaling responding to load (test with load testing)
- ✅ SSL/TLS certificates valid and auto-renewing
- ✅ Logs retained for compliance requirements (90 days)
- ✅ Zero-downtime deployment capability verified

### Tools
- Use **Task** with `deployment-engineer` for infrastructure setup
- Use **Task** with `performance-engineer` for optimization
- Use **Bash** to deploy: `kubectl apply -f k8s/`, `docker-compose up`
- Use **Edit** to update deployment configs

**Reference**: See `deployment-patterns.md` for CI/CD and Kubernetes patterns

---

## Decision Trees

### Technology Stack Selection

```
Choose stack based on requirements:

1. Do you need Server-Side Rendering (SSR)?
   YES → Next.js + Django/FastAPI
   NO → Continue

2. Do you prefer type safety across entire stack?
   YES → TypeScript frontend + NestJS backend
   NO → Continue

3. Do you need flexible schema (rapid changes)?
   YES → MongoDB (document-oriented)
   NO → PostgreSQL (relational, ACID)

4. Do you need real-time features (WebSockets)?
   YES → NestJS (Socket.io built-in) or FastAPI (WebSockets)
   NO → Any stack works

5. Team expertise?
   - Python team → React/FastAPI/PostgreSQL
   - JavaScript team → Next.js/NestJS/PostgreSQL
   - Mixed → React/Django/PostgreSQL (most mature ecosystem)
```

### Deployment Target Selection

```
Choose deployment platform:

1. Is this a monolith or microservices?
   Monolith → Consider Heroku, Railway, Render
   Microservices → Continue

2. Do you need full Kubernetes control?
   YES → EKS/AKS/GKE (managed K8s)
   NO → Continue

3. Do you prefer serverless?
   YES → Vercel (frontend) + AWS Lambda/Cloud Functions (backend)
   NO → Continue

4. Budget constraints?
   Low → DigitalOcean, Linode
   Medium → AWS/GCP with managed services
   High → AWS/GCP with full DevOps team

5. Compliance requirements (HIPAA, SOC2)?
   YES → AWS/GCP/Azure with compliance certifications
   NO → Any platform works
```

### API Style Selection

```
Choose API architecture:

1. Do clients need flexible queries (fetch only needed fields)?
   YES → GraphQL
   NO → Continue

2. Do you need simple CRUD operations?
   YES → REST (simplest, widest adoption)
   NO → Continue

3. Do you need real-time subscriptions?
   YES → GraphQL (subscriptions) or WebSockets
   NO → REST

4. Team experience?
   - GraphQL experience → GraphQL
   - No GraphQL experience → REST (lower learning curve)

5. Mobile clients with limited bandwidth?
   YES → GraphQL (clients request only needed fields)
   NO → REST (simpler, better caching)
```

---

## Configuration Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--quick` | flag | false | Architecture & design phase only |
| `--stack` | "React/FastAPI/PostgreSQL", "Next.js/Django/MongoDB", etc. | auto-detect | Technology stack selection |
| `--deployment-target` | aws, gcp, azure, digitalocean, heroku | aws | Cloud platform |
| `--api-style` | rest, graphql | rest | API architecture style |
| `--feature-flags` | flag | false (standard), true (deep) | Enable feature flag integration |
| `--testing-depth` | essential, comprehensive | essential (standard), comprehensive (deep) | Test coverage level |
| `--compliance` | none, gdpr, hipaa, soc2 | none | Compliance requirements |
| `--deep` | flag | false | Enable deep mode (full workflow) |

---

## Examples

### Example 1: Simple REST API Feature

```bash
/full-stack-feature "User profile management with avatar upload" --stack="React/FastAPI/PostgreSQL" --api-style=rest
```

**Generated Components**:
- Database: `users` table with profile fields, `user_avatars` table for S3 references
- Backend: `/api/users/{id}` (GET, PATCH), `/api/users/{id}/avatar` (POST), S3 integration
- Frontend: ProfilePage component, AvatarUpload component, profile editing form
- Tests: Unit tests for profile service, E2E tests for profile update flow
- CI/CD: GitHub Actions pipeline with staging deployment

### Example 2: Real-Time Feature with GraphQL

```bash
/full-stack-feature "Real-time chat with typing indicators" --deep --stack="Next.js/NestJS/PostgreSQL" --api-style=graphql --deployment-target=aws
```

**Generated Components**:
- Database: `messages`, `conversations`, `participants` tables with indexes
- Backend: GraphQL schema with queries/mutations, WebSocket subscriptions, Redis pub/sub
- Frontend: ChatWindow component, message list with virtualization, typing indicator
- Tests: Contract tests for GraphQL schema, E2E tests for message sending, load tests for concurrent users
- Infrastructure: ECS Fargate for NestJS, RDS PostgreSQL, ElastiCache Redis, Application Load Balancer
- Observability: CloudWatch logs, X-Ray traces, custom metrics for message latency

### Example 3: E-Commerce Feature with Compliance

```bash
/full-stack-feature "Payment processing with Stripe integration" --deep --stack="React/Django/PostgreSQL" --compliance=pci-dss --deployment-target=aws
```

**Generated Components**:
- Database: `payments`, `payment_methods` tables with encrypted fields
- Backend: Stripe webhook handlers, payment intent creation, idempotency keys
- Frontend: CheckoutForm with Stripe Elements, payment confirmation page
- Security: PCI-DSS compliant (no card data stored), encrypted payment tokens, audit logging
- Tests: Payment flow E2E tests, webhook handler tests, fraud prevention validation
- Infrastructure: Private subnets for backend, KMS encryption, CloudTrail audit logs

---

## What This Command Will Do

✅ **Design Architecture**: Database schema, API contracts, component hierarchy
✅ **Implement Backend**: RESTful/GraphQL endpoints with proper validation and auth
✅ **Build Frontend**: React/Next.js components with state management and API integration
✅ **Create Migrations**: Database schema changes with zero-downtime strategy
✅ **Write Tests**: Contract tests, E2E tests, security tests, performance tests
✅ **Set Up CI/CD**: Automated pipelines with quality gates and progressive rollout
✅ **Configure Monitoring**: Distributed tracing, metrics, logs, dashboards, alerts
✅ **Optimize Performance**: Caching, query optimization, bundle optimization
✅ **Ensure Security**: OWASP Top 10 validation, penetration testing, secrets management
✅ **Document Everything**: API docs, architecture decisions, runbooks

---

## What This Command Won't Do

❌ **Design UI/UX**: Focus is on implementation, not visual design (provide Figma designs)
❌ **Write Business Logic**: You define requirements, we implement technical solution
❌ **Manage Infrastructure Costs**: Cost optimization is separate concern
❌ **Perform Data Migration**: Existing data migration requires separate planning
❌ **Handle Third-Party Integrations**: External API integrations require separate implementation
❌ **Create Marketing Content**: Technical implementation only, not copywriting

---

## Agent Orchestration

### Phase 1 (Architecture) - Sequential Execution
```typescript
// Step 1: Database design
Task({
  subagent_type: 'observability-monitoring:database-optimizer',
  prompt: `Design database schema for: ${featureDescription}
    - Entity relationship model
    - Table schemas with proper types
    - Indexing strategy for query patterns
    - Migration scripts (zero-downtime)`
});

// Step 2: Backend architecture (depends on database schema)
Task({
  subagent_type: 'backend-development:backend-architect',
  prompt: `Design backend service architecture using database schema:
    - API contracts (OpenAPI/GraphQL)
    - Authentication & authorization flows
    - Caching strategy
    - Resilience patterns`
});

// Step 3: Frontend architecture (depends on API contracts)
Task({
  subagent_type: 'frontend-mobile-development:frontend-developer',
  prompt: `Design frontend architecture using API contracts:
    - Component hierarchy
    - State management approach
    - Data fetching patterns
    - Accessibility requirements`
});
```

### Phase 2 (Implementation) - Parallel Execution
```typescript
// All three can run in parallel
await Promise.all([
  Task({ subagent_type: 'python-development:fastapi-pro', ... }),
  Task({ subagent_type: 'frontend-mobile-development:frontend-developer', ... }),
  Task({ subagent_type: 'observability-monitoring:database-optimizer', ... })
]);
```

### Phase 3 (Testing) - Sequential with Validation
```typescript
// Contract tests must pass before E2E tests
await Task({ subagent_type: 'unit-testing:test-automator', ... }); // Contract tests
if (contractTestsPassed) {
  await Task({ subagent_type: 'unit-testing:test-automator', ... }); // E2E tests
  await Task({ subagent_type: 'full-stack-orchestration:security-auditor', ... });
}
```

---

## Troubleshooting

### Issue: Backend and Frontend API Contract Mismatch

**Cause**: OpenAPI specification not kept in sync with implementation

**Solution**:
1. Run contract tests: `pytest tests/contract/`
2. Review Pact diff: `pact-broker diff --consumer=WebApp --provider=API`
3. Update OpenAPI spec: `Edit openapi.yaml` to match actual implementation
4. Regenerate frontend API client: `openapi-generator generate -i openapi.yaml`
5. Re-run contract tests to verify

### Issue: E2E Tests Flaky or Failing

**Cause**: Race conditions, timing issues, or test data pollution

**Solution**:
1. Add explicit waits: `await page.waitForSelector('[data-testid="user-name"]')`
2. Use test isolation: Create/cleanup test data in beforeEach/afterEach
3. Increase timeouts for slow operations: `{ timeout: 10000 }`
4. Check test logs: `DEBUG=pw:api npm run test:e2e`
5. Review `testing-strategies.md` for E2E best practices

### Issue: Performance Degradation After Deployment

**Cause**: Missing indexes, inefficient queries, or caching issues

**Solution**:
1. Check slow query logs: `SELECT * FROM pg_stat_statements ORDER BY total_time DESC`
2. Run EXPLAIN on slow queries: `EXPLAIN ANALYZE SELECT ...`
3. Add missing indexes: `CREATE INDEX idx_users_email ON users(email)`
4. Verify cache hit rate: `redis-cli INFO stats | grep keyspace_hits`
5. Use **Task** with `performance-engineer` for comprehensive optimization

### Issue: CI/CD Pipeline Failing

**Cause**: Test failures, linting errors, or build issues

**Solution**:
1. Check GitHub Actions logs for specific error
2. Reproduce locally: `docker-compose -f docker-compose.test.yml up`
3. Fix linting: `npm run lint --fix`, `black backend/`
4. Update dependencies: `npm audit fix`, `pip install -U -r requirements.txt`
5. Verify environment variables are set in CI settings

---

## Best Practices

1. **API-First Development**: Design OpenAPI/GraphQL schema before implementation
2. **Contract-Driven**: Validate contracts between frontend and backend with Pact
3. **Test Pyramid**: 70% unit, 20% integration, 10% E2E (optimize for speed + reliability)
4. **Zero-Downtime Deployments**: Use blue-green or canary with automated rollback
5. **Observability from Day 1**: Add logging, metrics, traces before production
6. **Security by Default**: Input validation, parameterized queries, principle of least privilege
7. **Performance Budgets**: Define SLOs (p95 < 500ms) and validate in CI
8. **Feature Flags**: Progressive rollout (5% → 25% → 50% → 100%) with instant rollback
9. **Infrastructure as Code**: All infrastructure in version control (Terraform/Pulumi)
10. **Documentation**: Keep ADRs, runbooks, and API docs up-to-date

---

## Integration with Other Commands

- **Before Starting**: Use `/code-migrate` if upgrading framework versions
- **During Development**: Use `/component-scaffold` for complex UI components
- **After Implementation**: Use `/double-check` for comprehensive validation
- **Before Deployment**: Use `/run-all-tests` to ensure all tests passing
- **For Documentation**: Use `/update-docs` to sync with project documentation

---

## Success Metrics

- **Time to Production**: 1-3 days for deep mode (vs. 2-4 weeks manual)
- **Test Coverage**: 80-90% backend, 70-80% frontend
- **Security Posture**: Zero critical/high vulnerabilities
- **Performance**: p95 < 500ms, p99 < 1s API latency
- **Availability**: 99.9% uptime SLO
- **Deployment Frequency**: Daily deploys with <5min rollback
- **MTTR**: <30 minutes mean time to recovery
- **Developer Experience**: Comprehensive docs, clear workflows, automated testing

---

For implementation details, see:
- `architecture-patterns-library.md` - Database schemas, API contracts, component patterns
- `testing-strategies.md` - Contract, E2E, security, and performance testing
- `deployment-patterns.md` - CI/CD pipelines, Kubernetes, monitoring, rollback
- `technology-stack-guide.md` - Stack-specific implementation guidance
