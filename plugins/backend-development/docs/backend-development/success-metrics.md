# Success Metrics & Criteria

Quantifiable success criteria for feature development across all phases.

## Table of Contents
- [Phase-Specific Metrics](#phase-specific-metrics)
- [Technical Metrics](#technical-metrics)
- [Business Metrics](#business-metrics)
- [Quality Metrics](#quality-metrics)
- [Operational Metrics](#operational-metrics)
- [Measurement Tools](#measurement-tools)

---

## Phase-Specific Metrics

### Phase 1: Discovery & Requirements Planning

#### Business Analysis Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Requirements Completeness** | ≥ 90% | (Answered questions / Total identified questions) × 100 |
| **Stakeholder Alignment** | 100% sign-off | All key stakeholders approved requirements document |
| **User Story Coverage** | 100% | All acceptance criteria defined with measurable outcomes |
| **Risk Identification** | ≥ 80% | (Identified risks / Total risks discovered later) × 100 |
| **Scope Clarity** | ≥ 95% | (Clear requirements / Total requirements) × 100 |

**Validation Questions**:
- ✅ Are all user stories in "As a... I want... So that..." format?
- ✅ Do all user stories have acceptance criteria?
- ✅ Are success metrics defined and measurable?
- ✅ Have all stakeholders reviewed and approved?
- ✅ Are out-of-scope items explicitly documented?

#### Architecture Design Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **API Contract Coverage** | 100% | All endpoints documented in OpenAPI/GraphQL schema |
| **Data Model Completeness** | 100% | All entities, relationships, and constraints defined |
| **Scalability Assessment** | Documented | Load projections and scaling strategy defined |
| **Technology Stack Approval** | 100% | All tech choices reviewed by architecture team |
| **Design Review Completion** | ≥ 2 reviewers | At least 2 senior engineers reviewed |

**Deliverables Checklist**:
- [ ] Architecture diagram (C4 model recommended)
- [ ] API contract (OpenAPI 3.0 or GraphQL schema)
- [ ] Data model (ERD with constraints)
- [ ] Technology stack rationale
- [ ] Scalability and performance strategy
- [ ] Security architecture
- [ ] Integration points documented

#### Risk Assessment Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Security Review Completeness** | 100% | All OWASP Top 10 categories assessed |
| **Compliance Coverage** | 100% | All applicable regulations identified (GDPR, HIPAA, etc.) |
| **Risk Mitigation** | ≥ 90% | (Risks with mitigation strategy / Total risks) × 100 |
| **Critical Issues Resolution** | 100% | All critical/high risks have mitigation before implementation |

---

### Phase 2: Implementation & Development

#### Backend Implementation Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **API Coverage** | 100% | All endpoints from spec implemented |
| **Unit Test Coverage** | ≥ 80% | Code coverage tools (Jest, Pytest) |
| **API Response Time (p95)** | < 200ms | Load testing results |
| **Error Handling** | 100% | All external calls have try/catch with proper error responses |
| **Input Validation** | 100% | All endpoints validate inputs (Joi/Zod/class-validator) |
| **Feature Flag Integration** | 100% | All new code paths behind feature flags |
| **Documentation** | 100% | All endpoints documented in Swagger/Postman |

**Code Quality Checklist**:
- [ ] No hardcoded secrets or credentials
- [ ] All database queries use parameterized queries (SQL injection prevention)
- [ ] Proper logging with correlation IDs
- [ ] Circuit breakers for external service calls
- [ ] Connection pooling configured
- [ ] Health check endpoint implemented

#### Frontend Implementation Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Component Test Coverage** | ≥ 75% | React Testing Library/Vue Test Utils coverage |
| **Accessibility Score (WCAG 2.1 AA)** | 100% | axe DevTools / Lighthouse accessibility scan |
| **Performance (Lighthouse)** | ≥ 90 | Lighthouse performance score |
| **Bundle Size** | < 200KB (gzipped) | webpack-bundle-analyzer |
| **Load Time (First Contentful Paint)** | < 1.5s | Lighthouse / WebPageTest |
| **Mobile Responsiveness** | 100% | All breakpoints tested (320px, 768px, 1024px, 1440px) |

**Quality Checklist**:
- [ ] Semantic HTML elements used
- [ ] ARIA labels for interactive elements
- [ ] Keyboard navigation supported
- [ ] Color contrast ratio ≥ 4.5:1 (WCAG AA)
- [ ] Images have alt text
- [ ] Forms have proper labels and error messages
- [ ] Loading states implemented (skeleton screens)
- [ ] Error boundaries implemented

---

### Phase 3: Testing & Quality Assurance

#### Automated Testing Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Unit Test Coverage** | ≥ 80% | Jest/Pytest coverage report |
| **Integration Test Coverage** | ≥ 60% | Critical API paths tested |
| **E2E Test Coverage** | ≥ 3 critical paths | Cypress/Playwright tests for key user journeys |
| **Test Pass Rate** | 100% | All tests passing in CI/CD |
| **Test Execution Time** | < 5 min | CI/CD pipeline test duration |
| **Flaky Test Rate** | < 1% | (Flaky tests / Total tests) × 100 |

**Test Pyramid Adherence**:
- 70% Unit Tests (fast, isolated)
- 20% Integration Tests (API, database)
- 10% E2E Tests (critical user flows)

#### Security Validation Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Critical Vulnerabilities** | 0 | SAST/DAST scan results |
| **High Vulnerabilities** | 0 | Security scan results |
| **Medium Vulnerabilities** | ≤ 5 | Security scan with remediation plan |
| **Dependency Vulnerabilities** | 0 critical/high | Snyk/Dependabot scan |
| **OWASP Top 10 Coverage** | 100% | Manual penetration testing checklist |
| **Security Test Coverage** | ≥ 90% | (Security tests passed / Total security tests) × 100 |

**Security Checklist**:
- [ ] SQL injection tests passed
- [ ] XSS prevention verified
- [ ] CSRF protection enabled
- [ ] Authentication tests passed
- [ ] Authorization tests passed
- [ ] Rate limiting verified
- [ ] Input validation tests passed
- [ ] Secrets not in codebase (verified by git-secrets scan)

#### Performance Optimization Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **API Response Time (p95)** | < 200ms | Load testing (k6, JMeter, Artillery) |
| **API Response Time (p99)** | < 500ms | Load testing results |
| **Database Query Time (p95)** | < 50ms | APM tools (DataDog, New Relic) |
| **Frontend Load Time (LCP)** | < 2.5s | Lighthouse / Core Web Vitals |
| **Frontend Interaction (FID)** | < 100ms | Core Web Vitals |
| **Layout Shift (CLS)** | < 0.1 | Core Web Vitals |
| **Throughput** | ≥ 1000 req/s | Load testing at expected peak load |
| **Error Rate Under Load** | < 0.1% | Load testing error metrics |

**Performance Testing Scenarios**:
1. Baseline load (normal traffic)
2. Peak load (2x baseline)
3. Stress test (gradual increase until breaking point)
4. Spike test (sudden 5x traffic increase)
5. Soak test (sustained load for 2+ hours)

---

### Phase 4: Deployment & Monitoring

#### Deployment Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Deployment Success Rate** | ≥ 95% | (Successful deploys / Total deploys) × 100 |
| **Deployment Duration** | < 15 min | CI/CD pipeline timing |
| **Rollback Time** | < 5 min | Time from issue detection to rollback completion |
| **Zero-Downtime Deployment** | 100% | No 5xx errors during deployment |
| **Smoke Test Pass Rate** | 100% | Post-deployment smoke tests |
| **Feature Flag Functionality** | 100% | Feature can be toggled without redeployment |

**Deployment Validation**:
- [ ] Health check returns 200 OK
- [ ] Smoke tests pass
- [ ] No increase in error rate
- [ ] No increase in latency
- [ ] Feature flag controls work
- [ ] Logs are being written
- [ ] Metrics are being collected

#### Observability Success Criteria

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Metrics Collection** | 100% | All RED metrics (Rate, Errors, Duration) instrumented |
| **Log Coverage** | ≥ 95% | Critical paths have logging |
| **Trace Coverage** | ≥ 90% | Critical paths have distributed tracing |
| **Alert Coverage** | 100% | All critical metrics have alerts |
| **Dashboard Availability** | 100% | Real-time dashboard created |
| **MTTR (Mean Time To Recover)** | < 30 min | Average time to resolve incidents |
| **MTTD (Mean Time To Detect)** | < 5 min | Average time to detect incidents |

**Observability Checklist**:
- [ ] Request rate metrics collected
- [ ] Error rate metrics collected
- [ ] Latency metrics collected (p50, p95, p99)
- [ ] Business metrics tracked (conversion, usage)
- [ ] Structured logging implemented (JSON format)
- [ ] Correlation IDs in all logs
- [ ] Distributed tracing configured
- [ ] Alerts configured for:
  - [ ] Error rate > 1%
  - [ ] p95 latency > 200ms
  - [ ] Throughput drops > 50%
  - [ ] Service unavailable

---

## Technical Metrics

### Code Quality

| Metric | Target | Tool |
|--------|--------|------|
| **Cyclomatic Complexity** | < 10 per function | SonarQube, ESLint complexity rule |
| **Code Duplication** | < 3% | SonarQube, PMD |
| **Technical Debt Ratio** | < 5% | SonarQube |
| **Maintainability Index** | ≥ 70 | SonarQube, Code Climate |
| **Code Review Coverage** | 100% | All PRs reviewed by ≥ 1 engineer |

### Test Coverage

| Metric | Target | Tool |
|--------|--------|------|
| **Line Coverage** | ≥ 80% | Jest, Pytest, Istanbul |
| **Branch Coverage** | ≥ 75% | Coverage tools |
| **Function Coverage** | ≥ 85% | Coverage tools |
| **Mutation Test Score** | ≥ 70% | Stryker, PIT |

### Performance

| Metric | Target | Tool |
|--------|--------|------|
| **API p95 Latency** | < 200ms | APM (DataDog, New Relic) |
| **API p99 Latency** | < 500ms | APM tools |
| **Database Query p95** | < 50ms | Database monitoring |
| **Cache Hit Rate** | ≥ 80% | Redis metrics |
| **Error Rate** | < 0.5% | Error tracking (Sentry, Rollbar) |
| **Apdex Score** | ≥ 0.9 | APM tools |

---

## Business Metrics

### Feature Adoption

| Metric | Target | Measurement Period |
|--------|--------|-------------------|
| **Adoption Rate** | ≥ 50% of active users | 30 days post-launch |
| **Time to First Use** | < 7 days | Days from user signup to feature use |
| **Daily Active Users (DAU)** | ≥ 10% of total users | Daily |
| **Weekly Active Users (WAU)** | ≥ 30% of total users | Weekly |
| **Monthly Active Users (MAU)** | ≥ 60% of total users | Monthly |
| **Feature Retention (Day 1)** | ≥ 40% | Users who return next day |
| **Feature Retention (Day 7)** | ≥ 25% | Users who return after 7 days |
| **Feature Retention (Day 30)** | ≥ 15% | Users who return after 30 days |

### Engagement

| Metric | Target | Formula |
|--------|--------|---------|
| **Session Duration** | ≥ 5 min | Average time spent using feature per session |
| **Actions Per Session** | ≥ 3 | Average interactions per session |
| **Feature Stickiness** | ≥ 0.3 | DAU / MAU |
| **Conversion Rate** | Baseline + 10% | (Conversions with feature / Total sessions) × 100 |

### Business Impact

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Revenue Impact** | +X% | Incremental revenue attributed to feature |
| **Cost Reduction** | -X% | Operational cost savings |
| **Customer Satisfaction (CSAT)** | ≥ 4.5/5 | User surveys |
| **Net Promoter Score (NPS)** | ≥ 40 | User surveys |
| **Support Ticket Reduction** | -20% | Tickets related to replaced workflow |

---

## Quality Metrics

### Reliability

| Metric | Target | SLA |
|--------|--------|-----|
| **Uptime** | ≥ 99.9% | 99.9% (< 43.2 min downtime/month) |
| **Error Budget** | 0.1% | Allowed error rate |
| **Incident Frequency** | < 2 per month | Critical incidents |
| **Mean Time Between Failures (MTBF)** | > 30 days | Average time between incidents |

### Operational Excellence

| Metric | Target | Description |
|--------|--------|-------------|
| **Deployment Frequency** | ≥ 10 per week | Continuous deployment maturity |
| **Lead Time for Changes** | < 1 day | Time from commit to production |
| **Change Failure Rate** | < 15% | (Failed deploys / Total deploys) × 100 |
| **Mean Time to Restore (MTTR)** | < 1 hour | Time to recover from failure |

---

## Measurement Tools

### Development Phase

| Category | Tools | Purpose |
|----------|-------|---------|
| **Code Coverage** | Jest, Pytest, Istanbul, Coverage.py | Measure test coverage |
| **Code Quality** | SonarQube, ESLint, Pylint, Code Climate | Static analysis, code smells |
| **Security Scanning** | Snyk, Dependabot, SonarQube, Bandit | Vulnerability detection |
| **Performance Testing** | k6, JMeter, Artillery, Locust | Load and stress testing |
| **API Documentation** | Swagger UI, Postman, Redoc | API spec validation |

### Production Monitoring

| Category | Tools | Purpose |
|----------|-------|---------|
| **APM** | DataDog, New Relic, AppDynamics | Application performance monitoring |
| **Logging** | ELK Stack, Splunk, CloudWatch Logs | Centralized logging |
| **Tracing** | Jaeger, Zipkin, DataDog APM | Distributed tracing |
| **Error Tracking** | Sentry, Rollbar, Bugsnag | Error monitoring and alerting |
| **Metrics** | Prometheus, Grafana, DataDog | Custom metrics and dashboards |
| **Real User Monitoring** | Google Analytics, Amplitude, Mixpanel | User behavior and engagement |
| **Uptime Monitoring** | Pingdom, UptimeRobot, StatusCake | Service availability |

### Business Analytics

| Category | Tools | Purpose |
|----------|-------|---------|
| **Product Analytics** | Amplitude, Mixpanel, Heap | Feature usage and funnels |
| **A/B Testing** | Optimizely, VWO, LaunchDarkly | Experimentation |
| **Session Replay** | FullStory, LogRocket, Hotjar | User session recordings |
| **Heatmaps** | Hotjar, Crazy Egg, Mouseflow | User interaction patterns |
| **Surveys** | Typeform, SurveyMonkey, Qualtrics | Customer feedback (CSAT, NPS) |

---

## Success Criteria Dashboard

### Real-Time Feature Health Dashboard

```yaml
Feature Health: [Feature Name]
Status: 🟢 Healthy / 🟡 Warning / 🔴 Critical

Technical Metrics:
  - Error Rate: 0.15% (Target: < 0.5%) ✅
  - p95 Latency: 185ms (Target: < 200ms) ✅
  - Uptime: 99.95% (Target: ≥ 99.9%) ✅
  - Test Coverage: 87% (Target: ≥ 80%) ✅

Business Metrics:
  - Adoption Rate: 52% (Target: ≥ 50%) ✅
  - Daily Active Users: 12,450 (Target: ≥ 10,000) ✅
  - Conversion Rate: +12% (Target: +10%) ✅
  - CSAT Score: 4.6/5 (Target: ≥ 4.5) ✅

Quality Metrics:
  - Critical Bugs: 0 (Target: 0) ✅
  - Security Vulnerabilities: 2 medium (Target: 0 critical/high) ✅
  - Deployment Success: 96% (Target: ≥ 95%) ✅

Alerts (Last 24h):
  - 0 critical
  - 1 warning (p95 latency briefly exceeded 200ms at 14:32)
```

---

## Continuous Improvement

### Metrics Review Cadence

- **Daily**: Error rate, latency, uptime, deployment success
- **Weekly**: Test coverage, technical debt, adoption rate, engagement
- **Monthly**: Business impact, CSAT/NPS, security posture, cost efficiency
- **Quarterly**: Architecture review, tech stack modernization, team velocity

### Success Criteria Evolution

As the feature matures, adjust targets:

**Phase 1 (0-30 days)**: Focus on stability and adoption
- Error rate < 1%
- Adoption ≥ 20%

**Phase 2 (30-90 days)**: Optimize performance and engagement
- Error rate < 0.5%
- Adoption ≥ 50%
- p95 latency < 200ms

**Phase 3 (90+ days)**: Business impact and efficiency
- Error rate < 0.1%
- Adoption ≥ 70%
- Conversion rate improvement ≥ 15%
- Cost per transaction reduced by 20%

---

## References

- [Feature Development Command](../../commands/feature-development.md)
- [Phase Templates](./phase-templates.md)
- [Best Practices](./best-practices.md)
- [Deployment Strategies](./deployment-strategies.md)
