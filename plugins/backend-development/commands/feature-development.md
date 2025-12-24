---
version: "1.0.6"
category: "backend-development"
command: "/feature-development"

execution-modes:
  quick:
    description: "Rapid MVP development for simple features"
    time: "1-2 days"
    phases: "Skip discovery, minimal testing, direct deployment"
    steps: "Steps 4, 5, 7, 10 only"
    use-case: "Hot fixes, simple CRUD operations, urgent patches"

  standard:
    description: "Full 4-phase workflow for production features"
    time: "3-14 days"
    phases: "All 12 steps with comprehensive validation"
    use-case: "Standard feature development (default)"

  enterprise:
    description: "Extended workflow with compliance and governance"
    time: "2-4 weeks"
    phases: "All steps + compliance review, legal approval, multi-region deployment"
    use-case: "Enterprise features, regulated industries, multi-tenant"

documentation:
  detailed-guides: "../docs/backend-development/methodology-guides.md"
  phase-templates: "../docs/backend-development/phase-templates.md"
  agent-patterns: "../docs/backend-development/agent-orchestration.md"
  deployment: "../docs/backend-development/deployment-strategies.md"
  best-practices: "../docs/backend-development/best-practices.md"
  metrics: "../docs/backend-development/success-metrics.md"
---

Orchestrate end-to-end feature development from requirements to production deployment:

[Extended thinking: This workflow orchestrates specialized agents through comprehensive feature development phases - from discovery and planning through implementation, testing, and deployment. Each phase builds on previous outputs, ensuring coherent feature delivery. The workflow supports multiple development methodologies (traditional, TDD/BDD, DDD), feature complexity levels, and modern deployment strategies including feature flags, gradual rollouts, and observability-first development. Agents receive detailed context from previous phases to maintain consistency and quality throughout the development lifecycle.]

## Agent Reference

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

## Configuration Options

### Development Methodology
- **traditional**: Sequential development with testing after implementation
- **tdd**: Test-Driven Development with red-green-refactor cycles
- **bdd**: Behavior-Driven Development with scenario-based testing
- **ddd**: Domain-Driven Design with bounded contexts and aggregates

### Feature Complexity
- **simple**: Single service, minimal integration (1-2 days)
- **medium**: Multiple services, moderate integration (3-5 days)
- **complex**: Cross-domain, extensive integration (1-2 weeks)
- **epic**: Major architectural changes, multiple teams (2+ weeks)

### Deployment Strategy
- **direct**: Immediate rollout to all users
- **canary**: Gradual rollout starting with 5% of traffic
- **feature-flag**: Controlled activation via feature toggles
- **blue-green**: Zero-downtime deployment with instant rollback
- **a-b-test**: Split traffic for experimentation and metrics

## Phase 1: Discovery & Requirements Planning

1. **Business Analysis & Requirements**
   - **Action**: Manually analyze feature requirements for: $ARGUMENTS
   - **Deliverable**: Requirements document with user stories, success metrics, risk assessment [→ Guide](../docs/backend-development/phase-templates.md#business-analysis)
   - **Success**: Requirements completeness score >90%, stakeholder sign-off, clear scope boundaries

2. **Technical Architecture Design**
   - **Agent**: architect-review | **Context**: Business requirements from step 1
   - **Objective**: Design technical architecture with service boundaries, API contracts, data models, integration points
   - **Deliverable**: Technical design document with architecture diagrams, API specs [→ Guide](../docs/backend-development/phase-templates.md#architecture-design)

3. **Feasibility & Risk Assessment**
   - **Agent**: security-auditor | **Context**: Technical design from step 2
   - **Objective**: Assess security implications, compliance needs, data privacy concerns, potential vulnerabilities
   - **Deliverable**: Security assessment with risk matrix, compliance checklist, mitigation strategies [→ Guide](../docs/backend-development/phase-templates.md#risk-assessment)

## Phase 2: Implementation & Development

4. **Backend Services Implementation**
   - **Agent**: backend-architect | **Context**: Technical design from step 2
   - **Objective**: Implement backend services with RESTful/GraphQL APIs, business logic, resilience patterns, caching, feature flags
   - **Deliverable**: Backend services meeting API contracts [→ Implementation Guide](../docs/backend-development/phase-templates.md#backend-implementation)

5. **Frontend Implementation**
   - **Agent**: frontend-developer | **Context**: Backend APIs from step 4
   - **Objective**: Build responsive UI, state management, error handling, loading states, analytics tracking, feature flag integration
   - **Deliverable**: Frontend components with API integration [→ Implementation Guide](../docs/backend-development/phase-templates.md#frontend-implementation)

6. **Data Pipeline & Integration**
   - **Action**: Manually build data pipelines for: $ARGUMENTS
   - **Objective**: Design ETL/ELT processes, data validation, analytics events, data quality monitoring
   - **Deliverable**: Data pipelines, analytics events, quality checks [→ Guide](../docs/backend-development/phase-templates.md#data-pipeline)

## Phase 3: Testing & Quality Assurance

7. **Automated Test Suite**
   - **Agent**: test-automator | **Context**: Backend (step 4) and frontend (step 5) implementation
   - **Objective**: Create unit, integration, E2E, and performance tests with minimum 80% code coverage
   - **Deliverable**: Comprehensive test suite [→ Testing Guide](../docs/backend-development/phase-templates.md#automated-testing)

8. **Security Validation**
   - **Agent**: security-auditor | **Context**: Backend and frontend implementation from steps 4-5
   - **Objective**: Run OWASP checks, penetration testing, dependency scanning, compliance validation
   - **Deliverable**: Security test results, vulnerability report, remediation actions [→ Security Guide](../docs/backend-development/phase-templates.md#security-validation)

9. **Performance Optimization**
   - **Agent**: performance-engineer | **Context**: Backend (step 4) and frontend (step 5) implementation
   - **Objective**: Profile code, optimize queries, implement caching, reduce bundle sizes, improve load times
   - **Deliverable**: Performance improvements, optimization report, metrics [→ Performance Guide](../docs/backend-development/phase-templates.md#performance-optimization)

## Phase 4: Deployment & Monitoring

10. **Deployment Strategy & Pipeline**
    - **Agent**: deployment-engineer | **Context**: Test suites from step 7, infrastructure requirements
    - **Objective**: Create CI/CD pipeline with automated tests, feature flags, blue-green deployment, rollback procedures
    - **Deliverable**: CI/CD pipeline, deployment config, rollback plan [→ Deployment Guide](../docs/backend-development/deployment-strategies.md)

11. **Observability & Monitoring**
    - **Agent**: observability-engineer | **Context**: Feature implementation, success metrics
    - **Objective**: Implement distributed tracing, custom metrics, error tracking, alerting, dashboards, SLOs/SLIs
    - **Deliverable**: Monitoring dashboards, alerts, SLO definitions [→ Observability Guide](../docs/backend-development/phase-templates.md#observability)

12. **Documentation & Knowledge Transfer**
    - **Agent**: docs-architect | **Context**: All previous phases' outputs
    - **Objective**: Generate API documentation, user guides, deployment guides, troubleshooting runbooks
    - **Deliverable**: Comprehensive documentation package [→ Documentation Guide](../docs/backend-development/phase-templates.md#documentation)

## Execution Parameters

### Required Parameters
- **--feature**: Feature name and description
- **--methodology**: Development approach (traditional|tdd|bdd|ddd)
- **--complexity**: Feature complexity level (simple|medium|complex|epic)

### Optional Parameters
- **--mode**: Execution mode (quick|standard|enterprise) - default: standard
- **--deployment-strategy**: Deployment approach (direct|canary|feature-flag|blue-green|a-b-test)
- **--test-coverage-min**: Minimum test coverage threshold (default: 80%)
- **--performance-budget**: Performance requirements (e.g., <200ms response time)
- **--rollout-percentage**: Initial rollout percentage for gradual deployment (default: 5%)
- **--feature-flag-service**: Feature flag provider (launchdarkly|split|unleash|custom)
- **--analytics-platform**: Analytics integration (segment|amplitude|mixpanel|custom)
- **--monitoring-stack**: Observability tools (datadog|newrelic|grafana|custom)

## Success Criteria

### Phase-Specific Outcomes
- **Phase 1**: Requirements completeness >90%, risk matrix created, stakeholder sign-off
- **Phase 2**: API contract coverage 100%, feature flag configured, frontend responsive on all breakpoints
- **Phase 3**: Test coverage ≥80%, zero critical vulnerabilities, p95 latency <200ms, performance budget met
- **Phase 4**: Successful deployment, monitoring dashboards live, documentation published, rollback tested

### Overall Success
- All acceptance criteria from business requirements met
- Security scan shows no critical vulnerabilities
- Performance meets defined budgets and SLOs
- Feature flags configured for controlled rollout
- Monitoring and alerting fully operational
- Documentation complete and approved
- Successful deployment to production with rollback capability
- Product analytics tracking feature usage
- A/B test metrics configured (if applicable)

## Rollback Strategy

If issues arise during or after deployment:
1. **Immediate feature flag disable** (< 1 minute)
2. **Blue-green traffic switch** (< 5 minutes)
3. **Full deployment rollback via CI/CD** (< 15 minutes)
4. **Database migration rollback** if needed (coordinate with data team)
5. **Incident post-mortem** and fixes before re-deployment

[→ Detailed Rollback Procedures](../docs/backend-development/deployment-strategies.md#rollback-procedures)

Feature description: $ARGUMENTS
