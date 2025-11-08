# Risk Assessment Framework

Comprehensive framework for evaluating change risk in pull requests and code reviews, with quantitative scoring algorithms, risk mitigation strategies, and decision matrices.

## Risk Scoring Model

### Multi-Factor Risk Calculation

**Formula**:
```
Total Risk Score = (Size × 0.25) + (Complexity × 0.30) + (Coverage × 0.20) +
                   (Dependencies × 0.15) + (Security × 0.10)
```

Each factor scored 0-10, total range: 0-10

### Risk Factors Breakdown

#### 1. Size Factor (Weight: 25%)

**Calculation**:
```
Size Score = min(10, (lines_changed / 100))
```

**Thresholds**:
- **0-2**: Trivial (< 100 lines)
- **2-4**: Small (100-200 lines)
- **4-6**: Medium (200-400 lines)
- **6-8**: Large (400-600 lines)
- **8-10**: Very Large (> 600 lines)

**Rationale**: Larger changes have more opportunities for bugs and are harder to review thoroughly.

#### 2. Complexity Factor (Weight: 30%)

**Calculation**:
```
Complexity Score = (cyclomatic_complexity_avg / 10) +
                   (nesting_depth_max / 3) +
                   (function_length_avg / 50)
```

**Metrics**:
- **Cyclomatic Complexity**: Number of independent paths through code
- **Nesting Depth**: Maximum levels of nested control structures
- **Function Length**: Average lines per function

**Thresholds**:
- **0-2**: Simple (linear logic, few branches)
- **2-4**: Moderate (some conditionals and loops)
- **4-6**: Complex (multiple conditionals, nested logic)
- **6-8**: Very Complex (intricate control flow)
- **8-10**: Extremely Complex (refactoring recommended)

#### 3. Test Coverage Factor (Weight: 20%)

**Calculation**:
```
Coverage Score = 10 - (coverage_percentage / 10)
```
(Inverse relationship: lower coverage = higher risk)

**Thresholds**:
- **0-2**: Excellent (> 80% coverage)
- **2-4**: Good (60-80% coverage)
- **4-6**: Fair (40-60% coverage)
- **6-8**: Poor (20-40% coverage)
- **8-10**: Critical (< 20% coverage)

**Adjusted for Change Type**:
- New features: Target 80%+ coverage
- Bug fixes: Target 90%+ coverage (including regression tests)
- Refactoring: Must maintain existing coverage

#### 4. Dependencies Factor (Weight: 15%)

**Calculation**:
```
Dependencies Score = (external_deps_added × 2) +
                     (internal_modules_changed × 0.5) +
                     (breaking_changes × 5)
```

**Risk Indicators**:
- **New External Dependencies**: Each adds supply chain risk
- **Modified Internal Modules**: Ripple effect across codebase
- **Breaking Changes**: Requires coordination and migration
- **Circular Dependencies**: Critical architectural issue

**Thresholds**:
- **0-2**: Isolated (no new dependencies)
- **2-4**: Limited (1-2 internal modules)
- **4-6**: Moderate (3-5 modules or 1 external dep)
- **6-8**: Extensive (> 5 modules or 2+ external deps)
- **8-10**: Critical (breaking changes or circular deps)

#### 5. Security Factor (Weight: 10%)

**Calculation**:
```
Security Score = (auth_changes × 3) +
                 (data_handling_changes × 2) +
                 (input_validation_changes × 2) +
                 (crypto_changes × 3)
```

**High-Risk Areas**:
- Authentication/Authorization logic
- User input handling and validation
- Database queries (SQL injection risk)
- Cryptographic operations
- File system access
- Network communication
- Third-party API integration

**Thresholds**:
- **0-2**: No security-sensitive changes
- **2-4**: Minor security-adjacent changes
- **4-6**: Moderate security changes (needs review)
- **6-8**: High security impact (security review required)
- **8-10**: Critical security changes (security team + audit)

---

## Risk Level Determination

### Risk Matrix

| Total Score | Risk Level | Review Requirements | Deployment Strategy |
|-------------|-----------|-------------------|-------------------|
| 0.0 - 2.5 | 🟢 **Low** | Standard review, 1 approver | Deploy normally |
| 2.5 - 5.0 | 🟡 **Medium** | Thorough review, 2 approvers | Gradual rollout recommended |
| 5.0 - 7.5 | 🟠 **High** | Deep review, 2+ approvers, architecture review | Feature flag + staged rollout |
| 7.5 - 10.0 | 🔴 **Critical** | Multiple domain experts, security review, load testing | Canary deployment + monitoring |

### Special Risk Modifiers

**Increase Risk Level (+1 level)**:
- Production hotfix
- Customer-facing feature
- Payment/billing logic
- Data migration involved
- Legacy code modification
- No automated tests

**Decrease Risk Level (-1 level)**:
- Documentation-only changes
- Automated dependency updates (with passing tests)
- Revert of problematic change
- Well-tested refactoring
- Configuration with rollback plan

---

## Risk Mitigation Strategies

### Low Risk (0.0 - 2.5)

**Characteristics**: Small, well-tested, isolated changes

**Mitigation**:
- ✅ Standard code review process
- ✅ Automated testing (unit + integration)
- ✅ CI/CD pipeline validation
- ✅ Deploy during business hours

**Example**: Bug fix in utility function with comprehensive tests

---

### Medium Risk (2.5 - 5.0)

**Characteristics**: Moderate size/complexity, some integration points

**Mitigation**:
- ✅ Two-person review (code quality + domain expert)
- ✅ Extended test suite (include edge cases)
- ✅ Smoke testing in staging environment
- ✅ Monitor key metrics post-deployment
- ✅ Gradual rollout (10% → 50% → 100%)

**Example**: New API endpoint with database queries and business logic

---

### High Risk (5.0 - 7.5)

**Characteristics**: Large changes, complex logic, multiple integrations

**Mitigation**:
- ✅ Multi-perspective review (code + architecture + security)
- ✅ Load testing and performance benchmarks
- ✅ Feature flag for controlled rollout
- ✅ Comprehensive monitoring and alerting
- ✅ Staged deployment (dev → staging → canary → production)
- ✅ Rollback plan documented and tested
- ✅ Post-deployment validation checklist

**Example**: Payment processing integration with third-party API

---

### Critical Risk (7.5 - 10.0)

**Characteristics**: System-critical changes, security-sensitive, architectural

**Mitigation**:
- ✅ Expert review panel (architecture + security + domain experts)
- ✅ Security audit and penetration testing
- ✅ Chaos engineering tests (failure scenarios)
- ✅ Comprehensive load/stress testing
- ✅ Canary deployment with real-time monitoring
- ✅ Automated rollback triggers
- ✅ 24/7 on-call coverage during rollout
- ✅ Customer communication plan
- ✅ Post-mortem after deployment

**Example**: Database migration affecting core user data, authentication system rewrite

---

## Risk Decision Framework

### Decision Tree

```
START
  │
  ├─ Is this a security fix? ──YES──→ Risk Level: HIGH (minimum)
  │                            │
  │                            └─ Critical vulnerability? ──YES──→ Risk Level: CRITICAL
  │
  ├─ Is this production data migration? ──YES──→ Risk Level: HIGH (minimum)
  │
  ├─ Calculate Total Risk Score
  │   │
  │   ├─ Score < 2.5? ──YES──→ Risk Level: LOW
  │   ├─ Score 2.5-5.0? ──YES──→ Risk Level: MEDIUM
  │   ├─ Score 5.0-7.5? ──YES──→ Risk Level: HIGH
  │   └─ Score > 7.5? ──YES──→ Risk Level: CRITICAL
  │
  └─ Apply Risk Modifiers
      │
      └─ Final Risk Level → Select Mitigation Strategy
```

### When to Split a PR

**Split if ANY are true**:
- Total risk score > 7.0 (can split into lower-risk PRs)
- Size factor > 8 (> 600 lines changed)
- Multiple unrelated features/fixes
- Can separate infrastructure from feature code
- Can separate refactoring from functional changes

**Benefits of Splitting**:
- Lower individual PR risk scores
- Faster review cycles
- Easier to isolate issues
- Incremental deployment safety
- Better git history

---

## Risk Communication Template

### For PR Description

```markdown
## Risk Assessment

**Overall Risk Level**: 🟡 Medium

**Risk Breakdown**:
| Factor | Score | Notes |
|--------|-------|-------|
| Size | 3.5/10 | 250 lines changed |
| Complexity | 4.0/10 | Moderate conditional logic |
| Test Coverage | 2.0/10 | 85% coverage (good) |
| Dependencies | 3.0/10 | Modifies 3 internal modules |
| Security | 0.0/10 | No security-sensitive changes |
| **Total** | **3.1/10** | **Medium Risk** |

**Mitigation Plan**:
- ✅ Two reviewers assigned (backend + QA)
- ✅ Integration tests added covering edge cases
- ✅ Staging deployment planned before production
- ✅ Monitoring dashboard for key metrics
- ✅ Gradual rollout: 10% → 50% → 100% over 48 hours

**Rollback Plan**:
Feature flag `new_search_algorithm` can be toggled off without redeployment.
```

---

## Risk Monitoring Post-Deployment

### Metrics to Track

**Performance Metrics** (High/Critical Risk):
- Response time (p50, p95, p99)
- Error rate
- Throughput (requests per second)
- Database query performance

**Business Metrics** (Medium/High/Critical):
- Conversion rate
- User engagement
- Revenue impact
- Customer satisfaction

**System Metrics** (All Levels):
- CPU/Memory usage
- Disk I/O
- Network latency
- Cache hit rate

### Alert Thresholds

| Risk Level | Alert Threshold | Response Time | Rollback Trigger |
|-----------|----------------|---------------|-----------------|
| 🟢 Low | > 5% error increase | 1 hour | > 20% errors |
| 🟡 Medium | > 3% error increase | 30 minutes | > 10% errors |
| 🟠 High | > 2% error increase | 15 minutes | > 5% errors |
| 🔴 Critical | > 1% error increase | Immediate | > 2% errors |

---

## Risk Assessment Tools

### Automated Risk Calculation

**GitHub Actions Example**:
```yaml
name: Calculate PR Risk Score

on: [pull_request]

jobs:
  risk-assessment:
    runs-on: ubuntu-latest
    steps:
      - name: Calculate Risk
        run: |
          SIZE_SCORE=$(echo "${{ github.event.pull_request.additions }} + ${{ github.event.pull_request.deletions }}" | bc)
          COMPLEXITY_SCORE=$(npm run complexity-check)
          COVERAGE_SCORE=$(npm run coverage-report)

          # Calculate total risk
          TOTAL_RISK=$(calculate_risk $SIZE_SCORE $COMPLEXITY_SCORE $COVERAGE_SCORE)

          # Post comment with risk assessment
          gh pr comment ${{ github.event.pull_request.number }} \
            --body "🎯 Risk Score: $TOTAL_RISK"
```

### Manual Risk Checklist

**Use this checklist during PR creation**:

- [ ] Calculated size score (lines changed)
- [ ] Assessed complexity (cyclomatic complexity, nesting)
- [ ] Verified test coverage (% coverage, edge cases)
- [ ] Identified dependencies (external, internal, breaking)
- [ ] Evaluated security impact (auth, data, validation)
- [ ] Determined total risk score
- [ ] Selected appropriate mitigation strategy
- [ ] Documented rollback plan
- [ ] Assigned correct reviewers based on risk level
- [ ] Planned deployment strategy

---

## Case Studies

### Case Study 1: Low Risk - Documentation Update

**Change**: Update API documentation with new examples

**Risk Calculation**:
- Size: 0.5/10 (50 lines)
- Complexity: 0/10 (no code)
- Coverage: 0/10 (N/A for docs)
- Dependencies: 0/10 (no dependencies)
- Security: 0/10 (no security impact)
- **Total: 0.1/10** → 🟢 **Low Risk**

**Mitigation**: Single reviewer, deploy immediately

---

### Case Study 2: Medium Risk - New Feature

**Change**: Add user profile image upload functionality

**Risk Calculation**:
- Size: 4/10 (350 lines)
- Complexity: 5/10 (file handling, validation, storage)
- Coverage: 3/10 (75% coverage)
- Dependencies: 4/10 (S3 integration, image processing library)
- Security: 6/10 (file upload, validation, storage access)
- **Total: 4.5/10** → 🟡 **Medium Risk**

**Mitigation**:
- Two reviewers (backend + security)
- Extended testing (file types, sizes, malicious files)
- Feature flag deployment
- Monitor upload success rate and storage usage

---

### Case Study 3: High Risk - Payment Integration

**Change**: Integrate Stripe payment processing

**Risk Calculation**:
- Size: 6/10 (500 lines)
- Complexity: 7/10 (webhooks, error handling, idempotency)
- Coverage: 4/10 (70% coverage, external API mocking)
- Dependencies: 7/10 (Stripe SDK, webhook handling, database schema changes)
- Security: 9/10 (PCI compliance, API keys, customer data)
- **Total: 6.8/10** → 🟠 **High Risk**

**Mitigation**:
- Expert review panel (backend + security + payments specialist)
- Comprehensive testing (happy path, failures, webhooks)
- Security audit (PCI compliance check)
- Staging environment testing with Stripe test mode
- Feature flag with gradual rollout
- 24/7 monitoring of payment success rate
- Rollback plan with data reconciliation

---

### Case Study 4: Critical Risk - Database Migration

**Change**: Migrate user authentication from custom to OAuth2

**Risk Calculation**:
- Size: 9/10 (1200 lines)
- Complexity: 9/10 (authentication flows, session management, backward compatibility)
- Coverage: 6/10 (65% coverage, complex integration testing)
- Dependencies: 9/10 (OAuth provider, database schema changes, breaking changes)
- Security: 10/10 (authentication system, user sessions, data migration)
- **Total: 8.7/10** → 🔴 **Critical Risk**

**Mitigation**:
- Multi-expert review (architecture + security + auth specialist + QA)
- Security penetration testing
- Load testing (concurrent logins, session handling)
- Comprehensive migration script with rollback
- Canary deployment (internal users → beta users → 10% → 100%)
- Real-time monitoring (login success rate, session errors)
- Automated rollback triggers
- Customer communication plan
- Post-deployment validation for 1 week
- Post-mortem documentation

---

Use this framework to systematically assess and mitigate risk in all code changes, ensuring safe and reliable deployments.
