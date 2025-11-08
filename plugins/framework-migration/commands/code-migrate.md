---
version: 1.0.3
description: Orchestrate systematic code migration between frameworks and technology stacks with test-first discipline
argument-hint: <source-path> [--target <framework>] [--strategy <pattern>] [--mode quick|standard|deep]
category: framework-migration
purpose: Safe, incremental code migration with zero breaking changes and comprehensive testing
execution_time:
  quick: "30-60 minutes - Assessment and strategy planning only"
  standard: "2-6 hours - Complete single component migration"
  deep: "1-3 days - Enterprise migration with comprehensive validation"
color: blue
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, WebFetch
external_docs:
  - migration-patterns-library.md
  - testing-strategies.md
  - framework-specific-guides.md
  - rollback-procedures.md
agents:
  primary:
    - framework-migration:legacy-modernizer
    - framework-migration:architect-review
  conditional:
    - agent: unit-testing:test-automator
      trigger: pattern "test|coverage"
    - agent: comprehensive-review:security-auditor
      trigger: pattern "security|vulnerability"
    - agent: full-stack-orchestration:performance-engineer
      trigger: pattern "performance|optimization"
  orchestrated: true
tags: [migration, modernization, refactoring, framework-upgrade, code-transformation, zero-downtime]
---

# Code Migration Orchestrator

**Systematic framework and technology stack migration with test-first discipline, backward compatibility guarantees, and zero breaking changes**

## Execution Modes

Parse `$ARGUMENTS` to determine mode (default: standard):

### Quick Mode (30-60 min)
**Scope**: Assessment and strategy planning only
- Technology stack analysis
- Complexity assessment
- Migration strategy selection
- Risk identification
- Time/resource estimation

**Output**: Migration plan with strategy recommendation

**Use When**: Planning phase, seeking approval, estimating resources

### Standard Mode (2-6 hours) - RECOMMENDED
**Scope**: Complete single component migration
- All Quick mode deliverables
- Test coverage establishment
- Code transformation
- Integration validation
- Documentation updates

**Output**: Migrated component with tests and documentation

**Use When**: Migrating individual components, features, or modules

### Deep Mode (1-3 days)
**Scope**: Enterprise migration with comprehensive validation
- All Standard mode deliverables
- Performance benchmarking
- Security hardening
- Load testing
- Migration playbook creation
- Team training materials

**Output**: Production-ready migration with comprehensive validation

**Use When**: Critical systems, enterprise migrations, compliance requirements

---

## Configuration Options

- `--target <framework>`: Target technology (e.g., `--target react`, `--target python3`, `--target angular`)
- `--strategy <pattern>`: Migration pattern (`big-bang`, `strangler-fig`, `branch-by-abstraction`)
- `--mode <quick|standard|deep>`: Execution depth (default: standard)
- `--test-first`: Enforce characterization tests before any code changes
- `--parallel-run`: Enable side-by-side validation
- `--skip-security`: Skip security audit (not recommended for production)

---

## Your Task

**Source**: $ARGUMENTS
**Mode**: [Auto-detected or specified]

---

## Phase 1: Migration Assessment & Strategy Selection

**Objective**: Analyze source/target stacks and select optimal migration approach

### Step 1A: Technology Stack Analysis

**Use Task tool** with `subagent_type="framework-migration:architect-review"`:

```
Analyze codebase at $ARGUMENTS for migration assessment.

Identify:
- Current framework/language versions
- Architectural patterns in use
- External dependencies and versions
- Integration points and APIs
- Custom implementations vs framework features
- Code complexity metrics (cyclomatic complexity, duplication)

Generate technology inventory with migration complexity scores (1-10).

Reference: docs/framework-migration/framework-specific-guides.md
```

**Expected Output**: Technology inventory, complexity assessment, dependency tree

### Step 1B: Risk & Complexity Assessment

**Use Task tool** with `subagent_type="framework-migration:legacy-modernizer"`:

```
Assess migration risks for: $ARGUMENTS

Evaluate:
- Breaking changes between versions
- API compatibility issues
- Performance implications
- Security vulnerabilities in current/target versions
- Team skill gaps
- Test coverage gaps

Generate risk matrix with mitigation strategies.

Reference: docs/framework-migration/migration-patterns-library.md (breaking changes catalog)
```

**Expected Output**: Risk matrix (High/Medium/Low), mitigation strategies

### Step 1C: Strategy Selection

**Decision Tree** - Select based on assessment:

```
Migration Complexity > 7/10?
├─ Yes → Strangler Fig Pattern (incremental, parallel systems)
└─ No → Migration Timeline < 2 weeks?
    ├─ Yes → Big Bang (full cutover)
    └─ No → Branch by Abstraction (feature-by-feature)
```

**Migration Patterns**:

**1. Big Bang** (Low complexity, < 2 weeks):
- ✅ Fast completion
- ✅ No dual system maintenance
- ❌ High deployment risk
- ❌ Difficult rollback

**2. Strangler Fig** (High complexity, > 1 month):
- ✅ Zero downtime
- ✅ Instant rollback
- ✅ Incremental risk
- ❌ Dual system complexity

**3. Branch by Abstraction** (Medium complexity, 2-8 weeks):
- ✅ Feature-by-feature migration
- ✅ Continuous deployment
- ❌ Abstraction layer overhead

**📚 See**: [Strangler Fig Playbook](../docs/framework-migration/strangler-fig-playbook.md) for detailed implementation

**Success Criteria for Phase 1**:
- ✅ Technology stack fully documented
- ✅ Risk assessment complete with mitigation plans
- ✅ Migration strategy selected and justified
- ✅ Timeline and resource estimate provided

**🚨 Quick Mode Exits Here** - Deliver assessment and strategy recommendation

---

## Phase 2: Test Coverage Establishment

**Objective**: Create safety net before any code changes

### Step 2A: Characterization Tests

**Use Task tool** with `subagent_type="unit-testing:test-automator"`:

```
Create characterization tests for: $ARGUMENTS

Generate:
- Golden master tests for complex workflows
- Snapshot tests for UI components
- Contract tests for API integrations
- Behavior tests for business logic

Capture current behavior (even if buggy) to detect any changes.

Reference: docs/framework-migration/testing-strategies.md (characterization tests)
```

**Characterization Test Pattern**:
```javascript
// Captures current behavior before migration
describe('Legacy Payment Processor', () => {
  it('should match current behavior', () => {
    const result = legacyProcessor.process(testOrder);
    expect(result).toMatchSnapshot();  // Captures exact current output
  });
});
```

**📚 See**: [Testing Strategies - Characterization Tests](../docs/framework-migration/testing-strategies.md#characterization-tests)

### Step 2B: Integration Contract Tests

**Create tests for integration boundaries**:
- API contracts (request/response schemas)
- Database contracts (query interfaces)
- External service contracts (third-party APIs)
- Event contracts (message formats)

**Contract Test Example**:
```javascript
// Ensures API contract remains stable
describe('Payment API Contract', () => {
  it('should maintain response schema', async () => {
    const response = await paymentAPI.process(validRequest);
    expect(response).toMatchSchema({
      status: expect.stringMatching(/^(success|pending|failed)$/),
      transactionId: expect.any(String),
      amount: expect.any(Number)
    });
  });
});
```

### Step 2C: Performance Baseline Capture

**Establish performance benchmarks**:
```bash
# Capture baseline metrics
npm run benchmark > baseline-performance.json

# Key metrics:
# - Response time (p50, p95, p99)
# - Throughput (requests/second)
# - Memory usage
# - CPU utilization
```

**Success Criteria for Phase 2**:
- ✅ Test coverage > 80% for migration scope
- ✅ All integration points have contract tests
- ✅ Performance baseline documented
- ✅ All tests passing on current implementation

---

## Phase 3: Incremental Migration Implementation

**Objective**: Transform code using selected migration strategy

### Step 3A: Setup Migration Infrastructure

**For Strangler Fig**:
```
Use Task tool with subagent_type="cicd-automation:deployment-engineer"

Setup routing layer and feature flags for: $ARGUMENTS

Implement:
- API gateway or load balancer routing
- Feature flag system for gradual rollout
- Monitoring and metrics collection
- Rollback procedures

Reference: docs/framework-migration/strangler-fig-playbook.md
```

**For Big Bang / Branch by Abstraction**:
- Create migration branch
- Setup parallel build pipeline
- Configure dual testing (legacy + migrated)

### Step 3B: Automated Code Transformation

**Use codemods when available**:

**React Class → Hooks**:
```bash
npx react-codemod class-to-hooks src/components/
```

**Python 2 → 3**:
```bash
2to3 -w src/
```

**Custom Transformations**:
```
Use Task tool with subagent_type="framework-migration:legacy-modernizer"

Apply code transformations for: $ARGUMENTS

Transform:
- API calls to new framework syntax
- State management patterns
- Lifecycle methods to hooks/alternatives
- Import statements and module structure

Preserve:
- Business logic (no functional changes)
- Error handling patterns
- Existing tests (migrate separately)

Reference: docs/framework-migration/migration-patterns-library.md (transformation patterns)
```

**📚 See**: [Migration Patterns Library](../docs/framework-migration/migration-patterns-library.md) for transformation examples

### Step 3C: Manual Migration (Complex Cases)

**For components requiring manual rewrite**:

1. **Read original implementation** (understand business logic)
2. **Extract business rules** (separate from framework code)
3. **Implement in target framework** (following new patterns)
4. **Preserve identical behavior** (no functional changes)
5. **Update tests** (if syntax changes needed)

**Example - React Class to Functional**:
```javascript
// Before (Class Component)
class UserProfile extends React.Component {
  state = { loading: true, user: null };

  componentDidMount() {
    fetchUser(this.props.userId)
      .then(user => this.setState({ user, loading: false }));
  }

  render() {
    const { loading, user } = this.state;
    if (loading) return <Spinner />;
    return <div>{user.name}</div>;
  }
}

// After (Functional + Hooks)
function UserProfile({ userId }) {
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState(null);

  useEffect(() => {
    fetchUser(userId)
      .then(user => {
        setUser(user);
        setLoading(false);
      });
  }, [userId]);

  if (loading) return <Spinner />;
  return <div>{user.name}</div>;
}
```

**Success Criteria for Phase 3**:
- ✅ All target code compiles/builds successfully
- ✅ Characterization tests still pass (behavior unchanged)
- ✅ No console errors or warnings
- ✅ Code follows target framework best practices

---

## Phase 4: Integration & Validation

**Objective**: Ensure migrated code integrates correctly and performs well

### Step 4A: Integration Testing

**Run full test suite**:
```bash
# Run all tests
npm test

# Run integration tests
npm run test:integration

# Run E2E tests
npm run test:e2e
```

**Validate**:
- All unit tests pass
- Integration tests pass
- E2E workflows complete successfully
- No new test failures introduced

### Step 4B: Performance Comparison

**Compare against baseline**:
```bash
# Run benchmarks on migrated code
npm run benchmark > migrated-performance.json

# Compare
diff baseline-performance.json migrated-performance.json
```

**Acceptable Performance**:
- Response time within 110% of baseline
- Memory usage within 120% of baseline
- No critical performance regressions

**If performance regressed**:
```
Use Task tool with subagent_type="full-stack-orchestration:performance-engineer"

Analyze performance regression in: $ARGUMENTS

Profile and optimize:
- Identify bottlenecks
- Database query optimization
- Caching opportunities
- Bundle size optimization
```

### Step 4C: Security Audit

**Use Task tool** with `subagent_type="comprehensive-review:security-auditor"`:

```
Security audit migrated code at: $ARGUMENTS

Review:
- Input validation and sanitization
- Authentication/authorization patterns
- Dependency vulnerabilities
- OWASP Top 10 compliance
- Security headers and CORS

Flag any security regressions or new vulnerabilities.
```

**Success Criteria for Phase 4**:
- ✅ All tests passing (100% pass rate)
- ✅ Performance within acceptable range (<110% baseline)
- ✅ No security vulnerabilities introduced
- ✅ Integration points validated

---

## Phase 5: Deployment & Monitoring

**Objective**: Deploy migrated code safely with monitoring and rollback capability

### Step 5A: Progressive Rollout Strategy

**For Strangler Fig**:
1. Route 5% traffic to migrated implementation
2. Monitor error rates, latency, business metrics
3. If stable for 24 hours, increase to 25%
4. Continue: 25% → 50% → 75% → 100%

**For Big Bang**:
1. Deploy to staging environment
2. Run smoke tests
3. If passing, deploy to production
4. Monitor closely for first 2 hours

**Rollback Triggers** (immediate rollback if):
- Error rate > 5% (vs baseline < 1%)
- p95 latency > 2x baseline
- Any data corruption
- Critical functionality broken

**📚 See**: [Rollback Procedures](../docs/framework-migration/rollback-procedures.md)

### Step 5B: Monitoring Dashboard

**Key metrics to monitor**:
- Error rate (overall and by endpoint)
- Response time (p50, p95, p99)
- Throughput (requests/sec)
- Resource utilization (CPU, memory)
- Business metrics (conversion rate, revenue)

**Alert thresholds**:
- Error rate > 1%: Warning
- Error rate > 5%: Critical (rollback)
- p95 latency > 2x: Warning
- p95 latency > 3x: Critical (rollback)

### Step 5C: Documentation Updates

**Update**:
- README with new technology stack
- Architecture documentation
- API documentation (if endpoints changed)
- Deployment procedures
- Runbooks for new technology

**Success Criteria for Phase 5**:
- ✅ Deployed to production successfully
- ✅ Monitoring shows stable metrics
- ✅ No incidents or rollbacks needed
- ✅ Documentation updated

**🚨 Standard Mode Complete** - Migration deployed and validated

---

## Phase 6: Post-Migration Optimization (Deep Mode Only)

**Objective**: Optimize migrated code and create playbook for future migrations

### Step 6A: Performance Optimization

**Beyond functional equivalence**:
- Leverage target framework optimizations
- Implement caching strategies
- Bundle size optimization
- Code splitting

**Use Task tool** with `subagent_type="full-stack-orchestration:performance-engineer"`:

```
Optimize migrated code at: $ARGUMENTS

Apply target framework best practices:
- React: useMemo, useCallback, code splitting
- Vue: Computed properties, watchers, async components
- Python: List comprehensions, generators, async/await
- Node.js: Worker threads, streams, clustering

Target: 20-30% performance improvement over baseline.
```

### Step 6B: Migration Playbook Creation

**Document for future migrations**:
```markdown
# [Component Name] Migration Playbook

## Overview
- Source: [Technology/Version]
- Target: [Technology/Version]
- Duration: [Actual time taken]
- Complexity: [Actual vs estimated]

## Lessons Learned
1. [What worked well]
2. [What was challenging]
3. [Unexpected issues]

## Reusable Patterns
- [Pattern 1 with code example]
- [Pattern 2 with code example]

## Gotchas to Avoid
1. [Pitfall 1]
2. [Pitfall 2]

## Time Estimation Formula
[Refined estimate based on actual data]
```

### Step 6C: Team Training

**Knowledge transfer**:
- Conduct workshop on migrated technology
- Pair programming sessions
- Code review guidelines for new framework
- Best practices documentation

**Success Criteria for Phase 6**:
- ✅ Performance optimized (20-30% improvement)
- ✅ Migration playbook documented
- ✅ Team trained on new technology
- ✅ Ready for future migrations

**🎯 Deep Mode Complete** - Enterprise migration with comprehensive optimization

---

## Safety Guarantees

**This command will**:
- ✅ Create characterization tests before any changes
- ✅ Maintain backward compatibility at integration points
- ✅ Provide instant rollback capability
- ✅ Validate performance against baseline
- ✅ Run security audit on migrated code
- ✅ Generate comprehensive documentation

**This command will NEVER**:
- ❌ Modify code without test coverage
- ❌ Deploy without rollback plan
- ❌ Introduce breaking changes to APIs
- ❌ Skip security validation
- ❌ Delete code without backup
- ❌ Ignore performance regressions

---

## Usage Examples

### Basic Framework Migration
```bash
# Migrate React component to hooks
/code-migrate src/components/Dashboard.jsx --target react-hooks

# Migrate Python 2 to Python 3
/code-migrate src/legacy/ --target python3

# Migrate Angular 12 to 15
/code-migrate src/app --target angular15
```

### With Strategy Selection
```bash
# Use Strangler Fig for large migration
/code-migrate src/ --target react18 --strategy strangler-fig

# Quick assessment only
/code-migrate src/ --target vue3 --mode quick

# Full enterprise migration
/code-migrate src/ --target nextjs --mode deep --test-first
```

---

**Execute systematic code migration with test-first discipline, performance validation, and zero breaking changes**
