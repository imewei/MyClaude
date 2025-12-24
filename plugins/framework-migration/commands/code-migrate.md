---
version: "1.0.6"
description: Orchestrate systematic code migration between frameworks with test-first discipline
argument-hint: <source-path> [--target <framework>] [--strategy <pattern>] [--mode quick|standard|deep]
category: framework-migration
purpose: Safe, incremental code migration with zero breaking changes
execution_time:
  quick: "30-60 minutes"
  standard: "2-6 hours"
  deep: "1-3 days"
color: blue
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, WebFetch
external_docs:
  - migration-patterns-library.md
  - testing-strategies.md
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
  orchestrated: true
---

# Code Migration Orchestrator

Systematic framework migration with test-first discipline and zero breaking changes.

## Source

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| `--mode=quick` | 30-60 min | Assessment and strategy only |
| standard (default) | 2-6 hours | Complete single component migration |
| `--mode=deep` | 1-3 days | Enterprise migration with optimization |

**Options:** `--target react-hooks|python3|angular15`, `--strategy big-bang|strangler-fig|branch-by-abstraction`

---

## Phase 1: Assessment & Strategy

**Agents:** architect-review, legacy-modernizer

### Technology Analysis
- Current framework/language versions
- Architectural patterns in use
- External dependencies
- Integration points
- Complexity metrics

### Risk Assessment
- Breaking changes between versions
- API compatibility issues
- Performance implications
- Security vulnerabilities
- Team skill gaps

### Strategy Selection

```
Complexity > 7/10? → Strangler Fig (incremental)
Timeline < 2 weeks? → Big Bang (full cutover)
Otherwise → Branch by Abstraction (feature-by-feature)
```

| Strategy | Pros | Cons |
|----------|------|------|
| **Big Bang** | Fast, no dual system | High risk, hard rollback |
| **Strangler Fig** | Zero downtime, instant rollback | Dual system complexity |
| **Branch by Abstraction** | Feature-by-feature, continuous deploy | Abstraction overhead |

🚨 **Quick Mode exits here** - deliver assessment and strategy

---

## Phase 2: Test Coverage Establishment

**Agents:** test-automator

### Characterization Tests
- Golden master tests for workflows
- Snapshot tests for UI
- Contract tests for APIs
- Behavior tests for business logic

Capture current behavior (even if buggy) to detect changes.

### Contract Tests
- API request/response schemas
- Database query interfaces
- External service contracts
- Event/message formats

### Performance Baseline
```bash
npm run benchmark > baseline-performance.json
```
Capture: p50/p95/p99 latency, throughput, memory, CPU

**Success:** >80% coverage, all integration points tested, baseline documented

---

## Phase 3: Migration Implementation

### Codemods (when available)

| Migration | Codemod |
|-----------|---------|
| React Class → Hooks | `npx react-codemod class-to-hooks` |
| Python 2 → 3 | `2to3 -w src/` |
| Angular upgrade | `ng update` |
| Vue 2 → 3 | `npx @vue/compat-migration` |

### Manual Migration (complex cases)

1. Read original implementation
2. Extract business rules (separate from framework)
3. Implement in target framework
4. Preserve identical behavior
5. Update tests if syntax changes

### Strangler Fig Setup
- API gateway routing
- Feature flags for gradual rollout
- Monitoring for both systems
- Rollback procedures

**Success:** Code compiles, characterization tests pass, no console errors

---

## Phase 4: Validation

### Integration Testing
```bash
npm test                    # Unit
npm run test:integration    # Integration
npm run test:e2e           # E2E
```

### Performance Comparison
```bash
npm run benchmark > migrated-performance.json
diff baseline-performance.json migrated-performance.json
```

**Acceptable:** Response time <110%, memory <120% of baseline

### Security Audit
- Input validation
- Auth patterns
- Dependency vulnerabilities
- OWASP compliance

**Success:** All tests passing, performance acceptable, no security regressions

🚨 **Standard Mode exits here** - migration deployed and validated

---

## Phase 5: Deployment

### Progressive Rollout

| Stage | Traffic | Duration | Action |
|-------|---------|----------|--------|
| 1 | 5% | 24h | Monitor errors, latency |
| 2 | 25% | 24h | Check business metrics |
| 3 | 50% | 24h | Validate at scale |
| 4 | 100% | - | Full deployment |

### Rollback Triggers
- Error rate >5% (vs baseline <1%)
- p95 latency >2x baseline
- Any data corruption
- Critical functionality broken

### Documentation Updates
- README with new tech stack
- Architecture docs
- API documentation
- Deployment procedures

---

## Phase 6: Post-Migration (Deep Mode)

### Optimization
- Leverage target framework features
- Implement caching strategies
- Bundle optimization
- Code splitting

Target: 20-30% performance improvement over baseline

### Migration Playbook
Document lessons learned, reusable patterns, gotchas for future migrations

### Team Training
- Workshop on new technology
- Pair programming sessions
- Code review guidelines
- Best practices docs

🎯 **Deep Mode complete** - enterprise migration with optimization

---

## Safety Guarantees

**WILL:**
- ✅ Create characterization tests before changes
- ✅ Maintain backward compatibility
- ✅ Provide instant rollback
- ✅ Validate performance against baseline
- ✅ Run security audit

**NEVER:**
- ❌ Modify code without test coverage
- ❌ Deploy without rollback plan
- ❌ Introduce breaking API changes
- ❌ Skip security validation
- ❌ Ignore performance regressions

---

## Examples

```bash
# React to hooks
/code-migrate src/components/Dashboard.jsx --target react-hooks

# Python 2 to 3
/code-migrate src/legacy/ --target python3

# Large migration with Strangler Fig
/code-migrate src/ --target react18 --strategy strangler-fig

# Enterprise migration
/code-migrate src/ --target nextjs --mode deep --test-first
```
