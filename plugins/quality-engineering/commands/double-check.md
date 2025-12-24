---
version: "1.0.6"
command: /double-check
description: Multi-dimensional validation with automated testing, security scanning, and code review
argument-hint: [work-to-validate] [--deep] [--security] [--performance]
execution_modes:
  quick: "5-15 minutes"
  standard: "30-60 minutes"
  enterprise: "2-4 hours"
workflow_type: "sequential"
interactive_mode: true
color: orange
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, SlashCommand
agents:
  primary:
    - multi-agent-orchestrator
    - code-quality
    - code-reviewer
  conditional:
    - agent: research-intelligence
      trigger: pattern "research|paper|publication|methodology"
    - agent: systems-architect
      trigger: pattern "architecture|design.*pattern|system.*design|scalability"
    - agent: security-auditor
      trigger: pattern "security|auth|crypto|secrets|vulnerability"
    - agent: performance-engineer
      trigger: pattern "performance|optimization|bottleneck|latency"
    - agent: test-automator
      trigger: pattern "test|coverage|validation"
  orchestrated: true
---

# Comprehensive Double-Check & Validation

Systematic multi-dimensional validation with automated verification, security analysis, and code review.

## Context

Validation target: $ARGUMENTS

---

## Mode Selection

| Mode | Duration | Agents | Dimensions |
|------|----------|--------|------------|
| Quick | 5-15 min | code-quality | 5 (linting, tests, types, build, basic security) |
| Standard (default) | 30-60 min | + code-reviewer, test-automator | 10 (+ coverage, security, a11y, perf, infra) |
| Enterprise | 2-4h | + security-auditor, performance-engineer | 10 + deep analysis |

---

## Phase 1: Scope & Requirements Verification

| Step | Action |
|------|--------|
| 1 | Review conversation for original task |
| 2 | List explicit requirements and acceptance criteria |
| 3 | Define "complete" for this task |
| 4 | Traceability: Every requirement addressed? |

**Reference:** [Validation Dimensions](../docs/double-check/validation-dimensions.md)

---

## Phase 2: Automated Checks

### Quick Mode (5 Dimensions)

| Dimension | Tools | Pass Criteria |
|-----------|-------|---------------|
| Linting | ruff, eslint, clippy | No errors |
| Tests | pytest, jest, cargo test | All pass |
| Type checking | mypy, tsc, cargo check | No errors |
| Build | build command | Succeeds |
| Basic security | npm audit, pip-audit | No high/critical |

### Standard/Enterprise Mode (+5 Dimensions)

| Dimension | Tools | Pass Criteria |
|-----------|-------|---------------|
| Coverage | pytest-cov, jest --coverage | >80% |
| Security scan | semgrep, bandit, gitleaks | No high/critical |
| Accessibility | pa11y, axe | No violations |
| Performance | benchmark suite | Within SLOs |
| Infrastructure | terraform validate, kubectl dry-run | Valid |

**Scripts:** [Automated Validation Scripts](../docs/double-check/automated-validation-scripts.md)

---

## Phase 3: Manual Review (Standard+)

### Functional Correctness

| Check | Criteria |
|-------|----------|
| Happy path | Works as expected |
| Edge cases | null, empty, boundary values handled |
| Error handling | Robust, user-friendly messages |
| Silent failures | None |

### Code Quality

| Check | Criteria |
|-------|----------|
| Standards | Follows project conventions |
| Function size | <50 lines |
| Duplication | DRY principles |
| Abstraction | Appropriate levels |
| Documentation | Complete |

**Detailed checklist:** [Validation Dimensions](../docs/double-check/validation-dimensions.md)

---

## Phase 4: Security Analysis (Standard+)

### Automated
- semgrep, bandit (static analysis)
- gitleaks, trufflehog (secrets detection)
- npm audit, pip-audit (dependencies)

### Manual Review

| Category | Check |
|----------|-------|
| Secrets | No API keys, passwords, tokens in code |
| Input validation | All user input sanitized |
| Injection | SQL parameterized, XSS escaped |
| Auth/authz | Properly enforced |
| Dependencies | No known vulnerabilities |

**Deep dive:** [Security Validation Guide](../docs/double-check/security-validation-guide.md)

---

## Phase 5: Performance Analysis (Enterprise)

### Profiling

| Type | Tools |
|------|-------|
| CPU | cProfile, node --prof, perf |
| Memory | memory_profiler, heapdump |
| Load | wrk, k6, locust |

### Checklist

| Issue | Fix |
|-------|-----|
| N+1 queries | Eager loading |
| Missing cache | Add caching layer |
| No indexes | Add database indexes |
| Inefficient algorithm | Optimize or replace |
| No pagination | Implement pagination |

**Guide:** [Performance Analysis Guide](../docs/double-check/performance-analysis-guide.md)

---

## Phase 6: Production Readiness (Enterprise)

### Configuration
- [ ] No hardcoded values
- [ ] Secrets in vault
- [ ] Environment-specific configs separated

### Observability
- [ ] Structured logging (JSON)
- [ ] Metrics collection
- [ ] Error tracking
- [ ] Health check endpoints

### Deployment
- [ ] Rollback plan tested
- [ ] Migrations reversible
- [ ] CI/CD green
- [ ] Smoke tests defined

**Checklist:** [Production Readiness Checklist](../docs/double-check/production-readiness-checklist.md)

---

## Phase 7: Breaking Changes (Standard+)

| Check | Action |
|-------|--------|
| API compatibility | No breaking changes or version bump |
| Deprecation | Warnings for old patterns |
| Migration | Guide provided if breaking |
| Integration | All integration tests pass |

---

## Validation Report Format

```markdown
## Summary
- **Assessment**: ✅ Ready / ⚠️ Needs work / ❌ Not ready
- **Confidence**: High / Medium / Low
- **Mode**: Quick / Standard / Enterprise

## Issues Found
### Critical (Must Fix)
### Important (Should Fix)
### Minor (Nice to Fix)

## Recommendations
## Evidence (tests, coverage, scans)
```

---

## Advanced Options

| Flag | Purpose |
|------|---------|
| `--deep` | Property-based testing, fuzzing, dead code detection |
| `--security` | OWASP Top 10, penetration testing checklist, crypto review |
| `--performance` | Flamegraphs, memory profiling, load testing, query analysis |

---

## External Documentation

| Document | Content |
|----------|---------|
| [Validation Dimensions](../docs/double-check/validation-dimensions.md) | All 10 dimensions with checklists |
| [Automated Validation Scripts](../docs/double-check/automated-validation-scripts.md) | Ready-to-use scripts |
| [Security Validation Guide](../docs/double-check/security-validation-guide.md) | OWASP Top 10, security analysis |
| [Performance Analysis Guide](../docs/double-check/performance-analysis-guide.md) | Profiling, N+1, load testing |
| [Production Readiness Checklist](../docs/double-check/production-readiness-checklist.md) | Config, observability, deployment |

---

## Success Criteria

| Mode | Criteria |
|------|----------|
| Quick | All automated checks pass, no critical security issues, >70% coverage |
| Standard | + Manual review complete, all 10 dimensions validated, >80% coverage |
| Enterprise | + Security audit complete, performance meets SLOs, rollback tested |
