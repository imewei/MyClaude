---
name: comprehensive-validation-framework
version: "1.0.7"
maturity: "5-Expert"
specialization: Production Validation
description: Systematic multi-dimensional validation framework for code, APIs, and systems. Use when validating before deployment, running security scans (OWASP Top 10, dependency vulnerabilities), checking test coverage (>80% target), verifying accessibility (WCAG 2.1 AA), profiling performance, validating breaking changes, or preparing deployment readiness reports.
---

# Comprehensive Validation Framework

Systematic validation across 10 critical dimensions before production deployment.

---

## Validation Dimensions

| Dimension | What to Check | Tool/Method |
|-----------|---------------|-------------|
| Requirements | All requirements addressed | Review task description |
| Functional | Happy path, edge cases, errors | test_runner.py |
| Code Quality | Linting, formatting, complexity | lint_check.py |
| Security | OWASP, deps, secrets | security_scan.py |
| Performance | N+1, caching, profiling | performance_profiler.py |
| Accessibility | WCAG 2.1 AA compliance | accessibility_check.py |
| Testing | >80% coverage, edge cases | test_runner.py |
| Breaking Changes | API compatibility, migrations | Manual review |
| Operations | Logging, metrics, health checks | production-readiness.md |
| Documentation | README, API docs, runbooks | Manual review |

---

## Automated Scripts

### Master Orchestrator

```bash
# Run all validations
python scripts/run_all_validations.py

# Skip specific phases
python scripts/run_all_validations.py --skip-security --skip-tests

# What it runs: lint, format, type-check, tests, security, build
```

### Security Scanner

```bash
# Full security scan
python scripts/security_scan.py

# Fast mode (critical/high only)
python scripts/security_scan.py --fast

# Scans: dependency vulnerabilities (npm/pip/cargo audit),
#        SAST (Semgrep, Bandit), secret detection (Gitleaks)
```

### Test Runner

```bash
# Run tests with coverage
python scripts/test_runner.py

# Enforce minimum coverage
python scripts/test_runner.py --min-coverage 80

# Supported: Jest, pytest, cargo test, go test
```

### Performance Profiler

```bash
# Profile Python script
python scripts/performance_profiler.py script.py --top 30

# Supported: cProfile (Python), node --prof (JavaScript)
```

### Accessibility Checker

```bash
# Check WCAG compliance
python scripts/accessibility_check.py http://localhost:3000 --wcag-level AA

# Tools: pa11y, axe-core
```

---

## Validation Workflow

### Rapid Validation (15 min)

```bash
python scripts/run_all_validations.py

# Review: Security critical/high? Tests passing? Coverage >80%? Build OK?
```

### Comprehensive Validation (1-2 hours)

| Phase | Duration | Actions |
|-------|----------|---------|
| Automated | 15 min | run_all_validations.py |
| Manual Review | 45 min | Code quality, security deep-dive, reference docs |
| Documentation | 20 min | Fill validation-report-template.md |
| Sign-Off | 5 min | Approve / Approve with conditions / Reject |

---

## Validation Checklist

### Security

- [ ] No secrets in code (run security_scan.py)
- [ ] Input validation implemented
- [ ] SQL/XSS/Command injection prevented
- [ ] Auth/authz enforced
- [ ] Dependencies up to date

### Performance

- [ ] No N+1 queries
- [ ] Database indexes verified
- [ ] Caching appropriate
- [ ] Resource cleanup implemented

### Testing

- [ ] Unit tests >80% coverage
- [ ] Integration tests for API/DB
- [ ] E2E tests for critical flows
- [ ] Edge cases tested

### Operations

- [ ] No hardcoded config
- [ ] Secrets in vault/env
- [ ] Logging at appropriate levels
- [ ] Metrics instrumented
- [ ] Health check endpoint
- [ ] Graceful shutdown

---

## CI/CD Integration

```yaml
# .github/workflows/validate.yml
name: Comprehensive Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: '3.12' }
      - run: python scripts/run_all_validations.py
```

### Pre-commit Hook

```bash
#!/bin/bash
python scripts/lint_check.py --fix
python scripts/test_runner.py
[ $? -ne 0 ] && echo "Validation failed" && exit 1
```

---

## Reference Documentation

| Reference | When to Consult |
|-----------|-----------------|
| security-deep-dive.md | Auth code, input handling, crypto |
| performance-optimization.md | Slow operations, caching, load testing |
| testing-best-practices.md | Writing tests, improving coverage |
| production-readiness.md | Health checks, logging, deployment |
| accessibility-standards.md | WCAG compliance, ARIA patterns |
| breaking-changes-guide.md | API changes, migrations, versioning |

---

## Report Template

```markdown
# Validation Report

## Executive Summary
- Status: [Pass / Fail / Pass with Conditions]
- Critical Issues: [count]
- Recommendations: [summary]

## Findings by Dimension
[For each of 10 dimensions: ✅ Pass / ⚠️ Issues / ❌ Fail]

## Critical Issues (Must Fix)
1. [Issue, File:Line, Severity, Recommendation]

## Sign-Off
- Reviewer: [Name]
- Date: [Date]
- Approval: [Approved / Approved with Conditions / Rejected]
```

---

## Severity Classification

| Severity | Criteria | Action |
|----------|----------|--------|
| Critical | Security vuln, data loss, outage | Block deployment |
| High | Major bug, performance issue | Fix before deploy |
| Medium | Minor bug, code smell | Fix soon |
| Low | Enhancement, style | Track for later |

---

**Version**: 1.0.5
