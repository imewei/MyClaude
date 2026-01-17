---
description: Multi-dimensional validation with automated testing, security scanning,
  code review
triggers:
- /double-check
- workflow for double check
version: 1.0.7
command: /double-check
argument-hint: '[work] [--deep] [--security] [--performance]'
execution_modes:
  quick: 5-15min
  standard: 30-60min
  enterprise: 2-4h
workflow_type: sequential
color: orange
allowed-tools: [Read, Task, Bash]
---


# Comprehensive Validation

$ARGUMENTS

## Modes

| Mode | Time | Dimensions |
|------|------|------------|
| Quick | 5-15min | 5 (linting, tests, types, build, basic security) |
| Standard | 30-60min | 10 (+ coverage, security, a11y, perf, infra) |
| Enterprise | 2-4h | 10 + deep analysis |

Flags: `--deep`, `--security`, `--performance`

## Process

1. **Scope Verification**:
   - Review conversation for original task
   - List requirements and acceptance criteria
   - Define "complete" for this task
   - Traceability: Every requirement addressed?

2. **Automated Checks**:

### Quick (5 Dimensions)
| Dimension | Tools | Pass |
|-----------|-------|------|
| Linting | ruff, eslint, clippy | No errors |
| Tests | pytest, jest, cargo test | All pass |
| Types | mypy, tsc, cargo check | No errors |
| Build | build command | Succeeds |
| Basic security | npm audit, pip-audit | No high/critical |

### Standard/Enterprise (+5)
| Dimension | Tools | Pass |
|-----------|-------|------|
| Coverage | pytest-cov, jest | >80% |
| Security | semgrep, bandit, gitleaks | No high/critical |
| Accessibility | pa11y, axe | No violations |
| Performance | benchmark suite | Within SLOs |
| Infrastructure | terraform validate, kubectl dry-run | Valid |

3. **Manual Review** (Standard+):

**Functional**:
- Happy path works
- Edge cases (null, empty, boundary) handled
- Error handling robust, user-friendly messages
- No silent failures

**Code Quality**:
- Follows project conventions
- Function size <50 lines
- DRY principles
- Appropriate abstraction
- Complete documentation

4. **Security Analysis** (Standard+):

**Automated**: semgrep, bandit, gitleaks, npm audit, pip-audit

**Manual**:
- No secrets (API keys, passwords, tokens)
- All user input sanitized
- SQL parameterized, XSS escaped
- Auth/authz enforced
- No known dependency vulnerabilities

5. **Performance** (Enterprise):

**Profiling**: CPU (cProfile, node --prof, perf), Memory (memory_profiler, heapdump), Load (wrk, k6, locust)

**Checks**: No N+1, caching layer, DB indexes, efficient algorithm, pagination

6. **Production Readiness** (Enterprise):

**Config**: No hardcoded values, secrets in vault, env-specific configs separated

**Observability**: Structured logging (JSON), metrics collection, error tracking, health checks

**Deployment**: Rollback plan tested, reversible migrations, CI/CD green, smoke tests defined

7. **Breaking Changes** (Standard+):
- API compatibility (no breaking or version bump)
- Deprecation warnings
- Migration guide if breaking
- All integration tests pass

## Output

```markdown
## Summary
- Assessment: ✅ Ready / ⚠️ Needs work / ❌ Not ready
- Confidence: High / Medium / Low
- Mode: Quick / Standard / Enterprise

## Issues
### Critical (Must Fix)
### Important (Should Fix)
### Minor (Nice to Fix)

## Recommendations
## Evidence (tests, coverage, scans)
```

## Advanced

- `--deep`: Property-based testing, fuzzing, dead code detection
- `--security`: OWASP Top 10, penetration testing checklist, crypto review
- `--performance`: Flamegraphs, memory profiling, load testing, query analysis

## External Docs

- `validation-dimensions.md` - All 10 dimensions with checklists
- `automated-validation-scripts.md` - Ready-to-use scripts
- `security-validation-guide.md` - OWASP Top 10, security analysis
- `performance-analysis-guide.md` - Profiling, N+1, load testing
- `production-readiness-checklist.md` - Config, observability, deployment

## Success

| Mode | Criteria |
|------|----------|
| Quick | All automated pass, no critical security, >70% coverage |
| Standard | + Manual review, all 10 dimensions, >80% coverage |
| Enterprise | + Security audit, performance meets SLOs, rollback tested |
