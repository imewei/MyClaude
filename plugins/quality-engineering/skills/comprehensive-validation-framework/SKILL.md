---
name: comprehensive-validation-framework
description: Systematic multi-dimensional validation framework for code, APIs, and systems. Use when validating implementations before deployment, double-checking work for production readiness, or performing comprehensive quality assurance across 10 critical dimensions (requirements, functionality, code quality, security, performance, accessibility, testing, compatibility, operations, documentation). Includes automated scripts for security scanning, testing, linting, performance profiling, and accessibility checks, plus deep-dive reference guides for each validation dimension.
---

# Comprehensive Validation Framework

## Overview

This skill provides a systematic framework for validating code, APIs, and systems across 10 critical dimensions before deployment. It combines automated validation scripts with deep-dive reference material to ensure production readiness, quality, and reliability.

**Use this skill when**:
- Double-checking implementations before production deployment
- Validating new features for quality and security
- Performing comprehensive pre-launch audits
- Reviewing code changes for breaking changes or compatibility issues
- Ensuring production readiness across all dimensions

---

## Validation Workflow

Execute validation in this systematic order:

### Phase 1: Requirements & Scope (5 min)
1. Review original task/requirements
2. Verify all requirements addressed
3. Check for scope creep

### Phase 2: Automated Checks (10-20 min)
1. Run `scripts/run_all_validations.py` for comprehensive automation
2. Review automated findings
3. Run additional targeted scripts as needed

### Phase 3: Manual Review (20-40 min)
1. Functional correctness analysis
2. Code quality review
3. Security deep-dive (consult `references/security-deep-dive.md`)
4. Performance analysis (consult `references/performance-optimization.md`)
5. Accessibility verification (if UI - consult `references/accessibility-standards.md`)

### Phase 4: Production Readiness (10-15 min)
1. Verify observability (logging, metrics, tracing)
2. Check deployment configuration
3. Confirm rollback plan exists
4. Review documentation completeness

### Phase 5: Report Generation (10 min)
1. Use `assets/validation-report-template.md` as structure
2. Document all findings with severity and location
3. Provide actionable recommendations
4. Sign off with approval status

**Total Time**: 1-2 hours for comprehensive validation

---

## Automated Validation Scripts

All scripts located in `scripts/` directory. Run these first for efficient validation.

### Master Orchestrator

**Script**: `scripts/run_all_validations.py`

**Purpose**: Runs all automated checks in sequence (linting, formatting, type checking, tests, security, build).

**Usage**:
```bash
# Run all validations
python scripts/run_all_validations.py

# Skip specific phases
python scripts/run_all_validations.py --skip-security --skip-tests

# Verbose output
python scripts/run_all_validations.py --verbose
```

**What it checks**:
- Code linting (ESLint, Ruff, Clippy, etc.)
- Code formatting (Prettier, Black, etc.)
- Type checking (TypeScript, MyPy, etc.)
- Unit tests with coverage
- Security scanning (via security_scan.py)
- Build verification

**When to use**: Start every validation session with this script. It provides a comprehensive overview in minutes.

### Security Scanner

**Script**: `scripts/security_scan.py`

**Purpose**: Comprehensive security scanning across dependency vulnerabilities, SAST, and secret detection.

**Usage**:
```bash
# Full security scan
python scripts/security_scan.py

# Fast mode (critical/high only)
python scripts/security_scan.py --fast

# JSON output for CI/CD
python scripts/security_scan.py --json
```

**What it scans**:
1. **Dependency vulnerabilities**: npm audit, pip-audit, cargo audit
2. **SAST**: Semgrep, Bandit for code security issues
3. **Secret detection**: Gitleaks, Trufflehog for exposed secrets

**Exit codes**: Returns 1 if critical/high severity issues found.

**When to use**: Always run for security-sensitive features, authentication/authorization code, payment processing, data handling, or before production deployment.

### Test Runner

**Script**: `scripts/test_runner.py`

**Purpose**: Cross-language test execution with coverage analysis.

**Usage**:
```bash
# Run tests with coverage
python scripts/test_runner.py

# Enforce minimum coverage
python scripts/test_runner.py --min-coverage 80

# Generate HTML report
python scripts/test_runner.py --html-report
```

**Supported languages**: JavaScript/TypeScript (Jest), Python (pytest), Rust (cargo test), Go (go test).

**When to use**: After any code changes. Verify tests pass and coverage meets standards (target: >80% overall, >95% for critical paths).

### Linting & Formatting

**Script**: `scripts/lint_check.py`

**Purpose**: Run linters and formatters to ensure code quality standards.

**Usage**:
```bash
# Check code style
python scripts/lint_check.py

# Auto-fix issues
python scripts/lint_check.py --fix
```

**Checks**: ESLint, Prettier, Ruff, Black, Clippy, gofmt (auto-detected by project type).

**When to use**: Before committing code. Run with `--fix` to automatically resolve style issues.

### Performance Profiler

**Script**: `scripts/performance_profiler.py`

**Purpose**: Profile code execution to identify bottlenecks.

**Usage**:
```bash
# Profile Python script
python scripts/performance_profiler.py script.py

# Show top 30 functions
python scripts/performance_profiler.py script.py --top 30
```

**Supported**: Python (cProfile), JavaScript (node --prof).

**When to use**: When performance issues suspected, for optimization work, or for performance-critical code paths.

### Accessibility Checker

**Script**: `scripts/accessibility_check.py`

**Purpose**: Verify WCAG compliance for web applications.

**Usage**:
```bash
# Check accessibility
python scripts/accessibility_check.py http://localhost:3000

# Check specific WCAG level
python scripts/accessibility_check.py http://localhost:3000 --wcag-level AA

# Use specific tool
python scripts/accessibility_check.py http://localhost:3000 --tool axe
```

**Tools**: pa11y, axe-core.

**When to use**: For all UI changes. Ensure WCAG 2.1 Level AA compliance.

### Build Verifier

**Script**: `scripts/build_verify.py`

**Purpose**: Verify build configuration works correctly.

**Usage**:
```bash
# Run build
python scripts/build_verify.py

# Clean build
python scripts/build_verify.py --clean

# Release build
python scripts/build_verify.py --release
```

**Supported**: npm, Python (build), Cargo, Go.

**When to use**: Before deployment, after configuration changes, or when build issues suspected.

---

## Reference Documentation

Deep-dive guides located in `references/` directory. Consult these for manual review and expert guidance.

### Security Deep-Dive

**File**: `references/security-deep-dive.md`

**Contents**:
- OWASP Top 10 (2021/2025) with code examples
- Authentication & Authorization patterns (OAuth 2.0, JWT)
- Input validation & sanitization by input type
- Cryptography best practices (password hashing, encryption)
- Dependency security management
- Secret management (environment variables, vaults)
- Language-specific security (Python, JavaScript, Rust, Go)
- API security checklist (HTTPS, rate limiting, CORS, security headers)
- Infrastructure security (Docker, Kubernetes)
- Automated security testing in CI/CD

**When to consult**:
- Reviewing authentication/authorization code
- Validating input handling
- Checking cryptography implementation
- Assessing API security
- Before deploying security-sensitive features

**Search patterns**: Use grep to find specific topics:
```bash
grep -n "OWASP\|OAuth\|JWT\|SQL injection\|XSS" references/security-deep-dive.md
```

### Performance Optimization

**File**: `references/performance-optimization.md`

**Contents**:
- Profiling tools by language (Python, JavaScript, Rust, Go)
- Common bottlenecks (N+1 queries, missing indexes, inefficient algorithms)
- Caching strategies (multi-tier caching, Redis patterns)
- Load testing (k6, Locust, wrk)
- Database optimization (EXPLAIN ANALYZE, indexing, connection pooling)
- Frontend performance (bundle size, Core Web Vitals)
- Production monitoring (Golden Signals, APM)

**When to consult**:
- Investigating performance issues
- Optimizing slow operations
- Setting up performance monitoring
- Planning load testing
- Reviewing database queries

**Search patterns**:
```bash
grep -n "N+1\|caching\|profiling\|load test" references/performance-optimization.md
```

### Testing Best Practices

**File**: `references/testing-best-practices.md`

**Contents**:
- Testing pyramid (unit, integration, E2E distribution)
- AAA pattern (Arrange-Act-Assert)
- Test naming conventions
- Mocking and stubbing (pytest, Jest)
- Property-based testing (Hypothesis, fast-check)
- Test data management (factories)
- Mutation testing
- Coverage targets and strategies

**When to consult**:
- Writing new tests
- Improving test quality
- Debugging flaky tests
- Planning test strategy
- Reviewing test coverage gaps

**Search patterns**:
```bash
grep -n "AAA\|mock\|fixture\|coverage" references/testing-best-practices.md
```

### Production Readiness

**File**: `references/production-readiness.md`

**Contents**:
- Pre-launch checklist (configuration, security, observability, reliability)
- Health check implementation (liveness, readiness probes)
- Structured logging (Python, JavaScript)
- Metrics and monitoring (Prometheus, Datadog)
- Error tracking (Sentry integration)
- Graceful shutdown patterns
- Circuit breaker pattern
- Deployment strategies (blue-green, canary, feature flags)
- Runbook template

**When to consult**:
- Preparing for production deployment
- Setting up observability
- Implementing health checks
- Planning deployment strategy
- Creating runbooks

**Search patterns**:
```bash
grep -n "health check\|logging\|metrics\|circuit breaker" references/production-readiness.md
```

### Accessibility Standards

**File**: `references/accessibility-standards.md`

**Contents**:
- WCAG 2.1 principles (POUR: Perceivable, Operable, Understandable, Robust)
- Conformance levels (A, AA, AAA)
- Key success criteria with code examples
- ARIA roles, states, and properties
- Semantic HTML patterns
- Form accessibility
- Skip links and screen reader text
- Testing tools (axe-core, pa11y, Lighthouse)
- Common accessible patterns (modals, accordions, tables)

**When to consult**:
- Building UI components
- Ensuring WCAG compliance
- Debugging accessibility issues
- Implementing ARIA patterns
- Testing with screen readers

**Search patterns**:
```bash
grep -n "WCAG\|ARIA\|contrast\|keyboard" references/accessibility-standards.md
```

### Breaking Changes Guide

**File**: `references/breaking-changes-guide.md`

**Contents**:
- Definition of breaking changes
- Semantic versioning (SemVer)
- Strategies to avoid breaking changes
- Deprecation process and warnings
- API versioning strategies
- Database migrations (blue-green pattern)
- Feature flags for gradual rollout
- Rollback strategies
- Migration guide template
- Communication checklist

**When to consult**:
- Planning API changes
- Refactoring public interfaces
- Database schema changes
- Versioning libraries
- Writing migration guides

**Search patterns**:
```bash
grep -n "breaking\|deprecation\|migration\|rollback" references/breaking-changes-guide.md
```

---

## Validation Report Template

**File**: `assets/validation-report-template.md`

**Purpose**: Structured template for documenting validation findings and recommendations.

**Usage**:
1. Copy template to new file
2. Fill in each section systematically
3. Document findings with severity, location (file:line), and recommendations
4. Provide actionable next steps
5. Sign off with approval status

**Template sections**:
- Executive summary
- Results by validation dimension (10 dimensions)
- Summary of findings (strengths, critical/important/minor issues)
- Recommendations (immediate, follow-up, long-term)
- Verification evidence (test results, coverage, security scans)
- Sign-off with approval status

**Example usage**:
```bash
cp assets/validation-report-template.md validation-report-2024-10-27.md
# Fill in the template with findings
```

---

## Validation Dimensions Reference

### 1. Scope & Requirements Verification

**Check**:
- All explicit requirements addressed
- Implicit requirements identified
- No scope creep
- Acceptance criteria met

**Process**:
1. Review original task description
2. List all requirements
3. Verify each requirement has been implemented
4. Confirm no unintended features added

### 2. Functional Correctness Analysis

**Check**:
- Happy path works correctly
- Edge cases handled (null, empty, boundary values)
- Error handling comprehensive
- Integration with existing systems tested

**Process**:
1. Test typical usage scenarios
2. Test edge cases systematically
3. Verify error messages are actionable
4. Run automated tests: `python scripts/test_runner.py`

### 3. Code Quality & Maintainability

**Check**:
- Follows coding standards
- Functions are focused and single-purpose
- No code duplication
- Appropriate complexity levels
- Proper documentation

**Process**:
1. Run: `python scripts/lint_check.py`
2. Review code manually for complexity
3. Check documentation completeness
4. Verify naming consistency

### 4. Security Analysis

**Check**:
- No secrets in code
- Input validation implemented
- Injection prevention (SQL, XSS, command)
- Authentication/authorization enforced
- Dependencies up to date

**Process**:
1. Run: `python scripts/security_scan.py`
2. Consult: `references/security-deep-dive.md`
3. Review OWASP Top 10 checklist
4. Verify security headers set

### 5. Performance Analysis

**Check**:
- No N+1 query problems
- Database indexes verified
- Caching implemented appropriately
- Efficient algorithms used
- Resource cleanup implemented

**Process**:
1. Run: `python scripts/performance_profiler.py <script>`
2. Consult: `references/performance-optimization.md`
3. Check database queries with EXPLAIN
4. Run load tests if applicable

### 6. Accessibility & User Experience

**Check** (if UI):
- WCAG 2.1 Level AA compliance
- Keyboard navigation works
- Screen reader compatible
- Color contrast sufficient
- Form labels present

**Process**:
1. Run: `python scripts/accessibility_check.py <url>`
2. Consult: `references/accessibility-standards.md`
3. Test with keyboard only
4. Test with screen reader (VoiceOver, NVDA)

### 7. Testing Coverage & Strategy

**Check**:
- Unit tests for core logic (>80% coverage)
- Integration tests for API/DB interactions
- E2E tests for critical flows
- Tests are independent and reliable
- Edge cases tested

**Process**:
1. Run: `python scripts/test_runner.py --min-coverage 80 --html-report`
2. Consult: `references/testing-best-practices.md`
3. Review coverage report for gaps
4. Verify test quality (AAA pattern, proper naming)

### 8. Breaking Changes & Backward Compatibility

**Check**:
- No breaking changes to public APIs (or documented if necessary)
- Deprecation warnings added
- Migration guide provided
- Semantic versioning followed
- Rollback plan exists

**Process**:
1. Review API changes
2. Consult: `references/breaking-changes-guide.md`
3. Verify migration path documented
4. Test rollback procedure

### 9. Deployment & Operations Readiness

**Check**:
- No hardcoded configuration
- Secrets managed securely
- Logging at appropriate levels
- Metrics instrumented
- Health check endpoint
- Graceful shutdown implemented

**Process**:
1. Verify configuration management
2. Consult: `references/production-readiness.md`
3. Test health endpoints
4. Review CI/CD pipeline status

### 10. Documentation & Knowledge Transfer

**Check**:
- README updated
- API documentation complete
- Code comments for complex logic
- Runbook for operations
- Architecture diagrams (if significant changes)

**Process**:
1. Review all documentation
2. Verify setup instructions work
3. Check API docs are up to date
4. Ensure troubleshooting guide exists

---

## Quick Start Guide

### For Rapid Validation (15 minutes)

1. **Run automated checks**:
   ```bash
   python scripts/run_all_validations.py
   ```

2. **Review critical dimensions**:
   - Security: Critical/high issues found?
   - Tests: All passing? Coverage >80%?
   - Build: Successful?

3. **Quick report**:
   - Overall status: Pass/Fail
   - Critical issues: List any blockers
   - Recommendation: Approve or request fixes

### For Comprehensive Validation (1-2 hours)

1. **Phase 1 - Automated** (15 min):
   ```bash
   python scripts/run_all_validations.py --verbose
   ```

2. **Phase 2 - Manual Review** (45 min):
   - Review code quality manually
   - Consult relevant reference docs
   - Check each validation dimension

3. **Phase 3 - Documentation** (20 min):
   - Fill out `assets/validation-report-template.md`
   - Document all findings with locations
   - Provide actionable recommendations

4. **Phase 4 - Sign-Off** (5 min):
   - Approve, approve with conditions, or reject
   - List conditions if applicable
   - Set follow-up timeline

---

## Best Practices

### Running Validations

1. **Start with automation**: Always run automated scripts first to catch obvious issues quickly
2. **Prioritize critical dimensions**: Security, functionality, and testing are most critical
3. **Document everything**: Use the report template to capture findings systematically
4. **Be specific**: Provide exact file:line locations for all issues
5. **Be actionable**: Every finding should have a clear recommendation

### Using Reference Documentation

1. **Don't read everything**: Use grep to find specific topics
2. **Bookmark patterns**: Copy useful code examples for reuse
3. **Consult for deep-dive**: Reference docs provide expert guidance for complex topics
4. **Stay updated**: Security and best practices evolve, update references periodically

### Report Generation

1. **Use the template**: Consistency helps readers and establishes standards
2. **Severity matters**: Critical blocks deployment, high should fix soon, medium/low are improvements
3. **Evidence-based**: Include test results, coverage numbers, scan outputs
4. **Be constructive**: Frame findings as opportunities to improve, not criticism

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Comprehensive Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Run comprehensive validation
        run: |
          python scripts/run_all_validations.py

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        if: always()
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running validation checks..."
python scripts/lint_check.py --fix
python scripts/test_runner.py

if [ $? -ne 0 ]; then
  echo "Validation failed. Commit aborted."
  exit 1
fi
```

---

## Troubleshooting

### Scripts Not Found

**Issue**: `ModuleNotFoundError` or `command not found`

**Solution**:
- Ensure Python 3.12+ installed
- Install dependencies: `pip install -r requirements.txt` (if provided)
- For tool-specific errors (semgrep, gitleaks), install the tool or skip that check

### Validation Takes Too Long

**Solution**:
- Use fast mode: `--skip-tests --skip-build`
- Run critical checks only: security and tests
- Profile specific areas instead of full validation

### False Positives in Security Scan

**Solution**:
- Review findings carefully - not all are relevant
- Suppress false positives in tool config (`.semgrepignore`, etc.)
- Document why specific findings are not applicable

---

## Summary

This comprehensive validation framework provides systematic assurance across all critical dimensions before production deployment. Use automated scripts for efficiency, consult reference documentation for expert guidance, and leverage the report template for consistent documentation.

**Remember**: Validation is not about finding problems to criticize, but about ensuring quality, reliability, and production readiness. Every validation session improves the codebase and team knowledge.

**Key principle**: Validate early, validate often, validate systematically.
