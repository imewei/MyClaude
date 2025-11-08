---
version: 1.0.3
command: /double-check
description: Comprehensive multi-dimensional validation with automated testing, security scanning, and ultrathink reasoning across 3 execution modes
argument-hint: [work-to-validate] [--deep] [--security] [--performance]
execution_modes:
  quick:
    duration: "5-15 minutes"
    description: "Fast validation for small changes or PRs"
    agents: ["code-quality"]
    scope: "Automated checks only (linting, tests, basic security)"
    validations: "5 dimensions"
  standard:
    duration: "30-60 minutes"
    description: "Comprehensive validation for features or releases"
    agents: ["code-quality", "code-reviewer", "test-automator"]
    scope: "All automated checks + manual code review + test coverage analysis"
    validations: "10 dimensions"
  enterprise:
    duration: "2-4 hours"
    description: "Deep validation for production deployments"
    agents: ["multi-agent-orchestrator", "code-quality", "security-auditor", "performance-engineer"]
    scope: "Full validation suite + security audit + performance profiling + architecture review"
    validations: "10 dimensions + deep analysis"
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

Systematic multi-dimensional validation with automated verification, security analysis, and ultrathink reasoning.

## Context

The user needs comprehensive validation for: $ARGUMENTS

## Execution Mode Selection

<AskUserQuestion>
questions:
  - question: "Which validation depth best fits your needs?"
    header: "Validation Mode"
    multiSelect: false
    options:
      - label: "Quick (5-15 minutes)"
        description: "Fast validation for small changes or PRs. Automated checks only (linting, tests, basic security). Validates 5 core dimensions."

      - label: "Standard (30-60 minutes)"
        description: "Comprehensive validation for features or releases. All automated checks + manual review + coverage analysis. Validates all 10 dimensions."

      - label: "Enterprise (2-4 hours)"
        description: "Deep validation for production deployments. Full suite + security audit + performance profiling + architecture review. 10 dimensions + deep analysis."
</AskUserQuestion>

## Instructions

### Phase 1: Scope & Requirements Verification

Extract and validate original requirements:

1. **Review conversation history** for the original task
2. **List all explicit requirements** and acceptance criteria
3. **Define "complete"** for this specific task
4. **Requirement traceability**: Every requirement addressed? ✓/✗

**See comprehensive guide**: [Validation Dimensions](../docs/double-check/validation-dimensions.md)

---

### Phase 2: Automated Checks

Run all applicable automated checks based on execution mode:

#### Quick Mode: Core Checks (5 dimensions)

```bash
# 1. Linting and formatting
npm run lint || ruff check . || cargo clippy

# 2. Tests
npm test || pytest || cargo test

# 3. Type checking
npm run type-check || mypy . || cargo check

# 4. Build verification
npm run build || python -m build || cargo build

# 5. Basic security scan
npm audit --audit-level=moderate || pip-audit
```

#### Standard/Enterprise Mode: Full Suite (10 dimensions)

```bash
# Run all Quick checks plus:

# 6. Test coverage
npm run test:coverage || pytest --cov --cov-report=html

# 7. Security scanning
semgrep --config=auto . || bandit -r .
gitleaks detect --no-git || trufflehog filesystem .

# 8. Accessibility testing (if web UI)
npm run test:a11y || pa11y-ci

# 9. Performance profiling (if applicable)
npm run benchmark || pytest --benchmark-only

# 10. Infrastructure validation (if IaC present)
terraform validate || kubectl apply --dry-run=client -f k8s/
```

**See automation scripts**: [Automated Validation Scripts](../docs/double-check/automated-validation-scripts.md)

---

### Phase 3: Manual Review (Standard/Enterprise)

#### Functional Correctness
- [ ] Happy path scenarios work
- [ ] Edge cases handled (null, empty, boundary values)
- [ ] Error handling robust and user-friendly
- [ ] No silent failures

#### Code Quality
- [ ] Follows project coding standards
- [ ] Functions are focused (<50 lines)
- [ ] No code duplication (DRY)
- [ ] Appropriate abstraction levels
- [ ] Documentation complete

**See detailed checklist**: [Validation Dimensions](../docs/double-check/validation-dimensions.md)

---

### Phase 4: Security Analysis (Standard/Enterprise)

**Automated scanning** (already run in Phase 2)

**Manual security review**:
- [ ] No secrets in code (API keys, passwords, tokens)
- [ ] Input validation and sanitization
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (proper escaping)
- [ ] Authentication/authorization enforced
- [ ] Dependencies vulnerability-free

**See security deep-dive**: [Security Validation Guide](../docs/double-check/security-validation-guide.md)

---

### Phase 5: Performance Analysis (Enterprise Mode)

Run performance profiling and optimization analysis:

```bash
# CPU profiling
python -m cProfile -o profile.stats script.py
node --prof app.js

# Memory profiling
python -m memory_profiler script.py
node --inspect app.js

# Load testing
wrk -t12 -c400 -d30s http://localhost:3000
k6 run load-test.js
```

**Performance checklist**:
- [ ] No N+1 query problems
- [ ] Appropriate caching
- [ ] Database indexes on queried fields
- [ ] Efficient algorithms
- [ ] Pagination for large datasets

**See comprehensive guide**: [Performance Analysis Guide](../docs/double-check/performance-analysis-guide.md)

---

### Phase 6: Production Readiness (Enterprise Mode)

Validate deployment readiness:

**Configuration**:
- [ ] No hardcoded configuration
- [ ] Secrets in vault (not env vars)
- [ ] Environment-specific configs separated

**Observability**:
- [ ] Structured logging (JSON)
- [ ] Metrics collection enabled
- [ ] Error tracking configured
- [ ] Health check endpoints

**Deployment**:
- [ ] Rollback plan tested
- [ ] Database migrations reversible
- [ ] CI/CD pipeline green
- [ ] Smoke tests defined

**See production checklist**: [Production Readiness Checklist](../docs/double-check/production-readiness-checklist.md)

---

### Phase 7: Breaking Changes Analysis (Standard/Enterprise)

Assess impact on existing functionality:

- [ ] No breaking changes to public APIs
- [ ] Deprecation warnings for old patterns
- [ ] Migration guide provided if breaking
- [ ] Integration tests pass

---

## Validation Report

Provide structured report:

### Summary
- **Overall assessment**: ✅ Ready / ⚠️ Needs work / ❌ Not ready
- **Confidence level**: High / Medium / Low
- **Execution mode**: Quick / Standard / Enterprise

### Strengths
- What was done well
- Best practices followed
- Innovative solutions

### Issues Found

#### Critical (Must Fix)
1. [Issue description, location, impact]
2. [Recommended solution]

#### Important (Should Fix)
1. [Issue description, location, impact]
2. [Recommended solution]

#### Minor (Nice to Fix)
1. [Issue description, location, impact]
2. [Recommended solution]

### Recommendations
1. Immediate actions needed
2. Follow-up improvements
3. Long-term considerations

### Verification Evidence
- Tests run: [results]
- Coverage: [percentage]
- Security scans: [results]
- Performance benchmarks: [results]
- Build status: [success/failure]

---

## Advanced Options

### --deep Flag
Run extended analysis:
- Property-based testing (QuickCheck, Hypothesis)
- Fuzzing for input validation
- Dependency graph analysis
- Dead code detection
- Cyclomatic complexity metrics

### --security Flag
Enhanced security focus:
- OWASP Top 10 validation
- Penetration testing checklist
- Security headers verification
- Certificate validation
- Cryptographic implementation review

### --performance Flag
Deep performance analysis:
- CPU profiling with flamegraphs
- Memory profiling and leak detection
- Load testing under production conditions
- Database query analysis (EXPLAIN ANALYZE)
- Network latency analysis

---

## External Documentation

- [Validation Dimensions](../docs/double-check/validation-dimensions.md) - All 10 validation dimensions with detailed checklists
- [Automated Validation Scripts](../docs/double-check/automated-validation-scripts.md) - Ready-to-use validation scripts for all languages
- [Security Validation Guide](../docs/double-check/security-validation-guide.md) - Comprehensive security analysis and OWASP Top 10
- [Performance Analysis Guide](../docs/double-check/performance-analysis-guide.md) - Performance profiling, N+1 detection, load testing
- [Production Readiness Checklist](../docs/double-check/production-readiness-checklist.md) - Configuration, observability, deployment strategies

---

## Success Criteria

**Quick Mode**:
- ✅ All automated checks pass
- ✅ No critical security issues
- ✅ Tests pass with >70% coverage

**Standard Mode**:
- ✅ All Quick criteria met
- ✅ Manual code review complete
- ✅ All 10 dimensions validated
- ✅ No breaking changes or migration provided
- ✅ Tests pass with >80% coverage

**Enterprise Mode**:
- ✅ All Standard criteria met
- ✅ Security audit complete (no high/critical vulnerabilities)
- ✅ Performance benchmarks meet SLOs
- ✅ Production readiness checklist complete
- ✅ Architecture review approved
- ✅ Rollback plan tested

---

Execute comprehensive validation across selected execution mode, provide detailed findings with specific line numbers and actionable recommendations.
