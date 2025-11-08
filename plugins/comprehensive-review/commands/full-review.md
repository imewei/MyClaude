---
version: 1.0.3
category: code-review
purpose: Orchestrate comprehensive multi-dimensional code review using specialized review agents
execution_time:
  quick: "10-20 minutes - Core quality and security review (phases 1-2)"
  standard: "25-40 minutes - Full 4-phase multi-agent review"
  deep: "45-75 minutes - Complete analysis with metrics dashboard and automated remediation"
external_docs:
  - review-best-practices.md
  - risk-assessment-framework.md
  - pr-templates-library.md
tags: [code-review, multi-agent, orchestration, quality-assurance, security-audit, performance-analysis]
---

# Comprehensive Multi-Agent Code Review

Orchestrate exhaustive code review by coordinating specialized agents in sequential phases. Each phase builds upon previous findings to create comprehensive feedback covering quality, security, performance, testing, documentation, and best practices. Results are consolidated into actionable, prioritized remediation guidance.

## Execution Modes

### Quick Mode (10-20 min)
Core quality and security review only (Phases 1-2):
- Code quality analysis and architecture review
- Security vulnerability assessment
- Critical and high-priority issues only
- Use when: time-constrained reviews, hotfixes, small PRs

### Standard Mode (25-40 min) - DEFAULT
Full 4-phase multi-agent review:
- All phases (quality, security, performance, testing, docs, best practices)
- All priority levels (P0-P3)
- Comprehensive consolidated report
- Use when: standard PRs, feature development, general code changes

### Deep Mode (45-75 min)
Complete analysis with metrics and automation:
- All standard mode content
- Detailed metrics dashboard (complexity, duplication, coverage trends)
- Automated remediation suggestions
- Framework-specific deep analysis
- CI/CD integration recommendations
- Use when: major features, architectural changes, legacy code modernization

**To select mode**: Use `--mode=quick|standard|deep` flag. Default: standard.

## Configuration Options

- `--mode=<quick|standard|deep>`: Execution mode (default: standard)
- `--security-focus`: Prioritize security vulnerabilities and OWASP compliance
- `--performance-critical`: Emphasize performance bottlenecks and scalability
- `--tdd-review`: Include TDD compliance verification
- `--strict-mode`: Fail review on any critical issues
- `--framework=<name>`: Apply framework-specific best practices (React, Django, Spring, etc.)
- `--metrics-report`: Generate detailed quality metrics dashboard (auto-enabled in deep mode)

## Phase 1: Code Quality & Architecture Review

**Orchestrate** quality and architecture agents in parallel using Task tool:

### 1A. Code Quality Analysis
- **Agent**: `comprehensive-review:code-reviewer`
- **Focus**: Code complexity, maintainability, technical debt, duplication, SOLID principles
- **Tools**: SonarQube, CodeQL, Semgrep integration for static analysis
- **Output**: Quality metrics, code smell inventory, refactoring recommendations
- **Reference**: See `review-best-practices.md` for code smell patterns and fixes

**Prompt**:
```
Perform comprehensive code quality review for: $ARGUMENTS.

Analyze:
- Code complexity and maintainability index
- Technical debt and code duplication
- Naming conventions and Clean Code principles
- SOLID principle violations
- Code smells and anti-patterns

Integrate static analysis tools (SonarQube/CodeQL/Semgrep) if available.
Generate cyclomatic complexity metrics and identify refactoring opportunities.

Reference: docs/comprehensive-review/review-best-practices.md (code smells section)
```

### 1B. Architecture & Design Review
- **Agent**: `comprehensive-review:architect-review`
- **Focus**: Design patterns, structural integrity, microservices boundaries, API design
- **Analysis**: Circular dependencies, inappropriate coupling, architectural drift
- **Output**: Architecture assessment, design pattern analysis, structural recommendations

**Prompt**:
```
Review architectural design patterns and structural integrity in: $ARGUMENTS.

Evaluate:
- Microservices boundaries and API design
- Database schema and dependency management
- Domain-Driven Design adherence
- Circular dependencies and inappropriate coupling
- Enterprise architecture standards and cloud-native patterns

Verify compliance with modern architectural patterns.
```

## Phase 2: Security & Performance Review

**Orchestrate** security and performance agents, incorporating Phase 1 findings:

### 2A. Security Vulnerability Assessment
- **Agent**: `comprehensive-review:security-auditor`
- **Focus**: OWASP Top 10, dependency vulnerabilities, secrets detection, auth/authz
- **Tools**: Snyk/Trivy, GitLeaks, vulnerability scanning
- **Output**: Vulnerability report, CVE list, security risk matrix, remediation steps
- **Reference**: See `risk-assessment-framework.md` for security risk scoring

**Prompt**:
```
Execute comprehensive security audit on: $ARGUMENTS.

Perform:
- OWASP Top 10 analysis
- Dependency vulnerability scanning (Snyk/Trivy)
- Secrets detection (GitLeaks)
- Input validation and auth/authz review
- Cryptographic implementation assessment

Include architectural vulnerabilities from Phase 1: {phase1_architecture_context}

Check for: SQL injection, XSS, CSRF, insecure deserialization, configuration issues.

Reference: docs/comprehensive-review/risk-assessment-framework.md (security factor scoring)
```

### 2B. Performance & Scalability Analysis
- **Agent**: `full-stack-orchestration:performance-engineer`
- **Focus**: CPU/memory profiling, database query optimization, caching, async patterns
- **Analysis**: Memory leaks, resource contention, load testing bottlenecks
- **Output**: Performance metrics, bottleneck analysis, optimization recommendations

**Prompt**:
```
Conduct performance analysis and scalability assessment for: $ARGUMENTS.

Profile:
- CPU/memory hotspots and resource usage
- Database query performance and N+1 problems
- Caching strategies and connection pooling
- Asynchronous processing patterns

Consider architectural findings: {phase1_architecture_context}

Identify memory leaks, resource contention, and bottlenecks under load.
```

**Note**: Quick mode ends here. Continue for standard/deep modes.

## Phase 3: Testing & Documentation Review

**Orchestrate** test and documentation quality agents:

### 3A. Test Coverage & Quality Analysis
- **Agent**: `unit-testing:test-automator`
- **Focus**: Test coverage, test pyramid adherence, test quality metrics
- **Analysis**: Assertion density, test isolation, mock usage, flakiness
- **Output**: Coverage report, test quality metrics, testing gap analysis

**Prompt**:
```
Evaluate testing strategy and implementation for: $ARGUMENTS.

Analyze:
- Unit/integration/E2E test coverage and completeness
- Test pyramid adherence
- Test quality: assertion density, isolation, mocking, flakiness
- TDD practices compliance (if --tdd-review flag set)

Consider security and performance testing from Phase 2:
{phase2_security_context}, {phase2_performance_context}

Identify testing gaps and quality improvements.

Reference: docs/comprehensive-review/review-best-practices.md (testing checklist)
```

### 3B. Documentation & API Specification Review
- **Agent**: `code-documentation:docs-architect`
- **Focus**: Documentation completeness, API docs, ADRs, runbooks
- **Validation**: Documentation accuracy vs. actual implementation
- **Output**: Documentation coverage report, inconsistency list, improvement recommendations

**Prompt**:
```
Review documentation completeness and quality for: $ARGUMENTS.

Assess:
- Inline code documentation and comments
- API documentation (OpenAPI/Swagger)
- Architecture Decision Records (ADRs)
- README, deployment guides, runbooks

Verify documentation reflects actual implementation based on all previous findings:
{phase1_context}, {phase2_context}

Identify outdated docs, missing examples, unclear explanations.
```

## Phase 4: Best Practices & Standards Compliance

**Orchestrate** framework and DevOps best practices agents:

### 4A. Framework & Language Best Practices
- **Agent**: `framework-migration:legacy-modernizer`
- **Focus**: Modern language patterns, framework conventions, package management
- **Scope**: JavaScript/TypeScript, React, Python PEP, Java, Go idiomatic code (based on --framework)
- **Output**: Best practices compliance report, modernization recommendations

**Prompt**:
```
Verify adherence to framework and language best practices for: $ARGUMENTS.

Check (framework-specific based on --framework flag):
- Modern JavaScript/TypeScript patterns, React hooks
- Python PEP compliance, Java enterprise patterns, Go idiomatic code
- Package management and build configuration
- Environment handling and deployment practices

Synthesize all previous findings for framework-specific guidance:
{all_previous_contexts}
```

### 4B. CI/CD & DevOps Practices Review
- **Agent**: `cicd-automation:deployment-engineer`
- **Focus**: CI/CD pipeline, build automation, deployment strategies
- **Analysis**: Pipeline security, monitoring, rollback capabilities
- **Output**: Pipeline assessment, DevOps maturity evaluation, automation recommendations

**Prompt**:
```
Review CI/CD pipeline and DevOps practices for: $ARGUMENTS.

Evaluate:
- Build and test automation integration
- Deployment strategies (blue-green, canary)
- Infrastructure as Code (IaC)
- Monitoring, observability, incident response
- Pipeline security and artifact management

Consider operationalizing fixes for all critical issues: {all_critical_issues}
```

## Consolidated Report Generation

Compile all phase outputs into comprehensive, prioritized report:

### 🚨 Critical Issues (P0 - Must Fix Immediately)
- Security vulnerabilities with CVSS > 7.0
- Data loss or corruption risks
- Authentication/authorization bypasses
- Production stability threats
- Compliance violations (GDPR, PCI DSS, SOC2)

**Reference**: See `risk-assessment-framework.md` for CVSS scoring and risk levels

### ⚠️ High Priority (P1 - Fix Before Next Release)
- Performance bottlenecks impacting user experience
- Missing critical test coverage
- Architectural anti-patterns causing technical debt
- Outdated dependencies with known vulnerabilities
- Code quality issues affecting maintainability

### 📝 Medium Priority (P2 - Plan for Next Sprint)
- Non-critical performance optimizations
- Documentation gaps and inconsistencies
- Code refactoring opportunities
- Test quality improvements
- DevOps automation enhancements

### 💡 Low Priority (P3 - Track in Backlog)
- Style guide violations
- Minor code smell issues
- Nice-to-have documentation updates
- Cosmetic improvements

**Reference**: See `review-best-practices.md` for priority framework and communication guidelines

## Success Criteria

Review is successful when:
- ✅ All critical security vulnerabilities identified with remediation paths
- ✅ Performance bottlenecks profiled with optimization strategies
- ✅ Test coverage gaps mapped with priority recommendations
- ✅ Architecture risks assessed with mitigation strategies
- ✅ Documentation reflects actual implementation
- ✅ Framework best practices compliance verified
- ✅ CI/CD pipeline supports safe deployment
- ✅ Clear, actionable, prioritized feedback provided
- ✅ Metrics dashboard shows trends (deep mode)
- ✅ Team has actionable remediation plan

## Deep Mode Enhancements

When `--mode=deep` is specified, additionally provide:

1. **Metrics Dashboard**:
   - Cyclomatic complexity trends
   - Code duplication percentage
   - Test coverage evolution
   - Technical debt accumulation rate
   - Dependency vulnerability timeline

2. **Automated Remediation Suggestions**:
   - Sample code fixes for common issues
   - Refactoring scripts for detected patterns
   - Dependency update PRs (using Dependabot/Renovate patterns)

3. **Framework-Specific Deep Analysis**:
   - Performance benchmarks for framework patterns
   - Security configurations for framework
   - Best practice examples from framework docs

4. **CI/CD Integration Recommendations**:
   - Pre-commit hook templates
   - GitHub Actions workflow samples
   - Quality gate configurations

---

**Target**: `$ARGUMENTS`
**External Documentation**: `docs/comprehensive-review/` (review-best-practices.md, risk-assessment-framework.md, pr-templates-library.md)
