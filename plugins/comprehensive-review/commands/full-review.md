---
version: "1.0.6"
category: code-review
purpose: Orchestrate comprehensive multi-dimensional code review using specialized agents
execution_modes:
  quick: "10-20 minutes"
  standard: "25-40 minutes"
  deep: "45-75 minutes"
external_docs:
  - review-best-practices.md
  - risk-assessment-framework.md
  - pr-templates-library.md
tags: [code-review, multi-agent, orchestration, quality-assurance, security-audit]
---

# Comprehensive Multi-Agent Code Review

Orchestrate exhaustive code review by coordinating specialized agents in sequential phases.

## Target

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Phases | Scope |
|------|----------|--------|-------|
| Quick | 10-20 min | 1-2 | Core quality + security, critical/high issues only |
| Standard (default) | 25-40 min | 1-4 | All phases, all priority levels |
| Deep | 45-75 min | 1-4 + extras | + metrics dashboard, automated remediation, CI/CD |

---

## Configuration Options

| Flag | Purpose |
|------|---------|
| `--mode=quick\|standard\|deep` | Execution mode |
| `--security-focus` | Prioritize security/OWASP |
| `--performance-critical` | Emphasize performance |
| `--tdd-review` | Include TDD compliance |
| `--strict-mode` | Fail on critical issues |
| `--framework=<name>` | Framework-specific best practices |
| `--metrics-report` | Generate metrics dashboard |

---

## Phase 1: Code Quality & Architecture

**Agents:** code-reviewer, architect-review (parallel)

### 1A. Code Quality Analysis
| Focus | Deliverable |
|-------|-------------|
| Complexity, maintainability | Quality metrics |
| Technical debt, duplication | Code smell inventory |
| SOLID principles | Refactoring recommendations |
| Clean Code, naming | Style compliance |

### 1B. Architecture & Design Review
| Focus | Deliverable |
|-------|-------------|
| Design patterns | Architecture assessment |
| Microservices boundaries | API design analysis |
| Circular dependencies | Structural recommendations |
| DDD adherence | Domain model evaluation |

**Reference:** [review-best-practices.md](../docs/comprehensive-review/review-best-practices.md)

---

## Phase 2: Security & Performance

**Agents:** security-auditor, performance-engineer (incorporates Phase 1)

### 2A. Security Vulnerability Assessment
| Focus | Tools |
|-------|-------|
| OWASP Top 10 | Snyk, Trivy, GitLeaks |
| Dependency vulnerabilities | CVE scanning |
| Secrets detection | Code review |
| Auth/authz | Manual audit |

### 2B. Performance & Scalability
| Focus | Analysis |
|-------|----------|
| CPU/memory profiling | Hotspot identification |
| Database query optimization | N+1, indexing |
| Caching strategies | Connection pooling |
| Async patterns | Load testing bottlenecks |

**Reference:** [risk-assessment-framework.md](../docs/comprehensive-review/risk-assessment-framework.md)

**Quick mode ends here.**

---

## Phase 3: Testing & Documentation

**Agents:** test-automator, docs-architect

### 3A. Test Coverage & Quality
| Focus | Metrics |
|-------|---------|
| Unit/integration/E2E coverage | % coverage |
| Test pyramid adherence | Distribution |
| Assertion density, isolation | Quality score |
| TDD compliance | If --tdd-review |

### 3B. Documentation Review
| Focus | Validation |
|-------|------------|
| Inline documentation | Completeness |
| API docs (OpenAPI) | Accuracy |
| ADRs, runbooks | Currency |
| README, deployment guides | Clarity |

---

## Phase 4: Best Practices & Standards

**Agents:** legacy-modernizer, deployment-engineer

### 4A. Framework Best Practices
| Framework | Checks |
|-----------|--------|
| JavaScript/TypeScript | Modern patterns, React hooks |
| Python | PEP compliance |
| Java/Go | Enterprise/idiomatic patterns |
| All | Package management, build config |

### 4B. CI/CD & DevOps Practices
| Focus | Evaluation |
|-------|------------|
| Pipeline security | Artifact management |
| Deployment strategies | Blue-green, canary |
| IaC | Terraform, K8s |
| Monitoring, observability | Incident response |

---

## Consolidated Report Format

### Priority Levels

| Priority | Criteria | Examples |
|----------|----------|----------|
| P0 Critical | Must fix immediately | CVSS >7.0, data loss, auth bypass |
| P1 High | Before next release | Performance bottlenecks, missing coverage |
| P2 Medium | Next sprint | Refactoring, doc gaps |
| P3 Low | Backlog | Style violations, cosmetic |

---

## Deep Mode Enhancements

| Enhancement | Content |
|-------------|---------|
| Metrics Dashboard | Complexity trends, duplication %, coverage evolution |
| Automated Remediation | Sample fixes, refactoring scripts, dependency PRs |
| Framework Deep Analysis | Benchmarks, security configs, best practice examples |
| CI/CD Integration | Pre-commit hooks, Actions workflows, quality gates |

---

## Success Criteria

- [ ] All critical security vulnerabilities identified with remediation
- [ ] Performance bottlenecks profiled with optimization strategies
- [ ] Test coverage gaps mapped with priority recommendations
- [ ] Architecture risks assessed with mitigation strategies
- [ ] Documentation reflects actual implementation
- [ ] Framework best practices compliance verified
- [ ] CI/CD pipeline supports safe deployment
- [ ] Clear, actionable, prioritized feedback provided

---

## External Documentation

| Document | Purpose |
|----------|---------|
| [review-best-practices.md](../docs/comprehensive-review/review-best-practices.md) | Code smell patterns, communication guidelines |
| [risk-assessment-framework.md](../docs/comprehensive-review/risk-assessment-framework.md) | CVSS scoring, risk levels |
| [pr-templates-library.md](../docs/comprehensive-review/pr-templates-library.md) | PR templates and standards |
