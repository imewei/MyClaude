---
version: "1.0.6"
category: code-review
purpose: Multi-dimensional code review with specialized agents
execution_modes:
  quick: "10-20min"
  standard: "25-40min"
  deep: "45-75min"
external_docs: [review-best-practices.md, risk-assessment-framework.md, pr-templates-library.md]
tags: [code-review, multi-agent, quality, security]
---

# Multi-Agent Code Review

$ARGUMENTS

## Modes

| Mode | Time | Phases | Scope |
|------|------|--------|-------|
| Quick | 10-20min | 1-2 | Core quality + security, critical/high only |
| Standard | 25-40min | 1-4 | All phases, all priorities |
| Deep | 45-75min | 1-4+ | + Metrics, automation, CI/CD |

## Flags

`--mode=quick|standard|deep`, `--security-focus`, `--performance-critical`, `--tdd-review`, `--strict-mode`, `--framework=<name>`, `--metrics-report`

## Phase 1: Quality & Architecture

**Code Quality**: Complexity, debt, SOLID, Clean Code
**Architecture**: Patterns, boundaries, dependencies, DDD

## Phase 2: Security & Performance

**Security**: OWASP Top 10, CVE scan, secrets, auth (Snyk, Trivy, GitLeaks)
**Performance**: CPU/memory profiling, N+1, caching, async

🚨 **Quick mode exits**

## Phase 3: Testing & Docs

**Testing**: Coverage (unit/int/E2E), pyramid, isolation, TDD
**Docs**: Inline, API (OpenAPI), ADRs, README, guides

## Phase 4: Standards & DevOps

**Framework Best Practices**: JS/TS, Python PEP, Java/Go idioms
**CI/CD**: Pipeline security, deployment (blue-green, canary), IaC, monitoring

## Priority Levels

| Priority | Criteria | Examples |
|----------|----------|----------|
| P0 Critical | Fix immediately | CVSS >7, data loss, auth bypass |
| P1 High | Before release | Perf bottlenecks, missing tests |
| P2 Medium | Next sprint | Refactoring, doc gaps |
| P3 Low | Backlog | Style, cosmetic |

## Deep Mode

- Metrics dashboard (complexity, duplication %, coverage trends)
- Automated remediation (fixes, scripts, dep PRs)
- Framework deep analysis (benchmarks, security configs)
- CI/CD integration (hooks, workflows, quality gates)

## Success Criteria

- [ ] Critical vulns identified with remediation
- [ ] Perf bottlenecks profiled with strategies
- [ ] Test gaps mapped with priorities
- [ ] Architecture risks assessed with mitigation
- [ ] Docs reflect implementation
- [ ] Framework compliance verified
- [ ] CI/CD supports safe deployment
- [ ] Clear, actionable, prioritized feedback
