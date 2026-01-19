---
version: "2.1.0"
description: Unified code review and pull request management
argument-hint: <action> [target] [options]
category: quality-suite
execution_time:
  quick: "5-20 minutes"
  standard: "20-40 minutes"
  deep: "45-75 minutes"
color: cyan
allowed-tools: [Bash, Read, Write, Edit, Task, Glob, Grep]
external_docs:
  - review-best-practices.md
  - risk-assessment-framework.md
  - pr-templates-library.md
tags: [code-review, pull-request, quality, security, multi-agent]
---

# Code Review & PR Management

$ARGUMENTS

## Actions

| Action | Description |
|--------|-------------|
| `code` | Multi-dimensional code review with specialized agents |
| `pr` | Generate high-quality PR with automated checks |
| `enhance` | Enhance existing PR description and checklist |

**Examples:**
```bash
/review code src/ --security-focus
/review pr --mode enhanced
/review enhance #123 --mode enterprise
```

## Options

**All actions:**
- `--mode <depth>`: quick, standard (default), deep/enterprise

**Code review:**
- `--security-focus`: Prioritize security analysis
- `--performance-critical`: Focus on performance bottlenecks
- `--tdd-review`: Evaluate test-first discipline
- `--strict-mode`: No tolerance for P1+ issues
- `--framework <name>`: Framework-specific best practices
- `--metrics-report`: Generate metrics dashboard

**PR actions:**
- `--risk-only`: Only show risk assessment
- `--template <type>`: Force specific template (bug, feature, security, etc.)

---

# Action: Code Review

Multi-agent code review with quality, security, performance, and documentation analysis.

## Modes

| Mode | Time | Phases | Scope |
|------|------|--------|-------|
| Quick | 10-20min | 1-2 | Core quality + security, critical/high only |
| Standard | 25-40min | 1-4 | All phases, all priorities |
| Deep | 45-75min | 1-4+ | + Metrics, automation, CI/CD |

## Phase 1: Quality & Architecture

**Code Quality:**
- Complexity analysis (cyclomatic, cognitive)
- Technical debt identification
- SOLID principles adherence
- Clean Code evaluation

**Architecture:**
- Design pattern usage
- Module boundaries
- Dependency analysis
- DDD alignment

## Phase 2: Security & Performance

**Security (OWASP Top 10):**
- CVE vulnerability scan
- Secrets detection (GitLeaks)
- Authentication review
- Input validation
- Tools: Snyk, Trivy, GitLeaks

**Performance:**
- CPU/memory profiling
- N+1 query detection
- Caching opportunities
- Async optimization

**Quick mode exits here**

## Phase 3: Testing & Documentation

**Testing:**
- Coverage analysis (unit/integration/E2E)
- Test pyramid adherence
- Test isolation review
- TDD evaluation

**Documentation:**
- Inline documentation quality
- API docs (OpenAPI compliance)
- ADRs (Architecture Decision Records)
- README and guides

## Phase 4: Standards & DevOps

**Framework Best Practices:**
- JS/TS idioms
- Python PEP compliance
- Java/Go conventions

**CI/CD Review:**
- Pipeline security
- Deployment strategy (blue-green, canary)
- IaC quality
- Monitoring coverage

## Priority Levels

| Priority | Criteria | Examples |
|----------|----------|----------|
| P0 Critical | Fix immediately | CVSS >7, data loss, auth bypass |
| P1 High | Before release | Perf bottlenecks, missing tests |
| P2 Medium | Next sprint | Refactoring, doc gaps |
| P3 Low | Backlog | Style, cosmetic |

## Deep Mode Additions

- Metrics dashboard (complexity, duplication %, coverage trends)
- Automated remediation suggestions
- Framework deep analysis with benchmarks
- CI/CD integration recommendations

---

# Action: PR Enhancement

Generate high-quality PRs with comprehensive descriptions, automated review, and context-aware best practices.

## Modes

| Mode | Time | Content | Use Case |
|------|------|---------|----------|
| Basic | 5-10 min | Summary, changes, checklist | Small PRs, docs |
| Enhanced | 10-20 min | + Automated checks, risk | Features, fixes |
| Enterprise | 20-40 min | + Coverage, diagrams, splits | Major changes |

## Step 1: Change Analysis

```bash
git diff --name-status main...HEAD
git diff --shortstat main...HEAD
git log --oneline main..HEAD
```

**File Categorization:**
| Pattern | Category |
|---------|----------|
| `.js`, `.ts`, `.py`, `.go` | source |
| `test`, `spec`, `.test.` | test |
| `.json`, `.yml`, `.yaml` | config |
| `.md`, `README` | docs |

## Step 2: Automated Checks

| Check | Pattern | Severity |
|-------|---------|----------|
| Console logs | `console.(log\|debug)` | warning |
| Large functions | >50 lines | suggestion |
| Commented code | Large blocks | warning |
| TODOs | `TODO`, `FIXME` | info |
| Hardcoded values | Literal secrets | warning |
| SQL injection | `execute(.*\+.*)` | critical |
| XSS | `innerHTML =` | critical |
| Hardcoded secrets | `password.*=` | critical |

## Step 3: Risk Assessment

**Scoring (0-10):**
| Factor | Low (0-2) | Medium (3-5) | High (6-8) | Critical (9-10) |
|--------|-----------|--------------|------------|-----------------|
| Size | <100 lines | 100-400 | 400-1000 | >1000 |
| Files | <5 | 5-15 | 15-30 | >30 |
| Test coverage | Increases | Stable | Decrease | Major decrease |
| Dependencies | None | Minor | Major | New core deps |
| Security | None | Low risk | Auth/data | Crypto/secrets |

## Step 4: Large PR Detection

If >20 files or >1000 lines:
- Suggest splitting by feature area
- Provide cherry-pick commands
- Recommend split boundaries

## Step 5: Template Selection

| Change Type | Template Focus |
|-------------|----------------|
| Security fixes | CVSS, vulnerability details |
| Performance | Benchmarks, profiling |
| Bug fix | Root cause, reproduction |
| Refactoring | Motivation, metrics |
| New features | User stories, acceptance |
| Docs only | Audience, accuracy |
| Dependencies | Vulnerabilities, breaking |

## Output Format

```markdown
## Summary
[1-2 sentence description]

**Impact**: X files (Y additions, Z deletions)
**Risk Level**: 🟢 Low | 🟡 Medium | 🟠 High | 🔴 Critical
**Review Time**: ~N minutes

## What Changed
### 🔧 Source Changes
- [status]: `filename`

### ✅ Test Changes
- [status]: `filename`

## Why These Changes
[Motivation from commit messages]

## Checklist
- [ ] Self-review completed
- [ ] No debugging code left
- [ ] No sensitive data exposed
[Context-specific items based on changes]
```

---

## Success Criteria

**Code Review:**
- [ ] Critical vulns identified with remediation
- [ ] Perf bottlenecks profiled with strategies
- [ ] Test gaps mapped with priorities
- [ ] Architecture risks assessed
- [ ] Clear, actionable, prioritized feedback

**PR Enhancement:**
- [ ] Complete description with context
- [ ] All automated checks passed
- [ ] Risk level accurately assessed
- [ ] Appropriate checklist generated
- [ ] Split suggestions if needed
