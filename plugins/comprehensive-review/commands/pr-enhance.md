---
version: "2.1.0"
category: pr-optimization
purpose: Create high-quality pull requests with comprehensive descriptions, automated review checks, and best practices
execution_time:
  basic: "5-10 minutes"
  enhanced: "10-20 minutes"
  enterprise: "20-40 minutes"
external_docs:
  - pr-templates-library.md
  - review-best-practices.md
  - risk-assessment-framework.md
tags: [pull-request, code-review, automation, documentation, quality-assurance]
---

# Pull Request Enhancement

Generate high-quality PRs with comprehensive descriptions, automated review, and context-aware best practices.

## Requirements

$ARGUMENTS

---

## Execution Modes

| Mode | Time | Content | Use Case |
|------|------|---------|----------|
| `--mode=basic` | 5-10 min | Summary, changes, checklist | Small PRs, docs |
| enhanced (default) | 10-20 min | + Automated checks, risk assessment | Features, fixes |
| `--mode=enterprise` | 20-40 min | + Coverage, diagrams, split suggestions | Major changes |

---

## 1. Change Analysis

Run git analysis:
```bash
git diff --name-status main...HEAD
git diff --shortstat main...HEAD
git log --oneline main..HEAD
```

**Categorize files:**
| Pattern | Category |
|---------|----------|
| `.js`, `.ts`, `.py`, `.go` | source |
| `test`, `spec`, `.test.` | test |
| `.json`, `.yml`, `.yaml`, `config` | config |
| `.md`, `README`, `CHANGELOG` | docs |
| `.css`, `.scss` | styles |
| `Dockerfile`, `Makefile` | build |

---

## 2. PR Description Template

```markdown
## Summary

[1-2 sentence description]

**Impact**: X files changed (Y additions, Z deletions)
**Risk Level**: 🟢 Low | 🟡 Medium | 🟠 High | 🔴 Critical
**Review Time**: ~N minutes

## What Changed

### 🔧 Source Changes
- [status]: `filename`

### ✅ Test Changes
- [status]: `filename`

## Why These Changes

[Motivation from commit messages]

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## How Has This Been Tested?

[Testing approach]

## Breaking Changes

[If any, with migration guide]

## Checklist

### General
- [ ] Self-review completed
- [ ] No debugging code left
- [ ] No sensitive data exposed

### [Context-specific items based on changed files]
```

---

## 3. Automated Checks

**Detect common issues in diff:**

| Check | Pattern | Severity |
|-------|---------|----------|
| Console logs | `console.(log\|debug\|warn)` | warning |
| Large functions | Functions >50 lines | suggestion |
| Commented code | Large `//` or `/* */` blocks | warning |
| TODOs | `TODO`, `FIXME`, `HACK` | info |
| Hardcoded values | Literal secrets, URLs | warning |
| SQL injection | `execute(.*\+.*)` | critical |
| XSS | `innerHTML =` | critical |
| Hardcoded secrets | `password.*=.*["']` | critical |

---

## 4. Risk Assessment

**Calculate risk score (0-10):**

| Factor | Low (0-2) | Medium (3-5) | High (6-8) | Critical (9-10) |
|--------|-----------|--------------|------------|-----------------|
| Size | <100 lines | 100-400 | 400-1000 | >1000 |
| Files | <5 | 5-15 | 15-30 | >30 |
| Test coverage | Increases | Stable | Slight decrease | Major decrease |
| Dependencies | None | Minor updates | Major updates | New core deps |
| Security | None | Low risk | Auth/data changes | Crypto/secrets |

**Overall risk** = weighted average

**Mitigation strategies** based on risk level.

---

## 5. Context-Aware Checklist

Generate checklist items based on changed file types:

**If source files changed:**
- [ ] No code duplication
- [ ] Functions focused and small
- [ ] Error handling comprehensive

**If test files changed:**
- [ ] New code covered by tests
- [ ] Edge cases tested
- [ ] No flaky tests

**If config files changed:**
- [ ] No hardcoded values
- [ ] Environment variables documented
- [ ] Backwards compatible

**If security-related:**
- [ ] Input validation
- [ ] Auth/authz correct
- [ ] No sensitive data in logs

---

## 6. Large PR Detection

If >20 files or >1000 lines changed:

```
⚠️ Large PR Detected

This PR changes X files with Y total changes.
Consider splitting for easier review.

Suggested Splits:
1. [Feature area] - N files
2. [Feature area] - N files

How to Split:
git checkout -b feature/part-1
git cherry-pick <commits>
```

---

## 7. Template Selection

Auto-select based on changes:

| Change Type | Template |
|-------------|----------|
| Security fixes | Security (CVSS, vulnerability details) |
| Performance | Performance (benchmarks, profiling) |
| Bug fix | Bug Fix (root cause, reproduction) |
| Refactoring | Refactor (motivation, metrics) |
| New features | Feature (user stories, acceptance) |
| Docs only | Documentation (audience, accuracy) |
| Dependencies | Dependency (vulnerabilities, breaking) |
| Config | Configuration (env vars, rollback) |

---

## Output by Mode

### Basic
1. PR Summary
2. Detailed Description
3. Review Checklist

### Enhanced (default)
+ Automated Review Findings
+ Risk Assessment
+ Size Recommendations

### Enterprise
+ Test Coverage Report (before/after)
+ Architecture Diagrams (Mermaid)
+ PR Template Selection
+ Review Response Templates

---

**Focus**: Create PRs that are easy to review with all necessary context.
