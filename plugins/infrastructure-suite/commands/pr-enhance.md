---
version: "2.1.0"
description: Create high-quality pull requests with comprehensive descriptions, automated review, and best practices
allowed-tools: Bash(git:*), Read, Grep
argument-hint: [--mode=basic|enhanced|enterprise]
color: green
agents:
  primary:
    - quality-specialist
  conditional:
    - agent: quality-specialist
      trigger: pattern "security|auth|credentials"
  orchestrated: false
---

# Pull Request Enhancement

Generate high-quality PRs with comprehensive descriptions, automated review checks, and context-aware best practices.

## Context

$ARGUMENTS

---

## Mode Selection

| Mode | Time | Content |
|------|------|---------|
| `--mode=basic` | 5-10 min | Summary, changes, checklist |
| enhanced (default) | 10-20 min | + Automated checks, risk assessment |
| `--mode=enterprise` | 20-40 min | + Coverage, diagrams, split suggestions |

---

## 1. Analyze Changes

```bash
git diff --name-status main...HEAD
git diff --shortstat main...HEAD
git log --oneline main..HEAD
```

**File Categories:**

| Pattern | Category | Icon |
|---------|----------|------|
| `.js`, `.ts`, `.py`, `.go`, `.rs` | source | 🔧 |
| `test`, `spec`, `.test.` | test | ✅ |
| `.json`, `.yml`, `config` | config | ⚙️ |
| `.md`, `README`, `CHANGELOG` | docs | 📝 |
| `.css`, `.scss` | styles | 🎨 |
| `Dockerfile`, `Makefile` | build | 🏗️ |

---

## 2. Generate PR Description

```markdown
## Summary

[1-2 sentence purpose]

**Impact**: X files (Y+, Z-)
**Risk**: 🟢 Low | 🟡 Medium | 🟠 High | 🔴 Critical
**Review Time**: ~N min

## What Changed

### 🔧 Source Changes
- [status]: `filename`

### ✅ Test Changes
- [status]: `filename`

## Why These Changes

[Extract from commit messages]

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## How Tested?

[Testing approach]

## Breaking Changes

[If any, with migration guide]

## Checklist

[Context-aware items]
```

---

## 3. Automated Checks

Scan diff for issues:

| Check | Pattern | Severity |
|-------|---------|----------|
| Console logs | `console.(log\|debug\|warn)` | warning |
| Large functions | >50 lines | suggestion |
| TODOs | `TODO\|FIXME\|HACK` | info |
| Hardcoded secrets | `password.*=.*["']` | critical |
| SQL injection | `execute(.*\+.*)` | critical |
| XSS | `innerHTML =` | critical |

---

## 4. Risk Assessment

**Calculate risk score (0-10):**

| Factor | Low (0-2) | Medium (3-5) | High (6-8) | Critical (9-10) |
|--------|-----------|--------------|------------|-----------------|
| Size | <100 lines | 100-400 | 400-1000 | >1000 |
| Files | <5 | 5-15 | 15-30 | >30 |
| Test coverage | Increases | Stable | Decreases | Major decrease |
| Dependencies | None | Minor | Major | New core deps |
| Security | None | Low risk | Auth changes | Crypto/secrets |

---

## 5. Context-Aware Checklist

**Generate based on changed files:**

**Source files:**
- [ ] No code duplication
- [ ] Functions small and focused
- [ ] Error handling comprehensive

**Test files:**
- [ ] New code covered
- [ ] Edge cases tested
- [ ] No flaky tests

**Config files:**
- [ ] No hardcoded values
- [ ] Env vars documented
- [ ] Backwards compatible

**Security-related:**
- [ ] Input validation
- [ ] Auth/authz correct
- [ ] No sensitive data in logs

---

## 6. Large PR Detection

If >20 files or >1000 lines:

```
⚠️ Large PR Detected

This PR changes X files with Y total lines.
Consider splitting for easier review.

Suggested Splits:
1. [Feature area] - N files
2. [Feature area] - N files
```

---

## 7. PR Templates

Auto-select based on changes:

| Change Type | Template Focus |
|-------------|----------------|
| Security fixes | CVSS, vulnerability details |
| Performance | Benchmarks, profiling results |
| Bug fix | Root cause, reproduction steps |
| Refactoring | Motivation, metrics |
| New features | User stories, acceptance criteria |
| Docs only | Audience, accuracy checklist |
| Dependencies | Vulnerabilities, breaking changes |
| Config | Env vars, rollback plan |

---

## Output Format

**Basic:** Summary, Description, Checklist
**Enhanced:** + Automated findings, Risk assessment, Size recommendations
**Enterprise:** + Coverage report, Architecture diagrams, Response templates

---

**Focus**: Create PRs that are easy to review with all necessary context.
