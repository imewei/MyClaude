---
version: "1.0.5"
description: Safe dependency upgrade orchestration with breaking change management and security-first prioritization
argument-hint: [--security-only] [--mode quick|standard|deep] [--strategy incremental|batch]
category: framework-migration
purpose: Upgrade dependencies safely with automated testing and rollback procedures
execution_time:
  quick: "15-25 minutes"
  standard: "30-60 minutes"
  deep: "1-3 hours"
color: green
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task
external_docs:
  - dependency-strategies-guide.md
  - testing-strategies.md
  - rollback-procedures.md
agents:
  primary:
    - framework-migration:legacy-modernizer
  conditional:
    - agent: comprehensive-review:security-auditor
      trigger: argument "--security-only" OR pattern "security|vulnerability|CVE"
    - agent: unit-testing:test-automator
      trigger: pattern "test|coverage"
  orchestrated: true
---

# Dependency Upgrade Orchestrator

Safe, incremental dependency upgrades with security-first prioritization.

## Target

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope | Output |
|------|----------|-------|--------|
| `--mode=quick` | 15-25 min | Security patches only | Vulnerabilities patched |
| standard (default) | 30-60 min | Minor/patch upgrades | Dependencies updated, tests passing |
| `--mode=deep` | 1-3 hours | Major versions + migration | Major upgrades with documentation |

**Flags:** `--security-only`, `--strategy incremental|batch`, `--dry-run`, `--interactive`

---

## Phase 1: Analysis & Audit

### Vulnerability Scan

```bash
# Node.js
npm audit --json > audit-report.json

# Python
pip-audit --format json > audit-report.json
```

**Identify:** CVEs with CVSS scores, affected packages, remediation paths

### Outdated Detection

```bash
# Node.js
npm outdated --json

# Python
pip list --outdated --format=json
```

**Categorize:**
- **Patch** (1.2.3→1.2.4): Bug fixes, safe
- **Minor** (1.2.4→1.3.0): New features, backward compatible
- **Major** (1.3.0→2.0.0): Breaking changes, needs migration

🚨 **Quick Mode exits here** - deliver security vulnerability report

---

## Phase 2: Strategy & Prioritization

### Strategy Selection

```
Critical CVEs (CVSS>7.0)? → SECURITY-FIRST
>5 major upgrades? → INCREMENTAL (one at a time)
Otherwise → BATCH (group similar)
```

### Priority Order

| Priority | Type | Action |
|----------|------|--------|
| P0 | Critical security (CVSS>7.0) | Immediate |
| P1 | High security (CVSS 4-7) | Within 1 week |
| P2 | Core framework (React, Django) | Before dependencies |
| P3 | Direct dependencies | Higher impact |
| P4 | Minor/patch updates | Batch together |
| P5 | Dev dependencies | Lower priority |

---

## Phase 3: Upgrade Execution

### Backup

```bash
git add package.json package-lock.json
git commit -m "checkpoint: pre-upgrade"
git tag pre-upgrade-$(date +%Y%m%d-%H%M%S)
```

### Incremental Strategy

Upgrade one major at a time:
```bash
npm install react@17 react-dom@17
npm test
git commit -m "upgrade: React 16→17"

npm install react@18 react-dom@18
npm test
git commit -m "upgrade: React 17→18"
```

### Codemods (major upgrades)

| Framework | Codemod |
|-----------|---------|
| React 17→18 | `npx react-codemod update-react-imports` |
| Vue 2→3 | `npx @vue/compat-migration` |
| Python 2→3 | `2to3 -w src/` |

---

## Phase 4: Testing & Validation

```bash
npm test              # Unit
npm run test:e2e      # E2E
npx tsc --noEmit      # Types
```

**Performance criteria:**
- Bundle size: <+10%
- Build time: <+20%
- Latency: <+10%

🚨 **Standard Mode complete** - dependencies upgraded and validated

---

## Phase 5: Deployment (Deep Mode)

### Canary Rollout

| Day | Traffic | Action |
|-----|---------|--------|
| 1 | 5% | Monitor errors, latency |
| 2 | 25% | Check business metrics |
| 3-4 | 50%→100% | Full deployment |

**Rollback triggers:** Error rate >5%, p95 latency >2x baseline

### Documentation

Update: `CHANGELOG.md`, `README.md` if needed

---

## Phase 6: Automation (Deep Mode)

### Dependabot Setup

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

### Upgrade Cadence

| Frequency | Type |
|-----------|------|
| Weekly | Security patches (automated) |
| Monthly | Minor versions |
| Quarterly | Major versions (planned) |
| Immediate | Critical CVEs |

🎯 **Deep Mode complete** - automated dependency management established

---

## Safety Guarantees

**WILL:**
- ✅ Create backup before upgrades
- ✅ Upgrade incrementally (one major at a time)
- ✅ Run tests after each upgrade
- ✅ Provide instant rollback

**NEVER:**
- ❌ Skip security patches
- ❌ Multiple major versions at once
- ❌ Deploy without test validation
