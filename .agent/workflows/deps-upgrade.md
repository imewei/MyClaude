---
description: Safe dependency upgrade orchestration with breaking change management
  and security-first prioritization
triggers:
- /deps-upgrade
- workflow for deps upgrade
version: 1.0.7
argument-hint: '[--security-only] [--mode quick|standard|deep] [--strategy incremental|batch]'
category: framework-migration
execution_time:
  quick: '15-25m: Security patches only'
  standard: '30-60m: Minor/patch upgrades'
  deep: '1-3h: Major versions + automation'
color: green
allowed-tools: [Bash, Edit, Read, Task]
external_docs:
- dependency-strategies-guide.md
- testing-strategies.md
- rollback-procedures.md
---


# Dependency Upgrade

$ARGUMENTS

**Flags:** `--security-only`, `--strategy incremental|batch`, `--dry-run`, `--interactive`

## Phase 1: Analysis

**Scan vulnerabilities:**
```bash
npm audit --json > audit-report.json  # Node.js
pip-audit --format json               # Python
```

**Detect outdated:**
- Patch (1.2.3→1.2.4): Bug fixes, safe
- Minor (1.2.4→1.3.0): Features, backward compatible
- Major (1.3.0→2.0.0): Breaking changes

🚨 **Quick mode:** Deliver vulnerability report, exit

## Phase 2: Strategy & Prioritization

**Strategy:**
- Critical CVEs (CVSS>7.0) → Security-first
- >5 major upgrades → Incremental
- Otherwise → Batch

**Priority:**
- P0: Critical security (CVSS>7.0) → Immediate
- P1: High security (4-7) → 1 week
- P2: Core frameworks → Before dependencies
- P3: Direct deps → Higher impact
- P4: Minor/patch → Batch
- P5: Dev deps → Lower priority

## Phase 3: Execution

**Backup:**
```bash
git add package.json package-lock.json
git commit -m "checkpoint: pre-upgrade"
git tag pre-upgrade-$(date +%Y%m%d-%H%M%S)
```

**Incremental (one major at a time):**
```bash
npm install react@17 react-dom@17 && npm test
git commit -m "upgrade: React 16→17"
npm install react@18 react-dom@18 && npm test
git commit -m "upgrade: React 17→18"
```

**Codemods:**
- React 17→18: `npx react-codemod update-react-imports`
- Vue 2→3: `npx @vue/compat-migration`
- Python 2→3: `2to3 -w src/`

## Phase 4: Validation

```bash
npm test && npm run test:e2e && npx tsc --noEmit
```

**Performance limits:**
- Bundle size: <+10%
- Build time: <+20%
- Latency: <+10%

🚨 **Standard mode complete**

## Phase 5: Deployment (Deep)

**Canary rollout:**
- Day 1: 5% → Monitor errors/latency
- Day 2: 25% → Business metrics
- Day 3-4: 50%→100%

**Rollback triggers:** Error rate >5%, p95 latency >2x baseline

Update: `CHANGELOG.md`, `README.md`

## Phase 6: Automation (Deep)

**Dependabot:**
```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    schedule: {interval: "weekly"}
    open-pull-requests-limit: 5
```

**Cadence:** Weekly (security), Monthly (minor), Quarterly (major), Immediate (critical CVEs)

## Safety

- Backup before upgrades
- Incremental (one major at a time)
- Test after each upgrade
- Instant rollback available
