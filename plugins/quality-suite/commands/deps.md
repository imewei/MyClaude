---
version: "2.2.1"
description: Unified dependency management - security auditing, vulnerability scanning, and safe upgrades
argument-hint: "<action> [--mode quick|standard|deep] [--security-only] [--strategy incremental|batch]"
category: quality-suite
execution-modes:
  audit-quick: "2-5min"
  audit-standard: "5-15min"
  upgrade-quick: "15-25min"
  upgrade-standard: "30-60min"
  upgrade-deep: "1-3h"
color: green
allowed-tools: [Bash, Edit, Read, Task, Bash(uv:*)]
agents:
  primary:
    - quality-specialist
  conditional:
    - agent: sre-expert
      trigger: argument "--deep" OR critical vulnerabilities found
---

# Dependency Management

$ARGUMENTS

## Actions

| Action | Description | Modes |
|--------|-------------|-------|
| `audit` | Security scanning, vulnerability analysis, license compliance | quick, standard, comprehensive |
| `upgrade` | Safe dependency upgrades with breaking change management | quick, standard, deep |
| `report` | Combined audit + upgrade recommendations | standard |

**Usage:**
```bash
/deps audit                      # Standard security scan
/deps audit --mode=quick         # Critical/High vulns only
/deps upgrade --security-only    # Security patches only
/deps upgrade --mode=deep        # Full upgrade with automation
/deps report                     # Combined analysis
```

---

## Action: Audit

### Modes

| Mode | Time | Scope |
|------|------|-------|
| Quick | 2-5min | Direct deps, Critical/High vulns, license summary |
| Standard | 5-15min | Full tree, all severities, outdated |
| Comprehensive | 15-45min | + Supply chain, bundle size, automated PRs |

### Ecosystems

NPM/Yarn, Python, Go, Rust, Ruby, Java (package.json, requirements.txt, go.mod, Cargo.toml, Gemfile, pom.xml)

### Severity Scoring

| Level | CVSS | Examples |
|-------|------|----------|
| Critical | 9.0-10.0 | RCE, auth bypass |
| High | 7.0-8.9 | XSS, SSRF, path traversal |
| Moderate | 4.0-6.9 | Info disclosure, DoS |
| Low | 0.1-3.9 | Minor issues |

**Risk Formula:** `Risk = CVSS × DirectMult × ExploitMult × VectorMult × PatchMult`

### License Compliance

| Type | Examples | Risk |
|------|----------|------|
| Permissive | MIT, Apache, BSD | ✅ Low |
| Weak Copyleft | LGPL, MPL | ⚠️ Medium |
| Strong Copyleft | GPL, AGPL | 🔴 High |
| Unknown | - | ❌ Critical |

### Supply Chain Analysis (Comprehensive Mode)

- Typosquatting detection (Levenshtein distance)
- Maintainer changes
- Suspicious patterns (obfuscation, network calls)
- Package age (<30 days = risky)

---

## Action: Upgrade

### Modes

| Mode | Time | Scope |
|------|------|-------|
| Quick | 15-25min | Security patches only |
| Standard | 30-60min | Minor/patch upgrades |
| Deep | 1-3h | Major versions + automation setup |

### Flags

- `--security-only` - Only security-related upgrades
- `--strategy incremental|batch` - Upgrade strategy
- `--dry-run` - Preview without changes
- `--interactive` - Confirm each upgrade

### Phase 1: Analysis

```bash
npm audit --json > audit-report.json  # Node.js
pip-audit --format json               # Python
cargo audit                           # Rust
```

**Version Detection:**
- Patch (1.2.3→1.2.4): Bug fixes, safe
- Minor (1.2.4→1.3.0): Features, backward compatible
- Major (1.3.0→2.0.0): Breaking changes

🚨 **Quick mode:** Deliver vulnerability report and exit

### Phase 2: Strategy & Prioritization

**Priority:**
- P0: Critical security (CVSS>7.0) → Immediate
- P1: High security (4-7) → 1 week
- P2: Core frameworks → Before dependencies
- P3: Direct deps → Higher impact
- P4: Minor/patch → Batch
- P5: Dev deps → Lower priority

### Phase 3: Execution

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

### Phase 4: Validation

```bash
npm test && npm run test:e2e && npx tsc --noEmit
```

**Performance limits:**
- Bundle size: <+10%
- Build time: <+20%
- Latency: <+10%

🚨 **Standard mode complete**

### Phase 5: Deployment (Deep Mode)

**Canary rollout:**
- Day 1: 5% → Monitor errors/latency
- Day 2: 25% → Business metrics
- Day 3-4: 50%→100%

**Rollback triggers:** Error rate >5%, p95 latency >2x baseline

### Phase 6: Automation (Deep Mode)

**Dependabot:**
```yaml
version: "2.2.1"
updates:
  - package-ecosystem: "npm"
    schedule: {interval: "weekly"}
    open-pull-requests-limit: 5
```

**Cadence:** Weekly (security), Monthly (minor), Quarterly (major), Immediate (critical CVEs)

---

## Action: Report

Combines audit + upgrade analysis into a single report:

```markdown
# Dependency Report

## Security Summary
- Total Dependencies: 245 (187 direct)
- Vulnerabilities: 12 (2 Critical, 5 High)
- License Issues: 3
- Outdated: 37
- Risk Score: 47.2/100 ⚠️

## Recommended Actions
1. **Immediate:** axios 0.21.1 → 0.21.4 (CVE-2021-3749)
2. **This Week:** lodash 4.17.15 → 4.17.21
3. **Review:** Remove GPL-3.0 or change license

## Upgrade Plan
| Priority | Package | Current | Target | Type |
|----------|---------|---------|--------|------|
| P0 | axios | 0.21.1 | 0.21.4 | Security |
| P1 | lodash | 4.17.15 | 4.17.21 | Security |
| P2 | react | 17.0.2 | 18.2.0 | Major |
```

---

## Best Practices

1. Run audits daily (automated), weekly manual review
2. Fix Critical/High vulnerabilities immediately
3. Full test suite after any upgrades
4. Automate patch/minor PRs with Dependabot
5. Manual review for major version upgrades
6. Document deferred vulnerabilities with justification

## Safety

- Always backup before upgrades
- Use incremental strategy for major versions
- Test after each upgrade step
- Keep rollback available at all times
