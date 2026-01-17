---
description: Workflow for deps-audit
triggers:
- /deps-audit
- workflow for deps audit
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



# Dependency Audit

$ARGUMENTS

## Modes

| Mode | Time | Scope |
|------|------|-------|
| Quick | 2-5min | Direct deps, Critical/High vulns, license summary |
| Standard | 5-15min | Full tree, all severities, outdated |
| Comprehensive | 15-45min | + Supply chain, bundle size, automated PRs |

## Ecosystems

NPM/Yarn, Python, Go, Rust, Ruby, Java (package.json, requirements.txt, go.mod, Cargo.toml, Gemfile, pom.xml)

## Severity

| Level | CVSS | Examples |
|-------|------|----------|
| Critical | 9.0-10.0 | RCE, auth bypass |
| High | 7.0-8.9 | XSS, SSRF, path traversal |
| Moderate | 4.0-6.9 | Info disclosure, DoS |
| Low | 0.1-3.9 | Minor issues |

Risk = CVSS × DirectMult × ExploitMult × VectorMult × PatchMult

## License Compliance

| Type | Examples | Risk |
|------|----------|------|
| Permissive | MIT, Apache, BSD | ✅ Low |
| Weak Copyleft | LGPL, MPL | ⚠️ Medium |
| Strong Copyleft | GPL, AGPL | 🔴 High |
| Unknown | - | ❌ Critical |

## Outdated Priority

| Priority | Score | Criteria |
|----------|-------|----------|
| P0 | >80 | Security fixes, critical |
| P1 | 50-80 | Major versions, >1yr old |
| P2 | 20-50 | Minor updates |
| P3 | <20 | Patch updates |

## Supply Chain (Comprehensive)

- Typosquatting (Levenshtein)
- Maintainer changes
- Suspicious patterns (obfuscation, network calls)
- Package age (<30 days)

## Remediation

```bash
npm update pkg1@^2.1.5
pip install --upgrade pkg2==3.0.1
npm test && pytest
```

## CI/CD

```yaml
name: Daily Scan
on: {schedule: [{cron: '0 0 * * *'}]}
jobs:
  scan:
    steps: [{run: npm audit --json > audit.json}]
```

## Output

```markdown
# Audit Report
## Summary
- Deps: 245 (187 direct)
- Vulns: 12 (2 Critical, 5 High)
- License Issues: 3
- Outdated: 37
- Risk: 47.2/100 ⚠️

## Actions
1. axios 0.21.1 → 0.21.4 (CVE-2021-3749)
2. lodash 4.17.15 → 4.17.21
3. Remove GPL-3.0 or change license
```

## Best Practices

1. Run daily (automated), weekly manual
2. Fix Critical/High immediately
3. Full tests after updates
4. Automate patch/minor PRs
5. Manual review major versions
6. Document deferred vulns
