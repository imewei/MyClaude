---
version: "1.0.6"
category: codebase-cleanup
purpose: Comprehensive dependency security scanning and vulnerability analysis
execution_time:
  quick: 2-5 minutes
  standard: 5-15 minutes
  comprehensive: 15-45 minutes
external_docs:
  - dependency-security-guide.md
  - vulnerability-analysis-framework.md
  - automation-integration.md
---

# Dependency Audit and Security Analysis

Analyze project dependencies for vulnerabilities, licensing issues, and outdated packages.

## Target

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 2-5 min | Direct deps only, Critical/High vulns, license summary |
| Standard (default) | 5-15 min | Full dependency tree, all severities, outdated detection |
| Comprehensive | 15-45 min | + Supply chain analysis, bundle size, automated PRs |

---

## Phase 1: Dependency Discovery

### Supported Ecosystems

| Ecosystem | Files Scanned |
|-----------|---------------|
| NPM/Yarn | `package.json`, `package-lock.json`, `yarn.lock` |
| Python | `requirements.txt`, `pyproject.toml`, `poetry.lock` |
| Go | `go.mod`, `go.sum` |
| Rust | `Cargo.toml`, `Cargo.lock` |
| Ruby | `Gemfile`, `Gemfile.lock` |
| Java | `pom.xml`, `build.gradle` |

---

## Phase 2: Vulnerability Scanning

### Data Sources
- NPM Advisory Database
- PyPI Safety DB
- OSV (Open Source Vulnerabilities)
- GitHub Security Advisories

### Severity Classification

| Severity | CVSS | Examples |
|----------|------|----------|
| Critical | 9.0-10.0 | RCE, auth bypass |
| High | 7.0-8.9 | XSS, SSRF, path traversal |
| Moderate | 4.0-6.9 | Info disclosure, DoS |
| Low | 0.1-3.9 | Minor issues |

### Risk Scoring
```
Risk = CVSS × DirectMult × ExploitMult × VectorMult × PatchMult
```

---

## Phase 3: License Compliance

| License Type | Examples | Risk |
|--------------|----------|------|
| Permissive | MIT, Apache-2.0, BSD | ✅ Low |
| Weak Copyleft | LGPL, MPL | ⚠️ Medium |
| Strong Copyleft | GPL, AGPL | 🔴 High (requires source) |
| Unknown | - | ❌ Critical |

### Compatibility Check
- Verify project license compatibility
- Flag GPL in MIT/Apache projects
- Identify unknown/missing licenses

---

## Phase 4: Outdated Dependencies

### Priority Scoring

| Priority | Score | Criteria |
|----------|-------|----------|
| P0 | >80 | Security fixes, critical updates |
| P1 | 50-80 | Major versions, >1 year old |
| P2 | 20-50 | Minor updates |
| P3 | <20 | Patch updates |

---

## Phase 5: Supply Chain Security (Comprehensive)

| Check | Purpose |
|-------|---------|
| Typosquatting | Compare against popular packages (Levenshtein) |
| Maintainer Changes | Track recent additions/removals |
| Suspicious Patterns | Obfuscated code, network calls |
| Package Age | Flag <30 days old |

---

## Phase 6: Bundle Size Analysis (Comprehensive)

### Metrics
- Total bundle size (raw + gzipped)
- Per-package contribution
- Tree-shaking compatibility

### Recommendations
- Replace large packages with lighter alternatives
- Use lazy loading for heavy dependencies
- Remove unused dependencies

---

## Phase 7: Automated Remediation

### Update Script
```bash
# Critical security updates
npm update package1@^2.1.5
pip install --upgrade package2==3.0.1

# Verify
npm test && pytest
```

### PR Template
```markdown
## 🔒 Security Dependency Update
Updates {count} dependencies for {critical} critical vulns.

| Package | Current | Updated | CVE |
|---------|---------|---------|-----|
```

---

## Continuous Monitoring

### GitHub Actions
```yaml
name: Daily Dependency Scan
on:
  schedule:
    - cron: '0 0 * * *'
jobs:
  security-scan:
    steps:
      - run: npm audit --json > audit.json
```

---

## Output Format

```markdown
# Dependency Audit Report

## 📊 Executive Summary
- Total Dependencies: 245 (187 direct)
- Vulnerabilities: 12 (2 Critical, 5 High)
- License Issues: 3
- Outdated: 37
- Risk Score: 47.2/100 ⚠️

## 🚨 Immediate Actions
1. Update axios 0.21.1 → 0.21.4 (CVE-2021-3749)
2. Update lodash 4.17.15 → 4.17.21
3. Remove GPL-3.0 package or change license
```

---

## Success Criteria

- ✅ All dependencies scanned
- ✅ Vulnerabilities categorized by severity
- ✅ License compliance verified
- ✅ Outdated packages prioritized
- ✅ Remediation plan provided
- ✅ (Comprehensive) Supply chain analyzed

---

## Best Practices

1. **Run daily**: Automated scans, weekly manual review
2. **Prioritize**: Fix Critical/High immediately
3. **Test**: Full test suite after updates
4. **Automate**: Use automated PRs for patch/minor
5. **Review**: Manual review for major versions
6. **Document**: Record deferred vulnerabilities
