---
name: security-ci-template
version: "1.0.7"
maturity: "5-Expert"
specialization: CI/CD Security Scanning
description: Security scanning and lock file validation templates for CI/CD pipelines. Use when implementing SAST/DAST scanning, dependency vulnerability checks, lock file validation, or automated security gates in GitHub Actions or GitLab CI.
---

# Security CI Template

Security scanning and lock file validation for CI/CD pipelines.

---

## Security Tool Selection

| Category | Tools | Use Case |
|----------|-------|----------|
| Dependency Scan | Safety, Snyk, npm audit | Vulnerability detection |
| SAST | Bandit, Semgrep, CodeQL | Static code analysis |
| Container | Trivy, Anchore | Image vulnerabilities |
| Secrets | TruffleHog, GitGuardian | Credential detection |
| IaC | Checkov, tfsec | Infrastructure security |

---

## GitHub Actions Templates

### Dependency + SAST Scanning

```yaml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Security Scan
        run: |
          pip install safety bandit semgrep
          safety check --json
          bandit -r . -f json -o bandit-report.json
          semgrep --config auto .

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: security-reports
          path: bandit-report.json
```

### Lock File Validation

```yaml
- name: Validate Lock Files
  run: |
    pip install poetry
    poetry check && poetry lock --check
```

---

## GitLab CI Template

```yaml
security_scan:
  stage: test
  image: python:3.12
  script:
    - pip install safety bandit
    - safety check --json
    - bandit -r . -ll
  artifacts:
    reports:
      sast: bandit-report.json
```

---

## Lock File Validation

| Language | Lock File | Validation Command |
|----------|-----------|-------------------|
| Python (Poetry) | `poetry.lock` | `poetry lock --check` |
| Python (pip-tools) | `requirements.txt` | `pip-compile --dry-run` |
| Julia | `Manifest.toml` | `Pkg.resolve(); git diff --exit-code` |
| Node.js | `package-lock.json` | `npm ci --dry-run` |
| Rust | `Cargo.lock` | `cargo check` |

---

## Severity Thresholds

| Severity | Action | Configuration |
|----------|--------|---------------|
| CRITICAL | Block merge | `--fail-on critical` |
| HIGH | Block merge | `--fail-on high` |
| MEDIUM | Warn | `allow_failure: true` |
| LOW | Log only | Informational |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Fail on high severity | Block PRs with critical/high vulns |
| Scan every PR | Shift-left security |
| Cache dependencies | Speed up CI runs |
| Pin tool versions | Reproducible scans |
| Mask secrets | `::add-mask::` in logs |
| Document suppressions | Track false positives |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Ignoring warnings | Review all findings systematically |
| Outdated tools | Update security scanners quarterly |
| No lock validation | Add lock file checks to CI |
| Secrets in code | Use secret stores, not env files |
| Allow_failure everywhere | Reserve for low severity only |

---

## Checklist

- [ ] Dependency scanning enabled (Safety/Snyk)
- [ ] SAST scanning configured (Bandit/Semgrep)
- [ ] Lock file validation in CI
- [ ] Severity thresholds set
- [ ] Artifacts uploaded for review
- [ ] Secret scanning active
- [ ] Regular tool updates scheduled

---

**Version**: 1.0.5
