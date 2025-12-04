---
name: security-ci-template
description: Security scanning and lock file validation templates for CI/CD pipelines including dependency vulnerability scanning (Safety, Snyk, npm audit), SAST security analysis (Bandit, Semgrep, CodeQL), lock file validation (poetry.lock, package-lock.json, Cargo.lock), and compliance checking. Use when implementing security-focused CI/CD workflows with automated vulnerability detection, setting up dependency scanning for Python (Safety, Bandit), Node.js (npm audit, Snyk), or other languages, configuring SAST (Static Application Security Testing) tools like Semgrep or SonarQube, validating lock files to ensure dependency reproducibility and detect drift, implementing automated security gates to block merges with critical vulnerabilities, scanning Docker images for vulnerabilities with Trivy or Anchore, setting up DAST (Dynamic Application Security Testing) for runtime security testing, configuring secret scanning to detect exposed credentials in code, implementing software composition analysis (SCA) for open-source dependencies, setting up compliance scanning for HIPAA, PCI-DSS, or SOC2 requirements, creating security reports and uploading SARIF results to GitHub Security tab, implementing automated dependency updates with security patch automation, configuring fail-on-severity thresholds for vulnerability management, or integrating security scanning into pull request workflows for early detection. Use this skill when working with GitHub Actions security workflows, GitLab security scanning templates, security tool configurations, or any CI/CD pipeline requiring security automation and compliance validation.
---

# Security-Focused CI Template

**Purpose**: Security scanning and lock file validation templates for CI/CD pipelines

**Use Instead Of**: `/sci-ci-setup` command (removed in Week 2-3)

**Recommended**: Use `/cicd-automation:workflow-automate` command + this skill

## When to use this skill

- When implementing security-focused CI/CD workflows with automated vulnerability detection
- When setting up dependency vulnerability scanning for Python projects (Safety, Bandit, pip-audit)
- When configuring security scanning for Node.js projects (npm audit, Snyk, retire.js)
- When implementing SAST (Static Application Security Testing) with Semgrep, SonarQube, or CodeQL
- When validating lock files (poetry.lock, package-lock.json, Cargo.lock, Gemfile.lock) for dependency reproducibility
- When detecting lock file drift or inconsistencies between lock files and dependency declarations
- When scanning Docker images for vulnerabilities using Trivy, Anchore, or Clair
- When setting up automated security gates to block pull requests with critical or high-severity vulnerabilities
- When implementing DAST (Dynamic Application Security Testing) for runtime security analysis
- When configuring secret scanning with TruffleHog, GitGuardian, or git-secrets in pipelines
- When performing software composition analysis (SCA) for open-source dependency risks
- When implementing compliance scanning for HIPAA, PCI-DSS, SOC2, or industry regulations
- When generating security reports in SARIF format for GitHub Security or GitLab Security Dashboard
- When configuring automated dependency updates with security patch prioritization
- When setting severity thresholds (CRITICAL, HIGH, MEDIUM, LOW) for vulnerability blocking
- When integrating security scanning into pull request workflows for shift-left security
- When implementing license compliance scanning for open-source dependencies
- When setting up container scanning for Kubernetes deployments or container registries
- When configuring IaC (Infrastructure as Code) security scanning with Checkov, tfsec, or Terrascan
- When implementing security monitoring and alerting for CI/CD pipeline vulnerabilities
- When creating security dashboards and metrics for vulnerability tracking over time
- When working with GitHub Actions security scanning workflows or GitLab Security templates
- When integrating with security platforms like Snyk, WhiteSource, or Veracode
- When establishing security best practices and standards for development teams

---

## Security CI Features

### 1. Dependency Scanning Template

#### GitHub Actions
```yaml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Python dependency scanning
      - name: Safety Check
        run: |
          pip install safety
          safety check --json

      # SAST scanning
      - name: Bandit Security Scan
        run: |
          pip install bandit
          bandit -r . -f json -o bandit-report.json

      # Upload results
      - name: Upload Security Report
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            bandit-report.json
```

#### GitLab CI
```yaml
security_scan:
  stage: test
  image: python:3.12
  script:
    - pip install safety bandit
    - safety check --json
    - bandit -r . -f json -o bandit-report.json
  artifacts:
    reports:
      sast: bandit-report.json
```

### 2. Lock File Validation

#### Poetry Lock Check (GitHub Actions)
```yaml
name: Lock File Check

on: [push, pull_request]

jobs:
  lock-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install Poetry
        run: |
          pip install poetry

      - name: Validate Lock File
        run: |
          poetry check
          poetry lock --check

      - name: Verify Dependencies
        run: |
          poetry install --dry-run
```

#### pip-tools Lock Check
```yaml
- name: Validate pip lock
  run: |
    pip install pip-tools
    pip-compile --dry-run requirements.in
    diff requirements.txt <(pip-compile requirements.in)
```

#### Julia Manifest Check
```yaml
- name: Validate Julia Manifest
  run: |
    julia --project -e 'using Pkg; Pkg.resolve()'
    git diff --exit-code Manifest.toml
```

### 3. Combined Security + Lock Template

```yaml
name: Security & Dependency Validation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  security-and-locks:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      # Lock file validation
      - name: Install Poetry
        run: pip install poetry

      - name: Check Lock File
        run: |
          poetry check
          poetry lock --check

      # Security scanning
      - name: Security Scan
        run: |
          pip install safety bandit semgrep
          safety check
          bandit -r . -ll
          semgrep --config auto .

      # Dependency audit
      - name: Audit Dependencies
        run: |
          poetry export -f requirements.txt | safety check --stdin

      - name: Upload Results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            .sarif
            bandit-report.json
```

---

## Usage

### Step 1: Use Marketplace for Basic CI

Use the marketplace CI/CD plugin to set up basic pipeline:
```bash
# This will be handled by marketplace plugin automatically
# Or explicitly invoke if available
```

### Step 2: Add Security Template

Copy the appropriate template from above and add to `.github/workflows/` or `.gitlab-ci.yml`

### Step 3: Customize

Adjust security tools based on project needs:
- **Python**: safety, bandit, semgrep
- **Julia**: Aqua.jl for package quality
- **JavaScript**: npm audit, snyk
- **Multi-language**: semgrep, trivy

---

## Best Practices

1. **Run on every PR**: Catch security issues early
2. **Fail on high-severity**: Block merges with critical vulns
3. **Cache dependencies**: Speed up CI runs
4. **Regular updates**: Keep security tools current
5. **Document suppressions**: Track false positives

---

## Integration with Scientific Workflows

### For HPC/JAX Projects
```yaml
- name: Check JAX Dependencies
  run: |
    # Ensure JAX GPU dependencies are locked
    poetry show jax jaxlib | grep -E "version|cuda"
    poetry lock --check
```

### For Data Science Projects
```yaml
- name: Validate Data Pipeline Dependencies
  run: |
    # Check critical data science libraries are pinned
    poetry show numpy scipy pandas | grep version
    poetry check
```

---

**Replaced Command**: `/sci-ci-setup`
**Maintenance**: Update security tools quarterly
**Support**: Refer to marketplace `cicd-automation` plugin documentation
