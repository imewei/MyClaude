# Security-Focused CI Template

**Purpose**: Security scanning and lock file validation templates for CI/CD pipelines

**Use Instead Of**: `/sci-ci-setup` command (removed in Week 2-3)

**Recommended**: Use marketplace `cicd-automation:workflow-automate` + this skill

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
