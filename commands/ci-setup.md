---
description: Setup CI/CD pipeline with dependency version consistency enforcement
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
argument-hint: [platform] [--security] [--lock-check]
color: blue
agents:
  primary:
    - devops-security-engineer
  conditional:
    - agent: systems-architect
      trigger: pattern "microservice|distributed|scalability"
    - agent: fullstack-developer
      trigger: files "package.json|frontend|backend"
  orchestrated: false
---

# CI/CD Pipeline Setup with Dependency Version Consistency

## Context
- Repository: !`git remote get-url origin 2>/dev/null || echo "Not a git repo"`
- Tech stack: !`find . -name "package.json" -o -name "pyproject.toml" -o -name "uv.lock" -o -name "Cargo.toml" -o -name "go.mod" 2>/dev/null | head -5`
- Lock files: !`find . -name "package-lock.json" -o -name "yarn.lock" -o -name "pnpm-lock.yaml" -o -name "uv.lock" -o -name "Cargo.lock" -o -name "go.sum" 2>/dev/null | head -5`
- Existing CI: !`find .github .gitlab-ci.yml .circleci jenkins 2>/dev/null | head -5`

## Your Task: $ARGUMENTS

**Setup comprehensive CI/CD pipeline with dependency version consistency enforcement**:

### 1. Scope and Exclusions

**This command focuses on:**
- CI/CD pipeline configuration (GitHub Actions, GitLab CI, CircleCI, Jenkins)
- Dependency version consistency enforcement
- Security scanning and testing automation
- Deployment workflows and best practices

**Explicitly excluded from this command:**
- ❌ **Docker/Container configurations** - Not included in CI/CD setup
- ❌ **GitHub Pages deployment** - Static site hosting not covered
- ❌ **Container orchestration** (K8s, Docker Compose) - Separate infrastructure concern
- ❌ **Container registry operations** - Not part of core CI/CD pipeline

For container-specific CI/CD needs, use dedicated containerization tools and workflows separately.

---

### 2. Platform Selection
- **GitHub Actions** (Recommended): Native integration, free for public repos, matrix builds
- **GitLab CI**: Built-in, comprehensive DevOps, excellent integration
- **CircleCI**: Fast cloud-based, excellent caching
- **Jenkins**: Self-hosted, customizable, enterprise-ready

### 3. Core Pipeline Stages with Version Consistency
```yaml
stages: [validate-deps, test, build, security, deploy]

validate-deps:
  # CRITICAL: Ensure local and CI use identical dependency versions
  - Verify lock files exist and are committed
  - Install from lock files ONLY (npm ci, uv sync --frozen, etc.)
  - Fail if lock files are out of sync
  - Check for dependency drift
  - Validate reproducible builds

test:
  - Lint code (eslint/ruff/clippy)
  - Run unit tests with exact dependencies
  - Generate coverage report
  - Matrix testing (Node 20/22/23, Python 3.12/3.13)

build:
  - Compile/bundle with locked dependencies
  - Cache dependencies using lock file hash
  - Verify build reproducibility

security:
  - Dependency scan (npm audit/uv pip audit/cargo audit)
  - SAST (CodeQL/SonarQube/Semgrep)
  - License compliance check
  - Secrets detection (Gitleaks/TruffleHog)

deploy:
  - Staging (auto on develop)
  - Production (manual approval on main)
  - Rollback capability with version tracking
```

### 4. Dependency Version Consistency Enforcement

**Key Principle**: Lock files are the source of truth. Never use `npm install`, `pip install <package>`, or similar commands in CI.

#### Node.js/JavaScript (npm/yarn/pnpm)
```yaml
# GitHub Actions - npm
- name: Install dependencies
  run: npm ci  # Uses package-lock.json, fails if out of sync

# Verify lock file is up to date
- name: Verify lockfile
  run: |
    npm install --package-lock-only
    git diff --exit-code package-lock.json

# Yarn
- run: yarn install --frozen-lockfile

# pnpm
- run: pnpm install --frozen-lockfile
```

#### Python 3.12+ (uv - Modern, Fast)
```yaml
# uv with lock file (RECOMMENDED for Python 3.12+)
- name: Install uv
  uses: astral-sh/setup-uv@v4
  with:
    version: "latest"
    enable-cache: true

- name: Set up Python 3.12+
  uses: actions/setup-python@v5
  with:
    python-version: "3.12"

- name: Install dependencies with exact versions
  run: uv sync --frozen  # Fails if uv.lock is out of sync

- name: Run tests
  run: uv run pytest --cov

# Alternative: uv pip with requirements.txt (hashed)
- run: uv pip install --require-hashes -r requirements.txt
```

#### Rust (Cargo)
```yaml
# Cargo automatically uses Cargo.lock
- run: cargo build --locked  # Fail if lock file is stale
```

#### Go (go.mod)
```yaml
# Verify go.sum is current
- run: go mod verify
- run: go build -mod=readonly  # Don't modify go.mod/go.sum
```

### 5. Essential Optimizations
- **Lock-file based caching**: Hash lock file for cache key
  ```yaml
  # Node.js
  cache:
    key: ${{ runner.os }}-deps-${{ hashFiles('**/package-lock.json') }}
    paths: [~/.npm, node_modules]

  # Python with uv
  cache:
    key: ${{ runner.os }}-uv-${{ hashFiles('**/uv.lock') }}
    paths: [~/.cache/uv]
  ```
- **Parallel jobs**: Independent tests run simultaneously
- **Fail fast**: Stop on first critical failure
- **Conditional runs**: Skip unchanged areas using path filters

### 6. Security Best Practices
- **Lock files in version control**: Always commit lock files
- **Dependency integrity**: Use subresource integrity (SRI) hashes
- **Store secrets in vault**: GitHub Secrets, GitLab Variables, HashiCorp Vault
- **Scan dependencies for CVEs**: Automated alerts and blocking
- **Sign commits and artifacts**: GPG signing, SLSA provenance
- **Audit trail**: Track all deployments and changes
- **Least privilege access**: Minimal permissions per job

### 7. Platform-Specific Templates

#### GitHub Actions (`.github/workflows/ci.yml`) - Node.js
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  validate-dependencies:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Verify lockfile integrity
        run: |
          npm ci
          npm install --package-lock-only
          git diff --exit-code package-lock.json || {
            echo "❌ package-lock.json is out of sync!"
            echo "Run 'npm install' locally and commit the updated lock file."
            exit 1
          }

  test:
    needs: validate-dependencies
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [20, 22, 23]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - name: Install exact dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Run tests with coverage
        run: npm test -- --coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.node-version == 20

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: build
          path: dist/

  security:
    needs: validate-dependencies
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run dependency audit
        run: npm audit --audit-level=moderate

      - name: Run CodeQL analysis
        uses: github/codeql-action/analyze@v3

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'

  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    needs: [build, security]
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: echo "Deploy to staging"

  deploy-production:
    if: github.ref == 'refs/heads/main'
    needs: [build, security]
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to production
        run: echo "Deploy to production"
```

#### GitHub Actions - Python 3.12+ with uv
```yaml
name: Python CI with uv

on: [push, pull_request]

jobs:
  validate-dependencies:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
          enable-cache: true

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Verify uv.lock is up to date
        run: |
          uv lock --check
          git diff --exit-code uv.lock || {
            echo "❌ uv.lock is out of sync!"
            echo "Run 'uv lock' locally and commit the updated lock file."
            exit 1
          }

  test:
    needs: validate-dependencies
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.12', '3.13']
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
          enable-cache: true
          cache-dependency-glob: "uv.lock"

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies with exact versions
        run: uv sync --frozen

      - name: Run linter
        run: uv run ruff check .

      - name: Run type checker
        run: uv run mypy .

      - name: Run tests with coverage
        run: uv run pytest --cov --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.12'

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Build package
        run: uv build

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  security:
    needs: validate-dependencies
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: uv sync --frozen

      - name: Run safety check
        run: uv run safety check

      - name: Run bandit security linter
        run: uv run bandit -r src/

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'
```

#### GitLab CI (`.gitlab-ci.yml`) - Node.js
```yaml
stages:
  - validate
  - test
  - build
  - security
  - deploy

variables:
  NODE_VERSION: "20"

.node_template: &node_setup
  image: node:${NODE_VERSION}
  cache:
    key:
      files:
        - package-lock.json
    paths:
      - node_modules/
      - .npm/

validate-lockfile:
  <<: *node_setup
  stage: validate
  script:
    - npm ci
    - npm install --package-lock-only
    - git diff --exit-code package-lock.json || exit 1

test:
  <<: *node_setup
  stage: test
  needs: [validate-lockfile]
  parallel:
    matrix:
      - NODE_VERSION: ["20", "22", "23"]
  script:
    - npm ci
    - npm run lint
    - npm test -- --coverage
  coverage: '/All files[^|]*\|[^|]*\s+([\d\.]+)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

build:
  <<: *node_setup
  stage: build
  needs: [test]
  script:
    - npm ci
    - npm run build
  artifacts:
    paths:
      - dist/
    expire_in: 1 week

security-scan:
  <<: *node_setup
  stage: security
  needs: [validate-lockfile]
  script:
    - npm audit --audit-level=moderate
    - npx snyk test || true

deploy-staging:
  stage: deploy
  needs: [build, security-scan]
  environment: staging
  only:
    - develop
  script:
    - echo "Deploy to staging"

deploy-production:
  stage: deploy
  needs: [build, security-scan]
  environment: production
  when: manual
  only:
    - main
  script:
    - echo "Deploy to production"
```

#### GitLab CI - Python 3.12+ with uv
```yaml
stages:
  - validate
  - test
  - build
  - security
  - deploy

variables:
  PYTHON_VERSION: "3.12"

.python_template: &python_setup
  image: python:${PYTHON_VERSION}
  before_script:
    - curl -LsSf https://astral.sh/uv/install.sh | sh
    - source $HOME/.cargo/env
  cache:
    key:
      files:
        - uv.lock
    paths:
      - .venv/
      - ~/.cache/uv/

validate-lockfile:
  <<: *python_setup
  stage: validate
  script:
    - uv lock --check
    - git diff --exit-code uv.lock || exit 1

test:
  <<: *python_setup
  stage: test
  needs: [validate-lockfile]
  parallel:
    matrix:
      - PYTHON_VERSION: ["3.12", "3.13"]
  script:
    - uv sync --frozen
    - uv run ruff check .
    - uv run mypy .
    - uv run pytest --cov --cov-report=xml
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  <<: *python_setup
  stage: build
  needs: [test]
  script:
    - uv build
  artifacts:
    paths:
      - dist/
    expire_in: 1 week

security-scan:
  <<: *python_setup
  stage: security
  needs: [validate-lockfile]
  script:
    - uv sync --frozen
    - uv run safety check
    - uv run bandit -r src/

deploy-staging:
  stage: deploy
  needs: [build, security-scan]
  environment: staging
  only:
    - develop
  script:
    - echo "Deploy to staging"

deploy-production:
  stage: deploy
  needs: [build, security-scan]
  environment: production
  when: manual
  only:
    - main
  script:
    - echo "Deploy to production"
```

### 8. Pre-commit Hooks for Local Validation
```yaml
# .pre-commit-config.yaml - Node.js
repos:
  - repo: local
    hooks:
      - id: lockfile-check
        name: Verify lockfile is up to date
        entry: npm install --package-lock-only && git diff --exit-code package-lock.json
        language: system
        pass_filenames: false

# .pre-commit-config.yaml - Python with uv
repos:
  - repo: local
    hooks:
      - id: uv-lock-check
        name: Verify uv.lock is up to date
        entry: uv lock --check
        language: system
        pass_filenames: false
      - id: ruff
        name: Ruff linter
        entry: uv run ruff check --fix
        language: system
        types: [python]
```

### 9. Implementation Checklist
- [ ] Lock files exist and are committed to version control
- [ ] CI uses deterministic install commands (npm ci, uv sync --frozen, etc.)
- [ ] Lock file validation step in CI pipeline
- [ ] Dependency caching based on lock file hash
- [ ] Branch protection (require PR reviews)
- [ ] Automated testing on every PR
- [ ] Code coverage minimum threshold (80%)
- [ ] Security scans pass before merge
- [ ] Staging auto-deploy from develop branch
- [ ] Production requires manual approval
- [ ] Rollback procedure tested and documented
- [ ] Monitoring/alerts configured
- [ ] Dependency update automation (Renovate/Dependabot)

### 10. Dependency Version Drift Prevention

**Automated Dependency Updates**:
```yaml
# .github/dependabot.yml - Node.js
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "chore(deps)"
    groups:
      production-dependencies:
        dependency-type: "production"
      development-dependencies:
        dependency-type: "development"

# .github/dependabot.yml - Python with pip
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "chore(deps)"
```

**Renovate Configuration**:
```json
{
  "extends": ["config:recommended"],
  "lockFileMaintenance": {
    "enabled": true,
    "schedule": ["before 4am on monday"]
  },
  "rangeStrategy": "pin",
  "semanticCommits": "enabled"
}
```

### 11. Troubleshooting Common Issues

**Issue**: Lock file out of sync
```bash
# Node.js
npm install
git add package-lock.json
git commit -m "chore: update lock file"

# Python with uv
uv lock
git add uv.lock
git commit -m "chore: update lock file"
```

**Issue**: Cache invalidation problems
```bash
# Solution: Use lock file hash in cache key
# Node.js
key: ${{ runner.os }}-${{ hashFiles('**/package-lock.json') }}

# Python with uv
key: ${{ runner.os }}-${{ hashFiles('**/uv.lock') }}
```

**Issue**: Slow dependency installation
```bash
# Modern package managers are significantly faster:
npm ci → pnpm install --frozen-lockfile  # ~3x faster
pip install → uv pip install            # ~10-100x faster
```

**Provide complete, production-ready pipeline configuration with dependency version consistency enforcement and security best practices**
