---
description: Setup CI/CD pipeline with dependency version consistency enforcement
allowed-tools: Bash(git:*), Bash(find:*), Bash(grep:*), Read, Write, Edit, Glob, Grep
argument-hint: [platform] [--security] [--docker] [--lock-check]
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
- Tech stack: !`find . -name "package.json" -o -name "requirements.txt" -o -name "Cargo.toml" -o -name "go.mod" -o -name "pyproject.toml" 2>/dev/null | head -5`
- Lock files: !`find . -name "package-lock.json" -o -name "yarn.lock" -o -name "pnpm-lock.yaml" -o -name "poetry.lock" -o -name "Pipfile.lock" -o -name "Cargo.lock" -o -name "go.sum" 2>/dev/null | head -5`
- Existing CI: !`find .github .gitlab-ci.yml .circleci jenkins 2>/dev/null | head -5`

## Your Task: $ARGUMENTS

**Setup comprehensive CI/CD pipeline with dependency version consistency enforcement**:

### 1. Platform Selection
- **GitHub Actions** (Recommended): Native integration, free for public repos, matrix builds
- **GitLab CI**: Built-in, comprehensive DevOps, docker integration
- **CircleCI**: Fast cloud-based, excellent caching
- **Jenkins**: Self-hosted, customizable, enterprise-ready

### 2. Core Pipeline Stages with Version Consistency
```yaml
stages: [validate-deps, test, build, security, deploy]

validate-deps:
  # CRITICAL: Ensure local and CI use identical dependency versions
  - Verify lock files exist and are committed
  - Install from lock files ONLY (npm ci, pip install --no-deps, etc.)
  - Fail if lock files are out of sync
  - Check for dependency drift
  - Validate reproducible builds

test:
  - Lint code (eslint/ruff/clippy)
  - Run unit tests with exact dependencies
  - Generate coverage report
  - Matrix testing (Node 18/20/22, Python 3.10/3.11/3.12)

build:
  - Compile/bundle with locked dependencies
  - Build Docker image with multi-stage optimization
  - Cache dependencies using lock file hash
  - Verify build reproducibility

security:
  - Dependency scan (npm audit/pip-audit/cargo audit)
  - SAST (CodeQL/SonarQube/Semgrep)
  - Container scan (Trivy/Grype)
  - License compliance check
  - Secrets detection (Gitleaks/TruffleHog)

deploy:
  - Staging (auto on develop)
  - Production (manual approval on main)
  - Rollback capability with version tracking
```

### 3. Dependency Version Consistency Enforcement

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

#### Python (pip/poetry/uv)
```yaml
# pip with requirements.txt and hashes
- run: pip install --require-hashes -r requirements.txt

# Poetry with lock file
- run: poetry install --no-root --sync

# uv (modern, fast)
- run: uv pip install --require-hashes -r requirements.txt
- run: uv sync --frozen  # Uses uv.lock
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

### 4. Essential Optimizations
- **Lock-file based caching**: Hash lock file for cache key
  ```yaml
  cache:
    key: ${{ runner.os }}-deps-${{ hashFiles('**/package-lock.json') }}
    paths: [~/.npm, node_modules]
  ```
- **Parallel jobs**: Independent tests run simultaneously
- **Fail fast**: Stop on first critical failure
- **Conditional runs**: Skip unchanged areas using path filters

### 5. Security Best Practices
- **Lock files in version control**: Always commit lock files
- **Dependency integrity**: Use subresource integrity (SRI) hashes
- **Store secrets in vault**: GitHub Secrets, GitLab Variables, HashiCorp Vault
- **Scan dependencies for CVEs**: Automated alerts and blocking
- **Sign commits and artifacts**: GPG signing, SLSA provenance
- **Audit trail**: Track all deployments and changes
- **Least privilege access**: Minimal permissions per job

### 6. Platform-Specific Templates

#### GitHub Actions (`.github/workflows/ci.yml`)
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
        node-version: [18, 20, 22]
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

#### GitLab CI (`.gitlab-ci.yml`)
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
      - NODE_VERSION: ["18", "20", "22"]
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

#### Python with uv (Modern, Fast)
```yaml
name: Python CI with uv

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies with exact versions
        run: uv sync --frozen  # Fails if uv.lock is out of sync

      - name: Run tests
        run: uv run pytest --cov
```

#### Docker Multi-stage with Locked Dependencies
```dockerfile
# syntax=docker/dockerfile:1

FROM node:20-alpine AS deps
WORKDIR /app

# Copy lock file first for better layer caching
COPY package.json package-lock.json ./
RUN npm ci --only=production && npm cache clean --force

FROM node:20-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=deps --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --chown=nextjs:nodejs package.json ./

USER nextjs
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

### 7. Pre-commit Hooks for Local Validation
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: lockfile-check
        name: Verify lockfile is up to date
        entry: npm install --package-lock-only && git diff --exit-code package-lock.json
        language: system
        pass_filenames: false
```

### 8. Implementation Checklist
- [ ] Lock files exist and are committed to version control
- [ ] CI uses deterministic install commands (npm ci, --frozen-lockfile, etc.)
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

### 9. Dependency Version Drift Prevention

**Automated Dependency Updates**:
```yaml
# .github/dependabot.yml
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

### 10. Troubleshooting Common Issues

**Issue**: Lock file out of sync
```bash
# Solution: Regenerate lock file locally
npm install  # or yarn install, pnpm install
git add package-lock.json
git commit -m "chore: update lock file"
```

**Issue**: Cache invalidation problems
```bash
# Solution: Use lock file hash in cache key
key: ${{ runner.os }}-${{ hashFiles('**/package-lock.json') }}
```

**Issue**: Slow dependency installation
```bash
# Solution: Use modern package managers
npm ci → pnpm install --frozen-lockfile  # ~3x faster
pip install → uv pip install  # ~10-100x faster
```

**Provide complete, production-ready pipeline configuration with dependency version consistency enforcement and security best practices**
