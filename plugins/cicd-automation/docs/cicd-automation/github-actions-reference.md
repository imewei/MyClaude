# GitHub Actions Reference

**Version**: 1.0.3
**Command**: `/workflow-automate`
**Category**: CI/CD Automation

## Overview

Comprehensive GitHub Actions workflow patterns and complete production-ready examples covering multi-stage pipelines, matrix builds, security scanning, deployment automation, and monorepo orchestration.

---

## Table of Contents

1. [Node.js CI/CD Pipeline](#1-nodejs-cicd-pipeline)
2. [Python CI/CD Pipeline](#2-python-cicd-pipeline)
3. [Go CI/CD Pipeline](#3-go-cicd-pipeline)
4. [Rust CI/CD Pipeline](#4-rust-cicd-pipeline)
5. [Multi-Service Monorepo](#5-multi-service-monorepo)

---

## 1. Node.js CI/CD Pipeline

Complete multi-stage pipeline for Node.js/TypeScript applications with quality gates, testing, building, security scanning, and deployment.

### `.github/workflows/nodejs-ci-cd.yml`

```yaml
name: Node.js CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:

env:
  NODE_VERSION: '18.x'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ===== STAGE 1: QUALITY =====
  quality:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Lint
        run: npm run lint

      - name: Format check
        run: npm run format:check

      - name: Type check
        run: npm run typecheck

      - name: Security audit
        run: npm audit --audit-level=moderate
        continue-on-error: true

      - name: License check
        run: npx license-checker --summary
        continue-on-error: true

  # ===== STAGE 2: TEST =====
  test:
    name: Test (${{ matrix.os }}, Node ${{ matrix.node-version }})
    runs-on: ${{ matrix.os }}
    needs: quality
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        node-version: [16.x, 18.x, 20.x]
        exclude:
          # Reduce matrix size - skip older Node on Windows/macOS
          - os: windows-latest
            node-version: 16.x
          - os: macos-latest
            node-version: 16.x
      fail-fast: false

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run unit tests
        run: npm test -- --coverage --maxWorkers=2

      - name: Run integration tests
        run: npm run test:integration
        env:
          CI: true

      - name: Upload coverage to Codecov
        if: matrix.os == 'ubuntu-latest' && matrix.node-version == '18.x'
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage/coverage-final.json
          flags: unittests
          name: codecov-${{ matrix.os }}-${{ matrix.node-version }}

  # ===== STAGE 3: BUILD =====
  build:
    name: Build
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build
        env:
          NODE_ENV: production

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: build-artifacts
          path: |
            dist/
            package.json
            package-lock.json
          retention-days: 7

  # ===== STAGE 4: DOCKER BUILD & SCAN =====
  docker:
    name: Build & Scan Docker Image
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'push'
    permissions:
      contents: read
      packages: write
      security-events: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}

      - name: Build and push Docker image
        id: build-and-push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'

  # ===== STAGE 5: DEPLOY TO STAGING =====
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: docker
    if: github.ref == 'refs/heads/develop'
    environment:
      name: staging
      url: https://staging.example.com
    steps:
      - name: Deploy to Staging
        run: |
          echo "Deploying to staging environment..."
          # Add your deployment commands here
          # e.g., kubectl apply, helm upgrade, etc.

      - name: Run smoke tests
        run: |
          echo "Running smoke tests..."
          curl -f https://staging.example.com/health || exit 1

  # ===== STAGE 6: DEPLOY TO PRODUCTION =====
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: docker
    if: github.ref == 'refs/heads/main'
    environment:
      name: production
      url: https://example.com
    steps:
      - name: Deploy to Production
        run: |
          echo "Deploying to production environment..."
          # Add your deployment commands here

      - name: Run smoke tests
        run: |
          echo "Running smoke tests..."
          curl -f https://example.com/health || exit 1

      - name: Notify Slack
        if: always()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Deployment to production: ${{ job.status }}",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Deployment Status:* ${{ job.status }}\n*Environment:* Production\n*Commit:* ${{ github.sha }}"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

**Key Features**:
- ✅ Matrix testing across 3 OS × 3 Node versions
- ✅ Multi-stage pipeline with dependency gates
- ✅ Docker build with layer caching
- ✅ Trivy security scanning with SARIF upload
- ✅ Environment-based deployment (staging/production)
- ✅ Slack notifications

---

## 2. Python CI/CD Pipeline

Complete pipeline for Python applications with Poetry, pytest, Docker, and deployment automation.

### `.github/workflows/python-ci-cd.yml`

```yaml
name: Python CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: '3.11'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ===== QUALITY =====
  quality:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install Poetry
        uses: snok/install-poetry@v1
        with:
          version: 1.7.0
          virtualenvs-create: true
          virtualenvs-in-project: true

      - name: Load cached venv
        id: cached-poetry-dependencies
        uses: actions/cache@v4
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ env.PYTHON_VERSION }}-${{ hashFiles('**/poetry.lock') }}

      - name: Install dependencies
        if: steps.cached-poetry-dependencies.outputs.cache-hit != 'true'
        run: poetry install --no-interaction --no-root

      - name: Install project
        run: poetry install --no-interaction

      - name: Lint with ruff
        run: poetry run ruff check .

      - name: Format check with black
        run: poetry run black --check .

      - name: Type check with mypy
        run: poetry run mypy .

      - name: Security check with bandit
        run: poetry run bandit -r src/
        continue-on-error: true

  # ===== TEST =====
  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    needs: quality
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
      fail-fast: false

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Poetry
        uses: snok/install-poetry@v1

      - name: Cache dependencies
        uses: actions/cache@v4
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}

      - name: Install dependencies
        run: poetry install

      - name: Run tests with pytest
        run: |
          poetry run pytest \
            --cov=src \
            --cov-report=xml \
            --cov-report=term-missing \
            --junit-xml=pytest-results.xml \
            -v

      - name: Upload coverage
        if: matrix.python-version == '3.11'
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
          flags: pytest
          name: python-${{ matrix.python-version }}

      - name: Publish test results
        uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: pytest-results.xml

  # ===== BUILD =====
  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install Poetry
        uses: snok/install-poetry@v1

      - name: Build package
        run: poetry build

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: python-package
          path: dist/

  # ===== DOCKER =====
  docker:
    name: Build & Push Docker
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'push'
    permissions:
      contents: read
      packages: write
      security-events: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Scan with Snyk
        uses: snyk/actions/docker@master
        continue-on-error: true
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          args: --severity-threshold=high

  # ===== DEPLOY =====
  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    needs: docker
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - name: Deploy to production
        run: echo "Deploy to production"
```

**Key Features**:
- ✅ Poetry dependency management with caching
- ✅ Matrix testing across Python 3.10/3.11/3.12
- ✅ ruff, black, mypy, bandit for code quality
- ✅ pytest with coverage reporting
- ✅ Snyk security scanning

---

## 3. Go CI/CD Pipeline

Production pipeline for Go applications with matrix testing, static analysis, and containerization.

### `.github/workflows/go-ci-cd.yml`

```yaml
name: Go CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  GO_VERSION: '1.21'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ===== QUALITY =====
  quality:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Go
        uses: actions/setup-go@v5
        with:
          go-version: ${{ env.GO_VERSION }}
          cache: true

      - name: Verify dependencies
        run: go mod verify

      - name: Run go vet
        run: go vet ./...

      - name: Run staticcheck
        uses: dominikh/staticcheck-action@v1
        with:
          version: latest

      - name: Run golangci-lint
        uses: golangci/golangci-lint-action@v4
        with:
          version: latest

      - name: Run gosec (security)
        uses: securego/gosec@master
        with:
          args: ./...

  # ===== TEST =====
  test:
    name: Test (Go ${{ matrix.go-version }}, ${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    needs: quality
    strategy:
      matrix:
        go-version: ['1.20', '1.21', '1.22']
        os: [ubuntu-latest, macos-latest]
      fail-fast: false

    steps:
      - uses: actions/checkout@v4

      - name: Setup Go
        uses: actions/setup-go@v5
        with:
          go-version: ${{ matrix.go-version }}
          cache: true

      - name: Run tests
        run: |
          go test -v -race -coverprofile=coverage.txt -covermode=atomic ./...

      - name: Upload coverage
        if: matrix.os == 'ubuntu-latest' && matrix.go-version == '1.21'
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.txt
          flags: go

  # ===== BUILD =====
  build:
    name: Build Binaries
    runs-on: ubuntu-latest
    needs: test
    strategy:
      matrix:
        goos: [linux, darwin, windows]
        goarch: [amd64, arm64]
        exclude:
          - goos: windows
            goarch: arm64

    steps:
      - uses: actions/checkout@v4

      - name: Setup Go
        uses: actions/setup-go@v5
        with:
          go-version: ${{ env.GO_VERSION }}
          cache: true

      - name: Build binary
        env:
          GOOS: ${{ matrix.goos }}
          GOARCH: ${{ matrix.goarch }}
        run: |
          go build -ldflags="-s -w" -o bin/app-${{ matrix.goos }}-${{ matrix.goarch }} ./cmd/app

      - name: Upload binary
        uses: actions/upload-artifact@v4
        with:
          name: binary-${{ matrix.goos }}-${{ matrix.goarch }}
          path: bin/

  # ===== DOCKER =====
  docker:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'push'
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

**Key Features**:
- ✅ Multiple Go versions (1.20, 1.21, 1.22)
- ✅ Cross-platform builds (Linux/macOS/Windows, amd64/arm64)
- ✅ golangci-lint, staticcheck, gosec
- ✅ Multi-architecture Docker images

---

## 4. Rust CI/CD Pipeline

Complete Rust pipeline with cargo checks, clippy, testing, and release builds.

### `.github/workflows/rust-ci-cd.yml`

```yaml
name: Rust CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RUST_VERSION: '1.75.0'

jobs:
  # ===== QUALITY =====
  quality:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: ${{ env.RUST_VERSION }}
          components: rustfmt, clippy

      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/bin/
            ~/.cargo/registry/index/
            ~/.cargo/registry/cache/
            ~/.cargo/git/db/
            target/
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Check formatting
        run: cargo fmt -- --check

      - name: Run clippy
        run: cargo clippy -- -D warnings

      - name: Security audit
        uses: rustsec/audit-check@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

  # ===== TEST =====
  test:
    name: Test (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    needs: quality
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
      fail-fast: false

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/bin/
            ~/.cargo/registry/index/
            ~/.cargo/registry/cache/
            ~/.cargo/git/db/
            target/
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Run tests
        run: cargo test --verbose --all-features

      - name: Generate coverage (Linux only)
        if: matrix.os == 'ubuntu-latest'
        uses: taiki-e/install-action@cargo-llvm-cov

      - name: Run coverage
        if: matrix.os == 'ubuntu-latest'
        run: cargo llvm-cov --all-features --lcov --output-path lcov.info

      - name: Upload coverage
        if: matrix.os == 'ubuntu-latest'
        uses: codecov/codecov-action@v4
        with:
          files: lcov.info
          flags: rust

  # ===== BUILD =====
  build:
    name: Build Release
    runs-on: ${{ matrix.os }}
    needs: test
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - os: macos-latest
            target: x86_64-apple-darwin
          - os: windows-latest
            target: x86_64-pc-windows-msvc

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}

      - name: Build release
        run: cargo build --release --target ${{ matrix.target }}

      - name: Upload binary
        uses: actions/upload-artifact@v4
        with:
          name: binary-${{ matrix.target }}
          path: |
            target/${{ matrix.target }}/release/app
            target/${{ matrix.target }}/release/app.exe
```

**Key Features**:
- ✅ rustfmt, clippy with strict warnings
- ✅ Security audit with rustsec
- ✅ cargo-llvm-cov for coverage
- ✅ Multi-platform release builds

---

## 5. Multi-Service Monorepo

Advanced monorepo pipeline with service-specific workflows, change detection, and coordinated deployment.

### `.github/workflows/monorepo-ci-cd.yml`

```yaml
name: Monorepo CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # ===== DETECT CHANGES =====
  changes:
    name: Detect Changes
    runs-on: ubuntu-latest
    outputs:
      frontend: ${{ steps.filter.outputs.frontend }}
      backend: ${{ steps.filter.outputs.backend }}
      shared: ${{ steps.filter.outputs.shared }}
      infra: ${{ steps.filter.outputs.infra }}
    steps:
      - uses: actions/checkout@v4

      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            frontend:
              - 'packages/frontend/**'
              - 'packages/shared/**'
            backend:
              - 'packages/backend/**'
              - 'packages/shared/**'
            shared:
              - 'packages/shared/**'
            infra:
              - 'infrastructure/**'
              - '.github/workflows/**'

  # ===== SHARED LIBRARY =====
  shared-lib:
    name: Shared Library
    runs-on: ubuntu-latest
    needs: changes
    if: needs.changes.outputs.shared == 'true'
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18.x'
          cache: 'npm'
          cache-dependency-path: packages/shared/package-lock.json

      - name: Install & Build
        working-directory: packages/shared
        run: |
          npm ci
          npm run build
          npm test

      - name: Upload shared lib
        uses: actions/upload-artifact@v4
        with:
          name: shared-lib
          path: packages/shared/dist/

  # ===== FRONTEND SERVICE =====
  frontend:
    name: Frontend Service
    runs-on: ubuntu-latest
    needs: [changes, shared-lib]
    if: |
      always() &&
      needs.changes.outputs.frontend == 'true' &&
      (needs.shared-lib.result == 'success' || needs.shared-lib.result == 'skipped')
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18.x'
          cache: 'npm'
          cache-dependency-path: packages/frontend/package-lock.json

      - name: Download shared lib
        if: needs.changes.outputs.shared == 'true'
        uses: actions/download-artifact@v4
        with:
          name: shared-lib
          path: packages/shared/dist/

      - name: Install dependencies
        working-directory: packages/frontend
        run: npm ci

      - name: Lint
        working-directory: packages/frontend
        run: npm run lint

      - name: Test
        working-directory: packages/frontend
        run: npm test -- --coverage

      - name: Build
        working-directory: packages/frontend
        run: npm run build
        env:
          NODE_ENV: production

      - name: E2E Tests
        working-directory: packages/frontend
        run: npm run test:e2e

      - name: Upload build
        uses: actions/upload-artifact@v4
        with:
          name: frontend-build
          path: packages/frontend/dist/

  # ===== BACKEND SERVICE =====
  backend:
    name: Backend Service
    runs-on: ubuntu-latest
    needs: [changes, shared-lib]
    if: |
      always() &&
      needs.changes.outputs.backend == 'true' &&
      (needs.shared-lib.result == 'success' || needs.shared-lib.result == 'skipped')

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Poetry
        uses: snok/install-poetry@v1

      - name: Cache dependencies
        uses: actions/cache@v4
        with:
          path: packages/backend/.venv
          key: backend-venv-${{ runner.os }}-${{ hashFiles('packages/backend/poetry.lock') }}

      - name: Install dependencies
        working-directory: packages/backend
        run: poetry install

      - name: Lint
        working-directory: packages/backend
        run: poetry run ruff check .

      - name: Type check
        working-directory: packages/backend
        run: poetry run mypy .

      - name: Run migrations
        working-directory: packages/backend
        run: poetry run alembic upgrade head
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/testdb

      - name: Run tests
        working-directory: packages/backend
        run: poetry run pytest --cov=src --cov-report=xml
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/testdb
          REDIS_URL: redis://localhost:6379/0

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: packages/backend/coverage.xml
          flags: backend

  # ===== DOCKER BUILD =====
  docker-build:
    name: Build Docker Images
    runs-on: ubuntu-latest
    needs: [frontend, backend]
    if: github.event_name == 'push'
    strategy:
      matrix:
        service: [frontend, backend]
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}/${{ matrix.service }}
          tags: |
            type=ref,event=branch
            type=sha
            type=semver,pattern={{version}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: packages/${{ matrix.service }}
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha,scope=${{ matrix.service }}
          cache-to: type=gha,mode=max,scope=${{ matrix.service }}

  # ===== DEPLOY =====
  deploy:
    name: Deploy to ${{ matrix.environment }}
    runs-on: ubuntu-latest
    needs: docker-build
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
    strategy:
      matrix:
        environment:
          - ${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }}
    environment:
      name: ${{ matrix.environment }}

    steps:
      - uses: actions/checkout@v4

      - name: Deploy to Kubernetes
        run: |
          echo "Deploying to ${{ matrix.environment }}"
          # kubectl apply -f k8s/${{ matrix.environment }}/

      - name: Run smoke tests
        run: |
          echo "Running smoke tests for ${{ matrix.environment }}"
          # Add smoke test commands

      - name: Notify deployment
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Deployed to ${{ matrix.environment }}: ${{ job.status }}",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Monorepo Deployment*\n*Environment:* ${{ matrix.environment }}\n*Status:* ${{ job.status }}\n*Commit:* ${{ github.sha }}"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

**Key Features**:
- ✅ Change detection with path filters
- ✅ Service-specific pipelines with dependencies
- ✅ Shared library build artifact passing
- ✅ Service containers (PostgreSQL, Redis)
- ✅ Coordinated multi-service deployment
- ✅ Environment-based deployment (staging/production)

---

## Best Practices Summary

### 1. Caching Strategies
```yaml
# NPM cache
- uses: actions/setup-node@v4
  with:
    cache: 'npm'

# Custom cache
- uses: actions/cache@v4
  with:
    path: ~/.cache
    key: ${{ runner.os }}-cache-${{ hashFiles('**/file.lock') }}
```

### 2. Matrix Builds
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest]
    version: [16, 18, 20]
  fail-fast: false
```

### 3. Security Scanning
```yaml
- name: Run Trivy
  uses: aquasecurity/trivy-action@master
  with:
    format: 'sarif'
    output: 'trivy-results.sarif'

- name: Upload to GitHub Security
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: 'trivy-results.sarif'
```

### 4. Artifact Management
```yaml
- uses: actions/upload-artifact@v4
  with:
    name: build-artifacts
    path: dist/
    retention-days: 7

- uses: actions/download-artifact@v4
  with:
    name: build-artifacts
```

---

For related documentation, see:
- [gitlab-ci-reference.md](gitlab-ci-reference.md)
- [terraform-cicd-integration.md](terraform-cicd-integration.md)
- [security-automation-workflows.md](security-automation-workflows.md)
- [workflow-orchestration-patterns.md](workflow-orchestration-patterns.md)
