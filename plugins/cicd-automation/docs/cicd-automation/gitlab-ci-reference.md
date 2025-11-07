# GitLab CI Reference

**Version**: 1.0.3
**Command**: `/workflow-automate`
**Category**: CI/CD Automation

## Overview

Comprehensive GitLab CI/CD pipeline patterns and production-ready examples covering parallel execution, caching strategies, dynamic pipelines, security scanning, and deployment automation.

---

## Table of Contents

1. [Node.js with Docker](#1-nodejs-with-docker)
2. [Python with Poetry](#2-python-with-poetry)
3. [Go Microservices](#3-go-microservices)
4. [Monorepo with Trigger Rules](#4-monorepo-with-trigger-rules)

---

## 1. Node.js with Docker

Complete pipeline for Node.js applications with testing, building, Docker containerization, and deployment.

### `.gitlab-ci.yml`

```yaml
# GitLab CI/CD for Node.js Application

variables:
  NODE_VERSION: "18"
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
  DOCKER_DRIVER: overlay2
  npm_config_cache: "$CI_PROJECT_DIR/.npm"
  CYPRESS_CACHE_FOLDER: "$CI_PROJECT_DIR/.cypress"

stages:
  - quality
  - test
  - build
  - deploy

# ===== TEMPLATES =====
.node_template: &node_template
  image: node:${NODE_VERSION}
  cache:
    key:
      files:
        - package-lock.json
    paths:
      - .npm
      - node_modules
  before_script:
    - npm ci --prefer-offline

# ===== QUALITY STAGE =====
lint:
  <<: *node_template
  stage: quality
  script:
    - npm run lint

format-check:
  <<: *node_template
  stage: quality
  script:
    - npm run format:check

typecheck:
  <<: *node_template
  stage: quality
  script:
    - npm run typecheck

security-audit:
  <<: *node_template
  stage: quality
  script:
    - npm audit --audit-level=moderate
  allow_failure: true

# ===== TEST STAGE =====
.test_template: &test_template
  <<: *node_template
  stage: test
  coverage: '/All files[^|]*\|[^|]*\s+([\d\.]+)/'
  artifacts:
    when: always
    paths:
      - coverage/
    reports:
      junit: junit.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

unit-test:
  <<: *test_template
  script:
    - npm test -- --coverage --reporters=default --reporters=jest-junit
  parallel:
    matrix:
      - NODE_VERSION: ["16", "18", "20"]

integration-test:
  <<: *test_template
  services:
    - postgres:15
    - redis:7
  variables:
    POSTGRES_DB: testdb
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
    DATABASE_URL: "postgresql://postgres:postgres@postgres:5432/testdb"
    REDIS_URL: "redis://redis:6379"
  script:
    - npm run test:integration

e2e-test:
  <<: *node_template
  stage: test
  image: cypress/base:${NODE_VERSION}
  cache:
    key:
      files:
        - package-lock.json
    paths:
      - .npm
      - .cypress
  services:
    - name: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
      alias: app
  variables:
    BASE_URL: http://app:3000
  script:
    - npm run test:e2e -- --headless
  artifacts:
    when: always
    paths:
      - cypress/videos
      - cypress/screenshots
    expire_in: 7 days
  needs:
    - docker-build

# ===== BUILD STAGE =====
build:
  <<: *node_template
  stage: build
  script:
    - npm run build
  artifacts:
    paths:
      - dist/
    expire_in: 1 week

docker-build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  variables:
    DOCKER_TLS_CERTDIR: "/certs"
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    # Build and push Docker image
    - docker build -t $DOCKER_IMAGE -t $CI_REGISTRY_IMAGE:latest .
    - docker push $DOCKER_IMAGE
    - docker push $CI_REGISTRY_IMAGE:latest

    # Run Trivy security scan
    - apk add --no-cache curl
    - curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    - trivy image --severity HIGH,CRITICAL --exit-code 1 $DOCKER_IMAGE
  dependencies:
    - build

# ===== DEPLOY STAGE =====
.deploy_template: &deploy_template
  stage: deploy
  image: alpine/k8s:1.28.3
  before_script:
    - kubectl config use-context $KUBE_CONTEXT

deploy-staging:
  <<: *deploy_template
  variables:
    KUBE_CONTEXT: staging
    KUBE_NAMESPACE: staging
  script:
    - kubectl set image deployment/myapp myapp=$DOCKER_IMAGE -n $KUBE_NAMESPACE
    - kubectl rollout status deployment/myapp -n $KUBE_NAMESPACE
    - |
      # Smoke test
      kubectl run smoke-test --rm -i --restart=Never --image=curlimages/curl -- \
        curl -f http://myapp.$KUBE_NAMESPACE.svc.cluster.local/health
  environment:
    name: staging
    url: https://staging.example.com
    on_stop: stop-staging
  only:
    - develop

deploy-production:
  <<: *deploy_template
  variables:
    KUBE_CONTEXT: production
    KUBE_NAMESPACE: production
  script:
    - kubectl set image deployment/myapp myapp=$DOCKER_IMAGE -n $KUBE_NAMESPACE
    - kubectl rollout status deployment/myapp -n $KUBE_NAMESPACE
  environment:
    name: production
    url: https://example.com
  when: manual
  only:
    - main

stop-staging:
  <<: *deploy_template
  variables:
    KUBE_CONTEXT: staging
    KUBE_NAMESPACE: staging
  script:
    - kubectl delete all -l app=myapp -n $KUBE_NAMESPACE
  when: manual
  environment:
    name: staging
    action: stop
```

**Key Features**:
- ✅ YAML anchors for template reuse
- ✅ Parallel matrix testing across Node versions
- ✅ Service containers (PostgreSQL, Redis)
- ✅ Cypress E2E testing
- ✅ Docker build with Trivy scanning
- ✅ Kubernetes deployment with smoke tests
- ✅ Environment stop action

---

## 2. Python with Poetry

Production pipeline for Python applications using Poetry, pytest, and containerization.

### `.gitlab-ci.yml`

```yaml
# GitLab CI/CD for Python Application

variables:
  PYTHON_VERSION: "3.11"
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  POETRY_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pypoetry"
  POETRY_VERSION: "1.7.0"

stages:
  - quality
  - test
  - build
  - deploy

# ===== TEMPLATES =====
.python_template: &python_template
  image: python:${PYTHON_VERSION}
  cache:
    key:
      files:
        - poetry.lock
    paths:
      - .cache/pip
      - .cache/pypoetry
      - .venv
  before_script:
    - pip install poetry==${POETRY_VERSION}
    - poetry config virtualenvs.in-project true
    - poetry install --no-interaction --no-ansi

# ===== QUALITY STAGE =====
lint:
  <<: *python_template
  stage: quality
  script:
    - poetry run ruff check .

format-check:
  <<: *python_template
  stage: quality
  script:
    - poetry run black --check .

type-check:
  <<: *python_template
  stage: quality
  script:
    - poetry run mypy .

security-check:
  <<: *python_template
  stage: quality
  script:
    - poetry run bandit -r src/ -f json -o bandit-report.json
    - poetry run safety check --json --output safety-report.json
  artifacts:
    reports:
      sast: bandit-report.json
    paths:
      - safety-report.json
  allow_failure: true

# ===== TEST STAGE =====
test:
  <<: *python_template
  stage: test
  parallel:
    matrix:
      - PYTHON_VERSION: ["3.10", "3.11", "3.12"]
  services:
    - postgres:15
    - redis:7
  variables:
    POSTGRES_DB: testdb
    POSTGRES_USER: testuser
    POSTGRES_PASSWORD: testpass
    DATABASE_URL: "postgresql://testuser:testpass@postgres:5432/testdb"
    REDIS_URL: "redis://redis:6379/0"
  script:
    # Run migrations
    - poetry run alembic upgrade head

    # Run tests with coverage
    - poetry run pytest -v --cov=src --cov-report=xml --cov-report=term --junit-xml=report.xml
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'
  artifacts:
    when: always
    reports:
      junit: report.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - coverage.xml

# ===== BUILD STAGE =====
build-package:
  <<: *python_template
  stage: build
  script:
    - poetry build
  artifacts:
    paths:
      - dist/
    expire_in: 1 month

build-docker:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  variables:
    DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    # Multi-stage Docker build
    - docker build
        --build-arg PYTHON_VERSION=${PYTHON_VERSION}
        --build-arg POETRY_VERSION=${POETRY_VERSION}
        -t $DOCKER_IMAGE
        -t $CI_REGISTRY_IMAGE:latest
        .
    - docker push $DOCKER_IMAGE
    - docker push $CI_REGISTRY_IMAGE:latest

    # Container scanning
    - apk add --no-cache curl
    - curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    - trivy image --format json --output trivy-report.json $DOCKER_IMAGE
  artifacts:
    reports:
      container_scanning: trivy-report.json

# ===== DEPLOY STAGE =====
deploy-staging:
  stage: deploy
  image: alpine/helm:3.13.1
  variables:
    KUBE_NAMESPACE: staging
  script:
    - helm upgrade --install myapp ./helm
        --namespace $KUBE_NAMESPACE
        --set image.tag=$CI_COMMIT_SHORT_SHA
        --set ingress.host=staging.example.com
        --wait
  environment:
    name: staging
    url: https://staging.example.com
    kubernetes:
      namespace: staging
  only:
    - develop

deploy-production:
  stage: deploy
  image: alpine/helm:3.13.1
  variables:
    KUBE_NAMESPACE: production
  script:
    - helm upgrade --install myapp ./helm
        --namespace $KUBE_NAMESPACE
        --set image.tag=$CI_COMMIT_SHORT_SHA
        --set ingress.host=example.com
        --set replicaCount=3
        --wait
  environment:
    name: production
    url: https://example.com
    kubernetes:
      namespace: production
  when: manual
  only:
    - main
```

**Key Features**:
- ✅ Poetry with caching
- ✅ Parallel Python version testing
- ✅ Alembic database migrations
- ✅ SAST with Bandit, Safety check
- ✅ Container scanning with Trivy
- ✅ Helm deployment to Kubernetes

---

## 3. Go Microservices

Pipeline for Go microservices with parallel builds, testing, and multi-service deployment.

### `.gitlab-ci.yml`

```yaml
# GitLab CI/CD for Go Microservices

variables:
  GO_VERSION: "1.21"
  GOPATH: $CI_PROJECT_DIR/.go
  GOCACHE: $CI_PROJECT_DIR/.cache/go-build

stages:
  - quality
  - test
  - build
  - deploy

# ===== TEMPLATES =====
.go_template: &go_template
  image: golang:${GO_VERSION}
  cache:
    key: go-mod-${CI_COMMIT_REF_SLUG}
    paths:
      - .go/pkg/mod
      - .cache/go-build
  before_script:
    - mkdir -p .go
    - go mod download

# ===== QUALITY STAGE =====
quality:
  <<: *go_template
  stage: quality
  parallel:
    matrix:
      - CHECK: [vet, fmt, staticcheck, gosec]
  script:
    - |
      case $CHECK in
        vet)
          go vet ./...
          ;;
        fmt)
          test -z "$(gofmt -l .)"
          ;;
        staticcheck)
          go install honnef.co/go/tools/cmd/staticcheck@latest
          staticcheck ./...
          ;;
        gosec)
          go install github.com/securego/gosec/v2/cmd/gosec@latest
          gosec -fmt json -out gosec-report.json ./...
          ;;
      esac
  artifacts:
    when: always
    paths:
      - gosec-report.json
    reports:
      sast: gosec-report.json
    expire_in: 7 days

# ===== TEST STAGE =====
test:
  <<: *go_template
  stage: test
  parallel:
    matrix:
      - SERVICE: [api, worker, scheduler]
  services:
    - postgres:15
    - redis:7
    - nats:2
  variables:
    POSTGRES_DB: testdb
    POSTGRES_USER: test
    POSTGRES_PASSWORD: test
    DATABASE_URL: "postgres://test:test@postgres:5432/testdb?sslmode=disable"
    REDIS_URL: "redis://redis:6379"
    NATS_URL: "nats://nats:4222"
  script:
    - cd services/$SERVICE
    - go test -v -race -coverprofile=coverage.out -covermode=atomic ./...
    - go tool cover -func=coverage.out
  coverage: '/total:\s+\(statements\)\s+(\d+\.\d+)%/'
  artifacts:
    paths:
      - services/$SERVICE/coverage.out
    reports:
      coverage_report:
        coverage_format: cobertura
        path: services/$SERVICE/coverage.xml

# ===== BUILD STAGE =====
build:
  <<: *go_template
  stage: build
  parallel:
    matrix:
      - SERVICE: [api, worker, scheduler]
        GOOS: [linux]
        GOARCH: [amd64, arm64]
  script:
    - cd services/$SERVICE
    - CGO_ENABLED=0 GOOS=$GOOS GOARCH=$GOARCH go build
        -ldflags="-s -w -X main.version=$CI_COMMIT_SHORT_SHA"
        -o ../../bin/$SERVICE-$GOOS-$GOARCH
        ./cmd/$SERVICE
  artifacts:
    paths:
      - bin/
    expire_in: 1 week

docker-build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  parallel:
    matrix:
      - SERVICE: [api, worker, scheduler]
  variables:
    DOCKER_IMAGE: $CI_REGISTRY_IMAGE/$SERVICE:$CI_COMMIT_SHORT_SHA
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    # Multi-platform build
    - docker buildx create --use
    - docker buildx build
        --platform linux/amd64,linux/arm64
        --build-arg SERVICE=$SERVICE
        -t $DOCKER_IMAGE
        -t $CI_REGISTRY_IMAGE/$SERVICE:latest
        --push
        services/$SERVICE
  needs:
    - build

# ===== DEPLOY STAGE =====
deploy-services:
  stage: deploy
  image: bitnami/kubectl:1.28
  parallel:
    matrix:
      - SERVICE: [api, worker, scheduler]
        ENVIRONMENT: [staging]
  variables:
    KUBE_NAMESPACE: $ENVIRONMENT
  script:
    - kubectl config use-context $KUBE_CONTEXT
    - kubectl set image deployment/$SERVICE
        $SERVICE=$CI_REGISTRY_IMAGE/$SERVICE:$CI_COMMIT_SHORT_SHA
        -n $KUBE_NAMESPACE
    - kubectl rollout status deployment/$SERVICE -n $KUBE_NAMESPACE
  environment:
    name: $ENVIRONMENT/$SERVICE
    kubernetes:
      namespace: $ENVIRONMENT
  only:
    - develop

deploy-production:
  stage: deploy
  image: bitnami/kubectl:1.28
  variables:
    KUBE_NAMESPACE: production
  script:
    # Deploy all services to production
    - |
      for service in api worker scheduler; do
        kubectl set image deployment/$service \
          $service=$CI_REGISTRY_IMAGE/$service:$CI_COMMIT_SHORT_SHA \
          -n $KUBE_NAMESPACE
        kubectl rollout status deployment/$service -n $KUBE_NAMESPACE
      done
  environment:
    name: production
    kubernetes:
      namespace: production
  when: manual
  only:
    - main
```

**Key Features**:
- ✅ Parallel quality checks (vet, fmt, staticcheck, gosec)
- ✅ Multi-service testing with shared services
- ✅ Cross-compilation (amd64, arm64)
- ✅ Multi-platform Docker builds with buildx
- ✅ Coordinated microservices deployment

---

## 4. Monorepo with Trigger Rules

Advanced monorepo pipeline with change detection and selective job execution.

### `.gitlab-ci.yml`

```yaml
# GitLab CI/CD for Monorepo

variables:
  DOCKER_DRIVER: overlay2

stages:
  - detect
  - quality
  - test
  - build
  - deploy

# ===== CHANGE DETECTION =====
detect-changes:
  stage: detect
  image: alpine/git
  script:
    - |
      # Detect changes compared to main branch
      git fetch origin main
      CHANGED_FILES=$(git diff --name-only origin/main...HEAD)

      echo "CHANGED_FILES:"
      echo "$CHANGED_FILES"

      # Detect which services changed
      echo "frontend=false" > changes.env
      echo "backend=false" >> changes.env
      echo "shared=false" >> changes.env
      echo "infra=false" >> changes.env

      if echo "$CHANGED_FILES" | grep -q "^packages/frontend/"; then
        echo "frontend=true" > changes.env
      fi
      if echo "$CHANGED_FILES" | grep -q "^packages/backend/"; then
        echo "backend=true" >> changes.env
      fi
      if echo "$CHANGED_FILES" | grep -q "^packages/shared/"; then
        echo "shared=true" >> changes.env
      fi
      if echo "$CHANGED_FILES" | grep -q "^infrastructure/"; then
        echo "infra=true" >> changes.env
      fi

      cat changes.env
  artifacts:
    reports:
      dotenv: changes.env
  only:
    - merge_requests
    - main
    - develop

# ===== SHARED LIBRARY =====
shared-lib:
  stage: quality
  image: node:18
  cache:
    key: shared-${CI_COMMIT_REF_SLUG}
    paths:
      - packages/shared/node_modules
  script:
    - cd packages/shared
    - npm ci
    - npm run lint
    - npm run typecheck
    - npm test
    - npm run build
  artifacts:
    paths:
      - packages/shared/dist/
    expire_in: 1 hour
  rules:
    - if: $shared == "true"
    - if: $CI_COMMIT_BRANCH == "main"
    - if: $CI_COMMIT_BRANCH == "develop"

# ===== FRONTEND =====
frontend-quality:
  stage: quality
  image: node:18
  cache:
    key: frontend-${CI_COMMIT_REF_SLUG}
    paths:
      - packages/frontend/node_modules
  script:
    - cd packages/frontend
    - npm ci
    - npm run lint
    - npm run typecheck
  rules:
    - if: $frontend == "true"
    - changes:
        - packages/frontend/**/*
        - packages/shared/**/*

frontend-test:
  stage: test
  image: node:18
  cache:
    key: frontend-${CI_COMMIT_REF_SLUG}
    paths:
      - packages/frontend/node_modules
  script:
    - cd packages/frontend
    - npm ci
    - npm test -- --coverage
  coverage: '/All files[^|]*\|[^|]*\s+([\d\.]+)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: packages/frontend/coverage/cobertura-coverage.xml
  rules:
    - if: $frontend == "true"
    - changes:
        - packages/frontend/**/*

frontend-build:
  stage: build
  image: node:18
  dependencies:
    - shared-lib
  script:
    - cd packages/frontend
    - npm ci
    - npm run build
  artifacts:
    paths:
      - packages/frontend/dist/
  rules:
    - if: $frontend == "true"
    - changes:
        - packages/frontend/**/*

# ===== BACKEND =====
backend-quality:
  stage: quality
  image: python:3.11
  cache:
    key: backend-${CI_COMMIT_REF_SLUG}
    paths:
      - packages/backend/.venv
  before_script:
    - cd packages/backend
    - pip install poetry
    - poetry config virtualenvs.in-project true
    - poetry install
  script:
    - poetry run ruff check .
    - poetry run mypy .
  rules:
    - if: $backend == "true"
    - changes:
        - packages/backend/**/*
        - packages/shared/**/*

backend-test:
  stage: test
  image: python:3.11
  services:
    - postgres:15
  variables:
    POSTGRES_DB: testdb
    POSTGRES_USER: test
    POSTGRES_PASSWORD: test
    DATABASE_URL: "postgresql://test:test@postgres:5432/testdb"
  cache:
    key: backend-${CI_COMMIT_REF_SLUG}
    paths:
      - packages/backend/.venv
  before_script:
    - cd packages/backend
    - pip install poetry
    - poetry config virtualenvs.in-project true
    - poetry install
  script:
    - poetry run pytest --cov=src --cov-report=xml
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: packages/backend/coverage.xml
  rules:
    - if: $backend == "true"
    - changes:
        - packages/backend/**/*

backend-build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  variables:
    DOCKER_IMAGE: $CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHORT_SHA
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - cd packages/backend
    - docker build -t $DOCKER_IMAGE -t $CI_REGISTRY_IMAGE/backend:latest .
    - docker push $DOCKER_IMAGE
    - docker push $CI_REGISTRY_IMAGE/backend:latest
  rules:
    - if: $backend == "true"
    - changes:
        - packages/backend/**/*

# ===== INFRASTRUCTURE =====
terraform-validate:
  stage: quality
  image: hashicorp/terraform:1.6
  script:
    - cd infrastructure
    - terraform init -backend=false
    - terraform fmt -check -recursive
    - terraform validate
  rules:
    - if: $infra == "true"
    - changes:
        - infrastructure/**/*

# ===== DEPLOY =====
deploy-staging:
  stage: deploy
  image: alpine/helm:3.13.1
  variables:
    KUBE_NAMESPACE: staging
  script:
    - helm upgrade --install monorepo ./helm
        --namespace $KUBE_NAMESPACE
        --set frontend.image.tag=$CI_COMMIT_SHORT_SHA
        --set backend.image.tag=$CI_COMMIT_SHORT_SHA
        --wait
  environment:
    name: staging
    url: https://staging.example.com
  rules:
    - if: $CI_COMMIT_BRANCH == "develop"
      when: on_success

deploy-production:
  stage: deploy
  image: alpine/helm:3.13.1
  variables:
    KUBE_NAMESPACE: production
  script:
    - helm upgrade --install monorepo ./helm
        --namespace $KUBE_NAMESPACE
        --set frontend.image.tag=$CI_COMMIT_SHORT_SHA
        --set backend.image.tag=$CI_COMMIT_SHORT_SHA
        --set replicaCount=3
        --wait
  environment:
    name: production
    url: https://example.com
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
      when: manual
```

**Key Features**:
- ✅ Dynamic change detection with dotenv artifacts
- ✅ Rules-based job execution
- ✅ Service-specific caching
- ✅ Shared library artifact passing
- ✅ Multi-package monorepo support
- ✅ Terraform infrastructure validation

---

## Best Practices Summary

### 1. Caching Strategy
```yaml
cache:
  key:
    files:
      - package-lock.json  # Cache based on lock file
  paths:
    - .npm
    - node_modules
```

### 2. Parallel Execution
```yaml
parallel:
  matrix:
    - NODE_VERSION: ["16", "18", "20"]
      OS: [ubuntu, alpine]
```

### 3. Service Containers
```yaml
services:
  - postgres:15
  - redis:7
variables:
  POSTGRES_DB: testdb
  DATABASE_URL: "postgresql://user:pass@postgres:5432/testdb"
```

### 4. Change Detection
```yaml
rules:
  - changes:
      - packages/frontend/**/*
  - if: $CI_COMMIT_BRANCH == "main"
```

---

For related documentation, see:
- [github-actions-reference.md](github-actions-reference.md)
- [terraform-cicd-integration.md](terraform-cicd-integration.md)
- [security-automation-workflows.md](security-automation-workflows.md)
