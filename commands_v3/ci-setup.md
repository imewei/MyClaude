---
description: Setup continuous integration and deployment pipeline
allowed-tools: Bash(git:*), Bash(find:*), Bash(grep:*)
argument-hint: [platform] [--security] [--docker]
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

# CI/CD Pipeline Setup

## Context
- Repository: !`git remote get-url origin 2>/dev/null || echo "Not a git repo"`
- Tech stack: !`find . -name "package.json" -o -name "requirements.txt" -o -name "Cargo.toml" -o -name "go.mod" 2>/dev/null | head -3`
- Existing CI: !`find .github .gitlab-ci.yml .circleci jenkins 2>/dev/null | head -5`

## Your Task: $ARGUMENTS

**Setup comprehensive CI/CD following this structure**:

### 1. Platform Selection
- **GitHub Actions**: Native integration, free for public repos
- **GitLab CI**: Built-in, comprehensive DevOps
- **CircleCI**: Fast cloud-based
- **Jenkins**: Self-hosted, customizable

### 2. Core Pipeline Stages
```yaml
stages: [test, build, security, deploy]

test:
  - Lint code (eslint/pylint/clippy)
  - Run unit tests
  - Generate coverage report
  - Matrix testing (multiple versions)

build:
  - Compile/bundle application
  - Build Docker image (if applicable)
  - Cache dependencies

security:
  - Dependency scan (npm audit/pip-audit)
  - SAST (CodeQL/SonarQube)
  - Container scan (Trivy/Snyk)
  - Secrets detection

deploy:
  - Staging (auto on develop)
  - Production (manual approval on main)
  - Rollback capability
```

### 3. Essential Optimizations
- **Caching**: Dependencies, build artifacts (~70% faster)
- **Parallel jobs**: Independent tests run simultaneously
- **Fail fast**: Stop on first critical failure
- **Conditional runs**: Skip unchanged areas

### 4. Security Best Practices
- Store secrets in vault (never in code)
- Scan dependencies for CVEs
- Sign commits and artifacts
- Audit trail for deployments
- Least privilege access

### 5. Quick Start Templates

**GitHub Actions** (`.github/workflows/ci.yml`):
```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with: {node-version: '18', cache: 'npm'}
      - run: npm ci && npm test && npm run build
```

**GitLab CI** (`.gitlab-ci.yml`):
```yaml
stages: [test, deploy]
test:
  script: npm ci && npm test
  cache: {paths: [node_modules]}
```

**Docker Multi-stage**:
```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine
COPY --from=builder /app/node_modules ./node_modules
COPY . .
CMD ["npm", "start"]
```

### 6. Implementation Checklist
- [ ] Branch protection (require PR reviews)
- [ ] Automated testing on PR
- [ ] Code coverage minimum (80%)
- [ ] Security scans pass
- [ ] Staging auto-deploy
- [ ] Production approval gate
- [ ] Rollback procedure tested
- [ ] Monitoring/alerts configured

**Provide complete, production-ready pipeline configuration with security best practices**