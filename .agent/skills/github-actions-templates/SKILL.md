---
name: github-actions-templates
version: "1.0.7"
maturity: "5-Expert"
specialization: CI/CD Automation
description: Create production GitHub Actions workflows for testing, building, and deploying. Use when setting up CI pipelines, Docker builds, Kubernetes deployments, matrix builds, security scans, or reusable workflows.
---

# GitHub Actions Templates

Production-ready workflow patterns for CI/CD automation.

---

## Workflow Patterns

| Pattern | Trigger | Use Case |
|---------|---------|----------|
| Test | push, PR | Automated testing, linting |
| Build & Push | push tags | Docker images to registry |
| Deploy | push main | Kubernetes, cloud platforms |
| Matrix | push, PR | Multi-version testing |
| Security | push, PR | Vulnerability scanning |
| Reusable | workflow_call | Organization standards |

---

## Test Workflow

```yaml
name: Test
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [18.x, 20.x]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      - run: npm ci
      - run: npm run lint
      - run: npm test
      - uses: codecov/codecov-action@v3
```

---

## Docker Build & Push

```yaml
name: Build and Push
on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=semver,pattern={{version}}
            type=ref,event=branch
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## Kubernetes Deployment

```yaml
name: Deploy to K8s
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2
      - run: aws eks update-kubeconfig --name cluster --region us-west-2
      - run: |
          kubectl apply -f k8s/
          kubectl rollout status deployment/my-app -n production
```

---

## Security Scanning

```yaml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          format: 'sarif'
          output: 'trivy-results.sarif'
      - uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

---

## Reusable Workflow

```yaml
# .github/workflows/reusable-test.yml
name: Reusable Test
on:
  workflow_call:
    inputs:
      node-version:
        required: true
        type: string
    secrets:
      NPM_TOKEN:
        required: true

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
      - run: npm ci && npm test
```

**Usage**:
```yaml
jobs:
  call-test:
    uses: ./.github/workflows/reusable-test.yml
    with:
      node-version: '20.x'
    secrets:
      NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
```

---

## Production Deploy with Approval

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://app.example.com
    steps:
      - uses: actions/checkout@v4
      - run: echo "Deploying..."
      - uses: slackapi/slack-github-action@v1
        if: success()
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Pin versions | Use @v4, not @latest |
| Cache dependencies | `cache: 'npm'` in setup actions |
| Use secrets | Never hardcode credentials |
| Set permissions | Minimal required permissions |
| Matrix builds | Test multiple versions |
| Reusable workflows | Organization standards |
| Approval gates | Use environments for prod |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| @latest actions | Pin to specific versions |
| No caching | Use built-in cache options |
| Secrets in logs | Use `::add-mask::` |
| Missing permissions | Explicitly set `permissions:` |

---

## Checklist

- [ ] Actions pinned to specific versions
- [ ] Dependencies cached
- [ ] Secrets properly managed
- [ ] Permissions explicitly set
- [ ] Matrix builds for multi-version
- [ ] Security scanning enabled
- [ ] Production uses environments
- [ ] Notifications configured

---

**Version**: 1.0.5
