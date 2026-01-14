---
name: deployment-pipeline-design
version: "1.0.7"
maturity: "5-Expert"
specialization: CI/CD Architecture
description: Design multi-stage CI/CD pipelines with approval gates, security checks, and progressive delivery (rolling, blue-green, canary, feature flags). Use when architecting deployment workflows, implementing GitOps, or establishing multi-environment promotion strategies.
---

# Deployment Pipeline Design

Multi-stage CI/CD architecture with approval gates and deployment strategies.

---

## Pipeline Stages

```
Build → Test → Staging → Approve → Production → Verify
```

| Stage | Purpose | Actions |
|-------|---------|---------|
| Build | Compile, containerize | Docker build, push |
| Test | Validate | Unit, integration, security scan |
| Staging | Pre-prod validation | Deploy, E2E tests |
| Approve | Gate | Manual/automated approval |
| Production | Release | Canary/blue-green deploy |
| Verify | Confirm | Health checks, metrics |

---

## Deployment Strategies

| Strategy | Rollback | Downtime | Best For |
|----------|----------|----------|----------|
| Rolling | Minutes | Zero | Most applications |
| Blue-Green | Instant | Zero | High-risk deploys |
| Canary | Gradual | Zero | Traffic-based testing |
| Feature Flags | Instant | Zero | A/B testing |

### Rolling

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
```

### Canary (Argo Rollouts)

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
spec:
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {duration: 5m}
      - setWeight: 50
      - pause: {duration: 5m}
      - setWeight: 100
```

### Blue-Green

```bash
kubectl apply -f green-deployment.yaml
# Test green environment
kubectl label service my-app version=green
# Rollback: kubectl label service my-app version=blue
```

---

## Approval Gates

### GitHub Actions Environment

```yaml
deploy-production:
  needs: staging
  environment:
    name: production
    url: https://app.example.com
  steps:
    - run: kubectl apply -f k8s/production/
```

### GitLab Delayed Start

```yaml
deploy:production:
  when: delayed
  start_in: 30 minutes
  only: [main]
```

### Azure Multi-Approver

```yaml
- deployment: Deploy
  environment:
    name: production
  strategy:
    runOnce:
      preDeploy:
        steps:
        - task: ManualValidation@0
          inputs:
            notifyUsers: 'team-leads@example.com'
```

---

## Complete Pipeline Example

```yaml
name: Production Pipeline
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t myapp:${{ github.sha }} .
      - run: docker push myapp:${{ github.sha }}

  test:
    needs: build
    steps:
      - run: make test
      - run: trivy image myapp:${{ github.sha }}

  deploy-staging:
    needs: test
    environment: staging
    steps:
      - run: kubectl apply -f k8s/staging/

  deploy-production:
    needs: deploy-staging
    environment: production
    steps:
      - run: kubectl apply -f k8s/production/

  verify:
    needs: deploy-production
    steps:
      - run: curl -f https://app.example.com/health
```

---

## Rollback

### Automated

```yaml
- name: Health check
  id: health
  run: curl -sf https://app.example.com/health || exit 1

- name: Rollback on failure
  if: failure()
  run: kubectl rollout undo deployment/my-app
```

### Manual

```bash
kubectl rollout history deployment/my-app
kubectl rollout undo deployment/my-app --to-revision=3
```

---

## Metrics (DORA)

| Metric | Target |
|--------|--------|
| Deployment Frequency | Daily+ |
| Lead Time | < 1 hour |
| Change Failure Rate | < 15% |
| MTTR | < 1 hour |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Fail fast | Quick tests first |
| Parallel jobs | Independent stages concurrent |
| Caching | Reuse dependencies |
| Environment parity | Consistent configs |
| Secrets management | Use Vault/AWS Secrets |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| No rollback plan | Automate rollback on failure |
| Manual deployments | Full automation with gates |
| Inconsistent environments | Infrastructure as code |
| Skipping staging | Always deploy to staging first |

---

## Checklist

- [ ] Multi-stage pipeline defined
- [ ] Approval gates for production
- [ ] Deployment strategy selected (rolling/canary/blue-green)
- [ ] Automated rollback configured
- [ ] Health checks after deployment
- [ ] DORA metrics tracked

---

**Version**: 1.0.5
