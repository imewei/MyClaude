---
name: deployment-pipelines
version: "1.0.0"
description: Design and implement robust CI/CD pipelines using GitHub Actions, GitLab CI, and Jenkins. Covers multi-stage workflows, security scanning, secrets management, and progressive delivery (canary, blue-green).
---

# Deployment Pipelines

Expert guide for building automated, secure, and reliable delivery pipelines for modern software.

## 1. Pipeline Architecture

### Core Stages
- **Build**: Compile, lint, and containerize applications.
- **Test**: Run unit, integration, and security scans (Trivy, Snyk).
- **Staging**: Deploy to a pre-production environment for E2E validation.
- **Approval**: Manual or automated gates before production.
- **Production**: Gradual rollout with automated health checks.

### Parallelization
- **Matrix Builds**: Run tests across multiple OS/language versions concurrently.
- **Test Sharding**: Split large test suites across parallel runners to reduce lead time.
- **Fan-out/Fan-in**: Execute independent tasks (e.g., frontend and backend builds) in parallel.

## 2. Progressive Delivery Strategies

| Strategy | Rollback | Best For |
|----------|----------|----------|
| **Rolling** | Minutes | Standard updates, zero-downtime. |
| **Blue-Green**| Instant | High-risk changes, infrastructure swaps. |
| **Canary** | Gradual | Traffic-based validation of new features. |

## 3. Security & Secrets Management

- **Environment Secrets**: Store sensitive data in encrypted environment variables (e.g., GitHub Secrets).
- **Least Privilege**: Use short-lived tokens and OIDC for cloud provider authentication.
- **Scanning**: Integrate SAST/DAST and dependency scanning into every push.

## 4. Automation & Rollback Checklist

- [ ] **Fail Fast**: Run the fastest, most critical tests first.
- [ ] **Health Checks**: Implement automated verification after every deployment step.
- [ ] **Rollback**: Configure `kubectl rollout undo` or equivalent automated triggers on failure.
- [ ] **Artifact Versioning**: Ensure every build produces a unique, immutable artifact.
