---
version: "1.0.6"
category: "cicd-automation"
command: "/workflow-automate"
execution-modes:
  quick-start:
    description: "Fast CI/CD bootstrap"
    time: "10-15 minutes"
    scope: "Single workflow (GitHub Actions OR GitLab CI)"
  standard:
    description: "Production-ready multi-stage pipeline"
    time: "30-45 minutes"
    scope: "Complete CI/CD with testing, security, deployment"
  enterprise:
    description: "Complete automation with compliance and IaC"
    time: "60-120 minutes"
    scope: "Multi-platform, infrastructure, security, compliance"
documentation:
  analysis-framework: "../docs/cicd-automation/workflow-analysis-framework.md"
  github-actions: "../docs/cicd-automation/github-actions-reference.md"
  gitlab-ci: "../docs/cicd-automation/gitlab-ci-reference.md"
  terraform-integration: "../docs/cicd-automation/terraform-cicd-integration.md"
  security-workflows: "../docs/cicd-automation/security-automation-workflows.md"
---

# Workflow Automation Expert

Create efficient CI/CD pipelines, GitHub Actions workflows, and automated development processes.

## Requirements

$ARGUMENTS

---

## Section Coverage by Mode

| Section | Quick | Standard | Enterprise |
|---------|-------|----------|------------|
| 1. Analysis | ✅ | ✅ | ✅ |
| 2. GitHub Actions | ✅* | ✅ | ✅ |
| 3. Release Automation | - | ✅ | ✅ |
| 4. GitLab CI | ✅* | ✅ | ✅ |
| 5. Terraform IaC | - | - | ✅ |
| 6. Security Scanning | - | ✅ | ✅ |
| 7. Monitoring | - | ✅ | ✅ |
| 8. Documentation | - | - | ✅ |
| 9. Compliance | - | - | ✅ |
| 10. Orchestration | - | ✅ | ✅ |

*Quick-start: GitHub Actions OR GitLab CI (user selects)

---

## 1. Workflow Analysis

**Detect:** Existing workflows, manual processes, automation opportunities

| Item | Sources |
|------|---------|
| Existing Workflows | `.github/workflows/*.yml`, `.gitlab-ci.yml`, `Jenkinsfile` |
| Manual Processes | Build/deploy scripts, README instructions |
| Recommendations | CI/CD setup, automation by priority |

---

## 2. GitHub Actions Pipeline

**Jobs:** quality → test → build → deploy → verify

| Stage | Components |
|-------|------------|
| Quality | Lint, type check, security audit, license check |
| Test | Unit + integration, matrix (OS × Node versions) |
| Build | Multi-environment (dev/staging/prod), artifacts |
| Deploy | ECS/K8s with environment gates |
| Verify | Smoke tests, E2E, performance |

**Features:** Matrix builds, caching, Docker scanning (Trivy), Slack notifications

---

## 3. Release Automation

**Tools:** semantic-release with commit-analyzer, changelog, npm, GitHub releases

**Branches:** `main` (releases), `beta` (prereleases)

---

## 4. GitLab CI Pipeline

**Stages:** quality → test → build → deploy

| Feature | Implementation |
|---------|----------------|
| Matrix testing | `parallel: matrix` with Node versions |
| Artifacts | `expire_in: 1 week` |
| Environments | staging/production with approval |

---

## 5. Terraform IaC (Enterprise)

**Workflow:** fmt check → init → validate → plan → apply (main only)

**Features:** PR comments with plan summary, remote state (S3)

---

## 6. Security Automation

| Tool | Purpose |
|------|---------|
| Trivy | Vulnerability scanning |
| Snyk | Dependency scanning |
| OWASP | Dependency check |
| SonarCloud | Code quality |
| Semgrep | SAST |
| Gitleaks | Secret scanning |

**Schedule:** On push, PR, weekly

---

## 7. Monitoring Automation

**Stack:** Prometheus + Grafana + Alertmanager via Helm

**Includes:** Custom dashboards, alert rules

---

## 8. Dependency Updates

**Renovate Config:**
- Auto-merge minor/patch, dev dependencies, `@types/*`
- Vulnerability alerts auto-merge
- Group ESLint packages
- Schedule: after 10pm weekdays
- Concurrent PR limit: 3

---

## 9. Orchestration Patterns

| Pattern | Use Case |
|---------|----------|
| Parallel | Independent pre-deployment tasks |
| Sequential | Deploy → smoke tests |
| Retry | Transient failures with backoff |
| Conditional | Environment-specific steps |

---

## Execution Options

```bash
# Quick-start
/workflow-automate --mode=quick-start --platform=github

# Standard
/workflow-automate --mode=standard --security-level=high

# Enterprise
/workflow-automate --mode=enterprise --compliance=soc2 --iac-tool=terraform
```

**Options:** `--platform=github|gitlab|both`, `--environment=dev,staging,prod`, `--security-level=basic|standard|high`, `--compliance=none|soc2|hipaa|pci`

---

## Deliverables by Mode

| Mode | Deliverables |
|------|--------------|
| Quick-Start | Single CI/CD workflow, basic security, setup guide |
| Standard | Multi-stage pipeline, release automation, security scanning, monitoring, pre-commit hooks |
| Enterprise | All Standard + Terraform IaC, compliance validation, auto-docs, dependency automation, comprehensive guide |

---

## Success Criteria

- ✅ CI/CD runs successfully on first commit
- ✅ Security scans block critical vulnerabilities
- ✅ Automated deployments to all environments
- ✅ Monitoring dashboards collecting metrics
- ✅ Zero manual steps for standard workflows
- ✅ Documentation generated automatically
