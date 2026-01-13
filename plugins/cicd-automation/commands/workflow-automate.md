---
version: "1.0.6"
category: "cicd-automation"
command: "/workflow-automate"
execution-modes:
  quick-start: "10-15m: Single workflow (GitHub/GitLab)"
  standard: "30-45m: Full pipeline + security + monitoring"
  enterprise: "60-120m: Multi-platform + IaC + compliance"
documentation:
  analysis-framework: "../docs/cicd-automation/workflow-analysis-framework.md"
  github-actions: "../docs/cicd-automation/github-actions-reference.md"
  gitlab-ci: "../docs/cicd-automation/gitlab-ci-reference.md"
  terraform-integration: "../docs/cicd-automation/terraform-cicd-integration.md"
  security-workflows: "../docs/cicd-automation/security-automation-workflows.md"
---

# Workflow Automation

$ARGUMENTS

## Mode Coverage

| Feature | Quick | Standard | Enterprise |
|---------|-------|----------|------------|
| Analysis + GitHub/GitLab* | ✅ | ✅ | ✅ |
| Release automation | - | ✅ | ✅ |
| Security scanning | - | ✅ | ✅ |
| Monitoring + orchestration | - | ✅ | ✅ |
| Terraform IaC + compliance | - | - | ✅ |

*Quick: Select one platform

## 1. Analysis

Detect existing workflows (`.github/workflows/*.yml`, `.gitlab-ci.yml`), manual processes (scripts, README), automation opportunities

## 2. GitHub Actions

**Pipeline:** quality → test → build → deploy → verify

- Quality: Lint, type check, security audit
- Test: Unit + integration, matrix (OS × versions)
- Build: Multi-environment, artifacts
- Deploy: ECS/K8s, environment gates
- Verify: Smoke tests, E2E, performance
- Features: Caching, Trivy scanning, notifications

## 3. Release Automation

semantic-release: commit-analyzer, changelog, npm, GitHub releases. Branches: `main`, `beta`

## 4. GitLab CI

**Stages:** quality → test → build → deploy

- Matrix testing: `parallel: matrix`
- Artifacts: `expire_in: 1 week`
- Environments: staging/production with approval

## 5. Terraform IaC (Enterprise)

fmt → init → validate → plan → apply (main only). PR comments, S3 remote state

## 6. Security

Trivy (vulnerabilities), Snyk (dependencies), OWASP, SonarCloud, Semgrep (SAST), Gitleaks (secrets). Run: push, PR, weekly

## 7. Monitoring

Prometheus + Grafana + Alertmanager (Helm). Custom dashboards, alerts

## 8. Dependency Updates

Renovate: Auto-merge minor/patch, dev deps, `@types/*`. Vulnerability auto-merge. Schedule: 10pm+ weekdays, max 3 PRs

## 9. Orchestration

Parallel (independent tasks), Sequential (deploy→tests), Retry (transient failures), Conditional (env-specific)

## Usage

```bash
/workflow-automate --mode=quick-start --platform=github
/workflow-automate --mode=standard --security-level=high
/workflow-automate --mode=enterprise --compliance=soc2 --iac-tool=terraform
```

**Options:** `--platform=github|gitlab|both`, `--environment=dev,staging,prod`, `--security-level=basic|standard|high`, `--compliance=none|soc2|hipaa|pci`

## Deliverables

- Quick: Single workflow, basic security, guide
- Standard: + Release automation, security scanning, monitoring, pre-commit hooks
- Enterprise: + Terraform IaC, compliance validation, auto-docs, dependency automation

## Success

CI/CD runs on first commit, security blocks critical vulns, automated deployments, monitoring active, zero manual steps
