---
name: ci-cd-pipelines
description: Meta-orchestrator for CI/CD pipeline design. Routes to GitHub Actions, GitLab CI, pipeline design, security scanning, and error resolution skills. Use when setting up GitHub Actions, GitLab CI, designing deployment pipelines, adding security scanning, or resolving CI/CD errors.
---

# CI/CD Pipelines

Orchestrator for continuous integration and delivery pipeline design. Routes to the appropriate specialized skill based on the platform, pipeline stage, or failure mode.

## Expert Agent

- **`automation-engineer`**: Specialist for pipeline architecture, deployment automation, and CI/CD reliability.
  - *Location*: `plugins/dev-suite/agents/automation-engineer.md`
  - *Capabilities*: Workflow design, parallelization, caching strategies, security gates, and iterative failure resolution.

## Core Skills

### [GitHub Actions Templates](../github-actions-templates/SKILL.md)
Reusable workflows, composite actions, matrix builds, and GitHub-hosted runner optimization.

### [GitLab CI Patterns](../gitlab-ci-patterns/SKILL.md)
`.gitlab-ci.yml` structure, DAG pipelines, GitLab runners, and artifact management.

### [Deployment Pipeline Design](../deployment-pipeline-design/SKILL.md)
Blue/green, canary, and rolling deployments with rollback strategies.

### [Security CI Template](../security-ci-template/SKILL.md)
SAST, dependency scanning, secrets detection, and SBOM generation in CI pipelines.

### [Iterative Error Resolution](../iterative-error-resolution/SKILL.md)
Systematic diagnosis of CI/CD pipeline failures, flaky CI tests, and build environment issues. Scoped to pipeline-level errors — for runtime application debugging, see `dev-workflows/debugging-toolkit`.

## Routing Decision Tree

```
What is the CI/CD concern?
|
+-- GitHub Actions workflow / composite action?
|   --> github-actions-templates
|
+-- GitLab CI YAML / DAG / runner config?
|   --> gitlab-ci-patterns
|
+-- Deployment strategy / rollback / promotion?
|   --> deployment-pipeline-design
|
+-- SAST / dependency scan / secrets detection?
|   --> security-ci-template
|
+-- Pipeline failure / flaky test / build error?
    --> iterative-error-resolution
```

## Routing Table

| Trigger                               | Sub-skill                      |
|---------------------------------------|--------------------------------|
| .github/workflows, actions, matrix    | github-actions-templates       |
| .gitlab-ci.yml, stages, needs, DAG    | gitlab-ci-patterns             |
| Blue/green, canary, rollback, deploy  | deployment-pipeline-design     |
| SAST, Snyk, Trivy, SBOM, secrets scan | security-ci-template           |
| CI pipeline error, flaky CI, timeout  | iterative-error-resolution     |

## Checklist

- [ ] Identify the CI platform (GitHub Actions vs GitLab CI) before templating
- [ ] Confirm deployment strategy matches the rollback tolerance of the service
- [ ] Verify security scanning gates block merges on critical findings
- [ ] Check that pipeline caching keys are deterministic and not over-broad
- [ ] Validate secrets are injected via environment variables, never hardcoded
- [ ] Ensure failed pipelines produce actionable log output for rapid diagnosis
