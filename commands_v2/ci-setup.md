---
title: "CI Setup"
description: "CI/CD pipeline setup and automation for multiple platforms"
category: devops
subcategory: pipeline-automation
complexity: intermediate
argument-hint: "[--platform=github|gitlab|jenkins] [--type=basic|security|enterprise] [--deploy=staging|production|both] [--monitoring] [--security] [--agents=auto|core|scientific|engineering|ai|domain|quality|research|all] [--implement] [--dry-run] [--backup] [--rollback] [--intelligent] [--orchestrate] [--parallel] [--validate]"
allowed-tools: "*"
model: inherit
tags: ci-cd, devops, automation, github-actions, gitlab-ci, jenkins
dependencies: []
related: [run-all-tests, check-code-quality, commit, fix-commit-errors, optimize, generate-tests]
workflows: [ci-cd-pipeline, deployment-automation, quality-gates]
version: "2.1"
last-updated: "2025-09-29"
---

# CI/CD Pipeline Setup

Set up CI/CD pipelines with automated testing, building, and deployment: **$ARGUMENTS**

## Quick Start

```bash
# Basic GitHub Actions pipeline
/ci-setup --platform=github --type=basic

# Enterprise pipeline with security
/ci-setup --platform=github --type=enterprise --security --monitoring

# Multi-environment deployment
/ci-setup --platform=gitlab --deploy=both --monitoring
```

## Arguments

- `--platform` - CI/CD platform: github, gitlab, jenkins
- `--type` - Pipeline type: basic, security, enterprise
- `--deploy` - Deployment target: staging, production, both
- `--monitoring` - Enable monitoring and observability
- `--security` - Enable security scanning and compliance
- `--agents=<agents>` - Agent selection (auto, core, scientific, engineering, ai, domain, quality, research, all)
- `--implement` - Automatically create and configure CI/CD pipelines
- `--dry-run` - Preview pipeline configuration without creating
- `--backup` - Create backup before modifying CI/CD configuration
- `--rollback` - Enable rollback capability for failed configurations
- `--intelligent` - Enable intelligent agent selection based on project analysis
- `--orchestrate` - Enable advanced 23-agent orchestration for complex setups
- `--parallel` - Run setup steps in parallel for efficiency
- `--validate` - Validate pipeline configuration and test execution

## Agent Integration

### DevOps Security Agent (`devops-security-engineer`)
- **Secure CI/CD**: Security-integrated pipeline design and automation
- **Infrastructure Security**: Cloud security architecture and compliance
- **Container Security**: Kubernetes hardening and runtime protection
- **Compliance Automation**: SOC 2, ISO 27001, regulatory compliance
- **Incident Response**: Automated security response and orchestration

### Quality Agent (`code-quality-master`)
- **Quality Gates**: Automated testing and validation in pipelines
- **Performance Testing**: Load testing and regression detection
- **Code Analysis**: Static analysis and security scanning integration
- **Accessibility Testing**: WCAG compliance automation
- **Build Optimization**: CI/CD performance and efficiency optimization

### Orchestrator Agent (`multi-agent-orchestrator`)
- **Pipeline Coordination**: Multi-stage workflow orchestration
- **Resource Management**: Intelligent allocation of CI/CD resources
- **Fault Tolerance**: Pipeline failure recovery and resilience
- **Performance Monitoring**: CI/CD system efficiency tracking
- **Workflow Optimization**: Pipeline performance and reliability enhancement

## Agent Selection Options

- `auto` - Intelligent agent selection based on project requirements
- `core` - Essential multi-agent team for standard CI/CD
- `scientific` - Scientific computing and research pipeline setup
- `engineering` - Software engineering and DevOps focus
- `ai` - AI/ML pipeline and deployment setup
- `domain` - Domain-specific specialized pipelines
- `quality` - Quality engineering focus for testing and validation
- `research` - Research workflow and publication pipelines
- `all` - Complete 23-agent CI/CD system with specialized expertise

## Safety Features

### Dry-Run Mode
Preview pipeline configuration before creating:
```bash
/ci-setup --dry-run --implement --platform=github --type=basic
```

### Backup and Rollback
Automatic backup before changes:
```bash
# Create backup before pipeline setup
/ci-setup --backup --implement --platform=github

# Enable rollback if setup fails
/ci-setup --backup --rollback --implement --platform=github --type=enterprise
```

The system creates versioned backups of existing CI/CD configurations with:
- Timestamp-based naming
- Git commit before changes (if available)
- Automatic restoration on failure

### Validation
Validate pipeline configuration after setup:
```bash
/ci-setup --implement --validate --platform=github --type=security
```

### Safe Setup Workflow
```bash
# 1. Preview pipeline configuration
/ci-setup --dry-run --implement --platform=github --type=basic

# 2. Review configuration output

# 3. Apply with safety features
/ci-setup --backup --rollback --validate --implement --platform=github --type=basic

# 4. Pipeline is tested and validated automatically
```

## Pipeline Types

### Basic Pipeline
- Code checkout and dependency installation
- Unit and integration tests
- Build and package application
- Deploy to single environment

### Security Pipeline
- Static code analysis (SAST)
- Dependency vulnerability scanning
- Container security scanning
- Policy compliance checks

### Enterprise Pipeline
- Multi-environment deployments
- Advanced security scanning
- Quality gates and code coverage
- Monitoring and alerting
- Infrastructure as code

## Platform Support

### GitHub Actions
- Workflow files in `.github/workflows/`
- Environment protection rules
- Secrets management
- Matrix builds for multiple environments

### GitLab CI/CD
- `.gitlab-ci.yml` configuration
- Pipeline variables and environments
- Container registry integration
- Auto DevOps capabilities

### Jenkins
- Declarative pipeline scripts
- Multi-branch pipeline support
- Plugin recommendations
- Build agent configuration

## Security Features

- Static Application Security Testing (SAST)
- Dynamic Application Security Testing (DAST)
- Dependency vulnerability scanning
- Container image security analysis
- Infrastructure security assessment
- Compliance reporting

## Monitoring Integration

- Application performance monitoring
- Infrastructure metrics collection
- Log aggregation and analysis
- Alert configuration
- Dashboard creation

## Deployment Strategies

- **Blue-Green**: Zero-downtime deployments with environment switching
- **Canary**: Gradual rollout with traffic splitting
- **Rolling**: Sequential instance updates
- **Feature Flags**: Controlled feature rollouts

## Examples

### Basic GitHub Workflow
```yaml
name: CI/CD Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: npm test
      - name: Build
        run: npm run build

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        run: echo "Deploying application"
```

### Enterprise Security Pipeline
```yaml
name: Enterprise Pipeline
on:
  push:
    branches: [main, develop]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Security scan
        run: npm audit
      - name: SAST analysis
        run: semgrep --config=auto .

  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Code quality
        run: npm run lint
      - name: Test coverage
        run: npm run test:coverage

  deploy:
    needs: [security, quality]
    strategy:
      matrix:
        environment: [staging, production]
    environment: ${{ matrix.environment }}
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to ${{ matrix.environment }}
        run: echo "Deploying to ${{ matrix.environment }}"
```

## Related Commands

**Prerequisites**: Commands to run before CI/CD setup
- `/generate-tests --coverage=90` - Ensure comprehensive test coverage
- `/check-code-quality --auto-fix` - Fix quality issues before automation
- `/run-all-tests --auto-fix` - Ensure all tests pass before CI setup
- `/commit --validate` - Ensure clean repository state

**Alternatives**: Different automation approaches
- Manual deployment processes
- Platform-specific CI tools (Actions only, GitLab only)
- Third-party CI services (CircleCI, Travis)

**Combinations**: Commands that work with CI setup
- `/optimize --implement` - Optimize before setting up performance monitoring
- `/fix-commit-errors` - Fix CI failures automatically
- `/update-docs --type=api` - Document CI/CD processes
- `/commit --template=ci` - Commit CI configuration changes

**Follow-up**: Commands to run after CI setup
- `/fix-commit-errors` - Monitor and fix CI failures
- `/run-all-tests --auto-fix` - Maintain test suite health
- `/reflection --type=instruction` - Analyze CI/CD effectiveness
- `/update-docs --type=readme` - Document deployment procedures

ARGUMENTS: [--platform=github|gitlab|jenkins] [--type=basic|security|enterprise] [--deploy=staging|production|both] [--monitoring] [--security] [--agents=auto|core|scientific|engineering|ai|domain|quality|research|all] [--implement] [--dry-run] [--intelligent] [--orchestrate] [--parallel] [--validate]