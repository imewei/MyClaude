---
version: "1.0.3"
category: "cicd-automation"
command: "/workflow-automate"

execution-modes:
  quick-start:
    description: "Fast CI/CD bootstrap for new projects"
    time: "10-15 minutes"
    scope: "Single workflow type (GitHub Actions OR GitLab CI)"
    sections: "1, 2 or 4"
    use-case: "Quick pipeline setup, MVP projects, simple automation"
    templates: "Predefined, minimal customization"

  standard:
    description: "Production-ready multi-stage pipeline"
    time: "30-45 minutes"
    scope: "Complete CI/CD with testing, building, deployment, security"
    sections: "1-7"
    use-case: "Standard production pipeline setup (default)"
    templates: "Customizable with best practices"

  enterprise:
    description: "Complete automation with compliance and IaC"
    time: "60-120 minutes"
    scope: "Multi-platform, infrastructure, security, compliance, documentation"
    sections: "All 10 sections"
    use-case: "Enterprise CI/CD, regulated industries, complete automation"
    templates: "Fully customized with compliance validation"
    integration: "Terraform + Security + Compliance automation"

documentation:
  analysis-framework: "../docs/cicd-automation/workflow-analysis-framework.md"
  github-actions: "../docs/cicd-automation/github-actions-reference.md"
  gitlab-ci: "../docs/cicd-automation/gitlab-ci-reference.md"
  terraform-integration: "../docs/cicd-automation/terraform-cicd-integration.md"
  security-workflows: "../docs/cicd-automation/security-automation-workflows.md"
  orchestration: "../docs/cicd-automation/workflow-orchestration-patterns.md"
---

# Workflow Automation Expert

Create efficient CI/CD pipelines, GitHub Actions workflows, and automated development processes that reduce manual work, improve consistency, and accelerate delivery.

## Requirements
$ARGUMENTS

## Workflow Automation Sections

| Section | Scope | Quick-Start | Standard | Enterprise | Documentation |
|---------|-------|-------------|----------|------------|---------------|
| 1. Analysis | Project analysis & tech detection | ✅ | ✅ | ✅ | [→ Framework](../docs/cicd-automation/workflow-analysis-framework.md) |
| 2. GitHub Actions | Multi-stage CI/CD pipeline | ✅* | ✅ | ✅ | [→ Reference](../docs/cicd-automation/github-actions-reference.md) |
| 3. Release Automation | Semantic versioning, changelogs | - | ✅ | ✅ | - |
| 4. GitLab CI | GitLab pipeline generation | ✅* | ✅ | ✅ | [→ Reference](../docs/cicd-automation/gitlab-ci-reference.md) |
| 5. Terraform | Infrastructure as Code | - | - | ✅ | [→ Integration](../docs/cicd-automation/terraform-cicd-integration.md) |
| 6. Security | SAST/DAST, scanning | - | ✅ | ✅ | [→ Workflows](../docs/cicd-automation/security-automation-workflows.md) |
| 7. Monitoring | Observability automation | - | ✅ | ✅ | - |
| 8. Documentation | Auto-docs generation | - | - | ✅ | - |
| 9. Compliance | Compliance automation | - | - | ✅ | - |
| 10. Orchestration | Workflow coordination | - | ✅ | ✅ | [→ Patterns](../docs/cicd-automation/workflow-orchestration-patterns.md) |

*Quick-start mode: GitHub Actions OR GitLab CI (user selects platform)

---

## 1. Workflow Analysis

Analyze existing processes and identify automation opportunities.

### WorkflowAnalyzer Interface
```python
class WorkflowAnalyzer:
    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """Analyze project to identify automation opportunities"""
        return {
            'current_workflows': self._find_existing_workflows(project_path),
            'manual_processes': self._identify_manual_processes(project_path),
            'automation_opportunities': self._generate_recommendations(analysis),
            'build_process': self._analyze_build_process(project_path),
            'test_process': self._analyze_test_process(project_path),
            'deployment_process': self._analyze_deployment_process(project_path)
        }
```

**Detection Coverage**:
- **Existing Workflows**: GitHub Actions (.github/workflows/*.yml), GitLab CI (.gitlab-ci.yml), Jenkins (Jenkinsfile)
- **Manual Processes**: Build/deploy/release scripts, README manual steps
- **Recommendations**: CI/CD setup, build/test/deployment automation by priority

[→ Complete WorkflowAnalyzer Implementation](../docs/cicd-automation/workflow-analysis-framework.md)

---

## 2. GitHub Actions Workflows

Create comprehensive multi-stage CI/CD pipelines.

### Pipeline Structure
```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline
on: [push, pull_request, release]
env:
  NODE_VERSION: '18'
  PYTHON_VERSION: '3.11'

jobs:
  quality:     # Linting, type checking, security audit, license check
  test:        # Unit + integration tests across OS/Node matrix
  build:       # Multi-environment builds (dev/staging/prod)
  deploy:      # ECS/Kubernetes deployment with environment gates
  verify:      # Smoke tests, E2E tests, performance validation
```

**Key Features**:
- **Matrix Builds**: Test across ubuntu/windows/macos + Node 16/18/20
- **Caching**: npm/yarn cache with `actions/cache@v3`
- **Security**: Docker image scanning with Trivy, SARIF uploads
- **Artifacts**: Build artifacts with 7-day retention
- **Environments**: staging/production with approval gates
- **Notifications**: Slack notifications on deployment

[→ Complete Pipeline Examples](../docs/cicd-automation/github-actions-reference.md)

---

## 3. Release Automation

Automate semantic versioning and changelog generation.

### Semantic Release Workflow
```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    branches: [main]

jobs:
  release:
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - run: npx semantic-release
```

### Release Configuration
```javascript
// .releaserc.js
module.exports = {
  branches: ['main', { name: 'beta', prerelease: true }],
  plugins: [
    '@semantic-release/commit-analyzer',
    '@semantic-release/release-notes-generator',
    ['@semantic-release/changelog', { changelogFile: 'CHANGELOG.md' }],
    '@semantic-release/npm',
    '@semantic-release/git',
    '@semantic-release/github'
  ]
};
```

**Capabilities**: Automated version bumps, CHANGELOG.md generation, GitHub releases, npm publishing

---

## 4. GitLab CI Pipeline

Generate complete GitLab CI/CD pipelines.

### Pipeline Template
```yaml
# .gitlab-ci.yml
stages: [quality, test, build, deploy]

variables:
  NODE_VERSION: "18"

quality:
  stage: quality
  script:
    - npm ci
    - npm run lint
    - npm run typecheck

test:
  stage: test
  parallel:
    matrix:
      - NODE_VERSION: [16, 18, 20]
  script:
    - npm ci
    - npm test -- --coverage

build:
  stage: build
  script:
    - npm run build
  artifacts:
    paths: [dist/]
    expire_in: 1 week
```

[→ Complete GitLab CI Patterns](../docs/cicd-automation/gitlab-ci-reference.md)

---

## 5. Terraform CI/CD Integration

Automate infrastructure provisioning in CI/CD.

### Terraform Workflow
```yaml
# .github/workflows/terraform.yml
name: Terraform
on:
  pull_request:
    paths: ['terraform/**']
  push:
    branches: [main]

jobs:
  terraform:
    steps:
      - uses: hashicorp/setup-terraform@v2
      - run: terraform fmt -check -recursive
      - run: terraform init
      - run: terraform validate
      - run: terraform plan -out=tfplan
      - run: terraform apply tfplan  # Only on main branch
```

**Features**:
- Format checking, validation, plan preview on PRs
- Automated apply on main branch merges
- PR comments with plan summary
- Remote state management (S3 backend)

[→ Complete Terraform Integration](../docs/cicd-automation/terraform-cicd-integration.md)

---

## 6. Security Automation

Automate security scanning and compliance.

### Security Scanning Workflow
```yaml
# .github/workflows/security.yml
name: Security Scan
on:
  push:
  pull_request:
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  security-scan:
    steps:
      - uses: aquasecurity/trivy-action@master     # Vulnerability scanning
      - uses: snyk/actions/node@master             # Dependency scanning
      - uses: dependency-check/Dependency-Check_Action@main  # OWASP check
      - uses: SonarSource/sonarcloud-github-action@master    # Code quality
      - uses: returntocorp/semgrep-action@v1       # SAST
      - uses: gitleaks/gitleaks-action@v2          # Secret scanning
```

**Coverage**: Vulnerabilities, dependencies, OWASP Top 10, code quality, secrets, SAST/DAST

[→ Complete Security Workflows](../docs/cicd-automation/security-automation-workflows.md)

---

## 7. Monitoring Automation

Automate monitoring stack deployment.

### Monitoring Stack
```yaml
# .github/workflows/monitoring.yml
jobs:
  deploy-monitoring:
    steps:
      - uses: azure/setup-helm@v3
      - run: |
          helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
            --namespace monitoring --values monitoring/prometheus-values.yaml
          kubectl apply -f monitoring/dashboards/
          kubectl apply -f monitoring/alerts/
```

**Components**: Prometheus, Grafana, Alertmanager, custom dashboards, alert rules

---

## 8. Dependency Update Automation

Automate dependency updates with Renovate/Dependabot.

### Renovate Configuration
```json
{
  "extends": ["config:base", ":automergeMinor"],
  "schedule": ["after 10pm every weekday"],
  "vulnerabilityAlerts": { "automerge": true },
  "packageRules": [
    { "matchDepTypes": ["devDependencies"], "automerge": true },
    { "matchPackagePatterns": ["^@types/"], "automerge": true },
    { "matchPackagePatterns": ["^eslint"], "groupName": "eslint packages" }
  ],
  "prConcurrentLimit": 3
}
```

**Features**: Automated PRs, security vulnerability auto-merge, grouped updates, scheduling

---

## 9. Documentation Automation

Automate API docs and architecture diagrams.

### Documentation Workflow
```yaml
# .github/workflows/docs.yml
jobs:
  generate-docs:
    steps:
      - run: npm run docs:api          # TypeDoc API docs
      - run: npm run docs:typescript   # TypeScript docs
      - run: mmdc -i docs/architecture.mmd -o docs/architecture.png  # Mermaid diagrams
      - run: npm run docs:build
      - uses: peaceiris/actions-gh-pages@v3  # Deploy to GitHub Pages
```

**Generates**: API reference, TypeScript docs, architecture diagrams, deployment guides

---

## 10. Workflow Orchestration

Manage complex multi-step workflows with intelligent orchestration.

### WorkflowOrchestrator Interface
```typescript
interface WorkflowStep {
  name: string;
  type: 'parallel' | 'sequential';
  steps?: WorkflowStep[];
  action?: () => Promise<any>;
  retries?: number;
  timeout?: number;
  condition?: () => boolean;
  onError?: 'fail' | 'continue' | 'retry';
}

class WorkflowOrchestrator extends EventEmitter {
  async execute(workflow: WorkflowStep): Promise<WorkflowResult> {
    // Executes workflow with:
    // - Parallel/sequential execution
    // - Retry logic with exponential backoff
    // - Timeout handling
    // - Conditional step execution
    // - Error handling strategies
    // - Event emission for monitoring
  }
}
```

**Example Deployment Workflow**:
```typescript
const deploymentWorkflow = {
  name: 'deployment',
  type: 'sequential',
  steps: [
    {
      name: 'pre-deployment',
      type: 'parallel',
      steps: [
        { name: 'backup-database', timeout: 300000 },
        { name: 'health-check', retries: 3 }
      ]
    },
    {
      name: 'deployment',
      type: 'sequential',
      steps: [
        { name: 'blue-green-switch', onError: 'retry', retries: 2 },
        { name: 'smoke-tests', onError: 'fail' }
      ]
    },
    {
      name: 'post-deployment',
      type: 'parallel',
      steps: [
        { name: 'notify-teams', onError: 'continue' },
        { name: 'update-monitoring' }
      ]
    }
  ]
};
```

[→ Complete Orchestration Patterns](../docs/cicd-automation/workflow-orchestration-patterns.md)

---

## Execution Parameters

### Platform Selection (Quick-Start Mode)
- `--platform=github`: Generate GitHub Actions workflows
- `--platform=gitlab`: Generate GitLab CI pipelines
- `--platform=both`: Generate both platforms

### Customization Options
- `--environment=[dev,staging,prod]`: Target environments
- `--security-level=[basic,standard,high]`: Security scanning depth
- `--compliance=[none,soc2,hipaa,pci]`: Compliance requirements
- `--iac-tool=[terraform,cloudformation,pulumi]`: IaC tool selection

### Examples
```bash
# Quick-start: GitHub Actions for new project
/workflow-automate --mode=quick-start --platform=github

# Standard: Full CI/CD with security
/workflow-automate --mode=standard --security-level=high

# Enterprise: Complete automation with compliance
/workflow-automate --mode=enterprise --compliance=soc2 --iac-tool=terraform
```

---

## Output Deliverables

### Quick-Start Mode
1. Single CI/CD workflow file (GitHub Actions OR GitLab CI)
2. Basic security scanning
3. Simple deployment script
4. Quick setup guide

### Standard Mode
1. Multi-stage CI/CD pipeline (quality, test, build, deploy, verify)
2. Release automation with semantic versioning
3. Security scanning workflows
4. Monitoring setup
5. Pre-commit hooks
6. Development environment setup script
7. Workflow orchestration patterns

### Enterprise Mode
1. **All Standard deliverables** plus:
2. Terraform infrastructure automation
3. Compliance validation workflows
4. Automated documentation generation
5. Dependency update automation (Renovate/Dependabot)
6. Advanced monitoring with dashboards and alerts
7. Multi-environment deployment strategies
8. Comprehensive implementation guide
9. Security compliance reports

---

## Implementation Sequence

1. **Run Workflow Analysis** → Identify current state and automation gaps
2. **Select Execution Mode** → quick-start/standard/enterprise based on requirements
3. **Generate Core CI/CD** → Primary pipeline for builds, tests, deployments
4. **Add Security Layer** → Scanning, SAST/DAST, dependency checks
5. **Integrate IaC** (if enterprise) → Terraform/CloudFormation automation
6. **Setup Monitoring** → Observability, dashboards, alerts
7. **Configure Automation** → Releases, dependencies, documentation
8. **Deploy Orchestration** → Complex workflow management
9. **Validate End-to-End** → Test complete automation flow
10. **Document & Handoff** → Implementation guide, runbooks

---

## Success Criteria

- ✅ CI/CD pipeline runs successfully on first commit
- ✅ All stages complete within defined time budgets (quick: 10-15min, standard: 30-45min, enterprise: 60-120min)
- ✅ Security scans identify and block critical vulnerabilities
- ✅ Automated deployments to all target environments
- ✅ Monitoring dashboards live and collecting metrics
- ✅ Zero manual steps required for standard workflows
- ✅ Documentation generated and published automatically
- ✅ Team onboarded and able to use automation independently

Focus on creating reliable, maintainable automation that reduces manual work while maintaining quality and security standards.
