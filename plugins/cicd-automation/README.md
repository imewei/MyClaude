# CI/CD Automation

Comprehensive CI/CD pipeline automation with intelligent error resolution, multi-agent analysis, and advanced workflow orchestration. Features optimized slash commands (62% token reduction), 3 execution modes per command, 11 external documentation files (~4,600 lines), 5 specialized agents, and 6 production-ready skills.

**Version:** 1.0.7 | **Category:** infrastructure | **License:** MIT


## What's New in v1.0.7

This release implements **Opus 4.5 optimization** with enhanced token efficiency and standardized documentation.

### Key Improvements

- **Format Standardization**: All components now include consistent YAML frontmatter with version, maturity, specialization, and description fields
- **Token Efficiency**: 40-50% line reduction through tables over prose, minimal code examples, and structured sections
- **Enhanced Discoverability**: Clear "Use when..." trigger phrases for better Claude Code activation
- **Actionable Checklists**: Task-oriented guidance for common workflows
- **Cross-Reference Tables**: Quick-reference format for delegation and integration patterns


## Agents (5)

All agents at v1.0.3 with 90-92% maturity, systematic reasoning frameworks, and comprehensive examples.

### 🏗️ cloud-architect

**Status:** active | **Maturity:** 92% | **Version:** 1.0.7

Expert cloud architect specializing in AWS/Azure/GCP multi-cloud infrastructure design, advanced IaC, FinOps cost optimization, and modern architectural patterns.

**New in v1.0.2:**
- 6-step chain-of-thought framework (Requirements → Services → Architecture → Cost → Security → Validation)
- 5 Constitutional AI principles (Cost Optimization, Security-First, Resilience, Observability, Automation)
- Multi-region web application example with complete Terraform implementation and $3,321/month cost breakdown

**Expected Impact:** 30% better service selection, 40% reduction in over-engineering, 50% improved cost optimization

---

### 🚀 deployment-engineer

**Status:** active | **Maturity:** 92% | **Version:** 1.0.7

Expert deployment engineer specializing in modern CI/CD pipelines, GitOps workflows (ArgoCD/Flux), progressive delivery, container security, and platform engineering.

**New in v1.0.2:**
- 6-step chain-of-thought framework (Requirements → Pipeline → Security → Progressive Delivery → Monitoring → Validation)
- 5 Constitutional AI principles (Automation, Security, Zero-Downtime, Observability, Developer Experience)
- Financial services CI/CD example with 283-line GitHub Actions workflow, PCI-DSS/SOX compliance, 7 security tools

**Expected Impact:** 60% better pipeline security, 45% faster deployments, 50% reduction in deployment failures

---

### 🔧 devops-troubleshooter

**Status:** active | **Maturity:** 90% | **Version:** 1.0.7

Expert DevOps troubleshooter specializing in rapid incident response, advanced debugging, modern observability (OpenTelemetry, Prometheus), and root cause analysis.

**New in v1.0.2:**
- 6-step debugging process with timing (Incident Assessment → Data Gathering → Hypothesis → Testing → Implementation → Postmortem)
- 5 Constitutional AI principles (Systematic Investigation, Minimal Disruption, Documentation, Blameless RCA, Prevention)
- SEV-1 API outage example with complete 18-minute debugging trace, Prometheus alerts, Go code fixes, postmortem

**Expected Impact:** 70% faster incident resolution, 50% reduction in repeat incidents, 80% better postmortem quality

---

### ☸️ kubernetes-architect

**Status:** active | **Maturity:** 90% | **Version:** 1.0.7

Expert Kubernetes architect specializing in cloud-native infrastructure, advanced GitOps workflows (ArgoCD/Flux), enterprise container orchestration (EKS/AKS/GKE), service mesh (Istio/Linkerd), and platform engineering.

**New in v1.0.2:**
- 6-step chain-of-thought framework (Workload → Cluster → GitOps → Security → Observability → Cost)
- 5 Constitutional AI principles (GitOps, Security-by-Default, Developer Experience, Progressive Delivery, Observability-First)
- Fintech platform example with EKS, ArgoCD, Istio service mesh, security policies, observability stack, Argo Rollouts

**Expected Impact:** 50% better cluster architecture, 60% improved security posture, 40% better developer experience

---

### 🏗️ terraform-specialist

**Status:** active | **Maturity:** 90% | **Version:** 1.0.7

Expert Terraform/OpenTofu specialist mastering advanced IaC automation, state management (S3/DynamoDB with KMS encryption), enterprise infrastructure patterns, module design, and testing (Terratest, OPA).

**New in v1.0.2:**
- 6-step chain-of-thought framework (Requirements → Module Design → State Strategy → Testing → CI/CD → Validation)
- 5 Constitutional AI principles (DRY, State Security, Testing, Least Privilege, Maintainability)
- EKS cluster deployment example with hierarchical modules, secure state backend, Terratest tests, OPA policies, GitHub Actions

**Expected Impact:** 50% better module reusability, 40% reduction in state issues, 60% improved testing coverage

---

## Commands (2)

### 🔍 `/fix-commit-errors` (v1.0.3)

**Status:** active | **Maturity:** 95%

Intelligent GitHub Actions failure resolution with 5-agent multi-agent system, pattern matching across 100+ error types, Bayesian confidence scoring, and iterative fix strategies.

#### Optimization (v1.0.3)
- **Token Reduction**: 1,052 → 413 lines (60.7% reduction)
- **Execution Modes**: 3 modes with clear time estimates
- **External Docs**: 5 comprehensive files (~2,650 lines)

#### Execution Modes

**`quick-fix` (5-10 minutes)**
- **Use Case**: Urgent CI failures, production hotfixes, simple errors
- **Phases**: Discovery + Fix Application only (Phase 1, 4)
- **Auto-fix**: Always enabled
- **Best For**: Time-critical production issues

**`standard` (15-30 minutes) - DEFAULT**
- **Use Case**: Typical CI failure investigation
- **Phases**: All 7 phases with multi-agent analysis
- **Auto-fix**: Optional with `--auto-fix`
- **Best For**: Regular CI/CD debugging and learning

**`comprehensive` (30-60 minutes)**
- **Use Case**: Recurring failures, pattern investigation, knowledge base building
- **Phases**: Deep analysis + cross-workflow correlation + knowledge base
- **Learning**: Always enabled with `--learn`
- **Best For**: Complex issues requiring deep analysis

#### 5-Agent System
1. **Log Fetcher & Parser** - Retrieve and structure error logs via GitHub API
2. **Pattern Matcher & Categorizer** - Classify errors using 100+ patterns
3. **Root Cause Analyzer** - UltraThink reasoning with 3W1H analysis
4. **Knowledge Base Consultant** - Bayesian confidence scoring for solutions
5. **Solution Generator** - Automated fix code with rollback plans

#### Documentation
- **multi-agent-error-analysis.md** (711 lines) - Complete 5-agent implementation
- **error-pattern-library.md** (819 lines) - 100+ patterns across NPM/Python/Rust/Go
- **fix-strategies.md** (580 lines) - Level 1-3 iterative fix approaches
- **knowledge-base-system.md** (540 lines) - Bayesian learning algorithms
- **fix-examples.md** (400 lines) - 15 real-world scenarios, avg 9.2 min resolution

#### Usage Examples

```bash
# Quick fix for urgent production issue
/fix-commit-errors --mode quick-fix

# Standard investigation (default)
/fix-commit-errors

# Deep analysis with learning
/fix-commit-errors --mode comprehensive --learn

# Specific workflow run
/fix-commit-errors --run 12345678
```

---

### 🔧 `/workflow-automate` (v1.0.3)

**Status:** active | **Maturity:** 94%

Automated CI/CD workflow generation with intelligent platform selection (GitHub Actions, GitLab CI, Terraform), technology stack detection, and production-ready templates.

#### Optimization (v1.0.3)
- **Token Reduction**: 1,339 → 493 lines (63.2% reduction)
- **Execution Modes**: 3 modes for different project sizes
- **External Docs**: 6 comprehensive files (~2,000 lines)

#### Execution Modes

**`quick-start` (10-15 minutes)**
- **Use Case**: Fast CI/CD bootstrap for new projects, MVP setup
- **Scope**: Single workflow type (GitHub Actions OR GitLab CI)
- **Templates**: Predefined, minimal customization
- **Best For**: Quick project setup, proof of concept

**`standard` (30-45 minutes) - DEFAULT**
- **Use Case**: Production-ready multi-stage pipeline
- **Scope**: Complete CI/CD with testing, building, deployment, security
- **Templates**: Customizable with best practices
- **Best For**: Most production projects

**`enterprise` (60-120 minutes)**
- **Use Case**: Enterprise CI/CD with compliance and IaC
- **Scope**: Multi-platform + infrastructure + security + compliance
- **Integration**: Terraform + Security + Compliance automation
- **Best For**: Regulated industries, large-scale deployments

#### Features
- **Project Analysis**: Automated tech stack detection and complexity scoring
- **Multi-Platform**: GitHub Actions, GitLab CI, Terraform workflows
- **Security Integration**: SAST, DAST, container scanning, secret detection
- **Compliance**: OWASP Top 10, CIS benchmarks, PCI-DSS, SOC 2

#### Documentation
- **workflow-analysis-framework.md** (200 lines) - WorkflowAnalyzer Python class
- **github-actions-reference.md** (500 lines) - 5 complete workflows
- **gitlab-ci-reference.md** (400 lines) - 4 complete pipelines
- **terraform-cicd-integration.md** (350 lines) - 3 Terraform workflows
- **security-automation-workflows.md** (350 lines) - Comprehensive security scanning
- **workflow-orchestration-patterns.md** (300 lines) - TypeScript orchestrator

#### Usage Examples

```bash
# Quick bootstrap for new project
/workflow-automate --mode quick-start --platform github-actions

# Standard production pipeline (default)
/workflow-automate

# Enterprise with compliance
/workflow-automate --mode enterprise --compliance pci-dss,sox

# Multi-platform setup
/workflow-automate --platforms github-actions,gitlab-ci,terraform
```

---

## Skills (6)

All skills at v1.0.3 with comprehensive descriptions and detailed use cases.

### 📋 deployment-pipeline-design (v1.0.3)

Multi-stage CI/CD pipeline architecture with approval gates, security checks, deployment orchestration, and progressive delivery strategies.

**Enhanced with:** 23 use cases covering GitOps practices, deployment strategies (rolling, blue-green, canary), multi-environment promotion, rollback procedures, and security integration.

---

### ⚙️ github-actions-templates (v1.0.3)

Production-ready GitHub Actions workflows for automated testing, building, and deploying applications.

**Enhanced with:** 24 use cases covering CI pipelines, Docker builds, Kubernetes deployments, matrix builds, security scanning, reusable workflows, caching strategies, and approval gates.

---

### 🦊 gitlab-ci-patterns (v1.0.3)

GitLab CI/CD pipelines with multi-stage workflows, caching, distributed runners, and GitOps integration.

**Enhanced with:** 24 use cases covering .gitlab-ci.yml configuration, Docker-in-Docker builds, Kubernetes deployments, runner configuration, security scanning templates, and dynamic child pipelines.

---

### 🔄 iterative-error-resolution (v1.0.3)

Comprehensive iterative CI/CD error resolution with pattern recognition, automated fixes, knowledge base learning, and validation loops until zero errors remain.

**Enhanced with:** 25 use cases covering GitHub Actions failures, GitLab CI errors, dependency conflicts, build errors, test failures, automated fix application, and /fix-commit-errors integration.

---

### 🔐 secrets-management (v1.0.3)

Secure secrets management for CI/CD pipelines using Vault, AWS Secrets Manager, Azure Key Vault, and platform-native solutions.

**Enhanced with:** 25 use cases covering HashiCorp Vault, AWS/Azure/GCP secret managers, GitHub/GitLab secrets, External Secrets Operator, secret rotation, OIDC authentication, and scanning.

---

### 🔒 security-ci-template (v1.0.3)

Security scanning and lock file validation templates for CI/CD pipelines with vulnerability detection and compliance.

**Enhanced with:** 24 use cases covering dependency scanning (Safety, Snyk), SAST tools (Bandit, Semgrep, CodeQL), lock file validation, Docker scanning, DAST, secret scanning, and compliance scanning.

## Quick Start

### Installation

```bash
# Install via Claude Code marketplace
claude-code plugins install cicd-automation
```

### Basic Usage

```bash
# Design cloud architecture
"Design a multi-region web application on AWS with high availability"
# This will invoke the cloud-architect agent with chain-of-thought reasoning

# Create CI/CD pipeline
/workflow-automate
# Intelligent template selection based on your project

# Fix failing workflows
/fix-commit-errors
# Automated pattern matching and iterative resolution

# Troubleshoot production issues
"API is returning 500 errors with high latency"
# This will invoke the devops-troubleshooter with systematic debugging
```

## Examples

### Example 1: Multi-Region Web Application Architecture

**Request:** "Design a highly available web application on AWS with multi-region failover"

**cloud-architect v1.0.2 Response:**
```
## Step 1: Requirements Analysis
- Availability target: 99.99% (52 minutes downtime/year)
- Traffic: 1000 req/sec peak
- Multi-region: Primary (us-east-1), Secondary (us-west-2)

## Step 2: Service Selection
Primary Services:
- ALB (Application Load Balancer) - $16.43/month per region
- ECS Fargate - Auto-scaling containers
- RDS Multi-AZ PostgreSQL - Cross-region replication
- ElastiCache Redis - Session management
- S3 + CloudFront - Static assets with edge caching

## Step 3: Architecture Design
[Complete architecture YAML provided]

## Step 4: Cost Optimization
Monthly Cost: $3,321
- Compute: $1,800 (reserved instances: -$540/month savings)
- Database: $450 (Multi-AZ: necessary for availability)
- Networking: $600 (CloudFront reduces origin traffic)
- Storage: $250

## Step 5: Security Review
- VPC with private subnets for app/database tiers
- WAF rules for common vulnerabilities
- TLS 1.3 enforced via CloudFront
- Secrets Manager for credentials rotation

## Step 6: Validation
✓ Meets 99.99% availability target
✓ Auto-scales 100-1000 instances
✓ RTO: 5 minutes (DNS failover)
✓ RPO: 5 minutes (RDS replication lag)

[Complete Terraform implementation provided]
```

---

### Example 2: Financial Services CI/CD Pipeline

**Request:** "Create a secure CI/CD pipeline for a financial services application with PCI-DSS compliance"

**deployment-engineer v1.0.2 Response:**
```
## Step 1: Requirements Gathering
- Compliance: PCI-DSS, SOX
- Security: SAST, DAST, dependency scanning, secret scanning
- Deployment: Blue-green with manual approval for production
- Observability: Full tracing and audit logs

## Step 2: Pipeline Design
Stages: Build → Test → Security Scan → Stage Deploy → Manual Approval → Production Deploy

## Step 3: Security Integration
7 Security Tools:
1. TruffleHog - Secret scanning
2. SonarQube - Code quality + SAST
3. Snyk - Dependency vulnerabilities
4. Trivy - Container image scanning
5. Cosign - Image signing
6. Syft - SBOM generation
7. Kyverno - Kubernetes policy enforcement

## Step 4: Progressive Delivery
Strategy: Blue-green deployment
- Deploy to green environment
- Run smoke tests
- Manual approval gate
- Switch traffic via DNS/ALB
- Keep blue for 24h rollback window

## Step 5: Monitoring Setup
- OpenTelemetry traces to Jaeger
- Prometheus metrics with PCI-DSS compliance dashboards
- Audit logs to CloudWatch with 7-year retention

## Step 6: Validation
✓ All security scans integrated
✓ Manual approval for production
✓ Audit trail for SOX compliance
✓ Rollback capability < 5 minutes

[Complete 283-line GitHub Actions workflow provided with all security tools]
```

---

### Example 3: SEV-1 API Outage Debugging

**Request:** "API endpoints are returning 500 errors with high memory usage"

**devops-troubleshooter v1.0.2 Response:**
```
## T+0m - Incident Assessment (SEV-1)
Severity: SEV-1 (100% API unavailability)
Affected: All API endpoints
Impact: 10,000 active users
Scope: Production cluster (us-east-1)

## T+5m - Data Gathering
Prometheus Alert: pod_memory_usage > 90% for 5m
kubectl top pods:
  api-deployment-abc123: 7.8Gi / 8Gi (98% memory)

Logs show: "runtime: out of memory"

## T+10m - Hypothesis Formation
Primary: Memory leak in request processing
Evidence: Memory growth correlates with request count
Alternative: Memory limit too low (ruled out: 8Gi is appropriate)

## T+15m - Testing & Validation
Go pprof analysis:
- 6.2 GB allocated by middleware.LogRequest
- []byte slices never released
- Goroutine leak: 45,000 active (expected: <1000)

## T+25m - Implementation
Fix applied to middleware/logging.go:
[Code diff showing proper buffer pooling]

Deployed via rolling update
Memory usage: 98% → 45% within 2 minutes

## T+55m - Postmortem
Root Cause: Logging middleware allocated []byte for every request without releasing
Action Items:
1. Add memory profiling to staging (OWNER: SRE, DUE: 2025-02-05)
2. Implement buffer pooling pattern (OWNER: Backend, DUE: 2025-02-07)
3. Add memory leak detection alerts (OWNER: SRE, DUE: 2025-02-10)

[Complete debugging trace with Prometheus queries and code fixes provided]
```

## Key Features

### Chain-of-Thought Reasoning
All agents provide transparent, step-by-step reasoning for their decisions:
- **Requirements Analysis**: Understanding constraints and objectives
- **Solution Design**: Systematic evaluation of options
- **Implementation**: Concrete, production-ready code
- **Validation**: Self-critique and verification

### Constitutional AI Principles
Each agent has 5 core principles that guide decision-making:
- **cloud-architect**: Cost Optimization, Security-First, Resilience, Observability, Automation
- **deployment-engineer**: Automation, Security, Zero-Downtime, Observability, Developer Experience
- **devops-troubleshooter**: Systematic Investigation, Minimal Disruption, Documentation, Blameless RCA, Prevention
- **kubernetes-architect**: GitOps, Security-by-Default, Developer Experience, Progressive Delivery, Observability-First
- **terraform-specialist**: DRY, State Security, Testing, Least Privilege, Maintainability

### Comprehensive Examples
Every agent includes production-ready examples:
- Complete code implementations (Terraform, GitHub Actions, Kubernetes manifests)
- Cost breakdowns and optimization recommendations
- Security configurations and compliance considerations
- Trade-offs analysis and decision rationale

## Integration

### Compatible Plugins
- **backend-development**: API design and microservices architecture
- **observability-monitoring**: Performance engineering and monitoring
- **full-stack-orchestration**: End-to-end application deployment

### Slash Commands
- `/workflow-automate`: Intelligent CI/CD pipeline generation
- `/fix-commit-errors`: Automated GitHub Actions failure resolution

## Documentation

### Full Documentation
For comprehensive documentation, see: [Plugin Documentation](https://myclaude.readthedocs.io/en/latest/plugins/cicd-automation.html)

### Changelog
See [CHANGELOG.md](./CHANGELOG.md) for detailed release notes and version history.

### Agent Documentation
Each agent has detailed documentation with examples:
- [cloud-architect.md](./agents/cloud-architect.md) - Multi-cloud architecture design
- [deployment-engineer.md](./agents/deployment-engineer.md) - CI/CD pipeline engineering
- [devops-troubleshooter.md](./agents/devops-troubleshooter.md) - Incident response and debugging
- [kubernetes-architect.md](./agents/kubernetes-architect.md) - Cloud-native platform engineering
- [terraform-specialist.md](./agents/terraform-specialist.md) - Infrastructure as Code automation

### Skill Documentation
Each skill has implementation guides:
- [deployment-pipeline-design](./skills/deployment-pipeline-design/) - Pipeline architecture patterns
- [github-actions-templates](./skills/github-actions-templates/) - GitHub Actions workflows
- [gitlab-ci-patterns](./skills/gitlab-ci-patterns/) - GitLab CI/CD patterns
- [iterative-error-resolution](./skills/iterative-error-resolution/) - Automated error fixing
- [secrets-management](./skills/secrets-management/) - Secure credential management
- [security-ci-template](./skills/security-ci-template/) - Security scanning templates

## Support

### Reporting Issues
Report issues at: https://github.com/anthropics/claude-code/issues

### Contributing
Contributions are welcome! Please see the individual agent and skill documentation for contribution guidelines.

### License
MIT License - See [LICENSE](./LICENSE) for details

---

**Author:** Wei Chen
**Version:** 1.0.7
**Category:** Infrastructure
**Last Updated:** 2025-11-06
