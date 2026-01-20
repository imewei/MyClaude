# Infrastructure & Ops Suite

Consolidated suite for CI/CD automation, observability monitoring, and Git PR workflows. Optimized for Claude Code v2.1.12 with parallel agent execution and MCP server integration.

## 🚀 Features

- **CI/CD Automation**: Intelligent pipeline generation and automated error resolution for GitHub Actions and GitLab CI.
- **Observability**: Complete monitoring stack setup including Prometheus, Grafana, and distributed tracing.
- **Advanced Git Workflows**: Streamlined collaboration with intelligent commits, atomic enforcement, and PR enhancement.
- **Cloud Architecture**: Scalable multi-cloud infrastructure design using Terraform and Kubernetes.
- **Site Reliability**: SRE best practices including SLO/SLI implementation and error budget management.
- **Team Onboarding**: Orchestrated onboarding workflows for new developers and projects.

## 🤖 Agents

| Agent | Specialization |
|-------|----------------|
| `devops-architect` | Cloud infrastructure (AWS/Azure/GCP), Kubernetes, and IaC |
| `automation-engineer` | CI/CD pipelines, Git workflows, and automation |
| `sre-expert` | Reliability, observability, and incident response |

## 🛠️ Commands

| Command | Description |
|---------|-------------|
| `/commit` | Intelligent git commit with automated analysis and quality validation |
| `/fix-commit-errors` | Automatically analyzes GitHub Actions failures and applies solutions |
| `/merge-all` | Merge all local branches into main and clean up |
| `/monitor-setup` | Set up Prometheus, Grafana, and distributed tracing stack |
| `/onboard` | Orchestrate complete onboarding for new team members |
| `/slo-implement` | Implement SLO/SLA monitoring, error budgets, and alerting |
| `/workflow-automate` | Automated CI/CD workflow generation for GitHub Actions and GitLab CI |
| `/code-analyze` | Semantic code analysis using Serena MCP for symbol navigation |
| `/github-assist` | GitHub operations using GitHub MCP for issues, PRs, and repos |

## 🛠️ Skills

- **Deployment Pipelines**: Robust CI/CD design and secrets management.
- **Git Workflow**: Advanced Git operations and collaborative patterns.
- **Observability**: Monitoring, logging, tracing, and SLO implementation.

## 📦 Installation

This plugin is designed for [Claude Code](https://claude.com/code).

```bash
# Add the marketplace
/plugin marketplace add imewei/MyClaude

# Install the suite
/plugin install infrastructure-suite@marketplace
```

**Note:** After installation, restart Claude Code for changes to take effect.

## 📜 License

MIT License
