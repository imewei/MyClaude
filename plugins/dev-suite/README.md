# Dev Suite

Unified software development lifecycle suite covering architecture, implementation, CI/CD, testing, debugging, and deployment. Merges engineering, infrastructure, and quality into one suite for zero-friction cross-concern agent delegation.

## Features

- **Architecture & Design**: Scalable backend systems, microservices, REST/GraphQL/gRPC APIs, and modernization planning.
- **Full-Stack Implementation**: Web, iOS, Android (React, Next.js, Flutter), systems programming (C/C++/Rust/Go), and CLI tools.
- **CI/CD & DevOps**: GitHub Actions, GitLab CI, Kubernetes, Terraform, and multi-cloud (AWS/Azure/GCP) infrastructure.
- **Testing & Quality**: Comprehensive test automation, code review, security auditing, and E2E testing with Playwright.
- **Debugging**: AI-assisted root cause analysis, log correlation, memory profiling, and production incident resolution.
- **Observability**: Prometheus, Grafana, distributed tracing, SLO/SLI monitoring, and incident response.
- **Documentation**: Technical writing, API docs, tutorials, and architecture documentation.

## Agents

| Agent | Model | Specialization |
|-------|-------|----------------|
| `software-architect` | opus | Backend systems, microservices, API design |
| `app-developer` | sonnet | Web/mobile apps, React, Next.js, Flutter |
| `systems-engineer` | sonnet | C/C++/Rust/Go, CLI tools, low-level systems |
| `devops-architect` | sonnet | Cloud (AWS/Azure/GCP), Kubernetes, IaC |
| `automation-engineer` | sonnet | CI/CD pipelines, GitHub Actions, Git workflows |
| `sre-expert` | sonnet | Reliability, observability, SLO/SLI, incidents |
| `debugger-pro` | opus | Root cause analysis, log correlation, profiling |
| `quality-specialist` | sonnet | Code review, security audit, test automation |
| `documentation-expert` | haiku | Technical docs, manuals, tutorials |

## Commands (27)

| Command | Description |
|---------|-------------|
| `/scaffold` | TypeScript/Python/React/Julia project scaffolding |
| `/eng-feature-dev` | End-to-end guided feature development |
| `/commit` | Intelligent git commit with analysis |
| `/fix-commit-errors` | Auto-fix GitHub Actions failures |
| `/double-check` | Multi-dimensional validation (security, perf, a11y) |
| `/run-all-tests` | Iterative test-and-fix until green |
| `/test-generate` | Generate comprehensive test suites |
| `/smart-debug` | Intelligent debugging with multi-mode RCA |
| `/refactor-clean` | Code quality and SOLID refactoring |
| `/deps` | Dependency auditing and safe upgrades |
| `/code-analyze` | Semantic code analysis via Serena MCP |
| `/docs` | Documentation generation and sync |
| *+ 15 more* | See plugin.json for full list |

## Skills (39)

Covers the complete SDLC:

- **Architecture**: API design, auth patterns, microservices, monorepo management
- **Implementation**: TypeScript, Python, async patterns, SQL optimization, error handling
- **Infrastructure**: Git workflows, GitHub Actions, GitLab CI, Prometheus, Grafana, secrets management
- **Quality**: Code review, debugging toolkit, test automation, E2E testing, validation
- **Documentation**: Standards and best practices

## Hooks

| Event | Purpose |
|-------|---------|
| PostToolUse | Auto-lint suggestions after Write/Edit (Python, JS/TS) |
| SubagentStop | Collect results from debugger-pro/quality-specialist |

## Installation

```bash
# Add the marketplace
/plugin marketplace add imewei/MyClaude

# Install the suite
/plugin install dev-suite@marketplace
```

After installation, restart Claude Code for changes to take effect.

## License

MIT License
