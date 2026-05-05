# Dev Suite

Unified software development lifecycle suite covering architecture, implementation, CI/CD, testing, debugging, and deployment. Merges engineering, infrastructure, and quality into one suite for zero-friction cross-concern agent delegation.

## Overview

Dev Suite covers the complete software development lifecycle with 9 specialized agents (2 opus, 6 sonnet, 1 haiku), 12 registered slash commands, and 9 hub skills routing to 49 sub-skills. From architecture design through CI/CD to production debugging, every engineering workflow is covered. Agents delegate across specializations automatically — debugger-pro hands off to sre-expert for reliability issues, software-architect delegates to devops-architect for infrastructure.

## Quick Start / Usage Examples

```bash
# End-to-end feature development with guided phases
/eng-feature-dev "Add user authentication with OAuth2"

# Run all tests and auto-fix failures
/run-all-tests

# Validate before shipping (10-dimension check)
/double-check --deep --security

# Ask the architect to design a system
@software-architect "Design a microservice for payment processing"
```

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

## Skills (9 hubs → 49 sub-skills)

Covers the complete SDLC:

- **Architecture**: API design, auth patterns, microservices, monorepo management, GraphQL
- **Data & Storage**: Database patterns (ORMs, migrations), caching (Redis), search (Elasticsearch)
- **Implementation**: TypeScript, Python, async patterns, SQL optimization, error handling, WebSocket
- **Infrastructure**: Git workflows, GitHub Actions, GitLab CI, Prometheus, Grafana, secrets management, Docker/K8s, cloud providers (AWS/GCP/Azure), message queues
- **Quality**: Code review, debugging toolkit, test automation, E2E testing, validation, accessibility (WCAG), mobile testing
- **Documentation**: Standards and best practices

## Hooks (7 events)

| Event | Purpose |
|-------|---------|
| SessionStart | Auto-detect project stack (language, framework, test runner) |
| PreToolUse | Guard destructive git ops (push --force, reset --hard, branch -D) |
| PostToolUse | Auto-lint suggestions after Write/Edit (Python, JS/TS) |
| SubagentStop | Collect results from debugger-pro/quality-specialist |
| TaskCompleted | Trigger validation checks and suggest git commit |
| SessionEnd | Persist structured progress summary for next session |
| StopFailure | Capture context when /stop fails mid-operation |

(`ExecutionError` was removed in v3.4.0 — not supported by the CC v2.1.126 CLI event schema.)

## Integration / Workflow

Dev Suite agents delegate across specializations automatically: `debugger-pro` hands off to `sre-expert` for reliability issues, `software-architect` delegates to `devops-architect` for infrastructure, `quality-specialist` coordinates with `systems-engineer` on low-level review. Cross-suite, dev-suite delegates *up* to `agent-core/orchestrator` for multi-agent coordination and *out* to `science-suite` when ML/DL/physics expertise is needed. See `docs/integration-map.rst` for the full delegation graph.

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
