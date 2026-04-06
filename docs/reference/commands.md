# Command Reference

**14 Registered Commands** | **22 Skill-Invoked Commands** | **Version:** 3.1.1

Commands fall into two categories:
- **Registered commands** are declared in `plugin.json` and available as user-facing `/slash-commands`.
- **Skill-invoked commands** exist on disk but are not registered in manifests — they are triggered by skills, not directly by users.

---

## Registered Commands

### Agent Core Suite (`agent-core`) — 2 Commands

| Command | Description |
|---------|-------------|
| `/ultra-think` | Comprehensive analysis with full reasoning framework execution |
| `/team-assemble` | Generate ready-to-use agent team configurations from pre-built templates |

### Dev Suite (`dev-suite`) — 12 Commands

| Command | Description |
|---------|-------------|
| `/commit` | Intelligent git commit with automated analysis and quality validation |
| `/docs` | Unified documentation management — generate, update, and sync |
| `/double-check` | Multi-dimensional validation with automated testing and security scanning |
| `/eng-feature-dev` | End-to-end feature development with customizable methodologies |
| `/fix-commit-errors` | Diagnose and fix CI/CD failures by analyzing logs and rerunning workflows |
| `/merge-all` | Merge all local branches into main and clean up |
| `/modernize` | Legacy code migration using Strangler Fig pattern |
| `/refactor-clean` | Analyze and refactor code for quality and maintainability |
| `/run-all-tests` | Iteratively run and fix all tests until zero failures |
| `/smart-debug` | Intelligent debugging with multi-mode execution and automated RCA |
| `/test-generate` | Generate comprehensive test suites with scientific computing support |
| `/workflow-automate` | Automated CI/CD workflow generation for GitHub Actions and GitLab CI |

### Science Suite (`science-suite`) — 0 Commands

No registered commands. All 3 science-suite commands are skill-invoked.

---

## Skill-Invoked Commands

These commands exist on disk and are triggered by skills during workflows. They are **not** available as direct `/slash-commands`.

### Agent Core — 4 Skill-Invoked

| Command | Description |
|---------|-------------|
| `agent-build` | AI agent creation, optimization, and prompt engineering |
| `ai-assistant` | Build production-ready AI assistants with NLU and response generation |
| `docs-lookup` | Query library documentation using Context7 MCP |
| `reflection` | AI reasoning analysis, session retrospectives, and research optimization |

### Dev Suite — 15 Skill-Invoked

| Command | Description |
|---------|-------------|
| `adopt-code` | Analyze and modernize scientific computing codebases |
| `c-project` | Scaffold production-ready C projects with Makefile/CMake |
| `code-analyze` | Semantic code analysis using Serena MCP for symbol navigation |
| `code-explain` | Detailed code explanation with visual aids |
| `deps` | Dependency management — security auditing and safe upgrades |
| `fix-imports` | Fix broken imports across the codebase |
| `github-assist` | GitHub operations using GitHub MCP for issues, PRs, repos |
| `monitor-setup` | Set up Prometheus, Grafana, and distributed tracing |
| `multi-platform` | Build and deploy features across web, mobile, and desktop |
| `onboard` | Onboarding orchestration with 30/60/90 day plans |
| `profile-performance` | Performance profiling with perf, flamegraph, and valgrind |
| `rust-project` | Scaffold production-ready Rust projects |
| `scaffold` | Project and component scaffolding for TypeScript, Python, React, Julia |
| `slo-implement` | SLO/SLA monitoring, error budgets, and burn rate alerting |
| `tech-debt` | Technical debt analysis with ROI-based roadmaps |

### Science Suite — 3 Skill-Invoked

| Command | Description |
|---------|-------------|
| `analyze-data` | Analyze data files with statistical tests, visualization, and reporting |
| `paper-review` | Scientific paper review with methodology and reproducibility assessment |
| `run-experiment` | Design and execute computational experiments with hypothesis tracking |

---

## Execution Modes

Most commands support three execution modes via `--mode=<mode>`:

| Mode | Scope | Description |
|------|-------|-------------|
| **quick** | Fast | Syntax checking, basic scaffolding |
| **standard** | Full | Complete implementation with testing |
| **comprehensive** | Deep | Advanced features, compliance, CI/CD |

---

## Hub Skill Routing

Commands often invoke hub skills, which route to specialized sub-skills automatically. For example, `/smart-debug` may trigger the `debugging-toolkit` sub-skill through the `dev-workflows` hub. See the suite reference docs for full hub → sub-skill mappings.

---

## Resources

- [Agent Reference](agents.md)
- [Quick Reference Cheatsheet](cheatsheet.md)
- [Integration Map](../integration-map.rst) — Suite dependencies and MCP server roles
- [Glossary](../glossary.rst) — Key terms (Hub Skill, Sub-Skill, Routing Decision Tree)

*Generated from v3.1.1 validated marketplace data.*
