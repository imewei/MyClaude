# Command Reference

**15 Registered Commands** | **19 Skill-Invoked Commands** | **Version:** 3.5.2

Commands fall into two categories:
- **Registered commands** are declared in `plugin.json` and available as user-facing `/slash-commands`.
- **Skill-invoked commands** exist on disk but are not registered in manifests — they are triggered by skills, not directly by users.

---

## Registered Commands

### Dev Suite (`dev-suite`) — 10 Commands

| Command | Description |
|---------|-------------|
| `/docs` | Unified documentation management — generate, update, and sync |
| `/double-check` | Multi-dimensional validation with automated testing and security scanning |
| `/eng-feature-dev` | End-to-end feature development with customizable methodologies |
| `/fix-commit-errors` | Diagnose and fix CI/CD failures by analyzing logs and rerunning workflows |
| `/merge-all` | Merge all local branches into main and clean up |
| `/modernize` | Legacy code migration using Strangler Fig pattern |
| `/run-all-tests` | Iteratively run and fix all tests until zero failures |
| `/smart-debug` | Intelligent debugging with multi-mode execution and automated RCA |
| `/test-generate` | Generate comprehensive test suites with scientific computing support |
| `/workflow-automate` | Automated CI/CD workflow generation for GitHub Actions and GitLab CI |

### Research Suite (`research-suite`) — 3 Commands

| Command | Description |
|---------|-------------|
| `/lit-review` | Systematic literature review with PRISMA-compliant search and evidence synthesis |
| `/paper-implement` | Reproduce a research paper end-to-end: theory → code → validation |
| `/replicate` | Computational replication of published experiments with deviation analysis |

### Science Suite (`science-suite`) — 2 Commands

| Command | Description |
|---------|-------------|
| `/md-sim` | Molecular dynamics simulation setup, running, and trajectory analysis |
| `/benchmark` | Scientific code benchmarking across backends and hardware targets |

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

### Dev Suite — 13 Skill-Invoked

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
| `scaffold` | Project and component scaffolding for TypeScript, Python, React, Julia |
| `tech-debt` | Technical debt analysis with ROI-based roadmaps |

### Science Suite — 2 Skill-Invoked

| Command | Description |
|---------|-------------|
| `analyze-data` | Analyze data files with statistical tests, visualization, and reporting |
| `run-experiment` | Design and execute computational experiments with hypothesis tracking |

### Research Suite — 0 Skill-Invoked

No on-disk command files. All research workflows are skill-driven (`scientific-review`, the 8-stage `research-spark` pipeline, `research-practice` hub, and the 3 registered slash commands above).

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

*Generated from v3.5.2 validated marketplace data.*
