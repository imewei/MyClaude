# Claude Code Plugin Marketplace - Quick Reference

**Total Resources:** 3 Suites | 23 Expert Agents | 33 Slash Commands | 132 Skills
**Version:** 2.2.1 | **Last Updated:** March 31, 2026

---

## The 3-Suite System

The MyClaude ecosystem is organized into three focused suites, consolidated from specialized legacy plugins.

### 1. Agent Core Suite (`agent-core`)
**Purpose:** Multi-agent coordination, advanced reasoning, and context engineering.
- **Agents:** `@orchestrator`, `@reasoning-engine`, `@context-specialist`
- **Commands:** `/agent-build`, `/ai-assistant`, `/docs-lookup`, `/reflection`, `/ultra-think`, `/team-assemble`
- **Skills:** `agent-orchestration`, `agent-performance-optimization`, `llm-application-patterns`, `mcp-integration`, `multi-agent-coordination`, `reasoning-frameworks`, `reflection-framework`
- **Use When:** Complex problem solving, building AI assistants, managing long-running project context.

### 2. Dev Suite (`dev-suite`)
**Purpose:** Full-stack engineering, infrastructure, CI/CD, quality assurance, and debugging.
- **Agents:** `@app-developer`, `@automation-engineer`, `@debugger-pro`, `@devops-architect`, `@documentation-expert`, `@quality-specialist`, `@software-architect`, `@sre-expert`, `@systems-engineer`
- **Commands:** `/adopt-code`, `/c-project`, `/code-analyze`, `/code-explain`, `/commit`, `/deps`, `/docs`, `/double-check`, `/eng-feature-dev`, `/fix-commit-errors`, `/fix-imports`, `/github-assist`, `/merge-all`, `/modernize`, `/monitor-setup`, `/multi-platform`, `/onboard`, `/profile-performance`, `/refactor-clean`, `/run-all-tests`, `/rust-project`, `/scaffold`, `/slo-implement`, `/smart-debug`, `/tech-debt`, `/test-generate`, `/workflow-automate`
- **Skills:** `airflow-scientific-workflows`, `api-design-principles`, `architecture-patterns`, `async-python-patterns`, `auth-implementation-patterns`, `code-review`, `comprehensive-validation`, `debugging-toolkit`, `deployment-pipeline-design`, `distributed-tracing`, `documentation-standards`, `e2e-testing-patterns`, `error-handling-patterns`, `frontend-mobile-engineering`, `git-workflow`, `github-actions-templates`, `gitlab-ci-patterns`, `grafana-dashboards`, `iterative-error-resolution`, `microservices-patterns`, `modern-javascript-patterns`, `modernization-migration`, `monorepo-management`, `nodejs-backend-patterns`, `observability-sre-practices`, `plugin-syntax-validator`, `prometheus-configuration`, `python-packaging`, `python-performance-optimization`, `secrets-management`, `security-ci-template`, `slo-implementation`, `sql-optimization-patterns`, `systems-cli-engineering`, `test-automation`, `testing-patterns`, `typescript-advanced-types`, `typescript-project-scaffolding`, `uv-package-manager`
- **Use When:** Building web/mobile apps, systems programming, backend architecture, CI/CD, monitoring, Git workflows, testing, debugging, code quality, documentation.

### 3. Scientific Computing Suite (`science-suite`)
**Purpose:** HPC, specialized simulations, and data science workflows.
- **Agents:** `@ai-engineer`, `@jax-pro`, `@julia-pro`, `@ml-expert`, `@neural-network-master`, `@nonlinear-dynamics-expert`, `@prompt-engineer`, `@python-pro`, `@research-expert`, `@simulation-expert`, `@statistical-physicist`
- **Commands:** *Specialized commands are planned for future releases. Use suite agents for guided execution.*
- **Skills:** Extensive scientific skills including `jax-mastery`, `julia-mastery`, `deep-learning`, `statistical-physics`, `parallel-computing`, `machine-learning`, and 86 specialized domain skills.
- **Use When:** HPC, JAX/Julia projects, physics simulations, ML engineering, scientific research.

---

## 🛠️ Common Workflows

### Engineering Feature Development
1. Use `@software-architect` to design the system.
2. Run `/dev-suite:eng-feature-dev` to implement the core logic.
3. Use `/dev-suite:test-generate` to create tests.
4. Run `/dev-suite:double-check` before submitting.

### Scientific Simulation
1. Use `@simulation-expert` to design the simulation.
2. Use `@jax-pro` or `@julia-pro` to implement the numerical kernels.
3. Use `@research-expert` to analyze and visualize results.

### Agent Teams
1. Run `/agent-core:team-assemble list` to see all 34 team templates.
2. Run `/agent-core:team-assemble <type>` to generate a team prompt.
3. Paste the prompt into Claude Code with agent teams enabled.
4. See [Agent Teams Guide](../agent-teams-guide.md) for full details.

---

## 📦 Installation

### Step 1: Add the Marketplace
```bash
/plugin marketplace add imewei/MyClaude
```

### Step 2: Install Suites
```bash
/plugin install agent-core@marketplace
/plugin install dev-suite@marketplace
/plugin install science-suite@marketplace
```

---

## Resources
- **Full Documentation:** [https://myclaude.readthedocs.io/en/latest/](https://myclaude.readthedocs.io/en/latest/)
- **Agent List:** [agents.md](agents.md)
- **Commands List:** [commands.md](commands.md)

---

*Generated from v2.2.1 validated marketplace data.*
