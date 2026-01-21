# Claude Code Plugin Marketplace - Quick Reference

**Total Resources:** 5 Suites | 22 Expert Agents | 32 Slash Commands
**Version:** 2.1.0 | **Last Updated:** January 20, 2026

---

## 🚀 The 5-Suite System

The MyClaude ecosystem is organized into five powerful suites, each consolidated from specialized legacy plugins.

### 1. Agent Core Suite (`agent-core`)
**Purpose:** Multi-agent coordination, advanced reasoning, and context engineering.
- **Agents:** `@orchestrator`, `@reasoning-engine`, `@context-specialist`
- **Commands:** `/agent-build`, `/ai-assistant`, `/docs-lookup`, `/reflection`, `/ultra-think`
- **Use When:** Complex problem solving, building AI assistants, managing long-running project context.

### 2. Software Engineering Suite (`engineering-suite`)
**Purpose:** Full-stack engineering, systems programming, and platform implementations.
- **Agents:** `@software-architect`, `@app-developer`, `@systems-engineer`
- **Commands:** `/scaffold`, `/rust-project`, `/c-project`, `/eng-feature-dev`, `/modernize`, `/multi-platform`, `/profile-performance`
- **Use When:** Building web/mobile apps, systems programming (Rust/C), backend architecture, legacy modernization.

### 3. Infrastructure & Ops Suite (`infrastructure-suite`)
**Purpose:** CI/CD automation, observability monitoring, and Git workflows.
- **Agents:** `@devops-architect`, `@automation-engineer`, `@sre-expert`
- **Commands:** `/commit`, `/fix-commit-errors`, `/merge-all`, `/monitor-setup`, `/onboard`, `/slo-implement`, `/workflow-automate`, `/code-analyze`, `/github-assist`
- **Use When:** Setting up CI/CD, monitoring/observability, Git workflow automation, cloud infrastructure.

### 4. Quality & Maintenance Suite (`quality-suite`)
**Purpose:** Code quality, test automation, and intelligent debugging.
- **Agents:** `@quality-specialist`, `@debugger-pro`, `@documentation-expert`
- **Commands:** `/run-all-tests`, `/test-generate`, `/double-check`, `/smart-debug`, `/refactor-clean`, `/tech-debt`, `/adopt-code`, `/code-explain`, `/deps`, `/docs`, `/fix-imports`
- **Use When:** Writing/fixing tests, debugging complex bugs, technical debt remediation, documentation generation.

### 5. Scientific Computing Suite (`science-suite`)
**Purpose:** HPC, specialized simulations, and data science workflows.
- **Agents:** `@jax-pro`, `@julia-pro`, `@ml-expert`, `@neural-network-master`, `@prompt-engineer`, `@python-pro`, `@research-expert`, `@simulation-expert`, `@statistical-physicist`, `@ai-engineer`
- **Commands:** *Specialized commands are planned for future releases. Use suite agents for guided execution.*
- **Use When:** HPC, JAX/Julia projects, physics simulations, ML engineering, scientific research.

---

## 🛠️ Common Workflows

### Feature Development
1. Use `@software-architect` to design the system.
2. Run `/engineering-suite:eng-feature-dev` to implement the core logic.
3. Use `/quality-suite:test-generate` to create tests.
4. Run `/quality-suite:double-check` before submitting.

### Scientific Simulation
1. Use `@simulation-expert` to design the simulation.
2. Use `@jax-pro` or `@julia-pro` to implement the numerical kernels.
3. Use `@research-expert` to analyze and visualize results.

---

## 📦 Installation

### Step 1: Add the Marketplace
```bash
/plugin marketplace add imewei/MyClaude
```

### Step 2: Install Suites
```bash
/plugin install agent-core@marketplace
/plugin install engineering-suite@marketplace
/plugin install infrastructure-suite@marketplace
/plugin install quality-suite@marketplace
/plugin install science-suite@marketplace
```

---

## Resources
- **Full Documentation:** [https://myclaude.readthedocs.io/en/latest/](https://myclaude.readthedocs.io/en/latest/)
- **Agent List:** [agents.md](agents.md)
- **Commands List:** [commands.md](commands.md)

---

*Generated from v2.1.0 validated marketplace data.*
