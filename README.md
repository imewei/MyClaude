# MyClaude

[![Plugins](https://img.shields.io/badge/Plugins-3-blue.svg)](https://myclaude.readthedocs.io/en/latest/plugins/)
[![Agents](https://img.shields.io/badge/Agents-24-green.svg)](docs/reference/agents.md)
[![Commands](https://img.shields.io/badge/Commands-14-orange.svg)](docs/reference/commands.md)
[![Skills](https://img.shields.io/badge/Skills-26_hubs_→_167_sub-purple.svg)](docs/reference/cheatsheet.md)
[![Version](https://img.shields.io/badge/Version-3.1.0-red.svg)](https://github.com/imewei/MyClaude)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-ReadTheDocs-brightgreen.svg)](https://myclaude.readthedocs.io/en/latest/)

Claude Code plugin marketplace with **3 focused suites**, **24 expert agents**, **14 registered commands**, and **26 hub skills** routing to **167 sub-skills**. Built for Claude Opus 4.6 with tiered model assignments (Opus/Sonnet/Haiku), 10 lifecycle hooks, and hub-skill architecture for zero-ambiguity skill routing.

## The 3-Suite Hub Architecture

MyClaude v3.1.0 uses a **hub-skill architecture**: skills are organized into hub skills (meta-orchestrators) that route to specialized sub-skills via decision trees. Only hubs are declared in `plugin.json`; sub-skills are discovered through hub routing.

| Suite | Agents | Commands | Hubs → Sub-skills | Hooks | Focus |
|-------|--------|----------|-------------------|-------|-------|
| [Agent Core](plugins/agent-core/) | 3 | 2 | 3 → 12 | 8 events | Orchestration, reasoning, context engineering |
| [Dev Suite](plugins/dev-suite/) | 9 | 12 | 9 → 49 | 2 events | Full SDLC: architecture, CI/CD, testing, debugging |
| [Science Suite](plugins/science-suite/) | 12 | 0 | 14 → 106 | — | JAX, Julia, physics, ML/DL/HPC, research |

## Specialist Agents

24 agents with tiered model assignments: **9 opus** (deep reasoning), **14 sonnet** (standard), **1 haiku** (fast).

| Agent | Suite | Model | Specialization |
|-------|-------|-------|----------------|
| `@orchestrator` | Agent Core | opus | Multi-agent coordination and task delegation |
| `@reasoning-engine` | Agent Core | opus | Advanced reasoning and structured thinking |
| `@software-architect` | Dev | opus | Backend systems, microservices, API design |
| `@debugger-pro` | Dev | opus | Root cause analysis, log correlation |
| `@neural-network-master` | Science | opus | Deep learning theory and architecture |
| `@statistical-physicist` | Science | opus | Correlation functions, non-equilibrium dynamics |
| `@simulation-expert` | Science | opus | Molecular dynamics, HPC, numerical methods |
| `@jax-pro` | Science | sonnet | JAX, Bayesian inference, physics apps |
| `@julia-pro` | Science | sonnet | Julia SciML, DifferentialEquations.jl |
| `@python-pro` | Science | sonnet | Python systems engineering, performance |

## Installation

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

**Note:** After installation, restart Claude Code for changes to take effect.

## Quick Start

**Using Specialized Agents**
```
Ask Claude: "@python-pro help me optimize this async function"
Ask Claude: "@orchestrator coordinate a team for this new feature"
Ask Claude: "@jax-pro implement this differentiable physics model"
```

**Running Commands**
```bash
/agent-core:ultra-think "Analyze the architecture of this system"
/dev-suite:double-check my-feature
/dev-suite:fix-commit-errors
```

## Documentation

- **[Full Documentation](https://myclaude.readthedocs.io/en/latest/)**
- **[Plugin Cheatsheet](docs/reference/cheatsheet.md)**
- **[Complete Agents List](docs/reference/agents.md)**
- **[Complete Commands List](docs/reference/commands.md)**
- **[Agent Teams Guide](docs/agent-teams-guide.md)** — 34 ready-to-use team configurations

## License

MIT License (see [LICENSE](LICENSE))

---

**Built by Wei Chen** | [Documentation](https://myclaude.readthedocs.io/en/latest/) | [GitHub](https://github.com/imewei/MyClaude)
