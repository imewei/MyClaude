# MyClaude

[![Plugins](https://img.shields.io/badge/Plugins-4-blue.svg)](https://myclaude.readthedocs.io/en/latest/plugins/)
[![Agents](https://img.shields.io/badge/Agents-25-green.svg)](docs/reference/agents.md)
[![Commands](https://img.shields.io/badge/Commands-14-orange.svg)](docs/reference/commands.md)
[![Skills](https://img.shields.io/badge/Skills-31_hubs_→_186_sub-purple.svg)](docs/reference/cheatsheet.md)
[![Version](https://img.shields.io/badge/Version-3.4.1-red.svg)](https://github.com/imewei/MyClaude)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-ReadTheDocs-brightgreen.svg)](https://myclaude.readthedocs.io/en/latest/)

Claude Code plugin marketplace with **4 focused suites**, **25 expert agents**, **14 registered commands**, and **31 hub skills** routing to **186 sub-skills**. Built for Claude Opus 4.7 with tiered model assignments (Opus/Sonnet/Haiku), 27 lifecycle hook events across all suites, and hub-skill architecture for zero-ambiguity skill routing.

## The 4-Suite Hub Architecture

MyClaude v3.4.1 uses a **hub-skill architecture**: skills are organized into hub skills (meta-orchestrators) that route to specialized sub-skills via decision trees. Only hubs are declared in `plugin.json`; sub-skills are discovered through hub routing.

| Suite | Agents | Commands | Hubs → Sub-skills | Hooks | Focus |
|-------|--------|----------|-------------------|-------|-------|
| [Agent Core](plugins/agent-core/) | 3 | 2 | 4 → 13 | 12 events | Orchestration, reasoning, context engineering |
| [Dev Suite](plugins/dev-suite/) | 9 | 12 | 9 → 49 | 7 events | Full SDLC: architecture, CI/CD, testing, debugging |
| [Research Suite](plugins/research-suite/) | 2 | 0 | 4 → 12 | 3 events | Peer review, 8-stage research-spark pipeline, methodology |
| [Science Suite](plugins/science-suite/) | 11 | 0 | 14 → 112 | 5 events | JAX, Julia, physics, ML/DL/HPC, nonlinear dynamics |

## Specialist Agents

25 agents with tiered model assignments: **11 opus** (deep reasoning), **13 sonnet** (standard), **1 haiku** (fast).

| Agent | Suite | Model | Specialization |
|-------|-------|-------|----------------|
| `@orchestrator` | Agent Core | opus | Multi-agent coordination and task delegation |
| `@reasoning-engine` | Agent Core | opus | Advanced reasoning and structured thinking |
| `@context-specialist` | Agent Core | opus | Dynamic context management, vector/memory systems |
| `@software-architect` | Dev | opus | Backend systems, microservices, API design |
| `@debugger-pro` | Dev | opus | Root cause analysis, log correlation |
| `@research-expert` | Research | opus | Literature reviews, experiment design, statistical rigor |
| `@research-spark-orchestrator` | Research | opus | 8-stage artifact-gated refinement pipeline |
| `@neural-network-master` | Science | opus | Deep learning theory and architecture |
| `@statistical-physicist` | Science | opus | Correlation functions, non-equilibrium dynamics |
| `@simulation-expert` | Science | opus | Molecular dynamics, HPC, numerical methods |
| `@nonlinear-dynamics-expert` | Science | opus | Bifurcations, chaos, network dynamics, pattern formation |
| `@jax-pro` | Science | sonnet | JAX, Bayesian inference, physics apps |
| `@julia-pro` | Science | sonnet | Julia SciML, DifferentialEquations.jl |
| `@python-pro` | Science | sonnet | Python systems engineering, performance |

See [complete agent list](docs/reference/agents.md) for all 25 agents.

## Installation

### Step 1: Add the Marketplace

```bash
/plugin marketplace add imewei/MyClaude
```

### Step 2: Install Suites

```bash
/plugin install agent-core@marketplace
/plugin install dev-suite@marketplace
/plugin install research-suite@marketplace
/plugin install science-suite@marketplace
```

**Note:** After installation, restart Claude Code for changes to take effect.

## Quick Start

**Using Specialized Agents**
```
Ask Claude: "@python-pro help me optimize this async function"
Ask Claude: "@orchestrator coordinate a team for this new feature"
Ask Claude: "@jax-pro implement this differentiable physics model"
Ask Claude: "@research-expert design a power analysis for this experiment"
```

**Running Commands**
```bash
/agent-core:ultra-think "Analyze the architecture of this system"
/agent-core:team-assemble scientific-computing-team
/dev-suite:double-check my-feature
/dev-suite:fix-commit-errors
```

## Documentation

- **[Full Documentation](https://myclaude.readthedocs.io/en/latest/)**
- **[Plugin Cheatsheet](docs/reference/cheatsheet.md)**
- **[Complete Agents List](docs/reference/agents.md)**
- **[Complete Commands List](docs/reference/commands.md)**
- **[Agent Teams Guide](docs/agent-teams-guide.md)** — 10 focused team templates with 20 variants (codebase-aware recommender since v3.1.4)

## License

MIT License (see [LICENSE](LICENSE))

---

**Built by Wei Chen** | [Documentation](https://myclaude.readthedocs.io/en/latest/) | [GitHub](https://github.com/imewei/MyClaude)
