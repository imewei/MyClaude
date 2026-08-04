# MyClaude

[![Plugins](https://img.shields.io/badge/Plugins-3-blue.svg)](https://myclaude.readthedocs.io/en/latest/plugins/)
[![Agents](https://img.shields.io/badge/Agents-20-green.svg)](docs/reference/agents.md)
[![Commands](https://img.shields.io/badge/Commands-15-orange.svg)](docs/reference/commands.md)
[![Skills](https://img.shields.io/badge/Skills-50_hubs_→_148_sub-purple.svg)](docs/reference/cheatsheet.md)
[![Version](https://img.shields.io/badge/Version-4.0.0-red.svg)](https://github.com/imewei/MyClaude)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-ReadTheDocs-brightgreen.svg)](https://myclaude.readthedocs.io/en/latest/)

Claude Code plugin marketplace with **3 focused suites**, **20 expert agents**, **15 registered commands**, and **50 hub skills** routing to **148 sub-skills**. Built for Claude Opus 4.7 with tiered model assignments (Opus/Sonnet/Haiku), 12 lifecycle hook events across all suites, and hub-skill architecture for zero-ambiguity skill routing.

## The 3-Suite Hub Architecture

MyClaude v4.0.0 uses a **hub-skill architecture**: skills are organized into hub skills (meta-orchestrators) that route to specialized sub-skills via decision trees. Only hubs are declared in `plugin.json`; sub-skills are discovered through hub routing.

| Suite | Agents | Commands | Hubs → Sub-skills | Hooks | Focus |
|-------|--------|----------|-------------------|-------|-------|
| [Dev Suite](plugins/dev-suite/) | 6 | 10 | 10 → 35 | 6 events | Full SDLC: architecture, CI/CD, testing, debugging |
| [Research Suite](plugins/research-suite/) | 2 | 3 | 11 → 6 | 2 events | Peer review, 8-stage research-spark pipeline, methodology |
| [Science Suite](plugins/science-suite/) | 12 | 2 | 30 → 107 | 4 events | JAX, Julia, physics, ML/DL/HPC, nonlinear dynamics |

## Specialist Agents

20 agents with tiered model assignments: **10 opus** (deep reasoning), **8 sonnet** (standard), **2 haiku** (fast).

| Agent | Suite | Model | Specialization |
|-------|-------|-------|----------------|
| `@software-architect` | Dev | opus | Backend systems, microservices, API design |
| `@research-expert` | Research | opus | Literature reviews, experiment design, statistical rigor |
| `@research-spark-orchestrator` | Research | opus | 8-stage artifact-gated refinement pipeline |
| `@jax-pro` | Science | opus | JAX/JIT, vmap/pmap, Flax NNX, NumPyro, physics apps |
| `@julia-pro` | Science | opus | Julia SciML, DifferentialEquations.jl, Turing.jl |
| `@neural-network-master` | Science | opus | Deep learning theory and architecture |
| `@statistical-physicist` | Science | opus | Correlation functions, non-equilibrium dynamics |
| `@nonlinear-dynamics-expert` | Science | opus | Bifurcations, chaos, network dynamics, pattern formation |
| `@pinn-engineer` | Science | opus | PINNs, BPINNs, NeuralPDE, MethodOfLines |
| `@continuum-mechanics-engineer` | Science | opus | FEM/FEA, constitutive modeling, DMA/rheology, nanocomposites |
| `@simulation-expert` | Science | sonnet | Molecular dynamics, HPC, numerical methods |
| `@sci-workflow-engineer` | Science | sonnet | Scientific workflow design and optimization |
| `@julia-ml-hpc` | Science | sonnet | Julia ML, Lux.jl, distributed/GPU computing |
| `@python-pro` | Science | sonnet | Python systems engineering, performance |

See [complete agent list](docs/reference/agents.md) for all 20 agents.

## Installation

### Step 1: Add the Marketplace

```bash
/plugin marketplace add imewei/MyClaude
```

### Step 2: Install Suites

```bash
/plugin install dev-suite@marketplace
/plugin install research-suite@marketplace
/plugin install science-suite@marketplace
```

**Note:** After installation, restart Claude Code for changes to take effect.

## Quick Start

**Using Specialized Agents**
```
Ask Claude: "@python-pro help me optimize this async function"
Ask Claude: "@jax-pro implement this differentiable physics model"
Ask Claude: "@research-expert design a power analysis for this experiment"
```

**Running Commands**
```bash
/dev-suite:double-check my-feature
/dev-suite:fix-commit-errors
```

## Documentation

- **[Full Documentation](https://myclaude.readthedocs.io/en/latest/)**
- **[Plugin Cheatsheet](docs/reference/cheatsheet.md)**
- **[Complete Agents List](docs/reference/agents.md)**
- **[Complete Commands List](docs/reference/commands.md)**

## License

MIT License (see [LICENSE](LICENSE))

---

**Built by Wei Chen** | [Documentation](https://myclaude.readthedocs.io/en/latest/) | [GitHub](https://github.com/imewei/MyClaude)
