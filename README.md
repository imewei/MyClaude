# MyClaude

[![Plugins](https://img.shields.io/badge/Plugins-3-blue.svg)](https://myclaude.readthedocs.io/en/latest/plugins/)
[![Agents](https://img.shields.io/badge/Agents-26-green.svg)](docs/reference/agents.md)
[![Commands](https://img.shields.io/badge/Commands-21-orange.svg)](docs/reference/commands.md)
[![Skills](https://img.shields.io/badge/Skills-42_hubs_→_157_sub-purple.svg)](docs/reference/cheatsheet.md)
[![Version](https://img.shields.io/badge/Version-4.0.4-red.svg)](https://github.com/imewei/MyClaude)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-ReadTheDocs-brightgreen.svg)](https://myclaude.readthedocs.io/en/latest/)

Claude Code plugin marketplace with **3 focused suites**, **26 expert agents**, **21 registered commands**, and **42 hub skills** routing to **157 sub-skills**. Built for the Claude 5 model generation (Opus 5, Sonnet 5, Haiku 4.5) with tiered model assignments, 16 hook events across all suites, and hub-skill architecture for zero-ambiguity skill routing.

## The 3-Suite Hub Architecture

MyClaude v4.0.4 uses a **hub-skill architecture**: skills are organized into hub skills (meta-orchestrators) that route to specialized sub-skills via decision trees. Only hubs are declared in `plugin.json`; sub-skills are discovered through hub routing.

| Suite | Agents | Commands | Hubs → Sub-skills | Hooks | Focus |
|-------|--------|----------|-------------------|-------|-------|
| [Dev Suite](plugins/dev-suite/) | 12 | 14 | 9 → 36 | 7 events | Full SDLC: architecture, CI/CD, testing, debugging |
| [Research Suite](plugins/research-suite/) | 2 | 3 | 10 → 7 | 4 events | Peer review, research-spark pipeline (5-stage core + optional extension), methodology |
| [Science Suite](plugins/science-suite/) | 12 | 4 | 23 → 114 | 5 events | JAX, Julia, physics, ML/DL/HPC, nonlinear dynamics |

## Specialist Agents

26 agents with tiered model assignments: **8 opus** (architect: research, planning, review, theory), **16 sonnet** (contractor: code edits, implementation, debugging), **2 haiku** (mechanical docs, comment triage).

Tiers name Claude Code model aliases, not pinned versions, so each agent tracks the current generation: `opus` → Opus 5, `sonnet` → Sonnet 5, `haiku` → Haiku 4.5. The 19 original `opus` and `sonnet` agents set `effort: high`; `xhigh` is deliberately unused, since it does not exist on Sonnet 4.6 or Opus 4.6. The 5 `sonnet` agents adopted from `ecc` as `/review-pr`'s fan-out passes (`code-reviewer`, `pr-test-analyzer`, `silent-failure-hunter`, `type-design-analyzer`, `code-simplifier`) set `effort: medium` instead — deliberately lighter, parallel passes, distinct from `quality-specialist`'s `effort: high` deep audit. The 2 `haiku` agents set no `effort` at all — the field is not supported on Haiku 4.5.

Two models sit outside the alias set and are reachable only by pinning them explicitly on a dispatch (`model:` on the Agent call, or in an agent's frontmatter):

| Model | When to pin it |
|-------|----------------|
| `fable` (Fable 5.1) | Hardest long-horizon agentic runs. ~2× Opus 5 per token, and it rejects forced `tool_choice: any\|tool` with a 400 — verify the agent's harness does not force tool choice before making it a default. No agent is pinned to it. |
| Opus 4.8 / Sonnet 4.6 | Previous generation, still served. Pin one to reproduce an older run, or as a refusal fallback for `fable`/`opus` work. Sonnet 4.6 costs *more* than Sonnet 5 ($3/$15 vs $2/$10 per MTok), so pin it for reproducibility, never to save money. |

| Agent | Suite | Model | Specialization |
|-------|-------|-------|----------------|
| `@software-architect` | Dev | opus | Backend systems, microservices, API design |
| `@research-expert` | Research | opus | Literature reviews, experiment design, statistical rigor |
| `@research-spark-orchestrator` | Research | opus | Artifact-gated refinement pipeline (5-stage core + optional extension) |
| `@jax-pro` | Science | sonnet | JAX/JIT, vmap/pmap, Flax NNX, NumPyro, physics apps |
| `@julia-pro` | Science | sonnet | Julia SciML, DifferentialEquations.jl, Turing.jl |
| `@neural-network-master` | Science | opus | Deep learning theory and architecture |
| `@statistical-physicist` | Science | opus | Correlation functions, non-equilibrium dynamics |
| `@nonlinear-dynamics-expert` | Science | opus | Bifurcations, chaos, network dynamics, pattern formation |
| `@pinn-engineer` | Science | sonnet | PINNs, BPINNs, NeuralPDE, MethodOfLines |
| `@continuum-mechanics-engineer` | Science | opus | FEM/FEA, constitutive modeling, DMA/rheology, nanocomposites |
| `@simulation-expert` | Science | sonnet | Molecular dynamics, HPC, numerical methods |
| `@sci-workflow-engineer` | Science | sonnet | Scientific workflow design and optimization |
| `@julia-ml-hpc` | Science | sonnet | Julia ML, Lux.jl, distributed/GPU computing |
| `@python-pro` | Science | sonnet | Python systems engineering, performance |

See [complete agent list](docs/reference/agents.md) for all 26 agents.

## Installation

### Step 1: Add the Marketplace

```bash
/plugin marketplace add imewei/MyClaude
```

### Step 2: Install Suites

```bash
/plugin install dev-suite@scientific-computing-workflows
/plugin install research-suite@scientific-computing-workflows
/plugin install science-suite@scientific-computing-workflows
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
