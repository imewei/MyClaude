# MyClaude

[![Plugins](https://img.shields.io/badge/Plugins-3-blue.svg)](https://myclaude.readthedocs.io/en/latest/plugins/)
[![Agents](https://img.shields.io/badge/Agents-26-green.svg)](docs/reference/agents.md)
[![Commands](https://img.shields.io/badge/Commands-21-orange.svg)](docs/reference/commands.md)
[![Skills](https://img.shields.io/badge/Skills-42_hubs_→_157_sub-purple.svg)](docs/reference/cheatsheet.md)
[![Version](https://img.shields.io/badge/Version-4.0.6-red.svg)](https://github.com/imewei/MyClaude/releases/tag/v4.0.6)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-ReadTheDocs-brightgreen.svg)](https://myclaude.readthedocs.io/en/latest/)

Claude Code plugin marketplace with **3 focused suites**, **27 expert agents**, **21 registered commands**, and **42 hub skills** routing to **169 sub-skills**. Built for the Claude 5.5 generation (Opus 5.5, Sonnet 5.5, Haiku 5.5, with Fable 5.1 as advisor) with role-tuned model tiers and effort, 16 hook events across all suites, and hub-skill architecture for zero-ambiguity skill routing.

## The 3-Suite Hub Architecture

MyClaude v4.0.6 uses a **hub-skill architecture**: skills are organized into hub skills (meta-orchestrators) that route to specialized sub-skills via decision trees. Only hubs are declared in `plugin.json`; sub-skills are discovered through hub routing.

| Suite | Agents | Commands | Hubs → Sub-skills | Hooks | Focus |
|-------|--------|----------|-------------------|-------|-------|
| [Dev Suite](plugins/dev-suite/) | 12 | 14 | 9 → 36 | 7 events | Full SDLC: architecture, CI/CD, testing, debugging |
| [Research Suite](plugins/research-suite/) | 2 | 3 | 10 → 7 | 4 events | Peer review, research-spark pipeline (5-stage core + optional extension), methodology |
| [Science Suite](plugins/science-suite/) | 13 | 4 | 23 → 126 | 5 events | JAX, Julia, physics, ML/DL/HPC, nonlinear dynamics |

## Specialist Agents

27 agents with tiered model assignments: **8 opus** (architect: research, planning, review, theory), **17 sonnet** (contractor: code edits, implementation, debugging), **2 haiku** (mechanical docs, comment triage).

Tiers name Claude Code model aliases, not pinned versions, so each agent tracks the current generation: `opus` → Opus 5.5 ($4/$20 per MTok), `sonnet` → Sonnet 5.5 ($2/$10), `haiku` → Haiku 5.5 ($0.10/$0.50). Effort is set per role, because the right level differs by model:

| Role | Model | `effort` | Why |
|------|-------|----------|-----|
| Architecture, theory, audit, research design (8 agents) | `opus` | `high` | Opus 5.5 defaults to `medium`; these agents are intelligence-sensitive, so `high` is deliberate |
| Numerics-critical code (`jax-pro`, `julia-pro`, `pinn-engineer`, `simulation-expert`) | `sonnet` | `high` | Silent numerical errors cost more than the extra tokens |
| Other code-writing agents (`app-developer`, `automation-engineer`, `sre-expert`, `julia-implementer`, `julia-ml-hpc`, `ml-expert`, `python-pro`, `sci-workflow-engineer`) | `sonnet` | `medium` | Medium is the recommended start for agentic coding and multistep tool use on Sonnet 5.5 |
| Mechanical documentation (`documentation-expert`) | `haiku` | `low` | Cheapest tier for low-judgment generation |
| `/review-pr` fan-out passes (`code-reviewer`, `pr-test-analyzer`, `silent-failure-hunter`, `type-design-analyzer`, `code-simplifier`, `comment-analyzer`) | `sonnet` / `haiku` | upstream | Adopted at byte parity from `ecc`; parallel, deliberately light passes |

`xhigh` is available on these models; set it per dispatch for the hardest long-horizon runs rather than as a default. Agent prompts avoid step-by-step scaffolds and self-check checklists that older models needed, since current models plan and verify on their own. Token controls: agents no longer preload skills (`skills:` frontmatter injected each hub's full text into every dispatch, 1k–5k tokens); they read the skill they need by path. Only the 42 hub skills are model-invocable, so the session skill listing stays small, and sub-skills are reached through hubs or `/suite:name`. Agent bodies were trimmed to what the model cannot infer, and the `UserPromptSubmit` routing hints in dev-suite and science-suite fire only when a prompt matches a hub's domain and repeat at most once per 12 prompts.

Two further models sit outside the alias set:

| Model | When to pin it |
|-------|----------------|
| `fable` (Fable 5.1) | Advisor role only (a stronger second opinion on the hardest problems); no agent is pinned to it. About 2.5× Opus 5.5 per token ($10/$50 per MTok), requires 30-day data retention (not available under zero data retention), and rejects forced `tool_choice: any\|tool` with a 400. |
| Opus 4.8 / Sonnet 4.6 | Previous generation, still served. Pin one to reproduce an older run, or as a refusal fallback for `opus` work. Sonnet 4.6 costs *more* than Sonnet 5.5 ($3/$15 vs $2/$10 per MTok), so pin it for reproducibility, never to save money. |

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
| `@julia-implementer` | Science | sonnet | Julia package lifecycle: tests, docs, exports, precompile, profile, debug, audit |
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
