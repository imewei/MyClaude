# Quick Reference Cheatsheet

**3 Suites** | **26 Agents** | **21 Registered Commands** | **42 Hub Skills** (routing to 157 sub-skills; 199 SKILL.md on disk)
**Version:** 4.0.4

---

## The Hub Architecture

MyClaude v4.0.4 uses a **hub-skill architecture**: skills are organized into hub skills (meta-orchestrators) that route to specialized sub-skills. Only hubs are declared in `plugin.json`; sub-skills are discovered through hub routing trees.

```
plugin.json → hub skill → routing decision tree → sub-skill
```

---

## Suite Overview

### 1. Dev Suite (`dev-suite`)

**Purpose:** Full-stack engineering, infrastructure, CI/CD, quality, and debugging.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 12 | 2 opus, 8 sonnet, 2 haiku (6 original + 6 `/review-pr` fan-out agents adopted from `ecc`) |
| Commands | 14 registered | `/docs`, `/double-check`, `/eng-feature-dev`, `/fix-commit-errors`, `/modernize`, `/run-all-tests`, `/smart-debug`, `/test-generate`, `/workflow-automate`, `/review-pr`, `/code-review`, `/commit`, `/refactor-clean`, `/git-branch` (finish/clean/rollback/worktree) |
| Skills | 9 hubs → 36 sub | dev-hub, three-brain, architecture-and-infra, backend-patterns, ci-cd-pipelines, data-and-security, dev-workflows (includes `config-gc`, adopted from `ecc`), observability-and-sre, testing-and-quality |
| Hooks | 7 events | SessionStart, UserPromptSubmit, PostToolUse, SubagentStop, TaskCompleted, SessionEnd, StopFailure |

### 2. Research Suite (`research-suite`)

**Purpose:** Peer review, research-spark pipeline (5-stage core + optional extension), and methodology orchestration.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 2 | research-expert (opus), research-spark-orchestrator (opus) |
| Commands | 3 registered | `/lit-review`, `/paper-implement`, `/replicate` |
| Skills | 10 hubs → 7 sub | research-hub, research-spark, scientific-review, spark-articulator, landscape-scanner, falsifiable-claim, theory-scaffold, numerical-prototype, experiment-designer, premortem-critique (research-practice is a sub-skill reached via research-hub) |
| Hooks | 4 events | SessionStart (artifact-resume), PostToolUse (scientific-review deliverable check), SubagentStop (research-spark artifact check), TaskCompleted (audit log) |

### 3. Science Suite (`science-suite`)

**Purpose:** HPC, physics simulations, ML/DL, Julia, JAX, and nonlinear dynamics.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 12 | 4 opus, 7 sonnet, 1 haiku |
| Commands | 4 registered | `/md-sim`, `/benchmark`, `/analyze-data`, `/run-experiment` |
| Skills | 23 hubs → 114 sub | science-hub, advanced-simulations, bayesian-inference, continuum-mechanics-and-rheology, deep-learning, deep-learning-hub, jax-computing, julia-language, julia-mastery, julia-ml-and-dl, llm-and-ai, machine-learning, ml-and-data-science, ml-deployment, nonlinear-dynamics, parallel-computing, python-development, research-and-domains, sciml-and-diffeq, simulation-and-hpc, statistical-physics, statistical-physics-hub, time-series-analysis |
| Hooks | 5 events | SessionStart, UserPromptSubmit, PostToolUse, SessionEnd, SubagentStop |

**Total hook events across all suites:** 16

---

## Common Workflows

### Engineering Feature Development
1. `@software-architect` — design the system
2. `/eng-feature-dev` — implement the core logic
3. `/test-generate` — create tests
4. `/double-check` — validate before submitting

### Scientific Simulation
1. `@simulation-expert` — design the simulation
2. `@jax-pro` or `@julia-pro` — implement numerical kernels
3. `@research-expert` — analyze and visualize results

### Research (idea to plan)
1. `@research-spark-orchestrator` — drive the pipeline
2. Stages 1-5 (core) emit `01_spark.md` → `05_formalism.tex` into `artifacts/`, ending in a fundable proposal; Stages 6-8 (optional) continue to `08_premortem.md` only on request
3. `SubagentStop` hook verifies each stage artifact before advancing

### Peer Review (manuscript to .docx)
1. Skill auto-triggers on "review this paper" phrasings
2. `scientific-review` produces a journal-adapted Six-Lens referee report
3. Output: `.docx` (with `python-docx`) or markdown fallback

---

## Model Tier Quick Reference

| Tier | Count | Use Case | Agents |
|------|-------|----------|--------|
| **opus** | 8 | Architect: research, planning, review/audit, theory | software-architect, quality-specialist, research-expert, research-spark-orchestrator, continuum-mechanics-engineer, neural-network-master, nonlinear-dynamics-expert, statistical-physicist |
| **sonnet** | 11 | Contractor: code edits, implementation, debugging | app-developer, automation-engineer, sre-expert, jax-pro, julia-pro, pinn-engineer, ml-expert, julia-ml-hpc, python-pro, sci-workflow-engineer, simulation-expert |
| **haiku** | 1 | Mechanical doc generation | documentation-expert |

---

## Installation

```bash
# Add the marketplace
/plugin marketplace add imewei/MyClaude

# Install suites
/plugin install dev-suite@marketplace
/plugin install research-suite@marketplace
/plugin install science-suite@marketplace
```

---

## Resources

- [Agent Reference](agents.md) — All 26 agents with model tiers and delegation patterns
- [Commands Reference](commands.md) — 21 registered commands
- [Integration Map](../integration-map.rst) — Suite dependencies, MCP server roles, skill coverage
- [Glossary](../glossary.rst) — Hub Skill, Sub-Skill, Routing Decision Tree
- [GitHub Repository](https://github.com/imewei/MyClaude)

*Generated from v4.0.4 validated marketplace data.*
