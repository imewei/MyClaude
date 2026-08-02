# Quick Reference Cheatsheet

**3 Suites** | **20 Agents** | **15 Registered Commands** | **51 Hub Skills** (routing to 148 sub-skills; 199 SKILL.md on disk)
**Version:** 4.0.0

---

## The Hub Architecture

MyClaude v4.0.0 uses a **hub-skill architecture**: skills are organized into hub skills (meta-orchestrators) that route to specialized sub-skills. Only hubs are declared in `plugin.json`; sub-skills are discovered through hub routing trees.

```
plugin.json → hub skill → routing decision tree → sub-skill
```

---

## Suite Overview

### 1. Dev Suite (`dev-suite`)

**Purpose:** Full-stack engineering, infrastructure, CI/CD, quality, and debugging.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 6 | 1 opus, 4 sonnet, 1 haiku |
| Commands | 10 registered | `/docs`, `/double-check`, `/eng-feature-dev`, `/fix-commit-errors`, `/merge-all`, `/modernize`, `/run-all-tests`, `/smart-debug`, `/test-generate`, `/workflow-automate` |
| Skills | 9 hubs → 35 sub | dev-hub, three-brain, architecture-and-infra, backend-patterns, ci-cd-pipelines, data-and-security, dev-workflows, observability-and-sre, testing-and-quality |
| Hooks | 6 events | SessionStart, PostToolUse, SubagentStop, TaskCompleted, SessionEnd, StopFailure |

### 2. Research Suite (`research-suite`)

**Purpose:** Peer review, 8-stage research-spark pipeline, and methodology orchestration.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 2 | research-expert (opus), research-spark-orchestrator (opus) |
| Commands | 3 registered | `/lit-review`, `/paper-implement`, `/replicate` |
| Skills | 11 hubs → 6 sub | research-hub, experiment-designer, falsifiable-claim, landscape-scanner, numerical-prototype, premortem-critique, research-practice, research-spark, scientific-review, spark-articulator, theory-scaffold |
| Hooks | 2 events | SessionStart (artifact-resume), TaskCompleted (audit log) |

### 3. Science Suite (`science-suite`)

**Purpose:** HPC, physics simulations, ML/DL, Julia, JAX, and nonlinear dynamics.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 12 | 7 opus, 4 sonnet, 1 haiku |
| Commands | 2 registered | `/md-sim`, `/benchmark` (plus `analyze-data`, `run-experiment` on disk, skill-invoked) |
| Skills | 30 hubs → 107 sub | science-hub, advanced-simulations, bayesian-inference, bayesian-ude-workflow, continuum-mechanics-and-rheology, correlation-analysis, deep-learning, deep-learning-hub, equation-discovery, jax-computing, julia-language, julia-mastery, julia-ml-and-dl, llm-and-ai, machine-learning, md-simulation-setup, ml-and-data-science, ml-deployment, neural-pde, nonlinear-dynamics, parallel-computing, python-development, research-and-domains, sciml-and-diffeq, sciml-modern-stack, self-improving-ai, simulation-and-hpc, statistical-physics, statistical-physics-hub, time-series-analysis |
| Hooks | 4 events | SessionStart, PostToolUse, SessionEnd, SubagentStop |

**Total hook events across all suites:** 12

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

### Research (idea → plan)
1. `@research-spark-orchestrator` — drive the 8-stage pipeline
2. Stages 1-8 emit `01_spark.md` → `08_premortem.md` into `artifacts/`
3. `SubagentStop` hook verifies each stage artifact before advancing

### Peer Review (manuscript → .docx)
1. Skill auto-triggers on "review this paper" phrasings
2. `scientific-review` produces a journal-adapted Six-Lens referee report
3. Output: `.docx` (with `python-docx`) or markdown fallback

### Agent Teams
See [Agent Teams Guide](../agent-teams-guide.md) for the full agent composition of each team and variant.

---

## Model Tier Quick Reference

| Tier | Count | Use Case | Agents |
|------|-------|----------|--------|
| **opus** | 10 | Deep reasoning, architecture, research | software-architect, research-expert, research-spark-orchestrator, continuum-mechanics-engineer, jax-pro, julia-pro, neural-network-master, nonlinear-dynamics-expert, pinn-engineer, statistical-physicist |
| **sonnet** | 8 | Standard development and analysis | app-developer, automation-engineer, quality-specialist, sre-expert, julia-ml-hpc, python-pro, sci-workflow-engineer, simulation-expert |
| **haiku** | 2 | Fast, simple tasks | documentation-expert, ml-expert |

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

- [Agent Reference](agents.md) — All 20 agents with model tiers and delegation patterns
- [Commands Reference](commands.md) — 15 registered + 2 skill-invoked commands
- [Integration Map](../integration-map.rst) — Suite dependencies, MCP server roles, skill coverage
- [Agent Teams Guide](../agent-teams-guide.md) — 10 focused teams with 19 variants (codebase-aware recommender)
- [Glossary](../glossary.rst) — Hub Skill, Sub-Skill, Agent Team, Routing Decision Tree
- [GitHub Repository](https://github.com/imewei/MyClaude)

*Generated from v4.0.0 validated marketplace data.*
