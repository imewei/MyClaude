# Quick Reference Cheatsheet

**4 Suites** | **25 Agents** | **14 Registered Commands** | **31 Hub Skills** (routing to 186 sub-skills; 217 SKILL.md on disk)
**Version:** 3.4.0

---

## The Hub Architecture

MyClaude v3.4.0 uses a **hub-skill architecture**: skills are organized into hub skills (meta-orchestrators) that route to specialized sub-skills. Only hubs are declared in `plugin.json`; sub-skills are discovered through hub routing trees.

```
plugin.json → hub skill → routing decision tree → sub-skill
```

---

## Suite Overview

### 1. Agent Core (`agent-core`)

**Purpose:** Multi-agent coordination, advanced reasoning, and context engineering.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 3 | orchestrator (opus), reasoning-engine (opus), context-specialist (opus) |
| Commands | 2 registered | `/ultra-think`, `/team-assemble` |
| Skills | 4 hubs → 13 sub | agent-systems, reasoning-and-memory, llm-engineering, thinkfirst |
| Hooks | 12 events | SessionStart, SessionEnd, PreToolUse, PostToolUse, PreCompact, PostCompact, SubagentStart, SubagentStop, PermissionDenied, TaskCreated, TaskCompleted, StopFailure |

### 2. Dev Suite (`dev-suite`)

**Purpose:** Full-stack engineering, infrastructure, CI/CD, quality, and debugging.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 9 | 2 opus, 6 sonnet, 1 haiku |
| Commands | 12 registered | `/commit`, `/docs`, `/double-check`, `/eng-feature-dev`, `/fix-commit-errors`, `/merge-all`, `/modernize`, `/refactor-clean`, `/run-all-tests`, `/smart-debug`, `/test-generate`, `/workflow-automate` |
| Skills | 9 hubs → 49 sub | backend-patterns, frontend-and-mobile, architecture-and-infra, testing-and-quality, ci-cd-pipelines, observability-and-sre, python-toolchain, data-and-security, dev-workflows |
| Hooks | 7 events | SessionStart, PreToolUse, PostToolUse, SubagentStop, TaskCompleted, SessionEnd, StopFailure |

### 3. Research Suite (`research-suite`)

**Purpose:** Peer review, 8-stage research-spark pipeline, and methodology orchestration.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 2 | research-expert (opus), research-spark-orchestrator (opus) |
| Commands | 0 registered | (all workflows are skill-driven) |
| Skills | 4 hubs → 12 sub | scientific-review, research-spark, research-practice, _research-commons |
| Hooks | 3 events | SessionStart (artifact-resume), TaskCompleted (audit log), SubagentStop (prompt-based artifact gating) |

### 4. Science Suite (`science-suite`)

**Purpose:** HPC, physics simulations, ML/DL, Julia, JAX, and nonlinear dynamics.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 11 | 4 opus, 7 sonnet |
| Commands | 0 registered | (skill-invoked reference templates on disk) |
| Skills | 14 hubs → 112 sub | nonlinear-dynamics, jax-computing, julia-language, julia-ml-and-dl, sciml-and-diffeq, correlation-analysis, statistical-physics-hub, deep-learning-hub, ml-and-data-science, llm-and-ai, ml-deployment, simulation-and-hpc, research-and-domains, bayesian-inference |
| Hooks | 5 events | SessionStart, PreToolUse, PostToolUse, SessionEnd, SubagentStop |

**Total hook events across all suites:** 27

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
1. `/team-assemble list` — see all team templates
2. `/team-assemble <type>` — generate a team prompt
3. See [Agent Teams Guide](../agent-teams-guide.md) for details

---

## Model Tier Quick Reference

| Tier | Count | Use Case | Agents |
|------|-------|----------|--------|
| **opus** | 11 | Deep reasoning, architecture, research | orchestrator, reasoning-engine, context-specialist, software-architect, debugger-pro, research-expert, research-spark-orchestrator, neural-network-master, nonlinear-dynamics-expert, simulation-expert, statistical-physicist |
| **sonnet** | 13 | Standard development and analysis | app-developer, automation-engineer, devops-architect, quality-specialist, sre-expert, systems-engineer, ai-engineer, jax-pro, julia-ml-hpc, julia-pro, ml-expert, prompt-engineer, python-pro |
| **haiku** | 1 | Fast, simple tasks | documentation-expert |

---

## Installation

```bash
# Add the marketplace
/plugin marketplace add imewei/MyClaude

# Install suites
/plugin install agent-core@marketplace
/plugin install dev-suite@marketplace
/plugin install research-suite@marketplace
/plugin install science-suite@marketplace
```

---

## Resources

- [Agent Reference](agents.md) — All 25 agents with model tiers and delegation patterns
- [Commands Reference](commands.md) — 14 registered + 21 skill-invoked commands
- [Integration Map](../integration-map.rst) — Suite dependencies, MCP server roles, skill coverage
- [Agent Teams Guide](../agent-teams-guide.md) — 10 focused teams with 20 variants (codebase-aware recommender)
- [Glossary](../glossary.rst) — Hub Skill, Sub-Skill, Agent Team, Routing Decision Tree
- [GitHub Repository](https://github.com/imewei/MyClaude)

*Generated from v3.4.0 validated marketplace data.*
