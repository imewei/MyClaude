# Quick Reference Cheatsheet

**3 Suites** | **24 Agents** | **14 Registered Commands** | **26 Hub Skills** (routing to 167 sub-skills)
**Version:** 3.1.1

---

## The Hub Architecture

MyClaude v3.1.1 uses a **hub-skill architecture**: skills are organized into hub skills (meta-orchestrators) that route to specialized sub-skills. Only hubs are declared in `plugin.json`; sub-skills are discovered through hub routing trees.

```
plugin.json → hub skill → routing decision tree → sub-skill
```

---

## Suite Overview

### 1. Agent Core (`agent-core`)

**Purpose:** Multi-agent coordination, advanced reasoning, and context engineering.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 3 | orchestrator (opus), reasoning-engine (opus), context-specialist (sonnet) |
| Commands | 2 registered | `/ultra-think`, `/team-assemble` |
| Skills | 3 hubs → 12 sub | agent-systems, reasoning-and-memory, llm-engineering |
| Hooks | 8 events | SessionStart, PreToolUse, PostToolUse, PreCompact, PostCompact, SubagentStop, PermissionDenied, TaskCompleted |

### 2. Dev Suite (`dev-suite`)

**Purpose:** Full-stack engineering, infrastructure, CI/CD, quality, and debugging.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 9 | 2 opus, 6 sonnet, 1 haiku |
| Commands | 12 registered | `/commit`, `/docs`, `/double-check`, `/eng-feature-dev`, `/fix-commit-errors`, `/merge-all`, `/modernize`, `/refactor-clean`, `/run-all-tests`, `/smart-debug`, `/test-generate`, `/workflow-automate` |
| Skills | 9 hubs → 49 sub | backend-patterns, frontend-and-mobile, architecture-and-infra, testing-and-quality, ci-cd-pipelines, observability-and-sre, python-toolchain, data-and-security, dev-workflows |
| Hooks | 2 events | PostToolUse, SubagentStop |

### 3. Science Suite (`science-suite`)

**Purpose:** HPC, physics simulations, ML/DL, Julia, JAX, and research workflows.

| Component | Count | Details |
|-----------|-------|---------|
| Agents | 12 | 5 opus, 7 sonnet |
| Commands | 0 registered | (3 skill-invoked: analyze-data, paper-review, run-experiment) |
| Skills | 14 hubs → 106 sub | nonlinear-dynamics, jax-computing, julia-language, julia-ml-and-dl, sciml-and-diffeq, correlation-analysis, statistical-physics-hub, deep-learning-hub, ml-and-data-science, llm-and-ai, ml-deployment, simulation-and-hpc, research-and-domains, bayesian-inference |
| Hooks | 0 | — |

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

### Agent Teams
1. `/team-assemble list` — see all team templates
2. `/team-assemble <type>` — generate a team prompt
3. See [Agent Teams Guide](../agent-teams-guide.md) for details

---

## Model Tier Quick Reference

| Tier | Use Case | Agents |
|------|----------|--------|
| **opus** (9) | Deep reasoning, architecture, research | orchestrator, reasoning-engine, software-architect, debugger-pro, neural-network-master, nonlinear-dynamics-expert, research-expert, simulation-expert, statistical-physicist |
| **sonnet** (14) | Standard development and analysis | context-specialist, app-developer, automation-engineer, devops-architect, quality-specialist, sre-expert, systems-engineer, ai-engineer, jax-pro, julia-ml-hpc, julia-pro, ml-expert, prompt-engineer, python-pro |
| **haiku** (1) | Fast, simple tasks | documentation-expert |

---

## Installation

```bash
# Add the marketplace
/plugin marketplace add imewei/MyClaude

# Install suites
/plugin install agent-core@marketplace
/plugin install dev-suite@marketplace
/plugin install science-suite@marketplace
```

---

## Resources

- [Agent Reference](agents.md) — All 24 agents with model tiers and delegation patterns
- [Commands Reference](commands.md) — 14 registered + 22 skill-invoked commands
- [Integration Map](../integration-map.rst) — Suite dependencies, MCP server roles, skill coverage
- [Agent Teams Guide](../agent-teams-guide.md) — 21 pre-built team configurations
- [Glossary](../glossary.rst) — Hub Skill, Sub-Skill, Agent Team, Routing Decision Tree
- [GitHub Repository](https://github.com/imewei/MyClaude)

*Generated from v3.1.1 validated marketplace data.*
