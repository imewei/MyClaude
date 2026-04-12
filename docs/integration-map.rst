Integration Map
===============

How the 3 MyClaude suites (24 agents, 26 hub skills routing to 180 sub-skills) connect to each other and to external tools.

.. contents:: Table of Contents
   :depth: 2

Suite Dependencies
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Suite
     - Integrates With
   * - **agent-core**
     - All suites (orchestration layer). MCP: Sequential Thinking, Context7.
   * - **dev-suite**
     - Internal: all 9 agents cross-delegate freely. MCP: Serena, GitHub.
   * - **science-suite**
     - agent-core (reasoning), dev-suite (packaging). Internal: julia-pro ↔ julia-ml-hpc (SciML vs ML/HPC boundary), neural-network-master ↔ julia-ml-hpc (theory vs Julia impl). 14 hub skills route to 117 sub-skills. MCP: Context7.

MCP Server Roles
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Server
     - Command
     - Used By
   * - **Serena**
     - ``/code-analyze``
     - engineering, infrastructure, quality agents for symbol navigation
   * - **Context7**
     - ``/docs-lookup``
     - All agents for up-to-date library documentation
   * - **Sequential Thinking**
     - ``/ultra-think``
     - ``@reasoning-engine`` for structured multi-step analysis

Intra-Suite Delegation Patterns
-------------------------------

**agent-core** (3 agents): Fully connected triangle — orchestrator, reasoning-engine,
and context-specialist each delegate to the other two.

**dev-suite** (9 agents): Free internal delegation with key edges:

- software-architect ↔ devops-architect (architecture ↔ infrastructure)
- quality-specialist ↔ debugger-pro (testing ↔ debugging)
- automation-engineer ↔ devops-architect (CI/CD ↔ deployment)
- sre-expert ↔ devops-architect (reliability ↔ infrastructure)

**science-suite** (12 agents): Hub-and-spoke with domain boundaries:

- julia-pro ↔ julia-ml-hpc: SciML/ODE boundary — julia-pro owns UDEs and Lux.jl-for-physics; julia-ml-hpc owns ML training, GPU, and HPC
- neural-network-master → julia-ml-hpc: DL theory → Julia implementation
- ml-expert → julia-ml-hpc: Python ML → Julia ML pipelines
- simulation-expert → julia-ml-hpc: HPC → Julia GPU kernels
- nonlinear-dynamics-expert → julia-pro / jax-pro: Theory → implementation
- statistical-physicist → jax-pro: Theory → JAX implementation

Skill Coverage
~~~~~~~~~~~~~~

All 26 hub skills route to 180 sub-skills with 100% Expert Agent coverage:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 30

   * - Suite
     - Agents
     - Hubs
     - Sub-Skills
     - Coverage
   * - agent-core
     - 3
     - 3
     - 14
     - 100% — agent-systems, reasoning-and-memory, llm-engineering
   * - dev-suite
     - 9
     - 9
     - 49
     - 100% — mapped across 9 domain agents
   * - science-suite
     - 12
     - 14
     - 117
     - 100% — including Julia ML/HPC and nonlinear dynamics hubs

Official Plugin Agents
----------------------

23 agents from `claude-plugins-official <https://github.com/anthropics/claude-plugins-official>`_
complement the 24 MyClaude domain experts. See the :doc:`Agent Teams Guide <agent-teams-guide>`
for teams 26-33 that integrate these agents.

Key integration patterns:

- **Build + Review**: MyClaude agents produce code, official plugin agents validate it.
- **Quality Gate Enhancers**: Add pr-review-toolkit or coderabbit agents to any team.
- **Agent SDK**: Use agent-sdk-dev verifiers alongside ``@ai-engineer`` for SDK projects.

Agent Teams
-----------

10 focused team configurations with 20 variants and long-running workflow protocol (v3.3.0):

1. **feature-dev** — Build any feature end-to-end
2. **debug** (5 variants) — All debugging + incident response
3. **quality-gate** (2 variants) — Code review + security audit
4. **api-infra** (2 variants) — APIs + cloud + CI/CD + config
5. **sci-compute** (7 variants) — All scientific computing (auto-detects domain)
6. **modernize** — Legacy migration + refactoring
7. **ai-engineering** (1 variant) — LLM apps + RAG + multi-agent
8. **ml-deploy** (2 variants) — Model deploy + data pipelines + performance
9. **docs-publish** (1 variant) — Documentation + reproducibility
10. **plugin-forge** — Claude Code extensions

Use ``/agent-core:team-assemble list`` to browse all teams, or run it with no
arguments in a project root for a codebase-aware recommendation.
