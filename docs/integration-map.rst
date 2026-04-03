Integration Map
===============

How the 3 MyClaude suites (24 agents, 142 skills) connect to each other and to external tools.

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
     - agent-core (reasoning), dev-suite (packaging). Internal: julia-pro ↔ julia-ml-hpc (SciML vs ML/HPC boundary), neural-network-master ↔ julia-ml-hpc (theory vs Julia impl). MCP: Context7.

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

All 142 skills have Expert Agent pointers (100% coverage):

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Suite
     - Agents
     - Skills
     - Coverage
   * - agent-core
     - 3
     - 7
     - 100% — all skills cross-referenced
   * - dev-suite
     - 9
     - 39
     - 100% — mapped across 9 domain agents
   * - science-suite
     - 12
     - 96
     - 100% — including 10 new Julia ML/HPC skills

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

34 pre-built team configurations span five categories:

1. **Development & Operations** (1-10): Feature dev, incident response, API design
2. **Scientific Computing** (11-16): Bayesian inference, MD simulations, ML force fields
3. **Cross-Suite Specialized** (17-25): HPC interop, prompt R&D, security hardening
4. **Official Plugin Integration** (26-33): PR review, plugin development, HuggingFace
5. **Debugging** (34-38): GUI threading, numerical/JAX, schema drift, triage, full audit

Use ``/agent-core:team-assemble list`` to browse all templates.
