Integration Map
===============

How the 3 MyClaude suites (20 agents, 50 registered hub skills routing to 148 sub-skills) connect to each other and to external tools.

.. contents:: Table of Contents
   :depth: 2

Suite Dependencies
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Suite
     - Integrates With
   * - **dev-suite**
     - Internal: all 6 agents cross-delegate freely. MCP: Serena, GitHub.
   * - **research-suite**
     - science-suite (Stage 6 JAX/Julia/MD delegation from research-spark-orchestrator). 11 registered skills (8-stage pipeline + 2 hubs + standalone ``scientific-review``) route to 6 sub-skills (5 methodology specialists + the ``_research-commons`` resource hub). MCP: Context7 for journal guideline lookups.
   * - **science-suite**
     - dev-suite (packaging), research-suite (invoked for Stage 6 implementation). Internal: julia-pro <-> julia-ml-hpc (SciML vs ML/HPC boundary), neural-network-master <-> julia-ml-hpc (theory vs Julia impl). 30 hub skills route to 107 sub-skills. MCP: Context7.

MCP Server Roles
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Server
     - Command
     - Used By
   * - **Serena**
     - --
     - dev-suite agents for symbol navigation and semantic code search
   * - **Context7**
     - --
     - All agents for up-to-date library documentation

Intra-Suite Delegation Patterns
-------------------------------

**dev-suite** (6 agents): Free internal delegation with key edges:

- software-architect <-> sre-expert (architecture <-> reliability/infrastructure)
- quality-specialist <-> software-architect (validation <-> design)
- automation-engineer <-> sre-expert (CI/CD <-> deployment and monitoring)
- documentation-expert <-> software-architect (docs <-> interface design)

**research-suite** (2 agents): Pipeline-gated with optional cross-suite fan-out.

- research-spark-orchestrator → research-expert: Off-pipeline methodology questions
- research-spark-orchestrator → jax-pro / julia-pro / simulation-expert (science-suite): Stage 6 numerical prototype implementation
- research-spark-orchestrator → nonlinear-dynamics-expert / statistical-physicist (science-suite): Stages 4-5 theory work
- research-expert: No intra-suite delegation (one-off methodology specialist)

**science-suite** (12 agents): Hub-and-spoke with domain boundaries.

- julia-pro <-> julia-ml-hpc: SciML/ODE boundary — julia-pro owns UDEs and Lux.jl-for-physics; julia-ml-hpc owns ML training, GPU, and HPC
- neural-network-master → julia-ml-hpc: DL theory → Julia implementation
- ml-expert → julia-ml-hpc: Python ML → Julia ML pipelines
- simulation-expert → julia-ml-hpc: HPC → Julia GPU kernels
- nonlinear-dynamics-expert → julia-pro / jax-pro: Theory → implementation
- statistical-physicist → jax-pro: Theory → JAX implementation

Skill Coverage
~~~~~~~~~~~~~~

All 50 registered hub skills route to 148 sub-skills with 100% Expert Agent coverage:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 30

   * - Suite
     - Agents
     - Registered Skills
     - Sub-Skills
     - Coverage
   * - dev-suite
     - 6
     - 10
     - 35
     - 100% — mapped across 6 domain agents
   * - research-suite
     - 2
     - 11
     - 6
     - 100% — scientific-review standalone + 8-stage pipeline + research-practice hub
   * - science-suite
     - 12
     - 30
     - 107
     - 100% — including Julia ML/HPC and nonlinear dynamics hubs

Official Plugin Agents
----------------------

18 agents from 8 official plugins complement the 20 MyClaude domain experts.
See the :doc:`Agent Teams Guide <agent-teams-guide>` for team configurations
that integrate these agents.

Key integration patterns:

- **Build + Review**: MyClaude agents produce code, official plugin agents validate it.
- **Quality Gate Enhancers**: Add pr-review-toolkit agents to any team for automated review.
- **Agent SDK**: Use agent-sdk-dev verifiers alongside ``@sci-workflow-engineer`` for SDK projects.

Agent Teams
-----------

10 focused team configurations with 19 variants and long-running workflow protocol (v4.0.0):

1. **feature-dev** — Build any feature end-to-end
2. **debug** (5 variants) — All debugging + incident response
3. **quality-gate** (2 variants) — Code review + security audit
4. **api-infra** (2 variants) — APIs + cloud + CI/CD + config
5. **sci-compute** (7 variants) — All scientific computing (auto-detects domain)
6. **modernize** — Legacy migration + refactoring
7. **ai-engineering** — LLM apps + RAG
8. **ml-deploy** (2 variants) — Model deploy + data pipelines + performance
9. **docs-publish** (1 variant) — Documentation + reproducibility
10. **plugin-forge** — Claude Code extensions

See the :doc:`Agent Teams Guide <agent-teams-guide>` for the full agent
composition of each team and variant.
