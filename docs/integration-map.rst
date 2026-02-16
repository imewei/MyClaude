Integration Map
===============

How the 5 MyClaude suites connect to each other and to external tools.

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
   * - **engineering-suite**
     - quality-suite (testing), infrastructure-suite (deployment). MCP: Serena.
   * - **infrastructure-suite**
     - engineering-suite (app config), quality-suite (validation). MCP: Serena, GitHub.
   * - **quality-suite**
     - All suites (quality gates). MCP: Serena.
   * - **science-suite**
     - agent-core (reasoning), engineering-suite (packaging). MCP: Context7.

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

Official Plugin Agents
----------------------

20 agents from `claude-plugins-official <https://github.com/anthropics/claude-plugins-official>`_
complement MyClaude domain experts. See the :doc:`Agent Teams Guide <agent-teams-guide>`
for teams 26-33 that integrate these agents.

Key integration patterns:

- **Build + Review**: MyClaude agents produce code, official plugin agents validate it.
- **Quality Gate Enhancers**: Add pr-review-toolkit or coderabbit agents to any team.
- **Agent SDK**: Use agent-sdk-dev verifiers alongside ``@ai-engineer`` for SDK projects.

Agent Teams
-----------

38 pre-built team configurations span five categories:

1. **Development & Operations** (1-10): Feature dev, incident response, API design
2. **Scientific Computing** (11-16): Bayesian inference, MD simulations, ML force fields
3. **Cross-Suite Specialized** (17-25): HPC interop, prompt R&D, security hardening
4. **Official Plugin Integration** (26-33): PR review, plugin development, HuggingFace
5. **Debugging** (34-38): GUI threading, numerical/JAX, schema drift, triage, full audit

Use ``/agent-core:team-assemble list`` to browse all templates.
