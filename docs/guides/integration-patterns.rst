Integration Patterns
====================

Patterns for combining agents and skills across multiple suites to solve
cross-cutting concerns.

Cross-Suite Agent Teams
-----------------------

The :doc:`Agent Teams Guide </agent-teams-guide>` provides 38 ready-to-use
team configurations. Teams 17-25 are specifically designed for cross-suite
collaboration.

Key cross-suite patterns:

- **Performance optimization** (Team 17): Combines ``@systems-engineer``
  from engineering-suite with ``@jax-pro`` from science-suite and
  ``@debugger-pro`` from quality-suite.

- **HPC interoperability** (Team 18): Bridges ``@julia-pro`` and
  ``@python-pro`` from science-suite with ``@systems-engineer`` from
  engineering-suite.

- **Reproducible research** (Team 19): Connects ``@research-expert`` from
  science-suite with ``@automation-engineer`` from infrastructure-suite.

MCP Server Integration
----------------------

MyClaude agents can leverage MCP servers for enhanced capabilities:

- **Serena** — Semantic code analysis via ``/code-analyze`` command.
  Used by engineering and quality agents for symbol-level navigation.

- **Context7** — Library documentation lookup via ``/docs-lookup`` command.
  Used by all agents to access up-to-date API references.

- **Sequential Thinking** — Structured reasoning via ``/ultra-think`` command.
  Used by ``@reasoning-engine`` for multi-step analysis.

Official Plugin Integration
---------------------------

Teams 26-33 integrate official plugin agents (pr-review-toolkit, feature-dev,
coderabbit, plugin-dev) as quality gates alongside MyClaude domain experts.

The **Quality Gate Enhancers** pattern lets you add review agents to any
existing team. See the :doc:`Agent Teams Guide </agent-teams-guide>` for
details.

Related
-------

- :doc:`/suites/agent-core` — Orchestration and coordination
- :doc:`/suites/infrastructure-suite` — CI/CD and automation
