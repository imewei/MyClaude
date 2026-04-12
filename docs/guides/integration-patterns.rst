Integration Patterns
====================

Patterns for combining agents and :term:`hub skills <Hub Skill>` across multiple suites to solve
cross-cutting concerns.

.. note::

   Since v3.1.0, skills use a two-tier :term:`Hub Skill` architecture (26 hubs
   routing to 180 sub-skills as of v3.2.0). Cross-suite workflows invoke hub skills which
   automatically dispatch to the right sub-skill.

Cross-Suite Agent Teams
-----------------------

The :doc:`Agent Teams Guide </agent-teams-guide>` provides 25 ready-to-use
team configurations (v3.1.4+). Teams 13-16 and 22-25 are specifically designed
for cross-suite collaboration.

Key cross-suite patterns:

- **Performance optimization** (Team 17): Combines ``@systems-engineer``
  from dev-suite with ``@jax-pro`` from science-suite and
  ``@debugger-pro`` from dev-suite.

- **HPC interoperability** (Team 18): Bridges ``@julia-pro`` and
  ``@python-pro`` from science-suite with ``@systems-engineer`` from
  dev-suite.

- **Reproducible research** (Team 19): Connects ``@research-expert`` from
  science-suite with ``@automation-engineer`` from dev-suite.

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

- :doc:`/integration-map` — Suite dependencies, MCP roles, and skill coverage
- :doc:`/suites/agent-core` — Orchestration and coordination (3 hubs → 12 sub-skills)
- :doc:`/suites/dev-suite` — CI/CD and automation (9 hubs → 49 sub-skills)
- :doc:`/glossary` — Hub Skill, Sub-Skill, and Routing Decision Tree definitions
