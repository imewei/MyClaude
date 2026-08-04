Integration Patterns
====================

Patterns for combining agents and :term:`hub skills <Hub Skill>` across multiple suites to solve
cross-cutting concerns.

.. note::

   Since v3.1.0, skills use a two-tier :term:`Hub Skill` architecture (50 hubs
   routing to 148 sub-skills as of v4.0.0). Cross-suite workflows invoke hub skills which
   automatically dispatch to the right sub-skill.

Cross-Suite Delegation
-----------------------

Key cross-suite patterns:

- **Performance optimization**: Combines ``@python-pro`` and
  ``@jax-pro`` from science-suite with ``@sre-expert`` from dev-suite.

- **HPC interoperability**: Bridges ``@julia-pro`` and
  ``@python-pro`` from science-suite with ``@software-architect`` from
  dev-suite.

- **Reproducible research**: Connects ``@research-expert`` from
  science-suite with ``@automation-engineer`` from dev-suite.

MCP Server Integration
----------------------

MyClaude agents can leverage MCP servers for enhanced capabilities:

- **Context7** — Library documentation lookup. Used by all agents to
  access up-to-date API references.

Official Plugin Integration
---------------------------

MyClaude agents integrate official plugin agents (pr-review-toolkit, feature-dev,
plugin-dev) as quality gates alongside MyClaude domain experts.

The **Quality Gate Enhancers** pattern lets you add review agents to any
existing workflow.

Related
-------

- :doc:`/integration-map` — Suite dependencies, MCP roles, and skill coverage
- :doc:`/suites/dev-suite` — CI/CD and automation (9 hubs → 35 sub-skills)
- :doc:`/glossary` — Hub Skill, Sub-Skill, and Routing Decision Tree definitions
