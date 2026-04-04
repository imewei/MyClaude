Development Workflows
=====================

Patterns for using the **dev-suite** agents and :term:`hub skills <Hub Skill>` in software development.

.. note::

   Since v3.1.0, skills use a two-tier :term:`Hub Skill` architecture. The hub
   skills listed below route to specialized sub-skills via their
   :term:`Routing Decision Tree`. You invoke the hub; it dispatches to the
   right sub-skill automatically.

Feature Development
-------------------

The standard feature workflow uses ``@software-architect`` for design and
``@app-developer`` for implementation.

1. Design the API contract (hub: ``backend-patterns`` → sub: ``api-design-principles``).
2. Scaffold the project structure (skill: ``engineering-suite:scaffold``).
3. Implement backend logic (hub: ``architecture-and-infra`` → sub: ``architecture-patterns``).
4. Build the frontend (hub: ``frontend-and-mobile`` → sub: ``frontend-mobile-engineering``).
5. Write tests (hub: ``testing-and-quality`` → sub: ``javascript-testing-patterns`` or ``python-testing-patterns``).
6. Validate with ``/double-check``.

**Agent team:** Use :doc:`Team 27 (feature-dev) </agent-teams-guide>` for
multi-agent feature development with review gates.

Code Quality Pipeline
---------------------

Maintain code quality with the **dev-suite** agents.

.. code-block:: bash

   # Generate tests for new code
   /dev-suite:test-generate src/new_module.py

   # Run all tests iteratively until green
   /dev-suite:run-all-tests

   # Full validation before merge
   /dev-suite:double-check --deep

**Agent team:** Use :doc:`Team 3 (quality-audit) </agent-teams-guide>` for
pre-release quality gates.

Legacy Modernization
--------------------

Modernize legacy codebases with the Strangler Fig pattern.

1. Analyze the existing codebase (command: ``/adopt-code``).
2. Plan the migration strategy (command: ``/modernize``).
3. Implement incrementally with backward compatibility.
4. Validate each phase with ``/double-check``.

**Agent team:** Use :doc:`Team 6 (modernization) </agent-teams-guide>` for
coordinated migration.

Related
-------

- :doc:`/suites/dev-suite` — Full dev-suite reference (9 hubs → 49 sub-skills)
- :doc:`/glossary` — Hub Skill, Sub-Skill, and Routing Decision Tree definitions
