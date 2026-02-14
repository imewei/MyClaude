Development Workflows
=====================

Patterns for using the **engineering-suite** and **quality-suite** agents in
software development.

Feature Development
-------------------

The standard feature workflow uses ``@software-architect`` for design and
``@app-developer`` for implementation.

1. Design the API contract (skill: ``api-design-principles``).
2. Scaffold the project structure (command: ``/scaffold``).
3. Implement backend logic (skill: ``architecture-patterns``).
4. Build the frontend (skill: ``frontend-mobile-engineering``).
5. Write tests (skill: ``python-testing-patterns`` or ``javascript-testing-patterns``).
6. Validate with ``/double-check``.

**Agent team:** Use :doc:`Team 1 (feature-dev) </agent-teams-guide>` for
multi-agent feature development.

Code Quality Pipeline
---------------------

Maintain code quality with the **quality-suite** agents.

.. code-block:: bash

   # Generate tests for new code
   /quality-suite:test-generate src/new_module.py

   # Run all tests iteratively until green
   /quality-suite:run-all-tests

   # Full validation before merge
   /quality-suite:double-check --deep

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

- :doc:`/suites/engineering-suite` — Full engineering-suite reference
- :doc:`/suites/quality-suite` — Full quality-suite reference
