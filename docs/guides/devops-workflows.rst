DevOps Workflows
================

Patterns for using the **dev-suite** agents and :term:`hub skills <Hub Skill>` to automate CI/CD,
monitoring, and Git workflows.

.. note::

   Since v3.1.0, skills use a two-tier :term:`Hub Skill` architecture. The hub
   skills listed below route to specialized sub-skills via their
   :term:`Routing Decision Tree`.

CI/CD Pipeline Setup
--------------------

Use ``@automation-engineer`` to create production CI/CD pipelines.

1. Generate GitHub Actions workflow (command: ``/workflow-automate``).
2. Add security scanning gates (hub: ``ci-cd-pipelines`` → sub: ``security-ci-template``).
3. Configure deployment stages (hub: ``ci-cd-pipelines`` → sub: ``deployment-pipeline-design``).

.. code-block:: yaml

   # Example: Generated GitHub Actions workflow
   name: CI/CD
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - run: uv sync && uv run pytest
     security:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - run: uv run ruff check . && uv run pip-audit

**Agent team:** Use :doc:`Team 5 (infra-setup) </agent-teams-guide>` for
full infrastructure provisioning.

Observability Stack
-------------------

Use ``@sre-expert`` to set up monitoring and alerting.

1. Deploy Prometheus + Grafana (command: ``/monitor-setup``).
2. Create dashboards (hub: ``observability-and-sre`` → sub: ``grafana-dashboards``).
3. Define SLOs and error budgets (command: ``/slo-implement``).
4. Add distributed tracing (hub: ``observability-and-sre`` → sub: ``distributed-tracing``).

**Agent team:** Use :doc:`Team 5 (infra-setup) </agent-teams-guide>` with
``@sre-expert`` leading the observability track.

Git Workflow Automation
-----------------------

Streamline Git operations with ``@automation-engineer``.

.. code-block:: bash

   # Intelligent commit with analysis
   /dev-suite:commit

   # Fix CI failures automatically
   /dev-suite:fix-commit-errors

   # Merge all branches and clean up
   /dev-suite:merge-all

Related
-------

- :doc:`/suites/dev-suite` — Full dev-suite reference (9 hubs → 49 sub-skills)
- :doc:`/glossary` — Hub Skill, Sub-Skill, and Routing Decision Tree definitions
