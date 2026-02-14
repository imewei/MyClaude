DevOps Workflows
================

Patterns for using the **infrastructure-suite** agents to automate CI/CD,
monitoring, and Git workflows.

CI/CD Pipeline Setup
--------------------

Use ``@automation-engineer`` to create production CI/CD pipelines.

1. Generate GitHub Actions workflow (command: ``/workflow-automate``).
2. Add security scanning gates (skill: ``security-ci-template``).
3. Configure deployment stages (skill: ``deployment-pipeline-design``).

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
2. Create dashboards (skill: ``grafana-dashboards``).
3. Define SLOs and error budgets (command: ``/slo-implement``).
4. Add distributed tracing (skill: ``distributed-tracing``).

**Agent team:** Use :doc:`Team 5 (infra-setup) </agent-teams-guide>` with
``@sre-expert`` leading the observability track.

Git Workflow Automation
-----------------------

Streamline Git operations with ``@automation-engineer``.

.. code-block:: bash

   # Intelligent commit with analysis
   /infrastructure-suite:commit

   # Fix CI failures automatically
   /infrastructure-suite:fix-commit-errors

   # Merge all branches and clean up
   /infrastructure-suite:merge-all

Related
-------

- :doc:`/suites/infrastructure-suite` — Full infrastructure-suite reference
- :doc:`/suites/quality-suite` — Quality gates for pipelines
