Infrastructure Workflows
========================

Patterns for provisioning cloud infrastructure and managing platform
operations with the **dev-suite** :term:`hub skills <Hub Skill>`.

.. note::

   Since v3.1.0, skills use a two-tier :term:`Hub Skill` architecture. The hub
   skills listed below route to specialized sub-skills via their
   :term:`Routing Decision Tree`.

Cloud Infrastructure
--------------------

Use ``@devops-architect`` for multi-cloud architecture decisions.

1. Define infrastructure with Terraform (hub: ``ci-cd-pipelines`` → sub: ``deployment-pipeline-design``).
2. Configure Kubernetes clusters (agent: ``@devops-architect``).
3. Implement secrets management (hub: ``data-and-security`` → sub: ``secrets-management``).
4. Set up monitoring and alerting (command: ``/monitor-setup``).

.. code-block:: hcl

   # Example: Terraform resource pattern
   resource "aws_ecs_service" "app" {
     name            = "my-service"
     cluster         = aws_ecs_cluster.main.id
     task_definition = aws_ecs_task_definition.app.arn
     desired_count   = 3

     load_balancer {
       target_group_arn = aws_lb_target_group.app.arn
       container_name   = "app"
       container_port   = 8080
     }
   }

Reliability Engineering
-----------------------

Use ``@sre-expert`` for SLO-driven reliability.

1. Define SLIs and SLOs (command: ``/slo-implement``).
2. Set up error budgets and burn-rate alerts (hub: ``observability-and-sre`` → sub: ``slo-implementation``).
3. Create incident response runbooks (agent: ``@sre-expert``).
4. Implement Prometheus alerting (hub: ``observability-and-sre`` → sub: ``prometheus-configuration``).

Production Incident Response
----------------------------

When production issues arise, use ``@debugger-pro`` and ``@sre-expert``
together for rapid resolution.

**Agent team:** Use :doc:`Team 2 (incident-response) </agent-teams-guide>` for
coordinated multi-hypothesis investigation.

Related
-------

- :doc:`/suites/dev-suite` — Full dev-suite reference (9 hubs → 49 sub-skills)
- :doc:`/glossary` — Hub Skill, Sub-Skill, and Routing Decision Tree definitions
