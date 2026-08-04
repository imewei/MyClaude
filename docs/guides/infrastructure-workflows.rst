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

Use ``@software-architect`` for deployment-topology decisions and ``@sre-expert``
for the running substrate.

1. Define infrastructure with Terraform (hub: ``ci-cd-pipelines`` → sub: ``deployment-pipeline-design``).
2. Configure Kubernetes clusters (hub: ``architecture-and-infra`` → sub: ``containerization-patterns``).
3. Implement secrets management (hub: ``data-and-security`` → sub: ``secrets-management``).
4. Set up monitoring and alerting (hub: ``observability-and-sre`` → sub: ``prometheus-configuration``).

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

1. Define SLIs and SLOs for each critical user journey (agent: ``@sre-expert``).
2. Set up error budgets and burn-rate alerts (hub: ``observability-and-sre`` → sub: ``slo-implementation``).
3. Create incident response runbooks (agent: ``@sre-expert``).
4. Implement Prometheus alerting (hub: ``observability-and-sre`` → sub: ``prometheus-configuration``).

Production Incident Response
----------------------------

When production issues arise, use ``@sre-expert`` and ``@quality-specialist``
together for rapid resolution.

**Agent team:** Use :doc:`Team 2 (incident-response) </agent-teams-guide>` for
coordinated multi-hypothesis investigation.

Related
-------

- :doc:`/suites/dev-suite` — Full dev-suite reference (9 hubs → 35 sub-skills)
- :doc:`/glossary` — Hub Skill, Sub-Skill, and Routing Decision Tree definitions
