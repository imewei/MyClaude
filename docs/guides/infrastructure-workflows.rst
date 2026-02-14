Infrastructure Workflows
========================

Patterns for provisioning cloud infrastructure and managing platform
operations with the **infrastructure-suite**.

Cloud Infrastructure
--------------------

Use ``@devops-architect`` for multi-cloud architecture decisions.

1. Define infrastructure with Terraform (skill: ``deployment-pipeline-design``).
2. Configure Kubernetes clusters (agent: ``@devops-architect``).
3. Implement secrets management (skill: ``secrets-management``).
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
2. Set up error budgets and burn-rate alerts (skill: ``slo-implementation``).
3. Create incident response runbooks (agent: ``@sre-expert``).
4. Implement Prometheus alerting (skill: ``prometheus-configuration``).

Production Incident Response
----------------------------

When production issues arise, use ``@debugger-pro`` and ``@sre-expert``
together for rapid resolution.

**Agent team:** Use :doc:`Team 2 (incident-response) </agent-teams-guide>` for
coordinated multi-hypothesis investigation.

Related
-------

- :doc:`/suites/infrastructure-suite` — Full infrastructure-suite reference
- :doc:`/suites/engineering-suite` — Application-level architecture
