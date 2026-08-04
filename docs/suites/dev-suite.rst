Dev Suite
=========

Full-stack engineering, infrastructure, CI/CD, quality assurance, and debugging. Uses the :term:`Hub Skill` architecture with 9 hubs routing to 35 sub-skills. Merges engineering, infrastructure, and quality capabilities into a single development powerhouse.

**Version:** 4.0.0 | **6 Agents** | **10 Registered Commands** | **9 Hubs → 35 Sub-skills** | **6 Hook Events**

Agents
------

.. agent:: software-architect
   :description: Expert in designing scalable backend systems, microservices, and high-performance APIs (REST/GraphQL/gRPC).
   :model: opus
   :version: 4.0.0

.. agent:: app-developer
   :description: Expert in building high-quality applications for Web, iOS, and Android. Masters React, Next.js, Flutter, and React Native.
   :model: sonnet
   :version: 4.0.0

.. agent:: automation-engineer
   :description: Expert in automating software delivery pipelines and optimizing Git collaboration workflows.
   :model: sonnet
   :version: 4.0.0

.. agent:: quality-specialist
   :description: Expert in ensuring software quality through rigorous code reviews, security audits, and test automation strategies.
   :model: sonnet
   :version: 4.0.0

.. agent:: sre-expert
   :description: Reliability Consultant expert in system reliability, observability, and incident response.
   :model: sonnet
   :version: 4.0.0

.. agent:: documentation-expert
   :description: Expert in creating clear, comprehensive, and accurate technical documentation and tutorials.
   :model: haiku
   :version: 4.0.0

Registered Commands
-------------------

.. command:: /docs
   :description: Unified documentation management — generate, update, and sync.

.. command:: /double-check
   :description: Multi-dimensional validation with automated testing and security scanning.

.. command:: /eng-feature-dev
   :description: End-to-end feature development with customizable methodologies and deployment strategies.

.. command:: /fix-commit-errors
   :description: Diagnose and fix CI/CD failures by analyzing logs, applying fixes, and rerunning workflows.

.. command:: /merge-all
   :description: Merge all local branches into main and clean up.

.. command:: /modernize
   :description: Legacy code migration using Strangler Fig pattern with incremental modernization.

.. command:: /run-all-tests
   :description: Iteratively run and fix all tests until zero failures with AI-driven RCA.

.. command:: /smart-debug
   :description: Intelligent debugging with multi-mode execution and automated RCA.

.. command:: /test-generate
   :description: Generate comprehensive test suites with scientific computing support.

.. command:: /workflow-automate
   :description: Automated CI/CD workflow generation for GitHub Actions and GitLab CI.

Hub Skills
----------

Skills use a hub architecture: 9 hub skills route to 35 specialized sub-skills.

Hub: dev-hub (top-level router)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Entry-point router for the whole suite. Dispatches to the 8 domain hubs below
(``architecture-and-infra``, ``backend-patterns``, ``ci-cd-pipelines``,
``data-and-security``, ``dev-workflows``, ``observability-and-sre``,
``testing-and-quality``, ``three-brain``).

Hub: three-brain (leaf hub)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-model routing between Claude, Codex, and Agy: a one-shot second opinion
(Route mode — code review, multimodal analysis, long-context scans) or a
persistent dev/content team that stays alive across a multi-round project
(Team mode). Also reachable from ``dev-workflows``.

Hub: backend-patterns (3 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

API design, error handling, and message queue patterns.

- ``api-design-principles`` — REST resource modeling, versioning, pagination, contracts
- ``error-handling-patterns`` — Exception hierarchies, retry, circuit breaker
- ``message-queue-patterns`` — Kafka, RabbitMQ, SQS, dead-letter queues

Hub: architecture-and-infra (7 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clean Architecture, DDD, microservices, monorepos, containers, cloud, and migration.

- ``architecture-patterns`` — Clean/Hexagonal Architecture, DDD patterns
- ``microservices-patterns`` — Service boundaries, event-driven communication, Saga
- ``monorepo-management`` — Turborepo, Nx, pnpm workspaces
- ``systems-cli-engineering`` — Systems programming and CLI tool design
- ``containerization-patterns`` — Docker, Kubernetes deployments
- ``cloud-provider-patterns`` — AWS, GCP, Azure cloud-native architecture
- ``modernization-migration`` — Legacy migration and dependency upgrades

Hub: testing-and-quality (5 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test automation, patterns, E2E, validation, and code review.

- ``test-automation`` — Test framework setup, runner configuration
- ``testing-patterns`` — Unit, integration, contract testing with mocks/fixtures
- ``e2e-testing-patterns`` — Playwright, Cypress, Selenium
- ``comprehensive-validation`` — Schema validation, data integrity, runtime assertions
- ``code-review`` — Structured review checklists, PR feedback standards

Hub: ci-cd-pipelines (5 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GitHub Actions, GitLab CI, deployment strategies, security scanning, error resolution.

- ``github-actions-templates`` — Reusable workflows, composite actions, matrix builds
- ``gitlab-ci-patterns`` — DAG pipelines, runners, artifact management
- ``deployment-pipeline-design`` — Blue/green, canary, rolling deployments
- ``security-ci-template`` — SAST, dependency scanning, SBOM generation
- ``iterative-error-resolution`` — CI/CD pipeline failure diagnosis

Hub: observability-and-sre (5 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Monitoring, alerting, tracing, dashboards, and SLO implementation.

- ``observability-sre-practices`` — OpenTelemetry, Prometheus, Grafana, incident management
- ``prometheus-configuration`` — Scrape configs, recording rules, alert rules
- ``grafana-dashboards`` — Panels, variables, alerts, RED/USE methods
- ``distributed-tracing`` — OpenTelemetry, Jaeger, Tempo, context propagation
- ``slo-implementation`` — SLI/SLO definitions, error budgets, burn rate alerting

Hub: data-and-security (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Databases, SQL, caching, search, authentication, and secrets.

- ``database-patterns`` — ORM patterns, migrations, connection pooling
- ``sql-optimization-patterns`` — EXPLAIN analysis, indexing, N+1 elimination
- ``caching-patterns`` — Redis, Memcached, CDN, cache invalidation
- ``search-patterns`` — Elasticsearch, OpenSearch, full-text search
- ``auth-implementation-patterns`` — JWT, OAuth2, RBAC/ABAC, session management
- ``secrets-management`` — Vault, AWS Secrets Manager, Azure Key Vault

Hub: dev-workflows (4 sub-skills + 1 hub)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Git, documentation, Airflow pipelines, and debugging. Also routes to the
``three-brain`` hub.

- ``git-workflow`` — Branch strategies, commit conventions, conflict resolution
- ``documentation-standards`` — README structure, API docs, ADRs
- ``airflow-scientific-workflows`` — DAG design, task dependencies, data pipelines
- ``debugging-toolkit`` — Systematic debugging methodology, profiler-guided diagnosis

Hooks
-----

7 hook events:

- ``SessionStart`` — Auto-detect project stack (language, framework, test runner)
- ``UserPromptSubmit`` — Remind agent to route through the matching hub skill before implementing
- ``PostToolUse`` — Auto-lint after Write/Edit (ruff for Python, eslint for JS/TS)
- ``SubagentStop`` — Collect subagent results for orchestrated workflows
- ``TaskCompleted`` — Trigger validation checks on task completion
- ``SessionEnd`` — Persist structured progress summary for next session
- ``StopFailure`` — Capture context when /stop fails mid-operation

(``ExecutionError`` was removed in v3.4.0 — not supported by the CC v2.1.113 CLI event schema.)
