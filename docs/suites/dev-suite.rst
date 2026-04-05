Dev Suite
=========

Full-stack engineering, infrastructure, CI/CD, quality assurance, and debugging. Uses the :term:`Hub Skill` architecture with 9 hubs routing to 49 sub-skills. Merges engineering, infrastructure, and quality capabilities into a single development powerhouse.

**Version:** 3.1.0 | **9 Agents** | **12 Registered Commands** | **9 Hubs → 49 Sub-skills** | **2 Hook Events**

Agents
------

.. agent:: software-architect
   :description: Expert in designing scalable backend systems, microservices, and high-performance APIs (REST/GraphQL/gRPC).
   :model: opus
   :version: 3.1.0

.. agent:: debugger-pro
   :description: Expert in AI-assisted debugging, log correlation, and complex root cause analysis across distributed systems.
   :model: opus
   :version: 3.1.0

.. agent:: app-developer
   :description: Expert in building high-quality applications for Web, iOS, and Android. Masters React, Next.js, Flutter, and React Native.
   :model: sonnet
   :version: 3.1.0

.. agent:: automation-engineer
   :description: Expert in automating software delivery pipelines and optimizing Git collaboration workflows.
   :model: sonnet
   :version: 3.1.0

.. agent:: devops-architect
   :description: Platform Owner expert in multi-cloud architecture (AWS/Azure/GCP), Kubernetes orchestration, and Infrastructure as Code.
   :model: sonnet
   :version: 3.1.0

.. agent:: quality-specialist
   :description: Expert in ensuring software quality through rigorous code reviews, security audits, and test automation strategies.
   :model: sonnet
   :version: 3.1.0

.. agent:: sre-expert
   :description: Reliability Consultant expert in system reliability, observability, and incident response.
   :model: sonnet
   :version: 3.1.0

.. agent:: systems-engineer
   :description: Expert in low-level systems programming (C, C++, Rust, Go) and production-grade CLI tool design.
   :model: sonnet
   :version: 3.1.0

.. agent:: documentation-expert
   :description: Expert in creating clear, comprehensive, and accurate technical documentation and tutorials.
   :model: haiku
   :version: 3.1.0

Registered Commands
-------------------

.. command:: /commit
   :description: Intelligent git commit with automated analysis and quality validation.

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

.. command:: /refactor-clean
   :description: Analyze and refactor code to improve quality and maintainability.

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

Skills use a hub architecture: 9 hub skills route to 49 specialized sub-skills.

Hub: backend-patterns (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Node.js, async Python, API design, GraphQL, WebSocket, and message queue patterns.

- ``nodejs-backend-patterns`` — Express/Fastify servers, middleware chains, Node.js performance
- ``async-python-patterns`` — FastAPI, asyncio concurrency, async I/O optimization
- ``api-design-principles`` — REST resource modeling, versioning, pagination, contracts
- ``graphql-patterns`` — Schema design, resolvers, DataLoader batching, federation
- ``websocket-patterns`` — Real-time bidirectional communication, pub/sub
- ``message-queue-patterns`` — Kafka, RabbitMQ, SQS, dead-letter queues

Hub: frontend-and-mobile (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-platform UI, JavaScript/TypeScript, accessibility, and mobile testing.

- ``frontend-mobile-engineering`` — Web, iOS, Android with Flutter and React Native
- ``modern-javascript-patterns`` — ES6-ES2024 patterns, async/await, modules
- ``typescript-advanced-types`` — Generics, conditional types, mapped types
- ``typescript-project-scaffolding`` — Production-ready TypeScript project setup
- ``accessibility-testing`` — WCAG 2.1/2.2 compliance testing
- ``mobile-testing-patterns`` — Detox, Maestro, Appium for mobile apps

Hub: architecture-and-infra (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clean Architecture, DDD, microservices, monorepos, containers, and cloud.

- ``architecture-patterns`` — Clean/Hexagonal Architecture, DDD patterns
- ``microservices-patterns`` — Service boundaries, event-driven communication, Saga
- ``monorepo-management`` — Turborepo, Nx, pnpm workspaces
- ``systems-cli-engineering`` — Systems programming and CLI tool design
- ``containerization-patterns`` — Docker, Kubernetes deployments
- ``cloud-provider-patterns`` — AWS, GCP, Azure cloud-native architecture

Hub: testing-and-quality (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test automation, patterns, E2E, validation, code review, and plugin validation.

- ``test-automation`` — Test framework setup, runner configuration
- ``testing-patterns`` — Unit, integration, contract testing with mocks/fixtures
- ``e2e-testing-patterns`` — Playwright, Cypress, Selenium
- ``comprehensive-validation`` — Schema validation, data integrity, runtime assertions
- ``code-review`` — Structured review checklists, PR feedback standards
- ``plugin-syntax-validator`` — Plugin frontmatter and manifest validation

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

Hub: python-toolchain (5 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python packaging, performance, uv, error handling, and migration.

- ``python-packaging`` — pyproject.toml, build backends, PyPI publishing
- ``python-performance-optimization`` — cProfile, Cython, native extensions
- ``uv-package-manager`` — uv workspace, lockfile, virtual environments
- ``error-handling-patterns`` — Python exception hierarchies, retry, circuit breaker
- ``modernization-migration`` — Python 2→3 migration, dependency upgrades

Hub: data-and-security (6 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Databases, SQL, caching, search, authentication, and secrets.

- ``database-patterns`` — ORM patterns, migrations, connection pooling
- ``sql-optimization-patterns`` — EXPLAIN analysis, indexing, N+1 elimination
- ``caching-patterns`` — Redis, Memcached, CDN, cache invalidation
- ``search-patterns`` — Elasticsearch, OpenSearch, full-text search
- ``auth-implementation-patterns`` — JWT, OAuth2, RBAC/ABAC, session management
- ``secrets-management`` — Vault, AWS Secrets Manager, Azure Key Vault

Hub: dev-workflows (4 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Git, documentation, Airflow pipelines, and debugging.

- ``git-workflow`` — Branch strategies, commit conventions, conflict resolution
- ``documentation-standards`` — README structure, API docs, ADRs
- ``airflow-scientific-workflows`` — DAG design, task dependencies, data pipelines
- ``debugging-toolkit`` — Systematic debugging methodology, profiler-guided diagnosis

Hooks
-----

2 hook events:

- ``PostToolUse`` — Post-tool-use validation
- ``SubagentStop`` — Subagent completion handling
