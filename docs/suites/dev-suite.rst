Dev Suite
=========

Consolidated suite for full-stack engineering, infrastructure, CI/CD, quality assurance, and debugging. Merges engineering, infrastructure, and quality capabilities into a single development powerhouse optimized for Claude Opus 4.6.

Agents
------

.. agent:: app-developer
   :description: Expert in building high-quality applications for Web, iOS, and Android. Masters React, Next.js, Flutter, and React Native.
   :model: sonnet
   :version: 3.0.0

.. agent:: automation-engineer
   :description: Expert in automating software delivery pipelines and optimizing Git collaboration workflows. Masters GitHub Actions, GitLab CI, and advanced Git history management.
   :model: sonnet
   :version: 3.0.0

.. agent:: debugger-pro
   :description: Expert in AI-assisted debugging, log correlation, and complex root cause analysis across distributed systems.
   :model: opus
   :version: 3.0.0

.. agent:: devops-architect
   :description: Platform Owner expert in multi-cloud architecture (AWS/Azure/GCP), Kubernetes orchestration, and Infrastructure as Code (Terraform/Pulumi).
   :model: sonnet
   :version: 3.0.0

.. agent:: documentation-expert
   :description: Expert in creating clear, comprehensive, and accurate technical documentation, manuals, and tutorials.
   :model: haiku
   :version: 3.0.0

.. agent:: quality-specialist
   :description: Expert in ensuring software quality through rigorous code reviews, comprehensive security audits, and robust test automation strategies.
   :model: sonnet
   :version: 3.0.0

.. agent:: software-architect
   :description: Expert in designing scalable backend systems, microservices, and high-performance APIs (REST/GraphQL/gRPC).
   :model: opus
   :version: 3.0.0

.. agent:: sre-expert
   :description: Reliability Consultant expert in system reliability, observability (monitoring, logging, tracing), and incident response.
   :model: sonnet
   :version: 3.0.0

.. agent:: systems-engineer
   :description: Expert in low-level systems programming (C, C++, Rust, Go) and production-grade CLI tool design.
   :model: sonnet
   :version: 3.0.0

Commands
--------

.. command:: /adopt-code
   :description: Analyze and modernize scientific computing codebases with accuracy.

.. command:: /c-project
   :description: Scaffold production-ready C projects with Makefile/CMake and memory safety tools.

.. command:: /code-analyze
   :description: Semantic code analysis using Serena MCP for symbol navigation.

.. command:: /code-explain
   :description: Detailed code explanation with visual aids and domain expertise.

.. command:: /commit
   :description: Intelligent git commit with automated analysis and quality validation.

.. command:: /deps
   :description: Unified dependency management - security auditing and safe upgrades.

.. command:: /docs
   :description: Unified documentation management - generate, update, and sync.

.. command:: /double-check
   :description: Multi-dimensional validation with automated testing and security scanning.

.. command:: /eng-feature-dev
   :description: Unified end-to-end feature development with customizable methodologies and deployment strategies.

.. command:: /fix-commit-errors
   :description: Automatically analyzes GitHub Actions failures and applies solutions.

.. command:: /fix-imports
   :description: Systematically fix broken imports across the codebase.

.. command:: /github-assist
   :description: GitHub operations using GitHub MCP for issues, PRs, and repos.

.. command:: /merge-all
   :description: Merge all local branches into main and clean up.

.. command:: /modernize
   :description: Unified code migration and legacy modernization with Strangler Fig pattern.

.. command:: /monitor-setup
   :description: Set up Prometheus, Grafana, and distributed tracing stack.

.. command:: /multi-platform
   :description: Build and deploy features across web, mobile, and desktop platforms.

.. command:: /onboard
   :description: Orchestrate complete onboarding for new team members.

.. command:: /profile-performance
   :description: Comprehensive performance profiling with perf, flamegraph, and valgrind.

.. command:: /refactor-clean
   :description: Analyze and refactor code to improve quality and maintainability.

.. command:: /run-all-tests
   :description: Iteratively run and fix all tests until zero failures with AI-driven RCA.

.. command:: /rust-project
   :description: Scaffold production-ready Rust projects with cargo tooling and idiomatic patterns.

.. command:: /scaffold
   :description: Unified project and component scaffolding for TypeScript, Python, React, and Julia.

.. command:: /slo-implement
   :description: Implement SLO/SLA monitoring, error budgets, and alerting.

.. command:: /smart-debug
   :description: Intelligent debugging with multi-mode execution and automated RCA.

.. command:: /tech-debt
   :description: Analyze, prioritize, and remediate technical debt using ROI metrics.

.. command:: /test-generate
   :description: Generate comprehensive test suites with scientific computing support.

.. command:: /workflow-automate
   :description: Automated CI/CD workflow generation for GitHub Actions and GitLab CI.

Skills
------

.. skill:: airflow-scientific-workflows
   :description: Design Apache Airflow DAGs for scientific data pipelines, batch computations, and distributed simulations.
   :version: 3.0.0

.. skill:: api-design-principles
   :description: Master REST and GraphQL API design including resource-oriented architecture, HTTP semantics, pagination, versioning, and documentation.
   :version: 3.0.0

.. skill:: architecture-patterns
   :description: Master Clean Architecture, Hexagonal Architecture, and DDD patterns including entities, value objects, aggregates, and repositories.
   :version: 3.0.0

.. skill:: async-python-patterns
   :description: Master Python asyncio, concurrent programming, and async/await patterns for high-performance non-blocking applications.
   :version: 3.0.0

.. skill:: auth-implementation-patterns
   :description: Master authentication patterns including JWT, OAuth2/OpenID Connect, RBAC/ABAC, and secure cookie management.
   :version: 3.0.0

.. skill:: code-review
   :description: Systematic process for code review focused on security, performance, maintainability, and knowledge sharing.
   :version: 3.0.0

.. skill:: comprehensive-validation
   :description: Multi-dimensional validation framework for code, APIs, and systems. Covers security scans, performance profiling, and production readiness checks.
   :version: 3.0.0

.. skill:: debugging-toolkit
   :description: Unified AI-assisted and systematic debugging toolkit covering automated stack trace analysis, intelligent RCA, ML log correlation, and distributed systems diagnostics.
   :version: 3.0.0

.. skill:: deployment-pipeline-design
   :description: Design multi-stage CI/CD pipelines with approval gates, security checks, and progressive delivery.
   :version: 3.0.0

.. skill:: distributed-tracing
   :description: Implement distributed tracing with OpenTelemetry, Jaeger, and Tempo including instrumentation, context propagation, and sampling strategies.
   :version: 3.0.0

.. skill:: documentation-standards
   :description: Guidelines for high-quality technical documentation, including API specs, READMEs, and internal runbooks.
   :version: 3.0.0

.. skill:: e2e-testing-patterns
   :description: Build reliable E2E tests with Playwright and Cypress for web testing, browser automation, and CI/CD integration.
   :version: 3.0.0

.. skill:: error-handling-patterns
   :description: Master error handling patterns including exception hierarchies, Result types, retries with backoff, and circuit breakers.
   :version: 3.0.0

.. skill:: frontend-mobile-engineering
   :description: Design and build multi-platform applications for web, iOS, and Android. Covers Flutter, React Native, and mobile-first UX.
   :version: 3.0.0

.. skill:: git-workflow
   :description: Master advanced Git workflows for collaborative development. Covers interactive rebasing, cherry-picking, and bisecting.
   :version: 3.0.0

.. skill:: github-actions-templates
   :description: Create production GitHub Actions workflows for testing, building, and deploying.
   :version: 3.0.0

.. skill:: gitlab-ci-patterns
   :description: Build GitLab CI/CD pipelines with multi-stage workflows, caching, Docker builds, and Kubernetes deployments.
   :version: 3.0.0

.. skill:: grafana-dashboards
   :description: Create production Grafana dashboards with panels, variables, alerts, and templates using RED/USE methods.
   :version: 3.0.0

.. skill:: iterative-error-resolution
   :description: Iterative CI/CD error resolution with pattern recognition, automated fixes, and learning from outcomes.
   :version: 3.0.0

.. skill:: microservices-patterns
   :description: Design microservices with proper boundaries, event-driven communication, Saga pattern, and resilience patterns.
   :version: 3.0.0

.. skill:: modern-javascript-patterns
   :description: Master modern JavaScript (ES6+) features including async/await, destructuring, spread operators, and functional programming patterns.
   :version: 3.0.0

.. skill:: modernization-migration
   :description: Strategy and patterns for legacy modernization, framework migrations, and database schema evolution.
   :version: 3.0.0

.. skill:: monorepo-management
   :description: Master monorepo management with Turborepo, Nx, and pnpm workspaces.
   :version: 3.0.0

.. skill:: nodejs-backend-patterns
   :description: Build scalable Node.js backends with Express/Fastify/NestJS, implementing middleware, authentication, and database integration.
   :version: 3.0.0

.. skill:: observability-sre-practices
   :description: Production observability, monitoring, SRE with OpenTelemetry, Prometheus, Grafana, and incident management.
   :version: 3.0.0

.. skill:: plugin-syntax-validator
   :description: Validates plugin structure, manifest correctness, and component syntax against official standards.
   :version: 3.0.0

.. skill:: prometheus-configuration
   :description: Configure Prometheus for metric collection, alerting, and monitoring with scrape configs and recording rules.
   :version: 3.0.0

.. skill:: python-packaging
   :description: Create distributable Python packages with pyproject.toml, proper project structure, and publishing to PyPI.
   :version: 3.0.0

.. skill:: python-performance-optimization
   :description: Profile and optimize Python code using cProfile, line_profiler, and memory_profiler.
   :version: 3.0.0

.. skill:: secrets-management
   :description: Implement secrets management with HashiCorp Vault, AWS Secrets Manager, and Azure Key Vault.
   :version: 3.0.0

.. skill:: security-ci-template
   :description: Security scanning and lock file validation templates for CI/CD pipelines.
   :version: 3.0.0

.. skill:: slo-implementation
   :description: Define SLIs, SLOs, error budgets, and burn rate alerting following SRE best practices.
   :version: 3.0.0

.. skill:: sql-optimization-patterns
   :description: Master SQL optimization with EXPLAIN analysis, indexing strategies, N+1 elimination, and cursor-based pagination.
   :version: 3.0.0

.. skill:: systems-cli-engineering
   :description: Design high-performance systems and production-grade CLI tools. Covers memory management, concurrency, and CLI UX design.
   :version: 3.0.0

.. skill:: test-automation
   :description: Expert guide for implementing automated testing across the pyramid (Unit, Integration, E2E).
   :version: 3.0.0

.. skill:: testing-patterns
   :description: Multi-language testing patterns for Python and JavaScript/TypeScript.
   :version: 3.0.0

.. skill:: typescript-advanced-types
   :description: Master TypeScript's advanced type system including generics, conditional types, mapped types, and discriminated unions.
   :version: 3.0.0

.. skill:: typescript-project-scaffolding
   :description: Set up production-ready TypeScript projects with modern tooling like Vite, ESLint, and Vitest.
   :version: 3.0.0

.. skill:: uv-package-manager
   :description: Master uv for blazing-fast Python dependency management, virtual environments, and reproducible builds.
   :version: 3.0.0
