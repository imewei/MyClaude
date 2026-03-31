Quality & Maintenance Suite
===========================

Consolidated suite for code quality, test automation, legacy modernization, and debugging. Enhanced with adaptive reasoning for Claude Opus 4.6.

Agents
------

.. agent:: debugger-pro
   :description: Expert in AI-assisted debugging, log correlation, and complex root cause analysis across distributed systems.
   :model: opus
   :version: 2.2.1

.. agent:: documentation-expert
   :description: Expert in creating clear, comprehensive, and accurate technical documentation and tutorials.
   :model: haiku
   :version: 2.2.1

.. agent:: quality-specialist
   :description: Expert in ensuring software quality through rigorous code reviews, security audits, and test automation.
   :model: sonnet
   :version: 2.2.1

Commands
--------

.. command:: /adopt-code
   :description: Analyze and modernize scientific computing codebases with accuracy.

.. command:: /code-explain
   :description: Detailed code explanation with visual aids and domain expertise.

.. command:: /deps
   :description: Unified dependency management - security auditing and safe upgrades.

.. command:: /docs
   :description: Unified documentation management - generate, update, and sync.

.. command:: /double-check
   :description: Multi-dimensional validation with automated testing and security scanning.

.. command:: /fix-imports
   :description: Systematically fix broken imports across the codebase.

.. command:: /refactor-clean
   :description: Analyze and refactor code to improve quality and maintainability.

.. command:: /run-all-tests
   :description: Iteratively run and fix all tests until zero failures with AI-driven RCA.

.. command:: /smart-debug
   :description: Intelligent debugging with multi-mode execution and automated RCA.

.. command:: /tech-debt
   :description: Analyze, prioritize, and remediate technical debt using ROI metrics.

.. command:: /test-generate
   :description: Generate comprehensive test suites with scientific computing support.

Skills
------

.. skill:: debugging-toolkit
   :description: Unified AI-assisted and systematic debugging toolkit covering automated stack trace analysis, intelligent RCA, ML log correlation, scientific method-based profiling, and distributed systems diagnostics.
   :version: 2.2.1

.. skill:: code-review
   :description: Systematic process for code review focused on security, performance, maintainability, and knowledge sharing.
   :version: 2.2.1

.. skill:: comprehensive-validation
   :description: Multi-dimensional validation framework for code, APIs, and systems. Covers security scans, performance profiling, and production readiness checks.
   :version: 2.2.1

.. skill:: documentation-standards
   :description: Guidelines for high-quality technical documentation, including API specs, READMEs, and internal runbooks.
   :version: 2.2.1

.. skill:: e2e-testing-patterns
   :description: Build reliable E2E tests with Playwright and Cypress for web testing, browser automation, and CI/CD integration. Use when writing E2E tests, implementing Page Object Model, mocking APIs, visual regression, or accessibility testing.
   :version: 2.2.1

.. skill:: observability-sre-practices
   :description: Production observability, monitoring, SRE with OpenTelemetry, Prometheus, Grafana, incident management. Use for tracing/metrics/logs, SLOs/SLIs, alerts with AlertManager, Golden Signals, incident response, post-mortems, error budgets.
   :version: 2.2.1

.. skill:: plugin-syntax-validator
   :description: Validates plugin structure, manifest correctness, and component syntax against official standards. Checks for required files (plugin.json, README.md, LICENSE), validates YAML frontmatter in agents/commands/skills, and ensures directory compliance. Use PROACTIVELY when creating or modifying plugins.
   :version: 2.2.1

.. skill:: test-automation
   :description: Expert guide for implementing automated testing across the pyramid (Unit, Integration, E2E). Masters Jest, Pytest, Playwright, and Cypress.
   :version: 2.2.1

