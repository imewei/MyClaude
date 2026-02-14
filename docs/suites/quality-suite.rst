Quality & Maintenance Suite
===========================

Consolidated suite for code quality, test automation, legacy modernization, and debugging. Enhanced with adaptive reasoning for Claude Opus 4.6.

Agents
------

.. agent:: debugger-pro
   :description: Expert in AI-assisted debugging, log correlation, and complex root cause analysis across distributed systems.
   :model: sonnet
   :version: 2.2.0

.. agent:: documentation-expert
   :description: Expert in creating clear, comprehensive, and accurate technical documentation and tutorials.
   :model: sonnet
   :version: 2.2.0

.. agent:: quality-specialist
   :description: Expert in ensuring software quality through rigorous code reviews, security audits, and test automation.
   :model: sonnet
   :version: 2.2.0

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

.. skill:: ai-assisted-debugging
   :description: AI/LLM for automated stack trace analysis, intelligent RCA, ML log correlation in distributed systems. Use for Python/JS/Go exceptions, K8s pod failures, automated anomaly detection on logs/metrics, correlating git commits with production incidents.
   :version: 2.2.0

.. skill:: code-review
   :description: Systematic process for code review focused on security, performance, maintainability, and knowledge sharing.
   :version: 2.2.0

.. skill:: comprehensive-validation
   :description: Multi-dimensional validation framework for code, APIs, and systems. Covers security scans, performance profiling, and production readiness checks.
   :version: 2.2.0

.. skill:: comprehensive-validation-framework
   :description: Systematic multi-dimensional validation framework for code, APIs, and systems. Use when validating before deployment, running security scans (OWASP Top 10, dependency vulnerabilities), checking test coverage (>80% target), verifying accessibility (WCAG 2.1 AA), profiling performance, validating breaking changes, or preparing deployment readiness reports.
   :version: 2.2.0

.. skill:: debugging-strategies
   :description: Systematic debugging with scientific method, profiling, RCA across any stack. Use for runtime errors, performance issues, memory leaks, flaky bugs, production debugging with Chrome DevTools, VS Code, pdb/ipdb, Delve, git bisect.
   :version: 2.2.0

.. skill:: documentation-standards
   :description: Guidelines for high-quality technical documentation, including API specs, READMEs, and internal runbooks.
   :version: 2.2.0

.. skill:: e2e-testing-patterns
   :description: Build reliable E2E tests with Playwright and Cypress for web testing, browser automation, and CI/CD integration. Use when writing E2E tests, implementing Page Object Model, mocking APIs, visual regression, or accessibility testing.
   :version: 2.2.0

.. skill:: observability-sre-practices
   :description: Production observability, monitoring, SRE with OpenTelemetry, Prometheus, Grafana, incident management. Use for tracing/metrics/logs, SLOs/SLIs, alerts with AlertManager, Golden Signals, incident response, post-mortems, error budgets.
   :version: 2.2.0

.. skill:: plugin-syntax-validator
   :description: Validates plugin structure, manifest correctness, and component syntax against official standards. Checks for required files (plugin.json, README.md, LICENSE), validates YAML frontmatter in agents/commands/skills, and ensures directory compliance. Use PROACTIVELY when creating or modifying plugins.
   :version: 2.2.0

.. skill:: test-automation
   :description: Expert guide for implementing automated testing across the pyramid (Unit, Integration, E2E). Masters Jest, Pytest, Playwright, and Cypress.
   :version: 2.2.0

