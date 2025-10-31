# Quality Engineering

Comprehensive quality assurance, validation, and correctness verification tools with automated scripts for security scanning, testing, linting, performance profiling, accessibility checks, and plugin syntax validation.

**Version:** 1.0.1 | **Category:** quality-assurance | **License:** MIT

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/quality-engineering.html)

## What's New in v1.0.1

### Skills Discoverability Enhancement
Both quality-engineering skills now have comprehensive "When to use this skill" sections with 21 specific use cases each:

- **Enhanced Descriptions**: Detailed coverage of automated scripts, reference guides, validation dimensions, and specific tools/frameworks
- **Automatic Discovery**: +50-75% improvement in Claude Code recognizing when to use skills during quality assurance work
- **Context Relevance**: Skills activate automatically during validation, testing, security checking, and plugin development
- **42 Use Cases**: Specific scenarios across validation frameworks, security scanning, performance profiling, accessibility checks, and plugin syntax validation

**Key Improvements:**
- comprehensive-validation-framework: Detailed all 7 automated scripts, 6 reference guides, and 10 validation dimensions
- plugin-syntax-validator: Explicit script names, file patterns, error types, and auto-fix capabilities

## Commands (2)

### `/double-check`

**Status:** active

Comprehensive multi-dimensional validation with automated testing, security scanning, and quality assurance across 10 critical dimensions: scope/requirements, functional correctness, code quality, security (OWASP Top 10, dependency vulnerabilities, SAST, secret detection), performance (N+1 queries, caching, profiling), accessibility (WCAG 2.1 AA, keyboard navigation, screen readers), testing coverage (>80% target), breaking changes, operations readiness (logging, metrics, health checks), and documentation completeness.

**Key Features:**
- Automated scripts: run_all_validations.py, security_scan.py, test_runner.py, lint_check.py, performance_profiler.py, accessibility_check.py, build_verify.py
- Reference guides: Security deep-dive, performance optimization, testing best practices, production readiness, accessibility standards, breaking changes guide
- Structured validation reports with severity classifications and actionable recommendations
- CI/CD integration support with GitHub Actions and pre-commit hooks

### `/lint-plugins`

**Status:** active

Validate Claude Code plugin syntax, structure, and cross-references with auto-fix capabilities. Checks plugin command files for correct agent/skill namespace syntax (plugin:agent single colon format), verifies agent file existence, detects syntax errors (double colons, missing namespaces), and generates comprehensive validation reports with file:line error locations.

**Key Features:**
- Auto-fix double colon errors with --fix flag
- Validates 200+ agent references in ~30 seconds
- Detects SYNTAX errors (double colons) and REFERENCE errors (missing agents)
- Generates reports with statistics and actionable suggestions
- CI/CD integration with GitHub Actions and pre-commit hooks
- Builds complete agent/skill maps across plugin ecosystems

## Skills (2)

### comprehensive-validation-framework

Systematic multi-dimensional validation framework for code, APIs, and systems with automated scripts (run_all_validations.py, security_scan.py, test_runner.py, lint_check.py, performance_profiler.py, accessibility_check.py, build_verify.py) and deep-dive reference guides for security (OWASP Top 10, OAuth 2.0, JWT, input validation, cryptography), performance optimization (profiling tools, N+1 queries, caching strategies, database optimization), testing best practices (testing pyramid, AAA pattern, mocking, property-based testing), production readiness (health checks, structured logging, metrics, circuit breakers, deployment strategies), accessibility standards (WCAG 2.1 principles, ARIA patterns, semantic HTML), and breaking changes (SemVer, deprecation, API versioning, database migrations) across 10 critical validation dimensions.

**When to use:** Running comprehensive validation before production deployment, executing automated validation scripts for quality/security checks, performing security validation for authentication/authorization code, validating code quality with linters, checking test coverage >80%, profiling performance bottlenecks, verifying WCAG 2.1 AA accessibility compliance, consulting reference guides for expert guidance, preparing validation reports, setting up CI/CD pipelines, and 11+ more scenarios.

**Key capabilities:**
- 7 automated validation scripts covering linting, testing, security, performance, accessibility, and builds
- 6 comprehensive reference guides (1,000+ lines each) for expert guidance
- 10 validation dimensions with structured checklists
- Validation report templates with severity classifications
- Integration with popular tools: ESLint, Prettier, Ruff, Black, pytest, Jest, Semgrep, Bandit, pa11y, axe-core

### plugin-syntax-validator

Comprehensive validation framework for Claude Code plugin syntax, structure, and cross-references with automated scripts/validate_plugin_syntax.py script for detecting and auto-fixing common errors (double colons, missing namespaces, invalid references). Validates plugin command files (*.md) for correct agent/skill namespace syntax (plugin:agent single colon format), checks agent file existence in plugins/*/agents/*.md, generates comprehensive validation reports with file:line error locations and actionable suggestions, supports CI/CD integration with GitHub Actions and pre-commit hooks, and maintains plugin quality standards before deployment or marketplace submission.

**When to use:** Validating plugin command files before commits/PRs, running validation scripts to check syntax across all plugins, auto-fixing double colon errors with --fix flag, checking subagent_type namespace format, verifying agent file existence, detecting SYNTAX/REFERENCE errors, preparing plugins for distribution, setting up CI/CD validation, creating pre-commit hooks, generating validation reports, building agent/skill maps, and 10+ more scenarios.

**Key capabilities:**
- Auto-fix double colon syntax errors automatically
- Validates 200+ agent references in ~30 seconds
- Detects SYNTAX errors (plugin::agent) and REFERENCE errors (missing agents)
- Provides file:line error locations with actionable suggestions
- Builds complete agent/skill maps across plugin ecosystems
- CI/CD integration with GitHub Actions and pre-commit hooks

## Quick Start

To use this plugin:

1. Ensure Claude Code is installed
2. Enable the `quality-engineering` plugin
4. Try a command (e.g., `/double-check`)

## Integration

See the full documentation for integration patterns and compatible plugins.

## Documentation

For comprehensive documentation, see: [Plugin Documentation](https://myclaude.readthedocs.io/en/latest/plugins/quality-engineering.html)

To build documentation locally:

```bash
cd docs/
make html
```
