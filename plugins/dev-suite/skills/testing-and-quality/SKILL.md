---
name: testing-and-quality
description: Meta-orchestrator for testing and code quality. Routes to test automation, testing patterns, E2E, validation, code review, and plugin validation skills. Use when writing tests, setting up test automation, implementing E2E testing, analyzing test coverage, designing test pyramids, validating code quality, conducting code reviews, or checking plugin structure.
---

# Testing and Quality

Orchestrator for testing strategy and code quality across the full development lifecycle. Routes to the appropriate specialized skill based on the test scope, validation type, or review need.

## Expert Agent

- **`quality-specialist`**: Specialist for test architecture, quality gates, and systematic validation.
  - *Location*: `plugins/dev-suite/agents/quality-specialist.md`
  - *Capabilities*: Test pyramid design, coverage analysis, E2E strategy, code review standards, and plugin integrity validation.

## Core Skills

### [Test Automation](../test-automation/SKILL.md)
Test framework setup, test runner configuration, and automation patterns. Focuses on test infrastructure — for CI/CD pipeline YAML and workflow files, see `ci-cd-pipelines`.

### [Testing Patterns](../testing-patterns/SKILL.md)
Unit, integration, and contract testing patterns with mocking and fixture strategies.

### [E2E Testing Patterns](../e2e-testing-patterns/SKILL.md)
Playwright, Cypress, and Selenium for end-to-end browser and API testing.

### [Comprehensive Validation](../comprehensive-validation/SKILL.md)
Schema validation, data integrity checks, and runtime assertion strategies.

### [Code Review](../code-review/SKILL.md)
Structured review checklists, PR feedback standards, and automated linting gates.

### [Plugin Syntax Validator](../plugin-syntax-validator/SKILL.md)
Frontmatter parsing, manifest cross-reference checks, and plugin integrity validation.

## Routing Decision Tree

```
What is the quality concern?
|
+-- CI pipeline / test runner / automation setup?
|   --> test-automation
|
+-- Unit / integration / contract test design?
|   --> testing-patterns
|
+-- Browser / full-stack / API end-to-end tests?
|   --> e2e-testing-patterns
|
+-- Schema / data / runtime validation?
|   --> comprehensive-validation
|
+-- PR review / linting / feedback standards?
|   --> code-review
|
+-- Plugin frontmatter / manifest integrity?
    --> plugin-syntax-validator
```

## Routing Table

| Trigger                              | Sub-skill                    |
|--------------------------------------|------------------------------|
| Test framework setup, jest, pytest   | test-automation              |
| Mocks, fixtures, contracts, spies    | testing-patterns             |
| Playwright, Cypress, E2E, browser    | e2e-testing-patterns         |
| Pydantic, zod, JSON schema, asserts  | comprehensive-validation     |
| PR review, lint, code standards      | code-review                  |
| Plugin YAML, manifest, frontmatter   | plugin-syntax-validator      |

## Checklist

- [ ] Identify the test scope (unit / integration / E2E) before selecting a sub-skill
- [ ] Confirm CI automation runs tests on every pull request
- [ ] Verify coverage thresholds are enforced and not just reported
- [ ] Check that E2E tests target realistic user flows, not implementation details
- [ ] Validate plugin manifests pass syntax checks before merge
- [ ] Ensure code review standards are documented and applied consistently
