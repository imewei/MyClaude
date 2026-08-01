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

## Routing Decision Tree

```
What is the quality concern?
|
+-- CI pipeline / test runner / automation setup?
|   --> dev-suite:test-automation
|
+-- Unit / integration / contract test design?
|   --> dev-suite:testing-patterns
|
+-- Browser / full-stack / API end-to-end tests?
|   --> dev-suite:e2e-testing-patterns
|
+-- Schema / data / runtime validation?
|   --> dev-suite:comprehensive-validation
|
+-- PR review / linting / feedback standards?
|   --> dev-suite:code-review
|
+-- None of the above / concern is ambiguous or spans multiple areas?
    --> Delegate to quality-specialist for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Routing Table

| Trigger                              | Sub-skill                    |
|--------------------------------------|------------------------------|
| Test framework setup, jest, pytest   | dev-suite:test-automation              |
| Mocks, fixtures, contracts, spies    | dev-suite:testing-patterns             |
| Playwright, Cypress, E2E, browser    | dev-suite:e2e-testing-patterns         |
| Pydantic, zod, JSON schema, asserts  | dev-suite:comprehensive-validation     |
| PR review, lint, code standards      | dev-suite:code-review                  |

## Checklist

- [ ] Identify the test scope (unit / integration / E2E) before selecting a sub-skill
- [ ] Confirm CI automation runs tests on every pull request
- [ ] Verify coverage thresholds are enforced and not just reported
- [ ] Check that E2E tests target realistic user flows, not implementation details
- [ ] Validate plugin manifests pass syntax checks before merge
- [ ] Ensure code review standards are documented and applied consistently
