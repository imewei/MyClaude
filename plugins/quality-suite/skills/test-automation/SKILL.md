---
name: test-automation
version: "2.2.0"
description: Expert guide for implementing automated testing across the pyramid (Unit, Integration, E2E). Masters Jest, Pytest, Playwright, and Cypress.
---

# Test Automation

Comprehensive strategy for building reliable, maintainable automated test suites.

## 1. The Testing Pyramid

- **Unit Tests**: Focus on isolated logic. Aim for >80% coverage.
- **Integration Tests**: Verify interactions between components (e.g., API to DB).
- **E2E Tests**: Validate critical user journeys using Playwright or Cypress.

## 2. E2E Testing Patterns

### Page Object Model (POM)
Encapsulate page-specific logic and selectors to improve test maintainability.
```typescript
// Example: login.page.ts
export class LoginPage {
  constructor(private page: Page) {}
  async login(user, pass) { ... }
}
```

### API Mocking
Isolate frontend tests from backend instability by mocking network responses.

## 3. Automation Best Practices

- **Stable Selectors**: Prefer `data-testid` over CSS classes or XPath.
- **No Hard Waits**: Use assertions and signal-based waits (e.g., `waitForSelector`).
- **Parallel Execution**: Shard tests across runners to minimize CI time.
- **Flakiness Management**: Identify and fix flaky tests; avoid over-reliance on retries.

## 4. Test Automation Checklist

- [ ] Critical paths covered by E2E tests.
- [ ] Unit test coverage meets the project baseline (e.g., 80%).
- [ ] Tests are isolated and do not share state.
- [ ] CI/CD integration automatically runs tests on every PR.
