---
name: e2e-testing-patterns
description: Build reliable end-to-end tests with Playwright and Cypress for web application testing, browser automation, and CI/CD integration. Use when writing or editing E2E test files (*.spec.ts, *.spec.js, *.test.ts, *.cy.ts, *.cy.js, e2e/*.ts). Use when configuring Playwright (playwright.config.ts) or Cypress (cypress.config.ts, cypress.config.js). Use when implementing Page Object Model patterns for test organization. Use when testing user workflows like login, signup, checkout, form submission, or navigation. Use when writing browser automation scripts or web scraping with Playwright. Use when mocking API responses, intercepting network requests, or stubbing endpoints in tests. Use when implementing visual regression testing with screenshot comparison. Use when running accessibility tests with axe-core or similar tools. Use when debugging flaky tests, fixing test reliability issues, or investigating test failures. Use when setting up E2E tests in CI/CD pipelines (GitHub Actions, GitLab CI, CircleCI). Use when testing across multiple browsers (Chrome, Firefox, Safari, WebKit, Edge). Use when implementing parallel test execution or test sharding. Use when testing responsive designs across different viewport sizes. Use when writing custom Playwright or Cypress commands and utilities.
---

# End-to-End Testing Patterns

Master end-to-end testing with Playwright and Cypress for reliable, maintainable test automation.

## When to use this skill

- Writing or editing E2E test files (*.spec.ts, *.spec.js, *.cy.ts, *.cy.js)
- Configuring Playwright (playwright.config.ts) or Cypress (cypress.config.ts)
- Implementing Page Object Model (POM) patterns for test organization
- Testing user workflows (login, signup, checkout, form submission, navigation)
- Writing browser automation scripts or web scraping with Playwright
- Mocking API responses or intercepting network requests in tests
- Stubbing endpoints, fixtures, or test data for isolated testing
- Implementing visual regression testing with screenshot comparison
- Running accessibility tests with axe-core, Lighthouse, or similar tools
- Debugging flaky tests or investigating intermittent test failures
- Setting up E2E tests in CI/CD pipelines (GitHub Actions, GitLab CI, CircleCI)
- Testing across multiple browsers (Chrome, Firefox, Safari, WebKit, Edge)
- Implementing parallel test execution or test sharding for faster CI
- Testing responsive designs across mobile, tablet, and desktop viewports
- Writing custom Playwright commands or Cypress custom commands
- Setting up test fixtures, factories, or test data management
- Implementing authentication handling in E2E tests (session storage, cookies)
- Testing file uploads, downloads, or drag-and-drop interactions
- Recording and viewing test traces, videos, or screenshots for debugging
- Using Playwright codegen or Cypress Studio for test generation

## Core Concepts

### 1. Test Pyramid

- **E2E Tests**: Top of pyramid, test full user workflows
- **Integration Tests**: Middle, test component interactions
- **Unit Tests**: Base, test individual functions

### 2. Key Principles

- **Isolation**: Tests should not depend on each other
- **Reliability**: No flaky tests in CI/CD
- **Speed**: Optimize for fast feedback
- **Maintainability**: DRY code with Page Objects

### 3. Framework Comparison

| Feature | Playwright | Cypress |
|---------|------------|---------|
| Multi-browser | Chrome, Firefox, Safari, WebKit | Chrome, Firefox, Edge |
| Parallel execution | Built-in | Paid feature |
| API testing | Full support | Limited |
| Mobile testing | Device emulation | Viewport only |
| Language | JS/TS/Python/Java | JS/TS |

## Quick Start

### Playwright Setup

```bash
pnpm create playwright
# or
pnpm add -D @playwright/test
pnpm playwright install
```

### Cypress Setup

```bash
pnpm add -D cypress
pnpm cypress open
```

## Testing Patterns

### Pattern 1: Page Object Model (Playwright)

```typescript
// pages/login.page.ts
import { Page, Locator } from '@playwright/test';

export class LoginPage {
  readonly page: Page;
  readonly emailInput: Locator;
  readonly passwordInput: Locator;
  readonly submitButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.emailInput = page.getByLabel('Email');
    this.passwordInput = page.getByLabel('Password');
    this.submitButton = page.getByRole('button', { name: 'Sign in' });
  }

  async goto() {
    await this.page.goto('/login');
  }

  async login(email: string, password: string) {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }
}

// tests/login.spec.ts
import { test, expect } from '@playwright/test';
import { LoginPage } from '../pages/login.page';

test('successful login', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.login('user@example.com', 'password123');

  await expect(page).toHaveURL('/dashboard');
});
```

### Pattern 2: API Mocking (Playwright)

```typescript
import { test, expect } from '@playwright/test';

test('displays user data from API', async ({ page }) => {
  // Mock API response
  await page.route('**/api/users/me', (route) => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        id: 1,
        name: 'Test User',
        email: 'test@example.com'
      })
    });
  });

  await page.goto('/profile');
  await expect(page.getByText('Test User')).toBeVisible();
});
```

### Pattern 3: Visual Regression (Playwright)

```typescript
import { test, expect } from '@playwright/test';

test('homepage visual regression', async ({ page }) => {
  await page.goto('/');

  // Full page screenshot comparison
  await expect(page).toHaveScreenshot('homepage.png', {
    fullPage: true,
    maxDiffPixels: 100
  });
});

test('component screenshot', async ({ page }) => {
  await page.goto('/components');

  const card = page.locator('.product-card').first();
  await expect(card).toHaveScreenshot('product-card.png');
});
```

### Pattern 4: Accessibility Testing

```typescript
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test('page has no accessibility violations', async ({ page }) => {
  await page.goto('/');

  const accessibilityScanResults = await new AxeBuilder({ page }).analyze();

  expect(accessibilityScanResults.violations).toEqual([]);
});
```

### Pattern 5: Cypress Commands

```typescript
// cypress/support/commands.ts
Cypress.Commands.add('login', (email: string, password: string) => {
  cy.session([email, password], () => {
    cy.visit('/login');
    cy.get('[data-cy=email]').type(email);
    cy.get('[data-cy=password]').type(password);
    cy.get('[data-cy=submit]').click();
    cy.url().should('include', '/dashboard');
  });
});

// Usage in tests
cy.login('user@example.com', 'password123');
```

## CI/CD Integration

### GitHub Actions (Playwright)

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v2
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'pnpm'

      - run: pnpm install
      - run: pnpm playwright install --with-deps
      - run: pnpm playwright test

      - uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: playwright-report
          path: playwright-report/
```

## Debugging Tips

```typescript
// Enable headed mode for debugging
// npx playwright test --headed

// Use pause for debugging
await page.pause();

// Enable trace on failure
// playwright.config.ts
export default {
  use: {
    trace: 'on-first-retry',
    video: 'on-first-retry',
    screenshot: 'only-on-failure'
  }
};

// View trace
// npx playwright show-trace trace.zip
```

## Best Practices

1. **Use data-testid**: Stable selectors over CSS classes
2. **Avoid Sleeps**: Use proper waits and assertions
3. **Isolate Tests**: No shared state between tests
4. **Parallel Execution**: Speed up CI with parallelization
5. **Retry Flaky Tests**: But fix root cause
