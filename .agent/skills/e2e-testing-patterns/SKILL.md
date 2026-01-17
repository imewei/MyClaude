---
name: e2e-testing-patterns
version: "1.0.7"
maturity: "5-Expert"
specialization: E2E Test Automation
description: Build reliable E2E tests with Playwright and Cypress for web testing, browser automation, and CI/CD integration. Use when writing E2E tests, implementing Page Object Model, mocking APIs, visual regression, or accessibility testing.
---

# E2E Testing Patterns

Reliable end-to-end testing with Playwright and Cypress.

---

## Framework Comparison

| Feature | Playwright | Cypress |
|---------|------------|---------|
| Multi-browser | Chrome, Firefox, Safari, WebKit | Chrome, Firefox, Edge |
| Parallel execution | Built-in | Paid feature |
| API testing | Full support | Limited |
| Mobile testing | Device emulation | Viewport only |
| Language | JS/TS/Python/Java | JS/TS |

---

## Page Object Model (Playwright)

```typescript
// pages/login.page.ts
export class LoginPage {
  constructor(private page: Page) {}

  readonly emailInput = this.page.getByLabel('Email');
  readonly passwordInput = this.page.getByLabel('Password');
  readonly submitButton = this.page.getByRole('button', { name: 'Sign in' });

  async goto() { await this.page.goto('/login'); }

  async login(email: string, password: string) {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }
}

// tests/login.spec.ts
test('successful login', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.login('user@example.com', 'password');
  await expect(page).toHaveURL('/dashboard');
});
```

---

## API Mocking

```typescript
await page.route('**/api/users/me', (route) => {
  route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ id: 1, name: 'Test User' })
  });
});
```

---

## Visual Regression

```typescript
await expect(page).toHaveScreenshot('homepage.png', {
  fullPage: true,
  maxDiffPixels: 100
});
```

---

## Accessibility Testing

```typescript
import AxeBuilder from '@axe-core/playwright';

test('no a11y violations', async ({ page }) => {
  await page.goto('/');
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});
```

---

## CI/CD Integration (GitHub Actions)

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
        with: { node-version: 20, cache: 'pnpm' }
      - run: pnpm install && pnpm playwright install --with-deps
      - run: pnpm playwright test
      - uses: actions/upload-artifact@v4
        if: failure()
        with: { name: playwright-report, path: playwright-report/ }
```

---

## Debugging

```typescript
// Headed mode: npx playwright test --headed
// Pause for debugging:
await page.pause();

// playwright.config.ts - trace on failure
export default {
  use: {
    trace: 'on-first-retry',
    video: 'on-first-retry',
    screenshot: 'only-on-failure'
  }
};
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Stable selectors | Use data-testid over CSS classes |
| Avoid sleeps | Use proper waits and assertions |
| Isolate tests | No shared state between tests |
| Parallel execution | Speed up CI |
| Retry flaky tests | But fix root cause |
| Page objects | Encapsulate selectors |

---

## Checklist

- [ ] Page Object Model for reusability
- [ ] data-testid for stable selectors
- [ ] No hard-coded waits (use assertions)
- [ ] Tests isolated (no dependencies)
- [ ] Parallel execution enabled
- [ ] API mocking for isolation
- [ ] CI/CD integration with artifacts
- [ ] Accessibility testing included

---

**Version**: 1.0.5
