---
name: accessibility-testing
description: Implement web accessibility testing for WCAG 2.1/2.2 compliance including automated scanning (axe-core, pa11y), screen reader testing, keyboard navigation, ARIA patterns, and color contrast validation. Use when auditing accessibility, implementing ARIA, or building inclusive interfaces.
---

# Accessibility Testing

## Expert Agent

For accessibility auditing, WCAG compliance, and inclusive interface testing, delegate to:

- **`quality-specialist`**: Implements testing strategies including accessibility validation and compliance checks.
  - *Location*: `plugins/dev-suite/agents/quality-specialist.md`


## WCAG Conformance Levels

| Level | Requirement | Examples |
|-------|------------|---------|
| A | Minimum | Alt text, keyboard access, no seizure triggers |
| AA | Standard (legal target) | Color contrast 4.5:1, resize to 200%, focus visible |
| AAA | Enhanced | Contrast 7:1, sign language, reading level |

Target **WCAG 2.1 Level AA** for most projects.


## Automated Scanning

### axe-core with Playwright

```typescript
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("homepage has no accessibility violations", async ({ page }) => {
  await page.goto("/");

  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .exclude("#third-party-widget")
    .analyze();

  expect(results.violations).toEqual([]);
});

test("form page meets WCAG AA", async ({ page }) => {
  await page.goto("/signup");

  const results = await new AxeBuilder({ page })
    .include("#signup-form")
    .withTags(["wcag2aa"])
    .analyze();

  for (const violation of results.violations) {
    console.error(`${violation.id}: ${violation.description}`);
    for (const node of violation.nodes) {
      console.error(`  - ${node.html}`);
    }
  }

  expect(results.violations).toHaveLength(0);
});
```

## ARIA Patterns

### Landmark Roles

```html
<header role="banner">
  <nav role="navigation" aria-label="Main">...</nav>
</header>
<main role="main">
  <section aria-labelledby="section-title">
    <h2 id="section-title">Products</h2>
  </section>
</main>
<aside role="complementary">...</aside>
<footer role="contentinfo">...</footer>
```

### Interactive Components

```html
<!-- Disclosure (expandable) -->
<button aria-expanded="false" aria-controls="panel-1">Details</button>
<div id="panel-1" role="region" hidden>Content here</div>

<!-- Tab panel -->
<div role="tablist" aria-label="Settings">
  <button role="tab" aria-selected="true" aria-controls="tab-general">General</button>
  <button role="tab" aria-selected="false" aria-controls="tab-security">Security</button>
</div>
<div role="tabpanel" id="tab-general">General settings...</div>
<div role="tabpanel" id="tab-security" hidden>Security settings...</div>

<!-- Live region for dynamic updates -->
<div aria-live="polite" aria-atomic="true" role="status">
  3 items in your cart
</div>
```

### ARIA Rules

- Use native HTML elements first (`button`, not `div role="button"`)
- Every interactive element needs an accessible name
- `aria-live` for dynamic content updates
- `aria-expanded` for disclosure patterns
- Never use `aria-hidden="true"` on focusable elements


## Keyboard Navigation

### Required Keyboard Support

| Key | Action |
|-----|--------|
| Tab | Move to next focusable element |
| Shift+Tab | Move to previous focusable element |
| Enter/Space | Activate button or link |
| Escape | Close modal, dismiss popup |
| Arrow keys | Navigate within composite widgets |

### Focus Management

```typescript
function openModal(modalId: string): void {
  const modal = document.getElementById(modalId);
  if (!modal) return;

  modal.removeAttribute("hidden");
  const firstFocusable = modal.querySelector<HTMLElement>(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  firstFocusable?.focus();

  // Trap focus inside modal
  modal.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Escape") closeModal(modalId);
    if (e.key === "Tab") trapFocus(e, modal);
  });
}
```

### Focus Testing Checklist

- [ ] All interactive elements reachable via Tab
- [ ] Focus order matches visual order
- [ ] Focus indicator visible (min 2px, 3:1 contrast)
- [ ] No focus traps (except modals)
- [ ] Skip-to-content link present


## Color Contrast

| Element | Minimum Ratio (AA) | Enhanced (AAA) |
|---------|-------------------|----------------|
| Normal text | 4.5:1 | 7:1 |
| Large text (18px+) | 3:1 | 4.5:1 |
| UI components | 3:1 | 3:1 |
| Focus indicator | 3:1 | 3:1 |

### Testing Tools

- Chrome DevTools: Inspect element > color contrast ratio
- axe-core: Automated contrast checks
- WebAIM Contrast Checker: Manual verification


## Audit Checklist

- [ ] axe-core integrated into test suite
- [ ] All pages pass WCAG 2.1 AA automated checks
- [ ] Keyboard navigation tested on all interactive flows
- [ ] Color contrast meets AA minimums
- [ ] ARIA landmarks and labels applied correctly
- [ ] Screen reader tested on primary user flows
- [ ] Focus management correct for modals and dynamic content
- [ ] Skip-to-content link present
