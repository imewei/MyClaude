---
name: frontend-and-mobile
description: >
  Meta-orchestrator for implementation-focused frontend and mobile engineering. Routes to
  React/Flutter/cross-platform UI architecture, modern JavaScript, TypeScript typing and
  scaffolding, WCAG/ARIA accessibility testing, and mobile E2E test patterns. Use when building
  or testing web/mobile UI code, configuring TypeScript frontend projects, implementing accessible
  components, or validating React Native/Flutter apps. For visual design, UX shaping, or high-polish
  landing pages, use frontend-design:frontend-design or ui-ux-pro-max; for scientific desktop GUIs
  backed by JAX/Julia/PyQt/Makie, use app-developer.
---

# Frontend and Mobile

> **SEE ALSO:** For visual design, UX shaping, or polished React/Next.js pages, use `frontend-design:frontend-design` or `ui-ux-pro-max`. For React/Next.js performance, use `vercel-react-best-practices`. For scientific desktop GUIs coupled to numerical backends, use `app-developer`.
> This hub handles **implementation patterns**: component architecture, JavaScript/TypeScript structure, accessibility validation, and mobile test strategy.

Orchestrator for frontend and mobile engineering. Routes to the correct implementation skill based on platform, language concern, accessibility requirement, or test scope.

## Expert Agent

- **`app-developer`**: Specialist for scientific application UIs tightly coupled to numerical backends.
  - *Location*: `plugins/dev-suite/agents/app-developer.md`
  - *Capabilities*: PyQt/PySide6, Makie/PyQtGraph, scientific dashboards, accessibility, and data-interface patterns.

## Core Skills

### [Frontend & Mobile Engineering](../frontend-mobile-engineering/SKILL.md)
React component architecture, Flutter widgets, and cross-platform UI patterns.

### [Modern JavaScript Patterns](../modern-javascript-patterns/SKILL.md)
ES2024+ features, module systems, async patterns, and runtime optimization.

### [TypeScript Advanced Types](../typescript-advanced-types/SKILL.md)
Conditional types, mapped types, template literals, and type inference techniques.

### [TypeScript Project Scaffolding](../typescript-project-scaffolding/SKILL.md)
tsconfig setup, monorepo TypeScript configuration, and build pipeline integration.

### [Accessibility Testing](../accessibility-testing/SKILL.md)
WCAG 2.1 AA compliance, ARIA roles, screen reader testing, and automated audits.

### [Mobile Testing Patterns](../mobile-testing-patterns/SKILL.md)
Unit, integration, and E2E testing for React Native and Flutter applications.

## Routing Decision Tree

```
What is the frontend or mobile concern?
|
+-- Component design / UI architecture / Flutter widgets?
|   --> frontend-mobile-engineering
|
+-- Modern JS syntax / async / module bundling?
|   --> modern-javascript-patterns
|
+-- TypeScript type-level programming?
|   --> typescript-advanced-types
|
+-- Project setup / tsconfig / build pipeline?
|   --> typescript-project-scaffolding
|
+-- WCAG compliance / screen readers / ARIA?
|   --> accessibility-testing
|
+-- Mobile unit / integration / E2E tests?
    --> mobile-testing-patterns
```

## Routing Table

| Trigger                              | Sub-skill                       |
|--------------------------------------|---------------------------------|
| React, Flutter, components, widgets  | frontend-mobile-engineering     |
| ES2024, modules, async/await, bundler| modern-javascript-patterns      |
| Generics, conditional types, utility | typescript-advanced-types       |
| tsconfig, paths, composite projects  | typescript-project-scaffolding  |
| WCAG, ARIA, axe, screen reader       | accessibility-testing           |
| Detox, Flutter test, Maestro, mocks  | mobile-testing-patterns         |

## Checklist

- [ ] Identify the target platform (web / iOS / Android / cross-platform) first
- [ ] Confirm TypeScript strictness level before scaffolding a project
- [ ] Verify WCAG 2.1 AA requirements are met for all interactive elements
- [ ] Check that accessibility tests run in CI alongside unit tests
- [ ] Validate mobile tests cover offline and low-connectivity scenarios
- [ ] Ensure component API is typed end-to-end with no `any` escape hatches
