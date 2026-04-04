---
name: frontend-and-mobile
description: Meta-orchestrator for frontend and mobile development. Routes to React/Flutter, JavaScript, TypeScript, accessibility, and mobile testing skills. Use when building React/Flutter apps, writing modern JavaScript/TypeScript, scaffolding frontend projects, implementing accessibility, or testing mobile apps.
---

# Frontend and Mobile

Orchestrator for frontend and mobile development across web and native ecosystems. Routes to the appropriate specialized skill based on the platform, language, or quality concern.

## Expert Agent

- **`app-developer`**: Specialist for cross-platform UI, component architecture, and mobile performance.
  - *Location*: `plugins/dev-suite/agents/app-developer.md`
  - *Capabilities*: React, Flutter, responsive design, accessibility, and mobile testing strategies.

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
