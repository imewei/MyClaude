---
description: Orchestrate production-ready React/React Native component generation
  with TypeScript, tests, styles, and documentation
triggers:
- /component-scaffold
- orchestrate production ready react/react native
allowed-tools: [Write, Read, Task]
version: 1.0.0
---



# React/React Native Component Scaffolding

Generate production-ready components with TypeScript, tests, styles, and documentation.

## Component

$ARGUMENTS

## Mode Selection

| Mode | Duration | Output |
|------|----------|--------|
| `--quick` | 5-10 min | ComponentSpec interface and recommendations only |
| standard (default) | 15-30 min | Complete component with TypeScript and styling |
| `--deep` | 30-60 min | Full scaffold with tests, Storybook, and accessibility |

**Options:** `--platform=web|native|universal`, `--styling=css-modules|styled-components|tailwind`, `--tests`, `--storybook`, `--accessibility`

## Phase 1: Requirements Analysis

**Agent:** frontend-developer or multi-platform-mobile

### Determine Component Characteristics

| Classification | Decision Rule |
|----------------|---------------|
| **Platform** | Web-specific APIs → web; Native features → native; Shared → universal |
| **Type** | Complex state/data → page; Input validation → form; Wraps children → layout; Structured data → data-display; Otherwise → functional |
| **Styling** | React Native → StyleSheet; Dynamic theming → styled-components; Type safety → CSS Modules; Rapid prototyping → Tailwind |

### Extract Component Spec
- Parse name (enforce PascalCase)
- Identify required vs optional props
- Determine state requirements and hooks
- Check for naming conflicts in codebase

**Success:** ComponentSpec populated, platform justified, styling recommended, type classified

🚨 **Quick Mode exits here**

## Phase 2: Component Generation

### Generate Files

| File | Content |
|------|---------|
| `{Name}.tsx` | Functional component with TypeScript props, destructured defaults, semantic HTML/native elements |
| `{Name}.types.ts` | Props interface with JSDoc comments |
| `index.ts` | Barrel exports for tree-shaking |

### Accessibility (if `--accessibility`)
- ARIA attributes (role, aria-label, aria-describedby)
- Keyboard navigation
- Focus management
- Screen reader compatibility

**Success:** TypeScript compiles, props typed, semantic structure, proper exports

## Phase 3: Styling Implementation

| Approach | Implementation |
|----------|----------------|
| **CSS Modules** | `.module.css` with CSS variables for theming |
| **styled-components** | Theme props, prop-based dynamic styling |
| **Tailwind CSS** | Utility classes with responsive modifiers |
| **React Native StyleSheet** | StyleSheet.create with Platform.select |

**Success:** Styles follow project conventions, design tokens used, responsive design implemented

## Phase 4: Testing & Documentation (Deep Mode)

**Agents:** frontend-developer, test-automator

### Generate Tests (if `--tests`)
- React Testing Library tests
- Basic rendering, prop validation, event handlers
- Accessibility tests (axe-core)

### Generate Storybook (if `--storybook`)
- Default story with typical props
- Variant stories (Primary, Secondary, Disabled)
- ArgTypes with controls
- Interactive play functions

**Success:** ≥90% coverage, Storybook renders, axe-core passes

## Phase 5: Validation & Integration

```bash
npx tsc --noEmit          # TypeScript check
npm test {Name}.test      # Run tests
npm run storybook         # Verify stories
npm run lint              # Lint check
```

**Success:** Zero TypeScript errors, all tests pass, linting passes

🎯 **Deep Mode complete**

## Agent Selection

| Platform | Agent |
|----------|-------|
| native or universal | multi-platform-mobile |
| web (default) | frontend-developer |

## Examples

```bash
# Web component with Tailwind
/component-scaffold ProductCard --platform=web --styling=tailwind

# React Native with tests
/component-scaffold UserProfile --platform=native --tests

# Universal with full suite
/component-scaffold CheckoutForm --deep --platform=universal --styling=styled-components
```

## Will / Won't

**Will:**
- ✅ Generate fully typed TypeScript components
- ✅ Apply consistent styling patterns
- ✅ Create comprehensive tests (deep mode)
- ✅ Generate Storybook documentation (deep mode)
- ✅ Follow accessibility best practices

**Won't:**
- ❌ Implement business logic
- ❌ Create API integrations
- ❌ Handle global state management
- ❌ Generate backend code
- ❌ Deploy components
