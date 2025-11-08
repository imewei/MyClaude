---
version: 1.0.3
description: Orchestrate production-ready React/React Native component generation with TypeScript, tests, styles, and documentation
execution_time:
  quick: "5-10 minutes - Requirements analysis and component specification only"
  standard: "15-30 minutes - Complete component with TypeScript and styling"
  deep: "30-60 minutes - Full scaffold with tests, Storybook, and accessibility validation"
external_docs:
  - component-patterns-library.md
  - testing-strategies.md
  - styling-approaches.md
  - storybook-integration.md
agents:
  primary:
    - frontend-mobile-development:frontend-developer
    - frontend-mobile-development:mobile-developer
  conditional: []
color: blue
tags: [component-scaffolding, react, react-native, typescript, testing, storybook]
allowed-tools: [Read, Write, Edit, Bash, Glob, Grep, Task]
---

# React/React Native Component Scaffolding

> **Purpose**: Orchestrate systematic component generation with production-ready patterns, comprehensive testing, and accessibility-first implementation

---

## Command Syntax

```bash
/component-scaffold <component-name> [options]

# Quick mode: Requirements analysis only
/component-scaffold UserProfile --quick

# Standard mode: Complete component with styling
/component-scaffold ProductCard --platform=web --styling=tailwind

# Deep mode: Full scaffold with tests and Storybook
/component-scaffold CheckoutForm --deep --tests --storybook --accessibility
```

---

## Execution Modes

### Quick Mode (5-10 minutes)
**Purpose**: Analyze requirements and generate component specification

**Output**:
- ComponentSpec interface with props, state, hooks
- Platform selection recommendation (web/native/universal)
- Styling approach recommendation
- Component type classification

**When to use**:
- Planning phase before implementation
- Reviewing component requirements
- Architecture decision documentation

### Standard Mode (15-30 minutes)
**Purpose**: Generate complete component with TypeScript and styling

**Output**:
- Fully typed React/React Native component
- TypeScript interfaces and prop types
- Styled implementation (CSS Modules, styled-components, or Tailwind)
- Index file with barrel exports
- Basic documentation

**When to use**:
- Production component implementation
- Consistent component architecture
- Type-safe component library

### Deep Mode (30-60 minutes)
**Purpose**: Full scaffold with tests, Storybook, and accessibility

**Output**:
- Everything from Standard Mode, plus:
- Comprehensive test suite (React Testing Library)
- Storybook stories with argTypes and controls
- Accessibility validation (axe-core integration)
- Multiple component variants

**When to use**:
- Design system components
- Shared component library
- Components requiring comprehensive documentation

---

## Phase 1: Requirements Analysis

### Objective
Extract component specifications from user input and classify component characteristics

### Key Activities

1. **Extract Component Name**
   - Parse component name from arguments
   - Validate PascalCase naming convention
   - Check for naming conflicts in codebase

2. **Infer Component Type**
   ```
   Component Types:
   - functional: Stateless presentational components
   - page: Top-level route components
   - layout: Wrapper components with slots
   - form: Input forms with validation
   - data-display: Tables, lists, cards
   ```

3. **Identify Props and State**
   - Parse required vs optional props
   - Identify state requirements
   - Determine hook dependencies (useState, useEffect, useContext)

4. **Detect Platform**
   ```
   Platform Decision Tree:
   - Web-only features → platform: web
   - Native-only features → platform: native
   - Shared codebase → platform: universal
   ```

5. **Select Styling Approach**
   ```
   Styling Strategy:
   - CSS Modules: Component libraries, type safety
   - styled-components: Dynamic theming, props-based styling
   - Tailwind CSS: Rapid prototyping, utility-first
   - React Native StyleSheet: Native mobile apps
   ```

### Success Criteria
- ✅ ComponentSpec interface populated with all required fields
- ✅ Platform selection justified with rationale
- ✅ Styling approach recommended based on project context
- ✅ Component type classified correctly

### Tools
- Use **Grep** to search for existing components with similar names
- Use **Read** to analyze project styling patterns
- Use **frontend-developer** or **mobile-developer** agent for platform-specific guidance

**Reference**: See `component-patterns-library.md` for ComponentSpec interface definition

---

## Phase 2: Component Generation

### Objective
Generate production-ready component files with TypeScript, proper structure, and best practices

### Key Activities

1. **Generate TypeScript Interfaces**
   ```typescript
   export interface {ComponentName}Props {
     /** Prop description */
     propName: PropType;
     optionalProp?: PropType;
   }
   ```

2. **Generate Component Body**
   - Functional component with proper typing
   - Destructured props with default values
   - State hooks (useState) for internal state
   - Effect hooks (useEffect) for side effects
   - Proper JSX structure with semantic HTML

3. **Add Accessibility Features** (if --accessibility flag)
   - ARIA attributes (role, aria-label, aria-describedby)
   - Keyboard navigation support
   - Focus management
   - Screen reader compatibility

4. **Generate Platform-Specific Code**
   - **Web**: React with DOM elements
   - **Native**: React Native with View, Text, TouchableOpacity
   - **Universal**: Conditional rendering for platform differences

5. **Create Index File**
   ```typescript
   export { ComponentName } from './ComponentName';
   export type { ComponentNameProps } from './ComponentName';
   ```

### Success Criteria
- ✅ Component compiles without TypeScript errors
- ✅ All props properly typed with JSDoc comments
- ✅ Semantic HTML structure (for web components)
- ✅ Proper exports for tree-shaking

### Tools
- Use **Write** to create component files
- Use **Task** to invoke **frontend-developer** for web components
- Use **Task** to invoke **mobile-developer** for native components

**Reference**: See `component-patterns-library.md` for implementation patterns

---

## Phase 3: Styling Implementation

### Objective
Generate styles based on selected styling approach with consistent design tokens

### Key Activities

1. **CSS Modules** (if --styling=css-modules)
   ```css
   .componentName {
     display: flex;
     padding: var(--spacing-md);
   }
   ```
   - Create `.module.css` file
   - Use CSS variables for theming
   - Compose styles for reusability

2. **styled-components** (if --styling=styled-components)
   ```typescript
   export const Container = styled.div`
     padding: ${({ theme }) => theme.spacing.md};
   `;
   ```
   - Create styled component definitions
   - Use theme props for consistency
   - Add prop-based dynamic styling

3. **Tailwind CSS** (if --styling=tailwind)
   ```typescript
   <div className="flex flex-col p-4 bg-white rounded-lg shadow">
   ```
   - Apply utility classes in JSX
   - Use responsive modifiers (sm:, md:, lg:)
   - Add hover/focus states

4. **React Native StyleSheet** (if --platform=native)
   ```typescript
   const styles = StyleSheet.create({
     container: { flex: 1, padding: 16 }
   });
   ```
   - Create StyleSheet with typed styles
   - Use platform-specific styles (Platform.select)
   - Optimize for performance

### Success Criteria
- ✅ Styles follow project conventions
- ✅ Design tokens used for spacing, colors, typography
- ✅ Responsive design implemented (for web)
- ✅ Platform-specific optimizations applied (for native)

### Tools
- Use **Read** to analyze existing styling patterns
- Use **Write** to create style files
- Use **Edit** to update component with style imports

**Reference**: See `styling-approaches.md` for detailed patterns and best practices

---

## Phase 4: Testing & Documentation

### Objective
Generate comprehensive test suite and Storybook stories for component validation

### Key Activities

1. **Generate Unit Tests** (if --tests flag)
   ```typescript
   describe('ComponentName', () => {
     it('renders without crashing', () => {
       render(<ComponentName {...defaultProps} />);
       expect(screen.getByRole('region')).toBeInTheDocument();
     });
   });
   ```
   - Basic rendering test
   - Prop validation tests
   - Event handler tests
   - Accessibility tests (axe-core)

2. **Generate Storybook Stories** (if --storybook flag)
   ```typescript
   export const Default: Story = {
     args: { prop1: 'value1', prop2: 'value2' }
   };
   ```
   - Default story with typical props
   - Variant stories (Primary, Secondary, Disabled, etc.)
   - Interactive story with play function
   - ArgTypes with controls

3. **Generate Documentation**
   - JSDoc comments for props
   - Usage examples in component file
   - README for complex components

### Success Criteria
- ✅ Test coverage ≥90% for component logic
- ✅ All Storybook stories render correctly
- ✅ Accessibility tests pass (axe-core)
- ✅ Interactive stories demonstrate user flows

### Tools
- Use **Write** to create test files and stories
- Use **Bash** to run tests: `npm test ComponentName.test`
- Use **frontend-developer** agent for accessibility validation

**Reference**:
- See `testing-strategies.md` for test patterns
- See `storybook-integration.md` for story generation

---

## Phase 5: Validation & Integration

### Objective
Validate generated component and integrate into project structure

### Key Activities

1. **TypeScript Compilation**
   ```bash
   npx tsc --noEmit
   ```
   - Verify no type errors
   - Check for missing imports
   - Validate prop types

2. **Run Tests**
   ```bash
   npm test ComponentName.test
   ```
   - Ensure all tests pass
   - Verify accessibility tests
   - Check code coverage

3. **Run Storybook** (if --storybook)
   ```bash
   npm run storybook
   ```
   - Verify stories load correctly
   - Test interactive controls
   - Check responsive behavior

4. **Lint & Format**
   ```bash
   npm run lint
   npm run format
   ```
   - Fix linting errors
   - Apply consistent formatting
   - Remove unused imports

5. **Integration Check**
   - Import component in parent component
   - Verify proper tree-shaking
   - Check bundle size impact

### Success Criteria
- ✅ Zero TypeScript errors
- ✅ All tests pass with ≥90% coverage
- ✅ Linting passes with zero errors
- ✅ Component integrates successfully

### Tools
- Use **Bash** to run validation commands
- Use **Read** to verify generated files
- Use **Edit** to fix any issues discovered

---

## Decision Trees

### Platform Selection

```
Choose platform based on:

1. Does the component use web-specific APIs (DOM, window, document)?
   YES → platform: web
   NO → Continue

2. Does the component use native-specific features (Camera, GPS, Biometrics)?
   YES → platform: native
   NO → Continue

3. Is the component shared across web and mobile?
   YES → platform: universal (use React Native Web)
   NO → Default to web
```

### Styling Approach

```
Choose styling based on:

1. Is this a React Native component?
   YES → Use StyleSheet.create
   NO → Continue

2. Do you need dynamic theming with props?
   YES → Use styled-components
   NO → Continue

3. Is this a design system component requiring type safety?
   YES → Use CSS Modules
   NO → Use Tailwind CSS (rapid prototyping)
```

### Component Type Classification

```
Classify component type:

1. Does the component fetch data or manage complex state?
   YES → Type: page
   NO → Continue

2. Does the component have input fields with validation?
   YES → Type: form
   NO → Continue

3. Does the component wrap children with layout structure?
   YES → Type: layout
   NO → Continue

4. Does the component display data in structured format (table, list, cards)?
   YES → Type: data-display
   NO → Type: functional (default)
```

---

## Options Reference

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--quick` | flag | false | Requirements analysis only |
| `--platform` | web, native, universal | web | Target platform |
| `--styling` | css-modules, styled-components, tailwind | auto-detect | Styling approach |
| `--tests` | flag | false (standard), true (deep) | Generate test suite |
| `--storybook` | flag | false (standard), true (deep) | Generate Storybook stories |
| `--accessibility` | flag | false (standard), true (deep) | Add a11y features |
| `--deep` | flag | false | Enable deep mode (tests + storybook + a11y) |

---

## Examples

### Example 1: Web Component with Tailwind

```bash
/component-scaffold ProductCard --platform=web --styling=tailwind
```

**Generated Files**:
- `ProductCard.tsx` - Component with Tailwind classes
- `ProductCard.types.ts` - TypeScript interfaces
- `index.ts` - Barrel export

### Example 2: React Native Component with Tests

```bash
/component-scaffold UserProfile --platform=native --tests
```

**Generated Files**:
- `UserProfile.tsx` - React Native component with StyleSheet
- `UserProfile.types.ts` - TypeScript interfaces
- `UserProfile.test.tsx` - Test suite
- `index.ts` - Barrel export

### Example 3: Universal Component with Full Suite

```bash
/component-scaffold CheckoutForm --deep --platform=universal --styling=styled-components
```

**Generated Files**:
- `CheckoutForm.tsx` - Universal component with styled-components
- `CheckoutForm.types.ts` - TypeScript interfaces
- `CheckoutForm.styles.ts` - Styled components
- `CheckoutForm.test.tsx` - Comprehensive tests
- `CheckoutForm.stories.tsx` - Storybook stories
- `index.ts` - Barrel export

---

## What This Command Will Do

✅ **Analyze Requirements**: Extract component specs from user input
✅ **Generate TypeScript Component**: Fully typed with prop interfaces
✅ **Apply Styling**: CSS Modules, styled-components, Tailwind, or StyleSheet
✅ **Create Tests**: Unit tests with React Testing Library (optional)
✅ **Generate Storybook**: Interactive documentation (optional)
✅ **Validate Accessibility**: axe-core testing and ARIA attributes (optional)
✅ **Ensure Type Safety**: Zero TypeScript errors
✅ **Follow Best Practices**: Semantic HTML, proper hooks usage, performance optimization

---

## What This Command Won't Do

❌ **Implement Business Logic**: Focus is on component structure, not application logic
❌ **Create API Integrations**: Use separate data fetching patterns
❌ **Generate Backend Code**: Frontend/mobile components only
❌ **Handle State Management**: Use Zustand, Redux, or Context separately
❌ **Deploy Components**: Local generation only, deployment is separate

---

## Agent Orchestration

### Primary Agent Selection

```typescript
if (platform === 'native' || platform === 'universal') {
  agent = 'frontend-mobile-development:mobile-developer';
} else {
  agent = 'frontend-mobile-development:frontend-developer';
}
```

### Agent Invocation Pattern

```typescript
// Phase 2: Component Generation
Task({
  subagent_type: agent,
  prompt: `Generate ${componentSpec.name} component:
    - Type: ${componentSpec.type}
    - Platform: ${componentSpec.platform}
    - Props: ${JSON.stringify(componentSpec.props)}
    - Styling: ${componentSpec.styling}
    - Include accessibility: ${options.accessibility}`
});
```

---

## Troubleshooting

### Issue: TypeScript Compilation Errors

**Cause**: Missing type definitions or incorrect prop types

**Solution**:
1. Run `npx tsc --noEmit` to identify specific errors
2. Verify all props have type annotations
3. Check for missing imports (`React`, `FC`, etc.)
4. Use **Edit** to fix type errors

### Issue: Tests Failing

**Cause**: Missing test utilities or incorrect assertions

**Solution**:
1. Verify `@testing-library/react` is installed
2. Check that `defaultProps` match component requirements
3. Use `screen.debug()` to inspect rendered output
4. Review `testing-strategies.md` for correct patterns

### Issue: Storybook Not Loading

**Cause**: Missing Storybook configuration or incorrect imports

**Solution**:
1. Verify Storybook is initialized: `npx storybook@latest init`
2. Check `.storybook/main.ts` includes component directory
3. Ensure story file uses correct imports
4. Run `npm run storybook` to see error details

---

## Best Practices

1. **Always Start with Requirements Analysis**: Use `--quick` mode first to validate specs
2. **Follow Project Conventions**: Analyze existing components before generating new ones
3. **Prioritize Accessibility**: Use `--accessibility` flag for user-facing components
4. **Generate Tests Early**: Include `--tests` to catch issues during development
5. **Use Storybook for Documentation**: `--storybook` flag creates living documentation
6. **Validate After Generation**: Run TypeScript, tests, and lint before committing
7. **Keep Components Focused**: Single responsibility principle - one concern per component
8. **Use Semantic HTML**: Proper element types (button, nav, article) for accessibility
9. **Implement Responsive Design**: Mobile-first approach with responsive utilities
10. **Optimize for Performance**: React.memo, useMemo, useCallback for expensive operations

---

## Integration with Other Commands

- **After Generation**: Use `/test-generate` for additional test coverage
- **Before Committing**: Use `/double-check` for comprehensive validation
- **For Documentation**: Use `/update-docs` to sync with project documentation
- **For Migration**: Use `/code-migrate` when upgrading component frameworks

---

## Success Metrics

- **Time to Component**: 5-60 minutes depending on mode
- **Type Safety**: 100% TypeScript coverage
- **Test Coverage**: ≥90% for deep mode components
- **Accessibility**: Zero axe-core violations
- **Bundle Size**: Optimized with tree-shaking
- **Developer Experience**: Consistent component structure across project

---

For implementation details, see:
- `component-patterns-library.md` - Component generator classes and patterns
- `testing-strategies.md` - Test generation strategies
- `styling-approaches.md` - CSS Modules, styled-components, Tailwind patterns
- `storybook-integration.md` - Storybook story generation
