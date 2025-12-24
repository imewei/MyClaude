---
version: "1.0.5"
command: /multi-platform
description: Build and deploy features across web, mobile, and desktop platforms with API-first architecture and multi-agent orchestration
execution_modes:
  quick:
    duration: "3-5 days"
    description: "Single-platform MVP (web OR mobile) with API-first design"
    agents: ["backend-architect", "ui-ux-designer", "frontend-developer OR mobile-developer"]
    scope: "1 platform, basic design system, API contract, core features"
  standard:
    duration: "2-3 weeks"
    description: "Web + Mobile (iOS/Android) with design system and feature parity"
    agents: ["backend-architect", "ui-ux-designer", "frontend-developer", "ios-developer", "mobile-developer", "test-automator"]
    scope: "3 platforms (web, iOS, Android), full design system, cross-platform testing, feature parity validation"
  enterprise:
    duration: "4-6 weeks"
    description: "All platforms (web, iOS, Android, desktop) with shared code and comprehensive optimization"
    agents: ["backend-architect", "ui-ux-designer", "frontend-developer", "ios-developer", "mobile-developer", "flutter-expert", "performance-engineer", "test-automator", "docs-architect"]
    scope: "4+ platforms, shared business logic (Kotlin Multiplatform/Flutter Web), performance optimization, comprehensive testing, production deployment"
workflow_type: "hybrid"
interactive_mode: true
---

# Multi-Platform Feature Development Workflow

Build and deploy the same feature consistently across web, mobile, and desktop platforms using API-first architecture and parallel implementation strategies.

[Extended thinking: This workflow orchestrates multiple specialized agents to ensure feature parity across platforms while maintaining platform-specific optimizations. The coordination strategy emphasizes shared contracts and parallel development with regular synchronization points. By establishing API contracts and data models upfront, teams can work independently while ensuring consistency.]

## Execution Mode Selection

Use AskUserQuestion to select execution mode based on project requirements:

- **Quick** (3-5 days): Single platform MVP (web OR mobile) with API contract validation
- **Standard** (2-3 weeks): Web + Mobile (iOS/Android) with feature parity and cross-platform testing
- **Enterprise** (4-6 weeks): All platforms with shared code, optimization, and comprehensive deployment

## Agent Coordination Reference

| Phase | Tasks | Agents | Mode | Duration |
|-------|-------|--------|------|----------|
| 1. Architecture | API contracts, Design system, Shared logic | backend-architect, ui-ux-designer, architect-review (optional) | Sequential | 2-3 days |
| 2. Implementation | Web, iOS, Android, Desktop (parallel) | frontend-developer, ios-developer, mobile-developer, flutter-expert | Parallel | 5-15 days |
| 3. Validation | Testing, Optimization, Documentation | test-automator, performance-engineer, docs-architect | Sequential | 2-5 days |

**Cross-Plugin Dependencies** (graceful degradation):
- architect-review (comprehensive-review): Shared business logic architecture
- test-automator (unit-testing/full-stack-orchestration): Cross-platform testing
- performance-engineer (full-stack-orchestration): Platform optimizations
- docs-architect (code-documentation): API documentation

## Phase 1: Architecture and API Design (Sequential)

### 1. Define Feature Requirements and API Contracts

- Use Task tool with subagent_type="backend-development:backend-architect"
- Prompt: "Design the API contract for feature: $ARGUMENTS. Create OpenAPI 3.1 specification with RESTful endpoints, GraphQL schema (if applicable), WebSocket events, request/response schemas, authentication requirements, rate limiting, error response formats. Define shared data models that all platforms will consume."
- Expected output: Complete API specification, data models, integration guidelines

**See**: [Platform Architecture Guide](../docs/multi-platform/platform-architecture.md#api-first-architecture)

### 2. Design System and UI/UX Consistency

- Use Task tool with subagent_type="multi-platform-apps:ui-ux-designer"
- Prompt: "Create cross-platform design system for feature using API spec: [previous output]. Include component specifications for each platform (Material Design, iOS HIG, Fluent), responsive layouts, native patterns, desktop considerations, accessibility requirements (WCAG 2.2 Level AA), dark/light theme specifications."
- Expected output: Design system documentation, component library specs, platform guidelines

**See**: [Design Systems Guide](../docs/multi-platform/design-systems.md)

### 3. Shared Business Logic Architecture

- Use Task tool with subagent_type="comprehensive-review:architect-review"
- Prompt: "Design shared business logic architecture for cross-platform feature. Define core domain models, business rules, validation logic, state management patterns (MVI/Redux/BLoC), caching and offline strategies, error handling, platform-specific adapter patterns. Consider Kotlin Multiplatform for mobile or TypeScript for web/desktop sharing."
- Expected output: Shared code architecture, platform abstraction layers, implementation guide

**See**: [Platform Architecture Guide](../docs/multi-platform/platform-architecture.md#shared-business-logic-strategies)

## Phase 2: Parallel Platform Implementation

### 4. Platform-Specific Implementations

Execute platform implementations in parallel based on selected execution mode:

**4a. Web (React/Next.js)** - Use Task with subagent_type="frontend-mobile-development:frontend-developer"
**4b. iOS (SwiftUI)** - Use Task with subagent_type="multi-platform-apps:ios-developer"
**4c. Android (Kotlin/Compose)** - Use Task with subagent_type="frontend-mobile-development:mobile-developer"
**4d. Desktop (Electron/Tauri)** - Use Task with subagent_type="frontend-mobile-development:frontend-developer"

Each implementation follows the shared API contract, design system, and business logic patterns defined in Phase 1.

**See**: [Implementation Guides](../docs/multi-platform/implementation-guides.md)

## Phase 3: Integration and Validation

### 5. Cross-Platform Testing and Feature Parity

- Use Task tool with subagent_type="unit-testing:test-automator"
- Prompt: "Validate feature parity across all platforms: functional testing matrix (features work identically), UI consistency verification (follows design system), performance benchmarks per platform, accessibility testing, network resilience testing (offline, slow connections), data synchronization validation, platform-specific edge cases, end-to-end user journey tests. Create test report with any platform discrepancies."
- Expected output: Test report, parity matrix (variance <5%), performance metrics

**See**: [Testing Strategies Guide](../docs/multi-platform/testing-strategies.md)

### 6. Platform-Specific Optimizations

- Use Task tool with subagent_type="full-stack-orchestration:performance-engineer"
- Prompt: "Optimize each platform implementation: Web (bundle size, lazy loading, CDN, SEO), iOS (app size, launch time, memory usage, battery), Android (APK size, startup time, frame rate, battery), Desktop (binary size, resource usage, startup time), API (response time, caching, compression). Maintain feature parity while leveraging platform strengths. Document optimization techniques and trade-offs."
- Expected output: Optimized implementations, performance improvements, documentation

**See**: [Best Practices Guide](../docs/multi-platform/best-practices.md)

### 7. API Documentation and Deployment

- Use Task tool with subagent_type="code-documentation:docs-architect"
- Prompt: "Create comprehensive API documentation including interactive OpenAPI/Swagger documentation, platform-specific integration guides, SDK examples for each platform, authentication flow diagrams, rate limiting and quota information, Postman/Insomnia collections, WebSocket connection examples, error handling best practices, API versioning strategy."
- Expected output: Complete API documentation portal, integration guides

**See**: [Deployment & Distribution Guide](../docs/multi-platform/deployment-distribution.md)

## Configuration Options

- **--platforms**: Specify target platforms (web,ios,android,desktop)
- **--api-first**: Generate API before UI implementation (default: true)
- **--shared-code**: Use Kotlin Multiplatform or similar (default: evaluate)
- **--design-system**: Use existing or create new (default: create)
- **--testing-strategy**: Unit, integration, e2e (default: all)

## Success Criteria

- API contract defined and validated before implementation
- All platforms achieve feature parity with <5% variance
- Performance metrics meet platform-specific standards:
  - Web: LCP < 2.5s, FID < 100ms, CLS < 0.1
  - iOS: Launch time < 1s, 60fps scrolling, memory < 100MB
  - Android: Startup < 1s, 60fps, APK < 20MB
  - Desktop: Launch < 1s, memory < 150MB
- Accessibility standards met (WCAG 2.2 AA minimum)
- Cross-platform testing shows consistent behavior
- Documentation complete for all platforms
- Code reuse >40% between platforms where applicable
- User experience optimized for each platform's conventions

## Platform-Specific Considerations

**Web**: PWA capabilities, SEO optimization, browser compatibility (Chrome, Firefox, Safari, Edge)
**iOS**: App Store guidelines, TestFlight distribution, iOS-specific features (Face ID, Haptics, Live Activities)
**Android**: Play Store requirements, Android App Bundles, device fragmentation, Material You
**Desktop**: Code signing, auto-updates, OS-specific installers (DMG, MSI, AppImage)

Initial feature specification: $ARGUMENTS
