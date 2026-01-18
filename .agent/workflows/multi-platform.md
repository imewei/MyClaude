---
description: Build and deploy features across web, mobile, and desktop platforms with
  API-first architecture and multi-agent orchestration
triggers:
- /multi-platform
- build and deploy features
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



# Multi-Platform Feature Development

Build features consistently across web, mobile, desktop using API-first architecture and parallel implementation.

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

## Execution Mode

Use AskUserQuestion to select:
- **Quick** (3-5 days): Single platform MVP + API contract
- **Standard** (2-3 weeks): Web + Mobile (iOS/Android) + feature parity
- **Enterprise** (4-6 weeks): All platforms + shared code + optimization

## Agent Coordination

| Phase | Tasks | Mode | Duration |
|-------|-------|------|----------|
| 1. Architecture | API contracts, Design system, Shared logic | Sequential | 2-3 days |
| 2. Implementation | Web, iOS, Android, Desktop | **Parallel** | 5-15 days |
| 3. Validation | Testing, Optimization, Docs | **Parallel** | 2-5 days |

## Phase 1: Architecture (Sequential)

### 1. API Contract
Design RESTful/GraphQL API with OpenAPI 3.1 spec, shared data models, auth, rate limiting, error formats.
**See**: [Platform Architecture](../../plugins/multi-platform-apps/docs/multi-platform/platform-architecture.md#api-first-architecture)

### 2. Design System
Cross-platform components (Material, iOS HIG, Fluent), responsive layouts, accessibility (WCAG 2.2 AA), themes.
**See**: [Design Systems](../../plugins/multi-platform-apps/docs/multi-platform/design-systems.md)

### 3. Shared Logic
Core domain models, business rules, validation, state management (MVI/Redux/BLoC), caching, error handling.
**See**: [Platform Architecture](../../plugins/multi-platform-apps/docs/multi-platform/platform-architecture.md#shared-business-logic-strategies)

## Phase 2: Platform Implementation (Parallel Execution)

> **Orchestration Note**: Execute platform-specific implementation streams concurrently.

Implement per platform following shared contracts:
- **Web (React/Next.js)**: frontend-developer
- **iOS (SwiftUI)**: multi-platform-mobile
- **Android (Kotlin/Compose)**: multi-platform-mobile
- **Desktop (Electron/Tauri)**: frontend-developer

**See**: [Implementation Guides](../../plugins/multi-platform-apps/docs/multi-platform/implementation-guides.md)

## Phase 3: Validation (Parallel Execution)

> **Orchestration Note**: Execute testing, optimization, and documentation work concurrently.

### Cross-Platform Testing
Functional parity, UI consistency, performance benchmarks, accessibility, network resilience, data sync validation, platform edge cases, e2e journeys.
**See**: [Testing Strategies](../../plugins/multi-platform-apps/docs/multi-platform/testing-strategies.md)

### Platform Optimization
- Web: Bundle size, lazy loading, CDN, SEO
- iOS: App size, launch time, memory, battery
- Android: APK size, startup, frame rate, battery
- Desktop: Binary size, resource usage, startup

**See**: [Best Practices](../../plugins/multi-platform-apps/docs/multi-platform/best-practices.md)

### API Documentation
Interactive OpenAPI/Swagger, platform integration guides, SDK examples, auth flows, rate limits, collections, error handling, versioning.
**See**: [Deployment & Distribution](../../plugins/multi-platform-apps/docs/multi-platform/deployment-distribution.md)

## Configuration

- `--platforms`: Target platforms (web,ios,android,desktop)
- `--api-first`: Generate API before UI (default: true)
- `--shared-code`: Use Kotlin Multiplatform or similar (default: evaluate)
- `--design-system`: Use existing or create (default: create)
- `--testing-strategy`: Unit, integration, e2e (default: all)

## Success Criteria

- API contract defined/validated before implementation
- Feature parity <5% variance
- Performance targets:
  - Web: LCP <2.5s, FID <100ms, CLS <0.1
  - iOS: Launch <1s, 60fps, memory <100MB
  - Android: Startup <1s, 60fps, APK <20MB
  - Desktop: Launch <1s, memory <150MB
- Accessibility WCAG 2.2 AA
- Cross-platform testing consistent
- Documentation complete
- Code reuse >40%
- UX optimized per platform

## Platform Considerations

- **Web**: PWA, SEO, browser compatibility
- **iOS**: App Store, TestFlight, Face ID, Haptics, Live Activities
- **Android**: Play Store, App Bundles, fragmentation, Material You
- **Desktop**: Code signing, auto-updates, installers (DMG, MSI, AppImage)

$ARGUMENTS
