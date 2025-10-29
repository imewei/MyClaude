# Multi-Platform Architecture Patterns

> **Strategic patterns for building scalable applications across mobile, web, and desktop platforms.**

---

## Skill Overview

Master the architectural patterns and decisions required for successful multi-platform development, including code sharing strategies, platform-specific optimization, and team organization.

**Target Audience**: Technical leads, architects, and teams planning multi-platform projects

**Estimated Learning Time**: 6-8 hours to master concepts

---

## Architecture Decision Framework

### 1. Platform Strategy Matrix

| Strategy | Code Sharing | Performance | Time to Market | Use Case |
|----------|--------------|-------------|----------------|----------|
| **Native** | 0% | Excellent | Slow | Gaming, AR/VR |
| **Hybrid (React Native/Flutter)** | 70-90% | Good | Fast | Most apps |
| **Web + PWA** | 95% | Moderate | Very Fast | Content apps |
| **Cross-compile (KMP)** | 60-80% | Excellent | Moderate | Enterprise |

### 2. Architecture Patterns

#### Clean Architecture for Multi-Platform

```
┌─────────────────────────────────────┐
│         Presentation Layer          │
│  ┌─────────┬─────────┬───────────┐ │
│  │   iOS   │   Web   │  Android  │ │
│  │ SwiftUI │  React  │  Compose  │ │
│  └─────────┴─────────┴───────────┘ │
├─────────────────────────────────────┤
│         Application Layer           │
│  ┌──────────────────────────────┐  │
│  │  ViewModels / Presenters     │  │
│  │  (Platform-Agnostic Logic)   │  │
│  └──────────────────────────────┘  │
├─────────────────────────────────────┤
│           Domain Layer              │
│  ┌──────────────────────────────┐  │
│  │  Entities  │  Use Cases      │  │
│  │  Business Logic (Pure)       │  │
│  └──────────────────────────────┘  │
├─────────────────────────────────────┤
│            Data Layer               │
│  ┌──────────────────────────────┐  │
│  │  Repositories │ Data Sources │  │
│  │  (API, DB, Cache)            │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## Code Sharing Strategies

### 1. Flutter Multi-Platform

**Shared Code: 85-95%**

```dart
// Shared business logic and UI
lib/
├── core/                    # 100% shared
│   ├── domain/
│   ├── data/
│   └── utils/
├── features/                # 90% shared
│   └── user/
│       ├── domain/
│       ├── data/
│       └── presentation/
│           ├── widgets/      # Mostly shared
│           └── pages/
└── platform/                # Platform-specific
    ├── mobile/
    ├── web/
    └── desktop/
```

**Platform-Specific Optimization:**

```dart
// Adaptive UI based on platform
Widget build(BuildContext context) {
  if (Platform.isIOS || Platform.isAndroid) {
    return MobileLayout();
  } else if (kIsWeb) {
    return WebLayout();
  } else {
    return DesktopLayout();
  }
}

// Platform channels for native features
const platform = MethodChannel('com.example.app/battery');

Future<int> getBatteryLevel() async {
  final int batteryLevel = await platform.invokeMethod('getBatteryLevel');
  return batteryLevel;
}
```

### 2. React Native + React Web

**Shared Code: 70-80%**

```typescript
// Monorepo structure
packages/
├── shared/                  # 100% shared
│   ├── api/
│   ├── state/
│   ├── types/
│   └── utils/
├── mobile/                  # React Native
│   ├── src/
│   │   ├── components/      # 40% shared
│   │   └── screens/
│   └── package.json
└── web/                     # React Web
    ├── src/
    │   ├── components/      # 40% shared
    │   └── pages/
    └── package.json
```

**Shared Components Pattern:**

```typescript
// shared/components/Button.tsx
export interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary';
}

// mobile/components/Button.tsx (React Native)
import { TouchableOpacity, Text } from 'react-native';
import { ButtonProps } from '@shared/components/Button';

export const Button: React.FC<ButtonProps> = ({ title, onPress, variant }) => {
  return (
    <TouchableOpacity onPress={onPress} style={styles[variant]}>
      <Text>{title}</Text>
    </TouchableOpacity>
  );
};

// web/components/Button.tsx (React Web)
import { ButtonProps } from '@shared/components/Button';

export const Button: React.FC<ButtonProps> = ({ title, onPress, variant }) => {
  return (
    <button onClick={onPress} className={`btn btn-${variant}`}>
      {title}
    </button>
  );
};
```

### 3. Native with Shared Backend

**Shared Code: 0% (UI), 100% (Backend)**

```
project/
├── backend/                 # 100% shared
│   ├── api/
│   ├── services/
│   └── database/
├── ios/                     # Platform-specific
│   └── MyApp/
│       ├── ViewModels/      # Can share logic patterns
│       └── Views/
└── android/                 # Platform-specific
    └── app/
        ├── ViewModels/      # Can share logic patterns
        └── Views/
```

---

## Backend for Frontend (BFF) Pattern

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│   iOS   │    │   Web   │    │ Android │
└────┬────┘    └────┬────┘    └────┬────┘
     │              │              │
     ├──────────────┼──────────────┤
     │              │              │
┌────▼──────┐  ┌───▼─────┐  ┌────▼──────┐
│ iOS BFF   │  │ Web BFF │  │ Droid BFF │
└────┬──────┘  └───┬─────┘  └────┬──────┘
     │              │              │
     └──────────────┼──────────────┘
                    │
            ┌───────▼────────┐
            │  Core API      │
            │  Services      │
            └────────────────┘
```

**Benefits:**
- Optimize payload for each platform
- Platform-specific features
- Independent deployment
- Reduced coupling

---

## Data Synchronization Strategies

### 1. Offline-First Architecture

```typescript
// Sync Manager Pattern
class SyncManager {
  private queue: SyncOperation[] = [];

  async syncChanges() {
    // Get pending operations
    const pending = await this.getPendingOperations();

    for (const operation of pending) {
      try {
        // Attempt sync
        await this.executeOperation(operation);
        await this.markComplete(operation.id);
      } catch (error) {
        if (error.isNetworkError) {
          // Retry later
          await this.requeueOperation(operation);
        } else {
          // Conflict resolution needed
          await this.handleConflict(operation, error);
        }
      }
    }
  }

  private async handleConflict(
    operation: SyncOperation,
    error: ConflictError
  ) {
    // Last-write-wins strategy
    if (operation.timestamp > error.serverTimestamp) {
      await this.forceUpdate(operation);
    } else {
      await this.mergeChanges(operation, error.serverData);
    }
  }
}
```

### 2. Real-Time Updates

```typescript
// WebSocket with Fallback
class RealtimeManager {
  private ws: WebSocket | null = null;
  private fallbackTimer: NodeJS.Timer | null = null;

  connect() {
    try {
      this.ws = new WebSocket('wss://api.example.com/ws');
      this.ws.onmessage = this.handleMessage;
      this.ws.onerror = this.handleError;
    } catch (error) {
      // Fallback to polling
      this.startPolling();
    }
  }

  private startPolling() {
    this.fallbackTimer = setInterval(async () => {
      await this.fetchUpdates();
    }, 5000); // Poll every 5 seconds
  }
}
```

---

## Performance Optimization Strategies

### 1. Code Splitting by Platform

```typescript
// Dynamic imports for web
const loadPlatformModule = async () => {
  if (Platform.OS === 'web') {
    return await import('./WebSpecificFeature');
  } else {
    return await import('./MobileSpecificFeature');
  }
};
```

### 2. Asset Optimization

```yaml
# Assets per platform
assets/
├── icons/
│   ├── ios/          # @1x, @2x, @3x
│   ├── android/      # mdpi, hdpi, xhdpi, xxhdpi
│   └── web/          # SVG preferred
└── images/
    ├── mobile/       # Optimized for mobile
    └── web/          # WebP with fallback
```

---

## Team Organization

### 1. Feature Teams (Recommended)

```
Feature Team: User Profile
├── iOS Developer
├── Android Developer
├── Web Developer
├── Backend Developer
└── QA Engineer

✅ Pros: Fast delivery, end-to-end ownership
❌ Cons: Platform knowledge duplication
```

### 2. Platform Teams

```
iOS Team          Android Team      Web Team
├── Dev 1         ├── Dev 1         ├── Dev 1
├── Dev 2         ├── Dev 2         ├── Dev 2
└── Dev 3         └── Dev 3         └── Dev 3

✅ Pros: Deep platform expertise
❌ Cons: Coordination overhead
```

---

## Decision Matrix

### When to Choose Each Approach

**Choose Flutter/React Native if:**
- Need fast time to market
- Team skilled in Dart/JavaScript
- UI consistency is priority
- Budget is limited

**Choose Native (Swift/Kotlin) if:**
- Performance is critical
- Platform-specific features needed
- Long-term investment
- Large team with platform expertise

**Choose Web + PWA if:**
- Content-focused app
- Frequent updates needed
- Discoverability important
- Limited native features needed

---

## Anti-Patterns

### ❌ Don't: One size fits all

```typescript
// Bad: Same component for all platforms
<ComplexComponent
  mobileLayout={true}
  webLayout={false}
  desktopLayout={false}
  // 50+ props for different platforms
/>
```

### ✅ Do: Platform-optimized components

```typescript
// Good: Separate optimized components
{Platform.OS === 'web' ? (
  <WebOptimizedComponent />
) : (
  <MobileOptimizedComponent />
)}
```

---

## Quick Reference

### Code Sharing Targets

- **Flutter**: 85-95% shared code
- **React Native**: 70-80% shared code
- **Native + Shared Logic**: 40-60% shared business logic
- **Kotlin Multiplatform**: 60-80% shared code

### Architecture Checklist

- [ ] Define platform strategy upfront
- [ ] Establish code sharing boundaries
- [ ] Plan offline-first if needed
- [ ] Design BFF pattern for complex APIs
- [ ] Implement conflict resolution
- [ ] Set up CI/CD per platform
- [ ] Define team structure
- [ ] Plan performance monitoring

---

**Skill Version**: 1.0.0
**Last Updated**: October 27, 2024
**Difficulty**: Advanced
**Estimated Time**: 6-8 hours
