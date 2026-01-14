---
name: multi-platform-architecture
version: "1.0.7"
maturity: "5-Expert"
specialization: Cross-Platform Strategy
description: Design multi-platform apps with clean architecture, code sharing (Flutter, React Native, KMP), BFF patterns, offline-first sync, and team organization. Use when planning mobile/web/desktop architecture, choosing frameworks, or maximizing code reuse.
---

# Multi-Platform Architecture

Strategic patterns for mobile, web, and desktop with maximum code sharing.

---

## Strategy Selection

| Strategy | Code Sharing | Performance | Time to Market | Use Case |
|----------|--------------|-------------|----------------|----------|
| Native | 0% | Excellent | Slow | Gaming, AR/VR |
| Flutter | 85-95% | Good | Fast | Most apps |
| React Native | 70-80% | Good | Fast | JS teams |
| KMP | 60-80% | Excellent | Moderate | Enterprise |
| Web + PWA | 95% | Moderate | Very Fast | Content apps |

---

## Clean Architecture Layers

```
┌─────────────────────────────────────┐
│   Presentation (Platform-specific)  │  iOS/Android/Web UI
├─────────────────────────────────────┤
│   Application (Shared ViewModels)   │  Platform-agnostic logic
├─────────────────────────────────────┤
│   Domain (Shared Business Logic)    │  Entities, Use Cases
├─────────────────────────────────────┤
│   Data (Shared + Platform-specific) │  API, DB, Cache
└─────────────────────────────────────┘
```

---

## Flutter Code Sharing

```dart
// lib/
// ├── core/            # 100% shared
// ├── features/        # 90% shared
// └── platform/        # Platform-specific

// Adaptive UI
Widget build(BuildContext context) {
  if (Platform.isIOS || Platform.isAndroid) return MobileLayout();
  if (kIsWeb) return WebLayout();
  return DesktopLayout();
}

// Platform channel for native features
const platform = MethodChannel('com.example.app/native');
Future<int> getBatteryLevel() async {
  return await platform.invokeMethod('getBatteryLevel');
}
```

---

## React Native + Web Monorepo

```typescript
// packages/
// ├── shared/    # 100% shared (API, state, types)
// ├── mobile/    # React Native
// └── web/       # React Web

// shared/components/Button.tsx (interface)
export interface ButtonProps {
  title: string;
  onPress: () => void;
}

// mobile/components/Button.tsx
import { TouchableOpacity, Text } from 'react-native';
export const Button: FC<ButtonProps> = ({ title, onPress }) => (
  <TouchableOpacity onPress={onPress}><Text>{title}</Text></TouchableOpacity>
);

// web/components/Button.tsx
export const Button: FC<ButtonProps> = ({ title, onPress }) => (
  <button onClick={onPress}>{title}</button>
);
```

---

## Backend for Frontend (BFF)

```
iOS App ──→ iOS BFF ──┐
Web App ──→ Web BFF ──┼──→ Core API
Android ──→ Droid BFF ┘
```

**Benefits**: Optimized payloads, platform-specific features, independent deployment

---

## Offline-First Sync

```typescript
class SyncManager {
  async syncChanges() {
    const pending = await this.getPendingOperations();
    for (const op of pending) {
      try {
        await this.executeOperation(op);
        await this.markComplete(op.id);
      } catch (error) {
        if (error.isNetworkError) await this.requeueOperation(op);
        else await this.handleConflict(op, error);  // Last-write-wins or merge
      }
    }
  }
}
```

---

## Real-Time Updates

```typescript
class RealtimeManager {
  connect() {
    try {
      this.ws = new WebSocket('wss://api.example.com/ws');
      this.ws.onmessage = this.handleMessage;
    } catch {
      this.startPolling();  // Fallback to polling
    }
  }
}
```

---

## Team Organization

| Structure | Pros | Cons |
|-----------|------|------|
| Feature Teams | Fast delivery, end-to-end ownership | Knowledge duplication |
| Platform Teams | Deep expertise | Coordination overhead |

---

## Decision Matrix

**Choose Flutter if**: Fast time to market, Dart team, UI consistency priority

**Choose Native if**: Performance critical, platform-specific features, long-term investment

**Choose Web + PWA if**: Content-focused, frequent updates, discoverability important

---

## Anti-Patterns

```typescript
// ❌ Bad: Over-parameterized component
<Component mobileLayout={true} webLayout={false} /* 50+ props */ />

// ✅ Good: Platform-optimized components
{Platform.OS === 'web' ? <WebComponent /> : <MobileComponent />}
```

---

## Code Sharing Targets

| Framework | Shared Code |
|-----------|-------------|
| Flutter | 85-95% |
| React Native | 70-80% |
| KMP | 60-80% |
| Native + Shared Logic | 40-60% |

---

## Checklist

- [ ] Platform strategy defined
- [ ] Code sharing boundaries established
- [ ] Clean architecture layers separated
- [ ] Offline-first if needed
- [ ] BFF for complex APIs
- [ ] CI/CD per platform
- [ ] Team structure aligned

---

**Version**: 1.0.5
