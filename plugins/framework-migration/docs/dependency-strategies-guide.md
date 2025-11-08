# Dependency Upgrade Strategies Guide

**Version:** 1.0.3 | **Category:** framework-migration | **Type:** Reference

Comprehensive guide for safe dependency upgrades, breaking change management, and version resolution strategies.

---

## Table of Contents

1. [Upgrade Strategy Selection](#upgrade-strategy-selection)
2. [Semantic Versioning Analysis](#semantic-versioning-analysis)
3. [Breaking Change Catalog](#breaking-change-catalog)
4. [Compatibility Matrix](#compatibility-matrix)
5. [Testing Strategies](#testing-strategies)
6. [Rollback Procedures](#rollback-procedures)

---

## Upgrade Strategy Selection

### Decision Tree

```
Is this a security vulnerability?
├─ Yes → IMMEDIATE UPGRADE (security-first strategy)
└─ No → Is it a major version bump?
    ├─ Yes → INCREMENTAL UPGRADE (one major version at a time)
    └─ No → Is it part of core dependencies?
        ├─ Yes → CAUTIOUS UPGRADE (test thoroughly)
        └─ No → BATCH UPGRADE (group with similar packages)
```

### Strategy Types

**1. Security-First Strategy**
- **Priority**: Patch CVEs immediately
- **Approach**: Upgrade vulnerable packages first, regardless of version
- **Testing**: Smoke tests + security validation
- **Timeline**: Same day for critical, 1 week for high

**2. Incremental Strategy**
- **Priority**: Minimize risk through small steps
- **Approach**: Upgrade one major version at a time
- **Testing**: Full regression suite between versions
- **Timeline**: 1-2 weeks per major version

**3. Batch Strategy**
- **Priority**: Efficiency for low-risk updates
- **Approach**: Group similar patch/minor updates
- **Testing**: Standard test suite
- **Timeline**: 1 sprint cycle

**4. Framework-First Strategy**
- **Priority**: Core framework drives ecosystem
- **Approach**: Upgrade main framework, then dependent packages
- **Testing**: Integration tests + E2E
- **Timeline**: 2-4 weeks for major framework upgrade

---

## Semantic Versioning Analysis

### Version Format: `MAJOR.MINOR.PATCH`

**PATCH (1.2.3 → 1.2.4)**:
- **Changes**: Bug fixes only
- **Compatibility**: 100% backward compatible
- **Risk**: Very Low
- **Testing**: Smoke tests
- **Strategy**: Safe to batch upgrade

**MINOR (1.2.4 → 1.3.0)**:
- **Changes**: New features, backward compatible
- **Compatibility**: 99% backward compatible (deprecations may appear)
- **Risk**: Low to Medium
- **Testing**: Feature tests + regression
- **Strategy**: Review changelog, test new features

**MAJOR (1.3.0 → 1.0.2)**:
- **Changes**: Breaking changes, API changes
- **Compatibility**: Not backward compatible
- **Risk**: High
- **Testing**: Full test suite + migration validation
- **Strategy**: Incremental, read migration guide

### Pre-Release Versions

- **Alpha (1.0.2-alpha.1)**: Early development, unstable
- **Beta (1.0.2-beta.1)**: Feature complete, testing phase
- **RC (1.0.2-rc.1)**: Release candidate, near production

**Recommendation**: Avoid pre-releases in production unless necessary for critical features.

---

## Breaking Change Catalog

### React Breaking Changes

**React 15 → 16**:
- PropTypes moved to separate `prop-types` package
- `React.createClass` deprecated (use ES6 classes or functions)
- String refs deprecated (use callback refs)
- Hydration behavior changed

**React 16 → 17**:
- Event delegation moved from document to root
- No event pooling (events no longer reused)
- `useEffect` cleanup timing changed
- JSX transform changed (new JSX runtime)

**React 17 → 18**:
- Automatic batching for all updates
- Concurrent features (Suspense, transitions)
- Stricter StrictMode checks
- New root API: `createRoot()`

### Node.js Breaking Changes

**Node 12 → 14**:
- V8 8.1 → 8.4 (performance improvements)
- Optional chaining and nullish coalescing supported
- Diagnostic reports stable
- WASI experimental support

**Node 14 → 16**:
- V8 9.0 (Atomics.waitAsync)
- Apple Silicon support
- npm 7 (peer dependencies auto-install)
- Timers Promises API

**Node 16 → 18**:
- V8 10.1
- Fetch API built-in (experimental)
- Test runner built-in
- Global `JSON.parse` improved

### Python Breaking Changes

**Python 3.6 → 3.7**:
- `async` and `await` are keywords
- Dataclasses introduced
- Context variables
- Performance improvements (method calls 20% faster)

**Python 3.7 → 3.9**:
- Dictionary merge operator `|`
- Type hinting generics (list[int] instead of List[int])
- zoneinfo for timezone handling
- String `removeprefix()` and `removesuffix()`

---

## Compatibility Matrix

### Example: React Ecosystem Compatibility

| Package | React 16 | React 17 | React 18 |
|---------|----------|----------|----------|
| react-router v5 | ✅ | ✅ | ✅ |
| react-router v6 | ❌ | ✅ | ✅ |
| Redux v4 | ✅ | ✅ | ✅ |
| React Query v3 | ✅ | ✅ | ⚠️ (partial) |
| React Query v4 | ❌ | ✅ | ✅ |
| Material-UI v4 | ✅ | ✅ | ⚠️ (issues) |
| MUI v5 | ❌ | ✅ | ✅ |

**Legend**:
- ✅ Fully compatible
- ⚠️ Partially compatible (workarounds needed)
- ❌ Not compatible

### Peer Dependency Resolution

**Example**: Upgrading React 17 → 18

```json
{
  "dependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "peerDependencies": {
    "react": "^17.0.0 || ^18.0.0"  // Flexible range
  }
}
```

**Strategy**:
1. Identify packages with peer dependency conflicts
2. Check if package has React 18 compatible version
3. Upgrade conflicting packages in same PR
4. Use npm overrides/resolutions if needed temporarily

---

## Testing Strategies

### Pre-Upgrade Testing

**Baseline Capture**:
```bash
# Capture current behavior
npm test -- --coverage --json > baseline-tests.json
npm run build -- --stats > baseline-build.json
npm run perf-test > baseline-perf.json
```

### Post-Upgrade Testing

**Comparison Testing**:
```bash
# After upgrade
npm test -- --coverage --json > upgraded-tests.json
npm run build -- --stats > upgraded-build.json
npm run perf-test > upgraded-perf.json

# Compare
diff baseline-tests.json upgraded-tests.json
```

### Progressive Testing Levels

**Level 1: Smoke Tests** (5 minutes):
- Application starts without errors
- Critical paths functional (login, core features)
- No console errors

**Level 2: Regression Tests** (30 minutes):
- Full test suite passes
- Visual regression tests (Chromatic, Percy)
- Integration tests pass

**Level 3: Performance Tests** (1 hour):
- Load testing (no degradation)
- Memory profiling (no leaks)
- Bundle size analysis (acceptable growth)

---

## Rollback Procedures

### Immediate Rollback

**Git-Based Rollback**:
```bash
# Create checkpoint before upgrade
git tag pre-upgrade-$(date +%Y%m%d)

# If issues detected, rollback immediately
git revert HEAD
npm ci
npm test
```

### Incremental Rollback

**Feature-Flag Rollback** (for gradual upgrades):
```javascript
// Use old implementation if new version has issues
const useNewImplementation = featureFlags.isEnabled('use-react-18');

if (useNewImplementation) {
  return <NewReactComponent />;
} else {
  return <LegacyReactComponent />;
}
```

### Deployment Rollback

**Blue-Green Deployment**:
- Keep old version (blue) running
- Deploy new version (green)
- Switch traffic to green
- If issues: instant switch back to blue

**Canary Deployment**:
- Route 10% traffic to new version
- Monitor error rates
- If error rate > threshold: rollback
- Gradually increase if stable (10% → 50% → 100%)

---

**For dependency upgrade workflows**, see `/deps-upgrade` command.
