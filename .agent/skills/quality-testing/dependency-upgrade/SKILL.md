---
name: dependency-upgrade
version: "1.0.7"
maturity: "5-Expert"
specialization: Package Management
description: Manage major dependency upgrades with semver analysis, compatibility matrices, staged rollouts, codemods, and testing. Use when upgrading frameworks (React, Angular, Vue), resolving peer dependency conflicts, or implementing automated updates with Renovate/Dependabot.
---

# Dependency Upgrade

Safe major version upgrades with compatibility analysis and staged rollouts.

---

## Semantic Versioning

```
MAJOR.MINOR.PATCH (e.g., 2.3.1)

^2.3.1 = >=2.3.1 <3.0.0 (minor updates allowed)
~2.3.1 = >=2.3.1 <2.4.0 (patch updates only)
2.3.1 = exact version
```

---

## Analysis Commands

| Task | npm | yarn |
|------|-----|------|
| Outdated packages | `npm outdated` | `yarn outdated` |
| Security audit | `npm audit` | `yarn audit` |
| Check updates | `npx npm-check-updates` | `npx npm-check-updates` |
| Why installed | `npm ls pkg` | `yarn why pkg` |
| Deduplicate | `npm dedupe` | `yarn dedupe` |

---

## Staged Upgrade Strategy

| Phase | Action | Validation |
|-------|--------|------------|
| 1. Plan | Review changelogs, breaking changes | Document impact |
| 2. TypeScript | Upgrade TS first | Build passes |
| 3. Framework | One major version at a time | Tests pass |
| 4. Dependencies | Update related packages | Integration tests |
| 5. Clean up | Remove unused, dedupe | Bundle size check |

```bash
# Phase 1: Check current state
npm list --depth=0
npm outdated

# Phase 2: Upgrade incrementally
npm install typescript@latest
npm test && npm run build

# Phase 3: One major version at a time
npm install react@17 react-dom@17
npm test

npm install react@18 react-dom@18
npm test
```

---

## Compatibility Matrix

```javascript
const compatibility = {
  'react': {
    '16.x': { 'react-dom': '^16.0.0', '@testing-library/react': '^11.0.0' },
    '17.x': { 'react-dom': '^17.0.0', '@testing-library/react': '^11.0.2' },
    '18.x': { 'react-dom': '^18.0.0', '@testing-library/react': '^13.0.0' }
  }
};
```

---

## Codemod Application

```bash
# React upgrade codemods
npx react-codeshift --parser tsx \
  --transform react-codeshift/transforms/rename-unsafe-lifecycles.js \
  src/

# Generic AST transforms
npx jscodeshift -t transform.js src/
```

### Custom Migration Script

```javascript
const fs = require('fs');
const glob = require('glob');

glob('src/**/*.tsx', (err, files) => {
  files.forEach(file => {
    let content = fs.readFileSync(file, 'utf8');
    // Replace deprecated APIs
    content = content.replace(/componentWillMount/g, 'UNSAFE_componentWillMount');
    fs.writeFileSync(file, content);
  });
});
```

---

## Automated Updates

### Renovate

```json
{
  "extends": ["config:base"],
  "packageRules": [
    { "matchUpdateTypes": ["minor", "patch"], "automerge": true },
    { "matchUpdateTypes": ["major"], "automerge": false, "labels": ["major"] }
  ],
  "schedule": ["before 3am on Monday"]
}
```

### Dependabot

```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

---

## Testing Strategy

| Test Type | Purpose | When |
|-----------|---------|------|
| Unit | API compatibility | After each upgrade |
| Integration | Component interaction | After related upgrades |
| Visual regression | UI unchanged | After UI library upgrades |
| E2E | User flows work | Before deploy |

```javascript
// Compatibility test
describe('React Compatibility', () => {
  it('should have matching React versions', () => {
    const react = require('react/package.json').version;
    const reactDom = require('react-dom/package.json').version;
    expect(react).toBe(reactDom);
  });
});
```

---

## Rollback Plan

```bash
#!/bin/bash
# Save state
git stash
git checkout -b upgrade-attempt

# Attempt upgrade
npm install package@latest

# Test
if npm test; then
  git commit -am "chore: upgrade package"
else
  git checkout main
  git branch -D upgrade-attempt
  npm ci  # Restore from lock file
fi
```

---

## Peer Dependency Resolution

```bash
# npm 7+: Strict peer deps
npm install --legacy-peer-deps  # Ignore conflicts
npm install --force             # Override

# Workspace upgrades
npm install pkg@latest --workspace=packages/app
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Read changelogs | Understand breaking changes |
| Upgrade incrementally | One major version at a time |
| Test after each | Unit, integration, E2E |
| Check peer deps | Resolve conflicts early |
| Use lock files | Reproducible installs |
| Automate updates | Renovate or Dependabot |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| All at once | Upgrade one package at a time |
| Skip testing | Test after each upgrade |
| Ignore peer warnings | Resolve before continuing |
| No rollback plan | Create backup branch |
| Skip major versions | Go through each (16→17→18) |

---

## Checklist

**Pre-Upgrade:**
- [ ] Review changelogs for breaking changes
- [ ] Create feature branch
- [ ] Tag current state (git tag pre-upgrade)
- [ ] Run baseline tests

**During Upgrade:**
- [ ] Upgrade one dependency at a time
- [ ] Update peer dependencies
- [ ] Run tests after each upgrade
- [ ] Apply codemods for deprecated APIs

**Post-Upgrade:**
- [ ] Full regression testing
- [ ] Check bundle size
- [ ] Deploy to staging
- [ ] Monitor for runtime errors

---

**Version**: 1.0.5
