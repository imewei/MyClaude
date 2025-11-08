---
name: dependency-upgrade
description: Manage major dependency version upgrades with semantic versioning analysis, compatibility matrix validation, staged rollout strategies, breaking change handling, automated codemod application, and comprehensive testing across npm, yarn, pnpm, pip, gem, and other package managers. Use when upgrading major framework versions (React 16→18, Angular 12→17, Vue 2→3, Next.js 12→14), updating security-vulnerable dependencies identified by npm audit or Snyk, modernizing legacy dependencies with EOL warnings, resolving peer dependency conflicts and version incompatibilities, planning incremental upgrade paths one major version at a time, implementing automated dependency update workflows with Renovate or Dependabot, testing compatibility across unit, integration, and E2E test suites, managing breaking changes through migration guides and codemods, handling transitive dependency upgrades and lock file conflicts, performing staged rollouts with feature flags and canary releases, validating semver ranges (^, ~, exact versions), auditing dependency trees for duplicates and security issues, migrating between package managers (npm to yarn, yarn to pnpm), upgrading build tools and bundlers (Webpack 4→5, Vite 2→4), updating TypeScript compiler and type definitions, managing workspace and monorepo dependency upgrades. Apply when working with package.json, package-lock.json, yarn.lock, pnpm-lock.yaml, requirements.txt, Gemfile.lock, composer.json files, CI/CD configuration for dependency updates, migration scripts for automated API updates, compatibility test suites, and when planning risk-mitigated upgrade strategies for production applications.
---

# Dependency Upgrade

Master major dependency version upgrades, compatibility analysis, staged upgrade strategies, and comprehensive testing approaches.

## When to Use This Skill

- When upgrading major framework versions one version at a time (React 16→17→18, Angular 12→15→17, Vue 2→3)
- When updating security-vulnerable dependencies identified by npm audit, yarn audit, or Snyk security scans
- When resolving peer dependency conflicts and version incompatibilities in package.json
- When planning incremental upgrade paths using compatibility matrices for framework ecosystems
- When applying automated codemods for breaking change migrations (react-codeshift, jscodeshift, ast-grep)
- When implementing Renovate or Dependabot configurations for automated PR-based dependency updates
- When testing dependency upgrades across unit tests, integration tests, E2E tests, and visual regression tests
- When handling breaking changes documented in CHANGELOG.md and MIGRATION.md files
- When working with package.json, package-lock.json, yarn.lock, pnpm-lock.yaml lock files
- When auditing dependency trees using npm ls, yarn why, pnpm why to find duplicate packages
- When deduplicating packages using npm dedupe or yarn dedupe
- When checking for outdated packages with npm outdated, yarn outdated, or npx npm-check-updates
- When upgrading TypeScript and @types/* packages while maintaining type compatibility
- When migrating from npm to yarn, yarn to pnpm, or managing workspace dependencies in monorepos
- When updating build tools (Webpack, Vite, Rollup, esbuild, Turbopack) and bundler configurations
- When handling transitive dependency updates that affect peer dependency requirements
- When implementing staged upgrade strategies with feature flags for gradual rollout
- When creating custom migration scripts for automated API transformations
- When validating semantic versioning ranges (^1.0.2 for minor updates, ~1.0.2 for patches, 1.0.2 for exact)
- When setting up CI/CD pipelines to run dependency audits and automated updates
- When managing workspace package updates in monorepo structures (npm workspaces, yarn workspaces, pnpm workspaces, Lerna, Nx)
- When upgrading testing libraries (@testing-library/react, Jest, Vitest, Cypress, Playwright)
- When updating linting and formatting tools (ESLint, Prettier, Biome) and their configurations
- When migrating deprecated APIs to new APIs using find-and-replace or AST transformation tools
- When implementing rollback procedures for failed dependency upgrades
- When documenting upgrade procedures and maintaining upgrade logs for team knowledge sharing

## Semantic Versioning Review

```
MAJOR.MINOR.PATCH (e.g., 2.3.1)

MAJOR: Breaking changes
MINOR: New features, backward compatible
PATCH: Bug fixes, backward compatible

^2.3.1 = >=2.3.1 <3.0.0 (minor updates)
~2.3.1 = >=2.3.1 <2.4.0 (patch updates)
2.3.1 = exact version
```

## Dependency Analysis

### Audit Dependencies
```bash
# npm
npm outdated
npm audit
npm audit fix

# yarn
yarn outdated
yarn audit

# Check for major updates
npx npm-check-updates
npx npm-check-updates -u  # Update package.json
```

### Analyze Dependency Tree
```bash
# See why a package is installed
npm ls package-name
yarn why package-name

# Find duplicate packages
npm dedupe
yarn dedupe

# Visualize dependencies
npx madge --image graph.png src/
```

## Compatibility Matrix

```javascript
// compatibility-matrix.js
const compatibilityMatrix = {
  'react': {
    '16.x': {
      'react-dom': '^16.0.0',
      'react-router-dom': '^5.0.0',
      '@testing-library/react': '^11.0.0'
    },
    '17.x': {
      'react-dom': '^17.0.0',
      'react-router-dom': '^5.0.0 || ^6.0.0',
      '@testing-library/react': '^11.0.2'
    },
    '18.x': {
      'react-dom': '^18.0.0',
      'react-router-dom': '^6.0.0',
      '@testing-library/react': '^13.0.0'
    }
  }
};

function checkCompatibility(packages) {
  // Validate package versions against matrix
}
```

## Staged Upgrade Strategy

### Phase 1: Planning
```bash
# 1. Identify current versions
npm list --depth=0

# 2. Check for breaking changes
# Read CHANGELOG.md and MIGRATION.md

# 3. Create upgrade plan
echo "Upgrade order:
1. TypeScript
2. React
3. React Router
4. Testing libraries
5. Build tools" > UPGRADE_PLAN.md
```

### Phase 2: Incremental Updates
```bash
# Don't upgrade everything at once!

# Step 1: Update TypeScript
npm install typescript@latest

# Test
npm run test
npm run build

# Step 2: Update React (one major version at a time)
npm install react@17 react-dom@17

# Test again
npm run test

# Step 3: Continue with other packages
npm install react-router-dom@6

# And so on...
```

### Phase 3: Validation
```javascript
// tests/compatibility.test.js
describe('Dependency Compatibility', () => {
  it('should have compatible React versions', () => {
    const reactVersion = require('react/package.json').version;
    const reactDomVersion = require('react-dom/package.json').version;

    expect(reactVersion).toBe(reactDomVersion);
  });

  it('should not have peer dependency warnings', () => {
    // Run npm ls and check for warnings
  });
});
```

## Breaking Change Handling

### Identifying Breaking Changes
```bash
# Use changelog parsers
npx changelog-parser react 16.0.0 17.0.0

# Or manually check
curl https://raw.githubusercontent.com/facebook/react/main/CHANGELOG.md
```

### Codemod for Automated Fixes
```bash
# React upgrade codemods
npx react-codeshift <transform> <path>

# Example: Update lifecycle methods
npx react-codeshift \
  --parser tsx \
  --transform react-codeshift/transforms/rename-unsafe-lifecycles.js \
  src/
```

### Custom Migration Script
```javascript
// migration-script.js
const fs = require('fs');
const glob = require('glob');

glob('src/**/*.tsx', (err, files) => {
  files.forEach(file => {
    let content = fs.readFileSync(file, 'utf8');

    // Replace old API with new API
    content = content.replace(
      /componentWillMount/g,
      'UNSAFE_componentWillMount'
    );

    // Update imports
    content = content.replace(
      /import { Component } from 'react'/g,
      "import React, { Component } from 'react'"
    );

    fs.writeFileSync(file, content);
  });
});
```

## Testing Strategy

### Unit Tests
```javascript
// Ensure tests pass before and after upgrade
npm run test

// Update test utilities if needed
npm install @testing-library/react@latest
```

### Integration Tests
```javascript
// tests/integration/app.test.js
describe('App Integration', () => {
  it('should render without crashing', () => {
    render(<App />);
  });

  it('should handle navigation', () => {
    const { getByText } = render(<App />);
    fireEvent.click(getByText('Navigate'));
    expect(screen.getByText('New Page')).toBeInTheDocument();
  });
});
```

### Visual Regression Tests
```javascript
// visual-regression.test.js
describe('Visual Regression', () => {
  it('should match snapshot', () => {
    const { container } = render(<App />);
    expect(container.firstChild).toMatchSnapshot();
  });
});
```

### E2E Tests
```javascript
// cypress/e2e/app.cy.js
describe('E2E Tests', () => {
  it('should complete user flow', () => {
    cy.visit('/');
    cy.get('[data-testid="login"]').click();
    cy.get('input[name="email"]').type('user@example.com');
    cy.get('button[type="submit"]').click();
    cy.url().should('include', '/dashboard');
  });
});
```

## Automated Dependency Updates

### Renovate Configuration
```json
// renovate.json
{
  "extends": ["config:base"],
  "packageRules": [
    {
      "matchUpdateTypes": ["minor", "patch"],
      "automerge": true
    },
    {
      "matchUpdateTypes": ["major"],
      "automerge": false,
      "labels": ["major-update"]
    }
  ],
  "schedule": ["before 3am on Monday"],
  "timezone": "America/New_York"
}
```

### Dependabot Configuration
```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    reviewers:
      - "team-leads"
    commit-message:
      prefix: "chore"
      include: "scope"
```

## Rollback Plan

```javascript
// rollback.sh
#!/bin/bash

# Save current state
git stash
git checkout -b upgrade-branch

# Attempt upgrade
npm install package@latest

# Run tests
if npm run test; then
  echo "Upgrade successful"
  git add package.json package-lock.json
  git commit -m "chore: upgrade package"
else
  echo "Upgrade failed, rolling back"
  git checkout main
  git branch -D upgrade-branch
  npm install  # Restore from package-lock.json
fi
```

## Common Upgrade Patterns

### Lock File Management
```bash
# npm
npm install --package-lock-only  # Update lock file only
npm ci  # Clean install from lock file

# yarn
yarn install --frozen-lockfile  # CI mode
yarn upgrade-interactive  # Interactive upgrades
```

### Peer Dependency Resolution
```bash
# npm 7+: strict peer dependencies
npm install --legacy-peer-deps  # Ignore peer deps

# npm 8+: override peer dependencies
npm install --force
```

### Workspace Upgrades
```bash
# Update all workspace packages
npm install --workspaces

# Update specific workspace
npm install package@latest --workspace=packages/app
```

## Resources

- **references/semver.md**: Semantic versioning guide
- **references/compatibility-matrix.md**: Common compatibility issues
- **references/staged-upgrades.md**: Incremental upgrade strategies
- **references/testing-strategy.md**: Comprehensive testing approaches
- **assets/upgrade-checklist.md**: Step-by-step checklist
- **assets/compatibility-matrix.csv**: Version compatibility table
- **scripts/audit-dependencies.sh**: Dependency audit script

## Best Practices

1. **Read Changelogs**: Understand what changed
2. **Upgrade Incrementally**: One major version at a time
3. **Test Thoroughly**: Unit, integration, E2E tests
4. **Check Peer Dependencies**: Resolve conflicts early
5. **Use Lock Files**: Ensure reproducible installs
6. **Automate Updates**: Use Renovate or Dependabot
7. **Monitor**: Watch for runtime errors post-upgrade
8. **Document**: Keep upgrade notes

## Upgrade Checklist

```markdown
Pre-Upgrade:
- [ ] Review current dependency versions
- [ ] Read changelogs for breaking changes
- [ ] Create feature branch
- [ ] Backup current state (git tag)
- [ ] Run full test suite (baseline)

During Upgrade:
- [ ] Upgrade one dependency at a time
- [ ] Update peer dependencies
- [ ] Fix TypeScript errors
- [ ] Update tests if needed
- [ ] Run test suite after each upgrade
- [ ] Check bundle size impact

Post-Upgrade:
- [ ] Full regression testing
- [ ] Performance testing
- [ ] Update documentation
- [ ] Deploy to staging
- [ ] Monitor for errors
- [ ] Deploy to production
```

## Common Pitfalls

- Upgrading all dependencies at once
- Not testing after each upgrade
- Ignoring peer dependency warnings
- Forgetting to update lock file
- Not reading breaking change notes
- Skipping major versions
- Not having rollback plan
