---
name: monorepo-management
description: Master monorepo management with Turborepo, Nx, pnpm workspaces, Yarn workspaces, and npm workspaces to build efficient, scalable multi-package repositories with optimized builds, intelligent caching, and dependency management. Use when setting up or configuring monorepo infrastructure files (turbo.json, nx.json, pnpm-workspace.yaml, lerna.json), when creating or organizing workspace packages in apps/* and packages/* directories, when optimizing build performance with task pipelines and caching strategies, when managing shared dependencies across multiple packages, when implementing code sharing patterns for UI components, utilities, types, and configurations, when setting up CI/CD pipelines for monorepos with affected builds and deployments, when configuring shared tooling including TypeScript configs, ESLint presets, and build configurations, when implementing versioning and publishing strategies with changesets or Lerna, when debugging monorepo-specific issues like circular dependencies and phantom dependencies, when migrating from multi-repo (polyrepo) to monorepo architecture, when creating workspace packages with proper exports and package.json configuration, when optimizing dependency hoisting and installation with pnpm shamefully-hoist or strict-peer-dependencies, when setting up remote caching for team collaboration and CI/CD, when implementing monorepo filtering and task running with --filter or affected commands, when debugging build performance and cache misses, when establishing monorepo conventions for package naming and directory structure, or when scaling monorepos for large teams with hundreds of packages and applications.
---

# Monorepo Management

Build efficient, scalable monorepos that enable code sharing, consistent tooling, and atomic changes across multiple packages and applications.

## When to Use This Skill

### Monorepo Configuration Files
- Creating or editing `turbo.json` for Turborepo pipeline configuration
- Setting up `nx.json` for Nx workspace configuration and caching
- Configuring `pnpm-workspace.yaml` for pnpm workspace definitions
- Managing `.npmrc` for pnpm-specific settings (shamefully-hoist, strict-peer-dependencies)
- Editing `lerna.json` for Lerna configuration and versioning
- Setting up root `package.json` with workspace definitions and scripts

### Workspace Structure and Organization
- Creating workspace packages in `apps/*` directory (web apps, mobile apps, documentation sites)
- Setting up shared packages in `packages/*` directory (UI components, utilities, configs)
- Organizing packages with proper naming conventions (`@repo/ui`, `@company/shared-utils`)
- Implementing barrel exports with `index.ts` or `index.js` for clean package APIs
- Structuring monorepo for multiple teams and domain boundaries

### Package Configuration
- Creating package-level `package.json` with proper `name`, `version`, `main`, `types`, `exports`
- Setting up `exports` field for subpath exports and conditional exports
- Configuring workspace dependencies with `workspace:*` protocol (pnpm)
- Managing peer dependencies and dev dependencies at package level
- Implementing package build scripts with `build`, `dev`, `lint`, `test`

### Build System Setup and Optimization
- Configuring Turborepo task pipelines with `dependsOn`, `outputs`, and `inputs`
- Setting up Nx target defaults for build, test, lint with caching
- Implementing build caching strategies to skip unnecessary builds
- Configuring remote caching for team collaboration (Turborepo Remote Cache, Nx Cloud)
- Optimizing task execution order and parallelization
- Setting up incremental builds with proper cache invalidation

### Dependency Management
- Installing dependencies in specific packages: `pnpm add react --filter @repo/ui`
- Managing workspace dependencies and inter-package references
- Hoisting shared dependencies to root `node_modules`
- Resolving phantom dependencies (using deps not declared in package.json)
- Updating all packages with `pnpm update -r`
- Removing duplicate dependencies across workspace

### Code Sharing Patterns
- Creating shared UI component libraries (`packages/ui`)
- Setting up shared TypeScript configurations (`packages/tsconfig`)
- Implementing shared ESLint and Prettier configurations (`packages/config`)
- Creating shared utility libraries with tree-shakeable exports
- Sharing types and interfaces across frontend and backend packages
- Building design system packages with reusable components

### Build and Development Scripts
- Running tasks in specific packages: `pnpm --filter web dev`
- Running tasks across all packages: `pnpm -r build`
- Executing tasks in parallel: `pnpm -r --parallel dev`
- Using Turborepo commands: `turbo run build`, `turbo run test`
- Using Nx commands: `nx build my-app`, `nx affected:build`
- Filtering packages by pattern: `pnpm --filter "@repo/*" build`

### CI/CD for Monorepos
- Setting up GitHub Actions workflows for monorepo builds
- Implementing affected builds to only build changed packages
- Configuring deploy workflows for multiple apps
- Setting up test parallelization across packages
- Implementing cache restoration and saving in CI
- Using Nx affected commands in CI: `nx affected:build --base=main`

### Migration and Refactoring
- Migrating from multi-repo (polyrepo) to monorepo architecture
- Converting standalone packages to workspace packages
- Migrating from Lerna to Turborepo or Nx
- Refactoring duplicated code into shared packages
- Extracting common configurations to shared config packages

### Performance Debugging and Optimization
- Analyzing build performance with Turborepo `--summarize` flag
- Debugging cache misses and investigating why tasks aren't cached
- Identifying slow packages and optimizing their build times
- Profiling task execution with verbose logging
- Optimizing dependency installation with pnpm store

### Versioning and Publishing
- Setting up Changesets for package versioning
- Creating changeset files: `pnpm changeset`
- Versioning packages: `pnpm changeset version`
- Publishing packages: `pnpm changeset publish`
- Implementing independent vs fixed versioning strategies
- Setting up automated release workflows

### Advanced Patterns
- Implementing task dependencies with `^build` (build dependencies first)
- Setting up persistent tasks for dev servers: `"cache": false, "persistent": true`
- Configuring global dependencies that invalidate all caches
- Using filters for complex package selection: `pnpm --filter "...web" build`
- Setting up multiple outputs for different build artifacts
- Implementing custom task runners and plugins

### Troubleshooting Common Issues
- Resolving circular dependencies between packages
- Fixing phantom dependencies by adding explicit dependencies
- Debugging "Module not found" errors in monorepo packages
- Resolving peer dependency conflicts across workspace
- Fixing cache corruption and stale build outputs
- Investigating why builds aren't using cache

### Team Collaboration and Standards
- Establishing naming conventions for workspace packages
- Documenting package dependencies and architecture
- Setting up pre-commit hooks for workspace validation
- Implementing code ownership with CODEOWNERS for packages
- Creating runbooks for common monorepo operations
- Training teams on monorepo workflows and best practices

## Core Concepts

### 1. Why Monorepos?

**Advantages:**
- Shared code and dependencies
- Atomic commits across projects
- Consistent tooling and standards
- Easier refactoring
- Simplified dependency management
- Better code visibility

**Challenges:**
- Build performance at scale
- CI/CD complexity
- Access control
- Large Git repository

### 2. Monorepo Tools

**Package Managers:**
- pnpm workspaces (recommended)
- npm workspaces
- Yarn workspaces

**Build Systems:**
- Turborepo (recommended for most)
- Nx (feature-rich, complex)
- Lerna (older, maintenance mode)

## Turborepo Setup

### Initial Setup

```bash
# Create new monorepo
npx create-turbo@latest my-monorepo
cd my-monorepo

# Structure:
# apps/
#   web/          - Next.js app
#   docs/         - Documentation site
# packages/
#   ui/           - Shared UI components
#   config/       - Shared configurations
#   tsconfig/     - Shared TypeScript configs
# turbo.json      - Turborepo configuration
# package.json    - Root package.json
```

### Configuration

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "!.next/cache/**"]
    },
    "test": {
      "dependsOn": ["build"],
      "outputs": ["coverage/**"]
    },
    "lint": {
      "outputs": []
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "type-check": {
      "dependsOn": ["^build"],
      "outputs": []
    }
  }
}
```

```json
// package.json (root)
{
  "name": "my-monorepo",
  "private": true,
  "workspaces": [
    "apps/*",
    "packages/*"
  ],
  "scripts": {
    "build": "turbo run build",
    "dev": "turbo run dev",
    "test": "turbo run test",
    "lint": "turbo run lint",
    "format": "prettier --write \"**/*.{ts,tsx,md}\"",
    "clean": "turbo run clean && rm -rf node_modules"
  },
  "devDependencies": {
    "turbo": "^1.10.0",
    "prettier": "^3.0.0",
    "typescript": "^5.0.0"
  },
  "packageManager": "pnpm@8.0.0"
}
```

### Package Structure

```json
// packages/ui/package.json
{
  "name": "@repo/ui",
  "version": "0.0.0",
  "private": true,
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "types": "./dist/index.d.ts"
    },
    "./button": {
      "import": "./dist/button.js",
      "types": "./dist/button.d.ts"
    }
  },
  "scripts": {
    "build": "tsup src/index.ts --format esm,cjs --dts",
    "dev": "tsup src/index.ts --format esm,cjs --dts --watch",
    "lint": "eslint src/",
    "type-check": "tsc --noEmit"
  },
  "devDependencies": {
    "@repo/tsconfig": "workspace:*",
    "tsup": "^7.0.0",
    "typescript": "^5.0.0"
  },
  "dependencies": {
    "react": "^18.2.0"
  }
}
```

## pnpm Workspaces

### Setup

```yaml
# pnpm-workspace.yaml
packages:
  - 'apps/*'
  - 'packages/*'
  - 'tools/*'
```

```json
// .npmrc
# Hoist shared dependencies
shamefully-hoist=true

# Strict peer dependencies
auto-install-peers=true
strict-peer-dependencies=true

# Performance
store-dir=~/.pnpm-store
```

### Dependency Management

```bash
# Install dependency in specific package
pnpm add react --filter @repo/ui
pnpm add -D typescript --filter @repo/ui

# Install workspace dependency
pnpm add @repo/ui --filter web

# Install in all packages
pnpm add -D eslint -w

# Update all dependencies
pnpm update -r

# Remove dependency
pnpm remove react --filter @repo/ui
```

### Scripts

```bash
# Run script in specific package
pnpm --filter web dev
pnpm --filter @repo/ui build

# Run in all packages
pnpm -r build
pnpm -r test

# Run in parallel
pnpm -r --parallel dev

# Filter by pattern
pnpm --filter "@repo/*" build
pnpm --filter "...web" build  # Build web and dependencies
```

## Nx Monorepo

### Setup

```bash
# Create Nx monorepo
npx create-nx-workspace@latest my-org

# Generate applications
nx generate @nx/react:app my-app
nx generate @nx/next:app my-next-app

# Generate libraries
nx generate @nx/react:lib ui-components
nx generate @nx/js:lib utils
```

### Configuration

```json
// nx.json
{
  "extends": "nx/presets/npm.json",
  "$schema": "./node_modules/nx/schemas/nx-schema.json",
  "targetDefaults": {
    "build": {
      "dependsOn": ["^build"],
      "inputs": ["production", "^production"],
      "cache": true
    },
    "test": {
      "inputs": ["default", "^production", "{workspaceRoot}/jest.preset.js"],
      "cache": true
    },
    "lint": {
      "inputs": ["default", "{workspaceRoot}/.eslintrc.json"],
      "cache": true
    }
  },
  "namedInputs": {
    "default": ["{projectRoot}/**/*", "sharedGlobals"],
    "production": [
      "default",
      "!{projectRoot}/**/?(*.)+(spec|test).[jt]s?(x)?(.snap)",
      "!{projectRoot}/tsconfig.spec.json"
    ],
    "sharedGlobals": []
  }
}
```

### Running Tasks

```bash
# Run task for specific project
nx build my-app
nx test ui-components
nx lint utils

# Run for affected projects
nx affected:build
nx affected:test --base=main

# Visualize dependencies
nx graph

# Run in parallel
nx run-many --target=build --all --parallel=3
```

## Shared Configurations

### TypeScript Configuration

```json
// packages/tsconfig/base.json
{
  "compilerOptions": {
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "incremental": true,
    "declaration": true
  },
  "exclude": ["node_modules"]
}

// packages/tsconfig/react.json
{
  "extends": "./base.json",
  "compilerOptions": {
    "jsx": "react-jsx",
    "lib": ["ES2022", "DOM", "DOM.Iterable"]
  }
}

// apps/web/tsconfig.json
{
  "extends": "@repo/tsconfig/react.json",
  "compilerOptions": {
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist"]
}
```

### ESLint Configuration

```javascript
// packages/config/eslint-preset.js
module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'prettier',
  ],
  plugins: ['@typescript-eslint', 'react', 'react-hooks'],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true,
    },
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
  rules: {
    '@typescript-eslint/no-unused-vars': 'error',
    'react/react-in-jsx-scope': 'off',
  },
};

// apps/web/.eslintrc.js
module.exports = {
  extends: ['@repo/config/eslint-preset'],
  rules: {
    // App-specific rules
  },
};
```

## Code Sharing Patterns

### Pattern 1: Shared UI Components

```typescript
// packages/ui/src/button.tsx
import * as React from 'react';

export interface ButtonProps {
  variant?: 'primary' | 'secondary';
  children: React.ReactNode;
  onClick?: () => void;
}

export function Button({ variant = 'primary', children, onClick }: ButtonProps) {
  return (
    <button
      className={`btn btn-${variant}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

// packages/ui/src/index.ts
export { Button, type ButtonProps } from './button';
export { Input, type InputProps } from './input';

// apps/web/src/app.tsx
import { Button } from '@repo/ui';

export function App() {
  return <Button variant="primary">Click me</Button>;
}
```

### Pattern 2: Shared Utilities

```typescript
// packages/utils/src/string.ts
export function capitalize(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function truncate(str: string, length: number): string {
  return str.length > length ? str.slice(0, length) + '...' : str;
}

// packages/utils/src/index.ts
export * from './string';
export * from './array';
export * from './date';

// Usage in apps
import { capitalize, truncate } from '@repo/utils';
```

### Pattern 3: Shared Types

```typescript
// packages/types/src/user.ts
export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user';
}

export interface CreateUserInput {
  email: string;
  name: string;
  password: string;
}

// Used in both frontend and backend
import type { User, CreateUserInput } from '@repo/types';
```

## Build Optimization

### Turborepo Caching

```json
// turbo.json
{
  "pipeline": {
    "build": {
      // Build depends on dependencies being built first
      "dependsOn": ["^build"],

      // Cache these outputs
      "outputs": ["dist/**", ".next/**"],

      // Cache based on these inputs (default: all files)
      "inputs": ["src/**/*.tsx", "src/**/*.ts", "package.json"]
    },
    "test": {
      // Run tests in parallel, don't depend on build
      "cache": true,
      "outputs": ["coverage/**"]
    }
  }
}
```

### Remote Caching

```bash
# Turborepo Remote Cache (Vercel)
npx turbo login
npx turbo link

# Custom remote cache
# turbo.json
{
  "remoteCache": {
    "signature": true,
    "enabled": true
  }
}
```

## CI/CD for Monorepos

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # For Nx affected commands

      - uses: pnpm/action-setup@v2
        with:
          version: 8

      - uses: actions/setup-node@v3
        with:
          node-version: 18
          cache: 'pnpm'

      - name: Install dependencies
        run: pnpm install --frozen-lockfile

      - name: Build
        run: pnpm turbo run build

      - name: Test
        run: pnpm turbo run test

      - name: Lint
        run: pnpm turbo run lint

      - name: Type check
        run: pnpm turbo run type-check
```

### Deploy Affected Only

```yaml
# Deploy only changed apps
- name: Deploy affected apps
  run: |
    if pnpm nx affected:apps --base=origin/main --head=HEAD | grep -q "web"; then
      echo "Deploying web app"
      pnpm --filter web deploy
    fi
```

## Best Practices

1. **Consistent Versioning**: Lock dependency versions across workspace
2. **Shared Configs**: Centralize ESLint, TypeScript, Prettier configs
3. **Dependency Graph**: Keep it acyclic, avoid circular dependencies
4. **Cache Effectively**: Configure inputs/outputs correctly
5. **Type Safety**: Share types between frontend/backend
6. **Testing Strategy**: Unit tests in packages, E2E in apps
7. **Documentation**: README in each package
8. **Release Strategy**: Use changesets for versioning

## Common Pitfalls

- **Circular Dependencies**: A depends on B, B depends on A
- **Phantom Dependencies**: Using deps not in package.json
- **Incorrect Cache Inputs**: Missing files in Turborepo inputs
- **Over-Sharing**: Sharing code that should be separate
- **Under-Sharing**: Duplicating code across packages
- **Large Monorepos**: Without proper tooling, builds slow down

## Publishing Packages

```bash
# Using Changesets
pnpm add -Dw @changesets/cli
pnpm changeset init

# Create changeset
pnpm changeset

# Version packages
pnpm changeset version

# Publish
pnpm changeset publish
```

```yaml
# .github/workflows/release.yml
- name: Create Release Pull Request or Publish
  uses: changesets/action@v1
  with:
    publish: pnpm release
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
```

## Resources

- **references/turborepo-guide.md**: Comprehensive Turborepo documentation
- **references/nx-guide.md**: Nx monorepo patterns
- **references/pnpm-workspaces.md**: pnpm workspace features
- **assets/monorepo-checklist.md**: Setup checklist
- **assets/migration-guide.md**: Multi-repo to monorepo migration
- **scripts/dependency-graph.ts**: Visualize package dependencies
