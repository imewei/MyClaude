---
name: monorepo-management
version: "1.0.7"
description: Master monorepo management with Turborepo, Nx, and pnpm workspaces. Use when setting up turbo.json or nx.json, organizing apps/* and packages/* directories, optimizing builds with caching, managing workspace dependencies, implementing shared configs, setting up CI/CD for monorepos, or publishing packages with changesets.
---

# Monorepo Management

Scalable multi-package repositories with optimized builds and dependency management.

## Tool Selection

| Tool | Use Case | Strengths |
|------|----------|-----------|
| Turborepo | Most projects | Simple, fast, great caching |
| Nx | Enterprise | Feature-rich, plugins, generators |
| pnpm workspaces | Package manager | Efficient disk usage, strict |
| Lerna | Legacy | Maintenance mode, migrate away |

## Turborepo Setup

### Project Structure

```
my-monorepo/
├── apps/
│   ├── web/           # Next.js app
│   └── docs/          # Documentation
├── packages/
│   ├── ui/            # Shared components
│   ├── config/        # ESLint, Prettier configs
│   └── tsconfig/      # TypeScript configs
├── turbo.json
├── package.json
└── pnpm-workspace.yaml
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
      "outputs": ["dist/**", ".next/**"]
    },
    "test": {
      "dependsOn": ["build"],
      "outputs": ["coverage/**"]
    },
    "lint": { "outputs": [] },
    "dev": { "cache": false, "persistent": true }
  }
}
```

```json
// package.json (root)
{
  "name": "my-monorepo",
  "private": true,
  "workspaces": ["apps/*", "packages/*"],
  "scripts": {
    "build": "turbo run build",
    "dev": "turbo run dev",
    "test": "turbo run test",
    "lint": "turbo run lint"
  },
  "devDependencies": {
    "turbo": "^1.10.0"
  },
  "packageManager": "pnpm@8.0.0"
}
```

## pnpm Workspaces

```yaml
# pnpm-workspace.yaml
packages:
  - 'apps/*'
  - 'packages/*'
```

```ini
# .npmrc
shamefully-hoist=true
auto-install-peers=true
strict-peer-dependencies=true
```

### Common Commands

```bash
# Install in specific package
pnpm add react --filter @repo/ui

# Install workspace dependency
pnpm add @repo/ui --filter web

# Run in specific package
pnpm --filter web dev

# Run in all packages
pnpm -r build

# Run in parallel
pnpm -r --parallel dev

# Filter by pattern
pnpm --filter "@repo/*" build
pnpm --filter "...web" build  # web + dependencies
```

## Package Configuration

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
    "dev": "tsup src/index.ts --format esm,cjs --dts --watch"
  },
  "devDependencies": {
    "@repo/tsconfig": "workspace:*"
  }
}
```

## Shared Configurations

### TypeScript

```json
// packages/tsconfig/base.json
{
  "compilerOptions": {
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "declaration": true
  }
}

// apps/web/tsconfig.json
{
  "extends": "@repo/tsconfig/react.json",
  "compilerOptions": { "outDir": "dist", "rootDir": "src" },
  "include": ["src"]
}
```

### ESLint

```javascript
// packages/config/eslint-preset.js
module.exports = {
  extends: ['eslint:recommended', 'plugin:@typescript-eslint/recommended', 'prettier'],
  parser: '@typescript-eslint/parser',
};

// apps/web/.eslintrc.js
module.exports = { extends: ['@repo/config/eslint-preset'] };
```

## Code Sharing Patterns

### Shared UI Components

```typescript
// packages/ui/src/button.tsx
export interface ButtonProps {
  variant?: 'primary' | 'secondary';
  children: React.ReactNode;
}

export function Button({ variant = 'primary', children }: ButtonProps) {
  return <button className={`btn-${variant}`}>{children}</button>;
}

// packages/ui/src/index.ts
export { Button, type ButtonProps } from './button';

// apps/web/src/app.tsx
import { Button } from '@repo/ui';
```

### Shared Types

```typescript
// packages/types/src/user.ts
export interface User {
  id: string;
  email: string;
  name: string;
}

// Used in frontend and backend
import type { User } from '@repo/types';
```

## CI/CD

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with: { fetch-depth: 0 }
      - uses: pnpm/action-setup@v2
        with: { version: 8 }
      - uses: actions/setup-node@v3
        with: { node-version: 18, cache: 'pnpm' }
      - run: pnpm install --frozen-lockfile
      - run: pnpm turbo run build test lint
```

## Publishing with Changesets

```bash
pnpm add -Dw @changesets/cli
pnpm changeset init

# Create changeset
pnpm changeset

# Version packages
pnpm changeset version

# Publish
pnpm changeset publish
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Consistent Versioning | Lock deps across workspace |
| Shared Configs | Centralize ESLint, TS, Prettier |
| Acyclic Deps | No circular dependencies |
| Effective Caching | Configure inputs/outputs correctly |
| Type Sharing | Share types between FE/BE |
| Testing | Unit in packages, E2E in apps |

## Common Pitfalls

| Issue | Solution |
|-------|----------|
| Circular Dependencies | Refactor shared code to separate package |
| Phantom Dependencies | Add explicit dependency in package.json |
| Cache Not Working | Check inputs/outputs in turbo.json |
| Slow Builds | Enable remote caching with Vercel |
| Module Not Found | Check exports field in package.json |

## Checklist

- [ ] pnpm-workspace.yaml configured
- [ ] turbo.json pipeline defined
- [ ] Shared tsconfig package created
- [ ] Shared ESLint config created
- [ ] Package exports properly configured
- [ ] workspace:* protocol for internal deps
- [ ] CI/CD with pnpm caching
- [ ] Remote caching enabled (Vercel/Nx Cloud)
- [ ] Changesets for versioning
