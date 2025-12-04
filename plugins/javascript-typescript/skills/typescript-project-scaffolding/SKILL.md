---
name: typescript-project-scaffolding
description: Set up production-ready TypeScript projects with modern tooling, configuration, and best practices. Use when initializing new TypeScript projects or creating project boilerplate. Use when writing or editing tsconfig.json, tsconfig.node.json, or TypeScript configuration files. Use when configuring package.json scripts for TypeScript builds, type checking, or development workflows. Use when setting up Vite, esbuild, tsc, or other TypeScript build tools. Use when configuring ESLint for TypeScript (.eslintrc, eslint.config.js) with @typescript-eslint plugins. Use when setting up Prettier with TypeScript projects. Use when creating Next.js 14/15 applications with TypeScript and App Router. Use when scaffolding React applications with Vite and TypeScript (react-ts template). Use when building Node.js APIs or backend services with TypeScript. Use when creating publishable TypeScript libraries or npm packages. Use when building CLI tools with TypeScript (commander, yargs, inquirer). Use when setting up monorepo workspaces with pnpm, Turborepo, or Nx. Use when configuring Vitest or Jest for TypeScript testing. Use when setting up path aliases (@/*) in TypeScript projects. Use when migrating JavaScript projects to TypeScript.
---

# TypeScript Project Scaffolding

Production-ready TypeScript project scaffolding with modern tooling for web applications, APIs, libraries, and CLI tools.

## When to use this skill

- Initializing new TypeScript projects from scratch
- Writing or editing tsconfig.json, tsconfig.node.json, or tsconfig.build.json files
- Configuring TypeScript compiler options (strict mode, module resolution, target)
- Setting up path aliases (@/*, ~/*, #/*) for clean imports
- Configuring package.json scripts for TypeScript (build, dev, typecheck, lint)
- Setting up Vite with TypeScript (vite.config.ts)
- Configuring esbuild or tsc for TypeScript compilation
- Setting up ESLint for TypeScript with @typescript-eslint/parser and plugins
- Configuring Prettier with TypeScript projects
- Creating Next.js 14/15 applications with TypeScript and App Router
- Scaffolding React SPAs with Vite react-ts template
- Building Node.js APIs with TypeScript (Express, Fastify, Hono)
- Creating publishable TypeScript libraries for npm
- Building CLI tools with TypeScript (commander, yargs, inquirer, chalk)
- Setting up monorepo workspaces (pnpm-workspace.yaml, turborepo, Nx)
- Configuring Vitest for TypeScript testing (vitest.config.ts)
- Configuring Jest for TypeScript (jest.config.ts, ts-jest)
- Migrating JavaScript projects to TypeScript
- Setting up TypeScript declaration files (.d.ts) for libraries
- Configuring module resolution (bundler, node, node16)

## Core Concepts

### 1. Project Types

- **Next.js App**: Full-stack React with App Router
- **React SPA**: Client-side single-page application
- **Node.js API**: Backend services and APIs
- **Library**: Publishable npm packages
- **CLI Tool**: Command-line applications

### 2. Modern Tooling Stack

- **Package Manager**: pnpm (fast, efficient)
- **Bundler**: Vite, esbuild, or Rollup
- **Testing**: Vitest (fast, Vite-native)
- **Linting**: ESLint with TypeScript plugin
- **Formatting**: Prettier with consistent config

### 3. TypeScript Configuration

- **Strict Mode**: Enable all strict checks
- **Path Aliases**: Clean imports with @/* paths
- **Module Resolution**: Bundler mode for modern projects
- **Declaration Files**: Generate .d.ts for libraries

## Quick Start

### Next.js 15 App

```bash
pnpm create next-app@latest my-app --typescript --tailwind --eslint --app --src-dir
cd my-app
pnpm add -D vitest @testing-library/react @vitejs/plugin-react
```

### React SPA with Vite

```bash
pnpm create vite my-app --template react-ts
cd my-app
pnpm install
pnpm add -D vitest @testing-library/react jsdom
```

### Node.js API

```bash
mkdir my-api && cd my-api
pnpm init
pnpm add -D typescript @types/node tsx vitest
pnpm add express zod
pnpm tsc --init
```

## Configuration Patterns

### Pattern 1: Optimized tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "react-jsx",
    "incremental": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### Pattern 2: ESLint Configuration

```javascript
// eslint.config.js (flat config)
import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';
import react from 'eslint-plugin-react';

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        project: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/consistent-type-imports': 'error',
    },
  }
);
```

### Pattern 3: Vitest Configuration

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

### Pattern 4: Package.json Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:coverage": "vitest --coverage",
    "lint": "eslint . --ext .ts,.tsx",
    "lint:fix": "eslint . --ext .ts,.tsx --fix",
    "format": "prettier --write .",
    "typecheck": "tsc --noEmit"
  }
}
```

## Monorepo Setup

```bash
# Initialize pnpm workspace
pnpm init
echo "packages:\n  - 'packages/*'\n  - 'apps/*'" > pnpm-workspace.yaml

# Add Turborepo
pnpm add -D turbo

# Create turbo.json
cat > turbo.json << 'EOF'
{
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**"]
    },
    "test": {
      "dependsOn": ["build"]
    },
    "lint": {}
  }
}
EOF
```

## Best Practices

1. **Strict TypeScript**: Enable all strict mode checks
2. **Path Aliases**: Use @/* for clean imports
3. **Consistent Formatting**: Prettier with shared config
4. **Fast Testing**: Vitest over Jest for speed
5. **Modern Bundling**: Vite or esbuild over Webpack
