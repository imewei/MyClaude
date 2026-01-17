---
name: typescript-project-scaffolding
version: "1.0.7"
description: Set up production-ready TypeScript projects with modern tooling. Use when initializing projects, configuring tsconfig.json, setting up Vite/ESLint/Vitest, creating Next.js/React apps, or scaffolding Node.js APIs and CLI tools.
---

# TypeScript Project Scaffolding

Production-ready TypeScript project setup with modern tooling.

## Project Types

| Type | Quick Start |
|------|-------------|
| Next.js App | `pnpm create next-app@latest --typescript --tailwind --eslint --app` |
| React SPA | `pnpm create vite my-app --template react-ts` |
| Node.js API | `pnpm add -D typescript @types/node tsx vitest` |
| CLI Tool | Add `commander`, `inquirer`, `chalk` |
| Library | Configure `tsconfig.build.json` with declarations |

## tsconfig.json (Optimized)

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
    "baseUrl": ".",
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

## ESLint (Flat Config)

```javascript
// eslint.config.js
import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  {
    languageOptions: {
      parserOptions: { project: true }
    },
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/consistent-type-imports': 'error'
    }
  }
);
```

## Vitest Configuration

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
    coverage: { provider: 'v8', reporter: ['text', 'json', 'html'] }
  },
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') }
  }
});
```

## Package.json Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "test": "vitest",
    "test:coverage": "vitest --coverage",
    "lint": "eslint . --ext .ts,.tsx",
    "typecheck": "tsc --noEmit",
    "format": "prettier --write ."
  }
}
```

## Monorepo Setup

```bash
# pnpm workspace
echo "packages:\n  - 'packages/*'\n  - 'apps/*'" > pnpm-workspace.yaml
pnpm add -D turbo

# turbo.json
{
  "pipeline": {
    "build": { "dependsOn": ["^build"], "outputs": ["dist/**"] },
    "test": { "dependsOn": ["build"] },
    "lint": {}
  }
}
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Strict mode | Enable all strict checks |
| Path aliases | Use `@/*` for clean imports |
| Fast testing | Vitest over Jest |
| Modern bundling | Vite over Webpack |
| Consistent formatting | Prettier with shared config |
| Type-only imports | `import type { T }` syntax |

## Checklist

- [ ] Strict TypeScript enabled
- [ ] Path aliases configured
- [ ] ESLint with TypeScript plugin
- [ ] Prettier for formatting
- [ ] Vitest for testing
- [ ] Package scripts defined
- [ ] .gitignore includes node_modules, dist
