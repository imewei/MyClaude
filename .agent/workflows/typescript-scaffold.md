---
description: Workflow for typescript-scaffold
triggers:
- /typescript-scaffold
- workflow for typescript scaffold
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



# TypeScript Project Scaffolding

Generate complete project structures with modern tooling following best practices.

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 10-15 min | Basic config, essential deps, prototype-ready |
| Standard (default) | 20-30 min | Full config, tooling, testing setup |
| Comprehensive | 40-60 min | + CI/CD, Docker, security hardening |

## Project Types

| Type | Use Case | Key Features |
|------|----------|--------------|
| Next.js | Full-stack React with SSR | App Router, API routes, Server Components |
| React + Vite | Client-side SPAs | Fast HMR, optimized builds, Vitest |
| Node.js API | Backend services | Express/Fastify, middleware, auth |
| Library | Reusable packages | Tree-shakeable, dual format |
| CLI | Command-line tools | Commander.js, interactive prompts |

**Decision guide:** [Project Scaffolding Guide](../../plugins/javascript-typescript/docs/project-scaffolding-guide.md)

## Core Scaffolding Steps

### 1. Initialize with pnpm

```bash
mkdir project-name && cd project-name
pnpm init && git init
```

### 2. Project-Specific Setup

| Type | Command | Full Guide |
|------|---------|------------|
| Next.js | `pnpm create next-app@latest .` | [nextjs-scaffolding.md](../../plugins/javascript-typescript/docs/nextjs-scaffolding.md) |
| React+Vite | `pnpm create vite . --template react-ts` | - |
| Node.js API | `pnpm add express zod dotenv` | [nodejs-api-scaffolding.md](../../plugins/javascript-typescript/docs/nodejs-api-scaffolding.md) |
| Library/CLI | Manual setup | [library-cli-scaffolding.md](../../plugins/javascript-typescript/docs/library-cli-scaffolding.md) |

### 3. TypeScript Configuration

| Mode | Settings |
|------|----------|
| Quick | `strict: false`, basic config |
| Standard | Full strict mode, recommended settings |
| Comprehensive | + Project references, incremental, paths |

**Guide:** [TypeScript Configuration](../../plugins/javascript-typescript/docs/typescript-configuration.md)

### 4. Development Tooling

| Mode | Tools |
|------|-------|
| Quick | ESLint + Prettier |
| Standard | + Vitest + Husky (lint-staged) |
| Comprehensive | + GitHub Actions + Docker + Bundle analyzer |

**Guide:** [Development Tooling](../../plugins/javascript-typescript/docs/development-tooling.md)

### 5. Project Structure

| Type | Directories |
|------|-------------|
| Next.js | src/app/, components/, lib/, hooks/ |
| React+Vite | src/components/, pages/, hooks/, lib/ |
| Node.js | src/routes/, controllers/, services/, middleware/ |
| Library | src/, tests/, dist/ |
| CLI | bin/, src/cli/commands/, src/core/ |

## Essential Files

| File | Purpose |
|------|---------|
| package.json | Scripts: dev, build, test, lint, type-check |
| tsconfig.json | TypeScript settings |
| .eslintrc | Linting rules |
| .prettierrc | Formatting |
| .env.example | Environment template |
| README.md | Setup instructions |
| tests/setup.ts | Test configuration |

## Output Checklist

### All Modes
- [ ] Project directory structure
- [ ] package.json with scripts
- [ ] tsconfig.json for project type
- [ ] ESLint + Prettier config
- [ ] .env.example, .gitignore
- [ ] README.md with setup
- [ ] Entry point files
- [ ] Example components/routes

### Standard+
- [ ] Vitest setup
- [ ] Path aliases configured

### Comprehensive
- [ ] GitHub Actions CI/CD
- [ ] Docker setup
- [ ] Husky git hooks
- [ ] Monorepo setup (if requested)

## Common Patterns

### Path Aliases
```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {"@/*": ["./src/*"]}
  }
}
```

### Input Validation
```typescript
import { z } from 'zod'
const schema = z.object({ email: z.string().email() })
```

## External Documentation

| Document | Content |
|----------|---------|
| [Project Scaffolding Guide](../../plugins/javascript-typescript/docs/project-scaffolding-guide.md) | Decision frameworks, patterns |
| [Next.js Scaffolding](../../plugins/javascript-typescript/docs/nextjs-scaffolding.md) | Server Components, API routes |
| [Node.js API Scaffolding](../../plugins/javascript-typescript/docs/nodejs-api-scaffolding.md) | Controllers, services |
| [Library & CLI Scaffolding](../../plugins/javascript-typescript/docs/library-cli-scaffolding.md) | Publishing, Commander.js |
| [TypeScript Configuration](../../plugins/javascript-typescript/docs/typescript-configuration.md) | Strict mode, paths |
| [Development Tooling](../../plugins/javascript-typescript/docs/development-tooling.md) | ESLint, Vitest, CI/CD |

## Next Steps

After scaffolding:
1. `pnpm install`
2. `pnpm dev`
3. `pnpm test`
4. `pnpm type-check`
