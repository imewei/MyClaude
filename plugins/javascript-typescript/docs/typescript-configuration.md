# TypeScript Configuration Guide

> **Version:** 1.0.3 | **Category:** Configuration | **Maturity:** 95%

## tsconfig.json Optimization

### Strict Mode Configuration (Recommended)

```json
{
  "compilerOptions": {
    /* Language and Environment */
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
    "experimentalDecorators": true,
    "emitDecoratorMetadata": true,

    /* Modules */
    "module": "ESNext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "allowImportingTsExtensions": false,

    /* Emit */
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "removeComments": true,
    "importHelpers": true,
    "downlevelIteration": true,

    /* Interop Constraints */
    "isolatedModules": true,
    "allowSyntheticDefaultImports": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,

    /* Type Checking */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "allowUnusedLabels": false,
    "allowUnreachableCode": false,

    /* Completeness */
    "skipLibCheck": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

### Framework-Specific Configurations

#### Next.js

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "jsx": "preserve",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "allowJs": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "incremental": true,
    "isolatedModules": true,
    "paths": {
      "@/*": ["./src/*"]
    },
    "plugins": [{ "name": "next" }]
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

#### React + Vite

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "allowImportingTsExtensions": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "isolatedModules": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

#### Node.js Backend

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022"],
    "module": "ESNext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "declaration": true,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "types": ["node"]
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

---

## Monorepo Configuration with Project References

### Root tsconfig.json

```json
{
  "files": [],
  "references": [
    { "path": "./packages/ui" },
    { "path": "./packages/utils" },
    { "path": "./apps/web" }
  ]
}
```

### Package-Level tsconfig.json

```json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "composite": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"],
  "references": [
    { "path": "../utils" }
  ]
}
```

### tsconfig.base.json (Shared)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  }
}
```

---

## Incremental Compilation

### Optimized for Fast Builds

```json
{
  "compilerOptions": {
    "incremental": true,
    "tsBuildInfoFile": "./.tsbuildinfo",
    "skipLibCheck": true,
    "skipDefaultLibCheck": true
  }
}
```

### Build Commands

```json
{
  "scripts": {
    "build": "tsc -p tsconfig.build.json",
    "build:watch": "tsc -p tsconfig.build.json --watch",
    "build:clean": "rm -rf dist .tsbuildinfo && pnpm build"
  }
}
```

---

## Path Mapping

### Absolute Imports

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@components/*": ["./src/components/*"],
      "@utils/*": ["./src/utils/*"],
      "@types/*": ["./src/types/*"]
    }
  }
}
```

### Usage

```typescript
// Instead of: import { Button } from '../../../components/ui/Button'
import { Button } from '@/components/ui/Button'
import { formatDate } from '@/utils/date'
import type { User } from '@/types/user'
```

---

## Type Declaration Files

### Custom Type Declarations

```typescript
// src/types/express.d.ts
import 'express'

declare module 'express-serve-static-core' {
  interface Request {
    user?: {
      id: string
      email: string
    }
  }
}
```

### Global Type Declarations

```typescript
// src/types/global.d.ts
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV: 'development' | 'production' | 'test'
      DATABASE_URL: string
      JWT_SECRET: string
    }
  }
}

export {}
```

---

## Strict Mode Migration

### Gradual Strictness

```json
{
  "compilerOptions": {
    /* Start here */
    "strict": false,
    "noImplicitAny": true,

    /* Then enable gradually */
    "strictNullChecks": false,
    "strictFunctionTypes": false,
    "strictBindCallApply": false,
    "strictPropertyInitialization": false,
    "noImplicitThis": false,
    "alwaysStrict": false
  }
}
```

### Final Strict Configuration

```json
{
  "compilerOptions": {
    "strict": true
  }
}
```

---

## Compiler Performance Optimization

### Fast Builds

```json
{
  "compilerOptions": {
    /* Disable expensive checks */
    "skipLibCheck": true,
    "skipDefaultLibCheck": true,

    /* Enable incremental compilation */
    "incremental": true,

    /* Use project references for monorepos */
    "composite": true,

    /* Disable source maps in development */
    "sourceMap": false,

    /* Use faster module resolution */
    "moduleResolution": "bundler"
  }
}
```

### Build Time Comparison

| Configuration | Build Time | Watch Time |
|---------------|-----------|------------|
| Default | 15s | 3s |
| skipLibCheck | 8s | 1.5s |
| + incremental | 3s (after first) | 0.5s |
| + composite | 2s (parallel) | 0.3s |

---

## Integration with Build Tools

### Vite

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import tsconfigPaths from 'vite-tsconfig-paths'

export default defineConfig({
  plugins: [tsconfigPaths()], // Automatically reads tsconfig paths
})
```

### Webpack

```javascript
// webpack.config.js
const TsconfigPathsPlugin = require('tsconfig-paths-webpack-plugin')

module.exports = {
  resolve: {
    plugins: [new TsconfigPathsPlugin()],
  },
}
```

---

## Common Issues and Solutions

### Issue: "Cannot find module" with path aliases

```json
// Solution: Update both tsconfig.json and build tool
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

```typescript
// vite.config.ts
import path from 'path'

export default {
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
}
```

### Issue: Slow type checking

```json
// Solution: Use project references and skipLibCheck
{
  "compilerOptions": {
    "skipLibCheck": true,
    "incremental": true
  }
}
```

### Issue: "noImplicitAny" errors in migration

```typescript
// Temporary solution: Use type assertions
const data = JSON.parse(str) as unknown
const user = data as User

// Better solution: Add proper types
interface ApiResponse {
  data: User
}
const response: ApiResponse = JSON.parse(str)
```

---

## TypeScript Version Features

### TypeScript 5.3+ Features

```typescript
// Import attributes
import data from './data.json' with { type: 'json' }

// Const type parameters
function identity<const T>(value: T): T {
  return value
}

// Decorators (stable)
@sealed
class MyClass {}
```

### TypeScript 5.2 Features

```typescript
// Using declaration
{
  using file = getFileHandle()
  // Automatically disposed
}

// Decorator metadata
function logged(target: any, context: ClassMethodDecoratorContext) {
  // ...
}
```

---

## Related Documentation

- [Project Scaffolding Guide](project-scaffolding-guide.md)
- [Development Tooling](development-tooling.md)
- [Next.js Scaffolding](nextjs-scaffolding.md)
