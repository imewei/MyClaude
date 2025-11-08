# Next.js Project Scaffolding Guide

> **Version:** 1.0.3 | **Framework:** Next.js 15+ | **Maturity:** 95%

## Complete Next.js Project Structure

```bash
# Create Next.js project with TypeScript
pnpm create next-app@latest . --typescript --tailwind --app --src-dir --import-alias "@/*"
```

### Full Directory Structure

```
nextjs-project/
├── package.json
├── tsconfig.json
├── next.config.ts
├── .env.example
├── .gitignore
├── README.md
├── public/
│   ├── favicon.ico
│   └── images/
├── src/
│   ├── app/
│   │   ├── layout.tsx              # Root layout
│   │   ├── page.tsx                # Home page
│   │   ├── loading.tsx             # Loading UI
│   │   ├── error.tsx               # Error boundary
│   │   ├── not-found.tsx           # 404 page
│   │   ├── (auth)/                 # Auth route group
│   │   │   ├── login/
│   │   │   │   └── page.tsx
│   │   │   ├── register/
│   │   │   │   └── page.tsx
│   │   │   └── layout.tsx
│   │   ├── (dashboard)/            # Dashboard route group
│   │   │   ├── dashboard/
│   │   │   │   ├── page.tsx
│   │   │   │   ├── loading.tsx
│   │   │   │   └── @analytics/    # Parallel route
│   │   │   │       └── page.tsx
│   │   │   └── layout.tsx
│   │   ├── api/                    # API routes
│   │   │   ├── auth/
│   │   │   │   └── route.ts
│   │   │   ├── users/
│   │   │   │   └── route.ts
│   │   │   └── health/
│   │   │       └── route.ts
│   │   └── globals.css
│   ├── components/
│   │   ├── ui/                     # Shadcn-style components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── input.tsx
│   │   │   └── dialog.tsx
│   │   ├── layout/
│   │   │   ├── header.tsx
│   │   │   ├── footer.tsx
│   │   │   └── sidebar.tsx
│   │   └── features/
│   │       ├── auth/
│   │       │   ├── login-form.tsx
│   │       │   └── register-form.tsx
│   │       └── dashboard/
│   │           └── stats-card.tsx
│   ├── lib/
│   │   ├── actions.ts              # Server Actions
│   │   ├── api.ts                  # API client
│   │   ├── auth.ts                 # Auth utilities
│   │   ├── db.ts                   # Database client
│   │   ├── utils.ts                # Utility functions
│   │   └── types.ts                # Shared types
│   ├── hooks/
│   │   ├── use-auth.ts
│   │   ├── use-toast.ts
│   │   └── use-media-query.ts
│   ├── middleware.ts               # Next.js middleware
│   └── instrumentation.ts          # Observability
└── tests/
    ├── setup.ts
    ├── e2e/
    │   └── auth.spec.ts
    └── unit/
        └── utils.test.ts
```

---

## Configuration Files

### package.json

```json
{
  "name": "nextjs-project",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "vitest",
    "test:e2e": "playwright test",
    "type-check": "tsc --noEmit",
    "format": "prettier --write .",
    "prepare": "husky install"
  },
  "dependencies": {
    "next": "^15.0.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "zod": "^3.22.4",
    "@tanstack/react-query": "^5.17.0"
  },
  "devDependencies": {
    "@types/node": "^20.11.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^5.3.0",
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0",
    "vitest": "^1.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "@playwright/test": "^1.40.0",
    "eslint": "^8.56.0",
    "eslint-config-next": "^15.0.0",
    "prettier": "^3.1.0",
    "husky": "^8.0.0",
    "lint-staged": "^15.2.0"
  }
}
```

### next.config.ts

```typescript
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // React strict mode for development
  reactStrictMode: true,

  // Enable experimental features
  experimental: {
    // Server Actions
    serverActions: {
      bodySizeLimit: '2mb',
    },
    // Type-safe links
    typedRoutes: true,
  },

  // Image optimization
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'example.com',
      },
    ],
  },

  // Environment variables available to browser
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },

  // Redirects
  async redirects() {
    return [
      {
        source: '/old-path',
        destination: '/new-path',
        permanent: true,
      },
    ]
  },

  // Headers for security
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
        ],
      },
    ]
  },
}

export default nextConfig
```

### tsconfig.json

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

---

## Key Implementation Patterns

### 1. Server Components (Default)

```tsx
// src/app/products/page.tsx
import { db } from '@/lib/db'

// This is a Server Component by default
export default async function ProductsPage() {
  // Direct database access, no API route needed
  const products = await db.product.findMany()

  return (
    <div>
      <h1>Products</h1>
      {products.map((product) => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  )
}
```

### 2. Client Components (Interactive)

```tsx
// src/components/features/search-bar.tsx
'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

export function SearchBar() {
  const [query, setQuery] = useState('')
  const router = useRouter()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    router.push(`/search?q=${query}`)
  }

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search..."
      />
    </form>
  )
}
```

### 3. Server Actions

```tsx
// src/lib/actions.ts
'use server'

import { revalidatePath } from 'next/cache'
import { db } from './db'
import { z } from 'zod'

const createProductSchema = z.object({
  name: z.string().min(1),
  price: z.number().positive(),
})

export async function createProduct(formData: FormData) {
  const data = createProductSchema.parse({
    name: formData.get('name'),
    price: Number(formData.get('price')),
  })

  const product = await db.product.create({ data })

  revalidatePath('/products')
  return { success: true, product }
}
```

### 4. API Routes

```ts
// src/app/api/users/route.ts
import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'

const createUserSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
})

export async function POST(request: NextRequest) {
  try {
    const json = await request.json()
    const data = createUserSchema.parse(json)

    // Create user logic
    const user = { id: '1', email: data.email }

    return NextResponse.json(user, { status: 201 })
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: error.errors },
        { status: 400 }
      )
    }
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
```

### 5. Middleware for Authentication

```ts
// src/middleware.ts
import { NextRequest, NextResponse } from 'next/server'

export function middleware(request: NextRequest) {
  const token = request.cookies.get('session')?.value

  // Redirect to login if not authenticated
  if (!token && request.nextUrl.pathname.startsWith('/dashboard')) {
    return NextResponse.redirect(new URL('/login', request.url))
  }

  return NextResponse.next()
}

export const config = {
  matcher: '/dashboard/:path*',
}
```

### 6. Loading and Error States

```tsx
// src/app/products/loading.tsx
export default function Loading() {
  return <div>Loading products...</div>
}

// src/app/products/error.tsx
'use client'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div>
      <h2>Something went wrong!</h2>
      <button onClick={reset}>Try again</button>
    </div>
  )
}
```

---

## Authentication Patterns

### NextAuth.js Setup (Recommended)

```typescript
// src/app/api/auth/[...nextauth]/route.ts
import NextAuth from 'next-auth'
import CredentialsProvider from 'next-auth/providers/credentials'

const handler = NextAuth({
  providers: [
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        // Validate credentials
        const user = await validateUser(credentials)
        if (user) {
          return user
        }
        return null
      }
    })
  ],
  session: {
    strategy: 'jwt',
  },
  pages: {
    signIn: '/login',
  },
})

export { handler as GET, handler as POST }
```

---

## Database Integration

### Prisma Setup

```typescript
// src/lib/db.ts
import { PrismaClient } from '@prisma/client'

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
}

export const db =
  globalForPrisma.prisma ??
  new PrismaClient({
    log: ['query'],
  })

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = db
```

---

## Testing Setup

### Vitest Configuration

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
```

---

## Deployment Considerations

### Vercel (Recommended)
- Automatic deployments from Git
- Preview deployments for PRs
- Edge Functions for middleware
- Built-in Analytics and Web Vitals

### Self-Hosted
```dockerfile
FROM node:20-alpine AS base

# Install dependencies
FROM base AS deps
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile

# Build application
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Production image
FROM base AS runner
WORKDIR /app
ENV NODE_ENV production
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
```

---

## Related Documentation

- [Project Scaffolding Guide](project-scaffolding-guide.md)
- [TypeScript Configuration](typescript-configuration.md)
- [Development Tooling](development-tooling.md)
