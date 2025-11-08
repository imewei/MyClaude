# TypeScript Project Scaffolding Guide

> **Version:** 1.0.3 | **Category:** Project Setup | **Maturity:** 95%

## Overview

This guide provides a comprehensive framework for selecting and scaffolding the right TypeScript project type for your needs. Use this as a decision-making resource when initializing new projects.

---

## Project Type Selection Framework

### Decision Tree

```
Start Here
├─ Need UI/Frontend?
│  ├─ YES → Full-stack with server rendering?
│  │  ├─ YES → Next.js (App Router + API Routes)
│  │  └─ NO → React + Vite (SPA)
│  └─ NO → Backend/Library/Tool?
│     ├─ Backend API → Node.js API (Express/Fastify)
│     ├─ Reusable code → Library
│     └─ Terminal tool → CLI Application
```

### Project Type Characteristics

#### 1. Next.js Application
**When to choose**:
- Full-stack React applications with server-side rendering
- SEO-critical applications (marketing sites, blogs, e-commerce)
- Applications with API routes and backend logic
- Need for static site generation (SSG) or incremental static regeneration (ISR)
- Team familiar with React ecosystem

**Not suitable for**:
- Pure client-side SPAs without SEO needs
- Non-React projects
- Microservices (consider Node.js API instead)

**Key features**:
- App Router with layouts and parallel routes
- Server Components and Client Components
- Built-in API routes
- Image optimization
- Automatic code splitting
- TypeScript-first

#### 2. React + Vite SPA
**When to choose**:
- Client-side single-page applications
- Internal dashboards and admin panels
- Applications without SEO requirements
- Component libraries and design systems
- Fast development iteration needs

**Not suitable for**:
- SEO-critical public websites
- Server-side rendering requirements
- Non-React projects

**Key features**:
- Lightning-fast HMR with Vite
- Optimized production builds
- Plugin ecosystem
- TypeScript support out-of-the-box
- Modern browser targets

#### 3. Node.js API
**When to choose**:
- RESTful APIs and microservices
- GraphQL backends
- WebSocket servers
- Background job processors
- Authentication services
- Data processing pipelines

**Not suitable for**:
- Frontend applications
- Static file serving (use CDN)
- CPU-intensive workloads (consider Rust/Go)

**Key features**:
- Express.js or Fastify frameworks
- Middleware architecture
- Database integration (PostgreSQL, MongoDB, Redis)
- Authentication patterns (JWT, OAuth2)
- OpenAPI documentation

#### 4. Library/Package
**When to choose**:
- Reusable utilities and functions
- Framework-agnostic code
- npm packages for distribution
- Shared code across multiple projects
- Component libraries

**Not suitable for**:
- Applications with UI
- Backend services
- Monolithic applications

**Key features**:
- Tree-shakeable exports
- Dual module format (ESM + CommonJS)
- TypeScript declaration files
- Minimal dependencies
- Comprehensive testing

#### 5. CLI Application
**When to choose**:
- Command-line tools
- Build scripts and automation
- Developer productivity tools
- File processing utilities
- System administration tools

**Not suitable for**:
- GUI applications
- Long-running services
- Web applications

**Key features**:
- Commander.js or Yargs for argument parsing
- Interactive prompts with Inquirer.js
- Progress bars and spinners
- Colorized output
- Executable binaries

---

## Architecture Patterns by Project Type

### Next.js Architecture

**Recommended structure**:
```
src/
├── app/                    # App Router
│   ├── (auth)/            # Route groups
│   ├── (dashboard)/
│   ├── api/               # API routes
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page
├── components/
│   ├── ui/                # Shadcn-style components
│   └── features/          # Feature-specific components
├── lib/
│   ├── actions.ts         # Server Actions
│   ├── api.ts             # API client
│   └── utils.ts
└── hooks/
    └── use-*.ts
```

**Key patterns**:
- Server Components by default, Client Components when needed
- Server Actions for mutations
- Route handlers for API endpoints
- Parallel routes for complex layouts
- Loading and error boundaries

### React + Vite Architecture

**Recommended structure**:
```
src/
├── components/
│   ├── common/
│   ├── layout/
│   └── features/
├── pages/                 # Route components
├── hooks/
├── lib/
│   ├── api.ts
│   ├── store.ts           # State management
│   └── utils.ts
├── styles/
└── App.tsx
```

**Key patterns**:
- React Router for client-side routing
- Zustand or Redux for state management
- React Query for server state
- Code splitting with React.lazy
- Context for theming and global state

### Node.js API Architecture

**Recommended structure**:
```
src/
├── config/
│   ├── database.ts
│   └── env.ts
├── routes/
│   ├── index.ts
│   ├── users.ts
│   └── auth.ts
├── controllers/
├── services/              # Business logic
├── models/                # Database models
├── middleware/
│   ├── auth.ts
│   ├── validate.ts
│   └── error.ts
├── types/
├── utils/
└── app.ts
```

**Key patterns**:
- Controller → Service → Model architecture
- Dependency injection for testability
- Middleware for cross-cutting concerns
- Repository pattern for data access
- OpenAPI/Swagger documentation

---

## Best Practices by Project Type

### Next.js Best Practices

1. **Use Server Components by default**
   ```tsx
   // app/products/page.tsx (Server Component)
   async function ProductsPage() {
     const products = await fetchProducts() // Direct DB query
     return <ProductList products={products} />
   }
   ```

2. **Client Components for interactivity**
   ```tsx
   'use client'
   export function SearchBar() {
     const [query, setQuery] = useState('')
     // Interactive component
   }
   ```

3. **Server Actions for mutations**
   ```tsx
   // lib/actions.ts
   'use server'
   export async function createProduct(formData: FormData) {
     const product = await db.product.create({...})
     revalidatePath('/products')
     return product
   }
   ```

4. **Route Groups for organization**
   ```
   app/
   ├── (marketing)/        # Public pages
   │   ├── about/
   │   └── pricing/
   └── (app)/             # Authenticated pages
       └── dashboard/
   ```

### React + Vite Best Practices

1. **Code splitting with lazy loading**
   ```tsx
   const Dashboard = lazy(() => import('./pages/Dashboard'))
   <Suspense fallback={<Loading />}>
     <Dashboard />
   </Suspense>
   ```

2. **Optimize bundle with tree-shaking**
   ```ts
   // Bad: imports entire library
   import _ from 'lodash'

   // Good: imports only what's needed
   import { debounce } from 'lodash-es'
   ```

3. **Use React Query for server state**
   ```tsx
   const { data, isLoading } = useQuery({
     queryKey: ['products'],
     queryFn: fetchProducts,
   })
   ```

### Node.js API Best Practices

1. **Validate all inputs**
   ```ts
   import { z } from 'zod'

   const createUserSchema = z.object({
     email: z.string().email(),
     password: z.string().min(8),
   })

   app.post('/users', async (req, res) => {
     const data = createUserSchema.parse(req.body)
     // ...
   })
   ```

2. **Use middleware for authentication**
   ```ts
   const authenticate = async (req, res, next) => {
     const token = req.headers.authorization?.split(' ')[1]
     if (!token) return res.status(401).json({error: 'Unauthorized'})
     req.user = await verifyToken(token)
     next()
   }
   ```

3. **Centralized error handling**
   ```ts
   app.use((err, req, res, next) => {
     logger.error(err)
     res.status(err.status || 500).json({
       error: err.message,
       ...(process.env.NODE_ENV === 'development' && {stack: err.stack})
     })
   })
   ```

---

## Common Pitfalls and Solutions

### Next.js Pitfalls

**Pitfall**: Using Client Components unnecessarily
```tsx
// ❌ Bad: Entire page is client-side
'use client'
export default function Page() {
  const [count, setCount] = useState(0)
  return <div>{/* static content + interactive button */}</div>
}

// ✅ Good: Only button is client-side
export default function Page() {
  return (
    <div>
      {/* Static server-rendered content */}
      <InteractiveButton />
    </div>
  )
}
```

**Pitfall**: Not revalidating after mutations
```tsx
// ❌ Bad: Data not refreshed
'use server'
export async function deleteProduct(id: string) {
  await db.product.delete({ where: { id } })
}

// ✅ Good: Revalidate affected routes
'use server'
export async function deleteProduct(id: string) {
  await db.product.delete({ where: { id } })
  revalidatePath('/products')
}
```

### React + Vite Pitfalls

**Pitfall**: Not splitting large dependencies
```tsx
// ❌ Bad: Chart library in main bundle
import Chart from 'chart.js'

// ✅ Good: Lazy load chart component
const ChartComponent = lazy(() => import('./ChartComponent'))
```

### Node.js API Pitfalls

**Pitfall**: Missing input validation
```ts
// ❌ Bad: Trust user input
app.post('/users', async (req, res) => {
  const user = await db.user.create(req.body) // SQL injection risk
})

// ✅ Good: Validate with Zod
app.post('/users', async (req, res) => {
  const data = createUserSchema.parse(req.body)
  const user = await db.user.create(data)
})
```

---

## Migration Paths

### JavaScript → TypeScript
1. Add TypeScript dependencies
2. Rename `.js` → `.ts` gradually
3. Enable `allowJs` in tsconfig.json
4. Increase strictness incrementally
5. Add type annotations
6. Enable `strict` mode

### React (webpack) → React (Vite)
1. Install Vite and plugins
2. Create `vite.config.ts`
3. Move `index.html` to root
4. Update import paths (@ alias)
5. Migrate env variables (VITE_ prefix)
6. Update scripts in package.json

### Monolithic → Microservices
1. Identify bounded contexts
2. Extract services one by one
3. Add API gateway/router
4. Implement service discovery
5. Add monitoring and logging
6. Handle distributed transactions

---

## Project Complexity Matrix

| Project Type | Small (<5k LOC) | Medium (5-20k LOC) | Large (>20k LOC) |
|--------------|-----------------|-------------------|------------------|
| **Next.js** | App Router + API | +Middleware +Auth | +Monorepo +Edge |
| **React+Vite** | React Router | +State Mgmt | +Code Splitting |
| **Node.js API** | Express | +Testing +Docs | +Microservices |
| **Library** | Single export | +Multiple APIs | +Plugin System |
| **CLI** | Commander | +Interactive | +Plugins |

---

## Quick Start Checklist

Before scaffolding, answer these questions:

- [ ] What is the primary use case? (Web app, API, library, tool)
- [ ] Do you need server-side rendering?
- [ ] Will this be published to npm?
- [ ] What is the expected team size? (1, 2-5, 6+)
- [ ] What is the expected codebase size? (<5k, 5-20k, >20k LOC)
- [ ] Do you need authentication?
- [ ] Do you need a database?
- [ ] What is your deployment target? (Vercel, AWS, self-hosted)

---

## Related Documentation

- [Next.js Scaffolding Guide](nextjs-scaffolding.md)
- [Node.js API Scaffolding Guide](nodejs-api-scaffolding.md)
- [Library & CLI Scaffolding Guide](library-cli-scaffolding.md)
- [TypeScript Configuration Guide](typescript-configuration.md)
- [Development Tooling Setup](development-tooling.md)
