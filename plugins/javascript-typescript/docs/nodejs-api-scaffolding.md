# Node.js API Scaffolding Guide

> **Version:** 1.0.3 | **Category:** Backend API | **Maturity:** 95%

## Complete Node.js API Structure

```
nodejs-api/
├── package.json
├── tsconfig.json
├── .env.example
├── src/
│   ├── index.ts                    # Entry point
│   ├── app.ts                      # App configuration
│   ├── config/
│   │   ├── database.ts
│   │   ├── env.ts
│   │   └── redis.ts
│   ├── routes/
│   │   ├── index.ts
│   │   ├── auth.routes.ts
│   │   ├── users.routes.ts
│   │   └── health.routes.ts
│   ├── controllers/
│   │   ├── auth.controller.ts
│   │   └── users.controller.ts
│   ├── services/
│   │   ├── auth.service.ts
│   │   └── users.service.ts
│   ├── models/
│   │   └── User.ts
│   ├── middleware/
│   │   ├── auth.ts
│   │   ├── validate.ts
│   │   ├── rateLimiter.ts
│   │   └── errorHandler.ts
│   ├── types/
│   │   ├── express.d.ts
│   │   └── index.ts
│   └── utils/
│       ├── logger.ts
│       └── helpers.ts
└── tests/
    ├── setup.ts
    ├── unit/
    └── integration/
```

---

## package.json

```json
{
  "name": "nodejs-api",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "tsx watch --clear-screen=false src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "test": "vitest",
    "test:integration": "vitest run --config vitest.integration.config.ts",
    "lint": "eslint src --ext .ts",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "express": "^4.18.2",
    "dotenv": "^16.4.0",
    "zod": "^3.22.4",
    "jsonwebtoken": "^9.0.2",
    "bcrypt": "^5.1.1",
    "winston": "^3.11.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.21",
    "@types/node": "^20.11.0",
    "@types/jsonwebtoken": "^9.0.5",
    "@types/bcrypt": "^5.0.2",
    "typescript": "^5.3.0",
    "tsx": "^4.7.0",
    "vitest": "^1.2.0",
    "supertest": "^6.3.3",
    "@types/supertest": "^6.0.2",
    "eslint": "^8.56.0",
    "@typescript-eslint/parser": "^6.19.0",
    "@typescript-eslint/eslint-plugin": "^6.19.0"
  }
}
```

---

## Core Files

### src/index.ts

```typescript
import { createApp } from './app.js'
import { env } from './config/env.js'
import { logger } from './utils/logger.js'

const app = createApp()

const server = app.listen(env.PORT, () => {
  logger.info(`Server running on port ${env.PORT}`)
})

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, closing server')
  server.close(() => {
    logger.info('Server closed')
    process.exit(0)
  })
})
```

### src/app.ts

```typescript
import express, { Express } from 'express'
import { authRouter } from './routes/auth.routes.js'
import { usersRouter } from './routes/users.routes.js'
import { healthRouter } from './routes/health.routes.js'
import { errorHandler } from './middleware/errorHandler.js'
import { logger } from './utils/logger.js'

export function createApp(): Express {
  const app = express()

  // Middleware
  app.use(express.json())
  app.use(express.urlencoded({ extended: true }))

  // Logging
  app.use((req, res, next) => {
    logger.info(`${req.method} ${req.path}`)
    next()
  })

  // Routes
  app.use('/health', healthRouter)
  app.use('/api/auth', authRouter)
  app.use('/api/users', usersRouter)

  // Error handling
  app.use(errorHandler)

  return app
}
```

### src/config/env.ts

```typescript
import { z } from 'zod'
import dotenv from 'dotenv'

dotenv.config()

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.string().transform(Number).default('3000'),
  DATABASE_URL: z.string().url(),
  JWT_SECRET: z.string().min(32),
  REDIS_URL: z.string().url().optional(),
})

export const env = envSchema.parse(process.env)
```

---

## Middleware

### src/middleware/auth.ts

```typescript
import { Request, Response, NextFunction } from 'express'
import jwt from 'jsonwebtoken'
import { env } from '../config/env.js'

export interface AuthRequest extends Request {
  user?: { id: string; email: string }
}

export const authenticate = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const token = req.headers.authorization?.split(' ')[1]

    if (!token) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const decoded = jwt.verify(token, env.JWT_SECRET) as {
      id: string
      email: string
    }

    req.user = decoded
    next()
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' })
  }
}
```

### src/middleware/validate.ts

```typescript
import { Request, Response, NextFunction } from 'express'
import { z, ZodError } from 'zod'

export const validate = (schema: z.ZodObject<any, any>) => {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      schema.parse(req.body)
      next()
    } catch (error) {
      if (error instanceof ZodError) {
        return res.status(400).json({
          error: 'Validation failed',
          details: error.errors,
        })
      }
      next(error)
    }
  }
}
```

### src/middleware/errorHandler.ts

```typescript
import { Request, Response, NextFunction } from 'express'
import { logger } from '../utils/logger.js'

export const errorHandler = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  logger.error(err.stack)

  const status = (err as any).status || 500
  const message = err.message || 'Internal server error'

  res.status(status).json({
    error: message,
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  })
}
```

---

## Controller-Service Pattern

### src/controllers/users.controller.ts

```typescript
import { Request, Response, NextFunction } from 'express'
import { UsersService } from '../services/users.service.js'
import { AuthRequest } from '../middleware/auth.js'

export class UsersController {
  constructor(private usersService: UsersService) {}

  async getUsers(req: Request, res: Response, next: NextFunction) {
    try {
      const users = await this.usersService.findAll()
      res.json(users)
    } catch (error) {
      next(error)
    }
  }

  async getUser(req: Request, res: Response, next: NextFunction) {
    try {
      const user = await this.usersService.findById(req.params.id)
      if (!user) {
        return res.status(404).json({ error: 'User not found' })
      }
      res.json(user)
    } catch (error) {
      next(error)
    }
  }

  async updateUser(req: AuthRequest, res: Response, next: NextFunction) {
    try {
      // Authorization check
      if (req.user?.id !== req.params.id) {
        return res.status(403).json({ error: 'Forbidden' })
      }

      const user = await this.usersService.update(req.params.id, req.body)
      res.json(user)
    } catch (error) {
      next(error)
    }
  }
}
```

### src/services/users.service.ts

```typescript
import { db } from '../config/database.js'

export class UsersService {
  async findAll() {
    return db.user.findMany({
      select: { id: true, email: true, createdAt: true },
    })
  }

  async findById(id: string) {
    return db.user.findUnique({
      where: { id },
      select: { id: true, email: true, createdAt: true },
    })
  }

  async update(id: string, data: { email?: string }) {
    return db.user.update({
      where: { id },
      data,
      select: { id: true, email: true },
    })
  }
}
```

---

## Authentication Example

### src/services/auth.service.ts

```typescript
import bcrypt from 'bcrypt'
import jwt from 'jsonwebtoken'
import { db } from '../config/database.js'
import { env } from '../config/env.js'

export class AuthService {
  async register(email: string, password: string) {
    const hashedPassword = await bcrypt.hash(password, 10)

    const user = await db.user.create({
      data: { email, password: hashedPassword },
      select: { id: true, email: true },
    })

    const token = this.generateToken(user)
    return { user, token }
  }

  async login(email: string, password: string) {
    const user = await db.user.findUnique({ where: { email } })

    if (!user) {
      throw new Error('Invalid credentials')
    }

    const valid = await bcrypt.compare(password, user.password)

    if (!valid) {
      throw new Error('Invalid credentials')
    }

    const token = this.generateToken({ id: user.id, email: user.email })
    return { user: { id: user.id, email: user.email }, token }
  }

  private generateToken(payload: { id: string; email: string }) {
    return jwt.sign(payload, env.JWT_SECRET, { expiresIn: '7d' })
  }
}
```

---

## Testing

### tests/integration/users.test.ts

```typescript
import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import request from 'supertest'
import { createApp } from '../src/app.js'

describe('Users API', () => {
  let app: Express.Application
  let token: string

  beforeAll(async () => {
    app = createApp()
    // Login and get token
    const res = await request(app)
      .post('/api/auth/login')
      .send({ email: 'test@example.com', password: 'password' })
    token = res.body.token
  })

  it('should get all users', async () => {
    const res = await request(app)
      .get('/api/users')
      .set('Authorization', `Bearer ${token}`)

    expect(res.status).toBe(200)
    expect(Array.isArray(res.body)).toBe(true)
  })

  it('should get user by id', async () => {
    const res = await request(app)
      .get('/api/users/1')
      .set('Authorization', `Bearer ${token}`)

    expect(res.status).toBe(200)
    expect(res.body).toHaveProperty('id')
    expect(res.body).toHaveProperty('email')
  })
})
```

---

## Fastify Alternative

```typescript
import Fastify from 'fastify'
import { z } from 'zod'

const app = Fastify({ logger: true })

// Schema-based validation
const userSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
})

app.post('/api/users', async (request, reply) => {
  const data = userSchema.parse(request.body)
  // Create user
  return { success: true }
})

app.listen({ port: 3000 })
```

---

## Related Documentation

- [Project Scaffolding Guide](project-scaffolding-guide.md)
- [TypeScript Configuration](typescript-configuration.md)
- [Development Tooling](development-tooling.md)
