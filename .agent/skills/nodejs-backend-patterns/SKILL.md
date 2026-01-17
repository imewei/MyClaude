---
name: nodejs-backend-patterns
version: "1.0.7"
description: Build scalable Node.js backends with Express/Fastify/NestJS. Implement middleware, authentication, database integration, and API design. Use when creating REST APIs, microservices, or backend services.
---

# Node.js Backend Patterns

Production-ready Node.js backend development with modern frameworks and architectural patterns.

## Framework Selection

| Framework | Use Case | Performance |
|-----------|----------|-------------|
| Express | General APIs | Moderate |
| Fastify | High performance | Fast |
| NestJS | Enterprise, DI | Moderate |
| Koa | Minimal, async | Fast |

## Express Setup

```typescript
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';

const app = express();
app.use(helmet());
app.use(cors({ origin: process.env.ALLOWED_ORIGINS?.split(',') }));
app.use(compression());
app.use(express.json({ limit: '10mb' }));

app.listen(3000);
```

## Layered Architecture

```
src/
├── controllers/     # HTTP handlers
├── services/        # Business logic
├── repositories/    # Data access
├── middleware/      # Request processing
├── routes/          # Route definitions
└── utils/           # Helpers, errors
```

### Controller

```typescript
export class UserController {
  constructor(private userService: UserService) {}

  async createUser(req: Request, res: Response, next: NextFunction) {
    try {
      const user = await this.userService.createUser(req.body);
      res.status(201).json(user);
    } catch (error) {
      next(error);
    }
  }
}
```

### Service

```typescript
export class UserService {
  constructor(private userRepository: UserRepository) {}

  async createUser(userData: CreateUserDTO): Promise<User> {
    const existing = await this.userRepository.findByEmail(userData.email);
    if (existing) throw new ValidationError('Email exists');

    const hashedPassword = await bcrypt.hash(userData.password, 10);
    return this.userRepository.create({ ...userData, password: hashedPassword });
  }
}
```

### Repository

```typescript
export class UserRepository {
  constructor(private db: Pool) {}

  async findByEmail(email: string): Promise<User | null> {
    const { rows } = await this.db.query('SELECT * FROM users WHERE email = $1', [email]);
    return rows[0] || null;
  }

  async create(user: CreateUserDTO): Promise<User> {
    const { rows } = await this.db.query(
      'INSERT INTO users (name, email, password) VALUES ($1, $2, $3) RETURNING *',
      [user.name, user.email, user.password]
    );
    return rows[0];
  }
}
```

## Middleware Patterns

### Authentication

```typescript
export const authenticate = async (req: Request, res: Response, next: NextFunction) => {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) throw new UnauthorizedError('No token');

  const payload = jwt.verify(token, process.env.JWT_SECRET!) as JWTPayload;
  req.user = payload;
  next();
};
```

### Validation (Zod)

```typescript
export const validate = (schema: AnyZodObject) => async (req: Request, res: Response, next: NextFunction) => {
  try {
    await schema.parseAsync({ body: req.body, query: req.query, params: req.params });
    next();
  } catch (error) {
    next(new ValidationError('Validation failed', error.errors));
  }
};
```

### Rate Limiting

```typescript
import rateLimit from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';

export const apiLimiter = rateLimit({
  store: new RedisStore({ client: redis }),
  windowMs: 15 * 60 * 1000,
  max: 100
});
```

## Error Handling

```typescript
// Custom errors
export class AppError extends Error {
  constructor(public message: string, public statusCode: number = 500) {
    super(message);
  }
}
export class ValidationError extends AppError { constructor(msg: string) { super(msg, 400); } }
export class NotFoundError extends AppError { constructor(msg: string) { super(msg, 404); } }
export class UnauthorizedError extends AppError { constructor(msg: string) { super(msg, 401); } }

// Global handler
export const errorHandler = (err: Error, req: Request, res: Response, next: NextFunction) => {
  if (err instanceof AppError) {
    return res.status(err.statusCode).json({ error: err.message });
  }
  logger.error({ error: err.message, stack: err.stack });
  res.status(500).json({ error: 'Internal server error' });
};
```

## Database Connection

```typescript
import { Pool } from 'pg';

export const pool = new Pool({
  host: process.env.DB_HOST,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  max: 20,
  idleTimeoutMillis: 30000
});

// Graceful shutdown
export const closeDatabase = () => pool.end();
```

## JWT Authentication

```typescript
export class AuthService {
  async login(email: string, password: string) {
    const user = await this.userRepository.findByEmail(email);
    if (!user || !await bcrypt.compare(password, user.password)) {
      throw new UnauthorizedError('Invalid credentials');
    }

    return {
      token: jwt.sign({ userId: user.id }, process.env.JWT_SECRET!, { expiresIn: '15m' }),
      refreshToken: jwt.sign({ userId: user.id }, process.env.REFRESH_SECRET!, { expiresIn: '7d' })
    };
  }
}
```

## Caching with Redis

```typescript
export class CacheService {
  async get<T>(key: string): Promise<T | null> {
    const data = await redis.get(key);
    return data ? JSON.parse(data) : null;
  }

  async set(key: string, value: any, ttl?: number): Promise<void> {
    ttl ? await redis.setex(key, ttl, JSON.stringify(value))
        : await redis.set(key, JSON.stringify(value));
  }

  async invalidatePattern(pattern: string): Promise<void> {
    const keys = await redis.keys(pattern);
    if (keys.length) await redis.del(...keys);
  }
}
```

## Best Practices

| Area | Practice |
|------|----------|
| **Structure** | Layered architecture (Controller → Service → Repository) |
| **Validation** | Zod/Joi for input validation |
| **Errors** | Custom error classes, global handler |
| **Auth** | JWT with refresh tokens, bcrypt passwords |
| **Database** | Connection pooling, parameterized queries |
| **Security** | Helmet, CORS, rate limiting, input sanitization |
| **Logging** | Pino/Winston structured logging |
| **Testing** | Supertest for API tests |

## Checklist

- [ ] Layered architecture implemented
- [ ] Input validation middleware
- [ ] Custom error classes
- [ ] Global error handler
- [ ] JWT authentication
- [ ] Database connection pooling
- [ ] Redis caching
- [ ] Rate limiting
- [ ] Health check endpoint
- [ ] Graceful shutdown
