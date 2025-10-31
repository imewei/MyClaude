---
name: nodejs-backend-patterns
description: Build production-ready Node.js backend services with Express.js, Fastify, NestJS, and Koa implementing middleware patterns, error handling, authentication, database integration, and API design best practices for scalable architectures. Use when creating or editing Node.js server files (server.ts, app.ts, index.ts), when building REST APIs with Express or Fastify with proper routing and controllers, when implementing GraphQL backends with Apollo Server or Mercurius, when designing microservices architectures with service communication patterns, when implementing middleware chains for request processing, logging, authentication, and error handling, when integrating databases (PostgreSQL, MongoDB, Redis) with connection pooling and query optimization, when setting up authentication and authorization with JWT, OAuth2, session management, and role-based access control (RBAC), when implementing API versioning, rate limiting, and request validation, when building WebSocket servers for real-time features with Socket.io or native WebSockets, when setting up background job processing with Bull, BullMQ, or node-cron, when implementing file uploads and streaming with multer or busboy, when creating API documentation with Swagger/OpenAPI specifications, when designing repository patterns and service layers for clean architecture, when implementing caching strategies with Redis or in-memory caching, when setting up health checks, metrics, and observability with Prometheus or custom endpoints, when implementing graceful shutdown patterns and process management, when configuring CORS, security headers with helmet, and input sanitization, when building event-driven architectures with event emitters or message queues, when setting up database migrations and seeding with Knex, TypeORM, or Prisma, when implementing request/response logging and structured logging with Winston or Pino, when optimizing performance with clustering, load balancing, and horizontal scaling, when handling errors globally with error middleware and custom error classes, or when establishing Node.js backend coding standards and architectural patterns for teams.
---

# Node.js Backend Patterns

Comprehensive guidance for building scalable, maintainable, and production-ready Node.js backend applications with modern frameworks, architectural patterns, and best practices.

## When to Use This Skill

### Server and Application Setup
- Creating or editing main server files: `server.ts`, `app.ts`, `index.ts`, `main.ts`
- Setting up Express.js applications with proper middleware stack
- Configuring Fastify servers with plugins and decorators
- Implementing NestJS modules, controllers, and providers
- Initializing Koa applications with context-based middleware

### API Development and Routing
- Building REST API routes with Express Router or Fastify routing
- Creating controller files for handling HTTP requests
- Implementing route handlers in `routes/`, `controllers/`, or `handlers/` directories
- Designing API versioning strategies (URL path, headers, query params)
- Setting up route parameter validation and type safety
- Implementing nested routes and route grouping

### Middleware Implementation
- Creating custom middleware functions for request/response processing
- Implementing authentication middleware for protected routes
- Setting up logging middleware with Morgan, Winston, or Pino
- Creating error handling middleware with proper error propagation
- Implementing rate limiting middleware with express-rate-limit or fastify-rate-limit
- Setting up CORS middleware with proper configuration
- Creating request validation middleware with Joi, Zod, or class-validator
- Implementing request timing and performance monitoring middleware

### Authentication and Authorization
- Implementing JWT-based authentication with token generation and verification
- Setting up session-based authentication with express-session or connect-redis
- Implementing OAuth2 flows with Passport.js or custom implementations
- Creating role-based access control (RBAC) middleware
- Implementing permission-based authorization
- Setting up refresh token rotation and token blacklisting
- Integrating with authentication providers (Auth0, Firebase, Keycloak)
- Implementing API key authentication for service-to-service communication

### Database Integration
- Setting up database connections with connection pooling
- Implementing repository pattern for database abstraction
- Creating database models with TypeORM, Sequelize, Mongoose, or Prisma
- Writing optimized database queries and preventing N+1 problems
- Implementing database transactions for data consistency
- Setting up database migrations with Knex, TypeORM migrations, or Prisma Migrate
- Creating database seeders for development and testing
- Implementing soft deletes and audit trails
- Working with multiple database connections

### Error Handling
- Creating global error handling middleware
- Implementing custom error classes for different error types
- Setting up async error handling with try/catch or error-first callbacks
- Implementing error logging and alerting
- Creating user-friendly error responses
- Handling validation errors from request validators
- Implementing error recovery strategies
- Setting up error monitoring with Sentry or similar tools

### API Design and Documentation
- Designing RESTful API endpoints following REST principles
- Implementing proper HTTP status codes and methods
- Creating request/response DTOs (Data Transfer Objects)
- Setting up Swagger/OpenAPI documentation with swagger-jsdoc or @nestjs/swagger
- Implementing HATEOAS links in API responses
- Designing consistent API response formats
- Creating API versioning strategies

### GraphQL Backend Development
- Setting up Apollo Server or Mercurius for GraphQL APIs
- Creating GraphQL schemas and resolvers
- Implementing GraphQL subscriptions for real-time updates
- Setting up DataLoader for batching and caching
- Implementing GraphQL authentication and authorization
- Creating custom GraphQL directives
- Setting up GraphQL federation for microservices

### Real-Time Features
- Implementing WebSocket servers with Socket.io
- Creating WebSocket event handlers and namespaces
- Setting up Server-Sent Events (SSE) for real-time updates
- Implementing real-time notifications and messaging
- Broadcasting events to multiple clients
- Implementing WebSocket authentication
- Handling reconnection and connection state management

### Background Jobs and Scheduling
- Setting up job queues with Bull, BullMQ, or Agenda
- Creating background workers for async tasks
- Implementing scheduled jobs with node-cron or node-schedule
- Processing file uploads asynchronously
- Implementing email sending with queues
- Creating retry mechanisms for failed jobs
- Monitoring job queue health and performance

### File Handling
- Implementing file uploads with Multer or Busboy
- Validating file types, sizes, and contents
- Storing files to local filesystem, S3, or cloud storage
- Implementing file streaming for large files
- Creating file download endpoints with proper headers
- Implementing image processing with Sharp
- Setting up file cleanup and garbage collection

### Caching Strategies
- Implementing Redis caching for frequently accessed data
- Creating in-memory caching with node-cache or lru-cache
- Implementing cache invalidation strategies
- Setting up HTTP caching headers (ETag, Cache-Control)
- Creating cache-aside patterns
- Implementing distributed caching for multiple server instances

### Performance Optimization
- Implementing clustering with Node.js cluster module
- Setting up load balancing with PM2 or custom strategies
- Optimizing database queries with indexes and query analysis
- Implementing response compression with compression middleware
- Using streams for processing large datasets
- Implementing pagination for list endpoints
- Optimizing JSON serialization
- Setting up horizontal scaling strategies

### Security Implementation
- Setting up Helmet for security headers
- Implementing input sanitization to prevent XSS
- Preventing SQL injection with parameterized queries
- Setting up CSRF protection
- Implementing rate limiting to prevent abuse
- Validating and sanitizing user input
- Setting up secure password hashing with bcrypt or argon2
- Implementing API security best practices

### Logging and Monitoring
- Setting up structured logging with Winston or Pino
- Implementing request/response logging
- Creating log levels (error, warn, info, debug)
- Setting up log aggregation with ELK stack or similar
- Implementing metrics collection with Prometheus
- Creating health check endpoints (`/health`, `/ready`)
- Setting up application performance monitoring (APM)
- Implementing distributed tracing

### Service Architecture
- Creating service layer for business logic separation
- Implementing repository pattern for data access
- Setting up dependency injection with InversifyJS or NestJS DI
- Creating DTOs and validators for data transfer
- Implementing event-driven architecture with EventEmitter
- Setting up microservices communication (HTTP, gRPC, message queues)
- Implementing circuit breaker patterns for resilience
- Creating saga patterns for distributed transactions

### Testing Backend Services
- Writing unit tests for services and repositories
- Creating integration tests for API endpoints with supertest
- Implementing test fixtures and factories
- Mocking database connections and external services
- Setting up test databases with Docker or in-memory databases
- Creating E2E tests for critical user flows

### Deployment and Operations
- Implementing graceful shutdown patterns
- Setting up process management with PM2
- Creating Docker configurations for Node.js apps
- Implementing environment-based configuration
- Setting up database connection pooling for production
- Implementing zero-downtime deployments
- Creating deployment health checks

## Core Frameworks

### Express.js - Minimalist Framework

**Basic Setup:**
```typescript
import express, { Request, Response, NextFunction } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';

const app = express();

// Security middleware
app.use(helmet());
app.use(cors({ origin: process.env.ALLOWED_ORIGINS?.split(',') }));
app.use(compression());

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging
app.use((req: Request, res: Response, next: NextFunction) => {
  console.log(`${req.method} ${req.path}`);
  next();
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### Fastify - High Performance Framework

**Basic Setup:**
```typescript
import Fastify from 'fastify';
import helmet from '@fastify/helmet';
import cors from '@fastify/cors';
import compress from '@fastify/compress';

const fastify = Fastify({
  logger: {
    level: process.env.LOG_LEVEL || 'info',
    transport: {
      target: 'pino-pretty',
      options: { colorize: true }
    }
  }
});

// Plugins
await fastify.register(helmet);
await fastify.register(cors, { origin: true });
await fastify.register(compress);

// Type-safe routes with schema validation
fastify.post<{
  Body: { name: string; email: string };
  Reply: { id: string; name: string };
}>('/users', {
  schema: {
    body: {
      type: 'object',
      required: ['name', 'email'],
      properties: {
        name: { type: 'string', minLength: 1 },
        email: { type: 'string', format: 'email' }
      }
    }
  }
}, async (request, reply) => {
  const { name, email } = request.body;
  return { id: '123', name };
});

await fastify.listen({ port: 3000, host: '0.0.0.0' });
```

## Architectural Patterns

### Pattern 1: Layered Architecture

**Structure:**
```
src/
├── controllers/     # Handle HTTP requests/responses
├── services/        # Business logic
├── repositories/    # Data access layer
├── models/          # Data models
├── middleware/      # Express/Fastify middleware
├── routes/          # Route definitions
├── utils/           # Helper functions
├── config/          # Configuration
└── types/           # TypeScript types
```

**Controller Layer:**
```typescript
// controllers/user.controller.ts
import { Request, Response, NextFunction } from 'express';
import { UserService } from '../services/user.service';
import { CreateUserDTO, UpdateUserDTO } from '../types/user.types';

export class UserController {
  constructor(private userService: UserService) {}

  async createUser(req: Request, res: Response, next: NextFunction) {
    try {
      const userData: CreateUserDTO = req.body;
      const user = await this.userService.createUser(userData);
      res.status(201).json(user);
    } catch (error) {
      next(error);
    }
  }

  async getUser(req: Request, res: Response, next: NextFunction) {
    try {
      const { id } = req.params;
      const user = await this.userService.getUserById(id);
      res.json(user);
    } catch (error) {
      next(error);
    }
  }

  async updateUser(req: Request, res: Response, next: NextFunction) {
    try {
      const { id } = req.params;
      const updates: UpdateUserDTO = req.body;
      const user = await this.userService.updateUser(id, updates);
      res.json(user);
    } catch (error) {
      next(error);
    }
  }

  async deleteUser(req: Request, res: Response, next: NextFunction) {
    try {
      const { id } = req.params;
      await this.userService.deleteUser(id);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
}
```

**Service Layer:**
```typescript
// services/user.service.ts
import { UserRepository } from '../repositories/user.repository';
import { CreateUserDTO, UpdateUserDTO, User } from '../types/user.types';
import { NotFoundError, ValidationError } from '../utils/errors';
import bcrypt from 'bcrypt';

export class UserService {
  constructor(private userRepository: UserRepository) {}

  async createUser(userData: CreateUserDTO): Promise<User> {
    // Validation
    const existingUser = await this.userRepository.findByEmail(userData.email);
    if (existingUser) {
      throw new ValidationError('Email already exists');
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(userData.password, 10);

    // Create user
    const user = await this.userRepository.create({
      ...userData,
      password: hashedPassword
    });

    // Remove password from response
    const { password, ...userWithoutPassword } = user;
    return userWithoutPassword as User;
  }

  async getUserById(id: string): Promise<User> {
    const user = await this.userRepository.findById(id);
    if (!user) {
      throw new NotFoundError('User not found');
    }
    const { password, ...userWithoutPassword } = user;
    return userWithoutPassword as User;
  }

  async updateUser(id: string, updates: UpdateUserDTO): Promise<User> {
    const user = await this.userRepository.update(id, updates);
    if (!user) {
      throw new NotFoundError('User not found');
    }
    const { password, ...userWithoutPassword } = user;
    return userWithoutPassword as User;
  }

  async deleteUser(id: string): Promise<void> {
    const deleted = await this.userRepository.delete(id);
    if (!deleted) {
      throw new NotFoundError('User not found');
    }
  }
}
```

**Repository Layer:**
```typescript
// repositories/user.repository.ts
import { Pool } from 'pg';
import { CreateUserDTO, UpdateUserDTO, UserEntity } from '../types/user.types';

export class UserRepository {
  constructor(private db: Pool) {}

  async create(userData: CreateUserDTO & { password: string }): Promise<UserEntity> {
    const query = `
      INSERT INTO users (name, email, password)
      VALUES ($1, $2, $3)
      RETURNING id, name, email, password, created_at, updated_at
    `;
    const { rows } = await this.db.query(query, [
      userData.name,
      userData.email,
      userData.password
    ]);
    return rows[0];
  }

  async findById(id: string): Promise<UserEntity | null> {
    const query = 'SELECT * FROM users WHERE id = $1';
    const { rows } = await this.db.query(query, [id]);
    return rows[0] || null;
  }

  async findByEmail(email: string): Promise<UserEntity | null> {
    const query = 'SELECT * FROM users WHERE email = $1';
    const { rows } = await this.db.query(query, [email]);
    return rows[0] || null;
  }

  async update(id: string, updates: UpdateUserDTO): Promise<UserEntity | null> {
    const fields = Object.keys(updates);
    const values = Object.values(updates);

    const setClause = fields
      .map((field, idx) => `${field} = $${idx + 2}`)
      .join(', ');

    const query = `
      UPDATE users
      SET ${setClause}, updated_at = CURRENT_TIMESTAMP
      WHERE id = $1
      RETURNING *
    `;

    const { rows } = await this.db.query(query, [id, ...values]);
    return rows[0] || null;
  }

  async delete(id: string): Promise<boolean> {
    const query = 'DELETE FROM users WHERE id = $1';
    const { rowCount } = await this.db.query(query, [id]);
    return rowCount > 0;
  }
}
```

### Pattern 2: Dependency Injection

**DI Container:**
```typescript
// di-container.ts
import { Pool } from 'pg';
import { UserRepository } from './repositories/user.repository';
import { UserService } from './services/user.service';
import { UserController } from './controllers/user.controller';
import { AuthService } from './services/auth.service';

class Container {
  private instances = new Map<string, any>();

  register<T>(key: string, factory: () => T): void {
    this.instances.set(key, factory);
  }

  resolve<T>(key: string): T {
    const factory = this.instances.get(key);
    if (!factory) {
      throw new Error(`No factory registered for ${key}`);
    }
    return factory();
  }

  singleton<T>(key: string, factory: () => T): void {
    let instance: T;
    this.instances.set(key, () => {
      if (!instance) {
        instance = factory();
      }
      return instance;
    });
  }
}

export const container = new Container();

// Register dependencies
container.singleton('db', () => new Pool({
  host: process.env.DB_HOST,
  port: parseInt(process.env.DB_PORT || '5432'),
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
}));

container.singleton('userRepository', () =>
  new UserRepository(container.resolve('db'))
);

container.singleton('userService', () =>
  new UserService(container.resolve('userRepository'))
);

container.register('userController', () =>
  new UserController(container.resolve('userService'))
);

container.singleton('authService', () =>
  new AuthService(container.resolve('userRepository'))
);
```

## Middleware Patterns

### Authentication Middleware

```typescript
// middleware/auth.middleware.ts
import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { UnauthorizedError } from '../utils/errors';

interface JWTPayload {
  userId: string;
  email: string;
}

declare global {
  namespace Express {
    interface Request {
      user?: JWTPayload;
    }
  }
}

export const authenticate = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');

    if (!token) {
      throw new UnauthorizedError('No token provided');
    }

    const payload = jwt.verify(
      token,
      process.env.JWT_SECRET!
    ) as JWTPayload;

    req.user = payload;
    next();
  } catch (error) {
    next(new UnauthorizedError('Invalid token'));
  }
};

export const authorize = (...roles: string[]) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    if (!req.user) {
      return next(new UnauthorizedError('Not authenticated'));
    }

    // Check if user has required role
    const hasRole = roles.some(role =>
      req.user?.roles?.includes(role)
    );

    if (!hasRole) {
      return next(new UnauthorizedError('Insufficient permissions'));
    }

    next();
  };
};
```

### Validation Middleware

```typescript
// middleware/validation.middleware.ts
import { Request, Response, NextFunction } from 'express';
import { AnyZodObject, ZodError } from 'zod';
import { ValidationError } from '../utils/errors';

export const validate = (schema: AnyZodObject) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      await schema.parseAsync({
        body: req.body,
        query: req.query,
        params: req.params
      });
      next();
    } catch (error) {
      if (error instanceof ZodError) {
        const errors = error.errors.map(err => ({
          field: err.path.join('.'),
          message: err.message
        }));
        next(new ValidationError('Validation failed', errors));
      } else {
        next(error);
      }
    }
  };
};

// Usage with Zod
import { z } from 'zod';

const createUserSchema = z.object({
  body: z.object({
    name: z.string().min(1),
    email: z.string().email(),
    password: z.string().min(8)
  })
});

router.post('/users', validate(createUserSchema), userController.createUser);
```

### Rate Limiting Middleware

```typescript
// middleware/rate-limit.middleware.ts
import rateLimit from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';
import Redis from 'ioredis';

const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT || '6379')
});

export const apiLimiter = rateLimit({
  store: new RedisStore({
    client: redis,
    prefix: 'rl:',
  }),
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

export const authLimiter = rateLimit({
  store: new RedisStore({
    client: redis,
    prefix: 'rl:auth:',
  }),
  windowMs: 15 * 60 * 1000,
  max: 5, // Stricter limit for auth endpoints
  skipSuccessfulRequests: true,
});
```

### Request Logging Middleware

```typescript
// middleware/logger.middleware.ts
import { Request, Response, NextFunction } from 'express';
import pino from 'pino';

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: {
    target: 'pino-pretty',
    options: { colorize: true }
  }
});

export const requestLogger = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const start = Date.now();

  // Log response when finished
  res.on('finish', () => {
    const duration = Date.now() - start;
    logger.info({
      method: req.method,
      url: req.url,
      status: res.statusCode,
      duration: `${duration}ms`,
      userAgent: req.headers['user-agent'],
      ip: req.ip
    });
  });

  next();
};

export { logger };
```

## Error Handling

### Custom Error Classes

```typescript
// utils/errors.ts
export class AppError extends Error {
  constructor(
    public message: string,
    public statusCode: number = 500,
    public isOperational: boolean = true
  ) {
    super(message);
    Object.setPrototypeOf(this, AppError.prototype);
    Error.captureStackTrace(this, this.constructor);
  }
}

export class ValidationError extends AppError {
  constructor(message: string, public errors?: any[]) {
    super(message, 400);
  }
}

export class NotFoundError extends AppError {
  constructor(message: string = 'Resource not found') {
    super(message, 404);
  }
}

export class UnauthorizedError extends AppError {
  constructor(message: string = 'Unauthorized') {
    super(message, 401);
  }
}

export class ForbiddenError extends AppError {
  constructor(message: string = 'Forbidden') {
    super(message, 403);
  }
}

export class ConflictError extends AppError {
  constructor(message: string) {
    super(message, 409);
  }
}
```

### Global Error Handler

```typescript
// middleware/error-handler.ts
import { Request, Response, NextFunction } from 'express';
import { AppError } from '../utils/errors';
import { logger } from './logger.middleware';

export const errorHandler = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  if (err instanceof AppError) {
    return res.status(err.statusCode).json({
      status: 'error',
      message: err.message,
      ...(err instanceof ValidationError && { errors: err.errors })
    });
  }

  // Log unexpected errors
  logger.error({
    error: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method
  });

  // Don't leak error details in production
  const message = process.env.NODE_ENV === 'production'
    ? 'Internal server error'
    : err.message;

  res.status(500).json({
    status: 'error',
    message
  });
};

// Async error wrapper
export const asyncHandler = (
  fn: (req: Request, res: Response, next: NextFunction) => Promise<any>
) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};
```

## Database Patterns

### PostgreSQL with Connection Pool

```typescript
// config/database.ts
import { Pool, PoolConfig } from 'pg';

const poolConfig: PoolConfig = {
  host: process.env.DB_HOST,
  port: parseInt(process.env.DB_PORT || '5432'),
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
};

export const pool = new Pool(poolConfig);

// Test connection
pool.on('connect', () => {
  console.log('Database connected');
});

pool.on('error', (err) => {
  console.error('Unexpected database error', err);
  process.exit(-1);
});

// Graceful shutdown
export const closeDatabase = async () => {
  await pool.end();
  console.log('Database connection closed');
};
```

### MongoDB with Mongoose

```typescript
// config/mongoose.ts
import mongoose from 'mongoose';

const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI!, {
      maxPoolSize: 10,
      serverSelectionTimeoutMS: 5000,
      socketTimeoutMS: 45000,
    });

    console.log('MongoDB connected');
  } catch (error) {
    console.error('MongoDB connection error:', error);
    process.exit(1);
  }
};

mongoose.connection.on('disconnected', () => {
  console.log('MongoDB disconnected');
});

mongoose.connection.on('error', (err) => {
  console.error('MongoDB error:', err);
});

export { connectDB };

// Model example
import { Schema, model, Document } from 'mongoose';

interface IUser extends Document {
  name: string;
  email: string;
  password: string;
  createdAt: Date;
  updatedAt: Date;
}

const userSchema = new Schema<IUser>({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
}, {
  timestamps: true
});

// Indexes
userSchema.index({ email: 1 });

export const User = model<IUser>('User', userSchema);
```

### Transaction Pattern

```typescript
// services/order.service.ts
import { Pool } from 'pg';

export class OrderService {
  constructor(private db: Pool) {}

  async createOrder(userId: string, items: any[]) {
    const client = await this.db.connect();

    try {
      await client.query('BEGIN');

      // Create order
      const orderResult = await client.query(
        'INSERT INTO orders (user_id, total) VALUES ($1, $2) RETURNING id',
        [userId, calculateTotal(items)]
      );
      const orderId = orderResult.rows[0].id;

      // Create order items
      for (const item of items) {
        await client.query(
          'INSERT INTO order_items (order_id, product_id, quantity, price) VALUES ($1, $2, $3, $4)',
          [orderId, item.productId, item.quantity, item.price]
        );

        // Update inventory
        await client.query(
          'UPDATE products SET stock = stock - $1 WHERE id = $2',
          [item.quantity, item.productId]
        );
      }

      await client.query('COMMIT');
      return orderId;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }
}
```

## Authentication & Authorization

### JWT Authentication

```typescript
// services/auth.service.ts
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import { UserRepository } from '../repositories/user.repository';
import { UnauthorizedError } from '../utils/errors';

export class AuthService {
  constructor(private userRepository: UserRepository) {}

  async login(email: string, password: string) {
    const user = await this.userRepository.findByEmail(email);

    if (!user) {
      throw new UnauthorizedError('Invalid credentials');
    }

    const isValid = await bcrypt.compare(password, user.password);

    if (!isValid) {
      throw new UnauthorizedError('Invalid credentials');
    }

    const token = this.generateToken({
      userId: user.id,
      email: user.email
    });

    const refreshToken = this.generateRefreshToken({
      userId: user.id
    });

    return {
      token,
      refreshToken,
      user: {
        id: user.id,
        name: user.name,
        email: user.email
      }
    };
  }

  async refreshToken(refreshToken: string) {
    try {
      const payload = jwt.verify(
        refreshToken,
        process.env.REFRESH_TOKEN_SECRET!
      ) as { userId: string };

      const user = await this.userRepository.findById(payload.userId);

      if (!user) {
        throw new UnauthorizedError('User not found');
      }

      const token = this.generateToken({
        userId: user.id,
        email: user.email
      });

      return { token };
    } catch (error) {
      throw new UnauthorizedError('Invalid refresh token');
    }
  }

  private generateToken(payload: any): string {
    return jwt.sign(payload, process.env.JWT_SECRET!, {
      expiresIn: '15m'
    });
  }

  private generateRefreshToken(payload: any): string {
    return jwt.sign(payload, process.env.REFRESH_TOKEN_SECRET!, {
      expiresIn: '7d'
    });
  }
}
```

## Caching Strategies

```typescript
// utils/cache.ts
import Redis from 'ioredis';

const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT || '6379'),
  retryStrategy: (times) => {
    const delay = Math.min(times * 50, 2000);
    return delay;
  }
});

export class CacheService {
  async get<T>(key: string): Promise<T | null> {
    const data = await redis.get(key);
    return data ? JSON.parse(data) : null;
  }

  async set(key: string, value: any, ttl?: number): Promise<void> {
    const serialized = JSON.stringify(value);
    if (ttl) {
      await redis.setex(key, ttl, serialized);
    } else {
      await redis.set(key, serialized);
    }
  }

  async delete(key: string): Promise<void> {
    await redis.del(key);
  }

  async invalidatePattern(pattern: string): Promise<void> {
    const keys = await redis.keys(pattern);
    if (keys.length > 0) {
      await redis.del(...keys);
    }
  }
}

// Cache decorator
export function Cacheable(ttl: number = 300) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      const cache = new CacheService();
      const cacheKey = `${propertyKey}:${JSON.stringify(args)}`;

      const cached = await cache.get(cacheKey);
      if (cached) {
        return cached;
      }

      const result = await originalMethod.apply(this, args);
      await cache.set(cacheKey, result, ttl);

      return result;
    };

    return descriptor;
  };
}
```

## API Response Format

```typescript
// utils/response.ts
import { Response } from 'express';

export class ApiResponse {
  static success<T>(res: Response, data: T, message?: string, statusCode = 200) {
    return res.status(statusCode).json({
      status: 'success',
      message,
      data
    });
  }

  static error(res: Response, message: string, statusCode = 500, errors?: any) {
    return res.status(statusCode).json({
      status: 'error',
      message,
      ...(errors && { errors })
    });
  }

  static paginated<T>(
    res: Response,
    data: T[],
    page: number,
    limit: number,
    total: number
  ) {
    return res.json({
      status: 'success',
      data,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    });
  }
}
```

## Best Practices

1. **Use TypeScript**: Type safety prevents runtime errors
2. **Implement proper error handling**: Use custom error classes
3. **Validate input**: Use libraries like Zod or Joi
4. **Use environment variables**: Never hardcode secrets
5. **Implement logging**: Use structured logging (Pino, Winston)
6. **Add rate limiting**: Prevent abuse
7. **Use HTTPS**: Always in production
8. **Implement CORS properly**: Don't use `*` in production
9. **Use dependency injection**: Easier testing and maintenance
10. **Write tests**: Unit, integration, and E2E tests
11. **Handle graceful shutdown**: Clean up resources
12. **Use connection pooling**: For databases
13. **Implement health checks**: For monitoring
14. **Use compression**: Reduce response size
15. **Monitor performance**: Use APM tools

## Testing Patterns

See `javascript-testing-patterns` skill for comprehensive testing guidance.

## Resources

- **Node.js Best Practices**: https://github.com/goldbergyoni/nodebestpractices
- **Express.js Guide**: https://expressjs.com/en/guide/
- **Fastify Documentation**: https://www.fastify.io/docs/
- **TypeScript Node Starter**: https://github.com/microsoft/TypeScript-Node-Starter
