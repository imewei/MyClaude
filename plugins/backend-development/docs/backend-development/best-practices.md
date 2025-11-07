# Feature Development Best Practices

Production-ready best practices for building reliable, scalable, and maintainable features.

## Table of Contents
- [Production Readiness](#production-readiness)
- [Feature Flags](#feature-flags)
- [Observability](#observability)
- [Security](#security)
- [Performance](#performance)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Production Readiness

### Production Readiness Checklist

```markdown
## Code Quality
- [ ] Code review completed by 2+ engineers
- [ ] No commented-out code or TODOs in production code
- [ ] Error handling for all external calls
- [ ] Input validation on all user inputs
- [ ] Logging with appropriate levels (info, warn, error)
- [ ] No hardcoded secrets or credentials

## Testing
- [ ] Unit test coverage ≥ 80%
- [ ] Integration tests for critical paths
- [ ] E2E tests for user journeys
- [ ] Performance tests completed
- [ ] Security scan (SAST/DAST) passed
- [ ] Load testing completed

## Infrastructure
- [ ] Health check endpoint implemented
- [ ] Readiness probe configured
- [ ] Resource limits defined (CPU, memory)
- [ ] Auto-scaling configured
- [ ] Database indexes created
- [ ] Connection pooling configured

## Observability
- [ ] Metrics instrumented (request rate, latency, errors)
- [ ] Distributed tracing configured
- [ ] Logs structured (JSON format)
- [ ] Alerts configured for critical metrics
- [ ] Dashboards created
- [ ] Runbooks written

## Security
- [ ] Authentication implemented
- [ ] Authorization enforced
- [ ] Input sanitization applied
- [ ] SQL injection prevention verified
- [ ] XSS prevention verified
- [ ] CSRF protection enabled
- [ ] Security headers configured
- [ ] Secrets in vault/secret manager

## Deployment
- [ ] Feature flags configured
- [ ] Rollback procedure documented
- [ ] Database migrations tested (up and down)
- [ ] Deployment runbook created
- [ ] Smoke tests defined
- [ ] Canary deployment plan (if applicable)

## Documentation
- [ ] API documentation updated
- [ ] Architecture diagrams current
- [ ] README updated
- [ ] Changelog entry added
- [ ] User-facing documentation updated
```

---

## Feature Flags

### Flag Lifecycle Management

**1. Creation**
```typescript
// Create flag with metadata
const flagConfig = {
  key: 'new-checkout-flow',
  name: 'New Checkout Flow',
  description: 'Redesigned checkout with one-click purchase',
  tags: ['checkout', 'revenue', 'q1-2024'],
  owner: 'payments-team',
  defaultValue: false,
  variations: {
    enabled: { value: true, description: 'New checkout enabled' },
    disabled: { value: false, description: 'Old checkout' }
  }
};
```

**2. Targeting Strategy**
```yaml
targeting:
  - name: Beta Testers
    enabled: true
    users:
      - beta-user-1@example.com
      - beta-user-2@example.com

  - name: Premium Tier
    percentage: 50
    conditions:
      - attribute: tier
        operator: equals
        value: premium

  - name: Gradual Rollout
    percentage: 10  # Start with 10%, increase gradually
```

**3. Monitoring**
```typescript
// Track flag evaluations
analytics.track('feature_flag_evaluated', {
  flagKey: 'new-checkout-flow',
  value: flagValue,
  userId: user.id,
  timestamp: new Date()
});

// Monitor flag usage
const metrics = await getFlag Metrics('new-checkout-flow');
// {
//   evaluations: 125000,
//   uniqueUsers: 45000,
//   variantDistribution: { enabled: 4500, disabled: 40500 },
//   conversionRate: { enabled: 15.2%, disabled: 12.4% }
// }
```

**4. Cleanup**
```typescript
// After 100% rollout for 2 weeks, remove flag
// 1. Change flag to always return true
// 2. Deploy and monitor for 1 week
// 3. Remove flag checks from code
// 4. Deploy cleaned code
// 5. Archive flag in flag management system

// Before:
if (await ldClient.variation('new-checkout-flow', user, false)) {
  return newCheckoutFlow();
} else {
  return oldCheckoutFlow();
}

// After cleanup:
return newCheckoutFlow();  // Old code removed
```

---

## Observability

### Three Pillars: Metrics, Logs, Traces

**Metrics (RED Method)**
```typescript
// Rate: Requests per second
const requestRate = new promClient.Counter({
  name: 'http_requests_total',
  help: 'Total HTTP requests',
  labelNames: ['method', 'route', 'status']
});

// Errors: Error rate
const errorRate = new promClient.Counter({
  name: 'http_errors_total',
  help: 'Total HTTP errors',
  labelNames: ['method', 'route', 'error_type']
});

// Duration: Request latency
const requestDuration = new promClient.Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration',
  labelNames: ['method', 'route'],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
});

// Middleware
app.use((req, res, next) => {
  const start = Date.now();

  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    requestRate.inc({ method: req.method, route: req.route?.path, status: res.statusCode });
    requestDuration.observe({ method: req.method, route: req.route?.path }, duration);

    if (res.statusCode >= 400) {
      errorRate.inc({ method: req.method, route: req.route?.path, error_type: getErrorType(res.statusCode) });
    }
  });

  next();
});
```

**Structured Logging**
```typescript
import winston from 'winston';

const logger = winston.createLogger({
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: {
    service: 'feature-service',
    environment: process.env.NODE_ENV
  },
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

// Usage
logger.info('User checkout initiated', {
  userId: user.id,
  cartValue: cart.total,
  itemCount: cart.items.length,
  correlationId: req.headers['x-correlation-id']
});
```

**Distributed Tracing**
```typescript
import { trace, context } from '@opentelemetry/api';

const tracer = trace.getTracer('feature-service');

app.get('/api/checkout', async (req, res) => {
  const span = tracer.startSpan('checkout', {
    attributes: {
      'user.id': req.user.id,
      'http.method': req.method,
      'http.route': req.route.path
    }
  });

  try {
    // Child span for database call
    const dbSpan = tracer.startSpan('database.query', { parent: span });
    const cart = await getCart(req.user.id);
    dbSpan.end();

    // Child span for payment processing
    const paymentSpan = tracer.startSpan('payment.process', { parent: span });
    const payment = await processPayment(cart);
    paymentSpan.end();

    span.setStatus({ code: SpanStatusCode.OK });
    res.json({ success: true, orderId: payment.id });
  } catch (error) {
    span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
    span.recordException(error);
    throw error;
  } finally {
    span.end();
  }
});
```

---

## Security

### Input Validation

**Backend Validation**
```typescript
import { z } from 'zod';

const CreateUserSchema = z.object({
  email: z.string().email().max(255),
  password: z.string().min(8).max(72).regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])/),
  name: z.string().min(1).max(100).regex(/^[a-zA-Z\s-']+$/),
  age: z.number().int().min(13).max(120)
});

app.post('/api/users', async (req, res) => {
  try {
    const validatedData = CreateUserSchema.parse(req.body);
    const user = await createUser(validatedData);
    res.status(201).json(user);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        error: 'Validation failed',
        details: error.errors
      });
    }
    throw error;
  }
});
```

### SQL Injection Prevention
```typescript
// ❌ VULNERABLE
const query = `SELECT * FROM users WHERE email = '${req.body.email}'`;

// ✅ SECURE (Parameterized queries)
const query = 'SELECT * FROM users WHERE email = $1';
const result = await db.query(query, [req.body.email]);

// ✅ SECURE (ORM)
const user = await User.findOne({ where: { email: req.body.email } });
```

### Authentication & Authorization
```typescript
import jwt from 'jsonwebtoken';

// Authentication middleware
const authenticate = async (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'No token provided' });

  try {
    const payload = jwt.verify(token, process.env.JWT_SECRET);
    req.user = await User.findById(payload.userId);
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};

// Authorization middleware
const authorize = (...roles) => {
  return (req, res, next) => {
    if (!req.user) return res.status(401).json({ error: 'Unauthorized' });
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Forbidden' });
    }
    next();
  };
};

// Usage
app.delete('/api/users/:id', authenticate, authorize('admin'), async (req, res) => {
  await User.delete(req.params.id);
  res.status(204).send();
});
```

---

## Performance

### Database Optimization

**Indexes**
```sql
-- Create indexes for frequently queried columns
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);

-- Composite index for common query patterns
CREATE INDEX idx_orders_user_status ON orders(user_id, status, created_at DESC);
```

**Query Optimization**
```typescript
// ❌ N+1 Query Problem
const users = await User.findAll();
for (const user of users) {
  user.orders = await Order.findAll({ where: { userId: user.id } });
}

// ✅ Eager Loading
const users = await User.findAll({
  include: [{ model: Order, as: 'orders' }]
});
```

### Caching Strategy
```typescript
import Redis from 'ioredis';
const redis = new Redis(process.env.REDIS_URL);

// Multi-tier caching
class CacheService {
  private memoryCache = new Map<string, { value: any; expiresAt: number }>();

  async get(key: string): Promise<any> {
    // L1: In-memory cache (fastest)
    const memoryCached = this.memoryCache.get(key);
    if (memoryCached && memoryCached.expiresAt > Date.now()) {
      return memoryCached.value;
    }

    // L2: Redis cache
    const redisCached = await redis.get(key);
    if (redisCached) {
      const value = JSON.parse(redisCached);
      this.memoryCache.set(key, { value, expiresAt: Date.now() + 60000 }); // 1min
      return value;
    }

    return null;
  }

  async set(key: string, value: any, ttlSeconds: number): Promise<void> {
    // Set in both caches
    this.memoryCache.set(key, {
      value,
      expiresAt: Date.now() + Math.min(ttlSeconds * 1000, 60000)
    });
    await redis.setex(key, ttlSeconds, JSON.stringify(value));
  }
}

// Usage
const cache = new CacheService();

app.get('/api/products/:id', async (req, res) => {
  const cacheKey = `product:${req.params.id}`;

  // Try cache first
  let product = await cache.get(cacheKey);

  if (!product) {
    // Cache miss - fetch from database
    product = await Product.findById(req.params.id);
    await cache.set(cacheKey, product, 300); // Cache for 5 minutes
  }

  res.json(product);
});
```

---

## Testing

### Test Pyramid

```
       /\
      /E2E\     10% - End-to-End Tests (critical user journeys)
     /______\
    /        \
   /Integration\  20% - Integration Tests (API, DB, external services)
  /____________\
 /              \
/   Unit Tests   \  70% - Unit Tests (business logic, utilities)
/__________________\
```

### Testing Best Practices

**Unit Tests**
```typescript
import { describe, it, expect } from 'vitest';

describe('OrderService', () => {
  it('should calculate total with tax correctly', () => {
    const order = new Order([
      { price: 100, quantity: 2 },
      { price: 50, quantity: 1 }
    ]);

    const total = order.calculateTotal({ taxRate: 0.1 });

    expect(total).toBe(275); // (100*2 + 50) * 1.1
  });

  it('should throw error when items array is empty', () => {
    expect(() => new Order([])).toThrow('Order must contain at least one item');
  });
});
```

**Integration Tests**
```typescript
import request from 'supertest';
import app from '../app';

describe('POST /api/orders', () => {
  it('should create order and return 201', async () => {
    const response = await request(app)
      .post('/api/orders')
      .set('Authorization', `Bearer ${authToken}`)
      .send({
        items: [{ productId: 'prod-123', quantity: 2 }]
      });

    expect(response.status).toBe(201);
    expect(response.body).toHaveProperty('orderId');
    expect(response.body.status).toBe('pending');
  });
});
```

---

## Documentation

### API Documentation (OpenAPI)
```yaml
openapi: 3.0.0
info:
  title: Feature API
  version: 1.0.0
paths:
  /api/users:
    post:
      summary: Create a new user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
            examples:
              basic:
                value:
                  email: user@example.com
                  password: SecurePass123!
                  name: John Doe
      responses:
        '201':
          description: User created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
```

### README Template
```markdown
# Feature Name

## Overview
Brief description of the feature and its business value.

## Architecture
High-level architecture diagram and design decisions.

## Getting Started

### Prerequisites
- Node.js 20+
- PostgreSQL 16+
- Redis 7+

### Installation
```bash
npm install
cp .env.example .env
npm run migrate
npm run dev
```

## Usage Examples
```typescript
// Example code showing common use cases
```

## API Documentation
Link to Swagger/Postman docs

## Testing
```bash
npm test
npm run test:integration
npm run test:e2e
```

## Deployment
See [deployment runbook](./docs/deployment.md)

## Troubleshooting
Common issues and solutions

## Contributing
Guidelines for contributing
```

---

## Quick Reference

### Pre-Deployment Checklist
```bash
# 1. Code quality
npm run lint
npm run type-check

# 2. Tests
npm test
npm run test:integration

# 3. Security
npm audit
npm run security-scan

# 4. Build
npm run build

# 5. Smoke tests
npm run smoke-test

# 6. Deploy
npm run deploy:staging
# Test in staging
npm run deploy:production
```

### Post-Deployment Monitoring
- ✅ Error rate < 0.5%
- ✅ p95 latency < 200ms
- ✅ CPU usage < 70%
- ✅ Memory usage < 80%
- ✅ No critical alerts
- ✅ Feature flags working
- ✅ Metrics being collected
