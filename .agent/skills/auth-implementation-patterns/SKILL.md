---
name: auth-implementation-patterns
version: "1.0.7"
description: Master authentication patterns including JWT (access/refresh tokens), OAuth2/OpenID Connect, session-based auth, RBAC/ABAC, API keys, MFA, SSO, password hashing (bcrypt/argon2), secure cookies, and CSRF protection. Use when implementing auth systems, securing APIs, or designing access control.
---

# Authentication & Authorization Patterns

Secure, scalable authentication and authorization systems.

## Strategy Selection

| Strategy | Use Case | Stateful |
|----------|----------|----------|
| Session-based | Traditional web apps | Yes (server) |
| JWT | SPAs, mobile, microservices | No (stateless) |
| OAuth2/OIDC | Social login, SSO | Delegated |
| API Key | Third-party integrations | No |

## JWT Implementation

### Token Generation
```typescript
function generateTokens(userId: string, email: string, role: string) {
    const accessToken = jwt.sign(
        { userId, email, role },
        process.env.JWT_SECRET!,
        { expiresIn: '15m' }
    );
    const refreshToken = jwt.sign(
        { userId },
        process.env.JWT_REFRESH_SECRET!,
        { expiresIn: '7d' }
    );
    return { accessToken, refreshToken };
}
```

### Auth Middleware
```typescript
function authenticate(req: Request, res: Response, next: NextFunction) {
    const authHeader = req.headers.authorization;
    if (!authHeader?.startsWith('Bearer '))
        return res.status(401).json({ error: 'No token' });

    try {
        req.user = jwt.verify(authHeader.slice(7), process.env.JWT_SECRET!);
        next();
    } catch {
        return res.status(401).json({ error: 'Invalid token' });
    }
}
```

### Refresh Token Flow
```typescript
async refreshAccessToken(refreshToken: string) {
    const payload = jwt.verify(refreshToken, process.env.JWT_REFRESH_SECRET!);

    // Verify token exists in DB (revocation check)
    const stored = await db.refreshTokens.findOne({
        token: await hash(refreshToken),
        userId: payload.userId,
        expiresAt: { $gt: new Date() }
    });
    if (!stored) throw new Error('Token revoked');

    const user = await db.users.findById(payload.userId);
    return jwt.sign({ userId: user.id, email: user.email, role: user.role },
                    process.env.JWT_SECRET!, { expiresIn: '15m' });
}
```

## Session-Based Auth

```typescript
import session from 'express-session';
import RedisStore from 'connect-redis';

app.use(session({
    store: new RedisStore({ client: redisClient }),
    secret: process.env.SESSION_SECRET!,
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: process.env.NODE_ENV === 'production',
        httpOnly: true,
        maxAge: 24 * 60 * 60 * 1000,
        sameSite: 'strict'
    }
}));
```

## OAuth2 / Social Login

```typescript
passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID!,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    callbackURL: '/api/auth/google/callback'
}, async (accessToken, refreshToken, profile, done) => {
    let user = await db.users.findOne({ googleId: profile.id });
    if (!user) {
        user = await db.users.create({
            googleId: profile.id,
            email: profile.emails?.[0]?.value,
            name: profile.displayName
        });
    }
    return done(null, user);
}));
```

## Authorization Patterns

### Role-Based (RBAC)
```typescript
const roleHierarchy = {
    admin: ['admin', 'moderator', 'user'],
    moderator: ['moderator', 'user'],
    user: ['user']
};

function requireRole(...roles: Role[]) {
    return (req, res, next) => {
        if (!req.user) return res.status(401).json({ error: 'Not authenticated' });
        if (!roles.some(r => roleHierarchy[req.user.role].includes(r)))
            return res.status(403).json({ error: 'Insufficient permissions' });
        next();
    };
}
```

### Permission-Based
```typescript
const rolePermissions = {
    user: ['read:posts', 'write:posts'],
    moderator: ['read:posts', 'write:posts', 'read:users'],
    admin: ['read:posts', 'write:posts', 'read:users', 'write:users', 'delete:users']
};

function requirePermission(...permissions: Permission[]) {
    return (req, res, next) => {
        const hasAll = permissions.every(p => rolePermissions[req.user.role]?.includes(p));
        if (!hasAll) return res.status(403).json({ error: 'Insufficient permissions' });
        next();
    };
}
```

### Resource Ownership
```typescript
async function requireOwnership(resourceType: string) {
    return async (req, res, next) => {
        if (req.user.role === 'admin') return next();

        const resource = await db[resourceType].findById(req.params.id);
        if (!resource) return res.status(404).json({ error: 'Not found' });
        if (resource.userId !== req.user.userId)
            return res.status(403).json({ error: 'Not authorized' });
        next();
    };
}
```

## Password Security

```typescript
import bcrypt from 'bcrypt';
import { z } from 'zod';

const passwordSchema = z.string()
    .min(12, 'Min 12 characters')
    .regex(/[A-Z]/, 'Need uppercase')
    .regex(/[a-z]/, 'Need lowercase')
    .regex(/[0-9]/, 'Need number')
    .regex(/[^A-Za-z0-9]/, 'Need special char');

async function hashPassword(password: string) {
    return bcrypt.hash(password, 12);
}

async function verifyPassword(password: string, hash: string) {
    return bcrypt.compare(password, hash);
}
```

## Rate Limiting

```typescript
import rateLimit from 'express-rate-limit';

const loginLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,  // 15 minutes
    max: 5,                     // 5 attempts
    message: 'Too many login attempts'
});

app.post('/api/auth/login', loginLimiter, loginHandler);
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Hash passwords | bcrypt (12 rounds) or argon2 |
| Short access tokens | 15-30 minutes max |
| Secure cookies | httpOnly, secure, sameSite=strict |
| Rate limit auth | 5 attempts per 15 min |
| HTTPS only | TLS for all traffic |
| Rotate secrets | Regular key rotation |
| Log security events | Failed logins, token refresh |

## Common Pitfalls

| Pitfall | Risk |
|---------|------|
| JWT in localStorage | XSS vulnerability |
| No token expiration | Infinite access |
| Client-side auth only | Easily bypassed |
| Weak passwords | Brute force attacks |
| No rate limiting | Credential stuffing |

## Checklist

- [ ] Passwords hashed with bcrypt/argon2
- [ ] JWT with short expiration + refresh tokens
- [ ] Secure cookie settings configured
- [ ] Rate limiting on auth endpoints
- [ ] HTTPS enforced
- [ ] RBAC or ABAC implemented
- [ ] Refresh token revocation supported
- [ ] Security events logged
