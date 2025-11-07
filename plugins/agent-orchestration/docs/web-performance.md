# Web Performance Optimization Patterns

## Backend Optimization

### Pattern 1: N+1 Query Elimination (100x Speedup)
```python
# Before: 1 + N queries (slow)
users = User.query.all()
for user in users:
    posts = user.posts  # Separate query per user!

# After: 2 queries (fast)
users = User.query.options(joinedload(User.posts)).all()
# Speedup: 100x for 100 users
```

### Pattern 2: Database Connection Pooling
```python
# SQLAlchemy with pooling
engine = create_engine(
    'postgresql://...',
    pool_size=20,          # Reuse connections
    max_overflow=10,
    pool_pre_ping=True     # Check connection health
)
```

### Pattern 3: Redis Caching (50x Speedup)
```python
import redis
r = redis.Redis()

def get_user_profile(user_id):
    cached = r.get(f'user:{user_id}')
    if cached:
        return json.loads(cached)

    # Cache miss: query database
    profile = db.query_user(user_id)
    r.setex(f'user:{user_id}', 3600, json.dumps(profile))
    return profile

# First call: DB query (120ms)
# Subsequent calls: Redis (2ms), 60x faster
```

## Frontend Optimization

### Pattern 4: Code Splitting (Faster Initial Load)
```javascript
// Lazy load route components
const Dashboard = React.lazy(() => import('./Dashboard'));
const Profile = React.lazy(() => import('./Profile'));

// Result: Initial bundle 80% smaller
```

### Pattern 5: Image Optimization
```html
<!-- Responsive images with WebP -->
<picture>
  <source srcset="hero.webp" type="image/webp">
  <img src="hero.jpg" loading="lazy" alt="Hero">
</picture>

<!-- Result: 70% smaller images, lazy load offscreen -->
```

---

**See also**: [Optimization Patterns](optimization-patterns.md), [Examples](examples/api-performance-optimization.md)
