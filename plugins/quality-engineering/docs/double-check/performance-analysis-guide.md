# Performance Analysis Guide

Comprehensive performance profiling, bottleneck detection, and optimization strategies for production applications.

## Overview

Performance analysis is critical for ensuring applications scale efficiently and provide excellent user experience. This guide covers:
- CPU and memory profiling tools
- Database optimization (N+1 queries, indexing)
- Caching strategies
- Load testing methodologies
- Performance budgets and SLOs

---

## 1. Performance Profiling Tools

### Python Applications

#### cProfile - Built-in CPU Profiler

**Basic profiling**:
```bash
# Profile a script
python -m cProfile -s cumtime script.py

# Save profile data
python -m cProfile -o profile.stats script.py

# Analyze with pstats
python -m pstats profile.stats
>>> sort cumtime
>>> stats 20
```

**Programmatic profiling**:
```python
import cProfile
import pstats
from pstats import SortKey

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()

    # Your code here
    expensive_function()

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
```

#### py-spy - Sampling Profiler

**Zero-overhead profiling of running processes**:
```bash
# Profile running process
py-spy record -o profile.svg --pid <PID>

# Top-like interface
py-spy top --pid <PID>

# Profile subprocess
py-spy record -o profile.svg -- python app.py

# Flamegraph generation
py-spy record --format flamegraph -o flamegraph.svg --pid <PID>
```

#### memory_profiler - Memory Usage Analysis

```bash
# Install
uv uv pip install memory-profiler

# Decorate functions
@profile
def memory_intensive_function():
    large_list = [0] * (10 ** 7)
    return sum(large_list)

# Run with profiling
python -m memory_profiler script.py
```

**Output**:
```
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     3   38.816 MiB   38.816 MiB           1   @profile
     4                                         def memory_intensive_function():
     5  115.289 MiB   76.473 MiB           1       large_list = [0] * (10 ** 7)
     6  115.289 MiB    0.000 MiB           1       return sum(large_list)
```

---

### JavaScript/Node.js Applications

#### Node.js Built-in Profiler

```bash
# Generate CPU profile
node --prof app.js

# Process tick profiler output
node --prof-process isolate-*.log > processed.txt

# Inspect heap snapshots
node --inspect app.js
# Open chrome://inspect in Chrome
```

#### Clinic.js - Comprehensive Performance Suite

```bash
# Install
npm install -g clinic

# Doctor - Diagnose performance issues
clinic doctor -- node app.js

# Flame - CPU profiling
clinic flame -- node app.js

# Bubbleprof - Async operations profiling
clinic bubbleprof -- node app.js

# Heap profiler
clinic heapprofiler -- node app.js
```

#### 0x - Flamegraph Profiler

```bash
# Install
npm install -g 0x

# Profile application
0x app.js

# With custom sample rate
0x -P 'autocannon -c100 localhost:3000' app.js
```

---

### Rust Applications

#### cargo-flamegraph

```bash
# Install
cargo install flamegraph

# Generate flamegraph
cargo flamegraph

# Profile specific binary
cargo flamegraph --bin my-app

# Profile tests
cargo flamegraph --test my-test
```

#### perf (Linux)

```bash
# Record performance data
perf record --call-graph dwarf ./target/release/my-app

# Analyze
perf report

# Top processes
perf top
```

---

### Go Applications

#### pprof - Built-in Profiler

```go
import (
    "net/http"
    _ "net/http/pprof"
)

func main() {
    go func() {
        http.ListenAndServe("localhost:6060", nil)
    }()

    // Your application code
}
```

**Usage**:
```bash
# CPU profile (30 seconds)
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30

# Heap profile
go tool pprof http://localhost:6060/debug/pprof/heap

# Goroutine profile
go tool pprof http://localhost:6060/debug/pprof/goroutine

# Generate flamegraph
go tool pprof -http=:8080 profile.pb.gz
```

---

## 2. N+1 Query Detection and Fixes

### What is the N+1 Query Problem?

**Example scenario**: Loading users and their posts

**Bad (N+1 queries)**:
```python
# 1 query to get users
users = User.query.all()

# N queries to get posts for each user
for user in users:
    posts = user.posts  # Triggers separate query!
    print(f"{user.name}: {len(posts)} posts")
```

**SQL executed**:
```sql
SELECT * FROM users;                     -- 1 query
SELECT * FROM posts WHERE user_id = 1;  -- Query 1
SELECT * FROM posts WHERE user_id = 2;  -- Query 2
SELECT * FROM posts WHERE user_id = 3;  -- Query 3
-- ... N more queries
```

### Detection Tools

#### Django Debug Toolbar

```python
# settings.py
INSTALLED_APPS += ['debug_toolbar']
MIDDLEWARE += ['debug_toolbar.middleware.DebugToolbarMiddleware']
INTERNAL_IPS = ['127.0.0.1']

# Shows query count and duplicates in UI
```

#### SQLAlchemy Query Logging

```python
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Logs all SQL queries to console
```

#### nplusone (Python Library)

```python
from nplusone.ext.flask_sqlalchemy import NPlusOne

app = Flask(__name__)
NPlusOne(app)

# Raises warnings when N+1 queries detected
```

### Solutions

#### Eager Loading (Recommended)

**Django**:
```python
# Use select_related for foreign keys
users = User.objects.select_related('profile').all()

# Use prefetch_related for many-to-many
users = User.objects.prefetch_related('posts').all()

# Multiple levels
users = User.objects.prefetch_related(
    'posts__comments__author'
).all()
```

**SQLAlchemy**:
```python
from sqlalchemy.orm import joinedload, selectinload

# Joined load (single query with JOIN)
users = session.query(User).options(
    joinedload(User.posts)
).all()

# Select in load (two queries)
users = session.query(User).options(
    selectinload(User.posts)
).all()
```

**ActiveRecord (Ruby)**:
```ruby
# Eager load associations
users = User.includes(:posts).all

# Multiple associations
users = User.includes(:posts, :comments).all
```

#### DataLoader Pattern (GraphQL)

```javascript
const DataLoader = require('dataloader');

const postLoader = new DataLoader(async (userIds) => {
  const posts = await Post.findAll({
    where: { userId: userIds }
  });

  // Group by user_id
  const grouped = userIds.map(id =>
    posts.filter(post => post.userId === id)
  );

  return grouped;
});

// Usage
const posts = await postLoader.load(userId);
```

---

## 3. Database Optimization

### Query Analysis

#### EXPLAIN ANALYZE (PostgreSQL)

```sql
EXPLAIN ANALYZE
SELECT u.name, COUNT(p.id) as post_count
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
GROUP BY u.id, u.name
ORDER BY post_count DESC
LIMIT 10;
```

**Output interpretation**:
```
Limit  (cost=1234.56..1234.78 rows=10 width=64) (actual time=45.123..45.145 rows=10 loops=1)
  ->  Sort  (cost=1234.56..1235.78 rows=1000 width=64) (actual time=45.120..45.130 rows=10 loops=1)
        Sort Key: (count(p.id)) DESC
        ->  HashAggregate  (cost=1200.00..1220.00 rows=1000 width=64)
```

**Key metrics**:
- `cost`: Query planner estimate
- `actual time`: Real execution time (ms)
- `rows`: Estimated vs actual row count
- `loops`: Number of times operation repeated

#### Query Performance Checklist

- [ ] Indexes exist for WHERE clause columns
- [ ] Indexes exist for JOIN columns
- [ ] Indexes exist for ORDER BY columns
- [ ] No full table scans (Seq Scan) on large tables
- [ ] Cardinality estimates are accurate
- [ ] Query plan is optimal (no unnecessary sorts/aggregations)

### Index Strategies

#### When to Add Indexes

**✅ Good candidates**:
```sql
-- Frequently queried columns
CREATE INDEX idx_users_email ON users(email);

-- Foreign keys
CREATE INDEX idx_posts_user_id ON posts(user_id);

-- Composite indexes for multi-column queries
CREATE INDEX idx_posts_user_status ON posts(user_id, status);

-- Partial indexes for filtered queries
CREATE INDEX idx_active_users ON users(email) WHERE deleted_at IS NULL;
```

**❌ Avoid indexing**:
- Low-cardinality columns (e.g., boolean with 50/50 distribution)
- Columns rarely used in WHERE/JOIN
- Very small tables (<1000 rows)

#### Index Optimization

**Covering indexes** (include all columns needed):
```sql
-- Query: SELECT name, email FROM users WHERE status = 'active'
CREATE INDEX idx_users_status_covering ON users(status) INCLUDE (name, email);

-- Avoids table lookup
```

**Index-only scans**:
```sql
-- All columns in index
CREATE INDEX idx_posts_composite ON posts(user_id, created_at, title);

-- Query uses only indexed columns
SELECT user_id, created_at, title
FROM posts
WHERE user_id = 123
ORDER BY created_at DESC;
```

### Query Optimization Patterns

#### Avoid SELECT *

**Bad**:
```sql
SELECT * FROM users WHERE email = 'user@example.com';
```

**Good**:
```sql
SELECT id, name, email FROM users WHERE email = 'user@example.com';
```

#### Use Pagination

**Bad**:
```sql
SELECT * FROM posts ORDER BY created_at DESC;
-- Returns all rows!
```

**Good (Offset-based)**:
```sql
SELECT * FROM posts
ORDER BY created_at DESC
LIMIT 20 OFFSET 0;
-- Page 2: OFFSET 20
```

**Better (Cursor-based)**:
```sql
-- First page
SELECT * FROM posts
ORDER BY created_at DESC
LIMIT 20;

-- Next page (using last created_at)
SELECT * FROM posts
WHERE created_at < '2024-01-01 12:00:00'
ORDER BY created_at DESC
LIMIT 20;
```

#### Batch Operations

**Bad**:
```python
for user_id in user_ids:
    User.objects.filter(id=user_id).update(status='active')
    # N queries!
```

**Good**:
```python
User.objects.filter(id__in=user_ids).update(status='active')
# Single query
```

---

## 4. Caching Strategies

### Caching Layers

```
┌─────────────────────────────────────────┐
│ Client-Side Cache (Browser/App)        │
│ • LocalStorage, IndexedDB               │
│ • Service Workers                       │
└─────────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ CDN Cache (Cloudflare, AWS CloudFront) │
│ • Static assets                         │
│ • Edge caching                          │
└─────────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ HTTP Cache (nginx, Varnish)            │
│ • Reverse proxy cache                   │
│ • Full page cache                       │
└─────────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ Application Cache (Redis, Memcached)   │
│ • Query results                         │
│ • Session data                          │
│ • Computed values                       │
└─────────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ Database Query Cache                    │
│ • PostgreSQL shared_buffers             │
│ • MySQL query cache                     │
└─────────────────────────────────────────┘
```

### Redis Caching

#### Basic Caching Pattern

```python
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache(ttl=300):
    """Cache decorator with TTL in seconds"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result)
            )

            return result
        return wrapper
    return decorator

@cache(ttl=600)
def get_user_stats(user_id):
    # Expensive database query
    return {
        'post_count': count_posts(user_id),
        'followers': count_followers(user_id),
    }
```

#### Cache Invalidation

```python
# Invalidate on update
def update_user(user_id, data):
    User.update(user_id, data)

    # Invalidate cache
    redis_client.delete(f"get_user_stats:{user_id}")
    redis_client.delete(f"user_profile:{user_id}")

# Pattern-based invalidation
def invalidate_user_caches(user_id):
    pattern = f"*:user_id:{user_id}:*"
    for key in redis_client.scan_iter(pattern):
        redis_client.delete(key)
```

### Cache-Aside Pattern (Lazy Loading)

```python
def get_product(product_id):
    # 1. Check cache first
    cached = redis_client.get(f"product:{product_id}")
    if cached:
        return json.loads(cached)

    # 2. Load from database
    product = Product.query.get(product_id)

    # 3. Store in cache
    redis_client.setex(
        f"product:{product_id}",
        3600,  # 1 hour TTL
        json.dumps(product.to_dict())
    )

    return product
```

### Write-Through Cache

```python
def update_product(product_id, data):
    # 1. Update database
    product = Product.query.get(product_id)
    product.update(data)
    db.session.commit()

    # 2. Update cache immediately
    redis_client.setex(
        f"product:{product_id}",
        3600,
        json.dumps(product.to_dict())
    )
```

### HTTP Caching Headers

```python
from flask import make_response

@app.route('/api/products/<int:id>')
def get_product_api(id):
    product = get_product(id)
    response = make_response(product)

    # Cache for 1 hour
    response.headers['Cache-Control'] = 'public, max-age=3600'

    # ETag for conditional requests
    response.headers['ETag'] = f'"{product.updated_at.timestamp()}"'

    return response
```

---

## 5. Load Testing Methodologies

### Apache Bench (ab)

**Basic load test**:
```bash
# 1000 requests, 100 concurrent
ab -n 1000 -c 100 http://localhost:3000/api/users

# With POST data
ab -n 1000 -c 100 -p data.json -T application/json http://localhost:3000/api/users

# With authentication
ab -n 1000 -c 100 -H "Authorization: Bearer <token>" http://localhost:3000/api/protected
```

### wrk - Modern HTTP Benchmark

```bash
# Install
brew install wrk  # macOS
sudo apt install wrk  # Ubuntu

# Basic test (12 threads, 400 connections, 30 seconds)
wrk -t12 -c400 -d30s http://localhost:3000

# With Lua script for complex scenarios
wrk -t12 -c400 -d30s -s post.lua http://localhost:3000
```

**post.lua**:
```lua
wrk.method = "POST"
wrk.body   = '{"username": "test", "password": "test123"}'
wrk.headers["Content-Type"] = "application/json"
```

### k6 - Modern Load Testing

```javascript
// script.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 200 },   // Ramp up to 200 users
    { duration: '5m', target: 200 },   // Stay at 200 users
    { duration: '2m', target: 0 },     // Ramp down to 0
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests < 500ms
    http_req_failed: ['rate<0.01'],    // Error rate < 1%
  },
};

export default function () {
  const res = http.get('http://localhost:3000/api/users');

  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1);
}
```

**Run test**:
```bash
k6 run script.js

# Cloud execution
k6 cloud script.js

# With custom tags
k6 run --tag testid=123 script.js
```

### Locust - Python-based Load Testing

```python
# locustfile.py
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    @task(3)
    def view_products(self):
        self.client.get("/api/products")

    @task(1)
    def view_product_detail(self):
        product_id = random.randint(1, 100)
        self.client.get(f"/api/products/{product_id}")

    @task(1)
    def create_order(self):
        self.client.post("/api/orders", json={
            "product_id": 1,
            "quantity": 1,
        })

    def on_start(self):
        # Login once per user
        self.client.post("/api/login", json={
            "username": "test",
            "password": "test123",
        })
```

**Run test**:
```bash
# Web UI
locust -f locustfile.py --host=http://localhost:3000

# Headless mode
locust -f locustfile.py --host=http://localhost:3000 \
  --users 100 --spawn-rate 10 --run-time 10m --headless
```

---

## 6. Performance Budgets

### Define Performance SLOs

```yaml
# performance-budget.yml
metrics:
  # Response time targets
  api_response_time:
    p50: 100ms
    p95: 500ms
    p99: 1000ms

  # Throughput targets
  requests_per_second: 1000

  # Error rates
  error_rate: <1%

  # Resource limits
  cpu_usage: <70%
  memory_usage: <80%

  # Frontend metrics
  first_contentful_paint: <1.5s
  time_to_interactive: <3.0s
  total_bundle_size: <200KB
```

### Monitoring and Alerts

```python
# Prometheus metrics
from prometheus_client import Histogram, Counter

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

@app.route('/api/users')
@request_duration.labels('GET', '/api/users').time()
def get_users():
    return User.query.all()
```

---

## Best Practices

1. **Profile First, Optimize Second**: Don't optimize without data
2. **Set Performance Budgets**: Define SLOs and monitor them
3. **Optimize Hot Paths**: Focus on frequently executed code
4. **Use Appropriate Caching**: Match cache strategy to use case
5. **Index Strategically**: Too many indexes slow writes
6. **Test Under Load**: Simulate production traffic patterns
7. **Monitor Production**: Use APM tools (DataDog, New Relic)
8. **Optimize for 95th Percentile**: Don't just look at averages

---

## Tools Summary

| Language/Stack | CPU Profiler | Memory Profiler | Load Testing |
|----------------|--------------|-----------------|--------------|
| Python | cProfile, py-spy | memory_profiler | locust |
| JavaScript | clinic.js, 0x | Chrome DevTools | k6 |
| Rust | flamegraph, perf | valgrind | wrk |
| Go | pprof | pprof | hey |
| Any | - | - | ab, wrk, k6 |
