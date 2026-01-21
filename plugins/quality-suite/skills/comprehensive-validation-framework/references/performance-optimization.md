# Performance Optimization Reference

Comprehensive guide to profiling, optimizing, and load testing applications.

## Performance Analysis Workflow

1. **Measure First** → 2. **Identify Bottlenecks** → 3. **Optimize** → 4. **Verify**

Never optimize without measuring. "Premature optimization is the root of all evil."

---

## Profiling Tools by Language

### Python

**CPU Profiling**:
```bash
# cProfile
python -m cProfile -o profile.stats script.py
python -m pstats profile.stats

# py-spy (sampling profiler, no code changes)
py-spy record -o profile.svg -- python script.py

# line_profiler (line-by-line)
kernprof -l -v script.py
```

**Memory Profiling**:
```bash
# memory_profiler
python -m memory_profiler script.py

# memray (fast memory profiler)
memray run script.py
memray flamegraph output.bin
```

### JavaScript/Node.js

**CPU Profiling**:
```bash
# Node.js built-in
node --prof app.js
node --prof-process isolate-*.log

# Chrome DevTools
node --inspect app.js
# Open chrome://inspect

# clinic.js
clinic doctor -- node app.js
clinic flame -- node app.js
```

**Memory Profiling**:
```bash
# Node.js heap snapshot
node --inspect app.js
# Use Chrome DevTools Memory tab

# clinic.js
clinic heapprofiler -- node app.js
```

### Rust

```bash
# CPU profiling with cargo-flamegraph
cargo flamegraph --bin myapp

# Benchmarking
cargo bench

# With perf (Linux)
cargo build --release
perf record --call-graph dwarf target/release/myapp
perf report
```

### Go

```bash
# CPU profiling
go test -cpuprofile=cpu.prof -bench=.
go tool pprof cpu.prof

# Memory profiling
go test -memprofile=mem.prof -bench=.
go tool pprof mem.prof

# HTTP server profiling
import _ "net/http/pprof"
# Visit http://localhost:6060/debug/pprof/
```

---

## Common Performance Bottlenecks

### 1. Database Queries (N+1 Problem)

**❌ Bad - N+1 Queries**:
```python
# Fetches users, then makes N queries for posts
users = User.query.all()
for user in users:
    posts = Post.query.filter_by(user_id=user.id).all()  # N queries!
```

**✅ Good - Eager Loading**:
```python
# Single query with JOIN
users = User.query.options(joinedload(User.posts)).all()
```

**SQL Monitoring**:
```python
# SQLAlchemy: Enable query logging
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### 2. Missing Database Indexes

**Check for missing indexes**:
```sql
-- PostgreSQL: Find slow queries
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 20;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0;
```

**Add indexes**:
```sql
-- Index on frequently queried columns
CREATE INDEX idx_users_email ON users(email);

-- Composite index for multi-column queries
CREATE INDEX idx_posts_user_date ON posts(user_id, created_at DESC);

-- Partial index for common filters
CREATE INDEX idx_active_users ON users(email) WHERE active = true;
```

### 3. Inefficient Algorithms

**Time Complexity Quick Reference**:
- O(1) - Constant: Hash table lookup
- O(log n) - Logarithmic: Binary search
- O(n) - Linear: Simple iteration
- O(n log n) - Log-linear: Efficient sorting (merge sort, quicksort)
- O(n²) - Quadratic: Nested loops (often bad)
- O(2ⁿ) - Exponential: Recursive without memoization (very bad)

**Example Optimization**:

**❌ Bad - O(n²)**:
```python
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                duplicates.append(arr[i])
    return duplicates
```

**✅ Good - O(n)**:
```python
def find_duplicates(arr):
    seen = set()
    duplicates = set()
    for item in arr:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)
```

### 4. Excessive Memory Allocations

**Python - Use Generators**:

**❌ Bad - Loads entire list into memory**:
```python
def process_large_file(filename):
    lines = open(filename).readlines()  # Entire file in memory!
    return [line.upper() for line in lines]
```

**✅ Good - Streaming with generator**:
```python
def process_large_file(filename):
    with open(filename) as f:
        for line in f:  # One line at a time
            yield line.upper()
```

**JavaScript - Avoid Unnecessary Copies**:

**❌ Bad**:
```javascript
const newArr = [...bigArray];  // Full copy
newArr.push(item);
```

**✅ Good**:
```javascript
bigArray.push(item);  // Mutate in place if safe
```

### 5. Synchronous I/O Blocking

**Python - Use Async**:

**❌ Bad - Synchronous**:
```python
import requests

def fetch_data():
    results = []
    for url in urls:
        response = requests.get(url)  # Blocks!
        results.append(response.json())
    return results
```

**✅ Good - Async**:
```python
import asyncio
import aiohttp

async def fetch_data():
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]
```

---

## Caching Strategies

### Multi-Tier Caching

1. **Application Cache** (in-memory)
2. **Distributed Cache** (Redis, Memcached)
3. **CDN Cache** (CloudFlare, Fastly)
4. **Database Query Cache**
5. **Browser Cache** (HTTP headers)

### Python Example with Redis

```python
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379)

def cache(expiry=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cached = redis_client.get(key)

            if cached:
                return json.loads(cached)

            result = func(*args, **kwargs)
            redis_client.setex(key, expiry, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache(expiry=600)
def expensive_computation(x):
    # Expensive operation here
    return result
```

### HTTP Caching Headers

```python
from flask import make_response

@app.route('/static-data')
def static_data():
    response = make_response(data)
    response.headers['Cache-Control'] = 'public, max-age=3600'
    response.headers['ETag'] = generate_etag(data)
    return response
```

---

## Load Testing

### Tools

- **wrk**: HTTP benchmarking tool (C, very fast)
- **k6**: Modern load testing (Go, JavaScript DSL)
- **Locust**: Python-based, distributed load testing
- **Apache JMeter**: GUI-based, comprehensive
- **Gatling**: Scala-based, detailed reports

### k6 Example

```javascript
// load_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% under 500ms
    http_req_failed: ['rate<0.01'],    // <1% failure rate
  },
};

export default function () {
  const res = http.get('https://api.example.com/endpoint');
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  sleep(1);
}
```

Run:
```bash
k6 run load_test.js
```

### Locust Example

```python
# locustfile.py
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def view_homepage(self):
        self.client.get("/")

    @task(1)
    def view_profile(self):
        self.client.get("/profile/123")

    def on_start(self):
        self.client.post("/login", json={
            "username": "test",
            "password": "test123"
        })
```

Run:
```bash
locust -f locustfile.py --host=https://example.com
```

---

## Database Optimization

### Query Optimization

**EXPLAIN ANALYZE**:
```sql
-- PostgreSQL
EXPLAIN ANALYZE
SELECT u.name, COUNT(p.id)
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
WHERE u.active = true
GROUP BY u.id;

-- Look for:
-- 1. Sequential Scans (bad on large tables)
-- 2. Missing indexes
-- 3. Inefficient joins
```

**Connection Pooling**:

```python
# SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@localhost/db',
    poolclass=QueuePool,
    pool_size=10,          # Normal connections
    max_overflow=20,       # Extra connections when needed
    pool_pre_ping=True,    # Check connection health
    pool_recycle=3600      # Recycle after 1 hour
)
```

### Pagination

**❌ Bad - OFFSET becomes slow with large offsets**:
```sql
SELECT * FROM posts
ORDER BY created_at DESC
OFFSET 1000000 LIMIT 20;  -- Very slow!
```

**✅ Good - Cursor-based (keyset) pagination**:
```sql
SELECT * FROM posts
WHERE created_at < '2024-01-01 00:00:00'
  AND id < 1000000
ORDER BY created_at DESC, id DESC
LIMIT 20;
```

---

## Frontend Performance

### Bundle Size Optimization

**Analyze Bundle**:
```bash
# Webpack
npx webpack-bundle-analyzer dist/stats.json

# Vite
npx vite-bundle-visualizer
```

**Code Splitting**:
```javascript
// React lazy loading
import { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HeavyComponent />
    </Suspense>
  );
}
```

### Image Optimization

- Use WebP format (fallback to JPG/PNG)
- Lazy load images below the fold
- Use appropriate sizes (srcset)
- Compress images (TinyPNG, ImageOptim)
- Use CDN for image delivery

### Core Web Vitals

**Target Metrics**:
- **LCP (Largest Contentful Paint)**: < 2.5s
- **FID (First Input Delay)**: < 100ms
- **CLS (Cumulative Layout Shift)**: < 0.1

**Measure with Lighthouse**:
```bash
lighthouse https://example.com --view
```

---

## Performance Budget

### Define Budgets

```json
{
  "performance": {
    "maxResponseTime": 200,
    "maxDatabaseQueries": 10,
    "maxBundleSize": 300,
    "minTestCoverage": 80
  }
}
```

### Enforce in CI/CD

```yaml
# .github/workflows/performance.yml
- name: Check Bundle Size
  run: |
    npm run build
    SIZE=$(du -sk dist | cut -f1)
    if [ $SIZE -gt 300 ]; then
      echo "Bundle size $SIZE KB exceeds limit 300 KB"
      exit 1
    fi
```

---

## Monitoring in Production

### Key Metrics

**Golden Signals**:
1. **Latency**: Response time distribution (p50, p95, p99)
2. **Traffic**: Requests per second
3. **Errors**: Error rate (5xx responses)
4. **Saturation**: Resource utilization (CPU, memory, disk, network)

**RED Method** (for services):
- **Rate**: Requests per second
- **Errors**: Error rate
- **Duration**: Latency distribution

### Application Performance Monitoring (APM)

Tools:
- Datadog
- New Relic
- Dynatrace
- Elastic APM
- Prometheus + Grafana

**Example: Prometheus Metrics**:
```python
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('http_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('http_request_duration_seconds', 'Request duration')

@app.route('/metrics')
def metrics():
    return generate_latest()
```

---

## Performance Checklist

### Before Every Release

- [ ] Profile critical paths
- [ ] Check for N+1 queries
- [ ] Verify database indexes exist
- [ ] Test with production-like data volume
- [ ] Run load tests
- [ ] Check bundle sizes (frontend)
- [ ] Verify caching is working
- [ ] Monitor key metrics in staging
- [ ] Set up alerts for performance regressions

---

## References

- [Web.dev Performance](https://web.dev/performance/)
- [Google PageSpeed Insights](https://pagespeed.web.dev/)
- [Mozilla Performance](https://developer.mozilla.org/en-US/docs/Web/Performance)
