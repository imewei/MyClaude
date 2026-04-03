---
name: caching-patterns
description: Design caching strategies with Redis, Memcached, and CDN layers including cache invalidation, write-through/write-behind, distributed caching, and cache stampede prevention. Use when implementing caching layers, Redis data structures, or optimizing response times.
---

# Caching Patterns

## Expert Agent

For caching architecture, distributed cache design, and performance optimization, delegate to:

- **`software-architect`**: Designs caching layers, invalidation strategies, and distributed cache topologies.
  - *Location*: `plugins/dev-suite/agents/software-architect.md`


## Caching Strategy Comparison

| Pattern | Description | Consistency | Use Case |
|---------|-------------|-------------|----------|
| Cache-Aside | App reads/writes cache explicitly | Eventual | General purpose, read-heavy |
| Read-Through | Cache loads on miss automatically | Eventual | ORM integration |
| Write-Through | Cache writes to DB synchronously | Strong | Write-heavy, consistency critical |
| Write-Behind | Cache writes to DB asynchronously | Eventual | High write throughput |
| Refresh-Ahead | Cache refreshes before expiry | Eventual | Predictable access patterns |


## Cache-Aside Pattern (Redis + Python)

```python
import json
import redis

client = redis.Redis(host="localhost", port=6379, decode_responses=True)

def get_user(user_id: int) -> dict:
    cache_key = f"user:{user_id}"
    cached = client.get(cache_key)
    if cached:
        return json.loads(cached)

    user = db.query_user(user_id)
    client.setex(cache_key, 3600, json.dumps(user))
    return user

def update_user(user_id: int, data: dict) -> None:
    db.update_user(user_id, data)
    client.delete(f"user:{user_id}")  # Invalidate, don't update
```


## Redis Data Structures

| Structure | Use Case | Example |
|-----------|----------|---------|
| String | Simple key-value, counters | Session tokens, rate limits |
| Hash | Object fields | User profiles, config |
| List | Queues, recent items | Activity feeds, job queues |
| Set | Unique collections, tags | Online users, feature flags |
| Sorted Set | Rankings, time-series | Leaderboards, rate windows |
| Stream | Event log, messaging | Event sourcing, audit trail |

### Sorted Set for Leaderboard

```python
def update_score(user_id: str, score: float) -> None:
    client.zadd("leaderboard", {user_id: score})

def get_top_players(count: int = 10) -> list:
    return client.zrevrange("leaderboard", 0, count - 1, withscores=True)

def get_rank(user_id: str) -> int:
    rank = client.zrevrank("leaderboard", user_id)
    return rank + 1 if rank is not None else -1
```


## TTL Strategies

| Data Type | TTL | Rationale |
|-----------|-----|-----------|
| Session data | 30 min | Security, memory |
| User profiles | 1 hour | Moderate change frequency |
| Product catalog | 5 min | Price/stock changes |
| Static config | 24 hours | Rarely changes |
| API rate limit | 1 min window | Sliding window |


## Cache Stampede Prevention

### Mutex Lock Pattern

```python
import time

def get_with_mutex(key: str, fetch_fn, ttl: int = 3600) -> dict:
    cached = client.get(key)
    if cached:
        return json.loads(cached)

    lock_key = f"lock:{key}"
    if client.set(lock_key, "1", nx=True, ex=10):
        try:
            data = fetch_fn()
            client.setex(key, ttl, json.dumps(data))
            return data
        finally:
            client.delete(lock_key)
    else:
        time.sleep(0.1)
        return get_with_mutex(key, fetch_fn, ttl)
```

### Stale-While-Revalidate

```python
def get_with_stale(key: str, fetch_fn, ttl: int = 3600, stale_ttl: int = 300):
    cached = client.get(key)
    remaining_ttl = client.ttl(key)

    if cached and remaining_ttl > 0:
        if remaining_ttl < stale_ttl:
            # Refresh in background
            refresh_in_background(key, fetch_fn, ttl)
        return json.loads(cached)

    data = fetch_fn()
    client.setex(key, ttl, json.dumps(data))
    return data
```


## Distributed Cache Topology

| Topology | Pros | Cons |
|----------|------|------|
| Single node | Simple, low latency | No HA, limited memory |
| Sentinel | Auto failover | More complexity |
| Cluster | Horizontal scale, HA | Client must support |
| Read replicas | Scale reads | Replication lag |


## Cache Invalidation Patterns

| Pattern | Mechanism | Consistency |
|---------|-----------|-------------|
| TTL expiry | Automatic timeout | Eventual |
| Explicit delete | Delete on write | Strong |
| Event-driven | Pub/sub on data change | Near real-time |
| Version tags | Cache key includes version | Strong |

### Event-Driven Invalidation

```python
def on_user_updated(event: dict) -> None:
    user_id = event["user_id"]
    client.delete(f"user:{user_id}")
    client.delete(f"user_feed:{user_id}")
    # Publish to other service caches
    client.publish("cache_invalidation", json.dumps({"type": "user", "id": user_id}))
```


## Design Checklist

- [ ] Caching strategy chosen based on read/write ratio
- [ ] TTL values set per data type, not globally
- [ ] Cache stampede prevention implemented for hot keys
- [ ] Invalidation strategy matches consistency requirements
- [ ] Redis memory limits and eviction policy configured
- [ ] Cache hit/miss ratio monitored (target > 90%)
- [ ] Graceful degradation when cache is unavailable
- [ ] No sensitive data cached without encryption
