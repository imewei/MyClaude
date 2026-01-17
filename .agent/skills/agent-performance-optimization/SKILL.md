---
name: agent-performance-optimization
version: "1.0.7"
maturity: "5-Expert"
specialization: Agent Performance Tuning
description: Optimize AI agent performance through monitoring, metrics collection, caching, and load balancing. Use when analyzing agent execution bottlenecks, implementing latency percentiles (P50/P95/P99), setting up multi-tier caching, tracking success/failure rates, load balancing across agent instances, or scaling agent systems for production.
---

# Agent Performance Optimization

Production-ready patterns for monitoring, caching, and load balancing agent systems.

---

<!-- SECTION: METRICS -->
## Performance Metrics

```python
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import time
import functools

@dataclass
class PerformanceMetrics:
    agent_name: str
    task_count: int = 0
    success_count: int = 0
    execution_times: list[float] = field(default_factory=list)
    error_types: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, execution_time: float, success: bool, error_type: str = None):
        self.task_count += 1
        self.execution_times.append(execution_time)
        if success:
            self.success_count += 1
        elif error_type:
            self.error_types[error_type] += 1

    @property
    def p50(self) -> float:
        return sorted(self.execution_times)[int(len(self.execution_times) * 0.5)]

    @property
    def p95(self) -> float:
        return sorted(self.execution_times)[int(len(self.execution_times) * 0.95)]

    @property
    def success_rate(self) -> float:
        return self.success_count / self.task_count if self.task_count > 0 else 0

class MetricsCollector:
    def __init__(self):
        self.metrics: dict[str, PerformanceMetrics] = {}
        self.start_time = datetime.now()

    def record(self, agent_name: str, execution_time: float, success: bool, error_type: str = None):
        if agent_name not in self.metrics:
            self.metrics[agent_name] = PerformanceMetrics(agent_name)
        self.metrics[agent_name].update(execution_time, success, error_type)

# Decorator for automatic tracking
def track_performance(agent_name: str, collector: MetricsCollector):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                collector.record(agent_name, time.time() - start, True)
                return result
            except Exception as e:
                collector.record(agent_name, time.time() - start, False, type(e).__name__)
                raise
        return wrapper
    return decorator

# Usage
collector = MetricsCollector()

@track_performance("ml-engineer", collector)
async def train_model(data):
    # Training logic
    pass
```
<!-- END_SECTION: METRICS -->

---

<!-- SECTION: CACHING -->
## Caching System

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
import hashlib
import json

@dataclass
class CacheEntry:
    key: str
    value: Any
    expires_at: Optional[datetime]
    access_count: int = 0

    def is_expired(self) -> bool:
        return self.expires_at and datetime.now() > self.expires_at

class LRUCache:
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: dict[str, CacheEntry] = {}
        self.access_order: list[str] = []
        self.hits = 0
        self.misses = 0

    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        key_str = json.dumps({'func': func_name, 'args': args, 'kwargs': kwargs}, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        entry = self.cache.get(key)
        if entry is None or entry.is_expired():
            self.misses += 1
            if entry: self.cache.pop(key)
            return None
        self.hits += 1
        entry.access_count += 1
        self.access_order.remove(key)
        self.access_order.append(key)
        return entry.value

    def put(self, key: str, value: Any, ttl: int = None):
        if key in self.cache:
            self.access_order.remove(key)
        if len(self.cache) >= self.max_size:
            lru_key = self.access_order.pop(0)
            self.cache.pop(lru_key)

        expires_at = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        self.cache[key] = CacheEntry(key, value, expires_at)
        self.access_order.append(key)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

# Decorator
def cached(cache: LRUCache, ttl: int = None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            key = cache._make_key(func.__name__, args, kwargs)
            value = cache.get(key)
            if value is not None:
                return value
            result = await func(*args, **kwargs)
            cache.put(key, result, ttl)
            return result
        return wrapper
    return decorator

# Multi-tier cache
class TieredCache:
    def __init__(self):
        self.hot = LRUCache(100, 300)    # 5 min
        self.warm = LRUCache(500, 1800)  # 30 min
        self.cold = LRUCache(2000, 3600) # 1 hour

    def get(self, key: str) -> Optional[Any]:
        # Try hot -> warm -> cold, promote on hit
        value = self.hot.get(key)
        if value: return value

        value = self.warm.get(key)
        if value:
            self.hot.put(key, value)
            return value

        value = self.cold.get(key)
        if value:
            self.warm.put(key, value)
            return value
        return None
```
<!-- END_SECTION: CACHING -->

---

<!-- SECTION: LOAD_BALANCING -->
## Load Balancing

```python
from dataclasses import dataclass
from enum import Enum
import random

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"

@dataclass
class AgentInstance:
    id: str
    agent_type: str
    current_load: int = 0
    max_capacity: int = 10
    weight: float = 1.0
    is_healthy: bool = True

    @property
    def available_capacity(self) -> int:
        return max(0, self.max_capacity - self.current_load)

class LoadBalancer:
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED):
        self.strategy = strategy
        self.instances: dict[str, list[AgentInstance]] = {}
        self.rr_index: dict[str, int] = {}

    def register(self, instance: AgentInstance):
        self.instances.setdefault(instance.agent_type, []).append(instance)

    def select(self, agent_type: str) -> Optional[AgentInstance]:
        instances = [i for i in self.instances.get(agent_type, [])
                     if i.is_healthy and i.current_load < i.max_capacity]
        if not instances:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            idx = self.rr_index.get(agent_type, 0) % len(instances)
            self.rr_index[agent_type] = idx + 1
            return instances[idx]
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return min(instances, key=lambda x: x.current_load / x.max_capacity)
        else:  # WEIGHTED
            weights = [i.weight * (1 - i.current_load / i.max_capacity) for i in instances]
            return random.choices(instances, weights=weights)[0]

    def acquire(self, agent_type: str) -> Optional[AgentInstance]:
        instance = self.select(agent_type)
        if instance:
            instance.current_load += 1
        return instance

    def release(self, instance: AgentInstance):
        instance.current_load = max(0, instance.current_load - 1)

# Usage
lb = LoadBalancer(LoadBalancingStrategy.LEAST_LOADED)
lb.register(AgentInstance("ml-1", "ml-engineer", max_capacity=5))
lb.register(AgentInstance("ml-2", "ml-engineer", max_capacity=5))

instance = lb.acquire("ml-engineer")
# Use instance...
lb.release(instance)
```
<!-- END_SECTION: LOAD_BALANCING -->

---

## Best Practices

| Area | Practice |
|------|----------|
| Metrics | Track P50/P95/P99, not just averages |
| Caching | Use TTL, track hit rates, multi-tier for hot/warm/cold |
| Load Balancing | Health check before routing, graceful degradation |
| Monitoring | Automated alerting on thresholds |
| Retries | Exponential backoff for transient failures |

---

## Performance Targets

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| P95 Latency | <500ms | >1s |
| Success Rate | >99% | <95% |
| Cache Hit Rate | >80% | <50% |
| Load Balance | Even distribution | Any instance >80% |

---

## Checklist

- [ ] Metrics collection on all agents
- [ ] P50/P95/P99 latency tracking
- [ ] Success/failure rate monitoring
- [ ] Error type categorization
- [ ] LRU caching for expensive operations
- [ ] Cache hit rate monitoring
- [ ] Load balancer with health checks
- [ ] Automated alerting configured
- [ ] Retry logic with exponential backoff

---

**Version**: 1.0.5
