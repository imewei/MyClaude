# Agent Performance Optimization and Monitoring

Expert guidance on optimizing agent performance, monitoring execution metrics, implementing caching strategies, and tuning agent behavior for production systems. Use when analyzing agent bottlenecks, improving response times, or scaling agent systems.

## Overview

This skill provides production-ready patterns for agent performance monitoring, optimization, and tuning including metrics collection, performance profiling, caching strategies, and load balancing.

## Core Topics

### 1. Performance Monitoring and Metrics

#### Comprehensive Agent Metrics Collection

```python
# agent_optimization/metrics.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import time
import functools
import asyncio

@dataclass
class PerformanceMetrics:
    """Agent performance metrics."""
    agent_name: str
    task_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    p50_execution_time: float = 0.0
    p95_execution_time: float = 0.0
    p99_execution_time: float = 0.0
    execution_times: List[float] = field(default_factory=list)
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_execution: Optional[datetime] = None

    def update(self, execution_time: float, success: bool, error_type: Optional[str] = None):
        """Update metrics with new execution."""
        self.task_count += 1
        self.execution_times.append(execution_time)
        self.total_execution_time += execution_time
        self.last_execution = datetime.now()

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            if error_type:
                self.error_types[error_type] += 1

        # Update min/max/avg
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.avg_execution_time = self.total_execution_time / self.task_count

        # Calculate percentiles
        if self.execution_times:
            sorted_times = sorted(self.execution_times)
            n = len(sorted_times)
            self.p50_execution_time = sorted_times[int(n * 0.5)]
            self.p95_execution_time = sorted_times[int(n * 0.95)]
            self.p99_execution_time = sorted_times[int(n * 0.99)]

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.success_count / self.task_count if self.task_count > 0 else 0.0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return self.failure_count / self.task_count if self.task_count > 0 else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            'agent_name': self.agent_name,
            'task_count': self.task_count,
            'success_rate': f"{self.success_rate:.1%}",
            'failure_rate': f"{self.failure_rate:.1%}",
            'avg_execution_time': f"{self.avg_execution_time:.2f}s",
            'p50_execution_time': f"{self.p50_execution_time:.2f}s",
            'p95_execution_time': f"{self.p95_execution_time:.2f}s",
            'p99_execution_time': f"{self.p99_execution_time:.2f}s",
            'min_execution_time': f"{self.min_execution_time:.2f}s",
            'max_execution_time': f"{self.max_execution_time:.2f}s",
            'top_errors': dict(sorted(
                self.error_types.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        }

class MetricsCollector:
    """Central metrics collection system."""

    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.start_time = datetime.now()

    def get_or_create_metrics(self, agent_name: str) -> PerformanceMetrics:
        """Get or create metrics for agent."""
        if agent_name not in self.metrics:
            self.metrics[agent_name] = PerformanceMetrics(agent_name=agent_name)
        return self.metrics[agent_name]

    def record_execution(
        self,
        agent_name: str,
        execution_time: float,
        success: bool,
        error_type: Optional[str] = None
    ):
        """Record agent execution."""
        metrics = self.get_or_create_metrics(agent_name)
        metrics.update(execution_time, success, error_type)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all agent metrics."""
        return {
            agent_name: metrics.get_summary()
            for agent_name, metrics in self.metrics.items()
        }

    def get_system_summary(self) -> Dict[str, Any]:
        """Get system-wide summary."""
        total_tasks = sum(m.task_count for m in self.metrics.values())
        total_successes = sum(m.success_count for m in self.metrics.values())
        total_failures = sum(m.failure_count for m in self.metrics.values())

        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'uptime': f"{uptime:.0f}s",
            'total_tasks': total_tasks,
            'total_successes': total_successes,
            'total_failures': total_failures,
            'overall_success_rate': f"{total_successes / total_tasks:.1%}" if total_tasks > 0 else "0%",
            'throughput': f"{total_tasks / uptime:.2f} tasks/s" if uptime > 0 else "0 tasks/s",
            'active_agents': len(self.metrics),
            'agent_rankings': self._get_agent_rankings()
        }

    def _get_agent_rankings(self) -> List[Dict[str, Any]]:
        """Get agents ranked by performance."""
        rankings = []
        for agent_name, metrics in self.metrics.items():
            score = metrics.success_rate * (1 / (metrics.avg_execution_time + 1))
            rankings.append({
                'agent': agent_name,
                'score': f"{score:.3f}",
                'success_rate': f"{metrics.success_rate:.1%}",
                'avg_time': f"{metrics.avg_execution_time:.2f}s"
            })

        return sorted(rankings, key=lambda x: float(x['score']), reverse=True)

# Decorator for automatic metrics collection
def track_performance(agent_name: str, collector: MetricsCollector):
    """Decorator to track function performance."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error_type = None

            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_type = type(e).__name__
                raise
            finally:
                execution_time = time.time() - start_time
                collector.record_execution(agent_name, execution_time, success, error_type)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error_type = None

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_type = type(e).__name__
                raise
            finally:
                execution_time = time.time() - start_time
                collector.record_execution(agent_name, execution_time, success, error_type)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

# Usage example
collector = MetricsCollector()

@track_performance("ml-engineer", collector)
async def train_model(data):
    """Example tracked function."""
    await asyncio.sleep(2)  # Simulate training
    return {"accuracy": 0.95}

async def demo():
    # Execute tasks
    for i in range(10):
        try:
            await train_model({"data": f"dataset-{i}"})
        except Exception:
            pass

    # Print metrics
    metrics = collector.get_all_metrics()
    print("\nAgent Metrics:")
    for agent, stats in metrics.items():
        print(f"\n{agent}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\nSystem Summary:")
    summary = collector.get_system_summary()
    for key, value in summary.items():
        if key != 'agent_rankings':
            print(f"  {key}: {value}")
```

### 2. Caching and Memoization

#### Multi-Level Caching System

```python
# agent_optimization/caching.py
from typing import Any, Callable, Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import json
import pickle
import asyncio

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def access(self) -> None:
        """Record access."""
        self.access_count += 1
        self.last_accessed = datetime.now()

class LRUCache:
    """LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.hits = 0
        self.misses = 0

    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from function and arguments."""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        entry = self.cache.get(key)

        if entry is None:
            self.misses += 1
            return None

        if entry.is_expired():
            self.cache.pop(key)
            self.access_order.remove(key)
            self.misses += 1
            return None

        # Update access
        entry.access()
        self.access_order.remove(key)
        self.access_order.append(key)
        self.hits += 1

        return entry.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        # Remove if exists
        if key in self.cache:
            self.access_order.remove(key)

        # Evict if full
        if len(self.cache) >= self.max_size:
            lru_key = self.access_order.pop(0)
            self.cache.pop(lru_key)

        # Create entry
        expires_at = None
        if ttl or self.default_ttl:
            expires_at = datetime.now() + timedelta(seconds=ttl or self.default_ttl)

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at
        )

        self.cache[key] = entry
        self.access_order.append(key)

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1%}",
            'total_requests': total_requests
        }

def cached(cache: LRUCache, ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key
            key = cache._make_key(func.__name__, args, kwargs)

            # Try cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            cache.put(key, result, ttl)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key
            key = cache._make_key(func.__name__, args, kwargs)

            # Try cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.put(key, result, ttl)

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

# Multi-tier cache system
class TieredCache:
    """Multi-tier cache with hot/warm/cold levels."""

    def __init__(
        self,
        hot_size: int = 100,
        warm_size: int = 500,
        cold_size: int = 2000,
        hot_ttl: int = 300,  # 5 minutes
        warm_ttl: int = 1800,  # 30 minutes
        cold_ttl: int = 3600  # 1 hour
    ):
        self.hot = LRUCache(max_size=hot_size, default_ttl=hot_ttl)
        self.warm = LRUCache(max_size=warm_size, default_ttl=warm_ttl)
        self.cold = LRUCache(max_size=cold_size, default_ttl=cold_ttl)

    def get(self, key: str) -> Optional[Any]:
        """Get from tiered cache."""
        # Try hot cache
        value = self.hot.get(key)
        if value is not None:
            return value

        # Try warm cache
        value = self.warm.get(key)
        if value is not None:
            # Promote to hot
            self.hot.put(key, value)
            return value

        # Try cold cache
        value = self.cold.get(key)
        if value is not None:
            # Promote to warm
            self.warm.put(key, value)
            return value

        return None

    def put(self, key: str, value: Any, priority: str = "warm") -> None:
        """Put in appropriate tier."""
        if priority == "hot":
            self.hot.put(key, value)
        elif priority == "warm":
            self.warm.put(key, value)
        else:
            self.cold.put(key, value)

    def get_stats(self) -> Dict[str, Any]:
        """Get stats for all tiers."""
        return {
            'hot': self.hot.get_stats(),
            'warm': self.warm.get_stats(),
            'cold': self.cold.get_stats()
        }

# Usage example
cache = LRUCache(max_size=100, default_ttl=300)

@cached(cache, ttl=60)
async def expensive_computation(x: int, y: int) -> int:
    """Expensive computation with caching."""
    await asyncio.sleep(2)  # Simulate expensive work
    return x ** y

async def demo_caching():
    # First call - cache miss
    result1 = await expensive_computation(2, 10)
    print(f"Result: {result1}")

    # Second call - cache hit
    result2 = await expensive_computation(2, 10)
    print(f"Result: {result2}")

    # Print stats
    stats = cache.get_stats()
    print(f"\nCache stats: {stats}")
```

### 3. Load Balancing and Resource Management

#### Intelligent Load Balancer

```python
# agent_optimization/load_balancing.py
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import random

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    RANDOM = "random"

@dataclass
class AgentInstance:
    """Agent instance for load balancing."""
    id: str
    agent_type: str
    current_load: int = 0
    max_capacity: int = 10
    weight: float = 1.0
    is_healthy: bool = True
    response_time_ms: float = 0.0

    @property
    def load_percentage(self) -> float:
        """Calculate load percentage."""
        return (self.current_load / self.max_capacity) * 100 if self.max_capacity > 0 else 0

    @property
    def available_capacity(self) -> int:
        """Get available capacity."""
        return max(0, self.max_capacity - self.current_load)

class LoadBalancer:
    """Intelligent load balancer for agent instances."""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED):
        self.strategy = strategy
        self.instances: Dict[str, List[AgentInstance]] = {}
        self.round_robin_index: Dict[str, int] = {}

    def register_instance(self, instance: AgentInstance) -> None:
        """Register agent instance."""
        if instance.agent_type not in self.instances:
            self.instances[instance.agent_type] = []
        self.instances[instance.agent_type].append(instance)

    def select_instance(self, agent_type: str) -> Optional[AgentInstance]:
        """Select instance based on strategy."""
        instances = self.instances.get(agent_type, [])
        if not instances:
            return None

        # Filter healthy instances with capacity
        available = [
            inst for inst in instances
            if inst.is_healthy and inst.current_load < inst.max_capacity
        ]

        if not available:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(agent_type, available)
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded_select(available)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted_select(available)
        else:  # RANDOM
            return random.choice(available)

    def _round_robin_select(
        self,
        agent_type: str,
        instances: List[AgentInstance]
    ) -> AgentInstance:
        """Round-robin selection."""
        if agent_type not in self.round_robin_index:
            self.round_robin_index[agent_type] = 0

        index = self.round_robin_index[agent_type] % len(instances)
        self.round_robin_index[agent_type] += 1

        return instances[index]

    def _least_loaded_select(self, instances: List[AgentInstance]) -> AgentInstance:
        """Select least loaded instance."""
        return min(instances, key=lambda x: x.load_percentage)

    def _weighted_select(self, instances: List[AgentInstance]) -> AgentInstance:
        """Weighted random selection."""
        # Calculate effective weights (higher weight for less loaded)
        weights = [
            inst.weight * (1 - inst.load_percentage / 100)
            for inst in instances
        ]

        return random.choices(instances, weights=weights)[0]

    def acquire(self, agent_type: str) -> Optional[AgentInstance]:
        """Acquire instance for task."""
        instance = self.select_instance(agent_type)
        if instance:
            instance.current_load += 1
        return instance

    def release(self, instance: AgentInstance) -> None:
        """Release instance after task."""
        instance.current_load = max(0, instance.current_load - 1)

    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status."""
        status = {}

        for agent_type, instances in self.instances.items():
            total_capacity = sum(inst.max_capacity for inst in instances)
            total_load = sum(inst.current_load for inst in instances)
            healthy_count = sum(1 for inst in instances if inst.is_healthy)

            status[agent_type] = {
                'instances': len(instances),
                'healthy': healthy_count,
                'total_capacity': total_capacity,
                'current_load': total_load,
                'load_percentage': f"{(total_load / total_capacity * 100):.1f}%" if total_capacity > 0 else "0%",
                'instances_detail': [
                    {
                        'id': inst.id,
                        'load': f"{inst.current_load}/{inst.max_capacity}",
                        'load_pct': f"{inst.load_percentage:.1f}%",
                        'healthy': inst.is_healthy
                    }
                    for inst in instances
                ]
            }

        return status

# Usage example
def demo_load_balancing():
    lb = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_LOADED)

    # Register instances
    for i in range(3):
        lb.register_instance(AgentInstance(
            id=f"ml-engineer-{i}",
            agent_type="ml-engineer",
            max_capacity=5
        ))

    # Simulate task distribution
    print("Distributing tasks...")
    active_instances = []

    for i in range(10):
        instance = lb.acquire("ml-engineer")
        if instance:
            print(f"Task {i} -> {instance.id} (load: {instance.current_load}/{instance.max_capacity})")
            active_instances.append(instance)

    # Release some instances
    for instance in active_instances[:5]:
        lb.release(instance)

    # Print status
    print("\nLoad Balancer Status:")
    status = lb.get_status()
    for agent_type, info in status.items():
        print(f"\n{agent_type}:")
        print(f"  Load: {info['current_load']}/{info['total_capacity']} ({info['load_percentage']})")
        for inst_detail in info['instances_detail']:
            print(f"    {inst_detail['id']}: {inst_detail['load']} ({inst_detail['load_pct']})")
```

## Best Practices

### Performance Monitoring
1. Track comprehensive metrics (latency, throughput, errors)
2. Calculate percentiles (P50, P95, P99) not just averages
3. Monitor success/failure rates
4. Track error types for pattern analysis
5. Implement automated alerting

### Caching Strategy
1. Use TTL to prevent stale data
2. Implement multi-tier caching (hot/warm/cold)
3. Track cache hit rates
4. Size caches appropriately for workload
5. Consider cache invalidation strategies

### Load Balancing
1. Health check instances before routing
2. Consider agent capabilities when balancing
3. Implement graceful degradation
4. Monitor and adjust capacity dynamically
5. Use appropriate strategy for workload

## Quick Reference

```python
# Metrics collection
collector = MetricsCollector()

@track_performance("agent-name", collector)
async def task():
    # Task implementation
    pass

# Caching
cache = LRUCache(max_size=1000, default_ttl=300)

@cached(cache, ttl=60)
async def expensive_task():
    # Task implementation
    pass

# Load balancing
lb = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_LOADED)
instance = lb.acquire("agent-type")
# Use instance
lb.release(instance)
```

## When to Use This Skill

Use when you need to:
- Monitor agent performance in production
- Optimize slow agent operations
- Implement caching for expensive tasks
- Balance load across agent instances
- Track and analyze agent metrics
- Scale agent systems efficiently
- Identify performance bottlenecks
- Improve system throughput

This skill provides production-ready patterns for optimizing agent performance at scale.
