# Performance Tuning Guide

Complete guide for optimizing the Claude Code command executor framework.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Performance Profiles](#performance-profiles)
4. [Adaptive Tuning](#adaptive-tuning)
5. [Cache Optimization](#cache-optimization)
6. [Parallel Execution](#parallel-execution)
7. [Benchmarking](#benchmarking)
8. [Best Practices](#best-practices)

---

## Overview

The performance tuning system provides:

- **5-10x performance improvements** through intelligent optimization
- **Adaptive configuration** based on system resources and workload
- **Multi-level caching** with 70%+ hit rates
- **Optimal worker pool sizing** for maximum parallelization
- **6 pre-configured profiles** for different scenarios
- **Comprehensive benchmarking** for validation

### Architecture

```
Performance Tuning System
├── Adaptive Tuning (auto_tuner.py)
│   ├── System profiling
│   ├── Workload analysis
│   └── Configuration generation
│
├── Cache Optimization (cache_tuner.py)
│   ├── Size calculation
│   ├── Hit rate prediction
│   └── Policy optimization
│
├── Parallel Execution (worker_pool_optimizer.py)
│   ├── Worker count optimization
│   ├── Thread vs process selection
│   └── Load balancing
│
├── Profiles (6 configurations)
│   ├── Small project (<1000 files)
│   ├── Medium project (1000-10000 files)
│   ├── Large project (>10000 files)
│   ├── High performance (16+ cores)
│   ├── Memory constrained (<4GB)
│   └── CPU constrained (<4 cores)
│
└── Benchmarks (validation suite)
    ├── Cache benchmarks
    ├── Parallel execution benchmarks
    └── End-to-end benchmarks
```

---

## Quick Start

### 1. Auto-Tune (Recommended)

Automatically detect system and workload, generate optimal configuration:

```python
from performance.performance_tuner import PerformanceTuner

# Create tuner
tuner = PerformanceTuner()

# Run full tuning (includes benchmarks)
result = tuner.tune_full(
    target_metric="throughput",  # or "latency", "memory"
    run_benchmarks=True
)

print(f"Estimated speedup: {result.estimated_speedup}x")
print(f"Selected profile: {result.profile_name}")
print(f"Recommendations: {result.recommendations}")
```

### 2. Select Pre-configured Profile

Choose a profile based on your scenario:

```python
# For small projects
config = tuner.select_profile(project_size="small")

# For high-end systems
config = tuner.select_profile(system_type="high_performance")

# For memory-constrained systems
config = tuner.select_profile(system_type="memory_constrained")
```

### 3. Manual Tuning

Fine-tune specific components:

```python
# Tune cache only
cache_config = tuner.tune_cache(target_hit_rate=75.0)

# Tune worker pool only
worker_config = tuner.tune_workers(task_type=TaskType.CPU_BOUND)
```

---

## Performance Profiles

### Small Project Profile
**Target:** Codebases < 1000 files

- **Cache:** 400MB total (L1: 50MB, L2: 100MB, L3: 250MB)
- **Workers:** 4 workers, 8 threads
- **Memory:** 512MB max
- **Expected Speedup:** 3-5x

**Use Cases:**
- Personal projects
- Small libraries
- Prototypes

### Medium Project Profile
**Target:** Codebases 1000-10000 files

- **Cache:** 2.6GB total (L1: 100MB, L2: 500MB, L3: 2GB)
- **Workers:** 8 workers, 16 threads
- **Memory:** 2GB max
- **Expected Speedup:** 5-8x

**Use Cases:**
- Medium applications
- Standard libraries
- Corporate projects

### Large Project Profile
**Target:** Codebases > 10000 files

- **Cache:** 6.2GB total (L1: 200MB, L2: 1GB, L3: 5GB)
- **Workers:** 16 workers, 32 threads
- **Memory:** 8GB max
- **Expected Speedup:** 8-10x
- **Special Features:** ARC eviction policy, distributed cache

**Use Cases:**
- Enterprise applications
- Large frameworks
- Monorepos

### High Performance Profile
**Target:** High-end systems (16+ cores, 32+ GB RAM)

- **Cache:** 10.5GB total (L1: 500MB, L2: 2GB, L3: 8GB)
- **Workers:** 32 workers, 64 threads
- **Memory:** 16GB max
- **Expected Speedup:** 10-20x
- **Special Features:** Aggressive caching, work stealing, SIMD

**Use Cases:**
- Workstations
- Servers
- GPU-accelerated systems

### Memory Constrained Profile
**Target:** Systems < 4GB RAM

- **Cache:** 250MB total (L1: 25MB, L2: 75MB, L3: 150MB)
- **Workers:** 2 workers, 4 threads
- **Memory:** 256MB max
- **Expected Speedup:** 2-3x
- **Special Features:** Streaming, lazy loading, aggressive eviction

**Use Cases:**
- Low-end laptops
- Small VMs
- Containers

### CPU Constrained Profile
**Target:** Systems < 4 cores

- **Cache:** 1.4GB total (L1: 100MB, L2: 300MB, L3: 1GB)
- **Workers:** 4 workers, 6 threads
- **Memory:** 1GB max
- **Expected Speedup:** 2-4x
- **Special Features:** Thread-only (no process overhead), high cache reliance

**Use Cases:**
- Low-end devices
- Single/dual core systems
- Shared hosting

---

## Adaptive Tuning

### Auto-Tuner

The auto-tuner analyzes system resources and workload patterns to generate optimal configurations.

```python
from performance.adaptive.auto_tuner import AutoTuner, WorkloadProfile

tuner = AutoTuner()

# Create workload profile
workload = WorkloadProfile(
    avg_file_size_kb=50.0,
    file_count=5000,
    io_intensive=True,
    cpu_intensive=False,
    memory_intensive=False
)

# Tune with specific workload
config = tuner.tune(
    workload=workload,
    target_metric="throughput"  # or "latency", "memory"
)

print(f"L1 Cache: {config.l1_cache_mb}MB")
print(f"Max Workers: {config.max_workers}")
print(f"Optimizations: streaming={config.enable_streaming}")
```

### Workload Analyzer

Analyzes runtime workload patterns:

```python
from performance.adaptive.workload_analyzer import WorkloadAnalyzer

analyzer = WorkloadAnalyzer()

# Record file accesses
for file in files:
    analyzer.record_file_access(file, "read", duration_ms=10.0)

# Record operations
analyzer.record_operation("compute", duration_ms=50.0)

# Analyze
analysis = analyzer.analyze()

print(f"Workload type: {analysis.workload_type}")
print(f"Intensity: {analysis.intensity}")
print(f"Patterns: {analysis.patterns}")
print(f"Recommendations: {analysis.recommendations}")
```

---

## Cache Optimization

### Cache Tuning

Optimize cache sizes for target hit rates:

```python
from performance.cache.cache_tuner import CacheTuner, CacheMetrics

tuner = CacheTuner()

# Provide current metrics
current_metrics = CacheMetrics(
    l1_hits=500,
    l2_hits=300,
    l3_hits=100,
    misses=200,
    evictions=50
)

print(f"Current hit rate: {current_metrics.hit_rate():.1f}%")

# Tune for 75% hit rate
config = tuner.tune(
    current_metrics=current_metrics,
    target_hit_rate=75.0
)

print(f"Optimal L1: {config.l1_size_mb}MB")
print(f"Optimal L2: {config.l2_size_mb}MB")
print(f"Optimal L3: {config.l3_size_mb}MB")
print(f"Policy: {config.eviction_policy}")
```

### Cache Policies

**LRU (Least Recently Used)**
- Default policy
- Works well for most workloads
- Simple and efficient

**LFU (Least Frequently Used)**
- Good for hot file concentration
- Tracks access frequency
- Higher overhead

**ARC (Adaptive Replacement Cache)**
- Self-tuning
- Balances recency and frequency
- Best for large, varied workloads
- Used in large project profile

### Estimating Hit Rates

```python
# Estimate hit rate for configuration
working_set_mb = 1000
hit_rate = tuner.estimate_hit_rate(config, working_set_mb)
print(f"Estimated hit rate: {hit_rate:.1f}%")
```

---

## Parallel Execution

### Worker Pool Optimization

Calculate optimal worker counts:

```python
from performance.parallel.worker_pool_optimizer import (
    WorkerPoolOptimizer,
    TaskType
)

optimizer = WorkerPoolOptimizer()

# Optimize for CPU-bound tasks
config = optimizer.optimize(
    task_type=TaskType.CPU_BOUND,
    avg_task_duration_ms=100.0,
    memory_per_task_mb=50.0
)

print(f"Max workers: {config.max_workers}")
print(f"Thread pool: {config.thread_pool_size}")
print(f"Process pool: {config.process_pool_size}")
print(f"Use threads: {config.use_threads}")
print(f"Use processes: {config.use_processes}")
```

### Task Types

**CPU-Bound Tasks**
- Compute-intensive operations
- Optimal workers: CPU count × 1.0-1.5
- Use process pools (avoid GIL)

**I/O-Bound Tasks**
- File operations, network requests
- Optimal workers: CPU count × 2.0-4.0
- Use thread pools (efficient context switching)

**Mixed Tasks**
- Combination of CPU and I/O
- Optimal workers: CPU count × 2.0
- Use hybrid approach

### Speedup Calculation

```python
# Calculate expected speedup
speedup = optimizer.calculate_speedup(
    num_workers=8,
    task_type=TaskType.CPU_BOUND,
    overhead_ms=5.0
)
print(f"Expected speedup: {speedup:.2f}x")

# Find optimal worker count (diminishing returns)
optimal_workers, max_speedup = optimizer.find_optimal_worker_count(
    task_type=TaskType.CPU_BOUND,
    max_workers=64
)
print(f"Optimal workers: {optimal_workers} (speedup: {max_speedup:.2f}x)")
```

---

## Benchmarking

### Running Benchmarks

```python
from performance.benchmarks.benchmark_suite import PerformanceBenchmarkSuite

suite = PerformanceBenchmarkSuite()

# Run all benchmarks
results = suite.run_all()

# View summary
summary = results.get_summary()
print(f"Total benchmarks: {summary['total_benchmarks']}")
print(f"Successful: {summary['successful']}")
print(f"Avg throughput: {summary['avg_throughput']:.1f} ops/s")

# Compare to baseline
comparison = results.compare_to_baseline()
for comp in comparison['comparisons']:
    print(f"{comp['benchmark']}: {comp['speedup']:.2f}x")
```

### Benchmark Categories

1. **Cache Benchmarks**
   - Write throughput
   - Read throughput (hits)
   - Miss handling

2. **Parallel Execution Benchmarks**
   - Thread pool speedup
   - Load balancing
   - Worker optimization

3. **Memory Benchmarks**
   - Allocation performance
   - Memory efficiency

4. **I/O Benchmarks**
   - Read throughput
   - Write throughput

### Setting Baselines

```bash
# Run benchmarks and save as baseline
python performance/benchmarks/benchmark_suite.py

# Results saved to:
# ~/.claude/performance/benchmarks/baseline.json
```

---

## Best Practices

### 1. Profile Selection

**Auto-detect (Recommended):**
```python
tuner = PerformanceTuner()
result = tuner.tune_full()  # Automatic selection
```

**Manual selection guidelines:**
- **Small projects (<1000 files):** Use `small_project_profile`
- **Medium projects (1000-10000 files):** Use `medium_project_profile`
- **Large projects (>10000 files):** Use `large_project_profile`
- **High-end systems (16+ cores, 32+ GB):** Use `high_performance_profile`
- **Low memory (<4GB):** Use `memory_constrained_profile`
- **Few cores (<4):** Use `cpu_constrained_profile`

### 2. Cache Configuration

**Target Hit Rates:**
- **Small projects:** 60-70%
- **Medium projects:** 70-80%
- **Large projects:** 80-90%

**Cache Size Guidelines:**
- Allocate 20-40% of available memory
- L1:L2:L3 ratio = 1:4:10
- Minimum L1: 50MB
- Maximum total: 50% of RAM

### 3. Worker Pool Configuration

**CPU-Bound Workloads:**
- Workers = CPU cores × 1.0
- Use process pools
- Max: 2× CPU cores

**I/O-Bound Workloads:**
- Workers = CPU cores × 3.0
- Use thread pools
- Max: 64 threads

**Mixed Workloads:**
- Workers = CPU cores × 2.0
- Use hybrid approach
- Balance between threads and processes

### 4. Optimization Flags

**Enable streaming when:**
- File sizes > 1MB
- Memory constrained
- Processing large datasets

**Enable lazy loading when:**
- Memory constrained
- Large number of files
- Not all data needed upfront

**Enable prefetching when:**
- Sequential access patterns
- High-end system
- Predictable workload

### 5. Monitoring and Validation

**Always:**
- Run benchmarks after configuration changes
- Monitor cache hit rates
- Track memory usage
- Check for regressions

**Regular maintenance:**
- Clear old cache files (>7 days)
- Update baselines monthly
- Re-tune after major workload changes

### 6. Common Pitfalls

**Avoid:**
- Over-allocating cache (>50% RAM)
- Too many workers (>4× CPU cores)
- Process pools for I/O-bound tasks
- Ignoring memory constraints
- Disabling throttling on constrained systems

### 7. Performance Targets

**Minimum acceptable:**
- Cache hit rate: >50%
- Speedup: >2x
- Memory overhead: <30%

**Good performance:**
- Cache hit rate: >70%
- Speedup: >5x
- Memory overhead: <20%

**Excellent performance:**
- Cache hit rate: >85%
- Speedup: >8x
- Memory overhead: <15%

---

## Advanced Topics

### Custom Profiles

Create custom YAML profiles:

```yaml
name: "custom_profile"
description: "Custom configuration"

cache:
  l1_cache_mb: 150
  l2_cache_mb: 600
  l3_cache_mb: 2500
  ttl_hours: 24

parallel:
  max_workers: 12
  thread_pool_size: 24
  process_pool_size: 12
```

### Dynamic Adjustment

Adjust configuration at runtime:

```python
# Monitor performance
from performance.adaptive.workload_analyzer import WorkloadAnalyzer

analyzer = WorkloadAnalyzer()
# ... record operations ...
analysis = analyzer.analyze()

# Adjust based on intensity
if analysis.intensity > 0.8:
    # High load - increase resources
    config.max_workers = int(config.max_workers * 1.5)
```

### Integration with Framework

```python
from executors.framework import BaseCommandExecutor
from performance.performance_tuner import PerformanceTuner

class MyExecutor(BaseCommandExecutor):
    def __init__(self):
        super().__init__("my_command", CommandCategory.OPTIMIZATION)

        # Apply performance tuning
        tuner = PerformanceTuner()
        self.perf_config = tuner.tune_full()

        # Configure cache
        self.cache_manager.configure(
            self.perf_config.optimal_config['cache']
        )
```

---

## Troubleshooting

### Low Cache Hit Rate

**Symptoms:** Hit rate < 50%

**Solutions:**
1. Increase cache size
2. Adjust TTL (longer for stable workloads)
3. Enable cache warming
4. Check access patterns (random vs sequential)

### Poor Parallel Speedup

**Symptoms:** Speedup < 2x

**Solutions:**
1. Check task type (CPU vs I/O bound)
2. Reduce worker overhead
3. Increase batch sizes
4. Check for GIL contention (use processes)

### Memory Pressure

**Symptoms:** OOM errors, high swapping

**Solutions:**
1. Reduce cache sizes
2. Enable streaming and lazy loading
3. Use memory_constrained profile
4. Reduce worker count

### High CPU Usage

**Symptoms:** CPU constantly at 100%

**Solutions:**
1. Enable throttling
2. Reduce worker count
3. Add delays between operations
4. Use cpu_constrained profile

---

## References

- [Performance Module Documentation](./README.md)
- [Benchmark Results](./BENCHMARK_RESULTS.md)
- [Optimization Cookbook](./OPTIMIZATION_COOKBOOK.md)
- [Best Practices](./PERFORMANCE_BEST_PRACTICES.md)

---

**Version:** 2.0
**Last Updated:** 2025-09-29
**Author:** Claude Code Framework