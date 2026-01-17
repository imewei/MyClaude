---
name: python-performance-optimization
version: "1.0.7"
description: Profile and optimize Python code using cProfile, line_profiler, memory_profiler, and py-spy. Use when analyzing slow code, optimizing CPU-intensive operations, reducing memory consumption, implementing caching with lru_cache, using NumPy for vectorization, multiprocessing for CPU-bound tasks, or async/await for I/O-bound tasks.
---

# Python Performance Optimization

## Profiling Tools

| Tool | Purpose | Command |
|------|---------|---------|
| cProfile | CPU profiling | `python -m cProfile -o out.prof script.py` |
| line_profiler | Line-by-line | `kernprof -l -v script.py` |
| memory_profiler | Memory usage | `python -m memory_profiler script.py` |
| py-spy | Production/live | `py-spy record -o profile.svg --pid 12345` |
| timeit | Quick benchmarks | `python -m timeit "expression"` |

## cProfile Usage

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
result = main()

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
stats.dump_stats("profile.prof")
```

## Line Profiler

```python
# Add @profile decorator (provided by line_profiler)
@profile
def process_data(data):
    result = []
    for item in data:
        processed = item * 2
        result.append(processed)
    return result

# Run: kernprof -l -v script.py
```

## Memory Profiler

```python
from memory_profiler import profile

@profile
def memory_intensive():
    big_list = [i for i in range(1000000)]
    big_dict = {i: i**2 for i in range(100000)}
    return sum(big_list)

# Run: python -m memory_profiler script.py
```

## Optimization Patterns

### List Comprehensions vs Loops

```python
# Slow
result = []
for i in range(n):
    result.append(i**2)

# Fast (2-3x faster)
result = [i**2 for i in range(n)]
```

### Generators for Memory

```python
# High memory: creates full list
data = [i**2 for i in range(1000000)]

# Low memory: generates on demand
data = (i**2 for i in range(1000000))
```

### String Concatenation

```python
# Slow: O(n²) due to immutable strings
result = ""
for item in items:
    result += str(item)

# Fast: O(n)
result = "".join(str(item) for item in items)
```

### Dict vs List Lookup

```python
# O(n) list search
if target in items_list:
    ...

# O(1) dict/set lookup
if target in items_set:
    ...
```

## Caching with lru_cache

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Check cache stats
print(fibonacci.cache_info())
```

## NumPy Vectorization

```python
import numpy as np

# Slow: Python loop
def python_sum(n):
    return sum(range(n))

# Fast: NumPy (10-100x faster)
def numpy_sum(n):
    return np.arange(n).sum()

# Vectorized operations
a = np.arange(100000)
b = np.arange(100000)
result = a * b  # Element-wise, no loop
```

## Multiprocessing (CPU-bound)

```python
import multiprocessing as mp

def cpu_task(n):
    return sum(i**2 for i in range(n))

if __name__ == "__main__":
    with mp.Pool(processes=4) as pool:
        results = pool.map(cpu_task, [1000000] * 4)
```

## Async I/O (I/O-bound)

```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

# 4 concurrent requests instead of sequential
asyncio.run(main())
```

## Memory Optimization

### __slots__ for Classes

```python
class Regular:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Slotted:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Slotted uses ~40% less memory per instance
```

### Weak References

```python
import weakref

# Allows garbage collection when no strong refs
weak_cache = weakref.WeakValueDictionary()
```

### Memory Leak Detection

```python
import tracemalloc

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()

# Run code
run_code()

snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
for stat in top_stats[:10]:
    print(stat)
```

## Database Optimization

```python
# Slow: individual commits
for item in items:
    cursor.execute("INSERT ...", (item,))
    conn.commit()

# Fast: batch with single commit
cursor.executemany("INSERT ...", items)
conn.commit()
```

## Benchmark Decorator

```python
import time
from functools import wraps

def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter() - start:.6f}s")
        return result
    return wrapper
```

## Optimization Hierarchy

| Priority | Technique | Typical Speedup |
|----------|-----------|-----------------|
| 1 | Better algorithm | 10-1000x |
| 2 | Use built-ins (C) | 2-10x |
| 3 | NumPy vectorization | 10-100x |
| 4 | Caching | 10-1000x |
| 5 | Multiprocessing | 2-Nx (N cores) |
| 6 | Async I/O | 2-10x |
| 7 | Micro-optimizations | 1.1-2x |

## Best Practices

| Practice | Guideline |
|----------|-----------|
| Profile first | Measure before optimizing |
| Focus on hot paths | Optimize frequently-run code |
| Algorithm over micro | Better algorithm > faster code |
| Use built-ins | Implemented in C |
| Avoid globals | Local variable lookup faster |
| Batch I/O | Reduce system calls |
| Connection pooling | Reuse database/HTTP connections |

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Optimizing without profiling | Profile first |
| Global variables | Use local variables |
| String += in loops | Use ''.join() |
| List for membership | Use set/dict |
| Unnecessary copies | Use views, generators |
| No connection pooling | Pool DB/HTTP connections |

## Checklist

- [ ] Profiled code with cProfile/py-spy
- [ ] Identified actual bottlenecks
- [ ] Used appropriate data structures
- [ ] Applied caching where beneficial
- [ ] Used generators for large datasets
- [ ] Multiprocessing for CPU-bound tasks
- [ ] Async I/O for I/O-bound tasks
- [ ] Benchmarked before/after optimization
