---
name: python-development
description: Master modern Python systems engineering for scientific computing. Covers type-driven design, Rust extensions (PyO3), structured concurrency (TaskGroups), async patterns (async context managers, async generators, pytest-asyncio), profiling (cProfile, line_profiler, memory_profiler, py-spy, tracemalloc), robust testing (Hypothesis), and uv-based packaging. Use when architecting Python packages, writing Rust extensions with PyO3, implementing async concurrency, profiling slow Python code, or setting up property-based testing.
---

# Python Systems Engineering

Advanced patterns and toolchains for high-performance, robust Python development.

## Expert Agent

For all advanced Python development, architecture, and optimization tasks, delegate to:

- **`python-pro`**: Expert Python Systems Engineer focusing on rigor, performance, and modern standards.
  - *Location*: `plugins/science-suite/agents/python-pro.md`

## Core Skills

### 1. [Type-Driven Design](../type-driven-design/SKILL.md)
Structural typing with Protocols, Generics for reusability, and strict static analysis with Pyright/Mypy.

### 2. [Rust Extensions](../rust-extensions/SKILL.md)
Writing performance-critical bottlenecks in Rust using PyO3 and Maturin for 100x speedups.

### 3. [Modern Concurrency](../modern-concurrency/SKILL.md)
Structured concurrency using `asyncio.TaskGroup` (Python 3.11+) for reliable task management and error propagation.

### 4. [Robust Testing](../robust-testing/SKILL.md)
Property-based testing with Hypothesis and advanced Pytest patterns to ensure algorithmic correctness.

### 5. [Modern Packaging](../python-packaging-advanced/SKILL.md)
Using `uv` for blazing-fast dependency management, workspaces (monorepos), and reproducible environments.

## Routing Decision Tree

```
What is the primary Python engineering concern?
|
+-- Structural typing, Protocols, Generics, or strict static analysis?
|   --> science-suite:type-driven-design
|
+-- Writing Rust extensions with PyO3 / Maturin for performance-critical code?
|   --> science-suite:rust-extensions
|
+-- Structured concurrency with asyncio.TaskGroup or async patterns?
|   --> science-suite:modern-concurrency
|
+-- Property-based testing with Hypothesis or advanced Pytest patterns?
|   --> science-suite:robust-testing
|
+-- Dependency management, uv workspaces, or reproducible packaging?
|   --> science-suite:python-packaging-advanced
|
+-- asyncio idioms (run/context managers/generators), profiling, or CPython speed?
|   --> stay here: `## Async Patterns` / `## Python Profiling` below
|
+-- Exception hierarchy design, retries, or circuit breakers?
|   --> dev-suite:error-handling-patterns
|
+-- Python 2->3 migration, deprecated APIs, or legacy dependency upgrades?
|   --> dev-suite:modernization-migration
|
+-- None of the above / concern is ambiguous or spans multiple areas?
    --> Delegate to python-pro for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## The Python Pro Mindset

1.  **Strict Typing**: `mypy --strict` or `pyright` strict mode must always pass.
2.  **Zero Global State**: Use dependency injection and avoid the `global` keyword.
3.  **Performance First**: Profile with `py-spy`, vectorize with NumPy/JAX, and offload to Rust when necessary.
4.  **Structured Concurrency**: Avoid `asyncio.gather` in favor of `TaskGroup`.
5.  **Modern Tooling**: Standardize on `uv`, `ruff`, and `hatchling`.

## Async Patterns

Task lifecycle, cancellation, and concurrency limiting live in
[Modern Concurrency](../modern-concurrency/SKILL.md) — use `asyncio.TaskGroup`,
not `asyncio.gather`. This section covers the surrounding asyncio idioms.

### Entry Point

```python
import asyncio

async def main() -> str:
    await asyncio.sleep(1)
    return "done"

asyncio.run(main())  # never call an event loop from inside a running loop
```

### Async Context Manager

```python
class AsyncDBConnection:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.connection = None

    async def __aenter__(self):
        self.connection = await connect(self.dsn)
        return self.connection

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.connection.close()

async with AsyncDBConnection("postgresql://localhost") as conn:
    result = await conn.fetch("SELECT * FROM users")
```

### Async Generator (streaming)

```python
from collections.abc import AsyncIterator

async def fetch_pages(url: str, max_pages: int) -> AsyncIterator[dict]:
    for page in range(1, max_pages + 1):
        yield await fetch(f"{url}?page={page}")

async for page_data in fetch_pages("https://api.example.com", 10):
    process(page_data)
```

### Testing Async Code

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    assert await fetch_data("https://api.example.com") is not None

@pytest.mark.asyncio
async def test_timeout():
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(1.0):
            await slow_operation(5)
```

### Common Pitfalls

| Pitfall | Wrong | Correct |
|---------|-------|---------|
| Forgetting await | `result = async_func()` | `result = await async_func()` |
| Blocking the loop | `time.sleep(1)` | `await asyncio.sleep(1)` |
| Blocking CPU work inline | `heavy_calc()` in `async def` | `await asyncio.to_thread(heavy_calc)` |
| Swallowing cancellation | bare `except Exception` | `except asyncio.CancelledError: cleanup(); raise` |
| Calling async from sync | `await func()` in a sync body | `asyncio.run(func())` |

### Async Ecosystem

| Library | Purpose |
|---------|---------|
| aiohttp / httpx | HTTP client and server |
| FastAPI | Async web framework |
| asyncpg | PostgreSQL driver |
| aiofiles | File I/O |
| pytest-asyncio | Async test support |

## Python Profiling

Profile before optimizing — confirm the bottleneck is in Python, not I/O or
the allocator. Once located, escalate through
[Rust Extensions](../rust-extensions/SKILL.md) or NumPy/JAX vectorization.

| Tool | Purpose | Invocation |
|------|---------|------------|
| cProfile | Deterministic CPU profile | `python -m cProfile -o out.prof script.py` |
| line_profiler | Line-by-line hot loop | `kernprof -l -v script.py` (`@profile` decorator) |
| memory_profiler | Per-line memory | `python -m memory_profiler script.py` |
| py-spy | Sampling profiler, live/production | `py-spy record -o profile.svg --pid 12345` |
| tracemalloc | Allocation diffing / leak hunting | stdlib, no instrumentation needed |
| timeit | Micro-benchmark | `python -m timeit "expression"` |

### cProfile in-process

```python
import cProfile, pstats

profiler = cProfile.Profile()
profiler.enable()
result = main()
profiler.disable()

stats = pstats.Stats(profiler).sort_stats("cumulative")
stats.print_stats(10)
stats.dump_stats("profile.prof")  # inspect with snakeviz / tuna
```

### Memory leak hunting with tracemalloc

```python
import tracemalloc

tracemalloc.start()
before = tracemalloc.take_snapshot()
run_code()
after = tracemalloc.take_snapshot()

for stat in after.compare_to(before, "lineno")[:10]:
    print(stat)
```

### Optimization Hierarchy

Work top-down; stop when the measured target is met.

| Priority | Technique | Typical Speedup |
|----------|-----------|-----------------|
| 1 | Better algorithm / data structure | 10-1000x |
| 2 | NumPy / JAX vectorization | 10-100x |
| 3 | Caching (`functools.lru_cache`) | 10-1000x |
| 4 | Rust extension (PyO3) for the hot kernel | 10-100x |
| 5 | `ProcessPoolExecutor` for CPU-bound work | ~N cores |
| 6 | asyncio for I/O-bound work | 2-10x |
| 7 | Micro-optimizations (`__slots__`, locals) | 1.1-2x |

`__slots__` cuts ~40% of per-instance memory for attribute-heavy classes;
`weakref.WeakValueDictionary` lets cached objects be collected.

## Checklist

- [ ] Strict type checking enabled and passing.
- [ ] `uv.lock` managed and consistent across the workspace.
- [ ] Performance bottlenecks identified via profiling and addressed.
- [ ] Profiled with cProfile or py-spy *before* any optimization work.
- [ ] Algorithmic properties verified with property-based tests.
- [ ] Async code uses structured concurrency primitives.
- [ ] Async resources acquired via `async with`; streams via `async for`.
- [ ] Async tests run under `pytest-asyncio`.
