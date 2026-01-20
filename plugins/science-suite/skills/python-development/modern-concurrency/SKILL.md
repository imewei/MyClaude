---
name: modern-concurrency
version: "1.0.0"
description: Master structured concurrency in Python using asyncio TaskGroups and modern primitives. Use when implementing concurrent I/O, managing task lifecycles, or optimizing async applications for Python 3.11+.
---

# Modern Concurrency in Python

Focus on structured concurrency to build reliable and maintainable asynchronous systems.

## Expert Agent

For complex async architecture, deadlock debugging, or performance optimization, delegate to:

- **`python-pro`**: Expert in structured concurrency, `TaskGroups`, and async systems.
  - *Location*: `plugins/science-suite/agents/python-pro.md`

## 1. Structured Concurrency (TaskGroups)

Python 3.11+ introduces `asyncio.TaskGroup`, which provides a robust way to manage multiple tasks with automatic cancellation and error propagation.

```python
import asyncio

async def process_item(item: str):
    await asyncio.sleep(1)
    if item == "fail":
        raise ValueError("Simulated failure")
    print(f"Processed {item}")

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(process_item("A"))
            tg.create_task(process_item("B"))
            tg.create_task(process_item("fail"))
    except* ValueError as eg:
        # Handle exceptions from the TaskGroup
        for e in eg.exceptions:
            print(f"Caught: {e}")
```

## 2. Modern Primitives

| Primitive | Use Case |
|-----------|----------|
| **`TaskGroup`** | Run multiple tasks; if one fails, others are cancelled. |
| **`Barrier`** | Wait for a fixed number of tasks to reach a point. |
| **`Queue`** | Producer-consumer patterns. |
| **`Semaphore`** | Limit concurrency (e.g., rate limiting). |

## 3. GIL Management

For CPU-bound tasks, use `asyncio.to_thread` or `ProcessPoolExecutor` to avoid blocking the event loop.

```python
import asyncio
import time

def blocking_cpu_task():
    time.sleep(2)
    return "Done"

async def main():
    # Runs in a separate thread, not blocking the loop
    result = await asyncio.to_thread(blocking_cpu_task)
    print(result)
```

## 4. Best Practices

- **Avoid `asyncio.gather`**: Use `TaskGroup` for better error handling and cancellation.
- **No Blocking Calls**: Never use `time.sleep()` or blocking I/O in `async def` functions.
- **Timeouts**: Always wrap network calls in `asyncio.timeout()`.
- **Cancellation**: Ensure your code handles `asyncio.CancelledError` if it needs to perform cleanup.

## Checklist

- [ ] `TaskGroup` used for managing multiple concurrent tasks.
- [ ] No blocking calls in the event loop.
- [ ] `asyncio.timeout` applied to external I/O.
- [ ] `CancelledError` handled for resource cleanup.
- [ ] `ExceptionGroup` (except*) used for error handling with TaskGroups.
