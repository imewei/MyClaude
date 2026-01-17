# Event Loop Architecture - Visual Guide

## Overview

The asyncio event loop is the heart of Python's async programming model. Understanding its architecture is crucial for writing efficient async code.

## Event Loop Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AsyncIO Event Loop                        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Task/Coroutine Queue                       │ │
│  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐         │ │
│  │  │Task1│  │Task2│  │Task3│  │Task4│  │Task5│   ...   │ │
│  │  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘         │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Event Loop Scheduler                       │ │
│  │                                                         │ │
│  │  • Selects ready task                                  │ │
│  │  • Executes until await/yield                          │ │
│  │  • Switches to next ready task                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  I/O Selector                           │ │
│  │                                                         │ │
│  │  • select() / poll() / epoll()                         │ │
│  │  • Monitors file descriptors                           │ │
│  │  • Returns ready I/O operations                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                Callback Queue                           │ │
│  │  ┌──────┐  ┌──────┐  ┌──────┐                         │ │
│  │  │ CB 1 │  │ CB 2 │  │ CB 3 │   ...                   │ │
│  │  └──────┘  └──────┘  └──────┘                         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Event Loop Execution Flow

```
┌────────────────────────────────────────────────────────────┐
│ 1. SELECT READY TASKS                                       │
│    Loop through all registered tasks                        │
│    Find tasks that are ready to run (not blocked)          │
│    ↓                                                        │
│ 2. EXECUTE TASK                                             │
│    Resume task execution                                    │
│    Run until hit await, yield, or completion               │
│    ↓                                                        │
│ 3. HANDLE I/O                                               │
│    Check if await was on I/O operation                     │
│    Register with I/O selector                              │
│    ↓                                                        │
│ 4. SWITCH CONTEXT                                           │
│    Save current task state                                  │
│    Load next ready task                                     │
│    ↓                                                        │
│ 5. CHECK I/O READINESS                                      │
│    Poll I/O selector for completed operations              │
│    Wake up tasks waiting on completed I/O                  │
│    ↓                                                        │
│ 6. EXECUTE CALLBACKS                                        │
│    Run scheduled callbacks                                  │
│    Handle futures completion                                │
│    ↓                                                        │
│ 7. REPEAT                                                   │
│    Go back to step 1                                        │
│    Continue until no tasks remain                           │
└────────────────────────────────────────────────────────────┘
```

## Task State Machine

```
                    ┌─────────────┐
                    │   PENDING   │
                    │  (created)  │
                    └──────┬──────┘
                           │ asyncio.create_task()
                           ↓
                    ┌─────────────┐
              ┌────→│   RUNNING   │←────┐
              │     │  (executing)│     │
              │     └──────┬──────┘     │
              │            │             │
              │            │ await       │
              │            ↓             │
              │     ┌─────────────┐     │
              │     │   WAITING   │     │
              │     │ (on I/O or  │     │
              │     │   another   │     │
              │     │    task)    │     │
              │     └──────┬──────┘     │
              │            │             │
              └────────────┘ I/O ready  │
                           │ or          │
                           │ task done   │
                           └─────────────┘
                           │
                           ↓
                    ┌─────────────┐
                    │    DONE     │
                    │ (completed) │
                    └─────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
             ┌──────▼──────┐  ┌──▼──────────┐
             │   SUCCESS   │  │  EXCEPTION  │
             │  (result)   │  │   (error)   │
             └─────────────┘  └─────────────┘
```

## Coroutine Execution Model

```python
# Coroutine definition
async def fetch_data(url):
    print("1. Start fetch")
    response = await http_client.get(url)  # ← Suspension point
    print("3. Got response")
    data = await response.json()  # ← Suspension point
    print("5. Parsed JSON")
    return data

# What happens internally:

Step 1: fetch_data() called
        ↓
        Creates coroutine object (not executed yet)

Step 2: await fetch_data() or asyncio.create_task(fetch_data())
        ↓
        Coroutine scheduled on event loop

Step 3: Event loop runs coroutine until first await
        ↓
        Prints "1. Start fetch"
        Hits await http_client.get(url)
        Suspends coroutine
        Yields control back to event loop

Step 4: Event loop continues with other tasks
        ↓
        I/O selector monitors HTTP connection

Step 5: HTTP response arrives
        ↓
        I/O selector notifies event loop
        Coroutine marked as ready

Step 6: Event loop resumes coroutine
        ↓
        Continues after first await
        Prints "3. Got response"
        Hits second await response.json()
        Suspends again

Step 7: (Repeat for JSON parsing)
        ↓
        Eventually completes
        Returns data
```

## Cooperative Multitasking

```
Time →

Task A: ████████──────────────████████──────────────████████
                 await                  await

Task B: ──────────████████──────────────████████──────────────
                         await                  await

Task C: ──────────────────────████████──────────────████████
                                     await                await

Legend:
████ = Task executing
──── = Task suspended (waiting)

Key insight: Only ONE task executes at a time (single-threaded)
Tasks cooperatively yield control at await points
```

## Memory Layout

```
┌─────────────────────────────────────────────────────────┐
│                    Python Process                        │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │              Main Thread                            │ │
│  │                                                     │ │
│  │  ┌──────────────────────────────────────────────┐ │ │
│  │  │        Event Loop                             │ │ │
│  │  │                                               │ │ │
│  │  │  Coroutine 1 Stack: [frames, locals]        │ │ │
│  │  │  Coroutine 2 Stack: [frames, locals]        │ │ │
│  │  │  Coroutine 3 Stack: [frames, locals]        │ │ │
│  │  │  ...                                          │ │ │
│  │  │                                               │ │ │
│  │  │  Shared Heap Memory                          │ │ │
│  │  │  ├─ Global variables                         │ │ │
│  │  │  ├─ Shared data structures                   │ │ │
│  │  │  └─ I/O buffers                              │ │ │
│  │  └──────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

Key: All coroutines share the same thread and memory space
No thread synchronization needed (no locks required!)
But: No true parallelism for CPU-bound work
```

## Performance Characteristics

```
┌────────────────────────────────────────────────────────┐
│         Async vs Threading vs Multiprocessing           │
└────────────────────────────────────────────────────────┘

Async (asyncio):
├─ Context Switch: ~1-10 microseconds (very fast)
├─ Memory per task: ~10-50 KB (lightweight)
├─ Max concurrent: 10,000+ tasks (excellent scalability)
├─ CPU usage: Single core only
└─ Best for: I/O-bound operations

Threading:
├─ Context Switch: ~50-100 microseconds
├─ Memory per thread: ~1-8 MB (heavier)
├─ Max concurrent: 100-1000 threads (limited)
├─ CPU usage: Can use multiple cores (with limitations)
└─ Best for: I/O-bound with C extensions

Multiprocessing:
├─ Context Switch: ~100-1000 microseconds
├─ Memory per process: ~10-50 MB (heavy)
├─ Max concurrent: CPU core count (typically 4-16)
├─ CPU usage: Full parallel execution
└─ Best for: CPU-bound operations
```

## Real-World Example

```python
import asyncio
import aiohttp
import time

async def fetch_url(session, url):
    """
    Execution visualization:

    T=0ms:   fetch_url() starts
    T=1ms:   await session.get(url) → SUSPENDS
    T=2ms:   [Other tasks run while waiting for network]
    T=150ms: Network response arrives → RESUMES
    T=151ms: await response.text() → SUSPENDS
    T=152ms: [Other tasks run while waiting for body]
    T=200ms: Body received → RESUMES
    T=201ms: Returns result

    Total wall time: 201ms
    Actual CPU time: ~3ms (execution time)
    """
    print(f"Starting {url}")
    async with session.get(url) as response:
        text = await response.text()
        print(f"Completed {url}: {len(text)} bytes")
        return text

async def main():
    """
    Concurrent execution visualization:

    URL1: ████──────────────████──────────────
    URL2: ──████──────────────████──────────
    URL3: ────████──────────────████────────
    URL4: ──────████──────────────████──────
    URL5: ────────████──────────────████────

    Sequential time: 5 × 200ms = 1000ms
    Concurrent time: ~250ms (overlapping I/O)
    Speedup: 4x
    """
    urls = [
        'https://example.com',
        'https://httpbin.org/delay/1',
        'https://api.github.com',
        'https://python.org',
        'https://pypi.org'
    ]

    async with aiohttp.ClientSession() as session:
        # Create all tasks at once
        tasks = [fetch_url(session, url) for url in urls]

        # Execute concurrently
        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        print(f"\nFetched {len(results)} URLs in {elapsed:.2f}s")
        print(f"Average per URL: {elapsed/len(results):.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

## Event Loop Implementation Comparison

```
┌─────────────────────────────────────────────────────────┐
│              Event Loop Implementations                  │
└─────────────────────────────────────────────────────────┘

asyncio (standard library):
├─ Based on: selectors module
├─ Selector: select/poll/epoll (platform-dependent)
├─ Performance: Good (10K-100K req/s)
├─ Maturity: Stable, well-tested
└─ Use case: General purpose

uvloop (3rd party):
├─ Based on: libuv (Node.js event loop)
├─ Selector: epoll/kqueue
├─ Performance: Excellent (2-4x faster than asyncio)
├─ Maturity: Production-ready
└─ Use case: High-performance servers

trio (3rd party):
├─ Based on: Custom implementation
├─ Selector: Platform-specific
├─ Performance: Good
├─ Maturity: Modern, well-designed
└─ Use case: Structured concurrency
```

## Key Takeaways

1. **Event loop is single-threaded** - No parallel execution, only concurrent
2. **Cooperative multitasking** - Tasks must explicitly yield control with `await`
3. **Lightweight tasks** - Can handle 10,000+ concurrent tasks
4. **Best for I/O** - Network, file I/O, databases
5. **Context switching is cheap** - ~1-10 microseconds per switch
6. **No GIL issues** - All in one thread, no lock contention
7. **Explicit suspension points** - Only at `await` keywords

## Common Misconceptions

❌ **"Async makes code faster"**
✅ Async makes I/O-bound code more efficient by avoiding blocking

❌ **"Async is parallel"**
✅ Async is concurrent but not parallel (single-threaded)

❌ **"You need async for everything"**
✅ Only beneficial for I/O-bound operations with many connections

❌ **"Async avoids the GIL"**
✅ Async still has the GIL, but it doesn't matter (single thread)

## Further Reading

- Task Lifecycle (Coming Soon) - Complete task state management
- [Async Anti-Patterns](async-anti-patterns.md) - Common mistakes
- Concurrency Models Compared (Coming Soon) - When to use what
