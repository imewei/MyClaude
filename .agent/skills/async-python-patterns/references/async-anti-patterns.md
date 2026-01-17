# Async Python Anti-Patterns

## Overview

This comprehensive guide catalogs common async programming mistakes, their consequences, and how to fix them. Learning what NOT to do is just as important as learning best practices.

---

## 🚫 Anti-Pattern 1: Blocking the Event Loop

### ❌ Bad: Synchronous I/O in Async Function

```python
import asyncio
import time
import requests  # Synchronous library

async def fetch_data(url):
    # WRONG: Blocks entire event loop for 2 seconds!
    response = requests.get(url)  # Synchronous blocking call
    return response.text

async def main():
    # All these run sequentially, defeating async purpose
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)  # Takes N × 2 seconds
```

**Problem**: `requests.get()` blocks the entire event loop. No other tasks can run while waiting for the network response.

**Impact**:
- ⏱️ Sequential execution (no concurrency benefit)
- 🔒 Entire event loop frozen during I/O
- 📉 Performance worse than sequential code (overhead without benefit)

### ✅ Good: Async I/O

```python
import asyncio
import aiohttp  # Async library

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()  # Non-blocking

async def main():
    # These truly run concurrently
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)  # Takes ~max(individual times)
```

**Fix**: Use async libraries (`aiohttp`, `aiomysql`, `aiofiles`, etc.)

---

## 🚫 Anti-Pattern 2: CPU-Intensive Work in Async

### ❌ Bad: Heavy Computation in Event Loop

```python
import asyncio

async def process_data(data):
    # WRONG: CPU-intensive work blocks event loop
    result = 0
    for i in range(10_000_000):  # Blocks for seconds
        result += i ** 2
    return result

async def main():
    # These run sequentially despite being "async"
    tasks = [process_data(data) for data in datasets]
    results = await asyncio.gather(*tasks)  # No concurrency!
```

**Problem**: CPU-bound work doesn't yield control. Event loop can't switch tasks.

**Impact**:
- 🐌 No concurrency benefit
- 🔥 Wasted async overhead
- ❌ Other tasks starved

### ✅ Good: Offload to Process Pool

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

def process_data_sync(data):
    # CPU-intensive work in separate process
    result = 0
    for i in range(10_000_000):
        result += i ** 2
    return result

async def process_data(data):
    # Run in process pool (true parallelism)
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, process_data_sync, data)
    return result

async def main():
    # CPU work parallelized across cores
    tasks = [process_data(data) for data in datasets]
    results = await asyncio.gather(*tasks)  # True parallel execution
```

**Fix**: Use `run_in_executor()` with ProcessPoolExecutor for CPU-bound work.

---

## 🚫 Anti-Pattern 3: Not Awaiting Coroutines

### ❌ Bad: Forgetting await

```python
import asyncio

async def save_to_database(data):
    await db.insert(data)
    return True

async def main():
    # WRONG: Coroutine not awaited!
    result = save_to_database(data)  # Returns coroutine object
    print(result)  # Prints <coroutine object>, data NOT saved!

    # Also wrong: Creating task but not awaiting
    task = asyncio.create_task(save_to_database(data))
    # Task never completes if we exit before it finishes
```

**Problem**: Coroutine created but never executed. Data never saved.

**Warning**: Python 3.11+ will show: `RuntimeWarning: coroutine was never awaited`

### ✅ Good: Always Await or Track Tasks

```python
import asyncio

async def save_to_database(data):
    await db.insert(data)
    return True

async def main():
    # Option 1: Await immediately
    result = await save_to_database(data)
    print(result)  # True

    # Option 2: Create task and track it
    task = asyncio.create_task(save_to_database(data))
    # ... do other work ...
    result = await task  # Ensure task completes

    # Option 3: Use gather for multiple tasks
    tasks = [save_to_database(d) for d in data_list]
    results = await asyncio.gather(*tasks)  # All complete
```

---

## 🚫 Anti-Pattern 4: Mixing Sync and Async Code Incorrectly

### ❌ Bad: Calling Async from Sync

```python
import asyncio

async def async_operation():
    await asyncio.sleep(1)
    return "done"

def sync_function():
    # WRONG: Can't await in sync function
    result = await async_operation()  # SyntaxError!

    # Also wrong:
    result = async_operation()  # Returns coroutine, not result

    # Still wrong:
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(async_operation())  # Nested event loops!
```

**Problem**: Cannot call async code from sync code without proper handling.

### ✅ Good: Proper Async/Sync Bridge

```python
import asyncio

async def async_operation():
    await asyncio.sleep(1)
    return "done"

def sync_function():
    # Option 1: Use asyncio.run() (creates new event loop)
    result = asyncio.run(async_operation())  # Only if no loop exists

    # Option 2: Run in thread pool (if loop already running)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, async_operation())
        result = future.result()

    # Option 3: Make sync_function async
    # (Best solution - go async all the way)
```

**Fix**: Either go fully async or use proper bridging techniques.

---

## 🚫 Anti-Pattern 5: Creating Too Many Tasks

### ❌ Bad: Unbounded Task Creation

```python
import asyncio

async def process_item(item):
    await asyncio.sleep(0.1)
    return item * 2

async def main():
    # WRONG: Creates 1,000,000 tasks at once!
    items = range(1_000_000)
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)  # Memory explosion!
```

**Problem**: Creates millions of tasks simultaneously, exhausting memory.

**Impact**:
- 💾 Memory exhaustion (tens of GB)
- 🐌 Task scheduling overhead
- 💥 Possible crash

### ✅ Good: Use Semaphore or Queue

```python
import asyncio

async def process_item(item, semaphore):
    async with semaphore:  # Limit concurrent tasks
        await asyncio.sleep(0.1)
        return item * 2

async def main():
    # Limit to 100 concurrent tasks
    semaphore = asyncio.Semaphore(100)

    items = range(1_000_000)
    tasks = [process_item(item, semaphore) for item in items]
    results = await asyncio.gather(*tasks)  # Controlled concurrency

# Even better: Use queue pattern
async def worker(queue):
    while True:
        item = await queue.get()
        if item is None:  # Sentinel
            break
        result = await process_item(item)
        queue.task_done()

async def main_with_queue():
    queue = asyncio.Queue()
    workers = [asyncio.create_task(worker(queue)) for _ in range(100)]

    # Feed queue
    for item in range(1_000_000):
        await queue.put(item)

    await queue.join()  # Wait for all items processed

    # Stop workers
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers)
```

---

## 🚫 Anti-Pattern 6: Not Handling Exceptions

### ❌ Bad: Silent Exception Swallowing

```python
import asyncio

async def risky_operation(item):
    if item % 3 == 0:
        raise ValueError(f"Item {item} is bad")
    return item * 2

async def main():
    # WRONG: One exception kills entire gather
    tasks = [risky_operation(i) for i in range(10)]
    try:
        results = await asyncio.gather(*tasks)
    except ValueError:
        # Which task failed? What about successful tasks?
        print("Something failed")  # Lost all results!

    # Also wrong: fire-and-forget tasks
    for i in range(10):
        asyncio.create_task(risky_operation(i))  # Exceptions ignored!
```

**Problem**: Exceptions lost or entire batch fails.

### ✅ Good: Proper Exception Handling

```python
import asyncio

async def risky_operation(item):
    if item % 3 == 0:
        raise ValueError(f"Item {item} is bad")
    return item * 2

async def safe_operation(item):
    """Wrapper that catches exceptions"""
    try:
        return await risky_operation(item)
    except Exception as e:
        return {"error": str(e), "item": item}

async def main():
    # Option 1: return_exceptions=True
    tasks = [risky_operation(i) for i in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # results contains both values and exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")

    # Option 2: Wrapper function
    tasks = [safe_operation(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # Option 3: TaskGroup (Python 3.11+)
    async with asyncio.TaskGroup() as tg:
        for i in range(10):
            tg.create_task(risky_operation(i))
    # Raises ExceptionGroup if any task fails
```

---

## 🚫 Anti-Pattern 7: Race Conditions with Shared State

### ❌ Bad: Unprotected Shared State

```python
import asyncio

counter = 0  # Shared state

async def increment():
    global counter
    # WRONG: Race condition! (even though single-threaded)
    temp = counter
    await asyncio.sleep(0)  # Suspension point!
    counter = temp + 1

async def main():
    tasks = [increment() for _ in range(1000)]
    await asyncio.gather(*tasks)
    print(counter)  # Expected: 1000, Actual: ~100-300 (unpredictable!)
```

**Problem**: Context switch during read-modify-write creates race condition.

**How it happens**:
```
Task 1: Read counter=0
Task 1: await (suspends)
Task 2: Read counter=0 (still 0!)
Task 2: await (suspends)
Task 1: Write counter=1
Task 2: Write counter=1 (overwrites!)
Result: counter=1, should be 2
```

### ✅ Good: Use Locks or Atomic Operations

```python
import asyncio

counter = 0
lock = asyncio.Lock()

async def increment_safe():
    global counter
    async with lock:  # Atomic section
        temp = counter
        await asyncio.sleep(0)
        counter = temp + 1

# Better: Use atomic operations (no await in critical section)
async def increment_atomic():
    global counter
    counter += 1  # Atomic if no await
    await asyncio.sleep(0)  # Suspension after atomic op

# Best: Avoid shared state (functional approach)
async def increment_functional(value):
    await asyncio.sleep(0)
    return value + 1

async def main():
    results = await asyncio.gather(*[increment_functional(i) for i in range(1000)])
    total = sum(results)  # Predictable!
```

---

## 🚫 Anti-Pattern 8: Deadlocks with Locks

### ❌ Bad: Lock Ordering Issues

```python
import asyncio

lock1 = asyncio.Lock()
lock2 = asyncio.Lock()

async def task_a():
    async with lock1:
        await asyncio.sleep(0.1)
        async with lock2:  # Waits for lock2
            print("Task A")

async def task_b():
    async with lock2:
        await asyncio.sleep(0.1)
        async with lock1:  # Waits for lock1
            print("Task B")

async def main():
    # DEADLOCK! task_a holds lock1, waits for lock2
    #           task_b holds lock2, waits for lock1
    await asyncio.gather(task_a(), task_b())  # Hangs forever
```

**Problem**: Circular lock dependency causes deadlock.

### ✅ Good: Consistent Lock Ordering

```python
import asyncio

lock1 = asyncio.Lock()
lock2 = asyncio.Lock()

async def task_a():
    # Always acquire in same order
    async with lock1:
        async with lock2:
            print("Task A")

async def task_b():
    # Same order prevents deadlock
    async with lock1:
        async with lock2:
            print("Task B")

# Even better: Use single lock for related resources
resource_lock = asyncio.Lock()

async def task_with_single_lock():
    async with resource_lock:
        # Access all related resources atomically
        pass
```

---

## 🚫 Anti-Pattern 9: Not Cleaning Up Resources

### ❌ Bad: Resource Leaks

```python
import asyncio
import aiohttp

async def fetch_data(url):
    # WRONG: Session never closed on exception
    session = aiohttp.ClientSession()
    response = await session.get(url)
    data = await response.text()
    await session.close()  # Skipped if exception!
    return data

async def main():
    tasks = [fetch_data(url) for url in urls]
    await asyncio.gather(*tasks)  # Leaked sessions!
```

**Problem**: Exceptions skip cleanup, leaking connections.

### ✅ Good: Use Async Context Managers

```python
import asyncio
import aiohttp

async def fetch_data(url):
    # Automatic cleanup even on exception
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# For custom resources
class AsyncResource:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()  # Always called
        return False

async def use_resource():
    async with AsyncResource() as resource:
        await resource.do_something()
    # Guaranteed cleanup
```

---

## 🚫 Anti-Pattern 10: Using asyncio.run() Multiple Times

### ❌ Bad: Multiple Event Loops

```python
import asyncio

async def task1():
    return "result1"

async def task2():
    return "result2"

def main():
    # WRONG: Creates multiple event loops
    result1 = asyncio.run(task1())
    result2 = asyncio.run(task2())  # New loop, old one destroyed

    # Inefficient and error-prone
```

**Problem**: Creates/destroys event loop repeatedly (expensive).

### ✅ Good: Single Event Loop

```python
import asyncio

async def task1():
    return "result1"

async def task2():
    return "result2"

async def main():
    # Run in single event loop
    result1 = await task1()
    result2 = await task2()

    # Or concurrently
    result1, result2 = await asyncio.gather(task1(), task2())

if __name__ == "__main__":
    # Single asyncio.run() call
    asyncio.run(main())
```

---

## Quick Reference: Common Mistakes

| Mistake | Impact | Fix |
|---------|--------|-----|
| Blocking I/O | No concurrency | Use async libraries |
| CPU-bound work | Event loop blocked | Use ProcessPoolExecutor |
| Forgot await | Task not executed | Always await coroutines |
| Unbounded tasks | Memory exhaustion | Use Semaphore/Queue |
| Silent exceptions | Data loss | Use return_exceptions |
| Shared state races | Data corruption | Use locks or avoid shared state |
| Resource leaks | Connection exhaustion | Use async context managers |
| Multiple asyncio.run() | Performance loss | Single event loop |
| Deadlocks | Hang forever | Consistent lock ordering |
| Mixing sync/async | Broken event loop | Go fully async |

## Testing for Anti-Patterns

```python
import asyncio
import pytest

# Test for blocking
async def test_not_blocking():
    start = asyncio.get_event_loop().time()
    await my_async_function()
    elapsed = asyncio.get_event_loop().time() - start
    assert elapsed < 1.0  # Should not block for >1s

# Test for proper cleanup
async def test_resource_cleanup():
    with pytest.raises(Exception):
        await failing_function()
    # Verify resources closed
    assert resource.is_closed()

# Test for race conditions
async def test_no_race_conditions():
    results = await asyncio.gather(*[increment() for _ in range(1000)])
    assert counter == 1000  # Should be deterministic
```

## Further Reading

- [Event Loop Architecture](event-loop-architecture.md) - How async really works
- Task Lifecycle (Coming Soon) - Understanding task states
- Concurrency Models Compared (Coming Soon) - When to use async
