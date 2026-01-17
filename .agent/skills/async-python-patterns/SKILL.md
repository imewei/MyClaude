---
name: async-python-patterns
version: "1.0.7"
description: Master Python asyncio, concurrent programming, and async/await patterns for high-performance non-blocking applications. Use when writing async/await code, implementing FastAPI or aiohttp applications, building WebSocket servers, creating async context managers or generators, handling async database operations with asyncpg, implementing async HTTP clients with aiohttp, writing async tests with pytest-asyncio, or converting synchronous code to asynchronous patterns.
---

# Async Python Patterns

## Core Concepts

| Concept | Description |
|---------|-------------|
| Event Loop | Single-threaded cooperative multitasking scheduler |
| Coroutine | `async def` function that can pause/resume |
| Task | Scheduled coroutine on event loop |
| Future | Low-level result placeholder |
| Async Context Manager | `async with` for resource cleanup |
| Async Iterator | `async for` for streaming data |

## Basic Patterns

### Pattern 1: Entry Point
```python
import asyncio

async def main():
    await asyncio.sleep(1)
    return "done"

asyncio.run(main())  # Python 3.7+
```

### Pattern 2: Concurrent Execution
```python
async def fetch_all(user_ids: list[int]) -> list[dict]:
    tasks = [fetch_user(uid) for uid in user_ids]
    return await asyncio.gather(*tasks)
```

### Pattern 3: Task Management
```python
async def main():
    task1 = asyncio.create_task(background_task("A", 2))
    task2 = asyncio.create_task(background_task("B", 1))

    # Do other work while tasks run
    await asyncio.sleep(0.5)

    result1 = await task1
    result2 = await task2
```

### Pattern 4: Error Handling
```python
async def process_items(items: list[int]):
    tasks = [safe_operation(i) for i in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    return successful, failed
```

### Pattern 5: Timeout
```python
async def with_timeout():
    try:
        result = await asyncio.wait_for(slow_operation(5), timeout=2.0)
    except asyncio.TimeoutError:
        print("Operation timed out")
```

## Advanced Patterns

### Async Context Manager
```python
class AsyncDBConnection:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.connection = None

    async def __aenter__(self):
        self.connection = await connect(self.dsn)
        return self.connection

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.connection.close()

async with AsyncDBConnection("postgresql://localhost") as conn:
    result = await conn.fetch("SELECT * FROM users")
```

### Async Generator
```python
async def fetch_pages(url: str, max_pages: int) -> AsyncIterator[dict]:
    for page in range(1, max_pages + 1):
        response = await fetch(f"{url}?page={page}")
        yield response

async for page_data in fetch_pages("https://api.example.com", 10):
    process(page_data)
```

### Producer-Consumer
```python
async def producer(queue: asyncio.Queue, producer_id: int):
    for i in range(10):
        await queue.put(f"Item-{producer_id}-{i}")
    await queue.put(None)  # Signal done

async def consumer(queue: asyncio.Queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        await process(item)
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=10)
    producers = [asyncio.create_task(producer(queue, i)) for i in range(2)]
    consumers = [asyncio.create_task(consumer(queue)) for i in range(3)]

    await asyncio.gather(*producers)
    await queue.join()
    for c in consumers:
        c.cancel()
```

### Rate Limiting with Semaphore
```python
async def rate_limited_requests(urls: list[str], max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_limit(url: str):
        async with semaphore:
            return await fetch(url)

    return await asyncio.gather(*[fetch_with_limit(url) for url in urls])
```

### Async Lock
```python
class AsyncCounter:
    def __init__(self):
        self.value = 0
        self.lock = asyncio.Lock()

    async def increment(self):
        async with self.lock:
            current = self.value
            await asyncio.sleep(0.01)
            self.value = current + 1
```

## Real-World Applications

### Web Scraping with aiohttp
```python
import aiohttp

async def scrape_urls(urls: list[str]) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        async def fetch_url(url: str):
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                return {"url": url, "status": resp.status, "text": await resp.text()}

        return await asyncio.gather(*[fetch_url(url) for url in urls])
```

### Concurrent Database Queries
```python
async def get_user_data(db, user_id: int) -> dict:
    user, orders, profile = await asyncio.gather(
        db.fetch_one(f"SELECT * FROM users WHERE id = {user_id}"),
        db.execute(f"SELECT * FROM orders WHERE user_id = {user_id}"),
        db.fetch_one(f"SELECT * FROM profiles WHERE user_id = {user_id}")
    )
    return {"user": user, "orders": orders, "profile": profile}
```

### Connection Pool
```python
async def with_connection_pool():
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [session.get(f"https://api.example.com/{i}") for i in range(50)]
        return await asyncio.gather(*tasks)
```

### Batch Processing
```python
async def batch_process(items: list[str], batch_size: int = 10):
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        await asyncio.gather(*[process_item(item) for item in batch])
```

### Run Blocking in Executor
```python
import concurrent.futures

async def run_blocking(data):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, blocking_operation, data)
```

## Common Pitfalls

| Pitfall | Wrong | Correct |
|---------|-------|---------|
| Forgetting await | `result = async_func()` | `result = await async_func()` |
| Blocking event loop | `time.sleep(1)` | `await asyncio.sleep(1)` |
| Not handling cancel | No cleanup | `except asyncio.CancelledError: cleanup(); raise` |
| Sync calling async | `result = await func()` in sync | `asyncio.run(func())` |

## Testing Async Code

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await fetch_data("https://api.example.com")
    assert result is not None

@pytest.mark.asyncio
async def test_timeout():
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_operation(5), timeout=1.0)
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Entry point | Use `asyncio.run()` (Python 3.7+) |
| Concurrency | Use `gather()` for multiple tasks |
| Rate limiting | Use `asyncio.Semaphore` |
| Connection reuse | Pool connections with `TCPConnector` |
| Error handling | Use `return_exceptions=True` in gather |
| Timeouts | Always use `wait_for()` for external calls |
| Blocking ops | Use `run_in_executor()` for CPU-bound work |
| Cancellation | Handle `CancelledError`, cleanup, re-raise |

## Async Libraries

| Library | Purpose |
|---------|---------|
| aiohttp | HTTP client/server |
| FastAPI | Async web framework |
| asyncpg | PostgreSQL driver |
| motor | MongoDB driver |
| aiofiles | File I/O |
| aiocache | Caching |

## Checklist

- [ ] Use `asyncio.run()` for entry point
- [ ] Always `await` coroutines
- [ ] Use `gather()` for concurrent execution
- [ ] Implement timeouts for external calls
- [ ] Pool connections for efficiency
- [ ] Handle task cancellation properly
- [ ] Use semaphores for rate limiting
- [ ] Test with `pytest-asyncio`
- [ ] Never block the event loop
