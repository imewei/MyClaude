# Rust Async Programming Patterns

Comprehensive guide to async/await patterns with Tokio, production-ready async patterns, error handling, and concurrency best practices.

---

## Tokio Runtime Basics

### Runtime Initialization

```rust
// Single-threaded runtime
#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Async code here
}

// Multi-threaded runtime (default)
#[tokio::main]
async fn main() {
    // Async code here
}

// Custom runtime
use tokio::runtime::Runtime;

fn main() {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        // Async code here
    });
}

// Custom runtime with configuration
let rt = tokio::runtime::Builder::new_multi_thread()
    .worker_threads(4)
    .thread_name("my-worker")
    .thread_stack_size(3 * 1024 * 1024)
    .enable_all()
    .build()
    .unwrap();
```

### Cargo.toml Configuration

```toml
[dependencies]
tokio = { version = "1.36", features = ["full"] }

# Or minimal features
tokio = { version = "1.36", features = [
    "rt-multi-thread",  # Multi-threaded runtime
    "macros",           # #[tokio::main] and #[tokio::test]
    "net",              # TCP/UDP
    "io-util",          # AsyncRead/AsyncWrite utilities
    "time",             # Sleep, interval, timeout
    "sync",             # Channels, mutex, semaphore
] }
```

---

## Async Functions and Await

### Basic Async Functions

```rust
// Async function
async fn fetch_data(url: &str) -> Result<String, Error> {
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}

// Using async fn
#[tokio::main]
async fn main() -> Result<(), Error> {
    let data = fetch_data("https://example.com").await?;
    println!("Data: {}", data);
    Ok(())
}

// Async closure
let future = async {
    // Async code
    42
};

let result = future.await;
```

### Async Traits (Rust 1.75+)

```rust
trait AsyncProcessor {
    async fn process(&self, data: &str) -> Result<String, Error>;
}

struct DataProcessor;

impl AsyncProcessor for DataProcessor {
    async fn process(&self, data: &str) -> Result<String, Error> {
        // Async implementation
        Ok(data.to_uppercase())
    }
}
```

---

## Spawning Tasks

### tokio::spawn

```rust
use tokio::task;

#[tokio::main]
async fn main() {
    // Spawn task
    let handle = task::spawn(async {
        // Task code
        42
    });

    // Wait for result
    let result = handle.await.unwrap();
    println!("Result: {}", result);
}

// Spawn multiple tasks
let handles: Vec<_> = (0..10)
    .map(|i| {
        task::spawn(async move {
            process_item(i).await
        })
    })
    .collect();

// Wait for all
for handle in handles {
    handle.await.unwrap();
}
```

### spawn_blocking for CPU-Intensive Work

```rust
use tokio::task;

let result = task::spawn_blocking(|| {
    // CPU-intensive synchronous code
    heavy_computation()
}).await.unwrap();
```

---

## Concurrency Patterns

### join! - Run Concurrently

```rust
use tokio::join;

let (result1, result2, result3) = join!(
    async_operation_1(),
    async_operation_2(),
    async_operation_3(),
);

// All operations run concurrently
// Waits for all to complete
```

### try_join! - Early Exit on Error

```rust
use tokio::try_join;

let (data1, data2) = try_join!(
    fetch_data("url1"),
    fetch_data("url2"),
)?;

// Stops if any operation fails
```

### select! - Race Multiple Futures

```rust
use tokio::select;

select! {
    result = operation1() => {
        println!("Operation 1 completed first: {:?}", result);
    }
    result = operation2() => {
        println!("Operation 2 completed first: {:?}", result);
    }
    _ = tokio::time::sleep(Duration::from_secs(5)) => {
        println!("Timeout!");
    }
}
```

---

## Channels and Message Passing

### mpsc (Multiple Producer, Single Consumer)

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<String>(100);

    // Spawn sender task
    tokio::spawn(async move {
        for i in 0..10 {
            tx.send(format!("Message {}", i)).await.unwrap();
        }
    });

    // Receive messages
    while let Some(msg) = rx.recv().await {
        println!("Received: {}", msg);
    }
}

// Multiple senders
let tx1 = tx.clone();
let tx2 = tx.clone();

tokio::spawn(async move {
    tx1.send("From sender 1").await.unwrap();
});

tokio::spawn(async move {
    tx2.send("From sender 2").await.unwrap();
});
```

### broadcast - Multiple Consumers

```rust
use tokio::sync::broadcast;

let (tx, mut rx1) = broadcast::channel(100);
let mut rx2 = tx.subscribe();

tokio::spawn(async move {
    while let Ok(msg) = rx1.recv().await {
        println!("Receiver 1: {}", msg);
    }
});

tokio::spawn(async move {
    while let Ok(msg) = rx2.recv().await {
        println!("Receiver 2: {}", msg);
    }
});

tx.send("Broadcast message").unwrap();
```

### oneshot - Single Value

```rust
use tokio::sync::oneshot;

let (tx, rx) = oneshot::channel();

tokio::spawn(async move {
    let result = expensive_computation().await;
    tx.send(result).unwrap();
});

let result = rx.await.unwrap();
```

---

## Synchronization Primitives

### Mutex

```rust
use tokio::sync::Mutex;
use std::sync::Arc;

let counter = Arc::new(Mutex::new(0));

let handles: Vec<_> = (0..10)
    .map(|_| {
        let counter = Arc::clone(&counter);
        tokio::spawn(async move {
            let mut num = counter.lock().await;
            *num += 1;
        })
    })
    .collect();

for handle in handles {
    handle.await.unwrap();
}

let final_count = *counter.lock().await;
println!("Count: {}", final_count);
```

### RwLock

```rust
use tokio::sync::RwLock;
use std::sync::Arc;

let data = Arc::new(RwLock::new(Vec::new()));

// Multiple readers
for i in 0..5 {
    let data = Arc::clone(&data);
    tokio::spawn(async move {
        let vec = data.read().await;
        println!("Reader {}: {:?}", i, *vec);
    });
}

// Single writer
let data_writer = Arc::clone(&data);
tokio::spawn(async move {
    let mut vec = data_writer.write().await;
    vec.push(42);
});
```

### Semaphore - Rate Limiting

```rust
use tokio::sync::Semaphore;
use std::sync::Arc;

let semaphore = Arc::new(Semaphore::new(3)); // Max 3 concurrent

let handles: Vec<_> = (0..10)
    .map(|i| {
        let semaphore = Arc::clone(&semaphore);
        tokio::spawn(async move {
            let _permit = semaphore.acquire().await.unwrap();
            println!("Task {} running", i);
            tokio::time::sleep(Duration::from_secs(1)).await;
            // Permit auto-released on drop
        })
    })
    .collect();

for handle in handles {
    handle.await.unwrap();
}
```

---

## Timeouts and Intervals

### timeout

```rust
use tokio::time::{timeout, Duration};

match timeout(Duration::from_secs(5), long_operation()).await {
    Ok(result) => println!("Completed: {:?}", result),
    Err(_) => println!("Timeout!"),
}
```

### sleep

```rust
use tokio::time::{sleep, Duration};

sleep(Duration::from_secs(1)).await;
```

### interval

```rust
use tokio::time::{interval, Duration};

let mut interval = interval(Duration::from_secs(1));

loop {
    interval.tick().await;
    println!("Tick!");
}
```

---

## Error Handling Patterns

### Result Propagation

```rust
async fn fetch_and_process(url: &str) -> Result<Data, Error> {
    let response = reqwest::get(url).await?;
    let text = response.text().await?;
    let data: Data = serde_json::from_str(&text)?;
    Ok(data)
}
```

### Custom Error Types

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Parse error: {0}")]
    Parse(#[from] serde_json::Error),

    #[error("Timeout")]
    Timeout,

    #[error("Not found: {0}")]
    NotFound(String),
}

async fn fetch_with_timeout(url: &str) -> Result<Data, AppError> {
    let future = reqwest::get(url);

    match tokio::time::timeout(Duration::from_secs(5), future).await {
        Ok(Ok(response)) => {
            let data = response.json().await?;
            Ok(data)
        }
        Ok(Err(e)) => Err(AppError::Network(e)),
        Err(_) => Err(AppError::Timeout),
    }
}
```

---

## Production Patterns

### Graceful Shutdown

```rust
use tokio::signal;
use tokio::sync::broadcast;

#[tokio::main]
async fn main() {
    let (shutdown_tx, _) = broadcast::channel(1);

    // Spawn workers
    for i in 0..5 {
        let mut shutdown_rx = shutdown_tx.subscribe();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        println!("Worker {} shutting down", i);
                        break;
                    }
                    _ = do_work() => {}
                }
            }
        });
    }

    // Wait for shutdown signal
    shutdown_signal().await;
    println!("Sending shutdown signal");
    shutdown_tx.send(()).unwrap();

    // Wait for workers to finish
    tokio::time::sleep(Duration::from_secs(1)).await;
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
```

### Worker Pool

```rust
use tokio::sync::mpsc;

struct WorkerPool {
    tx: mpsc::Sender<Job>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl WorkerPool {
    fn new(size: usize) -> Self {
        let (tx, mut rx) = mpsc::channel::<Job>(100);

        for _ in 0..size {
            let mut rx = rx.clone();
            tokio::spawn(async move {
                while let Some(job) = rx.recv().await {
                    job();
                }
            });
        }

        WorkerPool { tx }
    }

    async fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.tx.send(job).await.unwrap();
    }
}
```

### Connection Pool

```rust
use deadpool_postgres::{Config, Runtime};
use tokio_postgres::NoTls;

let mut cfg = Config::new();
cfg.host = Some("localhost".to_string());
cfg.dbname = Some("mydb".to_string());
cfg.pool = Some(deadpool_postgres::PoolConfig::new(10));

let pool = cfg.create_pool(Some(Runtime::Tokio1), NoTls).unwrap();

// Get connection from pool
let client = pool.get().await?;
let rows = client.query("SELECT * FROM users", &[]).await?;
```

### Rate Limiter

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{Duration, interval};

struct RateLimiter {
    semaphore: Arc<Semaphore>,
}

impl RateLimiter {
    fn new(requests_per_second: usize) -> Self {
        let semaphore = Arc::new(Semaphore::new(requests_per_second));
        let sem_clone = Arc::clone(&semaphore);

        // Refill permits
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                let current = sem_clone.available_permits();
                if current < requests_per_second {
                    sem_clone.add_permits(requests_per_second - current);
                }
            }
        });

        RateLimiter { semaphore }
    }

    async fn acquire(&self) {
        self.semaphore.acquire().await.unwrap().forget();
    }
}
```

---

## Testing Async Code

### Basic Async Test

```rust
#[tokio::test]
async fn test_async_function() {
    let result = async_function().await;
    assert_eq!(result, expected);
}

// With timeout
#[tokio::test(flavor = "multi_thread")]
async fn test_with_timeout() {
    tokio::time::timeout(
        Duration::from_secs(5),
        async_operation()
    ).await.unwrap();
}
```

### Mocking with tokio-test

```rust
use tokio_test::{assert_ok, assert_err};

#[tokio::test]
async fn test_with_assertions() {
    let result = async_function().await;
    assert_ok!(result);
}
```

---

## Common Pitfalls

### 1. Blocking in Async Context

**Bad:**
```rust
async fn bad() {
    std::thread::sleep(Duration::from_secs(1)); // Blocks the executor!
}
```

**Good:**
```rust
async fn good() {
    tokio::time::sleep(Duration::from_secs(1)).await; // Non-blocking
}

// Or for CPU-intensive work
let result = tokio::task::spawn_blocking(|| {
    heavy_computation()
}).await.unwrap();
```

### 2. Holding Mutex Across Await

**Bad:**
```rust
use tokio::sync::Mutex;

let mutex = Mutex::new(data);
let guard = mutex.lock().await;
some_async_fn().await; // Guard held across await!
```

**Good:**
```rust
let result = {
    let guard = mutex.lock().await;
    guard.clone() // Clone data and drop guard
};
some_async_fn().await;
```

### 3. Not Handling Spawned Task Errors

**Bad:**
```rust
tokio::spawn(async {
    might_fail().await.unwrap(); // Panic in spawned task!
});
```

**Good:**
```rust
let handle = tokio::spawn(async {
    might_fail().await
});

match handle.await {
    Ok(Ok(result)) => println!("Success: {:?}", result),
    Ok(Err(e)) => eprintln!("Task error: {}", e),
    Err(e) => eprintln!("Join error: {}", e),
}
```

---

## Summary: Async Best Practices

1. **Runtime**: Use `#[tokio::main]` for convenience
2. **Spawning**: Use `tokio::spawn` for concurrent tasks
3. **Blocking**: Use `spawn_blocking` for CPU-intensive work
4. **Concurrency**: Use `join!`, `try_join!`, `select!` for concurrent operations
5. **Channels**: Use mpsc for message passing, broadcast for pub-sub
6. **Synchronization**: Prefer message passing over shared state
7. **Timeouts**: Always add timeouts to external I/O
8. **Graceful Shutdown**: Implement shutdown signals
9. **Error Handling**: Use custom error types, propagate errors
10. **Testing**: Use `#[tokio::test]` with timeouts
