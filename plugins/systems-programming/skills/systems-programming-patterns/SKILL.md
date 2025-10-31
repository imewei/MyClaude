---
name: systems-programming-patterns
description: Master systems programming patterns and techniques for memory management, concurrency, performance optimization, and debugging across C, C++, Rust, and Go. Use when writing or editing low-level systems code (*.c, *.cpp, *.rs, *.go files), implementing memory allocators or pools, designing concurrent or lock-free data structures, optimizing performance-critical code paths, debugging memory issues (leaks, corruption, race conditions), profiling applications with perf/valgrind/flamegraph, implementing RAII patterns or smart pointers, working with atomics and memory ordering, building thread pools or work-stealing schedulers, optimizing cache performance with SoA layouts, implementing SIMD vectorization, designing zero-copy algorithms, debugging with AddressSanitizer/ThreadSanitizer/Miri, benchmarking with Google Benchmark/Criterion/Go testing, preventing common pitfalls (use-after-free, double-free, data races, deadlocks, false sharing, ABA problems), or architecting high-performance systems with minimal latency and maximum throughput. Includes battle-tested implementations for memory pool allocation, arena allocators, lock-free queues and stacks, RCU patterns, reader-writer locks, and comprehensive debugging workflows.
---

# Systems Programming Patterns

## Purpose

This skill provides battle-tested patterns, techniques, and workflows for systems programming across C, C++, Rust, and Go. It covers the fundamental challenges of systems development: memory management, concurrent programming, performance optimization, and debugging complex issues. The skill consolidates years of systems programming wisdom into actionable patterns that prevent common pitfalls and enable high-performance, reliable software.

## When to use this skill

- Writing or editing low-level systems code in C (*.c, *.h files), C++ (*.cpp, *.cc, *.cxx, *.hpp files), Rust (*.rs files), or Go (*.go files)
- Implementing custom memory management strategies: memory pool allocators for fixed-size objects, arena allocators for bulk allocation, RAII wrappers, or smart pointers (std::unique_ptr, std::shared_ptr, Box, Rc, Arc)
- Designing and implementing concurrent or lock-free data structures: SPSC/MPMC queues, lock-free stacks, atomic operations with CAS, hazard pointers, epoch-based reclamation
- Optimizing performance-critical code paths in hot loops, tight inner loops, real-time systems, game engines, or high-frequency trading systems
- Debugging memory issues: memory leaks detected by Valgrind, use-after-free caught by AddressSanitizer, double-free errors, buffer overflows, uninitialized reads, or memory corruption
- Debugging concurrency issues: data races detected by ThreadSanitizer, deadlocks, race conditions, or synchronization bugs
- Profiling applications with perf (Linux), flamegraph generation, Valgrind's Massif for memory profiling, heaptrack, cargo-flamegraph (Rust), or pprof (Go)
- Implementing thread pools, work-stealing schedulers (like rayon in Rust), or goroutine pools for concurrent workload distribution
- Working with atomic operations: compare-and-swap (CAS), fetch-and-add, memory ordering (relaxed, acquire, release, seq_cst), or memory barriers
- Optimizing cache performance: converting Array of Structures (AoS) to Structure of Arrays (SoA), cache line padding to prevent false sharing, or prefetching
- Implementing SIMD vectorization: auto-vectorization with compiler flags (-O3 -march=native), explicit SIMD with intrinsics (AVX2, AVX-512), or portable SIMD in Rust
- Designing zero-copy algorithms: move semantics in C++, ownership transfer in Rust, memory-mapped files (mmap), splice/sendfile system calls, or DMA transfers
- Benchmarking code with Google Benchmark (C++), Criterion (Rust), or Go's testing package to measure performance improvements
- Preventing common systems programming pitfalls: use-after-free, double-free, memory leaks, data races, deadlocks, false sharing in multi-threaded code, ABA problems in lock-free structures
- Architecting high-performance systems: network servers, databases, game engines, operating system components, embedded systems, real-time systems, or low-latency financial systems
- Choosing between memory management patterns based on use case: pools for frequent fixed-size allocations, arenas for batch operations, RAII for automatic cleanup
- Selecting appropriate concurrency primitives: mutexes vs spinlocks vs lock-free, reader-writer locks for read-heavy workloads, RCU for wait-free reads
- Implementing Read-Copy-Update (RCU) patterns for read-heavy data structures with occasional writes
- Working with cross-platform systems code that needs to run on Linux, Windows, macOS, or embedded platforms
- Debugging undefined behavior in C/C++ or unsafe code in Rust using sanitizers (ASan, TSan, UBSan, MSan) or Miri interpreter

## Core Patterns and Techniques

### Memory Management Patterns

#### Pattern 1: Memory Pool Allocation

**Purpose**: Reduce allocation overhead and fragmentation for fixed-size objects

**When to use**:
- Allocating/deallocating many small objects frequently
- Latency-sensitive code paths where malloc is too slow
- Embedded systems with limited dynamic memory
- Real-time systems requiring deterministic allocation

**Implementation approach**:

```c
// C: Fixed-size memory pool
typedef struct {
    void* free_list;
    size_t object_size;
    size_t pool_size;
    char* memory;
} MemoryPool;

MemoryPool* pool_create(size_t object_size, size_t capacity);
void* pool_alloc(MemoryPool* pool);
void pool_free(MemoryPool* pool, void* ptr);
void pool_destroy(MemoryPool* pool);
```

```cpp
// C++: RAII memory pool with type safety
template<typename T, size_t Capacity>
class ObjectPool {
    std::array<std::aligned_storage_t<sizeof(T), alignof(T)>, Capacity> storage;
    std::vector<T*> free_list;
public:
    template<typename... Args>
    T* allocate(Args&&... args);
    void deallocate(T* ptr);
};
```

```rust
// Rust: Type-safe pool with lifetime tracking
pub struct Pool<T> {
    objects: Vec<Option<T>>,
    free_indices: Vec<usize>,
}

impl<T> Pool<T> {
    pub fn with_capacity(capacity: usize) -> Self;
    pub fn allocate(&mut self, value: T) -> PoolHandle;
    pub fn get(&self, handle: PoolHandle) -> Option<&T>;
}
```

**Trade-offs**:
- ✅ Fast, predictable allocation/deallocation
- ✅ Reduced memory fragmentation
- ✅ Cache-friendly sequential layout
- ❌ Fixed object size (one pool per type)
- ❌ Memory overhead if pool is underutilized
- ❌ Potential waste if capacity is too large

**Common pitfalls**:
- Forgetting to return objects to pool (memory leak)
- Using objects after returning them (use-after-free)
- Pool exhaustion without graceful degradation
- Not aligning pool entries correctly (performance/correctness)

**Reference**: See `references/memory-pools.md` for detailed implementations and benchmarks

#### Pattern 2: Arena Allocator

**Purpose**: Fast bulk allocation with single deallocation point

**When to use**:
- Request-scoped allocations (HTTP requests, transactions)
- Parsing and AST construction (entire tree deallocated together)
- Game frame allocations (reset every frame)
- Temporary computations with known lifetime

**Implementation characteristics**:
- Bump pointer allocation (increment pointer, return old value)
- No individual deallocation - entire arena freed at once
- Extremely fast allocation (just pointer arithmetic)
- Excellent cache locality (sequential allocations)

**Language-specific implementations**:

**C**: Manual arena with linear bump allocator
```c
typedef struct {
    char* buffer;
    size_t capacity;
    size_t offset;
} Arena;

void* arena_alloc(Arena* arena, size_t size, size_t align);
void arena_reset(Arena* arena);  // Reset offset to 0
void arena_destroy(Arena* arena);
```

**C++**: RAII arena with automatic cleanup
```cpp
class Arena {
    std::vector<std::unique_ptr<char[]>> blocks;
    char* current;
    size_t remaining;
public:
    template<typename T, typename... Args>
    T* create(Args&&... args);  // Placement new in arena
    void reset();  // Call destructors, reset offset
};
```

**Rust**: Type-safe arena with borrow checker integration
```rust
pub struct Arena {
    chunks: Vec<Vec<u8>>,
    current_chunk: usize,
    offset: usize,
}

impl Arena {
    pub fn alloc<T>(&mut self, value: T) -> &mut T;
    pub fn alloc_slice<T>(&mut self, slice: &[T]) -> &mut [T];
}
```

**Go**: Arena pattern (less common due to GC)
```go
type Arena struct {
    buffer []byte
    offset int
}

func (a *Arena) Alloc(size int) []byte {
    // Bump allocate from buffer
}
```

**Trade-offs**:
- ✅ Extremely fast allocation (single pointer bump)
- ✅ Perfect for known-lifetime batch operations
- ✅ Automatic cleanup of entire scope
- ❌ No individual deallocation
- ❌ Memory held until arena is destroyed
- ❌ Potential waste if allocations vary widely in lifetime

#### Pattern 3: RAII and Smart Pointers

**Purpose**: Automatic resource management through scope-based cleanup

**C++**: Fundamental ownership patterns
```cpp
// Unique ownership
std::unique_ptr<Resource> resource = std::make_unique<Resource>();

// Shared ownership
std::shared_ptr<Resource> shared = std::make_shared<Resource>();

// Weak reference (no ownership)
std::weak_ptr<Resource> weak = shared;

// Custom deleters
auto file = std::unique_ptr<FILE, decltype(&fclose)>(
    fopen("data.txt", "r"), &fclose
);
```

**Rust**: Ownership and borrowing
```rust
// Owned value (exclusive ownership)
let owned = Box::new(Resource::new());

// Reference counted (shared ownership)
let shared = Rc::new(Resource::new());

// Thread-safe shared ownership
let atomic_shared = Arc::new(Resource::new());

// Interior mutability
let mutable = RefCell::new(Resource::new());
```

**Reference**: See `references/raii-patterns.md` for comprehensive examples

### Concurrent Programming Patterns

#### Pattern 4: Lock-Free Data Structures

**Purpose**: Enable concurrent access without blocking threads

**When to use**:
- High-contention scenarios where locks cause bottlenecks
- Real-time systems where blocking is unacceptable
- Producer-consumer queues with high throughput requirements
- Read-heavy data structures (RCU, hazard pointers)

**Atomic primitives** (all languages):
- Compare-and-swap (CAS): atomic conditional update
- Fetch-and-add: atomic increment/decrement
- Load/Store with memory ordering: acquire/release/seq_cst

**Common lock-free structures**:

1. **Lock-Free Queue** (Michael-Scott queue)
   - Single-producer single-consumer (SPSC): fastest
   - Multi-producer multi-consumer (MPMC): most complex
   - Bounded vs unbounded variants

2. **Lock-Free Stack**
   - Simple CAS-based push/pop
   - ABA problem mitigation (tagged pointers, hazard pointers)

3. **Read-Copy-Update (RCU)**
   - Read-side wait-free (no synchronization)
   - Writers create new versions
   - Deferred reclamation of old versions

**C++ example (lock-free stack)**:
```cpp
template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
    };
    std::atomic<Node*> head{nullptr};

public:
    void push(T value) {
        Node* new_node = new Node{std::move(value), head.load()};
        while (!head.compare_exchange_weak(new_node->next, new_node));
    }

    bool pop(T& result) {
        Node* old_head = head.load();
        while (old_head &&
               !head.compare_exchange_weak(old_head, old_head->next));
        if (!old_head) return false;
        result = std::move(old_head->data);
        delete old_head;  // ABA problem! Use hazard pointers in production
        return true;
    }
};
```

**Rust example (crossbeam channels)**:
```rust
use crossbeam::channel::{bounded, unbounded};

// Bounded channel (backpressure when full)
let (tx, rx) = bounded(100);

// Unbounded channel (grows indefinitely)
let (tx, rx) = unbounded();

// Lock-free MPMC
tx.send(value)?;
let value = rx.recv()?;
```

**Go example (channels)**:
```go
// Buffered channel (bounded)
ch := make(chan int, 100)

// Unbuffered channel (synchronous)
ch := make(chan int)

// Select for multiplexing
select {
case val := <-ch1:
    // Received from ch1
case ch2 <- val:
    // Sent to ch2
case <-time.After(timeout):
    // Timeout
}
```

**Memory ordering considerations**:
- **Relaxed**: No ordering guarantees (fastest)
- **Acquire/Release**: Synchronizes with matching release/acquire
- **SeqCst**: Total order across all threads (safest, slowest)

**Common pitfalls**:
- **ABA problem**: Value changes from A→B→A, CAS succeeds incorrectly
  - Solution: Tagged pointers, hazard pointers, epoch-based reclamation
- **Memory reclamation**: When can we free memory that might be accessed?
  - Solution: Hazard pointers, epoch-based reclamation (crossbeam-epoch)
- **False sharing**: Multiple threads modify adjacent cache lines
  - Solution: Pad structures to cache line size (64 bytes typically)

**Reference**: See `references/lock-free-patterns.md` for detailed implementations

#### Pattern 5: Thread Pools and Work Stealing

**Purpose**: Efficiently distribute work across threads

**Patterns**:

1. **Fixed Thread Pool**
   - Pre-allocate N worker threads
   - Work queue feeding all threads
   - Good for consistent load

2. **Work Stealing Pool**
   - Each thread has own queue
   - Idle threads steal from busy threads
   - Better load balancing for uneven work

**C++ example (simple thread pool)**:
```cpp
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;

public:
    ThreadPool(size_t threads);

    template<class F>
    auto enqueue(F&& f) -> std::future<decltype(f())>;

    ~ThreadPool();
};
```

**Rust example (rayon work stealing)**:
```rust
use rayon::prelude::*;

// Parallel iterator (work stealing)
let sum: i32 = (0..1000).into_par_iter()
    .map(|x| expensive_computation(x))
    .sum();

// Thread pool with custom size
let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(8)
    .build()?;

pool.install(|| {
    // Work runs on this pool
});
```

**Go example (goroutine pool)**:
```go
type WorkerPool struct {
    tasks chan func()
    wg    sync.WaitGroup
}

func NewWorkerPool(workers int) *WorkerPool {
    p := &WorkerPool{
        tasks: make(chan func(), 100),
    }
    for i := 0; i < workers; i++ {
        p.wg.Add(1)
        go p.worker()
    }
    return p
}

func (p *WorkerPool) Submit(task func()) {
    p.tasks <- task
}
```

#### Pattern 6: Reader-Writer Locks and RCU

**Purpose**: Optimize for read-heavy workloads

**Standard RWLock**:
- Multiple concurrent readers OR one writer
- Writers must wait for all readers to finish
- Readers block while writer holds lock

**C++**:
```cpp
std::shared_mutex mutex;

// Reader
std::shared_lock lock(mutex);
// Multiple readers can hold shared_lock

// Writer
std::unique_lock lock(mutex);
// Exclusive access
```

**Rust**:
```rust
use std::sync::RwLock;

let lock = RwLock::new(data);

// Multiple readers
let read_guard = lock.read().unwrap();

// Exclusive writer
let write_guard = lock.write().unwrap();
```

**Go**:
```go
var mu sync.RWMutex

// Reader
mu.RLock()
defer mu.RUnlock()

// Writer
mu.Lock()
defer mu.Unlock()
```

**RCU (Read-Copy-Update)**:
- Readers never block (wait-free reads)
- Writers create new versions
- Old versions reclaimed when no readers reference them
- Best for rarely-updated, frequently-read data

**Reference**: See `references/rcu-patterns.md` for RCU implementation details

### Performance Optimization Patterns

#### Pattern 7: Cache-Aware Programming

**Principles**:

1. **Data Locality**: Keep related data together
2. **Sequential Access**: Prefer arrays over linked lists
3. **Cache Line Alignment**: Avoid false sharing
4. **Prefetching**: Hint processor about future access

**Structure of Arrays (SoA) vs Array of Structures (AoS)**:

```cpp
// AoS: Poor cache utilization when accessing one field
struct Particle {
    float x, y, z;     // position
    float vx, vy, vz;  // velocity
    float mass;
};
std::vector<Particle> particles;

// Process velocities (loads unnecessary position/mass)
for (auto& p : particles) {
    p.vx += dt * force_x;
    p.vy += dt * force_y;
    p.vz += dt * force_z;
}
```

```cpp
// SoA: Excellent cache utilization
struct Particles {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
    std::vector<float> mass;
};

// Only load velocity data
for (size_t i = 0; i < n; ++i) {
    particles.vx[i] += dt * force_x;
    particles.vy[i] += dt * force_y;
    particles.vz[i] += dt * force_z;
}
```

**Cache line padding to prevent false sharing**:

```cpp
struct alignas(64) Counter {
    std::atomic<int> value;
    char padding[64 - sizeof(std::atomic<int>)];
};

// Each counter on separate cache line
std::array<Counter, NUM_THREADS> per_thread_counters;
```

**Rust example**:
```rust
#[repr(align(64))]
struct CacheLineAligned<T> {
    value: T,
}

// Prevent false sharing
struct PerThreadData {
    counter: CacheLineAligned<AtomicU64>,
}
```

#### Pattern 8: SIMD and Vectorization

**Purpose**: Process multiple data elements in parallel using CPU vector instructions

**When to use**:
- Processing arrays of numbers (image processing, physics)
- Embarrassingly parallel element-wise operations
- Algorithms operating on contiguous memory

**Auto-vectorization (compiler)**:
```cpp
// Simple loop that compilers can auto-vectorize
void add_arrays(float* a, float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
// Compile with: -O3 -march=native
```

**Explicit SIMD (C++ with intrinsics)**:
```cpp
#include <immintrin.h>  // AVX2

void add_arrays_simd(float* a, float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&c[i], vc);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

**Rust (portable SIMD)**:
```rust
use std::simd::*;

fn add_arrays(a: &[f32], b: &[f32], c: &mut [f32]) {
    let (a_chunks, a_remainder) = a.as_chunks::<8>();
    let (b_chunks, b_remainder) = b.as_chunks::<8>();
    let (c_chunks, c_remainder) = c.as_chunks_mut::<8>();

    for ((a, b), c) in a_chunks.iter()
        .zip(b_chunks)
        .zip(c_chunks) {
        let va = f32x8::from_array(*a);
        let vb = f32x8::from_array(*b);
        *c = (va + vb).to_array();
    }

    // Handle remainder
    for ((a, b), c) in a_remainder.iter()
        .zip(b_remainder)
        .zip(c_remainder) {
        *c = a + b;
    }
}
```

**Common pitfalls**:
- Assuming alignment (use unaligned loads or ensure alignment)
- Forgetting to handle remainder elements
- Not checking SIMD availability at runtime (use CPUID)
- Breaking auto-vectorization with complex control flow

#### Pattern 9: Zero-Copy and Move Semantics

**Purpose**: Avoid unnecessary data copies

**C++ move semantics**:
```cpp
class Buffer {
    char* data;
    size_t size;

public:
    // Move constructor (steal resources)
    Buffer(Buffer&& other) noexcept
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    // Move assignment
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};

// Usage
Buffer create_buffer() {
    return Buffer(1024);  // Move, not copy
}

Buffer buf = create_buffer();  // Move constructed
```

**Rust ownership and borrowing**:
```rust
// Ownership transfer (move)
let buf1 = vec![1, 2, 3];
let buf2 = buf1;  // buf1 is moved, can't use it anymore

// Borrowing (zero-copy access)
fn process(data: &[u8]) {
    // Read-only access, no copy
}

process(&buf2);  // Borrow, buf2 still usable

// Mutable borrow
fn modify(data: &mut Vec<u8>) {
    data.push(4);
}

modify(&mut buf2);
```

**Zero-copy I/O patterns**:
- Memory-mapped files (mmap)
- Splice/sendfile system calls
- Scatter-gather I/O (readv/writev)
- DMA transfers (kernel bypass)

### Debugging and Profiling

#### Pattern 10: Debugging Memory Issues

**Tools and techniques**:

1. **Valgrind (C/C++)**
   ```bash
   # Memory leaks
   valgrind --leak-check=full ./program

   # Use-after-free, double-free
   valgrind --tool=memcheck ./program

   # Uninitialized values
   valgrind --track-origins=yes ./program
   ```

2. **AddressSanitizer (ASan)**
   ```bash
   # Compile with ASan
   gcc -fsanitize=address -g program.c

   # Detects:
   # - Use-after-free
   # - Heap buffer overflow
   # - Stack buffer overflow
   # - Use-after-return
   # - Memory leaks
   ```

3. **ThreadSanitizer (TSan)**
   ```bash
   # Compile with TSan
   gcc -fsanitize=thread -g program.c

   # Detects data races
   ```

4. **Rust debugging**
   ```bash
   # Miri (interpreter for detecting UB)
   cargo +nightly miri test

   # Valgrind works with Rust too
   valgrind --leak-check=full ./target/debug/program
   ```

**Common memory bugs**:
- **Memory leaks**: Allocated but never freed
- **Use-after-free**: Accessing freed memory
- **Double free**: Freeing same memory twice
- **Buffer overflow**: Writing past allocation bounds
- **Uninitialized reads**: Reading before writing
- **Data races**: Concurrent unsynchronized access

#### Pattern 11: Performance Profiling

**CPU profiling**:

1. **perf (Linux)**
   ```bash
   # Sample CPU usage
   perf record -g ./program
   perf report

   # Hardware counters (cache misses, branch mispredictions)
   perf stat -e cache-misses,branch-misses ./program
   ```

2. **flamegraph**
   ```bash
   # Generate flamegraph
   perf record -g ./program
   perf script | ./flamegraph.pl > flame.svg
   ```

3. **cargo-flamegraph (Rust)**
   ```bash
   cargo flamegraph --bin myprogram
   ```

**Profiling workflow**:
1. **Measure**: Collect profile data
2. **Analyze**: Identify hotspots (functions consuming most time)
3. **Optimize**: Focus on top 3-5 hotspots
4. **Verify**: Re-profile to confirm improvement
5. **Iterate**: Repeat until performance goals met

**Memory profiling**:

1. **Massif (Valgrind)**
   ```bash
   valgrind --tool=massif ./program
   ms_print massif.out.<pid>
   ```

2. **Heaptrack (Linux)**
   ```bash
   heaptrack ./program
   heaptrack_gui heaptrack.program.<pid>.gz
   ```

3. **Rust memory profiling**
   ```bash
   # DHAT (dynamic heap analysis)
   valgrind --tool=dhat ./target/release/program
   ```

**Benchmarking**:

**C++ (Google Benchmark)**:
```cpp
#include <benchmark/benchmark.h>

static void BM_VectorPush(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<int> v;
        for (int i = 0; i < state.range(0); ++i) {
            v.push_back(i);
        }
    }
}
BENCHMARK(BM_VectorPush)->Range(8, 8<<10);
```

**Rust (Criterion)**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n-1) + fibonacci(n-2),
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
```

**Go (testing package)**:
```go
func BenchmarkFibonacci(b *testing.B) {
    for i := 0; i < b.N; i++ {
        fibonacci(20)
    }
}

// Run with: go test -bench=.
```

## Common Pitfalls and How to Avoid Them

### Memory Management Pitfalls

1. **Forgetting to free memory**
   - Use RAII (C++) or ownership (Rust)
   - Enable leak detection tools in CI
   - Regular Valgrind runs

2. **Use-after-free**
   - Nullify pointers after freeing (C)
   - Use smart pointers (C++)
   - Rust prevents this at compile time

3. **Double free**
   - Clear ownership semantics
   - Use smart pointers with single owner
   - Rust prevents this at compile time

### Concurrency Pitfalls

1. **Data races**
   - Always protect shared mutable state
   - Use ThreadSanitizer to detect races
   - Rust prevents data races at compile time

2. **Deadlock**
   - Always acquire locks in same order
   - Use lock hierarchies
   - Consider timeout-based locking
   - Use try_lock where appropriate

3. **False sharing**
   - Pad hot data to cache line boundaries
   - Use thread-local storage
   - Separate read-only and write-heavy data

4. **ABA problem in lock-free code**
   - Use tagged pointers
   - Use hazard pointers or epoch-based reclamation
   - Consider using proven libraries (crossbeam, etc.)

### Performance Pitfalls

1. **Premature optimization**
   - Profile before optimizing
   - Focus on algorithmic improvements first
   - Optimize based on data, not intuition

2. **Ignoring cache effects**
   - Prefer sequential access patterns
   - Use SoA for hot data
   - Align frequently-accessed data

3. **Unnecessary allocations**
   - Reuse buffers when possible
   - Use stack allocation for small objects
   - Consider object pools for frequently allocated types

## Reference Materials

The `references/` directory contains detailed documentation:

- `memory-pools.md`: Comprehensive memory pool implementations and benchmarks
- `lock-free-patterns.md`: Detailed lock-free data structure implementations
- `raii-patterns.md`: RAII and smart pointer patterns across languages
- `rcu-patterns.md`: Read-Copy-Update implementation and use cases
- `profiling-guide.md`: Step-by-step profiling workflows and tool usage
- `common-bugs.md`: Catalog of common systems programming bugs and fixes

## Workflow

When using this skill:

1. **Identify the challenge**: Memory management, concurrency, performance, or debugging
2. **Select appropriate pattern**: Use the pattern index above to find relevant techniques
3. **Review reference material**: Consult detailed docs in `references/` for implementation details
4. **Consider trade-offs**: Evaluate pros/cons for the specific use case
5. **Implement with safeguards**: Use appropriate tools (sanitizers, profilers) during development
6. **Verify correctness**: Run with debugging tools enabled
7. **Measure performance**: Profile before and after optimization

## Language-Specific Considerations

### C
- Manual memory management requires discipline
- No built-in concurrency primitives (use pthreads)
- Excellent performance but safety is developer's responsibility
- Use static analysis tools extensively (clang-tidy, scan-build)

### C++
- RAII enables automatic resource management
- Rich standard library (std::thread, std::atomic, smart pointers)
- Template metaprogramming for zero-cost abstractions
- Balance safety features with manual control

### Rust
- Ownership system prevents data races and memory bugs at compile time
- Zero-cost abstractions with safety guarantees
- Excellent async ecosystem (tokio, async-std)
- Steep learning curve but worth the investment

### Go
- Garbage collected (no manual memory management)
- Built-in concurrency with goroutines and channels
- Lower absolute performance than C/C++/Rust
- Excellent for network services and concurrent systems
- Less suitable for hard real-time or memory-constrained systems

Select language based on project requirements: safety vs control, performance vs productivity, team expertise, and ecosystem maturity.
