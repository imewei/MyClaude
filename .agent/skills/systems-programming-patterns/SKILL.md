---
name: systems-programming-patterns
version: "1.0.7"
description: Master systems programming patterns for memory management, concurrency, and performance optimization across C, C++, Rust, and Go. Use when implementing memory allocators, designing lock-free data structures, optimizing cache performance, debugging with sanitizers (ASan/TSan/Miri), or profiling with perf/flamegraph.
---

# Systems Programming Patterns

<!-- SECTION: MEMORY -->
## Memory Management Patterns

| Pattern | Use Case | Trade-offs |
|---------|----------|------------|
| Memory Pool | Fixed-size frequent allocations | Fast alloc, fixed size |
| Arena | Batch allocations, known lifetime | Very fast, no individual free |
| RAII/Smart Pointers | Automatic cleanup | Safe, slight overhead |
| Custom Allocator | Domain-specific needs | Maximum control |

### Memory Pool (Fixed-Size Objects)

```cpp
// C++: Type-safe pool
template<typename T, size_t Capacity>
class ObjectPool {
    std::array<std::aligned_storage_t<sizeof(T), alignof(T)>, Capacity> storage;
    std::vector<T*> free_list;
public:
    template<typename... Args>
    T* allocate(Args&&... args) {
        if (free_list.empty()) return nullptr;
        T* ptr = free_list.back();
        free_list.pop_back();
        return new(ptr) T(std::forward<Args>(args)...);
    }
    void deallocate(T* ptr) {
        ptr->~T();
        free_list.push_back(ptr);
    }
};
```

```rust
// Rust: Pool with handles
pub struct Pool<T> {
    objects: Vec<Option<T>>,
    free_indices: Vec<usize>,
}
impl<T> Pool<T> {
    pub fn allocate(&mut self, value: T) -> usize {
        if let Some(idx) = self.free_indices.pop() {
            self.objects[idx] = Some(value);
            idx
        } else {
            self.objects.push(Some(value));
            self.objects.len() - 1
        }
    }
}
```

### Arena Allocator (Bulk Allocation)

```cpp
// C++: Bump allocator
class Arena {
    std::vector<std::unique_ptr<char[]>> blocks;
    char* current; size_t remaining;
public:
    template<typename T, typename... Args>
    T* create(Args&&... args) {
        void* ptr = allocate(sizeof(T), alignof(T));
        return new(ptr) T(std::forward<Args>(args)...);
    }
    void reset();  // Reset all, call destructors
};
```

### Smart Pointers (C++)

```cpp
// Unique ownership
auto resource = std::make_unique<Resource>();

// Shared ownership
auto shared = std::make_shared<Resource>();
std::weak_ptr<Resource> weak = shared;

// Custom deleter
auto file = std::unique_ptr<FILE, decltype(&fclose)>(
    fopen("data.txt", "r"), &fclose);
```

### Rust Ownership

```rust
// Exclusive ownership
let owned = Box::new(Resource::new());

// Shared ownership
let shared = Rc::new(Resource::new());      // Single-threaded
let atomic = Arc::new(Resource::new());      // Thread-safe

// Interior mutability
let cell = RefCell::new(Resource::new());
let mutex = Mutex::new(Resource::new());
```
<!-- END_SECTION: MEMORY -->

---

<!-- SECTION: CONCURRENCY -->
## Concurrency Patterns

| Pattern | Use Case | Complexity |
|---------|----------|------------|
| Mutex | Shared mutable state | Low |
| RWLock | Read-heavy workloads | Low |
| Lock-Free Queue | High-throughput producer-consumer | High |
| RCU | Read-mostly, rare writes | High |
| Thread Pool | Task distribution | Medium |

### Lock-Free Stack (C++)

```cpp
template<typename T>
class LockFreeStack {
    struct Node { T data; Node* next; };
    std::atomic<Node*> head{nullptr};
public:
    void push(T value) {
        Node* new_node = new Node{std::move(value), head.load()};
        while (!head.compare_exchange_weak(new_node->next, new_node));
    }
    bool pop(T& result) {
        Node* old_head = head.load();
        while (old_head && !head.compare_exchange_weak(old_head, old_head->next));
        if (!old_head) return false;
        result = std::move(old_head->data);
        delete old_head;  // Use hazard pointers in production
        return true;
    }
};
```

### Reader-Writer Lock

```cpp
// C++
std::shared_mutex mutex;
{ std::shared_lock lock(mutex); /* read */ }
{ std::unique_lock lock(mutex); /* write */ }
```

```rust
// Rust
let lock = RwLock::new(data);
let read = lock.read().unwrap();
let write = lock.write().unwrap();
```

```go
// Go
var mu sync.RWMutex
mu.RLock(); defer mu.RUnlock()  // Read
mu.Lock(); defer mu.Unlock()     // Write
```

### Thread Pool (Rust with Rayon)

```rust
use rayon::prelude::*;

let sum: i32 = (0..1000).into_par_iter()
    .map(|x| expensive_computation(x))
    .sum();

let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(8)
    .build().unwrap();
pool.install(|| { /* work */ });
```

### Go Channels

```go
ch := make(chan int, 100)  // Buffered
ch <- value                 // Send
value := <-ch               // Receive

select {
case v := <-ch1: // Received
case ch2 <- v:   // Sent
case <-time.After(timeout): // Timeout
}
```
<!-- END_SECTION: CONCURRENCY -->

---

<!-- SECTION: MEMORY_ORDERING -->
## Memory Ordering

| Order | Guarantee | Performance |
|-------|-----------|-------------|
| Relaxed | None | Fastest |
| Acquire | Syncs with Release | Medium |
| Release | Syncs with Acquire | Medium |
| SeqCst | Total order | Slowest |

```cpp
// C++ atomics
std::atomic<int> counter;
counter.store(1, std::memory_order_release);
int val = counter.load(std::memory_order_acquire);
```
<!-- END_SECTION: MEMORY_ORDERING -->

---

<!-- SECTION: CACHE -->
## Cache Optimization

### Structure of Arrays (SoA)

```cpp
// AoS (poor cache for single-field access)
struct Particle { float x, y, z, vx, vy, vz, mass; };
std::vector<Particle> particles;

// SoA (excellent cache utilization)
struct Particles {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
    std::vector<float> mass;
};
```

### Prevent False Sharing

```cpp
struct alignas(64) Counter {
    std::atomic<int> value;
    char padding[64 - sizeof(std::atomic<int>)];
};
```

```rust
#[repr(align(64))]
struct CacheAligned<T>(T);
```
<!-- END_SECTION: CACHE -->

---

<!-- SECTION: SIMD -->
## SIMD Vectorization

```cpp
// Auto-vectorized (compile with -O3 -march=native)
void add_arrays(float* a, float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

// Explicit AVX2
#include <immintrin.h>
void add_simd(float* a, float* b, float* c, size_t n) {
    for (size_t i = 0; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&c[i], _mm256_add_ps(va, vb));
    }
}
```
<!-- END_SECTION: SIMD -->

---

<!-- SECTION: DEBUGGING -->
## Debugging Tools

| Tool | Purpose | Command |
|------|---------|---------|
| Valgrind | Memory leaks, UaF | `valgrind --leak-check=full ./prog` |
| ASan | Memory errors | `gcc -fsanitize=address` |
| TSan | Data races | `gcc -fsanitize=thread` |
| UBSan | Undefined behavior | `gcc -fsanitize=undefined` |
| Miri | Rust UB detection | `cargo +nightly miri test` |

### Common Memory Bugs

| Bug | Detection | Prevention |
|-----|-----------|------------|
| Memory leak | Valgrind, ASan | RAII, smart pointers |
| Use-after-free | ASan, Valgrind | Ownership tracking |
| Double free | ASan | Single owner pattern |
| Data race | TSan | Mutex, Rust borrow checker |
| Buffer overflow | ASan | Bounds checking |
<!-- END_SECTION: DEBUGGING -->

---

<!-- SECTION: PROFILING -->
## Profiling

```bash
# CPU profiling
perf record -g ./program
perf report

# Flamegraph
perf script | ./flamegraph.pl > flame.svg

# Rust
cargo flamegraph --bin myprogram

# Cache analysis
perf stat -e cache-misses,branch-misses ./program
```

### Benchmarking

```cpp
// Google Benchmark (C++)
static void BM_Function(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(function_under_test());
    }
}
BENCHMARK(BM_Function);
```

```rust
// Criterion (Rust)
fn bench(c: &mut Criterion) {
    c.bench_function("fib", |b| b.iter(|| fibonacci(black_box(20))));
}
```

```go
// Go testing
func BenchmarkFunc(b *testing.B) {
    for i := 0; i < b.N; i++ { functionUnderTest() }
}
```
<!-- END_SECTION: PROFILING -->

---

<!-- SECTION: PITFALLS -->
## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| ABA problem | Tagged pointers, hazard pointers |
| False sharing | Cache line alignment (64 bytes) |
| Deadlock | Consistent lock ordering |
| Premature optimization | Profile first |
| Unnecessary allocations | Object pools, arena |
<!-- END_SECTION: PITFALLS -->

---

## Language Selection

| Language | Memory | Concurrency | Use Case |
|----------|--------|-------------|----------|
| C | Manual | pthreads | Embedded, OS |
| C++ | RAII | std::thread | Performance, games |
| Rust | Ownership | async, channels | Safety-critical |
| Go | GC | goroutines | Network services |

## Checklist

## Language Selection

| Language | Memory | Concurrency | Use Case |
|----------|--------|-------------|----------|
| C | Manual | pthreads | Embedded, OS |
| C++ | RAII | std::thread | Performance, games |
| Rust | Ownership | async, channels | Safety-critical |
| Go | GC | goroutines | Network services |

## Checklist

- [ ] Memory management pattern selected (pool/arena/RAII)
- [ ] Concurrency primitives appropriate (mutex/lock-free)
- [ ] Memory ordering correct for atomics
- [ ] Cache-friendly data layout (SoA if needed)
- [ ] False sharing prevented (alignment)
- [ ] Sanitizers run (ASan, TSan)
- [ ] Profiled before optimizing
- [ ] Benchmarks verify improvements
