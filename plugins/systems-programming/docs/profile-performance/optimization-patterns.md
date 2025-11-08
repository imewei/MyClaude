# Performance Optimization Patterns

Comprehensive guide to optimization strategies for CPU, memory, cache, and algorithmic improvements with before/after examples.

---

## Algorithm Optimization

### Time Complexity Reduction

**Example: O(n²) → O(n log n)**

**Before: Nested loop sorting**
```c
void bubble_sort(int* arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
// O(n²) time
```

**After: Quicksort**
```c
#include <stdlib.h>

void quicksort(int* arr, size_t n) {
    qsort(arr, n, sizeof(int), compare);
}
// O(n log n) time
```

**Speedup: 100× for n=10,000**

### Hash Table for O(1) Lookup

**Before: Linear search O(n)**
```c
bool contains(int* arr, size_t n, int value) {
    for (size_t i = 0; i < n; i++) {
        if (arr[i] == value) return true;
    }
    return false;
}
```

**After: Hash set O(1)**
```rust
use std::collections::HashSet;

let set: HashSet<i32> = vec![1, 2, 3, 4, 5].into_iter().collect();
let found = set.contains(&3); // O(1)
```

---

## Cache Optimization

### Structure of Arrays (SoA) Layout

**Before: Array of Structures (AoS) - Poor Cache Locality**
```c
struct Particle {
    float x, y, z;     // 12 bytes
    float vx, vy, vz;  // 12 bytes
    float mass;        // 4 bytes
    int id;            // 4 bytes
};  // Total: 32 bytes per particle

void update_positions(struct Particle* particles, size_t n) {
    for (size_t i = 0; i < n; i++) {
        particles[i].x += particles[i].vx;
        particles[i].y += particles[i].vy;
        particles[i].z += particles[i].vz;
        // Loads entire struct (32 bytes) but only uses position+velocity
    }
}
```

**After: Structure of Arrays (SoA) - Excellent Cache Locality**
```c
struct Particles {
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
    float* mass;
    int* id;
    size_t count;
};

void update_positions(struct Particles* p) {
    for (size_t i = 0; i < p->count; i++) {
        p->x[i] += p->vx[i];  // Sequential access, cache-friendly
        p->y[i] += p->vy[i];
        p->z[i] += p->vz[i];
    }
}
```

**Speedup: 3-5× due to better cache utilization**

### Cache Line Alignment

```c
// Before: False sharing
struct Counter {
    int count1;  // Same cache line as count2
    int count2;  // Modified by different threads = false sharing
};

// After: Align to cache lines (typically 64 bytes)
struct Counter {
    int count1 __attribute__((aligned(64)));
    int count2 __attribute__((aligned(64)));
};
```

**Rust:**
```rust
#[repr(align(64))]
struct Counter {
    count1: i32,
    _pad: [u8; 60],
    count2: i32,
}
```

---

## Memory Optimization

### Object Pooling

**Before: Frequent Allocations**
```c
void process_requests(void) {
    for (int i = 0; i < 1000000; i++) {
        Request* req = malloc(sizeof(Request));
        handle_request(req);
        free(req);  // Expensive!
    }
}
```

**After: Memory Pool**
```c
typedef struct {
    void** free_list;
    size_t free_count;
    size_t capacity;
    size_t object_size;
} Pool;

Pool* pool_create(size_t object_size, size_t capacity);
void* pool_alloc(Pool* pool);
void pool_free(Pool* pool, void* ptr);

void process_requests(void) {
    Pool* pool = pool_create(sizeof(Request), 1000);

    for (int i = 0; i < 1000000; i++) {
        Request* req = pool_alloc(pool);
        handle_request(req);
        pool_free(pool, req);  // Fast! Just adds to free list
    }

    pool_destroy(pool);
}
```

**Speedup: 10-50× reduction in allocation time**

### Arena Allocator

```c
typedef struct {
    char* buffer;
    size_t size;
    size_t offset;
} Arena;

Arena* arena_create(size_t size) {
    Arena* arena = malloc(sizeof(Arena));
    arena->buffer = malloc(size);
    arena->size = size;
    arena->offset = 0;
    return arena;
}

void* arena_alloc(Arena* arena, size_t size) {
    if (arena->offset + size > arena->size) return NULL;
    void* ptr = arena->buffer + arena->offset;
    arena->offset += size;
    return ptr;
}

void arena_destroy(Arena* arena) {
    free(arena->buffer);
    free(arena);
}

// Usage: Allocate many objects, free all at once
Arena* arena = arena_create(1024 * 1024);  // 1 MB
for (int i = 0; i < 1000; i++) {
    Data* d = arena_alloc(arena, sizeof(Data));
    // Use d...
}
arena_destroy(arena);  // Free all at once
```

### String Interning

**Before: Duplicate strings**
```rust
let s1 = String::from("hello");
let s2 = String::from("hello");  // Duplicate allocation
let s3 = String::from("hello");  // Another duplicate
```

**After: String interning**
```rust
use std::collections::HashMap;

struct StringInterner {
    map: HashMap<String, &'static str>,
}

impl StringInterner {
    fn intern(&mut self, s: &str) -> &'static str {
        if let Some(&interned) = self.map.get(s) {
            return interned;
        }
        let interned = Box::leak(s.to_string().into_boxed_str());
        self.map.insert(s.to_string(), interned);
        interned
    }
}

// All three share the same allocation
let s1 = interner.intern("hello");
let s2 = interner.intern("hello");
let s3 = interner.intern("hello");
```

---

## SIMD Vectorization

### Auto-Vectorization

**Enable compiler auto-vectorization:**
```bash
gcc -O3 -march=native -ffast-math program.c
rustc -C opt-level=3 -C target-cpu=native
```

**Vectorizable pattern:**
```c
// Simple loop - compiler can auto-vectorize
void add_arrays(float* a, float* b, float* result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}
```

### Explicit SIMD (AVX2)

```c
#include <immintrin.h>

void add_arrays_simd(float* a, float* b, float* result, size_t n) {
    size_t i = 0;

    // Process 8 floats at a time (256-bit AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }

    // Handle remainder
    for (; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}
```

**Speedup: 4-8× for floating-point operations**

**Rust portable_simd:**
```rust
#![feature(portable_simd)]
use std::simd::*;

fn add_arrays(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in (0..a.len()).step_by(f32x8::LANES) {
        let va = f32x8::from_slice(&a[i..]);
        let vb = f32x8::from_slice(&b[i..]);
        let vr = va + vb;
        result[i..].copy_from_slice(vr.as_array());
    }
}
```

---

## Branch Optimization

### Branch Prediction Hints

**C/C++:**
```c
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

if (unlikely(error_condition)) {
    handle_error();  // Rare path
} else {
    normal_path();   // Common path
}
```

### Branchless Programming

**Before: Conditional**
```c
int max(int a, int b) {
    if (a > b) return a;
    else return b;
}
```

**After: Branchless**
```c
int max(int a, int b) {
    return a * (a > b) + b * (a <= b);
}

// Or with ternary (compiler often optimizes to branchless)
int max(int a, int b) {
    return (a > b) ? a : b;
}
```

### Lookup Table

**Before: Branchy logic**
```c
int score_grade(int score) {
    if (score >= 90) return 'A';
    else if (score >= 80) return 'B';
    else if (score >= 70) return 'C';
    else if (score >= 60) return 'D';
    else return 'F';
}
```

**After: Lookup table**
```c
const char grade_table[101] = {
    ['0' ... '59'] = 'F',
    ['60' ... '69'] = 'D',
    ['70' ... '79'] = 'C',
    ['80' ... '89'] = 'B',
    ['90' ... '100'] = 'A',
};

int score_grade(int score) {
    return grade_table[score];
}
```

---

## Inlining and Function Call Overhead

### Manual Inlining

**Before: Function call overhead**
```c
static int add(int a, int b) {
    return a + b;
}

int main() {
    for (int i = 0; i < 1000000; i++) {
        result += add(i, i + 1);  // Function call overhead
    }
}
```

**After: Inline**
```c
static inline int add(int a, int b) {
    return a + b;
}

// Or let compiler decide
__attribute__((always_inline)) static int add(int a, int b) {
    return a + b;
}
```

**Rust:**
```rust
#[inline]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// Force inline
#[inline(always)]
fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

---

## Parallelization

### Rayon (Rust)

**Before: Sequential**
```rust
let sum: i32 = (0..1_000_000)
    .map(|x| expensive_computation(x))
    .sum();
```

**After: Parallel**
```rust
use rayon::prelude::*;

let sum: i32 = (0..1_000_000)
    .into_par_iter()
    .map(|x| expensive_computation(x))
    .sum();
```

**Speedup: Linear with cores (e.g., 8× on 8-core)**

### OpenMP (C/C++)

```c
#include <omp.h>

void parallel_sum(int* arr, size_t n) {
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; i++) {
        sum += arr[i];
    }

    printf("Sum: %d\n", sum);
}
```

```bash
gcc -fopenmp -O3 program.c
```

---

## I/O Optimization

### Buffered I/O

**Before: Unbuffered write**
```c
for (int i = 0; i < 1000000; i++) {
    fprintf(file, "%d\n", i);  // Syscall for every write!
}
```

**After: Buffered write**
```c
char buffer[1024 * 1024];  // 1 MB buffer
setvbuf(file, buffer, _IOFBF, sizeof(buffer));

for (int i = 0; i < 1000000; i++) {
    fprintf(file, "%d\n", i);  // Buffered
}
```

### Memory-Mapped Files

```c
#include <sys/mman.h>
#include <fcntl.h>

int fd = open("data.bin", O_RDONLY);
struct stat sb;
fstat(fd, &sb);

void* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
// Access data directly without read() syscalls
int* array = (int*)data;

munmap(data, sb.st_size);
close(fd);
```

---

## Lazy Evaluation

**Rust:**
```rust
// Eager (computes all)
let sum: i32 = (0..1_000_000)
    .map(|x| expensive(x))
    .sum();

// Lazy (short-circuits)
let found = (0..1_000_000)
    .map(|x| expensive(x))
    .find(|&x| x > 1000);  // Stops at first match
```

---

## Common Optimization Principles

### 1. Measure First

```bash
# Profile before optimizing
perf record -g ./program
perf report
```

### 2. Optimize Hot Paths

Focus on functions with >5% CPU time.

### 3. Reduce Allocations

```rust
// Before: Allocates on every call
fn process(input: &str) -> String {
    input.to_uppercase()
}

// After: Reuse buffer
fn process(input: &str, output: &mut String) {
    output.clear();
    output.push_str(&input.to_uppercase());
}
```

### 4. Prefer Stack Over Heap

```rust
// Heap allocation
let vec = vec![0; 1000];

// Stack allocation (if size is known)
let arr = [0; 1000];
```

### 5. Use Const Generics (Rust)

```rust
// Runtime size parameter
fn process(data: &[i32]) { }

// Compile-time size (enables optimizations)
fn process<const N: usize>(data: &[i32; N]) { }
```

---

## Profiling-Guided Optimization (PGO)

### Generate Profile

```bash
# Rust
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release
./target/release/program  # Run with representative workload
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# Rust: Use profile for optimized build
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" cargo build --release
```

### GCC PGO

```bash
# Generate instrumented binary
gcc -O3 -fprofile-generate program.c -o program

# Run with representative workload
./program

# Build optimized binary with profile
gcc -O3 -fprofile-use program.c -o program
```

**Speedup: 5-20% typically**

---

## Summary: Optimization Hierarchy

1. **Algorithm**: Choose better algorithm (biggest impact)
2. **Data structures**: Use appropriate data structures
3. **Memory layout**: SoA, cache alignment, pooling
4. **SIMD**: Vectorize hot loops
5. **Branches**: Reduce mispredictions
6. **Inlining**: Eliminate call overhead
7. **Parallelization**: Use all cores
8. **I/O**: Buffer, memory-map
9. **PGO**: Profile-guided optimization

**Remember: Always measure before and after!**
