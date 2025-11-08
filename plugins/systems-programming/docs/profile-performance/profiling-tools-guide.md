# Performance Profiling Tools Guide

Comprehensive guide to profiling tools for systems programming: perf, valgrind, flamegraphs, hardware counters, and language-specific profilers.

---

## perf - Linux Performance Analysis

### Basic CPU Profiling

```bash
# Record CPU profiling data
perf record -g ./program arguments

# View report
perf report

# Record with frequency (99 Hz is good default)
perf record -g -F 99 ./program

# Record specific process
perf record -g -p PID

# Record for duration
perf record -g -a sleep 30  # System-wide for 30 seconds
```

### Understanding perf report

```
Samples: 10K of event 'cycles:u', Event count (approx.): 8427652000
Overhead  Command  Shared Object       Symbol
  45.23%  program  program             [.] hot_function
  23.45%  program  libc-2.31.so        [.] __memcpy_avx_unaligned
  12.34%  program  program             [.] another_function
   5.67%  program  [kernel.kallsyms]   [k] entry_SYSCALL_64
```

**Interpreting:**
- **Overhead**: Percentage of samples in this function
- **Shared Object**: Library or binary
- **Symbol**: Function name
- `[.]` = userspace, `[k]` = kernel

### Hardware Counter Analysis

```bash
# List available events
perf list

# Cache misses
perf stat -e L1-dcache-load-misses,LLC-load-misses ./program

# Branch mispredictions
perf stat -e branches,branch-misses ./program

# Instructions per cycle (IPC)
perf stat -e cycles,instructions ./program

# Comprehensive stats
perf stat ./program
```

**Example output:**
```
 Performance counter stats for './program':

       1,234.56 msec task-clock                #    0.999 CPUs utilized
             12      context-switches          #    9.724 /sec
              0      cpu-migrations            #    0.000 /sec
            456      page-faults               #  369.371 /sec
  4,567,890,123      cycles                    #    3.700 GHz
  6,789,012,345      instructions              #    1.49  insn per cycle
  1,234,567,890      branches                  # 1000.000 M/sec
     12,345,678      branch-misses             #    1.00% of all branches
```

### Advanced perf Features

```bash
# Line-level profiling (requires debug symbols)
perf annotate

# Generate flamegraph
perf record -g -F 99 ./program
perf script | flamegraph.pl > flame.svg

# Top-like interface
perf top

# Memory access profiling
perf mem record ./program
perf mem report

# TLB (Translation Lookaside Buffer) misses
perf stat -e dTLB-load-misses,iTLB-load-misses ./program
```

---

## Flamegraphs

### Generating Flamegraphs

```bash
# Install flamegraph tools
git clone https://github.com/brendangregg/FlameGraph
cd FlameGraph

# Capture perf data
perf record -g -F 99 ./program

# Generate SVG
perf script | ./stackcollapse-perf.pl | ./flamegraph.pl > flame.svg

# Open in browser
firefox flame.svg
```

### Rust Flamegraphs with cargo-flamegraph

```bash
# Install
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bin my-program

# Opens flamegraph.svg automatically
```

### Reading Flamegraphs

**Interpretation:**
- **Width**: Time spent (wider = more samples)
- **Height**: Call stack depth
- **Flat tops**: Direct CPU consumers (optimize these!)
- **Color**: Random (for visual distinction)

**What to look for:**
- Wide plateaus at the top (hot functions)
- Unexpected library calls
- Excessive allocations (malloc/free)

---

## Valgrind Tools

### Callgrind - Detailed Profiling

```bash
# Profile with callgrind
valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./program

# View with kcachegrind (GUI)
kcachegrind callgrind.out.12345

# Command-line viewer
callgrind_annotate callgrind.out.12345
```

**Callgrind output:**
```
I refs:        10,234,567,890
I1  misses:         1,234,567
LLi misses:            12,345
Branches:        1,234,567,890
Mispredicts:        12,345,678
```

### Massif - Heap Profiling

```bash
# Run massif
valgrind --tool=massif ./program

# View text report
ms_print massif.out.12345

# Visualize
massif-visualizer massif.out.12345
```

**Massif output:**
```
    KB
19.71^                                               #
     |                                               #
     |                                      @@@@@@@@@#
     |                                  @@@:#        #
     |                              @@@@:  #        #
   0 +----------------------------------------------------------------------->Mi
     0                                                                   113.1

Number of snapshots: 50
 Detailed snapshots: [9, 19, 29, 39, 49]
```

### DHAT - Dynamic Heap Analysis

```bash
# Run DHAT
valgrind --tool=dhat ./program

# Opens dh_view.html automatically
firefox dh_view.html
```

**DHAT metrics:**
- Total bytes allocated
- Total blocks allocated
- Short-lived blocks
- Allocation hotspots

### Cachegrind - Cache Profiling

```bash
# Profile cache usage
valgrind --tool=cachegrind ./program

# View report
cg_annotate cachegrind.out.12345
```

**Cachegrind output:**
```
I refs:      10,234,567,890
I1  misses:      1,234,567  (0.01%)
LLi misses:         12,345  (0.00%)

D refs:       4,567,890,123
D1  misses:      45,678,901  (1.00%)
LLd misses:       1,234,567  (0.03%)
```

---

## Language-Specific Profilers

### Rust: cargo-flamegraph

```bash
cargo install flamegraph

# Profile binary
cargo flamegraph --bin my-app

# Profile specific test
cargo flamegraph --test my-test

# With release optimizations
cargo flamegraph --release
```

### Rust: Criterion Benchmarks

```toml
[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "my_benchmark"
harness = false
```

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

```bash
cargo bench
```

### C/C++: gprof

```bash
# Compile with profiling
gcc -pg -O2 program.c -o program

# Run program (generates gmon.out)
./program

# View report
gprof program gmon.out > analysis.txt
gprof program gmon.out | less
```

### C/C++: Google Benchmark

```cpp
#include <benchmark/benchmark.h>
#include <vector>

static void BM_VectorPush(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<int> v;
        for (int i = 0; i < state.range(0); ++i) {
            v.push_back(i);
        }
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_VectorPush)->Range(8, 8<<10)->Complexity();
BENCHMARK_MAIN();
```

```bash
g++ -std=c++17 -O3 benchmark.cpp -lbenchmark -lpthread -o bench
./bench
```

### Go: pprof

```go
import (
    "runtime/pprof"
    "os"
)

func main() {
    // CPU profiling
    f, _ := os.Create("cpu.prof")
    pprof.StartCPUProfile(f)
    defer pprof.StopCPUProfile()

    // Run code
    doWork()
}
```

```bash
# Analyze profile
go tool pprof cpu.prof

# Web interface
go tool pprof -http=:8080 cpu.prof
```

---

## System-Wide Profiling

### perf top

Real-time profiling:
```bash
# System-wide live profiling
perf top

# Specific process
perf top -p PID

# With call graph
perf top -g
```

### htop

Interactive process viewer:
```bash
htop

# Show threads
H

# Show tree view
t

# Filter by user
u
```

### iotop

I/O monitoring:
```bash
# Requires root
sudo iotop

# Show only active processes
sudo iotop -o
```

---

## Micro-Benchmarking Best Practices

### 1. Prevent Optimization

**C/C++:**
```c
static void DoNotOptimize(void* ptr) {
    asm volatile("" : : "r,m"(ptr) : "memory");
}

int result = expensive_function();
DoNotOptimize(&result);
```

**Rust:**
```rust
use std::hint::black_box;

let result = black_box(expensive_function());
```

### 2. Warmup

```rust
// Criterion does this automatically
c.bench_function("test", |b| {
    b.iter(|| {
        // Benchmark code
    });
});
```

### 3. Statistical Significance

```bash
# Run multiple iterations
cargo bench -- --sample-size 1000
```

---

## Profiling Checklist

**Before profiling:**
- [ ] Compile with optimizations (-O2 or -O3)
- [ ] Include debug symbols (-g)
- [ ] Use representative workload
- [ ] Warm up caches

**During profiling:**
- [ ] Profile multiple runs
- [ ] Use appropriate tool (CPU vs memory vs cache)
- [ ] Minimize system noise
- [ ] Profile realistic scenarios

**After profiling:**
- [ ] Identify hot functions (>5% CPU)
- [ ] Check cache miss rates
- [ ] Verify IPC (instructions per cycle)
- [ ] Document findings
- [ ] Re-profile after optimization

---

## Hardware Counter Events Reference

### Cache Events

```bash
# L1 data cache
perf stat -e L1-dcache-loads,L1-dcache-load-misses ./program

# L1 instruction cache
perf stat -e L1-icache-loads,L1-icache-load-misses ./program

# Last-level cache (L3)
perf stat -e LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses ./program
```

### Branch Events

```bash
perf stat -e branches,branch-misses,branch-loads,branch-load-misses ./program
```

### TLB Events

```bash
perf stat -e dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses ./program
```

### Memory Events

```bash
perf stat -e mem-loads,mem-stores ./program
```

---

## Interpreting Results

### Good Performance Indicators

- **IPC > 1.0**: Good instruction throughput
- **Cache miss rate < 1%**: Good cache usage
- **Branch misprediction < 5%**: Predictable branches
- **TLB miss rate < 0.1%**: Good memory access patterns

### Red Flags

- **IPC < 0.5**: Stalls, memory bottlenecks
- **L3 miss rate > 10%**: Poor locality
- **Branch misprediction > 10%**: Unpredictable branches
- **High malloc/free**: Excessive allocations

---

## Summary: Tool Selection

**CPU profiling**: perf, flamegraph, gprof
**Memory profiling**: valgrind massif, DHAT
**Cache profiling**: valgrind cachegrind, perf stat
**System-wide**: perf top, htop
**Micro-benchmarks**: criterion (Rust), Google Benchmark (C++), pprof (Go)
**Hardware counters**: perf stat
**Call graph**: valgrind callgrind, kcachegrind
