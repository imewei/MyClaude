# Comprehensive Profiling Guide

## CPU Profiling Workflow

### 1. Sampling Profilers (perf, Instruments, VTune)

**Linux: perf**
```bash
# Basic CPU profiling
perf record -g ./program arg1 arg2
perf report

# Record with call graphs
perf record --call-graph dwarf ./program

# Profile specific PID
perf record -p <pid> -g

# Record for N seconds
perf record -g -- sleep 10 &
perf record -p $(pgrep program)

# Hardware counters
perf stat -e cycles,instructions,cache-references,cache-misses ./program

# Branch prediction analysis
perf stat -e branch-instructions,branch-misses ./program

# Cache analysis
perf stat -e L1-dcache-load-misses,LLC-load-misses ./program
```

**Interpreting perf output**:
```
# perf report output
  50.23%  program  program  [.] hot_function
  20.15%  program  libc     [.] memcpy
  10.08%  program  program  [.] warm_function
```
- Focus on functions taking >5-10% of total time
- Check if time is in your code vs libraries
- Look for unexpected functions (surprising hot spots)

**Flamegraph generation**:
```bash
# Install flamegraph
git clone https://github.com/brendangregg/FlameGraph
cd FlameGraph

# Generate flamegraph
perf record -g -F 99 ./program
perf script | ./stackcollapse-perf.pl | ./flamegraph.pl > flame.svg

# Open in browser
firefox flame.svg
```

**Reading flamegraphs**:
- X-axis: Alphabetical ordering (NOT time!)
- Y-axis: Stack depth
- Width: Time spent in function + callees
- Flat tops: Function directly consuming CPU (optimization target)
- Tall towers: Deep call stacks (may indicate abstraction overhead)

### 2. Instrumentation Profilers (gprof, callgrind)

**gprof (compile-time instrumentation)**:
```bash
# Compile with profiling
gcc -pg -O2 program.c -o program

# Run program (generates gmon.out)
./program

# Generate report
gprof program gmon.out > analysis.txt
```

**Valgrind callgrind (runtime instrumentation)**:
```bash
# Run with callgrind
valgrind --tool=callgrind --dump-instr=yes ./program

# Visualize with kcachegrind
kcachegrind callgrind.out.<pid>
```

**callgrind advantages**:
- No recompilation needed
- Very detailed call graphs
- Cache simulation available
- Per-line profiling

**callgrind disadvantages**:
- 10-100x slowdown
- May change timing-dependent behavior

### 3. Rust-Specific Profiling

**cargo-flamegraph**:
```bash
# Install
cargo install flamegraph

# Profile release build
cargo flamegraph --bin myprogram

# Profile with specific features
cargo flamegraph --release --features="production"

# Opens flame.svg in browser automatically
```

**perf + cargo**:
```bash
# Build with debug symbols in release mode
cargo build --release

# Profile
perf record --call-graph dwarf ./target/release/myprogram
perf report
```

**Criterion benchmarking**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_algorithm(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("Algorithm A", size), size,
            |b, &size| {
                b.iter(|| algorithm_a(black_box(size)));
            });
        group.bench_with_input(BenchmarkId::new("Algorithm B", size), size,
            |b, &size| {
                b.iter(|| algorithm_b(black_box(size)));
            });
    }

    group.finish();
}

criterion_group!(benches, bench_algorithm);
criterion_main!(benches);
```

```bash
# Run benchmarks
cargo bench

# View detailed HTML report
open target/criterion/report/index.html
```

## Memory Profiling

### 1. Heap Profiling with Massif

**Valgrind massif**:
```bash
# Profile heap usage
valgrind --tool=massif --massif-out-file=massif.out ./program

# Visualize
ms_print massif.out

# GUI visualization
massif-visualizer massif.out
```

**Massif output interpretation**:
```
    MB
60.00 ^                                     #
      |                                   @ #
      |                                 @ @ #
40.00 +                               @ @ @ #
      |                             @ @ @ @ #
      |                           @ @ @ @ @ #
20.00 +                         @ @ @ @ @ @ #
      |                       @ @ @ @ @ @ @ #
      |                     @ @ @ @ @ @ @ @ #
 0.00 +--------------------------------------------->
         0   10  20  30  40  50  60  70  80  90  100
                         Time (seconds)
```

Look for:
- Growing memory that never decreases (potential leak)
- Sudden spikes (large allocations)
- Steady growth (accumulation without cleanup)

### 2. Heaptrack (Linux)

```bash
# Install
sudo apt install heaptrack heaptrack-gui

# Profile
heaptrack ./program

# Visualize
heaptrack_gui heaptrack.program.<pid>.gz
```

**Heaptrack features**:
- Allocation tracking with call stacks
- Peak memory usage identification
- Temporary allocation detection
- Leak detection

### 3. Rust Memory Profiling

**DHAT (dynamic heap analysis tool)**:
```bash
# Use valgrind's DHAT
valgrind --tool=dhat ./target/release/program

# View report
dhat/dh_view.html
```

**bytehound** (Rust-specific profiler):
```bash
# Install
cargo install bytehound-cli

# Record
bytehound record ./target/release/program

# Analyze
bytehound server memory-profiling_*.dat
# Opens web UI
```

**jemalloc profiling**:
```toml
# Cargo.toml
[dependencies]
jemallocator = "0.5"

[profile.release]
debug = true  # Keep symbols
```

```rust
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    // Your code
}
```

```bash
# Run with profiling
MALLOC_CONF=prof:true,lg_prof_interval:30 ./target/release/program

# Analyze heap dumps
jeprof --pdf ./target/release/program jeprof.*.heap > profile.pdf
```

## Profiling Best Practices

### 1. Profile Release Builds

```bash
# Rust
cargo build --release
perf record ./target/release/program

# C++
g++ -O3 -march=native -g program.cpp
perf record ./a.out

# Go
go build -o program .
perf record ./program
```

**Always include debug symbols** (`-g`) even in release builds for readable profiling output.

### 2. Use Representative Workloads

- Profile with production-like data
- Include both hot and cold paths
- Run long enough for patterns to emerge (30+ seconds)
- Test under realistic concurrency

### 3. Profile Multiple Times

- Run 3-5 profiling sessions
- Look for consistent hotspots across runs
- Investigate variance (timing-dependent behavior?)
- Compare before/after optimization

### 4. Start Broad, Then Focus

1. **Coarse profiling**: Identify top 5 functions
2. **Focused profiling**: Zoom into those functions
3. **Line-level profiling**: Identify specific bottlenecks
4. **Micro-benchmarking**: Test optimization hypotheses

## Hardware Performance Counters

### Useful Counters

```bash
# Cache analysis
perf stat -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores \
          -e LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses \
          ./program

# Branch prediction
perf stat -e branches,branch-misses \
          -e branch-load-misses,branch-loads \
          ./program

# TLB analysis
perf stat -e dTLB-loads,dTLB-load-misses,dTLB-stores,dTLB-store-misses \
          ./program

# Instructions per cycle (IPC)
perf stat -e cycles,instructions,stalled-cycles-frontend,stalled-cycles-backend \
          ./program
```

### Interpreting Counter Data

**Cache miss rate**:
```
L1-dcache-load-misses: 10,000,000
L1-dcache-loads:       100,000,000
Miss rate: 10%
```
- <5%: Excellent
- 5-10%: Good
- 10-20%: Moderate (optimize if hot path)
- >20%: Poor (likely optimization target)

**Branch prediction**:
```
branches:        100,000,000
branch-misses:     5,000,000
Mispredict rate: 5%
```
- <2%: Excellent
- 2-5%: Good
- 5-10%: Moderate
- >10%: Poor (consider branch elimination)

**IPC (Instructions Per Cycle)**:
```
instructions:           1,000,000,000
cycles:                   500,000,000
IPC: 2.0
```
- >3: Excellent (near peak)
- 2-3: Good
- 1-2: Moderate
- <1: Poor (stalls, cache misses)

## Profiling Concurrent Programs

### Thread-Specific Profiling

```bash
# Profile all threads
perf record -g -s ./program

# Per-thread report
perf report --sort comm,dso,symbol

# Thread timeline
perf script | grep -A 5 "thread_name"
```

### Lock Contention Analysis

**perf lock**:
```bash
# Record lock events
perf lock record ./program

# Analyze contention
perf lock report

# Top contended locks
perf lock report --sort=contended
```

**Identifying contention hotspots**:
```
Name                   acquired  contended  avg wait (ns)
lock_name                 10000       5000        100000
```
- High contention %: Consider lock-free alternatives
- Long wait times: Lock held too long
- Many acquisitions: Consider coarser locking

### ThreadSanitizer Profiling

```bash
# Compile with TSan
clang++ -fsanitize=thread -O2 -g program.cpp

# Run (detects data races)
./a.out

# Rust
RUSTFLAGS="-Z sanitizer=thread" cargo run --target x86_64-unknown-linux-gnu
```

## Profiling Checklist

Before optimizing:
- [ ] Profile release build with debug symbols
- [ ] Use representative workload
- [ ] Run multiple times for consistency
- [ ] Collect CPU profile (perf/flamegraph)
- [ ] Check hardware counters (cache, branches, IPC)
- [ ] Profile memory allocations (massif/heaptrack)
- [ ] Identify top 3-5 hotspots

During optimization:
- [ ] Micro-benchmark specific changes
- [ ] Re-profile after each change
- [ ] Verify improvement (don't trust assumptions!)
- [ ] Check for regression in other metrics

After optimization:
- [ ] Full regression test suite
- [ ] Measure end-to-end impact
- [ ] Document changes and reasoning
- [ ] Update benchmarks

## Common Profiling Mistakes

1. **Profiling debug builds**: 10-100x slower, misleading results
2. **Profiling trivial workloads**: Startup/shutdown dominates
3. **Optimizing without profiling**: Wasted effort on non-hotspots
4. **Ignoring statistical significance**: Single run isn't enough
5. **Forgetting debug symbols**: Unreadable output
6. **Not profiling in production**: Dev workload differs from prod

## Tool Quick Reference

| Tool | Purpose | Overhead | Platform |
|------|---------|----------|----------|
| perf | CPU profiling | ~1-5% | Linux |
| Valgrind (callgrind) | Detailed profiling | 10-100x | Linux |
| Valgrind (massif) | Heap profiling | 10-50x | Linux |
| Valgrind (DHAT) | Allocation analysis | 10-50x | Linux |
| heaptrack | Heap profiling | ~5% | Linux |
| gprof | CPU profiling | ~5% | Unix-like |
| flamegraph | Visualization | N/A | All (uses perf) |
| Instruments | All-in-one | Low | macOS |
| VTune | CPU profiling | ~1-5% | Linux/Windows |
| cargo-flamegraph | CPU profiling | ~1-5% | All (Rust) |
| criterion | Micro-benchmarks | N/A | All (Rust) |
| google-benchmark | Micro-benchmarks | N/A | All (C++) |
| pprof | Go profiling | ~5% | All (Go) |

## Example Optimization Case Study

**Initial profile**:
```
60% - process_data()
  40% - parse_json()
    30% - allocate_string()
    10% - validate()
  20% - compute()
```

**Optimization steps**:

1. **Reduce allocations in parse_json()** (targeted 30%)
   - Pre-allocate string buffer
   - Reuse allocations across calls
   - Result: 30% → 5% (25% improvement)

2. **Optimize compute()** (now 20% of total)
   - Vectorize loop
   - Improve cache locality
   - Result: 20% → 8% (12% improvement)

3. **Re-profile to verify**:
```
35% - process_data()  (was 60%)
  10% - parse_json()  (was 40%)
    5% - allocate_string()  (was 30%)
    5% - validate()  (was 10%)
  8% - compute()  (was 20%)
  17% - other functions now visible
```

Total speedup: ~1.7x (60% → 35% of runtime)

**Key lessons**:
- Focus on largest hotspots first (30% allocation)
- Measure impact (don't assume)
- New hotspots emerge after optimization
- Document what was tried and why
