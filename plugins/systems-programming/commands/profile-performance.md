---
version: 1.0.3
command: /profile-performance
description: Comprehensive performance profiling workflow with 3 execution modes from quick hotspot identification to enterprise-grade optimization
argument-hint: [target-binary-or-code]
execution_modes:
  quick:
    duration: "30 minutes - 1 hour"
    description: "Basic profiling and hotspot identification"
    scope: "perf record, flamegraph generation, identify functions >5% CPU time"
    tools: "perf, flamegraph"
    deliverables: "Flamegraph SVG, list of hot functions"
  standard:
    duration: "2-3 hours"
    description: "Comprehensive profiling with optimization recommendations"
    scope: "CPU profiling (perf), memory profiling (valgrind massif), cache analysis (perf stat), hardware counters, optimization strategy"
    tools: "perf, valgrind, flamegraph, cachegrind"
    deliverables: "Performance report with metrics, optimization recommendations"
  enterprise:
    duration: "1 day"
    description: "Full performance audit with benchmarking suite"
    scope: "Multi-dimensional profiling (CPU/memory/cache/I/O), hardware counter analysis, micro-benchmarks, before/after validation, regression testing"
    tools: "perf, valgrind, criterion/google-benchmark, hardware counters, DHAT"
    deliverables: "Complete performance audit, benchmarking suite, optimization roadmap"
workflow_type: "sequential"
interactive_mode: true
color: red
allowed-tools: Bash, Read, Grep, Write
---

# Performance Profiling Workflow

Systematic performance analysis using industry-standard tools for profiling and optimizing systems-level code.

## Context

Target for profiling: $ARGUMENTS

## Execution Mode Selection

<AskUserQuestion>
questions:
  - question: "What level of performance analysis do you need?"
    header: "Profiling Depth"
    multiSelect: false
    options:
      - label: "Quick (30min-1h)"
        description: "Basic CPU profiling with perf and flamegraph. Identify hot functions (>5% CPU time) and generate visualization."

      - label: "Standard (2-3h)"
        description: "Comprehensive profiling: CPU (perf), memory (valgrind massif), cache analysis (perf stat), hardware counters. Includes optimization recommendations."

      - label: "Enterprise (1 day)"
        description: "Full performance audit: multi-dimensional profiling (CPU/memory/cache/I/O), hardware counter analysis, micro-benchmarks, regression testing, and optimization roadmap."
</AskUserQuestion>

## Instructions

### 1. Identify Profiling Goals

**Determine optimization target:**
- **Latency**: Reduce time for single operation
- **Throughput**: Increase operations per second
- **Memory**: Reduce allocation or peak usage
- **Cache**: Improve cache hit rates
- **Scalability**: Performance under load

**Mode-Specific Focus:**
- **Quick**: CPU hotspots only
- **Standard**: CPU + memory + cache
- **Enterprise**: Full multi-dimensional analysis

---

### 2. Build with Profiling Symbols

**C/C++:**
```bash
# Optimized with debug symbols
gcc -O3 -march=native -g program.c -o program

# Or with CMake
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

**Rust:**
```bash
# Release with debug info
cargo build --release

# Or configure in Cargo.toml
[profile.release]
debug = true
```

**Go:**
```bash
go build -o program
```

---

### 3. CPU Profiling (All Modes)

**Quick Mode: perf + Flamegraph**
```bash
# Record CPU profiling
perf record -g -F 99 ./program

# Generate flamegraph
perf script | flamegraph.pl > flame.svg

# Or with cargo-flamegraph (Rust)
cargo flamegraph --bin my-program
```

**Interpreting Flamegraph:**
- **Width**: Time spent (wider = more samples)
- **Flat tops**: Direct CPU consumers → **Optimize these!**
- Look for functions >5% CPU time

**Standard Mode: Add perf report**
```bash
# View detailed report
perf report

# Annotate source code
perf annotate
```

**Enterprise Mode: Multi-run analysis**
```bash
# Multiple profiling runs for statistical significance
for i in {1..10}; do
    perf record -g -F 99 -o perf.data.$i ./program
done

# Aggregate and analyze
```

**Full reference:** [Profiling Tools Guide](../docs/profile-performance/profiling-tools-guide.md)

---

### 4. Hardware Counter Analysis (Standard+)

**Cache Misses:**
```bash
perf stat -e L1-dcache-load-misses,LLC-load-misses ./program
```

**Branch Mispredictions:**
```bash
perf stat -e branches,branch-misses ./program
```

**Instructions Per Cycle (IPC):**
```bash
perf stat -e cycles,instructions ./program
```

**Interpreting:**
- **IPC > 1.0**: Good throughput
- **Cache miss rate < 1%**: Good locality
- **Branch misprediction < 5%**: Predictable branches

**Reference:** [Profiling Tools - Hardware Counters](../docs/profile-performance/profiling-tools-guide.md#hardware-counter-events-reference)

---

### 5. Memory Profiling (Standard+)

**Heap Profiling with Massif:**
```bash
# Run massif
valgrind --tool=massif ./program

# View report
ms_print massif.out.12345

# Visualize
massif-visualizer massif.out.12345
```

**Look for:**
- Growing memory without decreasing (leak)
- Frequent allocations (optimization opportunity)
- Peak usage (temporary allocations)

**Dynamic Heap Analysis (Enterprise):**
```bash
valgrind --tool=dhat ./program
firefox dh_view.html
```

**Reference:** [Profiling Tools - Valgrind](../docs/profile-performance/profiling-tools-guide.md#valgrind-tools)

---

### 6. Optimization Strategy (Standard+)

**Based on profiling results, apply optimizations:**

**CPU Optimization:**
1. Algorithm improvement (O(n²) → O(n log n))
2. Cache optimization (SoA layout)
3. Branch optimization (reduce unpredictable branches)
4. SIMD vectorization
5. Inlining

**Memory Optimization:**
1. Object pooling
2. Arena allocators
3. Reduce allocations
4. Lazy initialization

**Reference:** [Optimization Patterns Guide](../docs/profile-performance/optimization-patterns.md)

---

### 7. Micro-Benchmarking (Enterprise)

**Rust (Criterion):**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark(c: &mut Criterion) {
    c.bench_function("function", |b| {
        b.iter(|| function(black_box(input)));
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
```

```bash
cargo bench
```

**C++ (Google Benchmark):**
```cpp
#include <benchmark/benchmark.h>

static void BM_Function(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(function(input));
    }
}
BENCHMARK(BM_Function);
BENCHMARK_MAIN();
```

**Reference:** [Profiling Tools - Benchmarking](../docs/profile-performance/profiling-tools-guide.md#micro-benchmarking-best-practices)

---

### 8. Verification Workflow (Standard+)

**Before Optimization:**
1. Profile and record baseline metrics
2. Create benchmark suite
3. Run tests to ensure correctness

**After Optimization:**
1. Re-run benchmarks (verify improvement)
2. Re-profile (check for new hotspots)
3. Run full test suite (no regressions)
4. Measure end-to-end impact

**Enterprise Mode: Regression Testing**
```bash
# Automated performance regression tests in CI/CD
cargo bench --bench my_benchmark -- --save-baseline baseline
# After changes
cargo bench --bench my_benchmark -- --baseline baseline
```

---

## Output Deliverables

### Quick Mode (30min-1h):
✅ Flamegraph visualization (SVG)
✅ List of hot functions (>5% CPU)
✅ Basic optimization recommendations

### Standard Mode (2-3h):
✅ Comprehensive performance report
✅ CPU profiling (perf report + flamegraph)
✅ Memory profiling (massif)
✅ Cache analysis (perf stat)
✅ Hardware counter metrics
✅ Prioritized optimization recommendations

### Enterprise Mode (1 day):
✅ Full performance audit report
✅ Multi-dimensional profiling data
✅ Micro-benchmark suite
✅ Before/after metrics
✅ Optimization roadmap
✅ Regression test suite
✅ CI/CD integration

---

## External Documentation

Comprehensive profiling and optimization guides:

- **[Profiling Tools Guide](../docs/profile-performance/profiling-tools-guide.md)** (~527 lines)
  - perf usage and hardware counters
  - Flamegraph generation and interpretation
  - Valgrind tools (massif, DHAT, cachegrind, callgrind)
  - Language-specific profilers (Rust criterion, C++ Google Benchmark, Go pprof)
  - System-wide profiling (perf top, htop, iotop)

- **[Optimization Patterns](../docs/profile-performance/optimization-patterns.md)** (~629 lines)
  - Algorithm optimization (O(n²) → O(n log n))
  - Cache optimization (SoA layout, alignment)
  - Memory optimization (pooling, arenas, string interning)
  - SIMD vectorization (AVX2, auto-vectorization)
  - Branch optimization (branchless programming, lookup tables)
  - Parallelization (Rayon, OpenMP)
  - I/O optimization (buffering, mmap)

---

## Profiling Checklist

**Before Profiling:**
- [ ] Compile with optimizations (-O2 or -O3)
- [ ] Include debug symbols (-g)
- [ ] Use representative workload
- [ ] Warm up caches

**During Profiling:**
- [ ] Profile multiple runs
- [ ] Use appropriate tool (CPU vs memory vs cache)
- [ ] Minimize system noise
- [ ] Profile realistic scenarios

**After Profiling:**
- [ ] Identify hot functions (>5% CPU)
- [ ] Check cache miss rates
- [ ] Verify IPC (instructions per cycle)
- [ ] Document findings
- [ ] Re-profile after optimization

---

## Optimization Principles

1. **Measure First**: Always profile before optimizing
2. **Optimize Hot Paths**: Focus on functions >5% CPU time
3. **Algorithm First**: Best algorithm choice has biggest impact
4. **Cache Matters**: Memory access patterns dominate performance
5. **Validate**: Always measure before/after
