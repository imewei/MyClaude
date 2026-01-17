---
description: Comprehensive performance profiling with perf, flamegraph, and valgrind
triggers:
- /profile-performance
- comprehensive performance profiling with
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `['target-binary-or-code']`
The agent should parse these arguments from the user's request.

# Performance Profiling Workflow

Systematic performance analysis using industry-standard profiling tools.

## Context

Target for profiling: $ARGUMENTS

## Mode Selection

| Mode | Duration | Scope | Tools |
|------|----------|-------|-------|
| Quick | 30min-1h | CPU hotspots, flamegraph | perf, flamegraph |
| Standard (default) | 2-3h | CPU + memory + cache + recommendations | + valgrind massif, cachegrind |
| Enterprise | 1 day | Full audit + benchmarks + regression suite | + criterion, DHAT, hardware counters |

## Phase 1: Profiling Goals

| Goal | Focus | Metrics |
|------|-------|---------|
| Latency | Single operation time | p50/p95/p99 |
| Throughput | Operations per second | ops/sec |
| Memory | Allocation/peak usage | MB, allocs |
| Cache | Hit rates | L1/LLC miss % |
| Scalability | Performance under load | Scaling factor |

**Mode Focus:**
- Quick: CPU hotspots only
- Standard: CPU + memory + cache
- Enterprise: Multi-dimensional + regression testing

## Phase 2: Build with Profiling Symbols

| Language | Command | Notes |
|----------|---------|-------|
| C/C++ | `gcc -O3 -march=native -g` | Optimized + debug symbols |
| Rust | `cargo build --release` | Add `debug = true` in profile.release |
| Go | `go build` | Symbols included by default |

## Phase 3: CPU Profiling (All Modes)

### Core Commands

| Tool | Command | Output |
|------|---------|--------|
| perf record | `perf record -g -F 99 ./program` | perf.data |
| flamegraph | `perf script \| flamegraph.pl > flame.svg` | SVG visualization |
| cargo-flamegraph | `cargo flamegraph --bin program` | Rust-specific |
| perf report | `perf report` | Interactive analysis |
| perf annotate | `perf annotate` | Source-level hotspots |

### Flamegraph Interpretation

| Indicator | Meaning | Action |
|-----------|---------|--------|
| Wide bars | More samples | Optimization target |
| Flat tops | Direct CPU consumers | Optimize these first |
| >5% CPU | Hotspot | Prioritize |

**Reference:** [Profiling Tools Guide](../../plugins/systems-programming/docs/profile-performance/profiling-tools-guide.md)

## Phase 4: Hardware Counter Analysis (Standard+)

| Metric | Command | Target |
|--------|---------|--------|
| Cache misses | `perf stat -e L1-dcache-load-misses,LLC-load-misses` | <1% miss rate |
| Branch mispredictions | `perf stat -e branches,branch-misses` | <5% misprediction |
| IPC | `perf stat -e cycles,instructions` | >1.0 IPC |

**Interpretation:**
- IPC > 1.0: Good throughput
- Cache miss rate < 1%: Good locality
- Branch misprediction < 5%: Predictable branches

## Phase 5: Memory Profiling (Standard+)

| Tool | Command | Purpose |
|------|---------|---------|
| massif | `valgrind --tool=massif ./program` | Heap usage over time |
| ms_print | `ms_print massif.out.PID` | Text report |
| DHAT (Enterprise) | `valgrind --tool=dhat ./program` | Allocation analysis |

### Memory Issues

| Pattern | Indication | Fix |
|---------|------------|-----|
| Growing without decreasing | Memory leak | Track allocations |
| Frequent small allocs | Optimization opportunity | Object pooling |
| High peak usage | Temporary allocations | Reduce scope |

**Reference:** [Profiling Tools - Valgrind](../../plugins/systems-programming/docs/profile-performance/profiling-tools-guide.md#valgrind-tools)

## Phase 6: Optimization Strategy (Standard+)

### Priority by Impact

| Category | Typical Speedup | Examples |
|----------|-----------------|----------|
| Algorithm | 10-100x | O(n^2) → O(n log n) |
| Cache layout | 2-10x | SoA, alignment |
| Branch optimization | 1.5-3x | Reduce unpredictable branches |
| SIMD | 2-8x | Vectorization |
| Memory | 1.5-5x | Pooling, arena allocators |

**Reference:** [Optimization Patterns Guide](../../plugins/systems-programming/docs/profile-performance/optimization-patterns.md)

## Phase 7: Micro-Benchmarking (Enterprise)

| Language | Framework | Command |
|----------|-----------|---------|
| Rust | Criterion | `cargo bench` |
| C++ | Google Benchmark | Custom binary |
| Go | testing.B | `go test -bench=.` |
| Python | pytest-benchmark | `pytest --benchmark-only` |

### Regression Testing

```bash
# Save baseline
cargo bench -- --save-baseline baseline
# Compare after changes
cargo bench -- --baseline baseline
```

## Phase 8: Verification Workflow

### Before/After Protocol

| Step | Before | After |
|------|--------|-------|
| 1 | Profile baseline | Re-profile |
| 2 | Create benchmark suite | Re-run benchmarks |
| 3 | Run correctness tests | Verify no regressions |
| 4 | Document metrics | Measure improvement |

## Output Deliverables

| Mode | Deliverables |
|------|--------------|
| Quick | Flamegraph SVG, hot function list (>5% CPU), basic recommendations |
| Standard | + perf report, massif data, cache analysis, optimization roadmap |
| Enterprise | + benchmark suite, regression tests, CI/CD integration, full audit |

## External Documentation

| Document | Content | Lines |
|----------|---------|-------|
| [Profiling Tools Guide](../../plugins/systems-programming/docs/profile-performance/profiling-tools-guide.md) | perf, flamegraph, valgrind, language profilers | ~527 |
| [Optimization Patterns](../../plugins/systems-programming/docs/profile-performance/optimization-patterns.md) | Algorithm, cache, memory, SIMD, parallelization | ~629 |

## Profiling Checklist

### Before
- [ ] Compile with optimizations (-O2/-O3)
- [ ] Include debug symbols (-g)
- [ ] Use representative workload
- [ ] Warm up caches

### During
- [ ] Multiple runs for statistical significance
- [ ] Minimize system noise
- [ ] Profile realistic scenarios

### After
- [ ] Identify functions >5% CPU
- [ ] Check cache miss rates
- [ ] Verify IPC
- [ ] Document findings
- [ ] Re-profile after optimization

## Optimization Principles

1. **Measure First**: Always profile before optimizing
2. **Hot Paths**: Focus on >5% CPU functions
3. **Algorithm First**: Best algorithm choice = biggest impact
4. **Cache Matters**: Memory access patterns dominate
5. **Validate**: Always measure before/after
