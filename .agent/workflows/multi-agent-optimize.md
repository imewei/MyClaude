---
description: Coordinate specialized agents for comprehensive code optimization across
  scientific computing, web, and ML domains
triggers:
- /multi-agent-optimize
- coordinate specialized agents for
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<target-path> [--mode=scan|analyze|apply] [--agents=AGENTS] [--focus=AREA] [--parallel]`
The agent should parse these arguments from the user's request.

# Multi-Agent Code Optimization

Coordinate specialized agents for comprehensive optimization analysis.

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

## Target

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Output |
|------|----------|--------|
| scan | 2-5 min | Priority list of quick wins and bottlenecks |
| analyze (default) | 10-30 min | Comprehensive report with code patches |
| apply | varies | Applied patches with validation |

---

## Mode: scan (Quick Bottleneck Detection)

### Execution Steps

1. **Verify target** exists
2. **Detect stack**: NumPy, JAX, PyTorch, Julia, etc.
3. **Quick pattern analysis**: Common anti-patterns
4. **Use the `agents` skill**: Available agents analyze in parallel
5. **Generate report**: `.optimization/<target>-scan-<date>.json`

### Pattern Detection

| Pattern | Impact |
|---------|--------|
| `for.*in range` in Python | Vectorization opportunity |
| `.apply(` in pandas | 10-100x speedup possible |
| Missing `@jit` on pure functions | 5-50x speedup possible |
| Missing `@lru_cache` | Repeated call optimization |

### Scan Output

```
Quick Wins (High Impact, Low Effort):
🚀 1. Vectorize for-loop in compute_correlation() [line 145]
   → Expected: 50x speedup | Effort: 10 min | Confidence: 95%
🚀 2. Replace pandas.apply() with vectorized ops [line 203]
   → Expected: 20x speedup | Effort: 15 min | Confidence: 90%

Available Agents: 4/8
✅ multi-agent-orchestrator, systems-architect, hpc-numerical-coordinator
⚠️ scientific-computing unavailable (install jax-implementation for GPU optimizations)
```

---

## Mode: analyze (Deep Analysis)

### Execution Flow

1. **Discover agents**: Check plugin dependencies
2. **Trigger conditionals**: Based on stack detection
3. **Parallel execution**: All agents analyze simultaneously
4. **Collect results**: Stream findings as agents complete
5. **Synthesize**: Deduplicate, resolve conflicts, rank by score
6. **Generate artifacts**: Report, patches, benchmarks

### Agent Coordination (Parallel Execution)

> **Orchestration Note**: Execute analysis agents concurrently.

**Parallel Streams:**
- **Systems Architect**: Architecture, memory patterns, I/O bottlenecks
- **HPC Coordinator**: Numerical methods, parallelization, vectorization
- **JAX Pro**: GPU acceleration, XLA compilation, functional transforms

**Synthesis:**
• 3 convergent recommendations
• 2 complementary optimizations
• 1 conflict resolved (memory vs speed)
• Total expected improvement: 200-500x

---

## Mode: apply (Safe Application)

### Execution Flow

1. Load recommendations from analyze run
2. User review: Display patches, improvements, risks
3. Create backup: `.optimization/backups/`
4. Apply patches sequentially with validation
5. Run validation gates
6. Auto-rollback on failure
7. Commit on success

### Validation Gates

| Gate | Requirement |
|------|-------------|
| Tests | All pass (no regressions) |
| Performance | Improved or unchanged |
| Numerical accuracy | Within tolerance |
| Memory | Not increased >20% |

---

## Optimization Patterns

| Pattern | Speedup | Description |
|---------|---------|-------------|
| Vectorization | 10-100x | NumPy/Pandas loops → vectorized ops |
| JIT Compilation | 5-50x | JAX @jit, Numba @njit |
| Caching | 2-10x | @lru_cache for repeated calls |
| Parallelization | Nx (N=cores) | Independent iterations → parallel |
| GPU Acceleration | 10-1000x | NumPy → JAX/CuPy on GPU |

---

## Output Artifacts

| Artifact | Location |
|----------|----------|
| Scan Report | `.optimization/<target>-scan-<date>.json` |
| Analysis Report | `.optimization/<target>-report-<date>.md` |
| Code Patches | `.optimization/patches/<file>-optimized.patch` |
| Benchmarks | `.optimization/benchmarks/before-after.json` |
| Agent Logs | `.optimization/logs/<agent>-<date>.log` |

---

## Common Workflows

```bash
# 1. Quick scan (5 min)
/multi-agent-optimize src/simulation/ --mode=scan

# 2. Deep analysis with scientific agents (20 min)
/multi-agent-optimize src/simulation/ --mode=analyze --focus=scientific --parallel

# 3. Apply top 3 recommendations (10 min)
/multi-agent-optimize src/simulation/ --mode=apply --top=3

# 4. Benchmark improvements
pytest benchmarks/ --benchmark-only
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Performance improvement | ≥20% measured |
| Test suite | All pass |
| Functionality | No regressions |
| Code complexity | Not significantly increased |

---

## Best Practices

1. **Always start with scan**: Understand opportunities first
2. **Use parallel**: Maximize agent efficiency with `--parallel`
3. **Validate incrementally**: One optimization at a time for complex code
4. **Benchmark everything**: Verify claimed speedups
5. **Version control**: Commit after each successful optimization

---

## External Documentation

- `optimization-patterns.md` - Complete pattern library
- `scientific-patterns.md` - NumPy/SciPy/JAX patterns
- `ml-optimization.md` - PyTorch/TensorFlow optimization
