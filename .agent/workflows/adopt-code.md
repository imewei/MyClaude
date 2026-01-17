---
description: Analyze and modernize scientific computing codebases while preserving
  numerical accuracy
triggers:
- /adopt-code
- analyze and modernize scientific
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<path-to-code> [target-framework]`
The agent should parse these arguments from the user's request.

# Scientific Code Adoption

$ARGUMENTS

## Codebase Discovery

| Analysis | Command |
|----------|---------|
| Project files | `find -type f -name "*.{f90,c,cpp,py,jl}"` |
| LOC count | `find ... -exec wc -l` |
| Language | Count by extension |
| Build system | Find Makefile, CMake, setup.py |
| Dependencies | Grep includes/imports |
| Parallelization | Grep MPI, OpenMP, CUDA |
| Numerical libs | Grep BLAS, LAPACK, FFT |

## Phase 1: Architecture

**Algorithms:** Solvers (iterative, direct), Integration (time stepping, discretization), Optimization (gradient descent, Newton)

**Computational kernels:** Hotspots (80/20 profiling), Bound type (memory/compute/I/O), Complexity (O(N), O(N²), O(N³)), Access pattern (contiguous/strided/random)

**Data structures:** Memory layout (row/column-major), Precision (float32/64/128), Sparsity (dense/sparse)

**Ref:** [algorithm-analysis-framework.md](../../plugins/code-migration/docs/code-migration/algorithm-analysis-framework.md)

## Phase 2: Numerical Accuracy

**Precision:**
- Physics: float64
- ML: float32
- Financial: Decimal/float128

**Verification:** Reference solutions (analytical, higher-precision, legacy), Tolerance (absolute/relative/mixed), Test hierarchy (unit/integration/system)

**Reproducibility:** FP associativity → Deterministic reductions, Parallel reductions → Ordered operations, RNG → Seeded

**Ref:** [numerical-accuracy-guide.md](../../plugins/code-migration/docs/code-migration/numerical-accuracy-guide.md)

## Phase 3: Framework Migration

**Target selection:**

| Framework | Use Case | Strengths |
|-----------|----------|-----------|
| NumPy/SciPy | General scientific | Low learning curve |
| JAX | GPU/TPU, autodiff | Transformations, XLA |
| Julia | High-performance | Composability, speed |
| PyTorch/TensorFlow | ML-enhanced | Deep learning integration |
| Rust/C++20 | Systems-level | Safety + performance |
| Dask/Chapel | Distributed | Parallel computing |

**Approach:**
- Phased (wrapper-first): 6-12w, lower risk
- Direct (rewrite): 3-6w, higher risk

**Ref:** [framework-migration-strategies.md](../../plugins/code-migration/docs/code-migration/framework-migration-strategies.md)

## Phase 4: Performance

**Parallelization:**
- Vectorization: 10-100x (broadcasting, vmap)
- Multi-threading: 4-8x (data parallelism)
- GPU: 10-1000x (JAX jit, CUDA)
- Distributed: 10-100x (cluster)

**Priorities:** Algorithm complexity (highest, O(N²)→O(N log N)), Cache optimization (high, loop tiling), Memory reduction (medium, pre-allocation), JIT compilation (high, 10-50x)

**Ref:** [performance-optimization-techniques.md](../../plugins/code-migration/docs/code-migration/performance-optimization-techniques.md)

## Phase 5: Integration

**Tooling:** Version control (detailed commits), CI (numerical validation in GHA), Documentation (Sphinx + NumPy-style docstrings), Benchmarking (pytest-benchmark, custom)

**Package:** Build config (pyproject.toml, src/ layout), Dependencies (conda environment.yml), Distribution (wheels, conda packages)

**Ref:** [integration-testing-patterns.md](../../plugins/code-migration/docs/code-migration/integration-testing-patterns.md)

## Phase 6: Validation

**Numerical:** Regression (vs reference), Convergence (verify rates), Property-based (numerical properties), Edge cases (boundaries, singular)

**Performance:** Speed (new vs old), Scaling (weak/strong), Memory (peak, allocation), Platforms (Linux, macOS, Windows)

## Deliverables

- Migration Plan: Architecture analysis, timeline
- Code Translation: Side-by-side comparison
- Validation Report: Accuracy (<1e-11), performance
- Documentation: API, algorithms, tutorials
- Test Suite: 95%+ coverage, numerical validation

## Special Considerations

- Numerical stability: Never change algorithms without validation
- Performance: Maintain or exceed legacy
- Conservation laws: Energy, momentum, mass
- Legacy compatibility: Wrapper layer if needed

**Ref:** [scientific-computing-best-practices.md](../../plugins/code-migration/docs/code-migration/scientific-computing-best-practices.md)

## Action Items

1. Analyze codebase
2. Identify algorithms and numerical properties
3. Recommend optimal frameworks
4. Create migration plan
5. Implement proof-of-concept for critical kernel
6. Validate numerical accuracy
7. Benchmark performance
8. Document findings
