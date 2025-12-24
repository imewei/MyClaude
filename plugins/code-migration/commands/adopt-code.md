---
version: 1.0.5
category: code-migration
command: /adopt-code
description: Analyze and modernize scientific computing codebases while preserving numerical accuracy
argument-hint: <path-to-code> [target-framework]
color: purple
execution_modes:
  quick: "30-45 minutes"
  standard: "1-2 hours"
  comprehensive: "3-5 hours"
agents:
  primary:
    - hpc-numerical-coordinator
  conditional:
    - agent: jax-pro
      trigger: pattern "jax|flax" OR argument "jax"
    - agent: neural-architecture-engineer
      trigger: pattern "neural|torch|tensorflow"
    - agent: systems-architect
      trigger: complexity > 20 OR pattern "architecture"
  orchestrated: false
external_docs:
  - docs/code-migration/algorithm-analysis-framework.md
  - docs/code-migration/numerical-accuracy-guide.md
  - docs/code-migration/framework-migration-strategies.md
  - docs/code-migration/performance-optimization-techniques.md
  - docs/code-migration/integration-testing-patterns.md
  - docs/code-migration/scientific-computing-best-practices.md
allowed-tools: Bash(find:*), Bash(ls:*), Bash(grep:*), Bash(wc:*), Bash(du:*), Bash(head:*), Bash(tail:*), Bash(file:*)
---

# Scientific Code Adoption & Modernization

Target: $ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 30-45 min | Codebase analysis + migration strategy |
| Standard (default) | 1-2h | + framework selection + initial implementation |
| Comprehensive | 3-5h | + validation, benchmarking, production integration |

---

## Initial Codebase Discovery

### Automated Analysis Commands

| Analysis | Command Pattern |
|----------|-----------------|
| Project files | `find -type f -name "*.{f90,c,cpp,py,jl}"` |
| LOC count | `find ... -exec wc -l` |
| Language detection | Count by extension |
| Build system | Find Makefile, CMake, setup.py |
| Dependencies | Grep includes/imports |
| Parallelization | Grep MPI, OpenMP, CUDA |
| Numerical libs | Grep BLAS, LAPACK, FFT |

---

## Phase 1: Code Understanding & Architecture

### Algorithm Identification

| Category | Examples |
|----------|----------|
| Solvers | Iterative methods, direct solvers |
| Integration | Time stepping, discretization |
| Optimization | Gradient descent, Newton methods |

### Computational Kernel Analysis

| Metric | Purpose |
|--------|---------|
| Hotspots | 80/20 rule profiling |
| Bound type | Memory, compute, or I/O |
| Complexity | O(N), O(N^2), O(N^3) |
| Access pattern | Contiguous, strided, random |

### Data Structure Analysis

| Aspect | Consideration |
|--------|---------------|
| Memory layout | Row-major vs column-major |
| Precision | float32, float64, float128 |
| Sparsity | Dense vs sparse structures |

**Reference:** [algorithm-analysis-framework.md](../docs/code-migration/algorithm-analysis-framework.md)

---

## Phase 2: Numerical Accuracy Preservation

### Precision Requirements

| Domain | Typical Precision |
|--------|-------------------|
| Physics simulations | float64 |
| Machine learning | float32 |
| Financial | Decimal/float128 |

### Verification Strategy

| Test Type | Purpose |
|-----------|---------|
| Reference solutions | Analytical, higher-precision, legacy |
| Tolerance criteria | Absolute, relative, mixed |
| Test hierarchy | Unit, integration, system |

### Reproducibility

| Issue | Mitigation |
|-------|------------|
| FP associativity | Deterministic reductions |
| Parallel reductions | Ordered operations |
| RNG | Seeded, reproducible |

**Reference:** [numerical-accuracy-guide.md](../docs/code-migration/numerical-accuracy-guide.md)

---

## Phase 3: Framework Migration Strategy

### Target Framework Selection

| Framework | Use Case | Strengths |
|-----------|----------|-----------|
| NumPy/SciPy | General scientific | Low learning curve |
| JAX | GPU/TPU, autodiff | Transformations, XLA |
| Julia | High-performance | Composability, speed |
| PyTorch/TensorFlow | ML-enhanced | Deep learning integration |
| Rust/C++20 | Systems-level | Safety + performance |
| Dask/Chapel | Distributed | Parallel computing |

### Migration Approach

| Approach | Timeline | Risk |
|----------|----------|------|
| Phased (wrapper-first) | 6-12 weeks | Lower |
| Direct (rewrite) | 3-6 weeks | Higher |

**Reference:** [framework-migration-strategies.md](../docs/code-migration/framework-migration-strategies.md)

---

## Phase 4: Performance Optimization

### Parallelization Opportunities

| Technique | Speedup | Use Case |
|-----------|---------|----------|
| Vectorization | 10-100x | Broadcasting, vmap |
| Multi-threading | 4-8x | Data parallelism |
| GPU acceleration | 10-1000x | JAX jit, CUDA |
| Distributed | 10-100x | Cluster computing |

### Optimization Priorities

| Optimization | Impact |
|--------------|--------|
| Algorithm complexity | Highest (O(N^2)→O(N log N)) |
| Cache optimization | High (loop tiling) |
| Memory reduction | Medium (pre-allocation) |
| JIT compilation | High (10-50x) |

**Reference:** [performance-optimization-techniques.md](../docs/code-migration/performance-optimization-techniques.md)

---

## Phase 5: Integration & Ecosystem

### Modern Tooling

| Tool | Purpose |
|------|---------|
| Version control | Detailed commits, migration branches |
| CI | Numerical validation in GitHub Actions |
| Documentation | Sphinx with NumPy-style docstrings |
| Benchmarking | pytest-benchmark, custom suites |

### Package Structure

| Component | Standard |
|-----------|----------|
| Build config | pyproject.toml, src/ layout |
| Dependencies | conda environment.yml |
| Distribution | Wheels, conda packages |

**Reference:** [integration-testing-patterns.md](../docs/code-migration/integration-testing-patterns.md)

---

## Phase 6: Validation & Benchmarking

### Numerical Validation

| Test Type | Purpose |
|-----------|---------|
| Regression | Compare to reference outputs |
| Convergence | Verify expected rates |
| Property-based | Numerical properties |
| Edge cases | Boundaries, singular cases |

### Performance Benchmarking

| Metric | Comparison |
|--------|------------|
| Speed | New vs old |
| Scaling | Weak and strong |
| Memory | Peak, allocation patterns |
| Platforms | Linux, macOS, Windows |

---

## Deliverables

| Deliverable | Content |
|-------------|---------|
| Migration Plan | Architecture analysis, timeline |
| Code Translation | Side-by-side comparison |
| Validation Report | Accuracy (<1e-11), performance |
| Documentation | API, algorithms, tutorials |
| Test Suite | 95%+ coverage, numerical validation |

---

## Special Considerations

| Category | Requirement |
|----------|-------------|
| Numerical stability | Never change algorithms without validation |
| Performance | Maintain or exceed legacy |
| Conservation laws | Energy, momentum, mass |
| Legacy compatibility | Wrapper layer if needed |

**Reference:** [scientific-computing-best-practices.md](../docs/code-migration/scientific-computing-best-practices.md)

---

## Action Items

1. Analyze codebase comprehensively
2. Identify core algorithms and numerical properties
3. Recommend optimal framework(s)
4. Create detailed migration plan
5. Implement proof-of-concept for critical kernel
6. Validate numerical accuracy
7. Benchmark performance
8. Document findings and recommendations
