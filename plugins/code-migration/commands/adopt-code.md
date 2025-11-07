---
version: "1.0.3"
category: "code-migration"
command: "/adopt-code"
description: Analyze, integrate, and optimize scientific computing codebases for modern frameworks while preserving numerical accuracy and computational efficiency.
argument-hint: <path-to-code> [target-framework]
color: purple

execution_modes:
  quick: "30-45 minutes - Essential codebase analysis and migration strategy only"
  standard: "1-2 hours - Comprehensive analysis with framework selection and initial implementation (default)"
  comprehensive: "3-5 hours - Full migration workflow with validation, benchmarking, and production integration"

agents:
  primary:
    - hpc-numerical-coordinator
  conditional:
    - agent: jax-pro
      trigger: pattern "jax|flax" OR argument "jax"
    - agent: neural-architecture-engineer
      trigger: pattern "neural|torch|tensorflow" OR files "*.h5|*.pt"
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

## Initial Codebase Discovery

### Code Structure Analysis
- Project files: !`find $ARGUMENTS -type f -name "*.f90" -o -name "*.f" -o -name "*.c" -o -name "*.cpp" -o -name "*.m" -o -name "*.py" -o -name "*.jl" -o -name "*.R" 2>/dev/null | head -50`
- Total lines of code: !`find $ARGUMENTS -type f \( -name "*.f90" -o -name "*.f" -o -name "*.c" -o -name "*.cpp" -o -name "*.py" -o -name "*.jl" \) -exec wc -l {} + 2>/dev/null | tail -1`
- Project size: !`du -sh $ARGUMENTS 2>/dev/null`

### Language & Framework Detection
- Fortran files: !`find $ARGUMENTS -name "*.f90" -o -name "*.f" -o -name "*.for" 2>/dev/null | wc -l`
- C/C++ files: !`find $ARGUMENTS -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" 2>/dev/null | wc -l`
- MATLAB files: !`find $ARGUMENTS -name "*.m" 2>/dev/null | wc -l`
- Python files: !`find $ARGUMENTS -name "*.py" 2>/dev/null | wc -l`
- Julia files: !`find $ARGUMENTS -name "*.jl" 2>/dev/null | wc -l`

### Build System Analysis
- Makefiles: !`find $ARGUMENTS -name "Makefile" -o -name "*.mk" 2>/dev/null`
- CMake files: !`find $ARGUMENTS -name "CMakeLists.txt" -o -name "*.cmake" 2>/dev/null`
- Setup scripts: !`find $ARGUMENTS -name "setup.py" -o -name "setup.cfg" -o -name "pyproject.toml" 2>/dev/null`

### Dependency & Library Analysis
- Include statements: !`grep -r "^#include\|^use \|^import \|^from \|^using " $ARGUMENTS --include="*.{c,cpp,h,hpp,f90,py,jl}" 2>/dev/null | head -30`
- External libraries: !`grep -r "link.*lib\|LIBS.*=\|-l[a-z]" $ARGUMENTS --include="Makefile" --include="*.cmake" 2>/dev/null | head -20`

### Parallelization Patterns
- MPI usage: !`grep -r "MPI_\|mpif90\|mpicc\|mpirun" $ARGUMENTS --include="*.{c,cpp,f90,py}" 2>/dev/null | wc -l`
- OpenMP usage: !`grep -r "omp\|!$OMP\|#pragma omp" $ARGUMENTS --include="*.{c,cpp,f90}" 2>/dev/null | wc -l`
- CUDA/GPU code: !`grep -r "__global__\|__device__\|cudaMalloc\|\.cu$" $ARGUMENTS 2>/dev/null | wc -l`

### Numerical Computing Patterns
- BLAS/LAPACK: !`grep -ri "dgemm\|dgesv\|dsyev\|lapack\|blas" $ARGUMENTS --include="*.{c,cpp,f90,f}" 2>/dev/null | wc -l`
- FFT operations: !`grep -ri "fft\|fftw" $ARGUMENTS --include="*.{c,cpp,f90,py,m}" 2>/dev/null | wc -l`

### Documentation & Testing
- Documentation files: !`find $ARGUMENTS -name "README*" -o -name "*.md" -o -name "*.rst" 2>/dev/null`
- Test files: !`find $ARGUMENTS -name "*test*" -o -name "*spec*" 2>/dev/null | head -20`

### Key Source Files
@$ARGUMENTS

---

## Your Task: Scientific Code Adoption & Modernization

Based on the discovered codebase, perform a comprehensive adoption analysis following these six phases:

---

## Phase 1: Code Understanding & Architecture Analysis

**1.1 Algorithm Identification**
- Identify and document all numerical algorithms used (iterative methods, direct solvers, time integration, discretization, optimization)
- Analyze computational kernels and performance-critical sections
- Document data structure organization and memory layout

**Detailed guidance**: See [algorithm-analysis-framework.md](../docs/code-migration/algorithm-analysis-framework.md)

**1.2 Computational Kernel Analysis**
- Profile code to identify hotspots (80/20 rule)
- Classify kernels as memory-bound, compute-bound, or I/O-bound
- Calculate loop complexity (O(N), O(N²), O(N³))
- Analyze memory access patterns (contiguous, strided, random)

**1.3 Data Structure Analysis**
- Document memory layout (row-major vs. column-major)
- Identify precision requirements (float32, float64, float128)
- Calculate memory footprints and identify sparse vs. dense structures

---

## Phase 2: Numerical Accuracy Preservation

**2.1 Precision Requirements**
- Determine required precision levels based on domain (physics: float64, ML: float32)
- Analyze error propagation and accumulation in iterative algorithms
- Identify catastrophic cancellation risks

**Detailed guidance**: See [numerical-accuracy-guide.md](../docs/code-migration/numerical-accuracy-guide.md)

**2.2 Verification Strategy**
- Extract or create reference solutions (analytical, higher-precision, legacy outputs)
- Define tolerance criteria (absolute error, relative error, mixed)
- Build test case hierarchy (unit tests, integration tests, system tests)

**2.3 Reproducibility Analysis**
- Identify non-deterministic operations (floating-point associativity, parallel reductions, RNG)
- Design cross-platform validation strategy
- Document conservation law validation (energy, mass, momentum)

---

## Phase 3: Modern Framework Migration Strategy

**3.1 Target Framework Selection**
Use the framework selection matrix based on code characteristics:
- **NumPy/SciPy**: General scientific computing (CPU, low learning curve)
- **JAX**: GPU/TPU acceleration, auto-differentiation (medium learning curve)
- **Julia**: High-performance, composability (excellent CPU/GPU performance)
- **PyTorch/TensorFlow**: ML-enhanced numerical computing
- **Rust/C++20**: Systems-level performance with safety
- **Dask/Chapel**: Distributed parallel computing

**Detailed guidance**: See [framework-migration-strategies.md](../docs/code-migration/framework-migration-strategies.md)

**3.2 Migration Roadmap**
Choose phased or direct migration approach:
- **Phased** (wrapper-first): F2py/Ctypes → Pure NumPy → JAX optimization (6-12 weeks)
- **Direct** (rewrite): Algorithm analysis → Implementation → Validation (3-6 weeks)

**3.3 Dependency Modernization**
- Map legacy libraries to modern equivalents (BLAS/LAPACK → NumPy/JAX, Fortran I/O → HDF5)
- Identify deprecated functionality and plan for missing features

---

## Phase 4: Performance Optimization Strategy

**4.1 Parallelization Opportunities**
- **Vectorization**: NumPy broadcasting, JAX vmap (10-100x speedup)
- **Multi-threading**: Data parallelism for CPU (4-8x on 8-core)
- **GPU acceleration**: JAX jit/vmap, CUDA kernels (10-1000x speedup)
- **Distributed computing**: Dask, MPI alternatives (10-100x on cluster)

**Detailed guidance**: See [performance-optimization-techniques.md](../docs/code-migration/performance-optimization-techniques.md)

**4.2 Computational Efficiency**
- Reduce algorithm complexity (O(N²) → O(N log N) via FFT)
- Optimize cache usage (loop blocking/tiling)
- Minimize memory allocations (pre-allocate, in-place operations)
- Apply JIT compilation (JAX, Numba for 10-50x speedup)

**4.3 Hardware Utilization**
- CPU architecture considerations (AVX-512 SIMD, ARM NEON)
- GPU utilization strategy (kernel candidates, memory hierarchy)
- Memory hierarchy optimization (L1/L2/L3 cache, RAM, GPU VRAM)

---

## Phase 5: Integration & Ecosystem

**5.1 Modern Tooling Integration**
- Version control best practices (detailed commit messages, migration branches)
- Continuous integration setup (GitHub Actions for numerical validation)
- Documentation generation (Sphinx with NumPy-style docstrings)
- Performance benchmarking suite (pytest-benchmark, custom frameworks)

**Detailed guidance**: See [integration-testing-patterns.md](../docs/code-migration/integration-testing-patterns.md)

**5.2 Package Management**
- Package structure (modern pyproject.toml, src/ layout)
- Dependency specification (conda environment.yml, pip requirements.txt)
- Version pinning strategy (strict for production, flexible for development)
- Binary distribution (wheels, conda packages)

**5.3 API Design**
- Functional API for scientific computing (pure functions, no side effects)
- Type hints and contracts (Python 3.12+ type annotations)
- Error handling and validation (informative exceptions, input checking)
- Configuration management

---

## Phase 6: Validation & Benchmarking

**6.1 Numerical Validation Suite**
- **Regression tests**: Compare against Fortran/MATLAB reference outputs
- **Convergence tests**: Verify expected convergence rates
- **Property-based tests**: Test numerical properties (commutativity, conservation)
- **Edge cases**: Boundary conditions, singular cases, extreme values

**6.2 Performance Benchmarking**
- Speed comparison (new vs. old implementation)
- Scaling analysis (weak and strong scaling)
- Memory profiling (peak usage, allocation patterns)
- Benchmark across platforms (Linux, macOS, Windows)

**6.3 Reproducibility Verification**
- Cross-platform testing (identical results on different OSes)
- Precision testing (different hardware platforms)
- Determinism testing (repeated runs produce identical outputs)

---

## Deliverables

### 1. Migration Plan Document
- Detailed architecture analysis
- Risk assessment and mitigation
- Phase-by-phase implementation plan
- Resource requirements and timeline estimates

### 2. Code Translation
- Modernized implementation with side-by-side comparison
- Comments explaining numerical considerations
- Performance-critical sections highlighted

### 3. Validation Report
- Numerical accuracy verification (< 1e-11 relative error typical)
- Performance benchmarks (target: maintain or exceed legacy performance)
- Reproducibility confirmation
- Known limitations

### 4. Documentation Package
- API documentation (Sphinx-generated)
- Algorithm documentation with mathematical details
- Usage examples and tutorials
- Migration guide from legacy code

### 5. Test Suite
- Unit tests (95%+ code coverage)
- Integration tests
- Numerical validation tests
- Performance regression tests

---

## Special Considerations

**Detailed guidance**: See [scientific-computing-best-practices.md](../docs/code-migration/scientific-computing-best-practices.md)

### Numerical Stability
⚠️ **Never** change numerical algorithms without validation
- Analyze condition numbers for ill-conditioned systems
- Use stable algorithm variants (QR vs. normal equations for least squares)
- Document any approximations or algorithmic changes
- Consider mixed-precision opportunities

### Performance Sensitivity
- Profile before optimizing (avoid premature optimization)
- Preserve algorithmic complexity
- Document performance trade-offs
- Target: maintain or exceed legacy performance

### Domain-Specific Requirements
- Physics conservation laws (energy, momentum, mass)
- Boundary condition handling (Dirichlet, Neumann, periodic)
- Symmetry preservation
- Units and dimensional analysis

### Legacy Compatibility
- Provide compatibility layer if needed (wrapper for gradual migration)
- Document breaking changes with migration examples
- Offer migration utilities (automated converters where possible)
- Support gradual adoption (phased migration timeline)

---

## Action Items

Now, execute the following steps:

1. **Analyze** the codebase comprehensively using the data collected above
2. **Identify** the core algorithms and their numerical properties (see algorithm-analysis-framework.md)
3. **Recommend** the optimal modern framework(s) for migration (see framework-migration-strategies.md)
4. **Create** a detailed migration plan with specific code examples
5. **Implement** a proof-of-concept for the most critical computational kernel
6. **Validate** numerical accuracy against original implementation (see numerical-accuracy-guide.md)
7. **Benchmark** performance comparison (see performance-optimization-techniques.md)
8. **Document** all findings, decisions, and recommendations

Focus on preserving numerical correctness while achieving modern software engineering standards and computational efficiency.

---

**Execution Mode Guidelines**:

- **Quick** (30-45 min): Codebase discovery + migration strategy recommendation only
- **Standard** (1-2 hours): Analysis + framework selection + initial implementation (default)
- **Comprehensive** (3-5 hours): Full workflow with validation, benchmarking, and production integration
