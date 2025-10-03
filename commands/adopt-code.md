---
description: Analyze, integrate, and optimize scientific computing codebases for modern frameworks while preserving numerical accuracy and computational efficiency.
allowed-tools: Bash(find:*), Bash(ls:*), Bash(grep:*), Bash(wc:*), Bash(du:*), Bash(head:*), Bash(tail:*), Bash(file:*)
argument-hint: <path-to-code> [target-framework]
color: purple
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
- R files: !`find $ARGUMENTS -name "*.R" 2>/dev/null | wc -l`

### Build System Analysis
- Makefiles: !`find $ARGUMENTS -name "Makefile" -o -name "*.mk" 2>/dev/null`
- CMake files: !`find $ARGUMENTS -name "CMakeLists.txt" -o -name "*.cmake" 2>/dev/null`
- Setup scripts: !`find $ARGUMENTS -name "setup.py" -o -name "setup.cfg" -o -name "pyproject.toml" 2>/dev/null`
- Build scripts: !`find $ARGUMENTS -name "build.*" -o -name "compile.*" 2>/dev/null`

### Dependency & Library Analysis
- Include statements: !`grep -r "^#include\|^use \|^import \|^from \|^using " $ARGUMENTS --include="*.{c,cpp,h,hpp,f90,py,jl}" 2>/dev/null | head -30`
- External libraries: !`grep -r "link.*lib\|LIBS.*=\|-l[a-z]" $ARGUMENTS --include="Makefile" --include="*.cmake" 2>/dev/null | head -20`
- Package dependencies: @$ARGUMENTS/requirements.txt
- Conda environment: @$ARGUMENTS/environment.yml

### Parallelization Patterns
- MPI usage: !`grep -r "MPI_\|mpif90\|mpicc\|mpirun" $ARGUMENTS --include="*.{c,cpp,f90,py}" 2>/dev/null | wc -l`
- OpenMP usage: !`grep -r "omp\|!$OMP\|#pragma omp" $ARGUMENTS --include="*.{c,cpp,f90}" 2>/dev/null | wc -l`
- CUDA/GPU code: !`grep -r "__global__\|__device__\|cudaMalloc\|\.cu$" $ARGUMENTS 2>/dev/null | wc -l`
- Threading: !`grep -r "pthread\|std::thread\|@threads" $ARGUMENTS --include="*.{c,cpp,py,jl}" 2>/dev/null | wc -l`

### Numerical Computing Patterns
- BLAS/LAPACK: !`grep -ri "dgemm\|dgesv\|dsyev\|lapack\|blas" $ARGUMENTS --include="*.{c,cpp,f90,f}" 2>/dev/null | wc -l`
- FFT operations: !`grep -ri "fft\|fftw" $ARGUMENTS --include="*.{c,cpp,f90,py,m}" 2>/dev/null | wc -l`
- Linear solvers: !`grep -ri "solver\|jacobi\|gauss.*seidel\|conjugate.*gradient" $ARGUMENTS 2>/dev/null | head -20`
- Numerical integration: !`grep -ri "integrate\|quadrature\|trapz\|simpson" $ARGUMENTS 2>/dev/null | head -20`

### Documentation & Testing
- Documentation files: !`find $ARGUMENTS -name "README*" -o -name "*.md" -o -name "*.rst" 2>/dev/null`
- Test files: !`find $ARGUMENTS -name "*test*" -o -name "*spec*" 2>/dev/null | head -20`
- Verification data: !`find $ARGUMENTS -name "*benchmark*" -o -name "*validation*" -o -name "*reference*" 2>/dev/null | head -15`

## Core Analysis Content

### Key Source Files
@$ARGUMENTS

## Your Task: Scientific Code Adoption & Modernization

Based on the discovered codebase, perform a comprehensive adoption analysis following these phases:

---

## Phase 1: Code Understanding & Architecture Analysis

### 1.1 Algorithm Identification
- **Core Algorithms**: Identify and document all numerical algorithms used
  - Iterative methods (convergence criteria, stability)
  - Direct solvers (matrix decompositions, numerical stability)
  - Time integration schemes (explicit/implicit, order of accuracy)
  - Discretization methods (FDM, FEM, FVM, spectral methods)
  - Optimization algorithms (gradient-based, evolutionary, etc.)

### 1.2 Computational Kernel Analysis
- **Performance-Critical Sections**: Identify hotspots and bottlenecks
  - Loop structures and complexity analysis
  - Memory access patterns (cache efficiency)
  - I/O operations and data movement
  - Communication patterns in parallel code

### 1.3 Data Structure Analysis
- **Memory Layout**: Document data organization
  - Array ordering (row-major vs column-major)
  - Strided vs contiguous memory access
  - Data structure definitions and sizes
  - Precision requirements (single, double, extended)

---

## Phase 2: Numerical Accuracy Preservation

### 2.1 Precision Requirements
- **Floating-Point Analysis**:
  - Document required precision levels
  - Identify catastrophic cancellation risks
  - Analyze accumulation errors in loops
  - Check for numerical stability issues

### 2.2 Verification Strategy
- **Accuracy Testing Plan**:
  - Extract or create reference solutions
  - Define tolerance criteria (absolute/relative error)
  - Identify analytical test cases
  - Document expected convergence rates

### 2.3 Reproducibility Analysis
- **Determinism Requirements**:
  - Identify non-deterministic operations
  - Document random number generation
  - Analyze reduction operation ordering
  - Check for race conditions

---

## Phase 3: Modern Framework Migration Strategy

### 3.1 Target Framework Selection
Based on the code characteristics, recommend:
- **NumPy/SciPy** (Python): General scientific computing
- **JAX** (Python): Auto-differentiation, GPU/TPU acceleration
- **Julia**: High-performance scientific computing, composability
- **PyTorch/TensorFlow**: ML-enhanced numerical computing
- **Rust/C++20**: Systems-level performance with safety
- **Chapel/Dask**: Distributed parallel computing

### 3.2 Migration Roadmap
Create a phased migration plan:
1. **Phase 1 - Core Algorithms**: Translate numerical kernels
2. **Phase 2 - Data Pipeline**: Modernize I/O and data handling
3. **Phase 3 - Parallelization**: Implement modern parallel patterns
4. **Phase 4 - Integration**: Connect with modern ecosystem
5. **Phase 5 - Optimization**: Performance tuning and profiling

### 3.3 Dependency Modernization
- **Library Mapping**:
  - Map legacy libraries to modern equivalents
  - Identify deprecated functionality
  - Plan for missing features
  - Document compatibility layers needed

---

## Phase 4: Performance Optimization Strategy

### 4.1 Parallelization Opportunities
- **Modern Parallelism**:
  - Vectorization opportunities (SIMD)
  - Multi-threading strategies (data vs task parallelism)
  - GPU acceleration potential (kernel candidates)
  - Distributed computing needs (MPI alternatives)

### 4.2 Computational Efficiency
- **Optimization Targets**:
  - Algorithm complexity reduction
  - Cache optimization strategies
  - Memory allocation optimization
  - Lazy evaluation opportunities
  - JIT compilation benefits

### 4.3 Hardware Utilization
- **Platform Optimization**:
  - CPU architecture considerations (AVX-512, ARM NEON)
  - GPU utilization strategy (CUDA, ROCm, Metal)
  - TPU opportunities (for JAX/TensorFlow)
  - Memory hierarchy optimization

---

## Phase 5: Integration & Ecosystem

### 5.1 Modern Tooling Integration
- **Development Tools**:
  - Version control best practices
  - Continuous integration setup
  - Automated testing framework
  - Documentation generation
  - Performance benchmarking suite

### 5.2 Package Management
- **Distribution Strategy**:
  - Package structure (pip/conda/cargo/Julia registry)
  - Dependency specification
  - Version pinning strategy
  - Binary distribution (wheels, conda packages)

### 5.3 API Design
- **User Interface**:
  - Modern API patterns (fluent, functional)
  - Type hints and contracts
  - Error handling and validation
  - Configuration management
  - Logging and diagnostics

---

## Phase 6: Validation & Benchmarking

### 6.1 Numerical Validation Suite
Create comprehensive tests:
- **Accuracy Tests**: Compare against reference solutions
- **Convergence Tests**: Verify convergence rates
- **Edge Cases**: Boundary conditions, singular cases
- **Regression Tests**: Guard against future breakage

### 6.2 Performance Benchmarking
Establish baselines:
- **Speed Comparison**: New vs old implementation
- **Scaling Analysis**: Weak and strong scaling
- **Memory Profiling**: Peak usage and allocation patterns
- **Energy Efficiency**: Operations per joule (if applicable)

### 6.3 Reproducibility Verification
- **Cross-Platform Testing**: Linux, macOS, Windows
- **Precision Testing**: Different hardware platforms
- **Determinism Testing**: Repeated runs consistency

---

## Deliverables

### 1. Migration Plan Document
- Detailed architecture analysis
- Risk assessment and mitigation
- Phase-by-phase implementation plan
- Resource requirements
- Timeline estimates

### 2. Code Translation
- Modernized implementation
- Side-by-side comparison with original
- Comments explaining numerical considerations
- Performance-critical sections highlighted

### 3. Validation Report
- Numerical accuracy verification
- Performance benchmarks
- Reproducibility confirmation
- Known limitations

### 4. Documentation Package
- API documentation
- Algorithm documentation with mathematical details
- Usage examples and tutorials
- Migration guide from legacy code

### 5. Test Suite
- Unit tests for all components
- Integration tests
- Numerical validation tests
- Performance regression tests

---

## Special Considerations for Scientific Computing

### Numerical Stability
- ⚠️ **Never** change numerical algorithms without validation
- Document any approximations or algorithmic changes
- Preserve bit-exact reproducibility if required
- Consider condition numbers and error propagation

### Performance Sensitivity
- Profile before optimizing
- Preserve algorithmic complexity
- Document performance trade-offs
- Consider mixed-precision opportunities

### Domain-Specific Requirements
- Physics conservation laws (energy, momentum, mass)
- Boundary condition handling
- Symmetry preservation
- Units and dimensional analysis

### Legacy Compatibility
- Provide compatibility layer if needed
- Document breaking changes
- Offer migration utilities
- Support gradual adoption

---

## Action Items

Now, execute the following steps:

1. **Analyze** the codebase comprehensively using the data collected above
2. **Identify** the core algorithms and their numerical properties
3. **Recommend** the optimal modern framework(s) for migration
4. **Create** a detailed migration plan with specific code examples
5. **Implement** a proof-of-concept for the most critical computational kernel
6. **Validate** numerical accuracy against original implementation
7. **Benchmark** performance comparison
8. **Document** all findings, decisions, and recommendations

Focus on preserving numerical correctness while achieving modern software engineering standards and computational efficiency.
