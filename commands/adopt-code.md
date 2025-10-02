---
title: "Adopt Code"
description: "Analyze, integrate, and optimize scientific computing codebases for modern frameworks"
category: scientific-computing
subcategory: legacy-modernization
complexity: expert
argument-hint: "[--analyze] [--integrate] [--optimize] [--language=fortran|c|cpp|python|julia|mixed] [--target=python|jax|julia] [--parallel=mpi|openmp|cuda|jax] [--agents=scientific|quality|orchestrator|all] [--dry-run] [--backup] [--rollback] [--validate] [codebase-path]"
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, TodoWrite
model: inherit
tags: scientific-computing, legacy-code, modernization, optimization, numerical-accuracy
dependencies: []
related: [optimize, jax-essentials, julia-jit-like, debug, check-code-quality, refactor-clean, multi-agent-optimize, generate-tests, run-all-tests]
workflows: [legacy-modernization, scientific-migration, performance-optimization]
version: "3.0"
last-updated: "2025-09-29"
executor: adopt_code_executor.py
---

# Adopt Code

Analyze, integrate, and optimize scientific computing codebases for modern frameworks while preserving numerical accuracy and computational efficiency.

## Quick Start

```bash
# Analyze legacy codebase
/adopt-code ./legacy_fortran --analyze

# Integrate Fortran with Python
/adopt-code ./fortran_solver --integrate --language=fortran --target=python

# Optimize for GPU acceleration
/adopt-code ./md_simulation --optimize --parallel=cuda

# Complete modernization pipeline
/adopt-code ./scientific_code --analyze --integrate --optimize --target=jax --agents=all
```

## Usage

```bash
/adopt-code [options] [codebase-path]
```

**Parameters:**
- `options` - Analysis, integration, and optimization configuration
- `codebase-path` - Path to legacy codebase (directory or file)

## Options

### Operation Modes

- **`--analyze`** - Perform comprehensive codebase analysis
  - Algorithm and data structure identification
  - Dependency mapping and library detection
  - Performance bottleneck analysis
  - Numerical method classification
  - Modernization feasibility assessment

- **`--integrate`** - Enable cross-language integration
  - Python/JAX/Julia wrapper generation
  - Foreign function interface (FFI) setup
  - Data marshaling and type conversion
  - Error handling and validation
  - API design and documentation

- **`--optimize`** - Apply performance optimizations
  - Algorithmic improvements
  - Vectorization and SIMD
  - GPU acceleration porting
  - Memory layout optimization
  - Parallel scalability enhancement

### Language and Target Options

- **`--language=<lang>`** - Source language
  - `fortran` - Fortran 77/90/95/2003/2008
  - `c` - C89/C99/C11/C17
  - `cpp` - C++98/11/14/17/20
  - `python` - Legacy Python 2.x
  - `julia` - Julia 0.x/1.x
  - `mixed` - Multi-language codebase

- **`--target=<framework>`** - Target framework
  - `python` - Modern Python 3.x with NumPy/SciPy
  - `jax` - JAX ecosystem with GPU acceleration
  - `julia` - Julia with SciML ecosystem

- **`--parallel=<strategy>`** - Parallelization approach
  - `mpi` - MPI (Message Passing Interface)
  - `openmp` - OpenMP threading
  - `cuda` - NVIDIA CUDA GPU
  - `jax` - JAX parallelization (pmap, vmap)

### Multi-Agent Options

- **`--agents=<selection>`** - Agent team selection
  - `scientific` - Scientific computing specialists
  - `quality` - Code quality and testing experts
  - `orchestrator` - Multi-agent coordination
  - `all` - Complete 23-agent system

## Executor Architecture

The adopt-code command uses a sophisticated executor that leverages shared utilities for comprehensive code modernization:

### Executor Components

```python
class AdoptCodeExecutor(CommandExecutor):
    """Executor for legacy code adoption and modernization"""

    def __init__(self):
        super().__init__("adopt-code")

        # Shared utilities
        self.code_modifier = CodeModifier()
        self.ast_analyzer = CodeAnalyzer()
        self.test_runner = TestRunner()
        self.git = GitUtils()

        # Agent orchestration
        self.orchestrator = AgentOrchestrator()

    def execute(self, args):
        # Phase 1: Analysis
        if args.get('analyze'):
            analysis = self.analyze_codebase(args['codebase_path'])

        # Phase 2: Integration
        if args.get('integrate'):
            integration = self.integrate_code(analysis, args)

        # Phase 3: Optimization
        if args.get('optimize'):
            optimization = self.optimize_code(integration, args)

        return results
```

### Shared Utility Integration

**Code Modifier** - Safe code transformation
- Backup creation before modifications
- File modification with rollback support
- Import management and organization
- Code formatting and style consistency

**AST Analyzer** - Deep code analysis
- Function and class extraction
- Dependency analysis
- Complexity metrics
- Dead code detection

**Test Runner** - Multi-framework validation
- Numerical accuracy tests
- Performance benchmarking
- Regression testing
- Coverage analysis

**Git Utils** - Version control integration
- Branch creation for modernization
- Commit generation with detailed messages
- Change tracking and rollback

## Three-Phase Workflow

### Phase 1: Analysis (`--analyze`)

Comprehensive codebase assessment to understand structure, algorithms, and modernization opportunities.

#### Analysis Process

```
┌─────────────────────────────────────────┐
│ 1. Language Detection                   │
│    • Parse file extensions              │
│    • Detect language versions           │
│    • Identify mixed-language patterns   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ 2. Dependency Analysis                  │
│    • External library detection         │
│    • Internal module dependencies       │
│    • Build system analysis              │
│    • Platform-specific code             │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ 3. Algorithm Identification             │
│    • Computational kernels              │
│    • Numerical methods classification   │
│    • Data structure patterns            │
│    • Performance hotspots               │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ 4. Modernization Assessment             │
│    • Integration feasibility            │
│    • Optimization opportunities         │
│    • Risk analysis                      │
│    • Effort estimation                  │
└─────────────────────────────────────────┘
```

#### Analysis Output

```json
{
  "codebase_summary": {
    "total_files": 247,
    "languages": {"fortran": 189, "c": 52, "python": 6},
    "lines_of_code": 145000,
    "complexity_score": "high"
  },
  "algorithms": [
    {
      "name": "FFT solver",
      "method": "Fast Fourier Transform",
      "files": ["fft_solver.f90", "fft_utils.f90"],
      "performance_critical": true,
      "optimization_potential": "GPU acceleration"
    }
  ],
  "dependencies": {
    "external": ["BLAS", "LAPACK", "MPI", "HDF5"],
    "modernization_path": "NumPy/SciPy/JAX"
  },
  "recommendations": {
    "target": "jax",
    "parallel": "cuda",
    "effort": "3-6 months",
    "risk": "medium"
  }
}
```

### Phase 2: Integration (`--integrate`)

Bridge legacy code with modern frameworks through wrapper generation and API design.

#### Integration Strategies

**1. Foreign Function Interface (FFI)**
```python
# Generated Python wrapper for Fortran code
import ctypes
import numpy as np
from pathlib import Path

class FortranSolverWrapper:
    """Python interface to legacy Fortran solver"""

    def __init__(self):
        # Load compiled Fortran library
        lib_path = Path(__file__).parent / "libsolver.so"
        self.lib = ctypes.CDLL(str(lib_path))

        # Define function signatures
        self.lib.solve_system.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # matrix
            ctypes.POINTER(ctypes.c_double),  # vector
            ctypes.c_int,                      # size
            ctypes.POINTER(ctypes.c_double)   # result
        ]
        self.lib.solve_system.restype = ctypes.c_int

    def solve(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Solve linear system Ax = b

        Args:
            matrix: Coefficient matrix (n x n)
            vector: Right-hand side (n)

        Returns:
            Solution vector (n)
        """
        # Validate inputs
        assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
        assert matrix.shape[0] == vector.shape[0], "Dimension mismatch"

        # Prepare data (Fortran column-major order)
        n = matrix.shape[0]
        matrix_f = np.asfortranarray(matrix, dtype=np.float64)
        vector_f = np.asfortranarray(vector, dtype=np.float64)
        result = np.zeros(n, dtype=np.float64)

        # Call Fortran function
        status = self.lib.solve_system(
            matrix_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            vector_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )

        if status != 0:
            raise RuntimeError(f"Solver failed with status {status}")

        return result
```

**2. f2py Integration (Fortran → Python)**
```bash
# Automatic wrapper generation with f2py
f2py -c solver.f90 -m solver_module

# Usage in Python
import solver_module
result = solver_module.solve_system(matrix, vector)
```

**3. Pybind11 Integration (C++ → Python)**
```cpp
// C++ code with Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> solve_system(
    py::array_t<double> matrix,
    py::array_t<double> vector) {

    // Get buffer info
    auto mat_buf = matrix.request();
    auto vec_buf = vector.request();

    // Validate dimensions
    if (mat_buf.ndim != 2 || vec_buf.ndim != 1) {
        throw std::runtime_error("Invalid dimensions");
    }

    // Call C++ solver
    size_t n = mat_buf.shape[0];
    auto result = py::array_t<double>(n);
    auto result_buf = result.request();

    cpp_solve(
        static_cast<double*>(mat_buf.ptr),
        static_cast<double*>(vec_buf.ptr),
        static_cast<double*>(result_buf.ptr),
        n
    );

    return result;
}

PYBIND11_MODULE(cpp_solver, m) {
    m.def("solve_system", &solve_system, "Solve linear system");
}
```

**4. JAX Integration with Custom Operations**
```python
# JAX wrapper with custom C++/CUDA kernel
import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import xla

# Register custom operation
def legacy_solver_p(matrix, vector):
    """JAX primitive for legacy solver"""
    return legacy_solver_p.bind(matrix, vector)

# Define abstract evaluation
def legacy_solver_abstract_eval(matrix, vector):
    return core.ShapedArray(vector.shape, vector.dtype)

# XLA lowering to custom call
def legacy_solver_xla_translation(c, matrix, vector):
    return xla.custom_call(
        c,
        b"legacy_solver_kernel",
        operands=[matrix, vector],
        shape=vector.shape,
        dtype=vector.dtype
    )

# Register with JAX
legacy_solver_p = core.Primitive("legacy_solver")
legacy_solver_p.def_abstract_eval(legacy_solver_abstract_eval)
xla.backend_specific_translations['cpu'][legacy_solver_p] = \
    legacy_solver_xla_translation
```

### Phase 3: Optimization (`--optimize`)

Apply performance improvements while preserving numerical accuracy.

#### Optimization Categories

**1. Algorithmic Optimization**
```python
# Before: O(n³) naive matrix multiplication
def matmul_naive(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += A[i,k] * B[k,j]
    return C

# After: O(n²·⁸¹) Strassen or O(n³) optimized BLAS
def matmul_optimized(A, B):
    return np.dot(A, B)  # Uses optimized BLAS

# JAX with GPU acceleration
@jax.jit
def matmul_jax(A, B):
    return jnp.dot(A, B)  # Compiles to XLA, runs on GPU
```

**2. Vectorization**
```fortran
! Before: Scalar operations
do i = 1, n
    result(i) = a(i) * b(i) + c(i)
end do

! After: Vectorized with compiler directives
!$OMP SIMD
do i = 1, n
    result(i) = a(i) * b(i) + c(i)
end do
```

**3. Memory Layout Optimization**
```python
# Before: Poor cache locality
def stencil_naive(grid):
    n = grid.shape[0]
    result = np.zeros_like(grid)
    for i in range(1, n-1):
        for j in range(1, n-1):
            result[i,j] = (
                grid[i-1,j] + grid[i+1,j] +
                grid[i,j-1] + grid[i,j+1]
            ) / 4
    return result

# After: Blocked for cache efficiency
def stencil_blocked(grid, block_size=32):
    n = grid.shape[0]
    result = np.zeros_like(grid)

    for bi in range(1, n-1, block_size):
        for bj in range(1, n-1, block_size):
            # Process block
            for i in range(bi, min(bi+block_size, n-1)):
                for j in range(bj, min(bj+block_size, n-1)):
                    result[i,j] = (
                        grid[i-1,j] + grid[i+1,j] +
                        grid[i,j-1] + grid[i,j+1]
                    ) / 4
    return result
```

**4. GPU Acceleration**
```python
# JAX GPU-accelerated version
import jax
import jax.numpy as jnp

@jax.jit
def stencil_jax(grid):
    """GPU-accelerated stencil operation"""
    # JAX automatically compiles to GPU
    kernel = jnp.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]]) / 4

    # Use convolution for efficient stencil
    from jax.scipy.signal import convolve2d
    return convolve2d(grid, kernel, mode='same')

# CUDA kernel (for custom operations)
cuda_code = """
__global__ void stencil_kernel(
    const double* input,
    double* output,
    int nx, int ny) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        int idx = i * ny + j;
        output[idx] = 0.25 * (
            input[(i-1)*ny + j] +
            input[(i+1)*ny + j] +
            input[i*ny + (j-1)] +
            input[i*ny + (j+1)]
        );
    }
}
"""
```

## Multi-Agent Orchestration

The adopt-code command leverages the 23-agent system for comprehensive code modernization.

### Agent Selection

**Scientific Agents (`--agents=scientific`)**
- `scientific-computing-master` - Lead scientific code analysis
- `jax-pro` - JAX ecosystem integration
- `neural-networks-master` - ML framework integration
- `research-intelligence-master` - Algorithm understanding
- Domain-specific experts (quantum, soft-matter, stochastic)

**Quality Agents (`--agents=quality`)**
- `code-quality-master` - Code quality analysis
- Testing strategy and validation
- Numerical accuracy verification
- Performance benchmarking

**All Agents (`--agents=all`)**
- Complete 23-agent system activation
- Cross-domain analysis and synthesis
- Comprehensive optimization recommendations
- Multi-perspective validation

### Agent Workflow

```
┌─────────────────────────────────────────┐
│ Agent Orchestrator                      │
│ • Analyzes codebase characteristics     │
│ • Selects optimal agent combination     │
│ • Distributes analysis tasks            │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Scientific│ │ Quality  │ │ Domain   │
│ Agents   │ │ Agents   │ │ Experts  │
└─────┬────┘ └─────┬────┘ └─────┬────┘
      │            │            │
      └────────────┼────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │ Synthesis Agent  │
        │ • Integrate      │
        │ • Prioritize     │
        │ • Validate       │
        └──────────────────┘
```

## Testing and Validation

Comprehensive testing ensures numerical accuracy and performance preservation.

### Test Suite Generation

```python
class ScientificTestSuite:
    """Automated test generation for modernized code"""

    def generate_accuracy_tests(self, legacy_func, modern_func):
        """Generate numerical accuracy tests"""
        test_cases = [
            # Edge cases
            self.generate_edge_cases(),
            # Random cases
            self.generate_random_cases(n=1000),
            # Known analytical solutions
            self.generate_analytical_cases(),
            # Stress tests
            self.generate_stress_cases()
        ]

        return test_cases

    def verify_accuracy(self, legacy_result, modern_result, tolerance=1e-10):
        """Verify numerical accuracy preservation"""
        # Absolute error
        abs_error = np.abs(legacy_result - modern_result)

        # Relative error
        rel_error = abs_error / (np.abs(legacy_result) + 1e-16)

        # Statistical comparison
        mean_rel_error = np.mean(rel_error)
        max_rel_error = np.max(rel_error)

        assert max_rel_error < tolerance, \
            f"Accuracy loss: max relative error = {max_rel_error}"

        return {
            'mean_error': mean_rel_error,
            'max_error': max_rel_error,
            'passed': True
        }

    def benchmark_performance(self, func, test_cases):
        """Benchmark performance"""
        import time

        times = []
        for case in test_cases:
            start = time.perf_counter()
            result = func(case)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
```

### Validation Checklist

- [ ] **Numerical Accuracy**
  - [ ] Results match within tolerance (default: 1e-10)
  - [ ] Conservation laws preserved
  - [ ] Convergence rates maintained
  - [ ] Edge cases handled correctly

- [ ] **Performance**
  - [ ] Meets or exceeds baseline performance
  - [ ] Parallel scaling verified
  - [ ] Memory usage optimized
  - [ ] GPU acceleration validated

- [ ] **Correctness**
  - [ ] All test cases pass
  - [ ] No regression in functionality
  - [ ] Error handling robust
  - [ ] API compatibility maintained

- [ ] **Quality**
  - [ ] Code formatted and documented
  - [ ] Type hints added
  - [ ] Linting passes
  - [ ] Security vulnerabilities addressed

## Complete Workflow Examples

### Example 1: Legacy Fortran Climate Model → Python/JAX

```bash
# Step 1: Analyze legacy Fortran codebase
/adopt-code ./climate_model_f90 --analyze --language=fortran --agents=scientific

# Output:
# ✓ Detected: 189 Fortran files, 145K LOC
# ✓ Algorithms: Navier-Stokes solver, FFT, MPI communication
# ✓ Dependencies: BLAS, LAPACK, MPI, HDF5
# ✓ Recommendation: JAX with MPI parallelization
# ✓ Effort estimate: 4-6 months, Medium risk

# Step 2: Integrate with Python/JAX
/adopt-code ./climate_model_f90 --integrate \
  --language=fortran \
  --target=jax \
  --parallel=mpi \
  --agents=all

# Output:
# ✓ Generated Python wrappers for 47 Fortran subroutines
# ✓ Created JAX-compatible API
# ✓ Implemented MPI communication layer
# ✓ Added comprehensive type hints
# ✓ Generated test suite (523 tests)

# Step 3: Optimize for GPU acceleration
/adopt-code ./climate_model_jax --optimize \
  --parallel=cuda \
  --agents=scientific

# Output:
# ✓ Ported 12 computational kernels to JAX
# ✓ Applied JIT compilation
# ✓ Optimized memory layout for GPU
# ✓ Implemented mixed-precision computation
# ✓ Performance: 8.3x speedup on A100 GPU

# Step 4: Validate numerical accuracy
/generate-tests ./climate_model_jax --type=scientific --coverage=95
/run-all-tests --scientific --reproducible --gpu

# Output:
# ✓ 523/523 tests passed
# ✓ Mean relative error: 3.2e-12
# ✓ Max relative error: 8.7e-11
# ✓ Coverage: 96.4%
# ✓ All conservation laws preserved

# Step 5: Commit modernized code
/commit --all --ai-message --template=feat
```

### Example 2: C++ Molecular Dynamics → JAX

```bash
# Step 1: Analyze C++ codebase
/adopt-code ./md_cpp --analyze --language=cpp

# Step 2: Generate Python bindings
/adopt-code ./md_cpp --integrate --language=cpp --target=python

# Step 3: Convert to JAX for GPU
/adopt-code ./md_python --integrate --target=jax --parallel=cuda

# Step 4: Optimize force calculation kernels
/adopt-code ./md_jax --optimize --parallel=cuda --agents=scientific

# Step 5: Benchmark performance
/debug --gpu --profile --benchmark

# Results:
# Legacy C++: 1.2 ms/step (CPU)
# JAX: 0.18 ms/step (GPU) - 6.7x speedup
# Memory: 2.1 GB → 850 MB (60% reduction)
```

### Example 3: Mixed Fortran/C Quantum Chemistry Code

```bash
# Step 1: Analyze mixed codebase
/adopt-code ./quantum_chem --analyze \
  --language=mixed \
  --agents=all

# Step 2: Staged integration (Fortran first)
/adopt-code ./quantum_chem/fortran --integrate \
  --language=fortran \
  --target=python

# Step 3: Integrate C components
/adopt-code ./quantum_chem/c --integrate \
  --language=c \
  --target=python

# Step 4: Unified optimization
/adopt-code ./quantum_chem_python --optimize \
  --parallel=mpi \
  --agents=scientific

# Step 5: Quality assurance
/check-code-quality ./quantum_chem_python --auto-fix
/multi-agent-optimize ./quantum_chem_python \
  --mode=review \
  --agents=all \
  --implement

# Step 6: Comprehensive testing
/generate-tests ./quantum_chem_python --type=scientific
/run-all-tests --scientific --parallel --coverage
```

## Performance Targets

### Expected Speedups by Domain

| Domain | Baseline | Target | Typical Result |
|--------|----------|--------|----------------|
| **Molecular Dynamics** | 1x (CPU) | 5-10x | 6.7x (GPU) |
| **Climate Modeling** | 1x (CPU) | 3-8x | 7.2x (GPU + MPI) |
| **Quantum Chemistry** | 1x (CPU) | 5-15x | 11.3x (GPU) |
| **CFD Simulations** | 1x (CPU) | 4-12x | 8.9x (GPU) |
| **Monte Carlo** | 1x (CPU) | 10-50x | 32x (GPU) |

### Optimization Breakdown

```
Total Speedup = Algorithmic × Vectorization × Parallelization × Hardware

Example (Climate Model):
  Algorithmic:      1.5x (better FFT algorithm)
  Vectorization:    2.3x (SIMD operations)
  Parallelization:  3.2x (MPI + GPU)
  Hardware:         2.8x (V100 vs CPU)
  Total:           31.2x theoretical (18.5x realized)
```

## Common Issues and Solutions

### Issue 1: Numerical Precision Loss

**Problem**: Results differ after modernization

**Solution**:
```python
# Use higher precision for critical operations
import jax.numpy as jnp

# Default precision
result = jnp.dot(A, B)  # float32 on GPU

# High precision
jax.config.update("jax_enable_x64", True)
result = jnp.dot(A, B)  # float64

# Mixed precision (recommended)
A_high = A.astype(jnp.float64)
result = jnp.dot(A_high, B).astype(jnp.float32)
```

### Issue 2: Memory Layout Mismatch

**Problem**: Fortran (column-major) vs C/Python (row-major)

**Solution**:
```python
# Automatic conversion
import numpy as np

# Python/C array (row-major)
A_c = np.array([[1, 2], [3, 4]], order='C')

# Convert to Fortran order
A_f = np.asfortranarray(A_c)

# Pass to Fortran function
fortran_func(A_f)
```

### Issue 3: MPI Communication Overhead

**Problem**: Poor parallel scaling

**Solution**:
```python
# Overlap computation and communication
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Non-blocking communication
req = comm.Isend(data, dest=target)

# Perform computation while data transfers
local_result = compute_local()

# Wait for communication to complete
req.Wait()

# Continue with reduced results
```

## Related Commands

**Prerequisites**: Commands to run before code adoption
- `/explain-code --level=expert` - Understand legacy codebase deeply
- `/check-code-quality` - Assess current code quality
- `/debug --auto-fix` - Fix runtime issues before migration
- Version control - Backup original codebase

**Core Integration**: Scientific computing ecosystem
- `/jax-essentials` - JAX operations after JAX migration
- `/jax-performance --gpu-accel` - JAX GPU optimization
- `/julia-jit-like --type-stability` - Julia performance optimization
- `/python-type-hint --strict` - Add type hints to modernized code

**Optimization**: Performance improvement
- `/optimize --implement --category=all` - Apply comprehensive optimizations
- `/multi-agent-optimize --focus=performance` - Multi-agent optimization analysis
- `/refactor-clean --patterns=modern` - Code modernization and cleanup

**Validation**: Testing and quality assurance
- `/generate-tests --type=scientific --coverage=95` - Generate scientific tests
- `/run-all-tests --scientific --reproducible --gpu` - Comprehensive validation
- `/double-check --deep-analysis` - Verify adoption results
- `/reflection --type=scientific` - Analyze adoption effectiveness

## Integration Patterns

### Complete Modernization Pipeline

```bash
# 1. Analysis and planning
/adopt-code legacy/ --analyze --language=fortran --agents=all
/think-ultra "Fortran to JAX migration strategy" --agents=scientific

# 2. Staged integration
/adopt-code legacy/ --integrate --target=python --parallel=mpi
/generate-tests modern/ --type=scientific

# 3. Optimization and acceleration
/adopt-code modern/ --optimize --parallel=cuda --agents=scientific
/jax-performance modern/ --gpu-accel --technique=caching

# 4. Validation and quality
/run-all-tests --scientific --reproducible --gpu --coverage
/check-code-quality modern/ --auto-fix

# 5. Documentation and commit
/update-docs modern/ --type=all --format=markdown
/commit --all --ai-message --template=feat
```

### Performance-First Adoption

```bash
# Profile legacy code
/debug legacy/ --profile --benchmark

# Adopt with optimization focus
/adopt-code legacy/ --analyze --optimize --agents=scientific

# Target-specific optimization
/optimize modern/ --implement --category=algorithm,memory
/jax-essentials modern/ --operation=jit --static-args

# Validate performance gains
/run-all-tests --benchmark --profile
```

### Quality-Assured Migration

```bash
# Pre-migration quality check
/check-code-quality legacy/ --analysis=scientific
/explain-code legacy/ --level=expert --docs

# Adopt with quality focus
/adopt-code legacy/ --integrate --agents=quality,scientific

# Comprehensive testing
/generate-tests modern/ --type=all --coverage=95
/run-all-tests --scientific --auto-fix

# Multi-agent review
/multi-agent-optimize modern/ --mode=review --agents=all
/double-check "migration quality" --deep-analysis
```

## Best Practices

### 1. Scientific Accuracy First
- **Never sacrifice correctness for performance**
- Validate against known analytical solutions
- Test edge cases and numerical stability
- Preserve conservation laws and physical constraints

### 2. Incremental Modernization
- Adopt one module at a time
- Maintain backward compatibility during transition
- Create comprehensive test suites before changes
- Use feature flags for gradual rollout

### 3. Documentation and Version Control
- Document all mathematical assumptions
- Track performance metrics over time
- Create detailed migration guides
- Use semantic versioning for releases

### 4. Performance Validation
- Benchmark before and after changes
- Profile to identify remaining bottlenecks
- Test parallel scaling on target hardware
- Measure memory usage and I/O performance

### 5. Community and Reproducibility
- Share modernization experiences
- Publish performance benchmarks
- Ensure deterministic results when needed
- Provide containerized environments

## Exit Codes

- `0` - Successful adoption/analysis/optimization
- `1` - Analysis warnings (proceed with caution)
- `2` - Integration failures (check dependencies)
- `3` - Optimization issues (numerical accuracy concerns)
- `4` - Testing failures (validation errors)

## Requirements

### System Requirements
- Python 3.8+ (for Python target)
- JAX 0.4+ (for JAX target)
- Julia 1.6+ (for Julia target)
- CUDA 11.0+ (for GPU acceleration)
- MPI implementation (for MPI parallelization)

### Build Tools
- Fortran compiler (gfortran, ifort)
- C/C++ compiler (gcc, clang, icc)
- f2py or pybind11 (for integration)
- CMake 3.15+ (for build systems)

### Optional Dependencies
- BLAS/LAPACK (linear algebra)
- FFTW (Fourier transforms)
- HDF5 (data I/O)
- CuPy (GPU arrays)
- Numba (JIT compilation)

ARGUMENTS: [--analyze] [--integrate] [--optimize] [--language=fortran|c|cpp|python|julia|mixed] [--target=python|jax|julia] [--parallel=mpi|openmp|cuda|jax] [--agents=scientific|quality|orchestrator|all] [codebase-path]