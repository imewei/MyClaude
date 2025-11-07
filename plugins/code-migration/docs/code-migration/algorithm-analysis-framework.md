# Algorithm Analysis Framework

**Version**: 1.0.3
**Category**: code-migration
**Purpose**: Comprehensive framework for analyzing legacy code algorithms, computational kernels, and data structures

## Overview

This framework provides systematic methodologies for understanding legacy scientific computing code, identifying core algorithms, analyzing computational hotspots, and documenting numerical properties essential for successful migration.

## Algorithm Identification Methodology

### 1. Iterative Methods

**Detection Patterns**:
```fortran
! Fortran patterns
DO WHILE (error > tolerance)
  ! Iterative refinement
END DO

! Fixed iteration loops
DO iter = 1, maxiter
  ! Convergence check
  IF (ABS(residual) < tol) EXIT
END DO
```

**Key Characteristics to Document**:
- **Convergence Criteria**: Absolute vs. relative tolerance
- **Iteration Limits**: Maximum iteration count
- **Initial Guess Strategy**: Zero, random, problem-specific
- **Stability**: Convergence rate, divergence conditions
- **Examples**: Jacobi, Gauss-Seidel, Conjugate Gradient, GMRES

**Migration Considerations**:
- Preserve convergence criteria exactly
- Document iteration history for validation
- Consider vectorization opportunities
- GPU acceleration potential (large systems)

---

### 2. Direct Solvers

**Detection Patterns**:
```fortran
! LAPACK calls
CALL DGESV(N, NRHS, A, LDA, IPIV, B, LDB, INFO)  ! General linear system
CALL DGESVD(...)  ! SVD decomposition
CALL DSYEV(...)   ! Eigenvalue problem
```

**Key Characteristics**:
- **Matrix Decomposition**: LU, Cholesky, QR, SVD
- **Matrix Properties**: Symmetric, positive-definite, sparse, banded
- **Numerical Stability**: Condition number, ill-conditioning detection
- **Memory Requirements**: O(N²) or O(N³) storage
- **Precision Needs**: Single vs. double precision

**Migration Path**:
- **NumPy/SciPy**: `np.linalg.solve`, `scipy.linalg.lu`
- **JAX**: `jax.numpy.linalg.solve` (GPU-compatible)
- **Julia**: `LinearAlgebra.lu`, `LAPACK.gesv!`

---

### 3. Time Integration Schemes

**Common Algorithms**:
- **Explicit Methods**: Forward Euler, Runge-Kutta (RK2, RK4)
- **Implicit Methods**: Backward Euler, Crank-Nicolson, BDF
- **Adaptive Methods**: RK45, LSODE, CVODE

**Detection Examples**:
```fortran
! Forward Euler
y(n+1) = y(n) + dt * f(y(n), t(n))

! RK4 pattern
k1 = f(y, t)
k2 = f(y + dt/2*k1, t + dt/2)
k3 = f(y + dt/2*k2, t + dt/2)
k4 = f(y + dt*k3, t + dt)
y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

**Critical Properties**:
- **Order of Accuracy**: 1st, 2nd, 4th order
- **Stability Region**: A-stable, L-stable
- **Time Step Selection**: Fixed vs. adaptive
- **Stiffness Handling**: Implicit for stiff systems

**Modern Equivalents**:
- **SciPy**: `scipy.integrate.solve_ivp` (RK45, BDF, LSODE)
- **JAX**: `diffrax` library (GPU-compatible ODE solvers)
- **Julia**: `DifferentialEquations.jl` (most comprehensive)

---

### 4. Discretization Methods

**Method Types**:

**Finite Difference Method (FDM)**:
```fortran
! Central difference for second derivative
d2u_dx2(i) = (u(i+1) - 2*u(i) + u(i-1)) / dx**2
```
- Grid-based, simple implementation
- Structured meshes
- 2nd, 4th, 6th order accuracy available

**Finite Element Method (FEM)**:
- Weak formulation, variational principles
- Unstructured meshes, complex geometries
- Basis functions (linear, quadratic, higher-order)

**Finite Volume Method (FVM)**:
- Conservation laws, flux calculations
- Shock-capturing schemes
- Godunov, MUSCL, WENO

**Spectral Methods**:
- Fourier transforms, Chebyshev polynomials
- High accuracy for smooth solutions
- FFT-based implementations

**Migration Libraries**:
- **FDM**: NumPy array operations, JAX for GPU
- **FEM**: FEniCS, Firedrake, SfePy
- **Spectral**: NumPy FFT, SciPy special functions, JAX FFT

---

### 5. Optimization Algorithms

**Common Patterns**:

**Gradient-Based**:
```fortran
! Steepest descent
x_new = x - alpha * gradient(f, x)

! Newton's method
x_new = x - inv(Hessian) * gradient(f, x)
```

**Evolutionary Algorithms**:
- Genetic algorithms, particle swarm
- Population-based, stochastic

**Constrained Optimization**:
- Lagrange multipliers
- Interior point methods
- Penalty methods

**Modern Frameworks**:
- **SciPy**: `scipy.optimize.minimize` (BFGS, L-BFGS-B, trust-constr)
- **JAX**: Automatic differentiation for gradients (jax.grad, jax.hessian)
- **Julia**: JuMP.jl, Optim.jl

---

## Computational Kernel Analysis

### Hotspot Detection Methodology

**Step 1: Profiling**
```bash
# Fortran profiling with gprof
gfortran -pg -O2 code.f90 -o code
./code
gprof code gmon.out > analysis.txt

# Python profiling
python -m cProfile -o profile.stats script.py
python -m pstats profile.stats
```

**Step 2: Identify Top Functions** (80/20 rule)
- Focus on functions consuming >80% runtime
- Typically 10-20% of code accounts for 80% execution time

**Step 3: Classify Kernel Types**:
- **Memory-bound**: Large array operations, bandwidth-limited
- **Compute-bound**: Heavy arithmetic, FLOPs-limited
- **I/O-bound**: File reads/writes, data transfer

---

### Loop Complexity Analysis

**Nested Loop Detection**:
```fortran
! O(N³) complexity - matrix multiplication
DO i = 1, N
  DO j = 1, N
    DO k = 1, N
      C(i,j) = C(i,j) + A(i,k) * B(k,j)
    END DO
  END DO
END DO
```

**Complexity Classes**:
- **O(N)**: Linear scans, reductions
- **O(N log N)**: FFT, efficient sorting
- **O(N²)**: Pairwise interactions, dense matrix ops
- **O(N³)**: Matrix multiplication, Gaussian elimination

**Vectorization Opportunities**:
- **NumPy**: Broadcasting, `np.einsum` for tensors
- **JAX**: `jax.vmap` for automatic vectorization
- **SIMD**: AVX-512 intrinsics for low-level optimization

---

### Memory Access Pattern Analysis

**Access Types**:

**Contiguous (Efficient)**:
```fortran
! Row-major (C/Python) or column-major (Fortran)
DO i = 1, N
  sum = sum + array(i)  ! Sequential access
END DO
```

**Strided (Cache-Inefficient)**:
```fortran
! Accessing every K-th element
DO i = 1, N, K
  sum = sum + array(i)  ! Large stride
END DO
```

**Random (Cache-Hostile)**:
```fortran
! Indirect indexing
DO i = 1, N
  sum = sum + array(indices(i))  ! Random access
END DO
```

**Cache Optimization**:
- **Blocking/Tiling**: Improve cache reuse
- **Loop Interchange**: Optimize for memory layout
- **Array Padding**: Avoid cache line conflicts

---

## Data Structure Analysis

### Memory Layout Patterns

**Fortran (Column-Major)**:
```fortran
REAL*8 :: A(M, N)
! Memory: A(1,1), A(2,1), ..., A(M,1), A(1,2), ...
```

**C/Python (Row-Major)**:
```c
double A[M][N];
// Memory: A[0][0], A[0][1], ..., A[0][N-1], A[1][0], ...
```

**Migration Strategy**:
- **Preserve order**: Use `order='F'` in NumPy for Fortran arrays
- **Transpose**: Convert between layouts if needed
- **Document impact**: Performance implications of layout changes

**NumPy Layout Control**:
```python
# Fortran-order (column-major)
arr_f = np.array([[1, 2], [3, 4]], order='F')

# C-order (row-major) - default
arr_c = np.array([[1, 2], [3, 4]], order='C')

# Check layout
print(arr_f.flags['F_CONTIGUOUS'])  # True
```

---

### Precision Analysis

**Fortran Precision Types**:
```fortran
REAL*4   ! Single precision (float32) - ~7 digits
REAL*8   ! Double precision (float64) - ~15 digits
REAL*16  ! Quad precision (float128) - ~33 digits (if available)
```

**Python/NumPy Equivalents**:
```python
import numpy as np

arr32 = np.array([1.0], dtype=np.float32)   # Single
arr64 = np.array([1.0], dtype=np.float64)   # Double (default)
arr128 = np.array([1.0], dtype=np.float128) # Quad (limited support)
```

**Precision Requirements Analysis**:
1. **Identify critical variables**: Which need high precision?
2. **Error propagation**: How do errors accumulate?
3. **Mixed precision**: Use float32 for storage, float64 for computation
4. **Validation tolerance**: Define acceptable error bounds

---

### Data Structure Sizes

**Memory Footprint Calculation**:
```python
# Example: 3D grid (1000 x 1000 x 100)
N = 1000 * 1000 * 100  # 100 million elements

# Memory requirements
mem_float32 = N * 4 / 1e9  # 0.4 GB
mem_float64 = N * 8 / 1e9  # 0.8 GB
```

**Sparse vs. Dense**:
- **Dense**: Full matrix storage (O(N²))
- **Sparse**: Store only non-zeros (O(NNZ))
- **Threshold**: Use sparse if <10% non-zero

**Sparse Libraries**:
- **SciPy**: `scipy.sparse.csr_matrix`, `scipy.sparse.linalg`
- **JAX**: `jax.experimental.sparse`
- **Julia**: `SparseArrays.jl`

---

## Performance Profiling Techniques

### CPU Profiling

**Python - cProfile**:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
result = expensive_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)  # Top 20 functions
```

**Line-by-Line Profiling**:
```python
from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(hotspot_function)
lp.run('hotspot_function()')
lp.print_stats()
```

---

### Memory Profiling

**Memory Usage Tracking**:
```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    large_array = np.zeros((10000, 10000))
    # Memory usage tracked line-by-line
    return large_array.sum()
```

**Peak Memory Detection**:
```python
import tracemalloc

tracemalloc.start()

# Code execution
result = compute()

current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1e6:.2f} MB")
tracemalloc.stop()
```

---

### GPU Profiling

**JAX Profiling**:
```python
import jax
import jax.numpy as jnp

# Profile GPU kernel
@jax.jit
def gpu_kernel(x):
    return jnp.dot(x, x.T)

# Time execution
%timeit gpu_kernel(x).block_until_ready()
```

**NVIDIA Nsight**:
```bash
# Profile CUDA kernels
nsys profile --trace=cuda,nvtx python script.py
nsight-sys report.nsys-rep
```

---

## Code Pattern Recognition

### Common Scientific Computing Patterns

**Pattern 1: Reduction Operations**
```fortran
! Sum, product, min, max
sum = 0.0
DO i = 1, N
  sum = sum + array(i)
END DO
```
**Modern**: `np.sum(array)`, `jnp.sum(array)`

**Pattern 2: Element-wise Operations**
```fortran
DO i = 1, N
  result(i) = a(i) * b(i) + c(i)
END DO
```
**Modern**: `result = a * b + c` (broadcasting)

**Pattern 3: Stencil Operations**
```fortran
DO i = 2, N-1
  u_new(i) = 0.25 * (u(i-1) + 2*u(i) + u(i+1))
END DO
```
**Modern**: `np.convolve`, `scipy.ndimage.convolve`

**Pattern 4: Matrix Operations**
```fortran
CALL DGEMM('N', 'N', M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC)
```
**Modern**: `C = alpha * A @ B + beta * C`

---

## Usage Examples

### Example 1: Analyze Fortran Code

```python
# Read legacy code
with open('legacy.f90', 'r') as f:
    code = f.read()

# Identify LAPACK calls
lapack_calls = re.findall(r'CALL\s+(DGESV|DGESVD|DSYEV|DGEMM)', code)
print(f"LAPACK routines: {set(lapack_calls)}")

# Detect loop patterns
do_loops = re.findall(r'DO\s+(\w+)\s*=\s*(\d+),\s*(\w+)', code)
print(f"Found {len(do_loops)} DO loops")

# Identify precision
precision = 'REAL*8' in code
print(f"Double precision: {precision}")
```

### Example 2: Profile Performance Hotspots

```python
import cProfile
import pstats

# Profile legacy code translation
profiler = cProfile.Profile()
profiler.enable()

result = legacy_algorithm(data)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(10)

# Output:
# 1. matrix_multiply: 45.2s (60% total time)
# 2. iterative_solver: 22.1s (29%)
# 3. file_io: 8.3s (11%)
```

### Example 3: Memory Layout Verification

```python
import numpy as np

# Fortran-order array (column-major)
arr_f = np.zeros((1000, 1000), order='F')

# Verify layout
print(f"F-contiguous: {arr_f.flags['F_CONTIGUOUS']}")
print(f"C-contiguous: {arr_f.flags['C_CONTIGUOUS']}")

# Access pattern performance
# Column access (fast for Fortran order)
col_sum = arr_f[:, 100].sum()

# Row access (slower for Fortran order)
row_sum = arr_f[100, :].sum()
```

---

## Integration Points

This framework integrates with:
- **numerical-accuracy-guide.md**: Precision requirements feed into accuracy validation
- **framework-migration-strategies.md**: Algorithm analysis informs framework selection
- **performance-optimization-techniques.md**: Hotspot identification guides optimization
- **migration-examples.md**: Real-world applications of analysis framework

---

## Best Practices

### 1. Document Assumptions
- Record all implicit assumptions in legacy code
- Test assumptions during migration
- Validate assumptions don't break in new framework

### 2. Preserve Algorithmic Intent
- Understand "why" not just "what"
- Document algorithmic reasoning
- Maintain mathematical equivalence

### 3. Benchmark Systematically
- Profile before optimizing
- Measure apples-to-apples (same problem size, hardware)
- Document baseline performance

### 4. Validate Incrementally
- Test each component in isolation
- Compare against reference solutions
- Build regression test suite

---

## References

- **LAPACK**: Linear Algebra Package - http://www.netlib.org/lapack/
- **NumPy**: Numerical Python - https://numpy.org/doc/
- **SciPy**: Scientific Python - https://docs.scipy.org/
- **JAX**: Composable transformations - https://jax.readthedocs.io/
- **Julia**: High-performance computing - https://docs.julialang.org/
