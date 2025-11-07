# Scientific Computing Best Practices

**Version**: 1.0.3
**Category**: code-migration
**Purpose**: Domain-specific guidelines for numerical stability, performance sensitivity, and legacy compatibility

## Numerical Stability Guidelines

### Condition Number Analysis

**Problem Sensitivity to Perturbations**:
```python
import numpy as np

def analyze_condition_number(A):
    """Check if linear system is ill-conditioned"""
    cond = np.linalg.cond(A)

    if cond < 10:
        status = "Well-conditioned"
    elif cond < 1000:
        status = "Moderate condition"
    elif cond < 1e6:
        status = "Ill-conditioned (caution)"
    else:
        status = "Severely ill-conditioned (avoid direct solve)"

    print(f"Condition number: {cond:.2e}")
    print(f"Status: {status}")
    print(f"Expected accuracy loss: ~{np.log10(cond):.1f} decimal digits")

    return cond

# Example: Hilbert matrix (notoriously ill-conditioned)
n = 10
H = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])
analyze_condition_number(H)
# Output:
# Condition number: 1.60e+13
# Status: Severely ill-conditioned
# Expected accuracy loss: ~13 decimal digits
```

**Mitigation Strategies**:
1. **Regularization**: Add small diagonal term (ridge regression)
2. **Preconditioning**: Transform to better-conditioned problem
3. **Iterative refinement**: Improve solution accuracy
4. **Higher precision**: Use float128 for critical calculations

---

### Algorithm Selection for Stability

**Stable vs. Unstable Algorithms**:

| Problem | Unstable Method | Stable Method |
|---------|----------------|---------------|
| Linear system | Cramer's rule | LU decomposition |
| Eigenvalues | Power iteration (general) | QR algorithm |
| Least squares | Normal equations | QR factorization |
| Polynomial roots | Companion matrix | Jenkins-Traub |

**Example - Least Squares**:
```python
import numpy as np

# Unstable: Normal equations (A^T A can be ill-conditioned)
def unstable_least_squares(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)

# Stable: QR factorization
def stable_least_squares(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]

# Test with ill-conditioned matrix
A = np.random.rand(100, 10) * 1e-8
b = np.random.rand(100)

try:
    x_unstable = unstable_least_squares(A, b)
    print(f"Unstable residual: {np.linalg.norm(A @ x_unstable - b)}")
except np.linalg.LinAlgError:
    print("Unstable method failed (singular matrix)")

x_stable = stable_least_squares(A, b)
print(f"Stable residual: {np.linalg.norm(A @ x_stable - b):.2e}")
```

---

## Performance Sensitivity Considerations

### Profile Before Optimizing

**Premature Optimization is the Root of All Evil** - Donald Knuth

**Profiling Workflow**:
```python
import cProfile
import pstats

# 1. Profile to find bottlenecks
profiler = cProfile.Profile()
profiler.enable()

result = simulation()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(10)

# Output might show:
# 1. matrix_multiply:  45s  (60% of runtime)  ← Optimize this!
# 2. file_io:           8s  (11%)
# 3. visualization:     5s   (7%)
```

**Optimization Priority**:
1. Profile to identify hotspots (>80% of time)
2. Optimize algorithmic complexity first (O(N²) → O(N log N))
3. Then vectorize/parallelize
4. Then low-level optimization (SIMD, GPU)

---

### Preserve Algorithmic Complexity

**Don't Make it Slower!**:
```python
# Bad: O(N²) when O(N) exists
def slow_unique(arr):
    unique = []
    for val in arr:
        if val not in unique:  # O(N) membership test
            unique.append(val)
    return unique

# Good: O(N) using set
def fast_unique(arr):
    return list(set(arr))  # O(N) hashing

# Benchmark
arr = np.random.randint(0, 1000, size=10000)

%timeit slow_unique(arr)  # 450 ms
%timeit fast_unique(arr)  # 0.5 ms (900x faster!)
```

---

### Document Performance Trade-offs

**Example Documentation**:
```python
def compute_fft(signal, method='numpy'):
    """
    Compute Fast Fourier Transform.

    Parameters
    ----------
    signal : array_like
        Input signal
    method : str
        FFT implementation to use

    Performance Characteristics
    ---------------------------
    - numpy FFT: O(N log N), CPU-only, good for N < 10^6
    - JAX FFT: O(N log N), GPU-accelerated, best for N > 10^6
    - scipy FFT: Similar to numpy, slightly slower

    Memory Usage
    ------------
    - numpy/scipy: 2x signal size (in-place option available)
    - JAX: 3x signal size (GPU memory)

    Returns
    -------
    fft_result : array
        Fourier transform of input
    """
    if method == 'numpy':
        return np.fft.fft(signal)
    elif method == 'jax':
        return jnp.fft.fft(signal)
```

---

### Mixed-Precision Opportunities

**When to Use Lower Precision**:
```python
# Storage: float32 (half memory)
data = np.array(large_dataset, dtype=np.float32)

# Computation: upcast to float64
result = np.sum(data.astype(np.float64))

# ML training: float16 or bfloat16
# - Faster on modern GPUs (Tensor Cores)
# - Sufficient for gradient descent
# - ~2x speedup with minimal accuracy loss
```

---

## Domain-Specific Requirements

### Physics Conservation Laws

**Energy Conservation** (Hamiltonian systems):
```python
def validate_hamiltonian(trajectory):
    """Symplectic integrators should preserve energy"""
    energies = [total_energy(state) for state in trajectory]

    energy_drift = np.abs(energies - energies[0])
    relative_drift = energy_drift / abs(energies[0])

    # Symplectic methods: O(dt^2) drift
    # Non-symplectic: linear drift
    assert np.max(relative_drift) < 1e-6, "Energy not conserved"
```

**Momentum Conservation** (N-body simulations):
```python
def check_momentum_conservation(positions, velocities, masses):
    """Total momentum should be constant"""
    momentum = np.sum(velocities * masses[:, np.newaxis], axis=0)

    # Should be zero (or constant if initial momentum != 0)
    assert np.linalg.norm(momentum) < 1e-12, "Momentum not conserved"
```

**Mass Conservation** (Chemistry, CFD):
```python
def validate_mass_conservation(concentrations):
    """Total mass must be preserved"""
    total_mass = np.sum(concentrations, axis=1)  # Sum over species

    mass_change = np.abs(total_mass - total_mass[0])
    assert np.all(mass_change < 1e-14), "Mass conservation violated"
```

---

### Boundary Condition Handling

**Common Boundary Types**:

**Dirichlet** (fixed value):
```python
# u(0) = a, u(L) = b
u[0] = a
u[-1] = b
```

**Neumann** (fixed derivative):
```python
# du/dx(0) = 0 (no-flux)
u[0] = u[1]  # Forward difference → 0
```

**Periodic**:
```python
# u(0) = u(L)
u[0] = u[-1]
u[-1] = u[0]
```

---

### Symmetry Preservation

**Example: Mirror Symmetry in PDE**:
```python
def check_symmetry(solution, midpoint):
    """Solution should be symmetric about midpoint"""
    left = solution[:midpoint]
    right = solution[midpoint:][::-1]  # Reversed

    assert np.allclose(left, right, rtol=1e-10), "Symmetry broken"
```

---

### Units and Dimensional Analysis

**Always Work in Consistent Units**:
```python
class PhysicalQuantity:
    """Quantity with units"""

    def __init__(self, value, units):
        self.value = value
        self.units = units

    def __add__(self, other):
        if self.units != other.units:
            raise ValueError(f"Cannot add {self.units} and {other.units}")
        return PhysicalQuantity(self.value + other.value, self.units)

    def convert_to(self, target_units):
        """Unit conversion"""
        conversion_factors = {
            ('m', 'cm'): 100,
            ('s', 'ms'): 1000,
            ('kg', 'g'): 1000,
        }
        factor = conversion_factors.get((self.units, target_units), 1)
        return PhysicalQuantity(self.value * factor, target_units)

# Usage
length = PhysicalQuantity(5.0, 'm')
length_cm = length.convert_to('cm')
print(f"{length_cm.value} {length_cm.units}")  # 500.0 cm
```

---

## Legacy Compatibility Patterns

### Compatibility Layer Design

**Gradual Migration with Compatibility Shim**:
```python
class LegacyCompatibility:
    """Wrapper providing legacy API while using new implementation"""

    def __init__(self):
        self.new_solver = ModernSolver()

    def legacy_solve(self, input_array, param1, param2):
        """
        Legacy API: solve(array, param1, param2)
        Maps to new API: solve(data=array, params={'p1': param1, 'p2': param2})
        """
        # Convert legacy format to new format
        params = {'p1': param1, 'p2': param2}

        # Call new implementation
        result = self.new_solver.solve(data=input_array, params=params)

        # Convert result back to legacy format if needed
        return result.values  # Extract values from Result object

# Usage allows gradual migration
legacy_api = LegacyCompatibility()
result = legacy_api.legacy_solve(data, 1.0, 2.0)  # Works with old code
```

---

### Breaking Change Documentation

**Document All Breaking Changes**:
```markdown
# Migration Guide v2.0

## Breaking Changes

### 1. Solver API Changed
**Old**: `solve(array, tol, maxiter)`
**New**: `solve(array, tolerance=tol, max_iterations=maxiter)`

**Migration**:
```python
# Before
result = solve(data, 1e-6, 1000)

# After
result = solve(data, tolerance=1e-6, max_iterations=1000)
```

### 2. Return Type Changed
**Old**: Returns NumPy array
**New**: Returns SolverResult object

**Migration**:
```python
# Before
result = solve(data)
print(result[0])

# After
result = solve(data)
print(result.values[0])  # .values attribute
```
```

---

### Migration Utilities

**Automatic Legacy File Converter**:
```python
import re

def convert_fortran_to_python(fortran_code):
    """
    Convert simple Fortran patterns to Python

    NOTE: This is a starting point, manual review required!
    """
    conversions = [
        (r'REAL\*8\s+::\s+(\w+)', r'\1 = 0.0  # float64'),
        (r'DO\s+(\w+)\s*=\s*(\d+),\s*(\w+)', r'for \1 in range(\2, \3 + 1):'),
        (r'END DO', r''),
        (r'CALL\s+DGEMM\((.*?)\)', r'result = A @ B  # Matrix multiply'),
    ]

    python_code = fortran_code
    for pattern, replacement in conversions:
        python_code = re.sub(pattern, replacement, python_code)

    return python_code
```

---

## Version Support Strategy

**Semantic Versioning**:
```
Given version MAJOR.MINOR.PATCH:

MAJOR: Incompatible API changes
MINOR: Backward-compatible new features
PATCH: Backward-compatible bug fixes
```

**Deprecation Timeline**:
```python
import warnings

def old_function(x):
    """
    .. deprecated:: 2.0
       Use :func:`new_function` instead. Will be removed in version 3.0.
    """
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function(x)
```

---

## Best Practices Checklist

### Before Migration
- [ ] Profile legacy code to identify hotspots
- [ ] Extract reference outputs for validation
- [ ] Document all algorithms and their properties
- [ ] Identify conservation laws and invariants
- [ ] List all dependencies and their versions

### During Migration
- [ ] Preserve algorithmic complexity
- [ ] Validate numerically at each step
- [ ] Test conservation laws
- [ ] Benchmark performance continuously
- [ ] Document all assumptions and trade-offs

### After Migration
- [ ] Comprehensive regression test suite
- [ ] Performance benchmarks (vs. legacy)
- [ ] Cross-platform validation
- [ ] User migration guide
- [ ] Deprecation timeline for legacy code

---

## References

- **Numerical Recipes**: Press et al., Cambridge University Press
- **Accuracy and Stability of Numerical Algorithms**: Higham (2002)
- **Scientific Software Design**: Damian Rouson (2011)
- **Best Practices for Scientific Computing**: Wilson et al., PLOS Biology (2014)
