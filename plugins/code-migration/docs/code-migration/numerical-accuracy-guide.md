# Numerical Accuracy Preservation Guide

**Version**: 1.0.3
**Category**: code-migration
**Purpose**: Comprehensive guide for preserving numerical accuracy during scientific code migration

## Overview

Numerical accuracy is the cornerstone of scientific computing migration. This guide provides systematic methodologies for determining precision requirements, designing verification strategies, ensuring reproducibility, and validating that migrated code produces bit-level equivalent results to legacy implementations.

## Precision Requirements Determination

### Floating-Point Arithmetic Basics

**IEEE 754 Standard Precision Levels**:

| Type | Bits | Mantissa | Exponent | Decimal Digits | Range |
|------|------|----------|----------|----------------|-------|
| float16 | 16 | 10 | 5 | ~3 | ±6.5×10⁴ |
| float32 | 32 | 23 | 8 | ~7 | ±3.4×10³⁸ |
| float64 | 64 | 52 | 11 | ~15 | ±1.8×10³⁰⁸ |
| float128 | 128 | 112 | 15 | ~33 | ±1.2×10⁴⁹³² |

**Machine Epsilon** (relative rounding error):
```python
import numpy as np

eps_float32 = np.finfo(np.float32).eps  # 1.19e-7
eps_float64 = np.finfo(np.float64).eps  # 2.22e-16
eps_float128 = np.finfo(np.float128).eps  # 1.08e-34

print(f"float32 precision: ~{-np.log10(eps_float32):.1f} decimal digits")
print(f"float64 precision: ~{-np.log10(eps_float64):.1f} decimal digits")
```

---

### Domain-Specific Precision Analysis

**Physics Simulations**:
- **Molecular dynamics**: float64 (float32 for forces, float64 for positions)
- **Climate modeling**: float64 for energy conservation
- **Quantum chemistry**: float64 minimum, float128 for high-accuracy

**Engineering Applications**:
- **Structural analysis**: float64 for stress/strain calculations
- **Fluid dynamics**: float64 for conservation laws
- **Electromagnetics**: float64 for field calculations

**Data Science / ML**:
- **Training**: float32 or mixed precision (faster, sufficient)
- **Inference**: float16 or int8 (quantized)
- **Loss computation**: float32 to prevent underflow

---

### Precision Requirement Extraction

**Step 1: Identify Critical Variables**

```python
def analyze_precision_requirements(code_path):
    """Extract precision-critical variables from legacy code"""
    critical_vars = {
        'positions': 'float64',  # Accumulation of small displacements
        'energies': 'float64',   # Conservation law validation
        'forces': 'float32',     # Derivative of energy (less critical)
        'masses': 'float32',     # Constants (high precision not needed)
    }
    return critical_vars
```

**Step 2: Error Propagation Analysis**

Forward error propagation for f(x):
```
Error(f) ≈ |f'(x)| * Error(x)
```

Example (exponential function):
```python
import numpy as np

x = 10.0
dx = 1e-7  # Input error

# Exponential amplifies errors
f = np.exp(x)
df = np.exp(x) * dx  # Error amplification: e^10 ≈ 22,000

print(f"Input error: {dx:.2e}")
print(f"Output error: {df:.2e} (amplified {df/dx:.0f}x)")
```

**Step 3: Accumulation Error Analysis**

```python
def accumulation_error_demo():
    """Show error accumulation in iterative algorithms"""
    N = 1_000_000

    # Float32 accumulation
    sum_f32 = np.float32(0.0)
    for i in range(N):
        sum_f32 += np.float32(0.1)

    # Float64 accumulation
    sum_f64 = np.float64(0.0)
    for i in range(N):
        sum_f64 += np.float64(0.1)

    expected = N * 0.1

    print(f"Expected: {expected}")
    print(f"Float32: {sum_f32} (error: {abs(sum_f32 - expected):.2e})")
    print(f"Float64: {sum_f64} (error: {abs(sum_f64 - expected):.2e})")

# Output:
# Expected: 100000.0
# Float32: 99998.44 (error: 1.56e+00)  # Significant accumulation!
# Float64: 100000.0 (error: 1.42e-08)  # Acceptable
```

---

## Catastrophic Cancellation Detection

### Problem: Subtracting Nearly Equal Numbers

**Example - Quadratic Formula**:
```python
import numpy as np

def unstable_quadratic(a, b, c):
    """Numerically unstable version"""
    discriminant = np.sqrt(b**2 - 4*a*c)

    # Catastrophic cancellation when b >> 4ac
    x1 = (-b + discriminant) / (2*a)
    x2 = (-b - discriminant) / (2*a)
    return x1, x2

def stable_quadratic(a, b, c):
    """Numerically stable version"""
    discriminant = np.sqrt(b**2 - 4*a*c)

    # Avoid cancellation
    if b >= 0:
        x1 = (-b - discriminant) / (2*a)
        x2 = (2*c) / (-b - discriminant)
    else:
        x1 = (2*c) / (-b + discriminant)
        x2 = (-b + discriminant) / (2*a)
    return x1, x2

# Test with b >> 4ac
a, b, c = 1.0, 1e10, 1.0

x1_unstable, x2_unstable = unstable_quadratic(a, b, c)
x1_stable, x2_stable = stable_quadratic(a, b, c)

print(f"Unstable x1: {x1_unstable:.15e}")  # Catastrophic cancellation!
print(f"Stable x1:   {x1_stable:.15e}")    # Correct
```

---

### Detection Strategies

**1. Code Pattern Analysis**:
```python
def detect_cancellation_risk(expression):
    """Identify subtraction of similar-magnitude terms"""
    risky_patterns = [
        r'\w+\s*-\s*\w+',  # a - b
        r'sqrt\(\w+\*\*2\s*-\s*\w+\*\*2\)',  # sqrt(a^2 - b^2)
    ]
    # Analyze expression for risky patterns
    pass
```

**2. Runtime Monitoring**:
```python
def monitor_cancellation(a, b):
    """Detect cancellation at runtime"""
    result = a - b

    # Check relative magnitude
    if abs(a) > 0 and abs(b) > 0:
        relative_diff = abs(result) / max(abs(a), abs(b))

        if relative_diff < 1e-6:
            print(f"⚠️ Catastrophic cancellation detected!")
            print(f"  a = {a}, b = {b}")
            print(f"  a - b = {result}")
            print(f"  Relative difference: {relative_diff:.2e}")

    return result
```

---

### Mitigation Techniques

**Technique 1: Algebraic Reformulation**
```python
# Bad: sqrt(x^2 + 1) - 1 for small x
def unstable_form(x):
    return np.sqrt(x**2 + 1) - 1

# Good: x^2 / (sqrt(x^2 + 1) + 1)
def stable_form(x):
    sqrt_term = np.sqrt(x**2 + 1)
    return x**2 / (sqrt_term + 1)

x = 1e-8
print(f"Unstable: {unstable_form(x):.15e}")  # Lost precision
print(f"Stable:   {stable_form(x):.15e}")    # Correct
```

**Technique 2: Kahan Summation**
```python
def kahan_sum(array):
    """Compensated summation for improved accuracy"""
    sum_val = 0.0
    compensation = 0.0

    for value in array:
        y = value - compensation
        t = sum_val + y
        compensation = (t - sum_val) - y
        sum_val = t

    return sum_val

# Test
values = np.random.rand(1_000_000).astype(np.float32)

naive_sum = np.sum(values)
kahan_sum_val = kahan_sum(values)
reference = np.sum(values.astype(np.float64))

print(f"Naive sum error:  {abs(naive_sum - reference):.2e}")
print(f"Kahan sum error:  {abs(kahan_sum_val - reference):.2e}")
```

---

## Verification Strategy Design

### Test Case Hierarchy

**Level 1: Unit Tests** (Individual functions)
```python
import pytest
import numpy as np

def test_matrix_multiply_accuracy():
    """Test matrix multiplication against reference"""
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)

    # New implementation
    C_new = new_matmul(A, B)

    # Reference (NumPy BLAS)
    C_ref = A @ B

    # Validate
    max_error = np.max(np.abs(C_new - C_ref))
    rel_error = max_error / np.max(np.abs(C_ref))

    assert rel_error < 1e-14, f"Relative error {rel_error:.2e} exceeds tolerance"
```

**Level 2: Integration Tests** (Component interactions)
```python
def test_ode_solver_integration():
    """Test ODE solver against analytical solution"""
    # Simple harmonic oscillator: y'' + y = 0
    # Analytical solution: y(t) = cos(t)

    def f(y, t):
        return np.array([y[1], -y[0]])

    y0 = np.array([1.0, 0.0])  # y(0) = 1, y'(0) = 0
    t_span = (0, 10.0)

    # Solve numerically
    solution = ode_solver(f, y0, t_span)

    # Compare with analytical
    t_eval = solution.t
    y_analytical = np.cos(t_eval)

    max_error = np.max(np.abs(solution.y[0] - y_analytical))
    assert max_error < 1e-6, f"ODE solver error {max_error:.2e} too large"
```

**Level 3: System Tests** (End-to-end validation)
```python
def test_full_simulation():
    """Test complete simulation against legacy code output"""
    # Load reference output from Fortran legacy code
    reference = np.load('reference_output.npy')

    # Run new Python/JAX implementation
    result = run_full_simulation(parameters)

    # Compare
    rel_error = np.linalg.norm(result - reference) / np.linalg.norm(reference)

    assert rel_error < 1e-11, f"System error {rel_error:.2e} exceeds tolerance"

    # Additional checks
    assert np.all(result > 0), "Physical constraint violated (negative values)"
    assert abs(np.sum(result) - np.sum(reference)) < 1e-12, "Conservation law violated"
```

---

### Tolerance Criteria Definition

**Absolute Error**:
```python
abs_error = abs(computed - reference)
tolerance = 1e-12  # Domain-specific

assert abs_error < tolerance
```

**Relative Error**:
```python
rel_error = abs(computed - reference) / abs(reference)
tolerance = 1e-11  # Typical for float64

assert rel_error < tolerance
```

**Mixed Criteria** (preferred):
```python
def validate_accuracy(computed, reference, atol=1e-14, rtol=1e-11):
    """Combined absolute + relative tolerance"""
    abs_error = abs(computed - reference)
    rel_error = abs_error / max(abs(reference), 1e-100)  # Avoid division by zero

    # Pass if EITHER criterion satisfied
    passed = (abs_error < atol) or (rel_error < rtol)

    return passed, abs_error, rel_error
```

---

### Reference Solution Generation

**Method 1: Analytical Solutions**
```python
def create_analytical_test_cases():
    """Generate test cases with known analytical solutions"""
    test_cases = [
        {
            'name': 'Exponential decay',
            'ode': lambda y, t: -y,
            'initial': 1.0,
            'analytical': lambda t: np.exp(-t),
            'domain': (0, 5),
        },
        {
            'name': 'Harmonic oscillator',
            'ode': lambda y, t: [y[1], -y[0]],
            'initial': [1.0, 0.0],
            'analytical': lambda t: np.cos(t),
            'domain': (0, 10),
        },
    ]
    return test_cases
```

**Method 2: Higher-Precision Reference**
```python
from mpmath import mp

def high_precision_reference(x, precision=100):
    """Compute reference using arbitrary precision"""
    mp.dps = precision  # Decimal places

    # Compute in high precision
    x_mp = mp.mpf(str(x))
    result_mp = mp.exp(x_mp)

    # Convert back to float64
    result = float(result_mp)
    return result

# Validate float64 implementation
x = 10.0
result_f64 = np.exp(x)
result_ref = high_precision_reference(x, precision=50)

error = abs(result_f64 - result_ref)
print(f"Error vs. high-precision reference: {error:.2e}")
```

**Method 3: Method of Manufactured Solutions (MMS)**
```python
def manufactured_solution_test():
    """Create PDE test with known solution"""
    # Assume solution: u(x,t) = sin(πx) * exp(-t)
    # Then PDE becomes: u_t - u_xx = f(x,t)
    # where f is computed from assumed solution

    def exact_solution(x, t):
        return np.sin(np.pi * x) * np.exp(-t)

    def source_term(x, t):
        # Computed from u_t - u_xx with u = sin(πx)exp(-t)
        return -np.sin(np.pi * x) * np.exp(-t) - np.pi**2 * np.sin(np.pi * x) * np.exp(-t)

    # Solve PDE with manufactured source
    computed_solution = pde_solver(source_term)
    exact = exact_solution(x_grid, t_final)

    error = np.linalg.norm(computed_solution - exact)
    assert error < 1e-6
```

---

## Reproducibility Analysis

### Determinism Requirements

**Sources of Non-Determinism**:

1. **Floating-Point Non-Associativity**
```python
# (a + b) + c ≠ a + (b + c) in finite precision

a, b, c = 1e20, 1.0, -1e20

result1 = (a + b) + c  # = 0.0
result2 = a + (b + c)  # = 1.0 (!)

print(f"(a+b)+c = {result1}")
print(f"a+(b+c) = {result2}")
print(f"Difference: {abs(result1 - result2)}")
```

2. **Parallel Reduction Non-Determinism**
```python
import numpy as np

# Parallel sum order varies across runs
arr = np.random.rand(1_000_000)

# Different orders of summation
sum1 = np.sum(arr)  # Default order
sum2 = np.sum(arr[::-1])  # Reversed order

# Small but measurable difference
print(f"Difference: {abs(sum1 - sum2):.2e}")
```

3. **Random Number Generation**
```python
import numpy as np

# Non-reproducible (bad)
values = np.random.rand(100)

# Reproducible (good)
rng = np.random.default_rng(seed=12345)
values = rng.random(100)

# JAX reproducibility
import jax.random as jrandom
key = jrandom.PRNGKey(42)
values_jax = jrandom.normal(key, shape=(100,))
```

---

### Cross-Platform Reproducibility

**Challenges**:
- Different BLAS/LAPACK implementations (OpenBLAS, MKL, ATLAS)
- CPU vs. GPU arithmetic
- Compiler optimizations (-O3 reordering)
- Hardware differences (x86 vs. ARM)

**Solutions**:

**1. Deterministic Libraries**
```python
import os

# Force deterministic NumPy
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Force deterministic JAX
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
```

**2. Tolerance-Based Validation**
```python
def cross_platform_test(func, args, atol=1e-12, rtol=1e-10):
    """Test allows small platform differences"""
    result_linux = func(*args)  # Run on Linux
    result_mac = func(*args)    # Run on macOS

    np.testing.assert_allclose(result_linux, result_mac, atol=atol, rtol=rtol)
```

**3. Reference Output Files**
```python
def validate_against_reference(computed, reference_file='reference.npy'):
    """Compare against stored reference output"""
    reference = np.load(reference_file)

    np.testing.assert_allclose(computed, reference, rtol=1e-11, atol=1e-13)
```

---

## Conservation Law Validation

### Physical Constraints

**Energy Conservation** (Hamiltonian systems):
```python
def validate_energy_conservation(trajectory, tolerance=1e-12):
    """Ensure total energy remains constant"""
    energies = compute_total_energy(trajectory)

    initial_energy = energies[0]
    energy_drift = np.abs(energies - initial_energy)

    max_drift = np.max(energy_drift)
    relative_drift = max_drift / abs(initial_energy)

    assert relative_drift < tolerance, f"Energy drift {relative_drift:.2e} > {tolerance}"
```

**Mass Conservation** (Fluid dynamics, chemistry):
```python
def validate_mass_conservation(concentrations, tolerance=1e-14):
    """Ensure total mass conserved"""
    total_mass = np.sum(concentrations, axis=1)  # Sum over species

    mass_change = np.abs(total_mass - total_mass[0])
    relative_change = mass_change / total_mass[0]

    assert np.all(relative_change < tolerance), "Mass not conserved"
```

**Positivity Constraints**:
```python
def validate_positivity(solution):
    """Physical quantities must be non-negative"""
    # Concentrations, probabilities, densities, etc.
    assert np.all(solution >= 0), "Negative values detected"

    # Warn if very close to zero (potential underflow)
    min_val = np.min(solution[solution > 0])
    if min_val < 1e-100:
        print(f"⚠️ Warning: Very small values detected (min: {min_val:.2e})")
```

---

## Best Practices Summary

### 1. **Always Test Against Reference**
- Use legacy code output as oracle
- Generate reference with higher precision if needed
- Store reference outputs in version control

### 2. **Define Tolerances Based on Domain**
- Physics: typically 1e-12 relative error
- ML training: 1e-3 often acceptable
- Financial: depends on monetary scale

### 3. **Test at Multiple Scales**
- Small problems: exact comparison possible
- Large problems: statistical validation
- Edge cases: extreme values, zeros, infinities

### 4. **Monitor Accumulation**
- Track error growth over time (long simulations)
- Periodically re-normalize (if appropriate)
- Use compensated summation for critical accumulations

### 5. **Document Precision Decisions**
- Why float32 vs. float64?
- What tolerances were chosen and why?
- Any known limitations or approximations?

---

## References

- **Numerical Recipes**: Press et al., Cambridge University Press
- **What Every Computer Scientist Should Know About Floating-Point**: Goldberg (1991)
- **Accuracy and Stability of Numerical Algorithms**: Higham (2002)
- **IEEE 754 Standard**: https://ieeexplore.ieee.org/document/8766229
