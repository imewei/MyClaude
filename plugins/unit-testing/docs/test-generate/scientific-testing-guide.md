# Scientific Testing Guide

Comprehensive patterns for numerical correctness validation, tolerance-based assertions, gradient verification for JAX/PyTorch, property-based testing for scientific code, performance benchmarks, and convergence test patterns.

## Numerical Correctness Validation

### Floating Point Comparison

```python
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

def test_numerical_computation():
    """Test numerical accuracy with appropriate tolerance"""

    # ❌ Bad: Exact comparison (fails due to floating point precision)
    result = compute_mean([1.0, 2.0, 3.0])
    assert result == 2.0  # May fail!

    # ✅ Good: Tolerance-based comparison
    assert_allclose(result, 2.0, rtol=1e-10, atol=1e-12)

    # For arrays
    expected = np.array([1.0, 2.0, 3.0])
    assert_allclose(result_array, expected, rtol=1e-10)
```

### Tolerance Guidelines

```python
# Machine precision
EPSILON_FLOAT32 = 1e-6   # Single precision
EPSILON_FLOAT64 = 1e-15  # Double precision

# Recommended tolerances
TOLERANCES = {
    'strict': {'rtol': 1e-12, 'atol': 1e-14},      # Analytical solutions
    'standard': {'rtol': 1e-7, 'atol': 1e-9},      # Normal numerical tests
    'relaxed': {'rtol': 1e-4, 'atol': 1e-6},       # Iterative methods
    'loose': {'rtol': 1e-2, 'atol': 1e-3}          # Convergence tests
}

def test_with_appropriate_tolerance():
    # Analytical solution: strict tolerance
    assert_allclose(result, exact_solution, **TOLERANCES['strict'])

    # Numerical solution: standard tolerance
    assert_allclose(result, expected, **TOLERANCES['standard'])

    # Iterative solver: relaxed tolerance
    assert_allclose(result, converged_value, **TOLERANCES['relaxed'])
```

### Analytical Solution Tests

```python
def test_polynomial_integration():
    """Test against known analytical solution"""

    # f(x) = x^2, integral from 0 to 1 = 1/3
    def f(x):
        return x**2

    result = numerical_integrate(f, 0, 1)
    analytical = 1.0 / 3.0

    assert_allclose(result, analytical, rtol=1e-10,
                   err_msg="Integration does not match analytical solution")


def test_exponential_function():
    """Test exponential with known values"""

    # e^0 = 1
    assert_allclose(exp_func(0), 1.0, rtol=1e-12)

    # e^1 = e
    assert_allclose(exp_func(1), np.e, rtol=1e-12)

    # e^-inf = 0
    assert_allclose(exp_func(-np.inf), 0.0, rtol=1e-12)
```

## Property-Based Testing with Hypothesis

### Basic Property Tests

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np

@given(
    data=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 100)),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False
        )
    )
)
@settings(max_examples=100, deadline=None)
def test_function_properties(data):
    """Test mathematical properties hold for all inputs"""

    result = my_function(data)

    # Property 1: Output is finite
    assert np.all(np.isfinite(result)), "Result contains inf/nan"

    # Property 2: Output shape matches input
    assert result.shape == data.shape, "Shape mismatch"

    # Property 3: Output is non-negative
    assert np.all(result >= 0), "Result contains negative values"
```

### Linearity Property

```python
@given(
    x=arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 50))),
    y=arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 50))),
    a=st.floats(min_value=-10, max_value=10, allow_nan=False),
    b=st.floats(min_value=-10, max_value=10, allow_nan=False)
)
def test_linearity(x, y, a, b):
    """Test: f(aX + bY) = af(X) + bf(Y) for linear operations"""

    # Ensure same shape
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    left = linear_function(a * x + b * y)
    right = a * linear_function(x) + b * linear_function(y)

    assert_allclose(left, right, rtol=1e-10)
```

### Idempotency Property

```python
@given(data=arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 50))))
def test_idempotent(data):
    """Test: f(f(x)) = f(x) for idempotent operations"""

    result1 = idempotent_function(data)
    result2 = idempotent_function(result1)

    assert_allclose(result1, result2, rtol=1e-10)
```

### Commutativity Property

```python
@given(
    x=arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 50))),
    y=arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 50)))
)
def test_commutative(x, y):
    """Test: f(x, y) = f(y, x) for commutative operations"""

    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    result_xy = commutative_function(x, y)
    result_yx = commutative_function(y, x)

    assert_allclose(result_xy, result_yx, rtol=1e-10)
```

## JAX Gradient Verification

### Gradient Correctness Tests

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

def test_gradient_correctness():
    """Test gradient using finite differences"""

    def loss_function(x):
        return jnp.sum(my_function(x))

    # Analytical gradient
    grad_fn = grad(loss_function)
    x = jnp.array([1.0, 2.0, 3.0])
    analytical_grad = grad_fn(x)

    # Numerical gradient (finite differences)
    epsilon = 1e-5
    numerical_grad = jnp.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.at[i].add(epsilon)
        x_minus = x.at[i].add(-epsilon)

        grad_approx = (loss_function(x_plus) - loss_function(x_minus)) / (2 * epsilon)
        numerical_grad = numerical_grad.at[i].set(grad_approx)

    # Compare gradients
    assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6)
```

### JIT Compilation Equivalence

```python
def test_jit_equivalence():
    """Test that JIT-compiled version produces same results"""

    input_data = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    # Non-JIT version
    result_nojit = my_function(input_data)

    # JIT version
    jitted_fn = jit(my_function)
    result_jit = jitted_fn(input_data)

    assert_allclose(result_nojit, result_jit, rtol=1e-12)
```

### vmap Correctness

```python
def test_vmap_correctness():
    """Test vectorization with vmap"""

    # Single input
    single_input = jnp.array([1.0, 2.0, 3.0])
    single_result = my_function(single_input)

    # Batched input
    batch_input = jnp.stack([single_input] * 5)
    vmapped_fn = vmap(my_function)
    batch_result = vmapped_fn(batch_input)

    # Check all batch results match single result
    for i in range(5):
        assert_allclose(batch_result[i], single_result, rtol=1e-12)
```

## PyTorch Gradient Tests

```python
import torch

def test_pytorch_gradient():
    """Test PyTorch autograd gradients"""

    def loss_fn(x):
        return torch.sum(my_torch_function(x))

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Compute gradient
    loss = loss_fn(x)
    loss.backward()
    analytical_grad = x.grad.clone()

    # Numerical gradient
    x.grad.zero_()
    epsilon = 1e-5
    numerical_grad = torch.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.clone().detach()
        x_plus[i] += epsilon

        x_minus = x.clone().detach()
        x_minus[i] -= epsilon

        grad_approx = (loss_fn(x_plus) - loss_fn(x_minus)) / (2 * epsilon)
        numerical_grad[i] = grad_approx

    # Compare
    assert torch.allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6)
```

## Convergence Testing

### Iterative Solver Convergence

```python
def test_convergence_rate():
    """Test iterative solver converges at expected rate"""

    def solve_iterative(A, b, max_iter=100):
        """Iterative solver returning all iterations"""
        x = np.zeros_like(b)
        residuals = []

        for i in range(max_iter):
            x_new = iteration_step(A, b, x)
            residual = np.linalg.norm(A @ x_new - b)
            residuals.append(residual)

            if residual < 1e-10:
                break

            x = x_new

        return x, residuals

    # Test problem
    A = np.random.randn(10, 10)
    A = A.T @ A  # Make positive definite
    b = np.random.randn(10)

    x, residuals = solve_iterative(A, b)

    # Check convergence
    assert residuals[-1] < 1e-8, "Did not converge"

    # Check convergence rate (should be exponential)
    if len(residuals) > 10:
        rate = residuals[-1] / residuals[-2]
        assert rate < 0.9, "Convergence too slow"
```

### Newton's Method Convergence

```python
def test_newton_quadratic_convergence():
    """Test Newton's method achieves quadratic convergence"""

    def newton_solve(f, df, x0, max_iter=20):
        errors = []
        x = x0

        for _ in range(max_iter):
            x_new = x - f(x) / df(x)
            error = abs(x_new - x)
            errors.append(error)

            if error < 1e-12:
                break

            x = x_new

        return x, errors

    # Test function: f(x) = x^2 - 2 (root at sqrt(2))
    f = lambda x: x**2 - 2
    df = lambda x: 2*x

    x, errors = newton_solve(f, df, x0=1.0)

    # Check quadratic convergence: error_{n+1} ≈ C * error_n^2
    if len(errors) > 3:
        for i in range(len(errors) - 2):
            if errors[i] > 1e-6:  # Only check before very small errors
                ratio = errors[i+1] / (errors[i]**2)
                assert ratio < 10, "Not quadratic convergence"
```

## Performance Benchmarks

### Pytest Benchmarks

```python
import pytest

@pytest.mark.benchmark
def test_performance(benchmark):
    """Benchmark function performance"""

    input_data = np.random.randn(1000, 1000)

    result = benchmark(my_function, input_data)

    assert result is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("size", [10, 100, 1000, 10000])
def test_scaling(benchmark, size):
    """Test performance scaling with input size"""

    input_data = np.random.randn(size)

    result = benchmark(my_function, input_data)

    assert result is not None
```

### Memory Profiling

```python
import tracemalloc

def test_memory_usage():
    """Test memory usage is reasonable"""

    tracemalloc.start()

    input_data = np.random.randn(10000)
    _ = my_function(input_data)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Assert peak memory is reasonable (< 100 MB)
    assert peak < 100 * 1024 * 1024, f"Peak memory {peak} bytes too high"
```

## Numerical Stability Tests

### Condition Number Tests

```python
def test_numerical_stability():
    """Test numerical stability with ill-conditioned problems"""

    # Create ill-conditioned matrix
    U, s, Vt = np.linalg.svd(np.random.randn(10, 10))
    s_ill = np.array([1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
    A = U @ np.diag(s_ill) @ Vt

    # Test solution
    b = np.random.randn(10)
    x = solve(A, b)

    # Check residual (should still be small even if solution isn't accurate)
    residual = np.linalg.norm(A @ x - b)
    assert residual < 1e-6, "Unstable solver"
```

### Catastrophic Cancellation Detection

```python
def test_cancellation_stability():
    """Test for catastrophic cancellation"""

    # Computing (1 + x) - 1 for small x should use log1p
    small_values = np.array([1e-10, 1e-12, 1e-14])

    for x in small_values:
        # ❌ Bad: loses precision
        bad_result = (1 + x) - 1

        # ✅ Good: specialized function
        good_result = np.log1p(x) - np.log(1)  # Or use expm1

        # Good result should be more accurate
        relative_error_bad = abs(bad_result - x) / x
        assert relative_error_bad > 0.1, "Should detect cancellation"
```

## Best Practices

1. **Use tolerance-based assertions** for floating point comparisons
2. **Test against analytical solutions** when available
3. **Verify gradients** with finite differences
4. **Use property-based testing** for mathematical properties
5. **Test convergence rates** for iterative methods
6. **Benchmark performance** and track regressions
7. **Test numerical stability** with ill-conditioned problems
8. **Check for inf/nan** in all outputs
9. **Verify shape consistency** between inputs and outputs
10. **Test with edge values** (zero, very large, very small)
