---
description: Generate comprehensive test suites for Python, Julia, and JAX scientific computing projects with numerical validation, property-based testing, and performance benchmarks
allowed-tools: Bash(find:*), Bash(grep:*), Bash(python:*), Bash(julia:*), Bash(pytest:*), Bash(git:*)
argument-hint: <source-file-or-module> [--coverage] [--property-based] [--benchmarks]
color: cyan
---

# Scientific Computing Test Suite Generator

## Phase 0: Project Discovery & Language Detection

### Repository Context
- Working directory: !`pwd`
- Project structure: !`find . -maxdepth 2 -type d 2>/dev/null | head -20`
- Git repository: !`git rev-parse --show-toplevel 2>/dev/null || echo "Not a git repo"`

### Language & Framework Detection

#### Python Scientific Stack
- Python version: !`python --version 2>/dev/null || python3 --version 2>/dev/null || echo "Not found"`
- NumPy: !`python -c "import numpy; print(f'numpy {numpy.__version__}')" 2>/dev/null || echo "Not installed"`
- SciPy: !`python -c "import scipy; print(f'scipy {scipy.__version__}')" 2>/dev/null || echo "Not installed"`
- JAX: !`python -c "import jax; print(f'jax {jax.__version__}')" 2>/dev/null || echo "Not installed"`
- PyTorch: !`python -c "import torch; print(f'torch {torch.__version__}')" 2>/dev/null || echo "Not installed"`
- TensorFlow: !`python -c "import tensorflow as tf; print(f'tf {tf.__version__}')" 2>/dev/null || echo "Not installed"`

#### Testing Frameworks (Python)
- pytest: !`python -c "import pytest; print(f'pytest {pytest.__version__}')" 2>/dev/null || echo "Not installed"`
- hypothesis: !`python -c "import hypothesis; print(f'hypothesis {hypothesis.__version__}')" 2>/dev/null || echo "Not installed"`
- pytest-benchmark: !`python -c "import pytest_benchmark; print('installed')" 2>/dev/null || echo "Not installed"`
- pytest-cov: !`python -c "import pytest_cov; print('installed')" 2>/dev/null || echo "Not installed"`

#### Julia Environment
- Julia version: !`julia --version 2>/dev/null || echo "Not found"`
- Project.toml: @Project.toml
- Test dependencies: !`grep -A 10 "\[extras\]" Project.toml 2>/dev/null | grep -i test`
- Test directory: !`find . -type d -name "test" 2>/dev/null | head -3`

#### JAX-Specific Detection
- JAX devices: !`python -c "import jax; print(f'Devices: {jax.devices()}')" 2>/dev/null || echo "JAX not available"`
- JAX backend: !`python -c "import jax; print(f'Backend: {jax.default_backend()}')" 2>/dev/null`
- GPU availability: !`python -c "import jax; print(f'GPU: {any(d.platform == \"gpu\" for d in jax.devices())}')" 2>/dev/null`

### Source Code Analysis
**Target**: `$ARGUMENTS`

- Source files: !`find ${ARGUMENTS:-.} -name "*.py" -o -name "*.jl" 2>/dev/null | head -20`
- Line count: !`find ${ARGUMENTS:-.} -name "*.py" -o -name "*.jl" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1`
- Existing tests: !`find ${ARGUMENTS:-.} -name "*test*.py" -o -name "*test*.jl" 2>/dev/null | wc -l`

---

## Phase 1: Code Analysis & Test Planning

### Multi-Agent Code Analysis System

#### Agent 1: AST Parser & Function Extractor
**Mission**: Parse source code and extract all testable units

**Python Analysis**:
```python
import ast
import inspect

class FunctionExtractor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.classes = []

    def visit_FunctionDef(self, node):
        """Extract function metadata"""
        func_info = {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
            'returns': ast.unparse(node.returns) if node.returns else None,
            'docstring': ast.get_docstring(node),
            'lineno': node.lineno,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'complexity': self.calculate_complexity(node)
        }
        self.functions.append(func_info)

    def visit_ClassDef(self, node):
        """Extract class metadata"""
        class_info = {
            'name': node.name,
            'bases': [ast.unparse(base) for base in node.bases],
            'methods': [],
            'docstring': ast.get_docstring(node),
            'lineno': node.lineno
        }
        self.classes.append(class_info)
```

**Extraction Tasks**:
1. **Functions**: Name, signature, return type, docstring, complexity
2. **Classes**: Name, inheritance, methods, properties
3. **Constants**: Global constants and configurations
4. **Imports**: Dependencies and their usage
5. **Type Hints**: Input/output types for validation
6. **Decorators**: JIT, vectorization, custom decorators

**Julia Analysis**:
```julia
using MacroTools

function extract_functions(file_path)
    code = read(file_path, String)
    expr = Meta.parse("begin\n$code\nend")

    functions = []

    MacroTools.postwalk(expr) do ex
        if @capture(ex, function name_(args__) body__ end)
            push!(functions, Dict(
                :name => name,
                :args => args,
                :body => body,
                :lineno => get_line_number(ex)
            ))
        end
        ex
    end

    return functions
end
```

#### Agent 2: Numerical Analysis Specialist
**Mission**: Identify numerical computing patterns and requirements

**Pattern Detection**:

**1. Matrix Operations**
```python
patterns = {
    'matrix_multiplication': r'@|\.dot\(|matmul',
    'matrix_inversion': r'inv\(|solve\(',
    'eigenvalue': r'eig\(|eigvals\(|eigenvectors',
    'svd': r'svd\(',
    'qr': r'qr\(',
    'cholesky': r'cholesky\(',
    'lu': r'lu\('
}
```

**2. Iterative Methods**
```python
patterns = {
    'loops': r'for .* in range',
    'while_loops': r'while .*:',
    'convergence': r'converge|tolerance|eps|atol|rtol',
    'iterations': r'max_iter|n_iter|num_iterations'
}
```

**3. Numerical Integration/Differentiation**
```python
patterns = {
    'integration': r'integrate|quad|trapz|simpson',
    'differentiation': r'grad|gradient|jacobian|hessian',
    'autodiff': r'jax\.grad|torch\.autograd|tf\.GradientTape'
}
```

**4. Optimization**
```python
patterns = {
    'minimization': r'minimize|optimize|argmin',
    'constraints': r'constraint|bound',
    'objective': r'objective|loss|cost'
}
```

**5. Random Number Generation**
```python
patterns = {
    'random': r'random|rand|randn|randint',
    'seed': r'seed|random_state|rng',
    'distributions': r'normal|uniform|exponential|poisson'
}
```

**6. Parallel/Vectorized Operations**
```python
patterns = {
    'jax_vmap': r'jax\.vmap',
    'jax_pmap': r'jax\.pmap',
    'jax_jit': r'jax\.jit',
    'numpy_vectorize': r'np\.vectorize',
    'broadcasting': r'broadcast|[:, None]'
}
```

#### Agent 3: Test Requirements Analyzer
**Mission**: Determine what tests are needed using UltraThink intelligence

**UltraThink Analysis Framework**:

**1. Numerical Correctness Testing**

For each function, ask:
- **What is the mathematical truth?**
  - Is there a closed-form solution for simple cases?
  - Are there known analytical results to compare against?
  - Can we verify with independent implementation?

- **What edge cases exist?**
  - Zero inputs, negative values, infinity, NaN
  - Empty arrays, single element, very large arrays
  - Ill-conditioned matrices, singular matrices
  - Degenerate cases, boundary conditions

- **What numerical stability concerns exist?**
  - Catastrophic cancellation risks
  - Loss of precision in floating-point arithmetic
  - Accumulation of rounding errors
  - Overflow/underflow conditions

**Example Analysis**:
```python
def analyze_function(func_name, func_code):
    """
    Function: matrix_inversion(A)

    Test Requirements:

    1. CORRECTNESS:
       - Test: A @ inv(A) ≈ I (identity)
       - Test: inv(inv(A)) ≈ A
       - Test: det(inv(A)) ≈ 1/det(A)

    2. EDGE CASES:
       - Singular matrix (det=0) → should raise error
       - Near-singular (ill-conditioned) → warning
       - 1x1 matrix, 2x2 matrix, large matrix
       - Identity matrix → should return identity
       - Diagonal matrix → simple inversion

    3. NUMERICAL STABILITY:
       - Condition number test
       - Precision degradation test
       - Compare against stable algorithm (e.g., LU decomposition)

    4. PERFORMANCE:
       - Benchmark different matrix sizes
       - Compare with optimized libraries
       - Memory usage profiling

    5. TYPE CHECKING:
       - Input validation (square matrix)
       - Output type matches input type
       - Shape preservation
    """
```

**2. Property-Based Testing Requirements**

Identify mathematical properties:
```python
properties = {
    'matrix_multiply': [
        'associative: (A @ B) @ C = A @ (B @ C)',
        'distributive: A @ (B + C) = A @ B + A @ C',
        'identity: A @ I = I @ A = A',
        'dimension: (m,n) @ (n,p) = (m,p)'
    ],
    'matrix_transpose': [
        'involution: transpose(transpose(A)) = A',
        'product: transpose(A @ B) = transpose(B) @ transpose(A)',
        'shape: (m,n) → (n,m)'
    ],
    'optimization': [
        'convergence: f(x_{n+1}) ≤ f(x_n) for minimization',
        'stationary: gradient at minimum ≈ 0',
        'convex: local minimum is global minimum'
    ]
}
```

**3. JAX-Specific Testing**

For JAX code, additional requirements:
```python
jax_tests = {
    'gradient_correctness': [
        'finite_difference_check',
        'gradient_gradient_check (hessian)',
        'complex_step_derivative_check'
    ],
    'jit_equivalence': [
        'jitted and non-jitted produce same results',
        'compilation successful',
        'no python side effects'
    ],
    'vmap_correctness': [
        'vmap matches manual loop',
        'batch dimension handling',
        'nested vmap correctness'
    ],
    'device_placement': [
        'cpu/gpu equivalence',
        'dtype preservation',
        'memory layout consistency'
    ]
}
```

#### Agent 4: Test Data Generator
**Mission**: Create comprehensive test datasets and fixtures

**Test Data Categories**:

**1. Analytical Test Cases** (known exact solutions)
```python
analytical_cases = {
    'linear_solver': {
        'identity': (np.eye(3), np.array([1,2,3]), np.array([1,2,3])),
        'diagonal': (np.diag([2,3,4]), np.array([2,6,12]), np.array([1,2,3])),
        'simple_2x2': (np.array([[1,2],[3,4]]), np.array([5,11]), np.array([1,2]))
    },
    'eigenvalue': {
        'identity': (np.eye(3), np.ones(3)),  # eigenvalues all 1
        'diagonal': (np.diag([1,2,3]), np.array([1,2,3])),
        'symmetric': (np.array([[2,1],[1,2]]), np.array([1,3]))
    },
    'integration': {
        'polynomial': (lambda x: x**2, 0, 1, 1/3),  # ∫x²dx from 0 to 1
        'exponential': (lambda x: np.exp(x), 0, 1, np.e - 1),
        'sine': (lambda x: np.sin(x), 0, np.pi, 2)
    }
}
```

**2. Edge Case Test Data**
```python
edge_cases = {
    'arrays': [
        np.array([]),           # empty
        np.array([1]),          # single element
        np.array([1, 2, 3]),    # small
        np.random.randn(1000),  # large
        np.array([0, 0, 0]),    # all zeros
        np.array([np.inf, -np.inf, np.nan]),  # special values
    ],
    'matrices': [
        np.eye(3),              # identity
        np.zeros((3, 3)),       # zero matrix
        np.ones((3, 3)),        # ones
        np.diag([1, 2, 3]),     # diagonal
        np.array([[1, 2], [2, 4]]),  # singular
        np.random.randn(100, 100) + 1e10 * np.eye(100),  # ill-conditioned
    ],
    'scalars': [0, 1, -1, 1e-10, 1e10, np.inf, -np.inf, np.nan]
}
```

**3. Property-Based Test Strategies**
```python
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

@given(
    matrix=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 10), st.integers(1, 10)),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False
        )
    )
)
def test_matrix_property(matrix):
    """Property-based test template"""
    pass
```

**4. Benchmark Datasets**
```python
benchmark_datasets = {
    'small': {'size': 10, 'iterations': 10000},
    'medium': {'size': 100, 'iterations': 1000},
    'large': {'size': 1000, 'iterations': 100},
    'xlarge': {'size': 10000, 'iterations': 10}
}
```

---

## Phase 2: Test Suite Generation

### Test Template System

#### Python/JAX Test Template

```python
"""
Test suite for {module_name}

Generated by generate-tests command
Date: {date}
Coverage target: >90%
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import warnings

# Import module under test
from {module_path} import {functions}

# Optional: JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Optional: Property-based testing
try:
    from hypothesis import given
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rng():
    """Random number generator with fixed seed for reproducibility"""
    return np.random.RandomState(42)


@pytest.fixture
def sample_data(rng):
    """Sample test data"""
    return {
        'vector': rng.randn(10),
        'matrix': rng.randn(5, 5),
        'scalar': 3.14159
    }


# ============================================================================
# Unit Tests - Basic Functionality
# ============================================================================

class Test{ClassName}:
    """Test suite for {ClassName}"""

    def test_{function_name}_basic(self):
        """Test basic functionality of {function_name}"""
        # Arrange
        input_data = {test_input}
        expected = {expected_output}

        # Act
        result = {function_name}(input_data)

        # Assert
        assert_allclose(result, expected, rtol=1e-7, atol=1e-10)

    def test_{function_name}_shape(self):
        """Test output shape is correct"""
        input_data = np.random.randn({input_shape})
        result = {function_name}(input_data)
        assert result.shape == {expected_shape}

    def test_{function_name}_dtype(self):
        """Test output dtype matches input"""
        for dtype in [np.float32, np.float64]:
            input_data = np.array({test_input}, dtype=dtype)
            result = {function_name}(input_data)
            assert result.dtype == dtype


# ============================================================================
# Edge Case Tests
# ============================================================================

class Test{ClassName}EdgeCases:
    """Edge case tests for {ClassName}"""

    def test_{function_name}_empty_input(self):
        """Test behavior with empty input"""
        empty = np.array([])
        with pytest.raises(ValueError, match="empty"):
            {function_name}(empty)

    def test_{function_name}_single_element(self):
        """Test with single element input"""
        single = np.array([1.0])
        result = {function_name}(single)
        assert result.shape == (1,)

    def test_{function_name}_zero_input(self):
        """Test with all zeros"""
        zeros = np.zeros({shape})
        result = {function_name}(zeros)
        # Add appropriate assertion

    def test_{function_name}_large_values(self):
        """Test numerical stability with large values"""
        large = np.array([1e10, 1e11, 1e12])
        result = {function_name}(large)
        assert np.all(np.isfinite(result)), "Result contains inf/nan"

    def test_{function_name}_small_values(self):
        """Test numerical stability with small values"""
        small = np.array([1e-10, 1e-11, 1e-12])
        result = {function_name}(small)
        assert np.all(np.isfinite(result)), "Result contains inf/nan"

    @pytest.mark.parametrize("special_value", [np.inf, -np.inf, np.nan])
    def test_{function_name}_special_values(self, special_value):
        """Test handling of inf and nan"""
        data = np.array([1.0, special_value, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = {function_name}(data)
        # Add appropriate assertion or exception check


# ============================================================================
# Numerical Correctness Tests
# ============================================================================

class Test{ClassName}Correctness:
    """Numerical correctness tests"""

    def test_{function_name}_analytical_solution(self):
        """Test against known analytical solution"""
        # Test case with known exact solution
        input_data = {analytical_input}
        expected = {analytical_output}
        result = {function_name}(input_data)

        assert_allclose(result, expected, rtol=1e-12, atol=1e-14,
                       err_msg="Result doesn't match analytical solution")

    def test_{function_name}_reference_implementation(self):
        """Test against reference implementation"""
        import scipy  # or other reference library

        input_data = np.random.randn({shape})
        our_result = {function_name}(input_data)
        ref_result = {reference_function}(input_data)

        assert_allclose(our_result, ref_result, rtol=1e-10,
                       err_msg="Differs from reference implementation")

    def test_{function_name}_convergence_rate(self):
        """Test convergence rate for iterative methods"""
        # Only for iterative algorithms
        errors = []
        for n in [10, 20, 40, 80]:
            result = {function_name}({inputs}, max_iter=n)
            error = compute_error(result, {true_solution})
            errors.append(error)

        # Check convergence
        for i in range(len(errors) - 1):
            assert errors[i+1] < errors[i], "Not converging"

        # Check convergence rate
        rate = np.log(errors[-1] / errors[-2]) / np.log(0.5)
        assert rate > {expected_rate}, f"Convergence rate {rate} too slow"


# ============================================================================
# Property-Based Tests
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class Test{ClassName}Properties:
    """Property-based tests using Hypothesis"""

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
    def test_{function_name}_idempotent(self, data):
        """Test: f(f(x)) = f(x) for idempotent operations"""
        result1 = {function_name}(data)
        result2 = {function_name}(result1)
        assert_allclose(result1, result2, rtol=1e-10)

    @given(
        matrix=arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(2, 20), st.integers(2, 20)),
            elements=st.floats(min_value=-100, max_value=100,
                             allow_nan=False, allow_infinity=False)
        )
    )
    def test_{function_name}_linearity(self, matrix):
        """Test: f(aX + bY) = af(X) + bf(Y) for linear operations"""
        a, b = 2.0, 3.0
        X = matrix[:, 0]
        Y = matrix[:, 1] if matrix.shape[1] > 1 else matrix[:, 0]

        left = {function_name}(a * X + b * Y)
        right = a * {function_name}(X) + b * {function_name}(Y)

        assert_allclose(left, right, rtol=1e-10)


# ============================================================================
# JAX-Specific Tests
# ============================================================================

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class Test{ClassName}JAX:
    """JAX-specific tests"""

    def test_{function_name}_jit_equivalence(self):
        """Test that JIT-compiled version produces same results"""
        input_data = jnp.array({test_input})

        # Non-JIT version
        result_nojit = {function_name}(input_data)

        # JIT version
        jitted_fn = jit({function_name})
        result_jit = jitted_fn(input_data)

        assert_allclose(result_nojit, result_jit, rtol=1e-12)

    def test_{function_name}_gradient_correctness(self):
        """Test gradient using finite differences"""
        def fn(x):
            return jnp.sum({function_name}(x))

        # Analytical gradient
        grad_fn = grad(fn)
        x = jnp.array({test_input})
        analytical_grad = grad_fn(x)

        # Finite difference gradient
        epsilon = 1e-5
        numerical_grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.at[i].add(epsilon)
            x_minus = x.at[i].add(-epsilon)
            numerical_grad[i] = (fn(x_plus) - fn(x_minus)) / (2 * epsilon)

        assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6)

    def test_{function_name}_vmap_correctness(self):
        """Test vectorization with vmap"""
        # Single input
        single_input = jnp.array({single_input})
        single_result = {function_name}(single_input)

        # Batched input
        batch_input = jnp.stack([single_input] * 5)
        vmapped_fn = vmap({function_name})
        batch_result = vmapped_fn(batch_input)

        # Check all batch results match single result
        for i in range(5):
            assert_allclose(batch_result[i], single_result, rtol=1e-12)

    def test_{function_name}_device_consistency(self):
        """Test consistency across CPU/GPU"""
        input_data = jnp.array({test_input})

        results = {}
        for device in jax.devices():
            with jax.default_device(device):
                results[device.platform] = {function_name}(input_data)

        # Compare all device results
        platforms = list(results.keys())
        for i in range(len(platforms) - 1):
            assert_allclose(results[platforms[i]], results[platforms[i+1]],
                          rtol=1e-12, atol=1e-14)


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.benchmark
class Test{ClassName}Performance:
    """Performance benchmarks"""

    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_{function_name}_performance(self, benchmark, size):
        """Benchmark performance across different input sizes"""
        input_data = np.random.randn(size)
        result = benchmark({function_name}, input_data)
        assert result is not None

    def test_{function_name}_memory_usage(self):
        """Test memory usage is reasonable"""
        import tracemalloc

        tracemalloc.start()
        input_data = np.random.randn(1000)
        _ = {function_name}(input_data)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Assert peak memory is reasonable (adjust threshold)
        assert peak < 10 * 1024 * 1024, f"Peak memory {peak} bytes too high"


# ============================================================================
# Integration Tests
# ============================================================================

class Test{ClassName}Integration:
    """Integration tests combining multiple functions"""

    def test_{function_name}_pipeline(self):
        """Test full processing pipeline"""
        # Test realistic workflow
        input_data = {realistic_input}

        # Step 1
        intermediate1 = {function1}(input_data)
        assert intermediate1 is not None

        # Step 2
        intermediate2 = {function2}(intermediate1)
        assert intermediate2 is not None

        # Final result
        result = {function3}(intermediate2)
        assert result is not None
        # Add specific assertions


# ============================================================================
# Regression Tests
# ============================================================================

class Test{ClassName}Regression:
    """Regression tests for known issues"""

    def test_{function_name}_issue_123(self):
        """Regression test for issue #123: {issue_description}"""
        # Test case that previously failed
        input_data = {failing_input}
        result = {function_name}(input_data)
        # Should not raise or produce incorrect result
        assert {condition}
```

#### Julia Test Template

```julia
"""
Test suite for {module_name}

Generated by generate-tests command
Date: {date}
"""

using Test
using {ModuleName}
using LinearAlgebra
using Random
using Statistics

# Optional test packages
if isdefined(Main, :BenchmarkTools)
    using BenchmarkTools
end

# ============================================================================
# Test Set: Basic Functionality
# ============================================================================

@testset "{ModuleName} - Basic Tests" begin
    @testset "{function_name} basic" begin
        # Arrange
        input_data = {test_input}
        expected = {expected_output}

        # Act
        result = {function_name}(input_data)

        # Assert
        @test result ≈ expected atol=1e-10 rtol=1e-7
    end

    @testset "{function_name} type stability" begin
        input_data = {test_input}
        @inferred {function_name}(input_data)
    end

    @testset "{function_name} shape" begin
        input_data = randn({input_shape})
        result = {function_name}(input_data)
        @test size(result) == {expected_shape}
    end
end

# ============================================================================
# Test Set: Edge Cases
# ============================================================================

@testset "{ModuleName} - Edge Cases" begin
    @testset "{function_name} empty input" begin
        empty = Float64[]
        @test_throws ArgumentError {function_name}(empty)
    end

    @testset "{function_name} single element" begin
        single = [1.0]
        result = {function_name}(single)
        @test length(result) == 1
    end

    @testset "{function_name} special values" begin
        @test isfinite({function_name}([0.0, 0.0, 0.0]))
        @test_throws DomainError {function_name}([Inf, -Inf, NaN])
    end
end

# ============================================================================
# Test Set: Numerical Correctness
# ============================================================================

@testset "{ModuleName} - Numerical Correctness" begin
    @testset "{function_name} analytical solution" begin
        # Known exact solution
        input_data = {analytical_input}
        expected = {analytical_output}
        result = {function_name}(input_data)

        @test result ≈ expected atol=1e-14 rtol=1e-12
    end

    @testset "{function_name} mathematical properties" begin
        A = randn(5, 5)
        B = randn(5, 5)

        # Test specific properties
        # Example: {function_name}(A + B) ≈ {function_name}(A) + {function_name}(B)
        @test {property_test}
    end
end

# ============================================================================
# Test Set: Performance
# ============================================================================

if isdefined(Main, :BenchmarkTools)
    @testset "{ModuleName} - Performance" begin
        @testset "{function_name} benchmarks" begin
            for size in [10, 100, 1000]
                input_data = randn(size)
                @btime {function_name}($input_data)
            end
        end
    end
end

# ============================================================================
# Test Set: Gradient Tests (for differentiable functions)
# ============================================================================

if isdefined(Main, :ForwardDiff) || isdefined(Main, :Zygote)
    @testset "{ModuleName} - Gradient Tests" begin
        using ForwardDiff

        @testset "{function_name} gradient correctness" begin
            x = randn(10)
            f(x) = sum({function_name}(x))

            # Automatic differentiation
            auto_grad = ForwardDiff.gradient(f, x)

            # Finite differences
            function finite_diff_grad(f, x; ε=1e-5)
                grad = similar(x)
                for i in 1:length(x)
                    x_plus = copy(x); x_plus[i] += ε
                    x_minus = copy(x); x_minus[i] -= ε
                    grad[i] = (f(x_plus) - f(x_minus)) / (2ε)
                end
                return grad
            end

            numerical_grad = finite_diff_grad(f, x)

            @test auto_grad ≈ numerical_grad rtol=1e-4
        end
    end
end
```

---

## Phase 3: UltraThink Test Quality Analysis

### Deep Reasoning About Test Quality

**1. Coverage vs. Quality Tradeoff**

```
Question: What makes a good test suite for scientific computing?

BREADTH (Coverage):
- Test all functions ✓
- Test all branches ✓
- Test all edge cases ✓

DEPTH (Quality):
- Test numerical correctness ✓✓✓
- Test mathematical properties ✓✓✓
- Test stability and precision ✓✓✓
- Test performance characteristics ✓✓

BALANCE:
→ 70% focus on numerical correctness
→ 20% focus on edge cases and robustness
→ 10% focus on performance

Rationale: In scientific computing, wrong answers are worse
than slow answers. Correctness is paramount.
```

**2. Test Independence vs. Realism**

```
Tension: Unit tests should be independent, but scientific
         workflows are inherently composed pipelines

Resolution:
- Unit tests: Test individual numerical kernels
  → Pure functions, no state, reproducible

- Integration tests: Test composed operations
  → Realistic workflows, data pipelines

- End-to-end tests: Test full simulations
  → Verify against known benchmarks

Example:
Unit: test_matrix_multiply(A, B)
Integration: test_solve_linear_system(A, b)  # uses multiply internally
E2E: test_finite_element_solver(mesh, bc)     # uses entire stack
```

**3. Determinism vs. Randomness**

```
Challenge: Need random data for robustness, but tests must be reproducible

Solution Pattern:
```python
@pytest.fixture
def rng():
    """Fixed-seed RNG for reproducibility"""
    return np.random.RandomState(42)  # Fixed seed

@pytest.fixture
def random_matrix(rng):
    """Reproducible random test data"""
    return rng.randn(100, 100)

# Use in tests
def test_svd(random_matrix):
    U, s, Vt = np.linalg.svd(random_matrix)
    # Same random_matrix every time → reproducible test
```

**4. Tolerance Selection Strategy**

```
How to choose rtol and atol?

Context-dependent:
1. Float32: rtol=1e-6, atol=1e-8
2. Float64: rtol=1e-12, atol=1e-14
3. Iterative methods: rtol=1e-6 (convergence tolerance)
4. Ill-conditioned problems: rtol=1e-3 (condition number dependent)
5. Integration/Optimization: problem-specific

Rule of thumb:
rtol = 10 * machine_epsilon * condition_number
atol = rtol * expected_magnitude
```

---

## Phase 4: Advanced Testing Strategies

### Strategy 1: Metamorphic Testing

**Concept**: Test mathematical properties without knowing exact outputs

```python
def test_matrix_multiply_metamorphic():
    """Test properties that should always hold"""
    A = np.random.randn(5, 3)
    B = np.random.randn(3, 4)
    C = np.random.randn(4, 2)

    # Property: Associativity
    result1 = (A @ B) @ C
    result2 = A @ (B @ C)
    assert_allclose(result1, result2)

    # Property: Distributivity
    D = np.random.randn(4, 2)
    result1 = A @ (B @ C + B @ D)
    result2 = A @ B @ C + A @ B @ D
    assert_allclose(result1, result2)
```

### Strategy 2: Differential Testing

**Concept**: Compare against multiple reference implementations

```python
def test_fft_differential():
    """Compare our FFT against multiple implementations"""
    signal = np.random.randn(128)

    result_numpy = np.fft.fft(signal)
    result_scipy = scipy.fft.fft(signal)
    result_ours = our_fft(signal)

    # All should agree
    assert_allclose(result_ours, result_numpy, rtol=1e-12)
    assert_allclose(result_ours, result_scipy, rtol=1e-12)
```

### Strategy 3: Statistical Testing

**Concept**: Test statistical properties of algorithms

```python
def test_random_sampling_distribution():
    """Test that samples follow expected distribution"""
    samples = sample_normal(mean=5.0, std=2.0, size=10000)

    # Statistical tests
    assert abs(np.mean(samples) - 5.0) < 0.1  # Mean
    assert abs(np.std(samples) - 2.0) < 0.1   # Std

    # Shapiro-Wilk test for normality
    from scipy.stats import shapiro
    statistic, p_value = shapiro(samples)
    assert p_value > 0.05, "Samples don't follow normal distribution"
```

### Strategy 4: Convergence Testing

**Concept**: Test that iterative methods converge at expected rate

```python
def test_gradient_descent_convergence():
    """Test convergence rate of optimization"""
    x0 = np.array([10.0, 10.0])

    errors = []
    for lr in [0.1, 0.05, 0.025]:
        result = gradient_descent(f, x0, learning_rate=lr, max_iter=100)
        error = np.linalg.norm(result - true_minimum)
        errors.append(error)

    # Error should decrease as learning rate optimizes
    # Test linear convergence rate
    for i in range(len(errors) - 1):
        ratio = errors[i+1] / errors[i]
        assert 0.1 < ratio < 0.9, f"Convergence rate {ratio} not in expected range"
```

---

## Phase 5: Test Execution & Coverage Analysis

### Execution Strategy

```bash
# Run tests with coverage
pytest tests/ --cov={module} --cov-report=html --cov-report=term

# Run only fast tests
pytest -m "not slow"

# Run with verbose output
pytest -v --tb=short

# Run property-based tests with more examples
pytest --hypothesis-show-statistics --hypothesis-seed=42

# Run benchmarks
pytest --benchmark-only --benchmark-autosave

# Run in parallel
pytest -n auto  # requires pytest-xdist
```

### Coverage Requirements

```python
# .coveragerc
[run]
source = {module}
omit =
    */tests/*
    */test_*
    */__init__.py

[report]
precision = 2
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:

[html]
directory = coverage_html_report
```

### Julia Coverage

```julia
using Coverage

# Generate coverage
coverage = process_folder()

# Filter out test files
filtered = filter(x -> !occursin("test", x.filename), coverage)

# Calculate coverage percentage
covered_lines, total_lines = get_summary(filtered)
coverage_pct = 100 * covered_lines / total_lines

println("Coverage: $(round(coverage_pct, digits=2))%")

# Upload to Codecov
Codecov.submit(filtered)
```

---

## Phase 6: Test Report Generation

### Comprehensive Test Report

```markdown
# Test Suite Report for {module_name}

Generated: {timestamp}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {total_tests} |
| Passed | {passed} ✅ |
| Failed | {failed} ❌ |
| Skipped | {skipped} ⏭️ |
| Success Rate | {success_rate}% |
| Line Coverage | {line_cov}% |
| Branch Coverage | {branch_cov}% |
| Test Duration | {duration}s |

## Test Categories

### Unit Tests: {unit_count} tests
- ✅ Basic functionality: {basic_tests}
- ✅ Edge cases: {edge_tests}
- ✅ Type checking: {type_tests}

### Numerical Correctness: {numerical_count} tests
- ✅ Analytical solutions: {analytical_tests}
- ✅ Reference comparisons: {reference_tests}
- ✅ Convergence tests: {convergence_tests}

### Property-Based Tests: {property_count} tests
- ✅ Examples generated: {hypothesis_examples}
- ✅ Properties verified: {properties_verified}

### JAX-Specific Tests: {jax_count} tests
- ✅ Gradient correctness: {gradient_tests}
- ✅ JIT equivalence: {jit_tests}
- ✅ Vectorization: {vmap_tests}

### Performance Benchmarks: {benchmark_count} tests
- ⚡ Small inputs (n=10): {bench_small}
- ⚡ Medium inputs (n=100): {bench_medium}
- ⚡ Large inputs (n=1000): {bench_large}

## Coverage Analysis

### Overall Coverage
```
Lines:    {lines_covered}/{lines_total} ({line_cov}%)
Branches: {branches_covered}/{branches_total} ({branch_cov}%)
Functions: {funcs_covered}/{funcs_total} ({func_cov}%)
```

### Uncovered Code
{uncovered_lines}

### Coverage Recommendations
1. Add tests for uncovered functions: {uncovered_funcs}
2. Test missing branches in: {uncovered_branches}
3. Add edge case tests for: {missing_edge_cases}

## Test Quality Metrics

### Assertion Density
- Average assertions per test: {avg_assertions}
- Tests with <2 assertions: {low_assertion_tests} ⚠️

### Test Independence
- Tests with shared state: {stateful_tests} ⚠️
- Flaky tests detected: {flaky_tests} ⚠️

### Numerical Precision
- Tests with relaxed tolerance (>1e-6): {relaxed_tolerance_tests}
- Recommended tightening: {tighten_recommendations}

## Performance Insights

### Slowest Tests
1. {test1}: {time1}s
2. {test2}: {time2}s
3. {test3}: {time3}s

### Benchmark Comparison
{benchmark_table}

## Recommendations

### High Priority
1. 🔴 Increase coverage to >90% (current: {line_cov}%)
2. 🔴 Fix flaky tests: {flaky_test_list}
3. 🔴 Add gradient tests for: {missing_gradient_tests}

### Medium Priority
1. 🟡 Add property-based tests for: {missing_property_tests}
2. 🟡 Improve assertion density in: {weak_tests}
3. 🟡 Add benchmark baselines

### Low Priority
1. 🟢 Refactor slow tests
2. 🟢 Add more edge cases
3. 🟢 Document test patterns

## Next Steps

1. ✅ Review and merge generated tests
2. ✅ Run full test suite in CI
3. ✅ Set coverage requirements (recommend 90%)
4. ✅ Schedule regular test review
```

---

## Your Task: Generate Comprehensive Test Suite

**Arguments Received**: `$ARGUMENTS`

### Execution Plan

**Step 1: Analyze Source Code**
```bash
# Parse source files
python analyze_code.py $ARGUMENTS

# Extract functions, classes, and patterns
# Identify numerical computing patterns
# Determine test requirements
```

**Step 2: Generate Test Files**
```bash
# For each source file, generate corresponding test file
for source_file in $(find $ARGUMENTS -name "*.py" -o -name "*.jl"); do
    generate_test_suite $source_file
done
```

**Step 3: Create Fixtures and Data**
```bash
# Generate test data
python generate_test_data.py --analytical --edge-cases --random

# Create fixtures
create_pytest_fixtures
```

**Step 4: Add Property-Based Tests**
```bash
# Generate Hypothesis strategies
python generate_hypothesis_tests.py $ARGUMENTS

# Add metamorphic tests
add_metamorphic_properties
```

**Step 5: Add JAX-Specific Tests** (if applicable)
```bash
if jax_detected; then
    generate_gradient_tests
    generate_jit_tests
    generate_vmap_tests
    generate_device_tests
fi
```

**Step 6: Generate Benchmarks**
```bash
if [ "$BENCHMARKS" = true ]; then
    generate_performance_benchmarks
    create_baseline_data
fi
```

**Step 7: Run and Validate**
```bash
# Run generated tests
pytest tests/ -v

# Check coverage
pytest --cov=$ARGUMENTS --cov-report=term

# Generate report
python generate_test_report.py
```

---

## Execution Modes

### 1. Basic Mode (Default)
```bash
/generate-tests src/module.py
# Generates unit tests and edge cases
```

### 2. Full Mode (Comprehensive)
```bash
/generate-tests src/ --coverage --property-based --benchmarks
# Generates all test types with coverage analysis
```

### 3. JAX Mode
```bash
/generate-tests jax_module.py --property-based
# Includes gradient, JIT, and vmap tests
```

### 4. Julia Mode
```bash
/generate-tests src/Module.jl
# Generates Julia test suite with type stability tests
```

---

## Output Structure

```
tests/
├── test_{module}.py          # Main test file
├── conftest.py               # Shared fixtures
├── test_data/                # Test datasets
│   ├── analytical.npz
│   ├── edge_cases.npz
│   └── random_seed_42.npz
├── benchmarks/               # Performance tests
│   └── bench_{module}.py
└── property_tests/           # Property-based tests
    └── properties_{module}.py

reports/
├── test_report.md
├── coverage_report.html
└── benchmark_results.json
```

---

## Success Criteria

✅ **>90% code coverage**
✅ **All numerical correctness tests pass**
✅ **Property-based tests with >100 examples**
✅ **Gradient tests for all differentiable functions**
✅ **Performance benchmarks established**
✅ **Zero flaky tests**

---

Now execute comprehensive test generation with scientific computing best practices! 🧪
