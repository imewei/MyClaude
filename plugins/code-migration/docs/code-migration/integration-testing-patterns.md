# Integration & Testing Patterns

**Version**: 1.0.3
**Category**: code-migration
**Purpose**: Modern tooling integration, package management, API design, and comprehensive testing frameworks

## Modern Tooling Integration

### Version Control Best Practices

**Git Workflow for Migration**:
```bash
# Create migration branch
git checkout -b feature/python-migration

# Preserve legacy code
git mv legacy/fortran_solver.f90 legacy/archive/

# Add new implementation
git add src/python_solver.py

# Commit with detailed message
git commit -m "feat: migrate Fortran solver to Python/JAX

- Translate core algorithms to vectorized NumPy
- Preserve numerical accuracy (<1e-12 error)
- Add comprehensive test suite
- GPU acceleration via JAX jit/vmap
- Performance: 129x speedup on V100 GPU

Co-authored-by: Legacy Code <fortran77@example.com>"
```

---

### Continuous Integration Setup

**GitHub Actions for Scientific Code**:
```yaml
name: Test Migration

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          uv uv pip install numpy scipy jax pytest
          uv uv pip install -e .

      - name: Run numerical validation tests
        run: pytest tests/ -v --tb=short

      - name: Compare with Fortran reference
        run: python scripts/validate_against_fortran.py

      - name: Check numerical accuracy
        run: |
          python -c "import tests.accuracy as acc; acc.check_all(tol=1e-11)"
```

---

### Documentation Generation

**Sphinx for API Documentation**:
```bash
# Install Sphinx
uv uv pip install sphinx sphinx-rtd-theme

# Initialize docs
cd docs
sphinx-quickstart

# Configure autodoc
# In conf.py:
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.mathjax']

# Generate docs
make html
```

**Docstring Example**:
```python
def solve_ode(f, y0, t_span, method='RK45', **kwargs):
    """
    Solve ordinary differential equation.

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE system: dy/dt = f(y, t).
        Signature: f(y, t) -> array_like
    y0 : array_like, shape (n,)
        Initial state vector
    t_span : tuple (t0, tf)
        Time interval for integration
    method : str, optional
        Integration method. Options: 'RK45', 'BDF', 'LSODA'
        Default: 'RK45'

    Returns
    -------
    solution : ODESolution
        Object with fields:
        - t : array, time points
        - y : array, solution values
        - success : bool

    Notes
    -----
    For stiff systems, use method='BDF' or 'LSODA'.

    Examples
    --------
    >>> def exponential_decay(y, t):
    ...     return -y
    >>> sol = solve_ode(exponential_decay, y0=1.0, t_span=(0, 5))
    >>> sol.y[-1]  # Final value
    0.00673794699...

    References
    ----------
    .. [1] Hairer, E., Norsett, S. P., Wanner, G. (1993).
           Solving Ordinary Differential Equations I.
    """
    pass
```

---

## Package Management

### Python Package Structure

```
my_scientific_package/
├── pyproject.toml          # Modern Python packaging
├── README.md
├── LICENSE
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── core.py         # Core algorithms
│       ├── solvers.py      # Numerical solvers
│       └── utils.py        # Utilities
├── tests/
│   ├── test_core.py
│   ├── test_solvers.py
│   └── reference/          # Fortran reference outputs
│       └── data.npy
├── docs/
│   ├── conf.py
│   └── index.rst
└── benchmarks/
    └── performance.py
```

**pyproject.toml** (modern packaging):
```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-scientific-package"
version = "1.0.3"
description = "Modern Python/JAX implementation of legacy Fortran solver"
readme = "README.md"
requires-python = ">=3.12"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"}
]
keywords = ["scientific-computing", "numerical-methods", "jax", "gpu"]

dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "jax[cuda12]>=0.4.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "black", "ruff", "mypy"]
docs = ["sphinx", "sphinx-rtd-theme"]
```

---

### Dependency Specification

**Environment Management** (conda):
```yaml
name: scientific-migration
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - numpy=1.26
  - scipy=1.11
  - jax
  - pytest
  - pip:
    - diffrax
    - my-custom-package
```

**Version Pinning Strategy**:
```
# production.txt - strict pins
numpy==1.26.4
scipy==1.11.4
jax[cuda12]==0.4.25

# development.txt - flexible
numpy>=1.26,<2.0
scipy>=1.11
jax[cuda12]>=0.4
```

---

## API Design

### Modern API Patterns

**Functional API** (preferred for scientific computing):
```python
import jax.numpy as jnp

def compute_energy(positions, masses):
    """Pure function - no side effects"""
    velocities = jnp.diff(positions, axis=0)
    kinetic = 0.5 * jnp.sum(masses * velocities**2)
    return kinetic
```

**Object-Oriented API** (for stateful systems):
```python
class ODESolver:
    """Stateful solver with configuration"""

    def __init__(self, method='RK45', rtol=1e-6, atol=1e-9):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.solutions = []

    def solve(self, f, y0, t_span):
        """Solve ODE and store solution"""
        solution = integrate.solve_ivp(
            f, t_span, y0,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol
        )
        self.solutions.append(solution)
        return solution

    def get_history(self):
        """Retrieve all solutions"""
        return self.solutions
```

---

### Type Hints and Contracts

**Type Annotations** (Python 3.12+):
```python
from typing import Callable
import numpy as np
import jax.numpy as jnp
from numpy.typing import NDArray

def solve_system(
    matrix: NDArray[np.float64],
    rhs: NDArray[np.float64],
    method: str = "direct"
) -> NDArray[np.float64]:
    """
    Solve linear system Ax = b.

    Parameters
    ----------
    matrix : NDArray[float64], shape (n, n)
        Coefficient matrix A
    rhs : NDArray[float64], shape (n,)
        Right-hand side vector b
    method : str
        Solution method: 'direct' or 'iterative'

    Returns
    -------
    solution : NDArray[float64], shape (n,)
        Solution vector x
    """
    if method == "direct":
        return np.linalg.solve(matrix, rhs)
    else:
        return iterative_solve(matrix, rhs)
```

---

### Error Handling and Validation

**Input Validation**:
```python
def validate_inputs(x, y, method):
    """Validate inputs and raise informative errors"""
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be ndarray, got {type(x)}")

    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")

    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")

    valid_methods = ['RK45', 'BDF', 'LSODA']
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
```

---

## Numerical Validation Frameworks

### Regression Test Suite

**Pytest Fixtures for Reference Data**:
```python
import pytest
import numpy as np

@pytest.fixture
def fortran_reference():
    """Load Fortran reference output"""
    return np.load('tests/reference/fortran_output.npy')

@pytest.fixture
def test_parameters():
    """Standard test parameters"""
    return {
        'n_points': 1000,
        'dt': 0.01,
        'tolerance': 1e-11
    }

def test_solver_accuracy(fortran_reference, test_parameters):
    """Compare Python solver with Fortran reference"""
    result = python_solver(**test_parameters)

    # Numerical comparison
    np.testing.assert_allclose(
        result,
        fortran_reference,
        rtol=test_parameters['tolerance'],
        atol=1e-13
    )
```

---

### Property-Based Testing

**Hypothesis for Numerical Properties**:
```python
from hypothesis import given, strategies as st
import numpy as np

@given(
    x=st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=10, max_size=1000),
    y=st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=10, max_size=1000)
)
def test_commutative_property(x, y):
    """Test numerical properties"""
    x_arr = np.array(x)
    y_arr = np.array(y)

    # dot product is commutative
    result1 = np.dot(x_arr, y_arr)
    result2 = np.dot(y_arr, x_arr)

    assert np.isclose(result1, result2, rtol=1e-10)
```

---

### Convergence Testing

**Test Convergence Rates**:
```python
def test_ode_solver_convergence():
    """Verify expected convergence order"""
    # Analytical solution: y(t) = exp(-t)
    def f(y, t):
        return -y

    y0 = 1.0
    t_final = 5.0

    errors = []
    dt_values = [0.1, 0.05, 0.025, 0.0125]

    for dt in dt_values:
        solution = solve_ode(f, y0, t_span=(0, t_final), dt=dt)
        expected = np.exp(-t_final)
        error = abs(solution.y[-1] - expected)
        errors.append(error)

    # Check convergence order (should be 4 for RK4)
    log_errors = np.log(errors)
    log_dts = np.log(dt_values)

    # Linear regression: log(error) vs log(dt)
    slope, _ = np.polyfit(log_dts, log_errors, 1)

    # RK4 has order 4 convergence
    assert 3.5 < slope < 4.5, f"Convergence order {slope:.2f} not RK4"
```

---

## Performance Benchmarking Suites

### Benchmark Framework

**pytest-benchmark**:
```python
import pytest

def fibonacci(n):
    """Compute nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def test_fibonacci_performance(benchmark):
    """Benchmark Fibonacci computation"""
    result = benchmark(fibonacci, 20)
    assert result == 6765

# Run: pytest --benchmark-only
```

**Custom Benchmark Suite**:
```python
import time
import numpy as np

class PerformanceBenchmark:
    def __init__(self, name):
        self.name = name
        self.results = []

    def run(self, func, *args, n_iter=10):
        """Run benchmark and collect statistics"""
        times = []

        # Warm-up
        func(*args)

        # Timed runs
        for _ in range(n_iter):
            start = time.perf_counter()
            result = func(*args)
            if hasattr(result, 'block_until_ready'):  # JAX
                result.block_until_ready()
            end = time.perf_counter()
            times.append(end - start)

        stats = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }

        self.results.append((self.name, stats))
        return stats

# Usage
benchmark = PerformanceBenchmark("Matrix Multiply")
stats = benchmark.run(np.dot, A, B, n_iter=100)
print(f"Mean time: {stats['mean']*1000:.2f} ms")
```

---

## Best Practices Summary

1. **Version Control**: Detailed commit messages with migration context
2. **CI/CD**: Automated numerical validation against reference
3. **Documentation**: Sphinx with NumPy-style docstrings
4. **Packaging**: Modern pyproject.toml with strict dependency pins
5. **API Design**: Type hints, input validation, informative errors
6. **Testing**: Regression tests, property-based tests, convergence verification
7. **Benchmarking**: Statistical significance, warm-up runs, reproducible conditions

---

## References

- pytest Documentation: https://docs.pytest.org/
- Hypothesis: https://hypothesis.readthedocs.io/
- Sphinx: https://www.sphinx-doc.org/
- Python Packaging Guide: https://packaging.python.org/
