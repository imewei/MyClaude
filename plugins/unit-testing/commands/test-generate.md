---
description: Generate comprehensive test suites for Python, Julia, and JAX scientific computing projects with numerical validation, property-based testing, and performance benchmarks
allowed-tools: Bash(find:*), Bash(grep:*), Bash(python:*), Bash(julia:*), Bash(pytest:*), Bash(git:*)
argument-hint: <source-file-or-module> [--coverage] [--property-based] [--benchmarks] [--scientific]
color: cyan
agents:
  primary:
    - test-automator
  conditional:
    - agent: hpc-numerical-coordinator
      trigger: pattern "numpy|scipy|pandas|matplotlib|scientific.*computing|numerical|simulation" OR argument "--scientific"
    - agent: jax-pro
      trigger: pattern "jax|flax|@jit|@vmap|@pmap|grad\\(|optax"
    - agent: neural-architecture-engineer
      trigger: pattern "torch|pytorch|tensorflow|keras|neural.*network|deep.*learning"
    - agent: ml-pipeline-coordinator
      trigger: pattern "sklearn|tensorflow|torch|keras|mlflow|model|train|predict"
    - agent: code-quality
      trigger: argument "--coverage" OR pattern "quality|lint|test.*strategy"
  orchestrated: false
---

# Automated Unit Test Generation

You are a test automation expert specializing in generating comprehensive, maintainable unit tests across multiple languages and frameworks, including scientific computing. Create tests that maximize coverage, catch edge cases, validate numerical correctness, and follow best practices for assertion quality and test organization.

## Context

The user needs automated test generation that analyzes code structure, identifies test scenarios, and creates high-quality unit tests with proper mocking, assertions, edge case coverage, and scientific computing validation (numerical correctness, property-based testing, gradient verification, performance benchmarks). Focus on framework-specific patterns and maintainable test suites.

## Requirements

$ARGUMENTS

## Instructions

### 1. Analyze Code for Test Generation

Scan codebase to identify untested code and generate comprehensive test suites:

```python
import ast
from pathlib import Path
from typing import Dict, List, Any

class TestGenerator:
    def __init__(self, language: str):
        self.language = language
        self.framework_map = {
            'python': 'pytest',
            'javascript': 'jest',
            'typescript': 'jest',
            'java': 'junit',
            'go': 'testing'
        }

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Extract testable units from source file"""
        if self.language == 'python':
            return self._analyze_python(file_path)
        elif self.language in ['javascript', 'typescript']:
            return self._analyze_javascript(file_path)

    def _analyze_python(self, file_path: str) -> Dict:
        with open(file_path) as f:
            tree = ast.parse(f.read())

        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'returns': ast.unparse(node.returns) if node.returns else None,
                    'decorators': [ast.unparse(d) for d in node.decorator_list],
                    'docstring': ast.get_docstring(node),
                    'complexity': self._calculate_complexity(node)
                })
            elif isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                classes.append({
                    'name': node.name,
                    'methods': methods,
                    'bases': [ast.unparse(base) for base in node.bases]
                })

        return {'functions': functions, 'classes': classes, 'file': file_path}
```

### 2. Generate Python Tests with pytest

```python
def generate_pytest_tests(self, analysis: Dict) -> str:
    """Generate pytest test file from code analysis"""
    tests = ['import pytest', 'from unittest.mock import Mock, patch', '']

    module_name = Path(analysis['file']).stem
    tests.append(f"from {module_name} import *\n")

    for func in analysis['functions']:
        if func['name'].startswith('_'):
            continue

        test_class = self._generate_function_tests(func)
        tests.append(test_class)

    for cls in analysis['classes']:
        test_class = self._generate_class_tests(cls)
        tests.append(test_class)

    return '\n'.join(tests)

def _generate_function_tests(self, func: Dict) -> str:
    """Generate test cases for a function"""
    func_name = func['name']
    tests = [f"\n\nclass Test{func_name.title()}:"]

    # Happy path test
    tests.append(f"    def test_{func_name}_success(self):")
    tests.append(f"        result = {func_name}({self._generate_mock_args(func['args'])})")
    tests.append(f"        assert result is not None\n")

    # Edge case tests
    if len(func['args']) > 0:
        tests.append(f"    def test_{func_name}_with_empty_input(self):")
        tests.append(f"        with pytest.raises((ValueError, TypeError)):")
        tests.append(f"            {func_name}({self._generate_empty_args(func['args'])})\n")

    # Exception handling test
    tests.append(f"    def test_{func_name}_handles_errors(self):")
    tests.append(f"        with pytest.raises(Exception):")
    tests.append(f"            {func_name}({self._generate_invalid_args(func['args'])})\n")

    return '\n'.join(tests)

def _generate_class_tests(self, cls: Dict) -> str:
    """Generate test cases for a class"""
    tests = [f"\n\nclass Test{cls['name']}:"]
    tests.append(f"    @pytest.fixture")
    tests.append(f"    def instance(self):")
    tests.append(f"        return {cls['name']}()\n")

    for method in cls['methods']:
        if method.startswith('_') and method != '__init__':
            continue

        tests.append(f"    def test_{method}(self, instance):")
        tests.append(f"        result = instance.{method}()")
        tests.append(f"        assert result is not None\n")

    return '\n'.join(tests)
```

### 3. Generate JavaScript/TypeScript Tests with Jest

```typescript
interface TestCase {
  name: string;
  setup?: string;
  execution: string;
  assertions: string[];
}

class JestTestGenerator {
  generateTests(functionName: string, params: string[]): string {
    const tests: TestCase[] = [
      {
        name: `${functionName} returns expected result with valid input`,
        execution: `const result = ${functionName}(${this.generateMockParams(params)})`,
        assertions: ['expect(result).toBeDefined()', 'expect(result).not.toBeNull()']
      },
      {
        name: `${functionName} handles null input gracefully`,
        execution: `const result = ${functionName}(null)`,
        assertions: ['expect(result).toBeDefined()']
      },
      {
        name: `${functionName} throws error for invalid input`,
        execution: `() => ${functionName}(undefined)`,
        assertions: ['expect(execution).toThrow()']
      }
    ];

    return this.formatJestSuite(functionName, tests);
  }

  formatJestSuite(name: string, cases: TestCase[]): string {
    let output = `describe('${name}', () => {\n`;

    for (const testCase of cases) {
      output += `  it('${testCase.name}', () => {\n`;
      if (testCase.setup) {
        output += `    ${testCase.setup}\n`;
      }
      output += `    const execution = ${testCase.execution};\n`;
      for (const assertion of testCase.assertions) {
        output += `    ${assertion};\n`;
      }
      output += `  });\n\n`;
    }

    output += '});\n';
    return output;
  }

  generateMockParams(params: string[]): string {
    return params.map(p => `mock${p.charAt(0).toUpperCase() + p.slice(1)}`).join(', ');
  }
}
```

### 4. Generate React Component Tests

```typescript
function generateReactComponentTest(componentName: string): string {
  return `
import { render, screen, fireEvent } from '@testing-library/react';
import { ${componentName} } from './${componentName}';

describe('${componentName}', () => {
  it('renders without crashing', () => {
    render(<${componentName} />);
    expect(screen.getByRole('main')).toBeInTheDocument();
  });

  it('displays correct initial state', () => {
    render(<${componentName} />);
    const element = screen.getByTestId('${componentName.toLowerCase()}');
    expect(element).toBeVisible();
  });

  it('handles user interaction', () => {
    render(<${componentName} />);
    const button = screen.getByRole('button');
    fireEvent.click(button);
    expect(screen.getByText(/clicked/i)).toBeInTheDocument();
  });

  it('updates props correctly', () => {
    const { rerender } = render(<${componentName} value="initial" />);
    expect(screen.getByText('initial')).toBeInTheDocument();

    rerender(<${componentName} value="updated" />);
    expect(screen.getByText('updated')).toBeInTheDocument();
  });
});
`;
}
```

### 5. Coverage Analysis and Gap Detection

```python
import subprocess
import json

class CoverageAnalyzer:
    def analyze_coverage(self, test_command: str) -> Dict:
        """Run tests with coverage and identify gaps"""
        result = subprocess.run(
            [test_command, '--coverage', '--json'],
            capture_output=True,
            text=True
        )

        coverage_data = json.loads(result.stdout)
        gaps = self.identify_coverage_gaps(coverage_data)

        return {
            'overall_coverage': coverage_data.get('totals', {}).get('percent_covered', 0),
            'uncovered_lines': gaps,
            'files_below_threshold': self.find_low_coverage_files(coverage_data, 80)
        }

    def identify_coverage_gaps(self, coverage: Dict) -> List[Dict]:
        """Find specific lines/functions without test coverage"""
        gaps = []
        for file_path, data in coverage.get('files', {}).items():
            missing_lines = data.get('missing_lines', [])
            if missing_lines:
                gaps.append({
                    'file': file_path,
                    'lines': missing_lines,
                    'functions': data.get('excluded_lines', [])
                })
        return gaps

    def generate_tests_for_gaps(self, gaps: List[Dict]) -> str:
        """Generate tests specifically for uncovered code"""
        tests = []
        for gap in gaps:
            test_code = self.create_targeted_test(gap)
            tests.append(test_code)
        return '\n\n'.join(tests)
```

### 6. Mock Generation

```python
def generate_mock_objects(self, dependencies: List[str]) -> str:
    """Generate mock objects for external dependencies"""
    mocks = ['from unittest.mock import Mock, MagicMock, patch\n']

    for dep in dependencies:
        mocks.append(f"@pytest.fixture")
        mocks.append(f"def mock_{dep}():")
        mocks.append(f"    mock = Mock(spec={dep})")
        mocks.append(f"    mock.method.return_value = 'mocked_result'")
        mocks.append(f"    return mock\n")

    return '\n'.join(mocks)
```

## 7. Scientific Computing Test Generation

### NumPy/SciPy Numerical Correctness Tests

```python
def generate_numerical_tests(self, func: Dict) -> str:
    """Generate tests for numerical correctness"""
    tests = []

    # Analytical solution tests
    tests.append(f"""
def test_{func['name']}_analytical_solution():
    '''Test against known analytical solution'''
    # Known exact solution
    input_data = {generate_analytical_input(func)}
    expected = {generate_analytical_output(func)}
    result = {func['name']}(input_data)

    from numpy.testing import assert_allclose
    assert_allclose(result, expected, rtol=1e-12, atol=1e-14,
                   err_msg="Result doesn't match analytical solution")
""")

    # Edge case tests
    tests.append(f"""
def test_{func['name']}_edge_cases():
    '''Test edge cases and special values'''
    import numpy as np

    # Empty array
    with pytest.raises(ValueError):
        {func['name']}(np.array([]))

    # Single element
    result = {func['name']}(np.array([1.0]))
    assert result.shape == (1,)

    # Zero values
    zeros = np.zeros(10)
    result = {func['name']}(zeros)
    assert np.all(np.isfinite(result))

    # Large/small values (numerical stability)
    large = np.array([1e10, 1e11, 1e12])
    result = {func['name']}(large)
    assert np.all(np.isfinite(result)), "Result contains inf/nan"
""")

    return '\n'.join(tests)
```

### Property-Based Testing with Hypothesis

```python
def generate_property_tests(self, func: Dict) -> str:
    """Generate property-based tests using Hypothesis"""
    tests = []

    tests.append("""
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
""")

    # Generate property test based on function type
    if is_linear_operation(func):
        tests.append(f"""
@given(
    data=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 100)),
        elements=st.floats(min_value=-1e6, max_value=1e6,
                          allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=100, deadline=None)
def test_{func['name']}_linearity(data):
    '''Test: f(aX + bY) = af(X) + bf(Y) for linear operations'''
    a, b = 2.0, 3.0
    X = data[:len(data)//2]
    Y = data[len(data)//2:]

    left = {func['name']}(a * X + b * Y)
    right = a * {func['name']}(X) + b * {func['name']}(Y)

    from numpy.testing import assert_allclose
    assert_allclose(left, right, rtol=1e-10)
""")

    if is_idempotent(func):
        tests.append(f"""
@given(data=arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 50)),
                   elements=st.floats(min_value=-100, max_value=100)))
def test_{func['name']}_idempotent(data):
    '''Test: f(f(x)) = f(x) for idempotent operations'''
    result1 = {func['name']}(data)
    result2 = {func['name']}(result1)
    assert_allclose(result1, result2, rtol=1e-10)
""")

    return '\n'.join(tests)
```

### JAX-Specific Tests

```python
def generate_jax_tests(self, func: Dict) -> str:
    """Generate JAX-specific tests for gradient, JIT, vmap"""
    tests = []

    tests.append("""
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
""")

    # JIT equivalence test
    tests.append(f"""
def test_{func['name']}_jit_equivalence():
    '''Test that JIT-compiled version produces same results'''
    input_data = jnp.array({generate_test_input(func)})

    # Non-JIT version
    result_nojit = {func['name']}(input_data)

    # JIT version
    jitted_fn = jit({func['name']})
    result_jit = jitted_fn(input_data)

    from numpy.testing import assert_allclose
    assert_allclose(result_nojit, result_jit, rtol=1e-12)
""")

    # Gradient correctness test
    if is_differentiable(func):
        tests.append(f"""
def test_{func['name']}_gradient_correctness():
    '''Test gradient using finite differences'''
    def fn(x):
        return jnp.sum({func['name']}(x))

    # Analytical gradient
    grad_fn = grad(fn)
    x = jnp.array({generate_test_input(func)})
    analytical_grad = grad_fn(x)

    # Finite difference gradient
    epsilon = 1e-5
    numerical_grad = jnp.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.at[i].add(epsilon)
        x_minus = x.at[i].add(-epsilon)
        numerical_grad = numerical_grad.at[i].set(
            (fn(x_plus) - fn(x_minus)) / (2 * epsilon)
        )

    assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6)
""")

    # vmap correctness test
    tests.append(f"""
def test_{func['name']}_vmap_correctness():
    '''Test vectorization with vmap'''
    # Single input
    single_input = jnp.array({generate_single_input(func)})
    single_result = {func['name']}(single_input)

    # Batched input
    batch_input = jnp.stack([single_input] * 5)
    vmapped_fn = vmap({func['name']})
    batch_result = vmapped_fn(batch_input)

    # Check all batch results match single result
    for i in range(5):
        assert_allclose(batch_result[i], single_result, rtol=1e-12)
""")

    return '\n'.join(tests)
```

### Performance Benchmarks

```python
def generate_benchmark_tests(self, func: Dict) -> str:
    """Generate performance benchmark tests"""
    tests = []

    tests.append(f"""
@pytest.mark.benchmark
@pytest.mark.parametrize("size", [10, 100, 1000])
def test_{func['name']}_performance(benchmark, size):
    '''Benchmark performance across different input sizes'''
    import numpy as np
    input_data = np.random.randn(size)
    result = benchmark({func['name']}, input_data)
    assert result is not None

def test_{func['name']}_memory_usage():
    '''Test memory usage is reasonable'''
    import tracemalloc
    import numpy as np

    tracemalloc.start()
    input_data = np.random.randn(1000)
    _ = {func['name']}(input_data)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Assert peak memory is reasonable (adjust threshold)
    assert peak < 10 * 1024 * 1024, f"Peak memory {{peak}} bytes too high"
""")

    return '\n'.join(tests)
```

### Julia Test Generation

```julia
function generate_julia_tests(func_info::Dict)
    """Generate Julia test suite"""

    test_suite = """
using Test
using ${{func_info['module']}}
using LinearAlgebra
using Random

@testset "${{func_info['module']}} - ${{func_info['name']}}" begin
    @testset "basic functionality" begin
        input_data = ${{generate_test_input(func_info)}}
        expected = ${{generate_expected_output(func_info)}}
        result = ${{func_info['name']}}(input_data)

        @test result ≈ expected atol=1e-10 rtol=1e-7
    end

    @testset "type stability" begin
        input_data = ${{generate_test_input(func_info)}}
        @inferred ${{func_info['name']}}(input_data)
    end

    @testset "edge cases" begin
        # Empty input
        @test_throws ArgumentError ${{func_info['name']}}(Float64[])

        # Single element
        single = [1.0]
        result = ${{func_info['name']}}(single)
        @test length(result) == 1

        # Special values
        @test isfinite(${{func_info['name']}}([0.0, 0.0, 0.0]))
    end

    @testset "mathematical properties" begin
        A = randn(5, 5)
        B = randn(5, 5)

        # Test specific properties based on function type
        # Example: Linearity, idempotence, etc.
    end
end
"""

    return test_suite
end
```

### Test Requirements for Scientific Code

#### Numerical Correctness Priority
1. **Analytical Solutions** (highest priority): Test against known exact solutions
2. **Reference Implementations**: Compare with SciPy, NumPy, or reference libraries
3. **Mathematical Properties**: Verify algebraic properties hold
4. **Convergence**: Test iterative methods converge at expected rate

#### Numerical Stability Tests
- Catastrophic cancellation detection
- Loss of precision monitoring
- Overflow/underflow handling
- Condition number analysis
- Special value handling (inf, -inf, nan, 0)

#### Edge Cases for Scientific Computing
- Empty arrays
- Single-element arrays
- Zero inputs
- Identity matrices
- Singular/near-singular matrices
- Ill-conditioned problems
- Large/small magnitude values

#### JAX-Specific Requirements
- JIT compilation equivalence
- Gradient correctness (finite differences validation)
- vmap batching correctness
- Device consistency (CPU/GPU)
- Memory efficiency
- Pure function validation (no side effects)

### Execution Modes

```bash
# Standard test generation
/test-generate src/module.py

# Scientific computing mode (enables numerical tests)
/test-generate src/scientific_module.py --scientific --property-based

# Full suite with benchmarks
/test-generate src/ --coverage --property-based --benchmarks

# JAX-specific tests
/test-generate jax_module.py --scientific --property-based
# Automatically includes gradient, JIT, vmap tests
```

## Output Format

1. **Test Files**: Complete test suites ready to run
2. **Coverage Report**: Current coverage with gaps identified
3. **Mock Objects**: Fixtures for external dependencies
4. **Test Documentation**: Explanation of test scenarios
5. **CI Integration**: Commands to run tests in pipeline
6. **Numerical Validation**: Analytical solution tests and property-based tests (for scientific code)
7. **Performance Benchmarks**: Baseline performance data (when requested)

Focus on generating maintainable, comprehensive tests that catch bugs early, validate numerical correctness, and provide confidence in code changes.
