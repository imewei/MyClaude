---
version: 1.0.3
command: /test-generate
description: Generate comprehensive test suites with scientific computing support, numerical validation, property-based testing, and performance benchmarks across 3 execution modes
argument-hint: <source-file-or-module> [--coverage] [--property-based] [--benchmarks] [--scientific]
execution_modes:
  quick:
    duration: "30min-1h"
    description: "Fast scaffolding for single module/file"
    agents: ["test-automator"]
    scope: "Single module/file"
    test_types: "Unit tests only"
    coverage: "Generate for uncovered functions"
    scientific: "Basic numerical assertions"
    output: "~50-100 test cases"
    use_case: "Quick scaffolding, TDD workflow"
  standard:
    duration: "2-4h"
    description: "Comprehensive test suite for package/feature"
    agents: ["test-automator", "hpc-numerical-coordinator"]
    scope: "Package/feature module"
    test_types: "Unit + integration, edge cases, mocking"
    coverage: "Comprehensive (>80%)"
    scientific: "Numerical validation, gradient verification"
    property_based: "Hypothesis for critical functions"
    output: "~200-500 test cases"
    use_case: "Production test suite generation"
  enterprise:
    duration: "1-2d"
    description: "Exhaustive test suite for entire project"
    agents: ["test-automator", "hpc-numerical-coordinator", "jax-pro"]
    scope: "Entire project/codebase"
    test_types: "Unit + integration + E2E, property-based, mutation"
    coverage: "Exhaustive (>90%), mutation score"
    scientific: "Full numerical validation, convergence tests"
    documentation: "Test docs, coverage reports"
    output: "~1,000+ test cases"
    use_case: "New project test suite, compliance"
workflow_type: "generative"
interactive_mode: true
color: cyan
allowed-tools: Bash(find:*), Bash(grep:*), Bash(python:*), Bash(julia:*), Bash(pytest:*), Bash(git:*)
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

Generate comprehensive, maintainable test suites across Python, Julia, JAX, and JavaScript/TypeScript with scientific computing support, numerical correctness validation, and property-based testing.

## Context

The user needs automated test generation for: $ARGUMENTS

## Execution Mode Selection

<AskUserQuestion>
questions:
  - question: "Which test generation scope best fits your needs?"
    header: "Generation Mode"
    multiSelect: false
    options:
      - label: "Quick (30min-1h)"
        description: "Single module/file. Unit tests only with basic coverage. Generates ~50-100 test cases. Use for TDD workflow and quick scaffolding."

      - label: "Standard (2-4h)"
        description: "Package/feature module. Unit + integration tests with >80% coverage, property-based testing for critical functions. Generates ~200-500 test cases. Use for production test suites."

      - label: "Enterprise (1-2d)"
        description: "Entire project/codebase. Full test suite with >90% coverage, E2E, property-based, and mutation testing. Includes documentation and reports. Generates ~1,000+ test cases. Use for new projects and compliance."
</AskUserQuestion>

## Phase 1: Code Analysis

**See comprehensive guide**: [Test Generation Patterns](../docs/test-generate/test-generation-patterns.md)

### AST-Based Analysis

```python
import ast
from typing import List, Dict

# Analyze Python module
def analyze_python_module(file_path: str) -> Dict:
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
                'complexity': calculate_complexity(node),
                'docstring': ast.get_docstring(node)
            })
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes.append({
                'name': node.name,
                'methods': methods
            })

    return {'functions': functions, 'classes': classes}
```

### Coverage Gap Detection

```python
# Identify untested code from coverage reports
def analyze_coverage_gaps(coverage_file: str) -> List[Dict]:
    coverage_data = read_coverage_report(coverage_file)

    gaps = []
    for file_path, data in coverage_data['files'].items():
        missing_lines = data.get('missing_lines', [])

        if missing_lines:
            # Find uncovered functions
            uncovered_functions = find_uncovered_functions(
                file_path,
                missing_lines
            )

            for func in uncovered_functions:
                gaps.append({
                    'file': file_path,
                    'function': func['name'],
                    'priority': calculate_priority(func)
                })

    return sorted(gaps, key=lambda x: x['priority'], reverse=True)
```

## Phase 2: Test Generation

**See comprehensive guide**: [Test Generation Patterns](../docs/test-generate/test-generation-patterns.md)

### pytest Test Generation

```python
# Generate pytest tests
def generate_pytest_tests(func_info: Dict) -> str:
    """Generate comprehensive pytest test suite"""

    return f'''
def test_{func_info['name']}_success():
    """Test {func_info['name']} with valid input"""
    result = {func_info['name']}({generate_mock_args(func_info['args'])})
    assert result is not None

def test_{func_info['name']}_edge_cases():
    """Test {func_info['name']} with edge cases"""
    # Empty input
    with pytest.raises((ValueError, TypeError)):
        {func_info['name']}()

    # None input
    with pytest.raises((ValueError, TypeError)):
        {func_info['name']}(None)

@pytest.mark.parametrize("input,expected", [
    ({generate_test_cases(func_info)})
])
def test_{func_info['name']}_parametrized(input, expected):
    """Parametrized test for multiple scenarios"""
    result = {func_info['name']}(input)
    assert result == expected
'''
```

### Jest/Vitest Test Generation

```typescript
// Generate Jest/Vitest tests
function generateJestTests(functionName: string, params: string[]): string {
  return `
describe('${functionName}', () => {
  it('should return expected result with valid input', () => {
    const result = ${functionName}(${generateMockParams(params)});
    expect(result).toBeDefined();
  });

  it('should handle null input gracefully', () => {
    const result = ${functionName}(null);
    expect(result).toBeDefined();
  });

  it('should throw error for invalid input', () => {
    expect(() => ${functionName}(undefined)).toThrow();
  });
});
`;
}
```

## Phase 3: Scientific Computing Tests

**See comprehensive guide**: [Scientific Testing Guide](../docs/test-generate/scientific-testing-guide.md)

### Numerical Correctness Tests

```python
def generate_numerical_tests(func_info: Dict) -> str:
    """Generate tests for numerical correctness"""

    return f'''
import numpy as np
from numpy.testing import assert_allclose

def test_{func_info['name']}_analytical_solution():
    """Test against known analytical solution"""
    input_data = {generate_analytical_input(func_info)}
    expected = {generate_analytical_output(func_info)}
    result = {func_info['name']}(input_data)

    assert_allclose(result, expected, rtol=1e-12, atol=1e-14,
                   err_msg="Result doesn't match analytical solution")

def test_{func_info['name']}_edge_cases():
    """Test edge cases and special values"""
    # Empty array
    with pytest.raises(ValueError):
        {func_info['name']}(np.array([]))

    # Zero values
    zeros = np.zeros(10)
    result = {func_info['name']}(zeros)
    assert np.all(np.isfinite(result)), "Result contains inf/nan"

    # Large/small values (numerical stability)
    large = np.array([1e10, 1e11, 1e12])
    result = {func_info['name']}(large)
    assert np.all(np.isfinite(result)), "Numerical instability detected"
'''
```

### JAX Gradient Tests

```python
def generate_jax_tests(func_info: Dict) -> str:
    """Generate JAX-specific tests"""

    return f'''
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from numpy.testing import assert_allclose

def test_{func_info['name']}_jit_equivalence():
    """Test that JIT-compiled version produces same results"""
    input_data = jnp.array({generate_test_input(func_info)})

    result_nojit = {func_info['name']}(input_data)
    jitted_fn = jit({func_info['name']})
    result_jit = jitted_fn(input_data)

    assert_allclose(result_nojit, result_jit, rtol=1e-12)

def test_{func_info['name']}_gradient_correctness():
    """Test gradient using finite differences"""
    def fn(x):
        return jnp.sum({func_info['name']}(x))

    grad_fn = grad(fn)
    x = jnp.array({generate_test_input(func_info)})
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

def test_{func_info['name']}_vmap_correctness():
    """Test vectorization with vmap"""
    single_input = jnp.array({generate_single_input(func_info)})
    single_result = {func_info['name']}(single_input)

    batch_input = jnp.stack([single_input] * 5)
    vmapped_fn = vmap({func_info['name']})
    batch_result = vmapped_fn(batch_input)

    for i in range(5):
        assert_allclose(batch_result[i], single_result, rtol=1e-12)
'''
```

## Phase 4: Property-Based Testing

**See comprehensive guide**: [Property-Based Testing](../docs/test-generate/property-based-testing.md)

### Hypothesis Tests

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np

def generate_property_tests(func_info: Dict) -> str:
    """Generate property-based tests using Hypothesis"""

    return f'''
@given(
    data=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 100)),
        elements=st.floats(min_value=-1e6, max_value=1e6,
                          allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=100, deadline=None)
def test_{func_info['name']}_linearity(data):
    """Test: f(aX + bY) = af(X) + bf(Y) for linear operations"""
    a, b = 2.0, 3.0
    X = data[:len(data)//2]
    Y = data[len(data)//2:]

    left = {func_info['name']}(a * X + b * Y)
    right = a * {func_info['name']}(X) + b * {func_info['name']}(Y)

    assert_allclose(left, right, rtol=1e-10)

@given(data=arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 50))))
def test_{func_info['name']}_idempotent(data):
    """Test: f(f(x)) = f(x) for idempotent operations"""
    result1 = {func_info['name']}(data)
    result2 = {func_info['name']}(result1)
    assert_allclose(result1, result2, rtol=1e-10)
'''
```

## Phase 5: Performance Benchmarks

### pytest Benchmarks

```python
def generate_benchmark_tests(func_info: Dict) -> str:
    """Generate performance benchmark tests"""

    return f'''
@pytest.mark.benchmark
@pytest.mark.parametrize("size", [10, 100, 1000])
def test_{func_info['name']}_performance(benchmark, size):
    """Benchmark performance across different input sizes"""
    input_data = np.random.randn(size)
    result = benchmark({func_info['name']}, input_data)
    assert result is not None

def test_{func_info['name']}_memory_usage():
    """Test memory usage is reasonable"""
    import tracemalloc

    tracemalloc.start()
    input_data = np.random.randn(1000)
    _ = {func_info['name']}(input_data)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Assert peak memory < 100 MB
    assert peak < 100 * 1024 * 1024, f"Peak memory {{peak}} bytes too high"
'''
```

## Phase 6: Coverage Analysis & Reporting

**See comprehensive guide**: [Coverage Analysis Guide](../docs/test-generate/coverage-analysis-guide.md)

### Generate Coverage Report

```bash
# Python coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# JavaScript coverage
npm test -- --coverage

# Identify gaps
python analyze_coverage_gaps.py coverage.json
```

### Prioritized Test Generation

```python
# Focus on high-priority gaps
gaps = analyze_coverage_gaps('coverage.json')
high_priority = gaps[:20]  # Top 20% of gaps

for gap in high_priority:
    tests = generate_tests_for_gap(gap)
    write_test_file(gap['file'], tests)
```

## Phase 7: Test Organization

### Directory Structure

```
project/
├── src/
│   ├── module1.py
│   └── module2.py
└── tests/
    ├── unit/
    │   ├── test_module1.py  # Generated tests
    │   └── test_module2.py
    ├── integration/
    │   └── test_workflow.py
    └── conftest.py  # Shared fixtures
```

### Test File Template

```python
"""
Tests for {module_name}

Generated automatically by test-generate command.
"""

import pytest
from unittest.mock import Mock, patch

from {module_path} import {imports}


class Test{ClassName}:
    """Test suite for {ClassName}"""

    @pytest.fixture
    def instance(self):
        return {ClassName}()

    # Generated tests here


# Standalone function tests
# Parametrized tests
# Edge case tests
# Scientific computing tests (if applicable)
```

## Execution Modes

### Quick Mode (30min-1h)

```bash
/test-generate src/module.py

# Generates basic unit tests for single module
# Output: ~50-100 test cases
```

### Standard Mode (2-4h)

```bash
/test-generate src/ --coverage --property-based

# Generates comprehensive test suite with property-based tests
# Output: ~200-500 test cases
```

### Enterprise Mode (1-2d)

```bash
/test-generate . --coverage --property-based --benchmarks --scientific

# Generates exhaustive test suite for entire project
# Output: ~1,000+ test cases
```

## Framework-Specific Patterns

### Python (pytest)
- Fixtures for test data
- Parametrized tests for multiple scenarios
- Mocks for external dependencies
- Property-based tests with Hypothesis
- Numerical tests with numpy.testing

### JavaScript (Jest/Vitest)
- describe/it blocks for organization
- beforeEach for setup
- Mock functions with jest.fn()
- Snapshot testing for components
- Async/await for asynchronous tests

### Julia
- @testset for test organization
- @test for assertions
- @inferred for type stability
- Property-based tests with packages

## External Documentation

Comprehensive guides available in `docs/test-generate/`:

1. **test-generation-patterns.md** - AST parsing, test algorithms, mocking strategies, framework templates
2. **scientific-testing-guide.md** - Numerical correctness, tolerance-based assertions, gradient verification
3. **property-based-testing.md** - Hypothesis patterns, QuickCheck equivalents, stateful testing
4. **coverage-analysis-guide.md** - Coverage metrics, gap identification, prioritization, reporting

## Output

Generated test files include:

1. **Unit Tests**: Comprehensive coverage of all functions and classes
2. **Edge Case Tests**: Boundary values, null inputs, error handling
3. **Parametrized Tests**: Multiple scenarios with different inputs
4. **Property-Based Tests**: Mathematical properties for scientific code
5. **Performance Benchmarks**: Performance regression detection
6. **Coverage Reports**: HTML and terminal coverage reports
7. **Test Documentation**: Explanation of test scenarios

## Best Practices

1. **Analyze code structure** before generating tests
2. **Generate happy path first**, then edge cases
3. **Use parametrized tests** for multiple scenarios
4. **Mock external dependencies** to isolate tests
5. **Generate meaningful test names** describing what is being tested
6. **Include docstrings** explaining test purpose
7. **Follow AAA pattern** (Arrange, Act, Assert)
8. **Generate assertions based on return types**
9. **Handle async functions** appropriately
10. **Maintain consistency** with existing test patterns

## Now Execute

Begin automated test generation based on selected execution mode, analyzing code structure and creating comprehensive test suites with scientific computing support where applicable.
