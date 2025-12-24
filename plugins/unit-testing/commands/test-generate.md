---
version: "1.0.5"
command: /test-generate
description: Generate comprehensive test suites with scientific computing support, numerical validation, property-based testing, and performance benchmarks
argument-hint: <source-file-or-module> [--coverage] [--property-based] [--benchmarks] [--scientific]
execution_modes:
  quick:
    duration: "30min-1h"
    scope: "Single module/file"
    output: "~50-100 test cases"
  standard:
    duration: "2-4h"
    scope: "Package/feature module"
    output: "~200-500 test cases"
  enterprise:
    duration: "1-2d"
    scope: "Entire project/codebase"
    output: "~1,000+ test cases"
workflow_type: "generative"
color: cyan
allowed-tools: Bash(find:*), Bash(grep:*), Bash(python:*), Bash(julia:*), Bash(pytest:*), Bash(git:*)
agents:
  primary:
    - test-automator
  conditional:
    - agent: hpc-numerical-coordinator
      trigger: pattern "numpy|scipy|pandas|scientific.*computing|numerical|simulation" OR argument "--scientific"
    - agent: jax-pro
      trigger: pattern "jax|flax|@jit|@vmap|@pmap|grad\\(|optax"
    - agent: neural-architecture-engineer
      trigger: pattern "torch|pytorch|tensorflow|keras|neural.*network|deep.*learning"
    - agent: ml-pipeline-coordinator
      trigger: pattern "sklearn|mlflow|model|train|predict"
    - agent: code-quality
      trigger: argument "--coverage" OR pattern "quality|lint|test.*strategy"
  orchestrated: false
---

# Automated Test Generation

Generate comprehensive test suites across Python, Julia, JAX, and JavaScript/TypeScript with scientific computing support.

## Target

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope | Test Types | Coverage | Output |
|------|----------|-------|------------|----------|--------|
| Quick | 30min-1h | Single module | Unit only | Basic | ~50-100 tests |
| Standard | 2-4h | Package/feature | Unit + integration, property-based | >80% | ~200-500 tests |
| Enterprise | 1-2d | Entire project | Unit + integration + E2E + mutation | >90% | ~1,000+ tests |

**Options:** `--coverage`, `--property-based`, `--benchmarks`, `--scientific`

---

## Phase 1: Code Analysis

**Agent:** test-automator

### AST-Based Analysis
- Parse source files to extract functions, classes, methods
- Identify parameters, return types, complexity
- Extract docstrings for expected behavior

### Coverage Gap Detection
- Read existing coverage reports
- Identify uncovered functions
- Prioritize by complexity and risk

### Dependency Mapping
- External vs internal dependencies
- Mock candidates identification
- Integration points

---

## Phase 2: Test Generation

### Framework Selection

| Language | Framework | Style |
|----------|-----------|-------|
| Python | pytest | fixtures, parametrize, markers |
| JavaScript | Jest/Vitest | describe/it, beforeEach, mock |
| Julia | Test.jl | @testset, @test, @inferred |

### Test Categories

| Category | Description |
|----------|-------------|
| **Happy Path** | Valid input, expected behavior |
| **Edge Cases** | Empty, null, boundary values |
| **Error Handling** | Invalid input, exception testing |
| **Parametrized** | Multiple scenarios via parametrize/each |

### Test Template Pattern
```
AAA Pattern: Arrange → Act → Assert
- Arrange: Setup test data and mocks
- Act: Call function under test
- Assert: Verify expected behavior
```

---

## Phase 3: Scientific Computing Tests (if `--scientific`)

**Agent:** hpc-numerical-coordinator, jax-pro

### Numerical Correctness

| Test Type | Purpose |
|-----------|---------|
| Analytical solution | Compare against known mathematical results |
| Tolerance assertions | `assert_allclose(result, expected, rtol=1e-12)` |
| Edge values | Empty arrays, zeros, large/small values |
| Numerical stability | Verify no inf/nan in results |

### JAX-Specific Tests

| Test Type | Purpose |
|-----------|---------|
| JIT equivalence | `jit(fn)(x) == fn(x)` |
| Gradient correctness | Analytical vs finite difference gradients |
| vmap correctness | Batched results match individual results |

---

## Phase 4: Property-Based Testing (if `--property-based`)

**Agent:** test-automator

### Mathematical Properties

| Property | Test |
|----------|------|
| Idempotence | `f(f(x)) == f(x)` |
| Commutativity | `f(a, b) == f(b, a)` |
| Associativity | `f(f(a, b), c) == f(a, f(b, c))` |
| Linearity | `f(ax + by) == af(x) + bf(y)` |
| Inverse | `f_inv(f(x)) == x` |

### Hypothesis Strategies
- Use `hypothesis.strategies` for input generation
- `hypothesis.extra.numpy` for array strategies
- Configure `max_examples`, `deadline` appropriately

---

## Phase 5: Performance Benchmarks (if `--benchmarks`)

### Benchmark Tests
- Multiple input sizes (10, 100, 1000, 10000)
- Memory usage tracking via tracemalloc
- pytest-benchmark integration

### Performance Assertions
- Set reasonable upper bounds for execution time
- Track memory peak usage
- Fail tests if performance regresses significantly

---

## Phase 6: Coverage & Reporting

```bash
# Python
pytest --cov=src --cov-report=html --cov-report=term-missing

# JavaScript
npm test -- --coverage

# Identify gaps
python analyze_coverage_gaps.py coverage.json
```

**Prioritization:** Focus on high-complexity, high-risk gaps first

---

## Test Organization

```
tests/
├── unit/
│   ├── test_module1.py
│   └── test_module2.py
├── integration/
│   └── test_workflow.py
└── conftest.py  # Shared fixtures
```

**Naming:** `test_{module_name}.py`, `Test{ClassName}`, `test_{function_name}_{scenario}`

---

## Best Practices

| Practice | Description |
|----------|-------------|
| Analyze first | Understand code before generating |
| Happy path first | Then edge cases |
| Parametrize | Multiple scenarios in single test |
| Mock external | Isolate unit under test |
| Meaningful names | Describe what is tested |
| AAA pattern | Arrange, Act, Assert |
| Type-aware assertions | Based on return types |
| Async handling | Use appropriate async test patterns |
| Consistency | Match existing test patterns |

---

## External Documentation

Comprehensive guides in `docs/test-generate/`:
- `test-generation-patterns.md` - AST parsing, algorithms, templates
- `scientific-testing-guide.md` - Numerical correctness, tolerances
- `property-based-testing.md` - Hypothesis patterns, stateful testing
- `coverage-analysis-guide.md` - Gap identification, prioritization
