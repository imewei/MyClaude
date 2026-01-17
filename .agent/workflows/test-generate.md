---
description: Generate comprehensive test suites with scientific computing support,
  numerical validation, property-based testing, benchmarks
triggers:
- /test-generate
- generate comprehensive test suites
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<source> [--coverage] [--property-based] [--benchmarks] [--scientific]`
The agent should parse these arguments from the user's request.

# Automated Test Generation

$ARGUMENTS

## Modes

| Mode | Time | Scope | Types | Coverage | Output |
|------|------|-------|-------|----------|--------|
| Quick | 30min-1h | Single module | Unit only | Basic | ~50-100 tests |
| Standard | 2-4h | Package/feature | Unit+integration, property-based | >80% | ~200-500 tests |
| Enterprise | 1-2d | Entire project | Unit+int+E2E+mutation | >90% | ~1,000+ tests |

Options: `--coverage`, `--property-based`, `--benchmarks`, `--scientific`

## Process

1. **Code Analysis (AST)**:
   - Parse files, extract functions/classes/methods
   - Identify params, return types, complexity
   - Extract docstrings for expected behavior
   - Coverage gap detection (uncovered functions)
   - Dependency mapping (external vs internal, mocks, integration points)

2. **Framework Selection**:
   | Lang | Framework | Style |
   |------|-----------|-------|
   | Python | pytest | fixtures, parametrize, markers |
   | JavaScript | Jest/Vitest | describe/it, beforeEach, mock |
   | Julia | Test.jl | @testset, @test, @inferred |

3. **Test Categories** (AAA Pattern: Arrange → Act → Assert):
   - **Happy Path**: Valid input, expected behavior
   - **Edge Cases**: Empty, null, boundary values
   - **Error Handling**: Invalid input, exceptions
   - **Parametrized**: Multiple scenarios via parametrize/each

4. **Scientific Tests** (if `--scientific`):
   - **Numerical**: Analytical solution comparison, tolerance assertions (`assert_allclose(result, expected, rtol=1e-12)`), edge values (empty, zeros, large/small), no inf/nan
   - **JAX**: JIT equivalence (`jit(fn)(x) == fn(x)`), gradient correctness (analytical vs finite diff), vmap correctness (batched = individual)

5. **Property-Based** (if `--property-based`):
   - **Properties**: Idempotence (`f(f(x)) == f(x)`), Commutativity (`f(a,b) == f(b,a)`), Associativity, Linearity, Inverse
   - **Hypothesis**: Use `hypothesis.strategies`, `hypothesis.extra.numpy`, configure `max_examples`, `deadline`

6. **Benchmarks** (if `--benchmarks`):
   - Multiple input sizes (10, 100, 1k, 10k)
   - Memory via tracemalloc
   - pytest-benchmark integration
   - Performance assertions (upper bounds, regression checks)

7. **Coverage & Reporting**:
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
npm test -- --coverage
python analyze_coverage_gaps.py coverage.json
```

## Test Organization

```
tests/
├── unit/test_module1.py, test_module2.py
├── integration/test_workflow.py
└── conftest.py  # Shared fixtures
```

Naming: `test_{module}.py`, `Test{Class}`, `test_{function}_{scenario}`

## Best Practices

| Practice | Description |
|----------|-------------|
| Analyze first | Understand before generating |
| Happy first | Then edge cases |
| Parametrize | Multiple scenarios in one |
| Mock external | Isolate unit under test |
| Meaningful names | Describe what's tested |
| AAA pattern | Arrange, Act, Assert |
| Type-aware assertions | Based on return types |
| Async handling | Appropriate async patterns |
| Consistency | Match existing patterns |

## External Docs

- `test-generation-patterns.md` - AST parsing, algorithms, templates
- `scientific-testing-guide.md` - Numerical correctness, tolerances
- `property-based-testing.md` - Hypothesis patterns, stateful
- `coverage-analysis-guide.md` - Gap ID, prioritization
