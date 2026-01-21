---
name: robust-testing
version: "2.1.0"
description: Implement robust testing strategies using property-based testing, advanced fixtures, and mutation testing. Use when writing tests with Hypothesis, implementing complex pytest fixtures, or ensuring high reliability in scientific computations.
---

# Robust Testing in Python

Move beyond simple unit tests to rigorous verification using property-based testing and modern patterns.

## Expert Agent

For advanced testing architectures, property-based test design, or verifying complex numerical algorithms, delegate to:

- **`python-pro`**: Expert in property-based testing with Hypothesis and rigorous verification.
  - *Location*: `plugins/science-suite/agents/python-pro.md`

## 1. Property-Based Testing (Hypothesis)

Use `Hypothesis` to automatically find edge cases by specifying properties that should always hold true.

```python
from hypothesis import given, strategies as st

def sort_list(lst: list[int]) -> list[int]:
    return sorted(lst)

@given(st.lists(st.integers()))
def test_sort_is_idempotent(lst):
    # Property: sorting a sorted list shouldn't change it
    sorted_once = sort_list(lst)
    assert sort_list(sorted_once) == sorted_once

@given(st.lists(st.integers()))
def test_sort_preserves_length(lst):
    # Property: length should remain the same
    assert len(sort_list(lst)) == len(lst)
```

## 2. Advanced Pytest Fixtures

Leverage scopes and dependency injection for clean, reusable test setups.

```python
import pytest

@pytest.fixture(scope="session")
def api_client():
    # Setup session-wide client
    client = MyClient()
    yield client
    client.close()

@pytest.fixture
def mock_db(monkeypatch):
    # Use monkeypatch within a fixture
    db = MockDB()
    monkeypatch.setattr("myapp.core.db", db)
    return db
```

## 3. Testing Strategies

| Strategy | Tools | Goal |
|----------|-------|------|
| **Property-Based** | Hypothesis | Find edge cases automatically. |
| **Parameterized** | `pytest.mark.parametrize` | Test specific scenarios efficiently. |
| **Snapshot** | `syrupy` / `pytest-snapshot` | Verify large or complex outputs. |
| **Mutation** | `mutmut` | Verify the quality of your tests by mutating source code. |

## 4. Best Practices

- **Independent Tests**: Ensure tests can run in any order without side effects.
- **Shrinking**: Hypothesis automatically "shrinks" failing inputs to the smallest possible example.
- **Numerical Stability**: Use `math.isclose()` or `pytest.approx()` for floating-point comparisons.
- **Coverage**: Use `pytest-cov` but focus on branch coverage and critical paths over 100% line coverage.

## Checklist

- [ ] Property-based tests implemented for complex logic.
- [ ] Edge cases (empty inputs, large numbers, NaN) handled.
- [ ] Fixtures used for resource management (setup/teardown).
- [ ] Floating point comparisons use `approx`.
- [ ] Branch coverage verified for critical algorithms.
