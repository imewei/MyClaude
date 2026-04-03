---
name: testing-patterns
description: Cross-language testing strategy guide covering framework selection, test pyramid design, CI/CD integration, and parallel execution for Python (pytest) and JavaScript/TypeScript (Jest/Vitest). Use when choosing testing frameworks, designing test architecture, or setting up test infrastructure across ecosystems.
---

# Testing Patterns (Cross-Language Guide)

## Expert Agent

For testing strategy and quality gates across Python and JS/TS ecosystems, delegate to:

- **`quality-specialist`**: Designs comprehensive test strategies and enforces quality gates.
  - *Location*: `plugins/dev-suite/agents/quality-specialist.md`
- **`systems-engineer`**: Implements testing across Python, JavaScript, and TypeScript.
  - *Location*: `plugins/dev-suite/agents/systems-engineer.md`

## Framework Selection

| Ecosystem | Framework | Best For | Speed |
|-----------|-----------|----------|-------|
| Python | pytest | General purpose, fixtures, plugins | Fast |
| Python | Hypothesis | Property-based / fuzz testing | Medium |
| Python | pytest-asyncio | Async code testing | Fast |
| JS/TS | Vitest | Vite projects, ESM-first | Fast |
| JS/TS | Jest | General purpose, mature ecosystem | Moderate |
| JS/TS | Testing Library | Component tests (React, Vue) | - |
| JS/TS | Supertest | API integration testing | - |
| Any | Playwright | E2E browser testing | Slow |

## Test Pyramid

| Level | Scope | Ratio | Tools |
|-------|-------|-------|-------|
| Unit | Function/class | ~70% | pytest, Jest/Vitest |
| Integration | Components/APIs | ~20% | TestClient, Supertest |
| E2E | Full system | ~10% | Playwright, Cypress |

## Cross-Language Best Practices

| Practice | Python | JavaScript/TypeScript |
|----------|--------|----------------------|
| Pattern | AAA (Arrange, Act, Assert) | AAA (Arrange, Act, Assert) |
| Naming | `test_<what>_<when>_<expected>` | `should <expected> when <condition>` |
| Isolation | No shared state between tests | No shared state between tests |
| Assertions | One logical assertion per test | One logical assertion per test |
| Fixtures | `@pytest.fixture` + conftest.py | `beforeEach` + factory functions |
| Mocking | `unittest.mock` / `pytest-mock` | `vi.mock` / `vi.fn()` |
| Coverage | `pytest-cov` (80%+ critical paths) | `v8`/`istanbul` (80%+ critical paths) |
| Parallel | `pytest-xdist -n auto` | Vitest threads / Jest workers |

## Parallel Execution

### Python
| Method | Tool | Command |
|--------|------|---------|
| Multi-Process | pytest-xdist | `pytest -n auto` |
| Distributed | pytest-xdist (socket) | `pytest -d --tx socket=...` |
| Matrix | Tox / CI | Parallel environments |

### JavaScript/TypeScript
- **Vitest**: Parallel threads by default (`poolOptions.threads`)
- **Jest**: `jest --maxWorkers=50%`
- **Playwright**: `fullyParallel: true`

## CI/CD Integration

### Python
```yaml
- run: uv pip install -e ".[dev]"
- run: pytest --cov --cov-report=xml
- uses: codecov/codecov-action@v3
```

### JavaScript/TypeScript
```yaml
- run: npm ci
- run: npx vitest run --coverage
```

## Configuration Quick Reference

### Python (`pyproject.toml`)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v -n auto --cov=myapp --cov-report=term-missing"
markers = ["slow: marks slow tests", "integration: integration tests"]
```

### JavaScript/TypeScript (`vitest.config.ts`)
```typescript
export default defineConfig({
  test: {
    globals: true,
    coverage: {
      provider: 'v8',
      thresholds: { branches: 80, functions: 80, lines: 80 }
    }
  }
});
```

## Checklist

- [ ] Test pyramid balanced (70/20/10)
- [ ] All external dependencies mocked at unit level
- [ ] Integration tests use real DB where feasible
- [ ] Edge cases and error paths covered
- [ ] Async code properly tested
- [ ] Coverage measured for critical paths (80%+)
- [ ] Tests run in CI/CD pipeline
- [ ] Parallel execution configured
- [ ] Test names are descriptive and follow conventions
