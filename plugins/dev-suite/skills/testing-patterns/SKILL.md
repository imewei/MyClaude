---
name: testing-patterns
description: Comprehensive testing patterns for Python (pytest, fixtures, mocking, Hypothesis) and JavaScript/TypeScript (Jest/Vitest, Testing Library, Supertest). Covers unit tests, integration tests, TDD, async testing, parameterization, property-based testing, and CI/CD integration across both ecosystems.
---

# Testing Patterns (Multi-Language)

## Expert Agent

For multi-language testing patterns, pytest/Jest/Vitest strategies, and TDD workflows, delegate to:

- **`systems-engineer`**: Implements robust testing across Python, JavaScript, and TypeScript ecosystems.
  - *Location*: `plugins/dev-suite/agents/systems-engineer.md`
- **`quality-specialist`**: Designs comprehensive test strategies and enforces quality gates.
  - *Location*: `plugins/dev-suite/agents/quality-specialist.md`

Robust testing strategies covering Python and JavaScript/TypeScript ecosystems.

---

## Python Testing (pytest)

### Test Types

| Type | Scope | Speed | Tools |
|------|-------|-------|-------|
| Unit | Function/class | Fast | pytest, mock |
| Integration | Components | Medium | pytest, TestClient |
| E2E | Full system | Slow | pytest, Playwright |
| Property | Random inputs | Medium | Hypothesis |

### Basic Test Structure (AAA)

```python
import pytest

def test_division():
    # Arrange
    calc = Calculator()

    # Act
    result = calc.divide(10, 2)

    # Assert
    assert result == 5

def test_division_by_zero():
    calc = Calculator()
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calc.divide(10, 0)
```

### Fixtures

```python
# conftest.py
import pytest
from typing import Generator

@pytest.fixture
def db() -> Generator[Database, None, None]:
    """Connected database fixture."""
    database = Database("sqlite:///:memory:")
    database.connect()
    yield database
    database.disconnect()

@pytest.fixture(scope="session")
def app_config():
    """Session-scoped config."""
    return {"debug": True, "api_key": "test"}

@pytest.fixture(autouse=True)
def reset_state():
    """Auto-use: runs before each test."""
    yield
    # Cleanup after test

# Parametrized fixture
@pytest.fixture(params=["sqlite", "postgresql"])
def db_backend(request):
    return request.param
```

### Parameterized Tests

```python
@pytest.mark.parametrize("email,valid", [
    ("user@example.com", True),
    ("test@domain.co.uk", True),
    ("invalid.email", False),
    ("@example.com", False),
])
def test_email_validation(email, valid):
    assert is_valid_email(email) == valid

@pytest.mark.parametrize("a,b,expected", [
    pytest.param(2, 3, 5, id="positive"),
    pytest.param(-1, 1, 0, id="negative"),
    pytest.param(0, 0, 0, id="zero"),
])
def test_addition(a, b, expected):
    assert add(a, b) == expected
```

### Mocking

```python
from unittest.mock import Mock, patch

def test_api_call():
    mock_response = Mock()
    mock_response.json.return_value = {"id": 1, "name": "John"}
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response) as mock_get:
        user = api_client.get_user(1)

        assert user["name"] == "John"
        mock_get.assert_called_once_with("https://api.example.com/users/1")

@patch("requests.post")
def test_create_user(mock_post):
    mock_post.return_value.json.return_value = {"id": 2}
    result = api_client.create_user({"name": "Jane"})
    assert result["id"] == 2
```

### Async Testing

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_fetch():
    result = await fetch_data("https://api.example.com")
    assert "data" in result

@pytest.mark.asyncio
async def test_concurrent():
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    assert len(results) == len(urls)

@pytest.fixture
async def async_client():
    client = await create_client()
    yield client
    await client.close()
```

### Monkeypatch

```python
def test_env_variable(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    assert get_database_url() == "postgresql://test"

def test_env_not_set(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert get_database_url() == "sqlite:///:memory:"

def test_attribute(monkeypatch):
    monkeypatch.setattr(config, "api_key", "test-key")
    assert config.get_api_key() == "test-key"
```

### Temporary Files

```python
def test_file_operations(tmp_path):
    test_file = tmp_path / "data.txt"
    test_file.write_text("Hello, World!")

    assert test_file.exists()
    assert test_file.read_text() == "Hello, World!"
```

### Property-Based Testing (Hypothesis)

```python
from hypothesis import given, strategies as st

@given(st.text())
def test_reverse_twice_is_original(s):
    assert reverse(reverse(s)) == s

@given(st.integers(), st.integers())
def test_addition_commutative(a, b):
    assert a + b == b + a

@given(st.lists(st.integers()))
def test_sorted_is_ordered(lst):
    sorted_lst = sorted(lst)
    for i in range(len(sorted_lst) - 1):
        assert sorted_lst[i] <= sorted_lst[i + 1]
```

### Test Markers

```python
@pytest.mark.slow
def test_slow_operation():
    time.sleep(2)

@pytest.mark.integration
def test_database():
    pass

@pytest.mark.skip(reason="Not implemented")
def test_future():
    pass

@pytest.mark.skipif(os.name == "nt", reason="Unix only")
def test_unix_specific():
    pass

@pytest.mark.xfail(reason="Known bug #123")
def test_known_bug():
    assert False

# Run: pytest -m slow / pytest -m "not slow"
```

### Database Testing

```python
@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()

def test_create_user(db_session):
    user = User(name="Test", email="test@example.com")
    db_session.add(user)
    db_session.commit()

    assert user.id is not None
    assert db_session.query(User).count() == 1
```

### Python Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v -n auto --cov=myapp --cov-report=term-missing"
markers = [
    "slow: marks slow tests",
    "integration: integration tests"
]

[tool.coverage.run]
source = ["myapp"]
omit = ["*/tests/*"]

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "raise NotImplementedError"]
```

### Parallel Execution (Python)

| Method | Tool | Command |
|--------|------|---------|
| **Multi-Process** | pytest-xdist | `pytest -n auto` |
| **Distributed** | pytest-xdist (socket) | `pytest -d --tx socket=...` |
| **Async Tests** | pytest-asyncio | `asyncio.gather(*tests)` (within test) |
| **Matrix** | Tox / CI | Parallel environments |

---

## JavaScript/TypeScript Testing (Jest/Vitest)

### Framework Selection

| Framework | Use Case | Speed |
|-----------|----------|-------|
| Vitest | Vite projects | Fast |
| Jest | General purpose | Moderate |
| Testing Library | Component tests | - |
| Supertest | API integration | - |

### Vitest Configuration

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      thresholds: { branches: 80, functions: 80, lines: 80 }
    },
    setupFiles: ['./src/test/setup.ts'],
    poolOptions: {
      threads: {
        singleThread: false,
        maxThreads: 4
      }
    }
  }
});
```

### Parallelization (JS)

- **Vitest**: Runs in parallel threads by default. Use `poolOptions.threads` to tune.
- **Jest**: Use `maxWorkers` (e.g. `jest --maxWorkers=50%`).
- **Playwright**: Fully parallel by default. Configure `fullyParallel: true` in config.

### Unit Testing Patterns

#### Pure Functions

```typescript
describe('Calculator', () => {
  it('should add two numbers', () => {
    expect(add(2, 3)).toBe(5);
  });

  it('should throw on division by zero', () => {
    expect(() => divide(10, 0)).toThrow('Division by zero');
  });
});
```

#### Class Testing

```typescript
describe('UserService', () => {
  let service: UserService;

  beforeEach(() => {
    service = new UserService();
  });

  it('should create user', () => {
    const user = service.create({ id: '1', name: 'John' });
    expect(service.findById('1')).toEqual(user);
  });

  it('should throw if user exists', () => {
    service.create({ id: '1', name: 'John' });
    expect(() => service.create({ id: '1', name: 'Jane' })).toThrow('User already exists');
  });
});
```

#### Async Testing

```typescript
describe('ApiService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch user', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ id: '1', name: 'John' })
    });

    const user = await service.fetchUser('1');
    expect(user).toEqual({ id: '1', name: 'John' });
  });

  it('should throw on not found', async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: false });
    await expect(service.fetchUser('999')).rejects.toThrow('User not found');
  });
});
```

### Mocking Patterns

#### Module Mocking

```typescript
vi.mock('nodemailer', () => ({
  default: {
    createTransport: vi.fn(() => ({
      sendMail: vi.fn().mockResolvedValue({ messageId: '123' })
    }))
  }
}));
```

#### Dependency Injection

```typescript
describe('UserService', () => {
  let service: UserService;
  let mockRepository: IUserRepository;

  beforeEach(() => {
    mockRepository = { findById: vi.fn(), create: vi.fn() };
    service = new UserService(mockRepository);
  });

  it('should return user if found', async () => {
    vi.mocked(mockRepository.findById).mockResolvedValue({ id: '1', name: 'John' });
    const user = await service.getUser('1');
    expect(user.name).toBe('John');
  });
});
```

#### Spying

```typescript
const loggerSpy = vi.spyOn(logger, 'info');

await service.processOrder('123');

expect(loggerSpy).toHaveBeenCalledWith('Processing order 123');
expect(loggerSpy).toHaveBeenCalledTimes(2);
```

### Integration Testing (Supertest)

```typescript
describe('User API', () => {
  beforeEach(async () => {
    await pool.query('TRUNCATE TABLE users CASCADE');
  });

  it('should create user', async () => {
    const response = await request(app)
      .post('/api/users')
      .send({ name: 'John', email: 'john@example.com', password: 'pass123' })
      .expect(201);

    expect(response.body).toHaveProperty('id');
    expect(response.body.email).toBe('john@example.com');
  });

  it('should return 409 if email exists', async () => {
    await request(app).post('/api/users').send({ name: 'John', email: 'john@example.com', password: 'pass' });
    await request(app).post('/api/users').send({ name: 'Jane', email: 'john@example.com', password: 'pass' }).expect(409);
  });

  it('should require auth for protected routes', async () => {
    await request(app).get('/api/users/me').expect(401);
  });
});
```

### React Component Testing

```typescript
import { render, screen, fireEvent } from '@testing-library/react';

describe('UserForm', () => {
  it('should render form inputs', () => {
    render(<UserForm onSubmit={vi.fn()} />);
    expect(screen.getByPlaceholderText('Name')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Submit' })).toBeInTheDocument();
  });

  it('should call onSubmit with form data', () => {
    const onSubmit = vi.fn();
    render(<UserForm onSubmit={onSubmit} />);

    fireEvent.change(screen.getByTestId('name-input'), { target: { value: 'John' } });
    fireEvent.change(screen.getByTestId('email-input'), { target: { value: 'john@example.com' } });
    fireEvent.click(screen.getByRole('button', { name: 'Submit' }));

    expect(onSubmit).toHaveBeenCalledWith({ name: 'John', email: 'john@example.com' });
  });
});
```

### Hook Testing

```typescript
import { renderHook, act } from '@testing-library/react';

describe('useCounter', () => {
  it('should increment', () => {
    const { result } = renderHook(() => useCounter(0));
    act(() => result.current.increment());
    expect(result.current.count).toBe(1);
  });
});
```

### Test Fixtures (JS)

```typescript
import { faker } from '@faker-js/faker';

export function createUserFixture(overrides?: Partial<User>): User {
  return {
    id: faker.string.uuid(),
    name: faker.person.fullName(),
    email: faker.internet.email(),
    ...overrides
  };
}
```

### Timer Testing

```typescript
it('should call after delay', () => {
  vi.useFakeTimers();
  const callback = vi.fn();

  setTimeout(callback, 1000);
  expect(callback).not.toHaveBeenCalled();

  vi.advanceTimersByTime(1000);
  expect(callback).toHaveBeenCalled();

  vi.useRealTimers();
});
```

### JS Test Organization

```typescript
describe('UserService', () => {
  describe('createUser', () => {
    it('should create user', () => {});
    it('should throw if email exists', () => {});
  });

  describe('updateUser', () => {
    it('should update user', () => {});
    it('should throw if not found', () => {});
  });
});
```

### JS Commands

```bash
vitest                    # Watch mode
vitest --coverage         # With coverage
vitest --ui               # UI mode
vitest run                # Single run
vitest run src/user.test.ts  # Specific file
```

---

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

## CI/CD Integration

### Python
```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: "3.12" }
      - run: uv pip install -e ".[dev]"
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v3
```

### JavaScript/TypeScript
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - run: npm ci
      - run: npx vitest run --coverage
```

## Checklist

### Python
- [ ] Tests follow AAA pattern
- [ ] Fixtures handle setup/teardown
- [ ] External dependencies mocked
- [ ] Edge cases and errors tested
- [ ] Async code uses @pytest.mark.asyncio
- [ ] Coverage measured for critical paths
- [ ] Tests run in CI/CD pipeline
- [ ] Test names are descriptive

### JavaScript/TypeScript
- [ ] Unit tests for business logic
- [ ] Integration tests for APIs
- [ ] Component tests for UI
- [ ] Mocks for external dependencies
- [ ] Edge cases covered
- [ ] Error handling tested
- [ ] 80%+ coverage on critical paths
- [ ] CI/CD integration
