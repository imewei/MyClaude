---
name: python-testing-patterns
version: "1.0.7"
description: Comprehensive testing with pytest, fixtures, mocking, parameterization, and TDD. Use when writing test files (test_*.py), creating fixtures in conftest.py, using unittest.mock or pytest-mock, writing parameterized tests, testing async code with pytest-asyncio, implementing property-based testing with Hypothesis, or measuring coverage with pytest-cov.
---

# Python Testing Patterns

## Test Types

| Type | Scope | Speed | Tools |
|------|-------|-------|-------|
| Unit | Function/class | Fast | pytest, mock |
| Integration | Components | Medium | pytest, TestClient |
| E2E | Full system | Slow | pytest, Playwright |
| Property | Random inputs | Medium | Hypothesis |

## Basic Test Structure (AAA)

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

## Fixtures

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

## Parameterized Tests

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

## Mocking

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

## Async Testing

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

## Monkeypatch

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

## Temporary Files

```python
def test_file_operations(tmp_path):
    test_file = tmp_path / "data.txt"
    test_file.write_text("Hello, World!")

    assert test_file.exists()
    assert test_file.read_text() == "Hello, World!"
```

## Property-Based Testing (Hypothesis)

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

## Test Markers

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

## Database Testing

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

## Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=myapp --cov-report=term-missing"
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

## CI/CD Integration

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Best Practices

| Practice | Guideline |
|----------|-----------|
| Naming | `test_<what>_<when>_<expected>` |
| Isolation | No shared state between tests |
| Assertions | One logical assertion per test |
| Fixtures | Use for setup/teardown |
| Mocking | Mock external dependencies only |
| Coverage | Focus on critical paths, not % |
| Speed | Fast tests run more often |

## Checklist

- [ ] Tests follow AAA pattern
- [ ] Fixtures handle setup/teardown
- [ ] External dependencies mocked
- [ ] Edge cases and errors tested
- [ ] Async code uses @pytest.mark.asyncio
- [ ] Coverage measured for critical paths
- [ ] Tests run in CI/CD pipeline
- [ ] Test names are descriptive
