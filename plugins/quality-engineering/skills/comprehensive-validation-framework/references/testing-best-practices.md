# Testing Best Practices Reference

Comprehensive guide to test strategies, coverage, and quality assurance.

## Testing Pyramid

```
           /\
          /  \  E2E Tests (Few)
         /----\
        / Integration \ (Some)
       /     Tests     \
      /----------------\
     /   Unit Tests     \ (Many)
    /____________________\
```

**Distribution**: 70% Unit, 20% Integration, 10% E2E

---

## Unit Testing

### AAA Pattern

**Arrange → Act → Assert**

```python
def test_user_creation():
    # Arrange
    username = "testuser"
    email = "test@example.com"

    # Act
    user = User.create(username=username, email=email)

    # Assert
    assert user.username == username
    assert user.email == email
    assert user.id is not None
```

### Test Naming

**Pattern**: `test_<what>_<condition>_<expected>`

```python
def test_create_user_with_valid_data_succeeds()
def test_create_user_with_duplicate_email_raises_error()
def test_get_user_when_not_found_returns_none()
```

### Test Independence

**❌ Bad - Tests depend on each other**:
```python
user = None

def test_create_user():
    global user
    user = User.create("test")

def test_update_user():
    user.update(name="updated")  # Depends on previous test!
```

**✅ Good - Independent tests**:
```python
@pytest.fixture
def user():
    return User.create("test")

def test_create_user():
    user = User.create("test")
    assert user.name == "test"

def test_update_user(user):
    user.update(name="updated")
    assert user.name == "updated"
```

---

## Test Coverage

### Target Coverage

- **Overall**: >80%
- **Critical paths**: >95%
- **Utility code**: >90%
- **UI components**: >70%

### Coverage Tools

**JavaScript**:
```bash
npm test -- --coverage
npx jest --coverage
```

**Python**:
```bash
pytest --cov=src --cov-report=html --cov-report=term
```

**Rust**:
```bash
cargo tarpaulin --out Html
```

### Coverage != Quality

**100% coverage doesn't mean bug-free!**

```python
def divide(a, b):
    return a / b

def test_divide():
    assert divide(10, 2) == 5  # 100% coverage, but missing ZeroDivisionError test!
```

---

## Mocking and Stubbing

### When to Mock

- External APIs
- Databases
- File system
- Time-dependent code
- Random number generation

### Python Example (pytest)

```python
from unittest.mock import Mock, patch

def test_api_call_success(mocker):
    # Mock external API
    mock_response = Mock()
    mock_response.json.return_value = {"data": "success"}
    mock_response.status_code = 200

    mocker.patch('requests.get', return_value=mock_response)

    result = fetch_data()
    assert result == {"data": "success"}

def test_current_time(mocker):
    # Mock time
    mocker.patch('time.time', return_value=1609459200)  # 2021-01-01
    assert get_current_timestamp() == 1609459200
```

### JavaScript Example (Jest)

```javascript
import { fetchUser } from './api';

jest.mock('./api');

test('displays user data', async () => {
  fetchUser.mockResolvedValue({ name: 'Test User' });

  const user = await loadUserProfile(123);
  expect(user.name).toBe('Test User');
});
```

---

## Integration Testing

### Database Integration

**Use test database**:

```python
import pytest
from sqlalchemy import create_engine

@pytest.fixture(scope="function")
def db():
    # Create test database
    engine = create_engine("postgresql://localhost/test_db")
    Base.metadata.create_all(engine)

    yield engine

    # Cleanup
    Base.metadata.drop_all(engine)
```

### API Integration Tests

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_create_and_retrieve_user():
    # Create user
    response = client.post("/users", json={"name": "Test", "email": "test@example.com"})
    assert response.status_code == 201
    user_id = response.json()["id"]

    # Retrieve user
    response = client.get(f"/users/{user_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Test"
```

---

## End-to-End Testing

### Playwright Example

```javascript
// e2e/login.spec.js
const { test, expect } = require('@playwright/test');

test('user can login successfully', async ({ page }) => {
  await page.goto('https://example.com/login');

  await page.fill('input[name="email"]', 'test@example.com');
  await page.fill('input[name="password"]', 'password123');
  await page.click('button[type="submit"]');

  await expect(page).toHaveURL('https://example.com/dashboard');
  await expect(page.locator('h1')).toContainText('Welcome');
});
```

### Cypress Example

```javascript
// cypress/e2e/checkout.cy.js
describe('Checkout Flow', () => {
  it('completes purchase', () => {
    cy.visit('/products');
    cy.get('[data-testid="add-to-cart"]').first().click();
    cy.get('[data-testid="cart-icon"]').click();
    cy.get('[data-testid="checkout-button"]').click();
    cy.get('[data-testid="payment-form"]').should('be.visible');
  });
});
```

---

## Property-Based Testing

### Python (Hypothesis)

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers()))
def test_sort_is_idempotent(lst):
    """Sorting twice should equal sorting once."""
    assert sorted(sorted(lst)) == sorted(lst)

@given(st.integers(min_value=0), st.integers(min_value=0))
def test_addition_commutative(a, b):
    """a + b should equal b + a."""
    assert a + b == b + a
```

### JavaScript (fast-check)

```javascript
import fc from 'fast-check';

test('reverse is involutive', () => {
  fc.assert(
    fc.property(fc.array(fc.anything()), (arr) => {
      const reversed = reverse(reverse(arr));
      expect(reversed).toEqual(arr);
    })
  );
});
```

---

## Test Data Management

### Factories

**Python (factory_boy)**:
```python
import factory

class UserFactory(factory.Factory):
    class Meta:
        model = User

    username = factory.Sequence(lambda n: f'user{n}')
    email = factory.LazyAttribute(lambda obj: f'{obj.username}@example.com')
    is_active = True

# Usage
user = UserFactory.create()
users = UserFactory.create_batch(10)
```

**JavaScript**:
```javascript
const userFactory = (overrides = {}) => ({
  id: Math.random().toString(),
  name: 'Test User',
  email: 'test@example.com',
  ...overrides
});

const user = userFactory({ name: 'Custom Name' });
```

---

## Mutation Testing

**Verify test quality by introducing bugs**

**Python (mutmut)**:
```bash
mutmut run
mutmut results
mutmut show 1
```

**JavaScript (Stryker)**:
```bash
npx stryker run
```

**Example**: Mutmut changes `==` to `!=`, if tests still pass, tests are weak!

---

## Snapshot Testing

### React Components (Jest)

```javascript
import { render } from '@testing-library/react';

test('UserProfile renders correctly', () => {
  const { container } = render(<UserProfile name="Test" />);
  expect(container).toMatchSnapshot();
});
```

**Update snapshots**:
```bash
npm test -- -u
```

---

## Performance Testing

### Benchmarking

**Python (pytest-benchmark)**:
```python
def test_fibonacci_performance(benchmark):
    result = benchmark(fibonacci, 20)
    assert result == 6765
```

**JavaScript (benchmark.js)**:
```javascript
suite.add('Array#forEach', () => {
  array.forEach(x => x * 2);
}).add('Array#map', () => {
  array.map(x => x * 2);
}).run();
```

---

## Test Organization

### Directory Structure

```
tests/
├── unit/
│   ├── models/
│   ├── services/
│   └── utils/
├── integration/
│   ├── api/
│   └── database/
├── e2e/
│   └── user_flows/
└── fixtures/
    └── data.json
```

### Test Configuration

**pytest.ini**:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --cov=src
    --cov-report=html
    --cov-report=term
    -v
```

**jest.config.js**:
```javascript
module.exports = {
  testEnvironment: 'node',
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },
  collectCoverageFrom: ['src/**/*.js']
};
```

---

## Testing Checklist

### Every Feature Should Have

- [ ] Unit tests for core logic
- [ ] Integration tests for API/DB interactions
- [ ] E2E tests for critical user flows
- [ ] Edge case tests (null, empty, boundary values)
- [ ] Error path tests
- [ ] Performance tests (if performance-critical)

### Code Review Checklist

- [ ] All tests pass
- [ ] Coverage >80% for new code
- [ ] Tests are independent
- [ ] Tests have clear names
- [ ] Mocks are used appropriately
- [ ] No hardcoded test data in assertions
- [ ] No flaky tests

---

## References

- [Testing Best Practices (Microsoft)](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/)
- [Test Pyramid (Martin Fowler)](https://martinfowler.com/articles/practical-test-pyramid.html)
- [Effective Testing (Google)](https://testing.googleblog.com/)
