# Python Testing Anti-Patterns

## Overview

Common testing mistakes that lead to fragile, slow, or unreliable test suites.

---

## 🚫 Anti-Pattern 1: Testing Implementation Details

### ❌ Bad: Tightly Coupled Tests

```python
def test_user_creation_implementation():
    """WRONG: Tests internal implementation"""
    user = User()
    assert user._password_hash is None  # Private attribute
    assert user._validate_email_format("test@example.com")  # Private method
    assert isinstance(user._created_at, datetime)  # Internal detail
```

**Problem**: Tests break when refactoring internal code, even if behavior unchanged.

### ✅ Good: Test Public Interface

```python
def test_user_creation_behavior():
    """RIGHT: Tests public API and observable behavior"""
    user = User.create(email="test@example.com", password="secret123")
    assert user.email == "test@example.com"
    assert user.verify_password("secret123")
    assert user.is_active
```

---

## 🚫 Anti-Pattern 2: Flaky Tests (Time-Dependent)

### ❌ Bad: Non-Deterministic Tests

```python
import time

def test_cache_expiration():
    """WRONG: Depends on wall-clock time"""
    cache.set("key", "value", ttl=1)
    time.sleep(0.9)  # Almost expired
    assert cache.get("key") == "value"  # Flaky!
    time.sleep(0.2)  # Should be expired
    assert cache.get("key") is None  # Flaky!
```

**Problem**: Race conditions, timing issues cause intermittent failures.

### ✅ Good: Mock Time or Use Explicit Control

```python
from unittest.mock import patch
from freezegun import freeze_time

@freeze_time("2024-01-01 12:00:00")
def test_cache_expiration():
    """RIGHT: Deterministic time control"""
    cache.set("key", "value", ttl=60)

    with freeze_time("2024-01-01 12:00:30"):
        assert cache.get("key") == "value"

    with freeze_time("2024-01-01 12:01:01"):
        assert cache.get("key") is None  # Always passes
```

---

## 🚫 Anti-Pattern 3: Test Interdependence

### ❌ Bad: Tests Depend on Each Other

```python
class TestUserWorkflow:
    user_id = None  # Shared state!

    def test_1_create_user(self):
        """Must run first"""
        user = create_user("test@example.com")
        TestUserWorkflow.user_id = user.id

    def test_2_update_user(self):
        """Depends on test_1"""
        update_user(TestUserWorkflow.user_id, name="New Name")

    def test_3_delete_user(self):
        """Depends on test_1 and test_2"""
        delete_user(TestUserWorkflow.user_id)
```

**Problem**: Tests must run in order, can't run in parallel, brittle.

### ✅ Good: Independent Tests with Fixtures

```python
@pytest.fixture
def user():
    """Each test gets fresh user"""
    user = create_user("test@example.com")
    yield user
    delete_user(user.id)  # Cleanup

def test_create_user(user):
    """Independent"""
    assert user.email == "test@example.com"

def test_update_user(user):
    """Independent"""
    update_user(user.id, name="New Name")
    updated = get_user(user.id)
    assert updated.name == "New Name"

def test_delete_user(user):
    """Independent"""
    delete_user(user.id)
    assert get_user(user.id) is None
```

---

## 🚫 Anti-Pattern 4: Excessive Mocking

### ❌ Bad: Mock Everything

```python
def test_create_order():
    """WRONG: Over-mocked"""
    with patch('module.validate_user') as mock_validate:
        with patch('module.check_inventory') as mock_inventory:
            with patch('module.calculate_price') as mock_price:
                with patch('module.charge_payment') as mock_payment:
                    with patch('module.send_confirmation') as mock_email:
                        mock_validate.return_value = True
                        mock_inventory.return_value = True
                        mock_price.return_value = 100.0
                        mock_payment.return_value = True
                        mock_email.return_value = None

                        order = create_order(user_id=1, product_id=2)
                        # Not testing much real code!
```

**Problem**: Test doesn't verify actual logic, just mocking infrastructure.

### ✅ Good: Mock Only External Dependencies

```python
def test_create_order():
    """RIGHT: Mock only external systems"""
    with patch('module.payment_gateway') as mock_gateway:
        mock_gateway.charge.return_value = {'success': True, 'transaction_id': '123'}

        # Real validation, inventory, price calculation
        order = create_order(
            user_id=test_user.id,
            product_id=test_product.id
        )

        assert order.total == test_product.price
        assert order.status == 'confirmed'
        mock_gateway.charge.assert_called_once()
```

---

## 🚫 Anti-Pattern 5: Testing Multiple Things at Once

### ❌ Bad: Mega Test

```python
def test_entire_user_lifecycle():
    """WRONG: Tests too much"""
    # Create user
    user = User.create(email="test@example.com", password="secret")
    assert user.id is not None

    # Update user
    user.name = "Test User"
    user.save()
    assert user.name == "Test User"

    # Add posts
    post = user.create_post(title="Hello")
    assert len(user.posts) == 1

    # Delete user
    user.delete()
    assert User.get(user.id) is None
    # When this fails, which part broke?
```

**Problem**: When it fails, unclear what broke. Hard to debug.

### ✅ Good: One Test, One Concept

```python
def test_user_creation():
    """One concept: user creation"""
    user = User.create(email="test@example.com", password="secret")
    assert user.id is not None
    assert user.email == "test@example.com"

def test_user_update():
    """One concept: user update"""
    user = create_test_user()
    user.name = "Test User"
    user.save()
    assert user.name == "Test User"

def test_user_posts_relationship():
    """One concept: posts relationship"""
    user = create_test_user()
    post = user.create_post(title="Hello")
    assert len(user.posts) == 1
    assert post in user.posts
```

---

## 🚫 Anti-Pattern 6: Ignoring Test Performance

### ❌ Bad: Slow Test Suite

```python
def test_api_endpoint():
    """WRONG: Starts entire server for each test"""
    server = start_full_application()  # 5 seconds
    db = initialize_database()  # 3 seconds
    response = requests.get("http://localhost:8000/api/users")
    assert response.status_code == 200
    shutdown_server(server)  # 2 seconds
    # 10 seconds per test!
```

**Problem**: Slow tests = developers don't run them = broken main branch.

### ✅ Good: Fast Tests with Appropriate Scope

```python
# Unit test (milliseconds)
def test_user_validation():
    """Fast: No I/O"""
    is_valid = validate_email("test@example.com")
    assert is_valid

# Integration test (seconds)
@pytest.mark.integration
def test_api_endpoint(test_client):
    """Faster: Use test client, not real server"""
    response = test_client.get("/api/users")
    assert response.status_code == 200

# E2E test (minutes)
@pytest.mark.e2e
def test_full_user_flow(live_server):
    """Slow: Only for critical paths"""
    # Full browser automation
    pass
```

---

## 🚫 Anti-Pattern 7: Not Testing Error Cases

### ❌ Bad: Happy Path Only

```python
def test_divide():
    """WRONG: Only tests success case"""
    result = divide(10, 2)
    assert result == 5.0
    # What about divide by zero?
    # What about invalid inputs?
```

**Problem**: Production bugs from unexpected inputs.

### ✅ Good: Test Error Cases

```python
def test_divide_success():
    """Test happy path"""
    assert divide(10, 2) == 5.0

def test_divide_by_zero():
    """Test error case"""
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_divide_invalid_types():
    """Test validation"""
    with pytest.raises(TypeError):
        divide("10", "2")

def test_divide_edge_cases():
    """Test boundaries"""
    assert divide(0, 5) == 0.0
    assert divide(-10, 2) == -5.0
    assert divide(1, 3) == pytest.approx(0.333, rel=0.001)
```

---

## 🚫 Anti-Pattern 8: Asserting Too Little

### ❌ Bad: Weak Assertions

```python
def test_create_user():
    """WRONG: Doesn't verify enough"""
    user = User.create(email="test@example.com", password="secret")
    assert user is not None  # Weak!
    # Is email set correctly?
    # Is password hashed?
    # Is user active?
```

**Problem**: Tests pass but code is broken.

### ✅ Good: Thorough Assertions

```python
def test_create_user():
    """RIGHT: Verifies all expected behavior"""
    user = User.create(
        email="test@example.com",
        password="secret123"
    )

    # Verify object created
    assert user.id is not None

    # Verify attributes set correctly
    assert user.email == "test@example.com"
    assert user.is_active
    assert user.created_at <= datetime.now()

    # Verify password handling
    assert user.password != "secret123"  # Should be hashed
    assert user.verify_password("secret123")  # Should verify
    assert not user.verify_password("wrong")  # Should reject wrong password
```

---

## 🚫 Anti-Pattern 9: Using Production Database

### ❌ Bad: Test Against Production

```python
def test_delete_user():
    """WRONG: Uses production database!"""
    db = connect_to_production()  # Dangerous!
    user = db.users.find_one({"email": "test@example.com"})
    db.users.delete_one({"_id": user["_id"]})
    # Just deleted production data!
```

**Problem**: Data loss, side effects, impossible to reset.

### ✅ Good: Isolated Test Database

```python
@pytest.fixture(scope="function")
def db():
    """Fresh database for each test"""
    test_db = create_test_database()
    yield test_db
    drop_test_database(test_db)

def test_delete_user(db):
    """Safe: Uses isolated test database"""
    user = db.users.insert_one({"email": "test@example.com"})
    db.users.delete_one({"_id": user.inserted_id})
    assert db.users.count_documents({}) == 0
```

---

## 🚫 Anti-Pattern 10: Unclear Test Names

### ❌ Bad: Vague Names

```python
def test_user():
    """WRONG: What is being tested?"""
    pass

def test_1():
    """WRONG: Meaningless"""
    pass

def test_something():
    """WRONG: Too vague"""
    pass
```

**Problem**: Can't understand what failed without reading code.

### ✅ Good: Descriptive Names

```python
def test_create_user_with_valid_email_succeeds():
    """Clear: What, when, expected result"""
    pass

def test_create_user_with_invalid_email_raises_validation_error():
    """Clear: Specific scenario"""
    pass

def test_user_can_update_name_but_not_email():
    """Clear: Business rule"""
    pass
```

---

## Quick Reference: Testing Smells

| Smell | Problem | Fix |
|-------|---------|-----|
| Long test method | Tests too much | Split into focused tests |
| Many mocks | Over-mocking | Mock only external deps |
| Sleeps/waits | Flaky timing | Mock time or use callbacks |
| Shared state | Interdependent | Use fixtures |
| Private access | Implementation coupling | Test public API |
| No error tests | Missing coverage | Test failure paths |
| Prod database | Data corruption risk | Use test database |
| Vague names | Hard to debug | Descriptive names |
| Weak assertions | False positives | Assert all behavior |
| Slow suite | Developers skip | Optimize or split |

## Test Pyramid

```
         ┌──────────┐
         │   E2E    │  ← Few, slow, brittle
         │  Tests   │     (Full system)
         └──────────┘
      ┌──────────────┐
      │ Integration  │  ← Some, medium speed
      │    Tests     │     (Multiple units)
      └──────────────┘
   ┌──────────────────┐
   │   Unit Tests     │  ← Many, fast, reliable
   │  (Single units)  │     (Pure functions)
   └──────────────────┘
```

**Goal**: Most tests should be fast unit tests, few slow E2E tests.

## Further Reading

- [Test Fixtures Patterns](../SKILL.md#fixtures)
- [Mocking Best Practices](../SKILL.md#mocking)
- [Property-Based Testing](../../../../plugins/unit-testing/docs/test-generate/property-based-testing.md)
