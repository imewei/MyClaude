# Multi-Language Testing Guide

Comprehensive patterns for testing across Python, JavaScript/TypeScript, Rust, Go, Java, and other languages in monorepo and polyglot environments.

## Language-Specific Patterns

### Python Testing Patterns

**Test Organization**:
```
project/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── user.py
│       └── api.py
└── tests/
    ├── unit/
    │   ├── test_user.py
    │   └── test_api.py
    ├── integration/
    │   └── test_workflow.py
    └── conftest.py  # Shared fixtures
```

**conftest.py** (Shared Fixtures):
```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def db_engine():
    """Database engine for all tests"""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture
def db_session(db_engine):
    """Database session with automatic rollback"""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def sample_user():
    """Sample user data"""
    return {
        'name': 'John Doe',
        'email': 'john@example.com',
        'age': 30
    }
```

**Test Patterns**:
```python
# Unit test
def test_user_creation(db_session, sample_user):
    user = User(**sample_user)
    db_session.add(user)
    db_session.commit()

    assert user.id is not None
    assert user.name == sample_user['name']

# Parametrized test
@pytest.mark.parametrize("email,valid", [
    ("user@example.com", True),
    ("invalid", False),
    ("@example.com", False),
    ("user@", False),
])
def test_email_validation(email, valid):
    result = validate_email(email)
    assert result == valid

# Exception testing
def test_invalid_age():
    with pytest.raises(ValueError, match="Age must be positive"):
        User(name="John", email="john@example.com", age=-1)

# Async test
@pytest.mark.asyncio
async def test_async_function():
    result = await async_fetch_data()
    assert result is not None
```

### JavaScript/TypeScript Testing Patterns

**Test Organization**:
```
project/
├── src/
│   ├── user.ts
│   └── api.ts
├── __tests__/
│   ├── user.test.ts
│   └── api.test.ts
└── test/
    ├── setup.ts
    └── fixtures/
        └── userData.ts
```

**Jest Configuration**:
```typescript
// jest.config.ts
import type { Config } from 'jest';

const config: Config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.ts', '**/*.test.ts'],
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/**/*.test.ts'
  ],
  setupFilesAfterEnv: ['<rootDir>/test/setup.ts'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1'
  }
};

export default config;
```

**Test Patterns**:
```typescript
// Unit test
describe('User', () => {
  let user: User;

  beforeEach(() => {
    user = new User('John', 'john@example.com');
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should create user with valid data', () => {
    expect(user.name).toBe('John');
    expect(user.email).toBe('john@example.com');
  });

  it('should validate email format', () => {
    expect(() => new User('John', 'invalid')).toThrow('Invalid email');
  });

  // Mocking
  it('should fetch user from API', async () => {
    const mockFetch = jest.fn().mockResolvedValue({
      json: async () => ({ id: 1, name: 'John' })
    });
    global.fetch = mockFetch;

    const user = await fetchUser(1);

    expect(mockFetch).toHaveBeenCalledWith('/api/users/1');
    expect(user.name).toBe('John');
  });

  // Snapshot testing
  it('should match snapshot', () => {
    expect(user.toJSON()).toMatchSnapshot();
  });
});

// Parametrized test (using test.each)
describe.each([
  ['user@example.com', true],
  ['invalid', false],
  ['@example.com', false],
])('Email validation for %s', (email, expected) => {
  it(`should return ${expected}`, () => {
    expect(validateEmail(email)).toBe(expected);
  });
});
```

### Rust Testing Patterns

**Test Organization**:
```
project/
├── src/
│   ├── lib.rs
│   ├── user.rs
│   └── api.rs
└── tests/
    └── integration_test.rs
```

**Unit Tests** (in `src/user.rs`):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_creation() {
        let user = User::new("John", "john@example.com");
        assert_eq!(user.name, "John");
        assert_eq!(user.email, "john@example.com");
    }

    #[test]
    #[should_panic(expected = "Invalid email")]
    fn test_invalid_email() {
        User::new("John", "invalid");
    }

    #[test]
    fn test_user_age_validation() -> Result<(), String> {
        let user = User::with_age("John", "john@example.com", 30)?;
        assert_eq!(user.age, 30);
        Ok(())
    }

    // Parametrized test (using rstest)
    use rstest::rstest;

    #[rstest]
    #[case("user@example.com", true)]
    #[case("invalid", false)]
    #[case("@example.com", false)]
    fn test_email_validation(#[case] email: &str, #[case] expected: bool) {
        assert_eq!(validate_email(email), expected);
    }

    // Async test
    #[tokio::test]
    async fn test_async_fetch() {
        let result = fetch_user(1).await;
        assert!(result.is_ok());
    }
}
```

**Integration Tests** (in `tests/integration_test.rs`):
```rust
use my_crate::*;

#[test]
fn test_full_workflow() {
    let user = User::new("John", "john@example.com");
    let saved = save_user(&user);
    assert!(saved.is_ok());

    let fetched = get_user(user.id);
    assert_eq!(fetched.unwrap().name, "John");
}

// Test fixtures
fn setup() -> TestContext {
    TestContext {
        db: Database::new_in_memory(),
    }
}

#[test]
fn test_with_setup() {
    let ctx = setup();
    // Use ctx.db for tests
}
```

### Go Testing Patterns

**Test Organization**:
```
project/
├── pkg/
│   └── user/
│       ├── user.go
│       └── user_test.go
└── internal/
    └── api/
        ├── api.go
        └── api_test.go
```

**Test Patterns**:
```go
package user

import (
    "testing"
)

// Basic test
func TestUserCreation(t *testing.T) {
    user := NewUser("John", "john@example.com")

    if user.Name != "John" {
        t.Errorf("Expected name John, got %s", user.Name)
    }
}

// Table-driven test
func TestEmailValidation(t *testing.T) {
    tests := []struct {
        name  string
        email string
        want  bool
    }{
        {"valid email", "user@example.com", true},
        {"invalid email", "invalid", false},
        {"missing domain", "user@", false},
        {"missing local", "@example.com", false},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := ValidateEmail(tt.email)
            if got != tt.want {
                t.Errorf("ValidateEmail(%s) = %v, want %v",
                    tt.email, got, tt.want)
            }
        })
    }
}

// Subtests
func TestUser(t *testing.T) {
    user := NewUser("John", "john@example.com")

    t.Run("name", func(t *testing.T) {
        if user.Name != "John" {
            t.Error("Incorrect name")
        }
    })

    t.Run("email", func(t *testing.T) {
        if user.Email != "john@example.com" {
            t.Error("Incorrect email")
        }
    })
}

// Benchmarks
func BenchmarkUserCreation(b *testing.B) {
    for i := 0; i < b.N; i++ {
        NewUser("John", "john@example.com")
    }
}

// Test helpers
func TestMain(m *testing.M) {
    // Setup
    setup()

    // Run tests
    code := m.Run()

    // Teardown
    teardown()

    os.Exit(code)
}
```

### Java Testing Patterns (JUnit 5)

**Test Organization**:
```
project/
└── src/
    ├── main/
    │   └── java/
    │       └── com/example/
    │           ├── User.java
    │           └── UserService.java
    └── test/
        └── java/
            └── com/example/
                ├── UserTest.java
                └── UserServiceTest.java
```

**Test Patterns**:
```java
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.*;

class UserTest {
    private User user;

    @BeforeEach
    void setUp() {
        user = new User("John", "john@example.com");
    }

    @AfterEach
    void tearDown() {
        // Cleanup
    }

    @Test
    @DisplayName("Should create user with valid data")
    void testUserCreation() {
        assertEquals("John", user.getName());
        assertEquals("john@example.com", user.getEmail());
    }

    @Test
    void testInvalidEmail() {
        assertThrows(IllegalArgumentException.class, () -> {
            new User("John", "invalid");
        });
    }

    // Parametrized test
    @ParameterizedTest
    @CsvSource({
        "user@example.com, true",
        "invalid, false",
        "@example.com, false"
    })
    void testEmailValidation(String email, boolean expected) {
        assertEquals(expected, User.validateEmail(email));
    }

    // Timeout test
    @Test
    @Timeout(value = 100, unit = TimeUnit.MILLISECONDS)
    void testPerformance() {
        // Should complete within 100ms
        heavyOperation();
    }

    // Conditional test
    @Test
    @EnabledOnOs(OS.LINUX)
    void testOnLinuxOnly() {
        // Only runs on Linux
    }

    @Test
    @EnabledIf("customCondition")
    void testConditional() {
        // Runs if customCondition() returns true
    }

    boolean customCondition() {
        return System.getenv("ENV") != null;
    }
}
```

## Cross-Language Test Organization

### Monorepo Structure

```
monorepo/
├── packages/
│   ├── backend/          (Python)
│   │   ├── src/
│   │   ├── tests/
│   │   └── pytest.ini
│   ├── frontend/         (TypeScript)
│   │   ├── src/
│   │   ├── __tests__/
│   │   └── jest.config.ts
│   ├── api-gateway/      (Go)
│   │   ├── pkg/
│   │   └── *_test.go
│   └── services/
│       └── auth/         (Rust)
│           ├── src/
│           ├── tests/
│           └── Cargo.toml
├── scripts/
│   └── run-all-tests.sh
└── .github/
    └── workflows/
        └── test.yml
```

### Unified Test Runner

```bash
#!/bin/bash
# scripts/run-all-tests.sh

set -e

echo "Running tests across all packages..."

# Python tests
echo " Testing Python backend..."
cd packages/backend
pytest --cov=src --cov-report=xml
cd ../..

# TypeScript tests
echo " Testing TypeScript frontend..."
cd packages/frontend
npm test -- --coverage
cd ../..

# Go tests
echo " Testing Go API gateway..."
cd packages/api-gateway
go test ./... -coverprofile=coverage.out
cd ../..

# Rust tests
echo " Testing Rust auth service..."
cd packages/services/auth
cargo test --all-features
cd ../../..

echo " All tests completed successfully!"
```

### GitHub Actions for Monorepo

```yaml
name: Monorepo Tests

on: [push, pull_request]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      backend: ${{ steps.filter.outputs.backend }}
      frontend: ${{ steps.filter.outputs.frontend }}
      go-api: ${{ steps.filter.outputs.go-api }}
      rust-auth: ${{ steps.filter.outputs.rust-auth }}
    steps:
      - uses: actions/checkout@v3
      - uses: dorny/paths-filter@v2
        id: filter
        with:
          filters: |
            backend:
              - 'packages/backend/**'
            frontend:
              - 'packages/frontend/**'
            go-api:
              - 'packages/api-gateway/**'
            rust-auth:
              - 'packages/services/auth/**'

  test-backend:
    needs: detect-changes
    if: needs.detect-changes.outputs.backend == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: |
          cd packages/backend
          pip install -r requirements-dev.txt
          pytest --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3

  test-frontend:
    needs: detect-changes
    if: needs.detect-changes.outputs.frontend == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '20'
      - run: |
          cd packages/frontend
          npm ci
          npm test -- --coverage
      - uses: codecov/codecov-action@v3

  test-go-api:
    needs: detect-changes
    if: needs.detect-changes.outputs.go-api == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.21'
      - run: |
          cd packages/api-gateway
          go test ./... -coverprofile=coverage.out
          go tool cover -html=coverage.out -o coverage.html
      - uses: codecov/codecov-action@v3

  test-rust-auth:
    needs: detect-changes
    if: needs.detect-changes.outputs.rust-auth == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - run: |
          cd packages/services/auth
          cargo test --all-features
          cargo tarpaulin --out Xml
      - uses: codecov/codecov-action@v3
```

## Framework Comparison Matrix

| Feature | Python (pytest) | JavaScript (Jest) | Rust (cargo test) | Go (go test) | Java (JUnit 5) |
|---------|-----------------|-------------------|-------------------|--------------|----------------|
| **Parallel Execution** | ✅ (pytest-xdist) | ✅ (built-in) | ✅ (built-in) | ✅ (built-in) | ✅ (JUnit Platform) |
| **Parametrized Tests** | ✅ (@pytest.mark.parametrize) | ✅ (test.each) | ✅ (rstest) | ✅ (table-driven) | ✅ (@ParameterizedTest) |
| **Fixtures/Setup** | ✅ (@pytest.fixture) | ✅ (beforeEach) | ⚠️ (manual) | ✅ (TestMain) | ✅ (@BeforeEach) |
| **Mocking** | ✅ (unittest.mock) | ✅ (jest.mock) | ✅ (mockall) | ⚠️ (manual/gomock) | ✅ (Mockito) |
| **Coverage** | ✅ (pytest-cov) | ✅ (built-in) | ✅ (tarpaulin) | ✅ (built-in) | ✅ (JaCoCo) |
| **Async Support** | ✅ (pytest-asyncio) | ✅ (built-in) | ✅ (tokio) | ✅ (built-in) | ⚠️ (manual) |
| **Snapshot Testing** | ✅ (pytest-snapshot) | ✅ (built-in) | ✅ (insta) | ❌ | ❌ |

## Best Practices for Polyglot Projects

### 1. Consistent Test Structure

```
Adopt consistent patterns across languages:
- tests/ or __tests__/ directory
- test_*.* or *.test.* naming
- unit/, integration/, e2e/ subdirectories
- Shared fixtures or test data
```

### 2. Unified CI/CD Pipeline

```yaml
# Use matrix builds for different languages
strategy:
  matrix:
    project:
      - { path: 'backend', lang: 'python', cmd: 'pytest' }
      - { path: 'frontend', lang: 'node', cmd: 'npm test' }
      - { path: 'api', lang: 'go', cmd: 'go test ./...' }
```

### 3. Shared Test Data

```
# Share test fixtures across languages
test-data/
├── users.json
├── products.json
└── orders.json

# All languages load from same source
```

### 4. Consistent Coverage Requirements

```
# Set same coverage threshold across all projects
- Python: 80% (in pytest.ini)
- JavaScript: 80% (in jest.config.js)
- Go: 80% (in CI script)
- Rust: 80% (in Cargo.toml)
```

### 5. Cross-Language Integration Tests

```python
# Python integration test calling Go service
def test_go_service_integration():
    # Start Go service
    process = subprocess.Popen(['./go-service'])

    try:
        # Wait for service to start
        time.sleep(2)

        # Call Go service from Python
        response = requests.get('http://localhost:8080/api/users')
        assert response.status_code == 200

    finally:
        process.terminate()
```

## Language-Specific Performance Tips

### Python
- Use `pytest-xdist` for parallel execution
- Mock external dependencies
- Use `pytest-benchmark` for performance tests

### JavaScript/TypeScript
- Enable `maxWorkers` for Jest
- Use `--runInBand` for debugging
- Mock network calls with MSW

### Rust
- Use `cargo nextest` for faster test execution
- Enable `--release` for benchmarks
- Use `proptest` for property-based testing

### Go
- Use `-short` flag for quick tests
- Enable `-race` detector in CI
- Use `testify` for better assertions

### Java
- Enable parallel execution in JUnit
- Use `@ExtendWith` for custom extensions
- Mock with Mockito for better performance

## Troubleshooting Multi-Language Projects

### Issue: Tests pass locally but fail in CI

**Solution**: Ensure consistent environments

```yaml
# Use Docker for consistent environments
test-backend:
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pytest

test-frontend:
  image: node:20
  script:
    - npm ci
    - npm test
```

### Issue: Slow test execution in monorepo

**Solution**: Run only changed packages

```bash
# Detect changed packages
CHANGED=$(git diff --name-only HEAD~1 | grep -o 'packages/[^/]*' | sort -u)

# Run tests only for changed packages
for pkg in $CHANGED; do
  cd $pkg && run-tests && cd -
done
```

### Issue: Dependency conflicts between languages

**Solution**: Use language version managers

```bash
# .tool-versions (asdf)
python 3.11.0
nodejs 20.0.0
golang 1.21.0
rust 1.70.0
```
