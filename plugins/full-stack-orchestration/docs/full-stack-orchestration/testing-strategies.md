# Testing Strategies

> **Reference**: Comprehensive testing patterns for contract, integration, E2E, security, and performance testing

---

## Contract Testing

### Pattern 1: API Contract Testing with Pact

**Use Case**: Ensure API contracts between frontend and backend remain compatible

**Provider (Backend) Test**:
```python
# tests/contract/test_user_provider.py
from pact import Provider
import pytest

@pytest.fixture
def provider():
    return Provider('UserAPI')

def test_user_api_contract(provider):
    """Verify backend implements the contract"""

    (provider
        .given('A user with ID 550e8400-e29b-41d4-a716-446655440000 exists')
        .upon_receiving('A request for user details')
        .with_request('GET', '/api/users/550e8400-e29b-41d4-a716-446655440000')
        .will_respond_with(200, body={
            'id': '550e8400-e29b-41d4-a716-446655440000',
            'email': 'user@example.com',
            'username': 'johndoe',
            'createdAt': '2024-01-01T00:00:00Z'
        }))

    # Verify against running service
    provider.verify('http://localhost:8000', provider_states_setup_url='http://localhost:8000/_pact/setup')
```

**Consumer (Frontend) Test**:
```typescript
// tests/contract/userApi.test.ts
import { Pact } from '@pact-foundation/pact';
import { getUserById } from '../../api/users';

describe('User API Contract', () => {
  const provider = new Pact({
    consumer: 'WebApp',
    provider: 'UserAPI',
    port: 8080,
  });

  beforeAll(() => provider.setup());
  afterAll(() => provider.finalize());
  afterEach(() => provider.verify());

  test('get user by ID', async () => {
    await provider.addInteraction({
      state: 'A user with ID 550e8400-e29b-41d4-a716-446655440000 exists',
      uponReceiving: 'A request for user details',
      withRequest: {
        method: 'GET',
        path: '/api/users/550e8400-e29b-41d4-a716-446655440000',
        headers: {
          Accept: 'application/json',
        },
      },
      willRespondWith: {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
        },
        body: {
          id: '550e8400-e29b-41d4-a716-446655440000',
          email: 'user@example.com',
          username: 'johndoe',
          createdAt: '2024-01-01T00:00:00Z',
        },
      },
    });

    const user = await getUserById('550e8400-e29b-41d4-a716-446655440000');

    expect(user.email).toBe('user@example.com');
    expect(user.username).toBe('johndoe');
  });
});
```

---

### Pattern 2: OpenAPI Validation with Dredd

**Use Case**: Validate backend API against OpenAPI specification

**Dredd Configuration**:
```yaml
# dredd.yml
reporter: markdown
loglevel: info
only:
  - "Users > Get user by ID"
  - "Users > Create user"
  - "Users > Update user"
  - "Users > Delete user"
hookfiles: ./tests/hooks.js
server: npm start
server-wait: 3
language: nodejs
require: ./tests/setup.js
```

**Dredd Hooks**:
```javascript
// tests/hooks.js
const hooks = require('hooks');
const { createUser, deleteUser } = require('./helpers');

let createdUserId;

hooks.before('Users > Get user by ID', async (transaction) => {
  // Setup: Create test user
  const user = await createUser({
    email: 'test@example.com',
    username: 'testuser',
    password: 'password123'
  });
  createdUserId = user.id;

  // Replace URL parameter with actual ID
  transaction.fullPath = transaction.fullPath.replace('{userId}', createdUserId);
});

hooks.after('Users > Get user by ID', async (transaction) => {
  // Cleanup: Delete test user
  if (createdUserId) {
    await deleteUser(createdUserId);
  }
});
```

---

## Integration Testing

### Pattern 3: Backend Integration Tests

**Use Case**: Test API endpoints with database integration

**FastAPI Integration Test**:
```python
# tests/integration/test_user_endpoints.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db
from app.models.user import User

# Test database setup
SQLALCHEMY_TEST_DATABASE_URL = "postgresql://test:test@localhost/test_db"
engine = create_engine(SQLALCHEMY_TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(test_db):
    return TestClient(app)

class TestUserEndpoints:
    def test_create_user(self, client):
        """Test user creation endpoint"""
        response = client.post(
            "/api/users",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "SecureP@ssw0rd"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["username"] == "newuser"
        assert "id" in data
        assert "password" not in data  # Password should not be returned

    def test_create_user_duplicate_email(self, client):
        """Test duplicate email rejection"""
        user_data = {
            "email": "duplicate@example.com",
            "username": "user1",
            "password": "Password123"
        }

        # First creation should succeed
        response1 = client.post("/api/users", json=user_data)
        assert response1.status_code == 201

        # Second creation with same email should fail
        user_data["username"] = "user2"  # Different username
        response2 = client.post("/api/users", json=user_data)
        assert response2.status_code == 400
        assert "already registered" in response2.json()["detail"].lower()

    def test_get_user_by_id(self, client):
        """Test fetching user by ID"""
        # Create user first
        create_response = client.post(
            "/api/users",
            json={
                "email": "gettest@example.com",
                "username": "gettest",
                "password": "Password123"
            }
        )
        user_id = create_response.json()["id"]

        # Fetch user
        get_response = client.get(f"/api/users/{user_id}")
        assert get_response.status_code == 200
        data = get_response.json()
        assert data["id"] == user_id
        assert data["email"] == "gettest@example.com"

    def test_get_nonexistent_user(self, client):
        """Test 404 for non-existent user"""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/users/{fake_id}")
        assert response.status_code == 404
```

---

## End-to-End Testing

### Pattern 4: E2E Testing with Playwright

**Use Case**: Test complete user journeys across frontend and backend

**Playwright E2E Test**:
```typescript
// tests/e2e/userRegistration.spec.ts
import { test, expect } from '@playwright/test';

test.describe('User Registration Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000');
  });

  test('successful registration creates account and logs in user', async ({ page }) => {
    const timestamp = Date.now();
    const testEmail = `test${timestamp}@example.com`;
    const testUsername = `user${timestamp}`;

    // Navigate to registration page
    await page.click('text=Sign Up');
    await expect(page).toHaveURL(/.*\/register/);

    // Fill registration form
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="username"]', testUsername);
    await page.fill('input[name="password"]', 'SecureP@ssw0rd123');
    await page.fill('input[name="confirmPassword"]', 'SecureP@ssw0rd123');

    // Submit form
    await page.click('button[type="submit"]');

    // Wait for redirect to dashboard
    await expect(page).toHaveURL(/.*\/dashboard/, { timeout: 5000 });

    // Verify user is logged in
    await expect(page.locator('text=Welcome')).toBeVisible();
    await expect(page.locator(`text=${testUsername}`)).toBeVisible();

    // Verify API call succeeded
    const response = await page.waitForResponse(
      (response) => response.url().includes('/api/users/me') && response.status() === 200
    );
    const userData = await response.json();
    expect(userData.email).toBe(testEmail);
    expect(userData.username).toBe(testUsername);
  });

  test('validation prevents invalid registration', async ({ page }) => {
    await page.click('text=Sign Up');

    // Try to submit empty form
    await page.click('button[type="submit"]');

    // Check for validation errors
    await expect(page.locator('text=Email is required')).toBeVisible();
    await expect(page.locator('text=Username is required')).toBeVisible();
    await expect(page.locator('text=Password is required')).toBeVisible();

    // Fill with invalid email
    await page.fill('input[name="email"]', 'invalid-email');
    await page.blur('input[name="email"]');
    await expect(page.locator('text=Invalid email format')).toBeVisible();

    // Fill with weak password
    await page.fill('input[name="password"]', '123');
    await page.blur('input[name="password"]');
    await expect(page.locator('text=Password must be at least 8 characters')).toBeVisible();

    // Mismatched passwords
    await page.fill('input[name="password"]', 'SecureP@ssw0rd123');
    await page.fill('input[name="confirmPassword"]', 'DifferentP@ssw0rd');
    await page.blur('input[name="confirmPassword"]');
    await expect(page.locator('text=Passwords do not match')).toBeVisible();

    // Form should not submit with errors
    await page.click('button[type="submit"]');
    await expect(page).toHaveURL(/.*\/register/);  // Still on registration page
  });

  test('displays server error on duplicate email', async ({ page }) => {
    const existingEmail = 'existing@example.com';

    // Pre-create user via API
    await page.request.post('http://localhost:8000/api/users', {
      data: {
        email: existingEmail,
        username: 'existinguser',
        password: 'Password123'
      }
    });

    // Try to register with same email
    await page.click('text=Sign Up');
    await page.fill('input[name="email"]', existingEmail);
    await page.fill('input[name="username"]', 'newuser');
    await page.fill('input[name="password"]', 'SecureP@ssw0rd123');
    await page.fill('input[name="confirmPassword"]', 'SecureP@ssw0rd123');
    await page.click('button[type="submit"]');

    // Check for server error message
    await expect(page.locator('text=Email already registered')).toBeVisible();
    await expect(page).toHaveURL(/.*\/register/);
  });
});
```

---

## Security Testing

### Pattern 5: OWASP Top 10 Security Tests

**Use Case**: Validate security against common vulnerabilities

**SQL Injection Test**:
```python
# tests/security/test_sql_injection.py
import pytest
from fastapi.testclient import TestClient

def test_sql_injection_in_query_param(client: TestClient):
    """Test that SQL injection attempts are blocked"""
    # Try SQL injection in user ID parameter
    malicious_inputs = [
        "1' OR '1'='1",
        "1'; DROP TABLE users; --",
        "1' UNION SELECT * FROM users --",
        "1' AND 1=1 --"
    ]

    for malicious_input in malicious_inputs:
        response = client.get(f"/api/users/{malicious_input}")

        # Should return 400 (validation error) or 404 (not found), never 200 or 500
        assert response.status_code in [400, 404], \
            f"SQL injection attempt not properly handled: {malicious_input}"

        # Should not leak database error messages
        if response.status_code == 500:
            assert "sql" not in response.text.lower()
            assert "database" not in response.text.lower()
```

**XSS Prevention Test**:
```typescript
// tests/security/xss.spec.ts
test('prevents XSS attacks in user input', async ({ page }) => {
  const xssPayloads = [
    '<script>alert("XSS")</script>',
    '<img src=x onerror="alert(\'XSS\')">',
    '<iframe src="javascript:alert(\'XSS\')">',
    'javascript:alert("XSS")'
  ];

  for (const payload of xssPayloads) {
    // Try to inject XSS in username field
    await page.goto('http://localhost:3000/profile');
    await page.fill('input[name="bio"]', payload);
    await page.click('button[type="submit"]');

    // Wait for save
    await page.waitForTimeout(1000);

    // Reload page and check that script is escaped, not executed
    await page.reload();

    // XSS should be rendered as text, not executed
    const bioContent = await page.locator('[data-testid="user-bio"]').textContent();
    expect(bioContent).toContain(payload);  // Text content should match

    // Script should not have executed (no alert)
    const dialogPromise = page.waitForEvent('dialog', { timeout: 100 }).catch(() => null);
    const dialog = await dialogPromise;
    expect(dialog).toBeNull();  // No alert dialog should appear
  }
});
```

---

## Performance Testing

### Pattern 6: Load Testing with k6

**Use Case**: Validate system performance under load

**k6 Load Test Script**:
```javascript
// tests/performance/load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 200 },  // Ramp up to 200 users
    { duration: '5m', target: 200 },  // Stay at 200 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],  // 95% < 500ms, 99% < 1s
    http_req_failed: ['rate<0.01'],  // Error rate < 1%
    errors: ['rate<0.1'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  // Test 1: Get users list
  let response = http.get(`${BASE_URL}/api/users?page=1&limit=20`);
  check(response, {
    'users list status 200': (r) => r.status === 200,
    'users list duration < 500ms': (r) => r.timings.duration < 500,
  }) || errorRate.add(1);

  sleep(1);

  // Test 2: Get specific user
  const userId = '550e8400-e29b-41d4-a716-446655440000';
  response = http.get(`${BASE_URL}/api/users/${userId}`);
  check(response, {
    'user detail status 200': (r) => r.status === 200,
    'user detail duration < 200ms': (r) => r.timings.duration < 200,
    'user detail has email': (r) => JSON.parse(r.body).email !== undefined,
  }) || errorRate.add(1);

  sleep(1);

  // Test 3: Search users
  response = http.get(`${BASE_URL}/api/users/search?q=john`);
  check(response, {
    'search status 200': (r) => r.status === 200,
    'search duration < 1000ms': (r) => r.timings.duration < 1000,
  }) || errorRate.add(1);

  sleep(2);
}
```

**Run k6 Test**:
```bash
# Install k6
brew install k6  # macOS
# or
apt-get install k6  # Linux

# Run load test
k6 run tests/performance/load-test.js

# Run with custom environment variable
k6 run --env BASE_URL=https://staging.example.com tests/performance/load-test.js

# Run with increased virtual users
k6 run --vus 500 --duration 10m tests/performance/load-test.js
```

---

## Visual Regression Testing

### Pattern 7: Visual Testing with Percy

**Use Case**: Detect unintended UI changes

**Percy Snapshot Test**:
```typescript
// tests/visual/components.spec.ts
import { test } from '@playwright/test';
import percySnapshot from '@percy/playwright';

test.describe('Visual Regression Tests', () => {
  test('button components', async ({ page }) => {
    await page.goto('http://localhost:6006/iframe.html?id=button--primary');
    await percySnapshot(page, 'Button - Primary');

    await page.goto('http://localhost:6006/iframe.html?id=button--secondary');
    await percySnapshot(page, 'Button - Secondary');

    await page.goto('http://localhost:6006/iframe.html?id=button--disabled');
    await percySnapshot(page, 'Button - Disabled');
  });

  test('user profile page - responsive', async ({ page }) => {
    await page.goto('http://localhost:3000/profile/testuser');
    await page.waitForLoadState('networkidle');

    // Desktop view
    await page.setViewportSize({ width: 1280, height: 720 });
    await percySnapshot(page, 'Profile Page - Desktop');

    // Tablet view
    await page.setViewportSize({ width: 768, height: 1024 });
    await percySnapshot(page, 'Profile Page - Tablet');

    // Mobile view
    await page.setViewportSize({ width: 375, height: 667 });
    await percySnapshot(page, 'Profile Page - Mobile');
  });
});
```

---

## Test Coverage Targets

| Test Type | Coverage Target | Purpose |
|-----------|----------------|---------|
| **Unit Tests** | 80-90% | Individual function/method correctness |
| **Integration Tests** | 70-80% | API endpoint and service integration |
| **Contract Tests** | 100% | All API contracts validated |
| **E2E Tests** | Critical paths | User journey completeness |
| **Security Tests** | OWASP Top 10 | Vulnerability prevention |
| **Performance Tests** | SLO compliance | Scalability and responsiveness |
| **Visual Tests** | Key components | UI consistency |

---

## CI/CD Integration

### GitHub Actions Test Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main, develop]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run unit tests
        run: pytest tests/unit --cov=app --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: pytest tests/integration

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install Playwright
        run: npx playwright install --with-deps
      - name: Run E2E tests
        run: npm run test:e2e
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: playwright-report
          path: playwright-report/

  security-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run OWASP ZAP scan
        uses: zaproxy/action-baseline@v0.7.0
        with:
          target: 'http://localhost:8000'
```
