# Debugging Strategies for Test Failures

Comprehensive guide for AI-driven root cause analysis, log correlation, flaky test detection, and systematic debugging approaches for test failures across frameworks.

## Overview

This guide provides systematic strategies for debugging test failures, from common patterns to advanced techniques using AI-driven analysis, distributed tracing, and automated root cause detection.

## AI-Driven Root Cause Analysis

### Ultra-Think Framework for Test Debugging

**Step 1: Problem Space Understanding**

```
Current State:
- Failed test: test_user_login
- Failure type: AssertionError
- Frequency: Intermittent (70% pass rate)
- Environment: CI only, passes locally
- Recent changes: Database connection pooling added
```

**Step 2: Hypothesis Generation**

```python
hypotheses = [
    {
        "hypothesis": "Race condition in connection pool",
        "confidence": 0.8,
        "evidence": [
            "Passes locally (single-threaded)",
            "Fails in CI (parallel execution)",
            "Recent connection pool changes"
        ],
        "test_approach": "Add explicit connection cleanup"
    },
    {
        "hypothesis": "Test data collision",
        "confidence": 0.6,
        "evidence": [
            "Intermittent failure",
            "Multiple tests create users"
        ],
        "test_approach": "Use unique test data per test"
    },
    {
        "hypothesis": "Timing issue",
        "confidence": 0.5,
        "evidence": [
            "Async operation in login flow",
            "No explicit waits"
        ],
        "test_approach": "Add proper async waits"
    }
]
```

**Step 3: Hypothesis Testing**

```bash
# Test Hypothesis 1: Race condition
# Add connection cleanup
pytest test_user_login.py -n 4 --count=10
# Result: 100% pass → Hypothesis confirmed!

# Test Hypothesis 2: Data collision
# Use unique usernames
pytest test_user_login.py --count=20
# Result: Still fails → Hypothesis rejected

# Test Hypothesis 3: Timing
# Add explicit waits
pytest test_user_login.py
# Result: Partial improvement → Contributing factor
```

**Step 4: Root Cause Identification**

```
Root Cause: Connection pool exhaustion
- Connection pool size: 5
- Parallel tests: 8
- Not releasing connections properly in teardown
- Race condition when pool is full

Fix: Increase pool size + Add proper cleanup
```

### AI Pattern Recognition

```python
class TestFailureAnalyzer:
    """
    AI-powered test failure analyzer using pattern matching
    """

    def analyze_failure(self, test_output: str) -> Dict[str, Any]:
        """Analyze test failure and suggest fixes"""

        # Extract failure information
        failure_info = self.extract_failure_info(test_output)

        # Pattern matching
        patterns = {
            'null_pointer': {
                'pattern': r'(NoneType|null|undefined).*has no attribute',
                'category': 'RuntimeError',
                'common_causes': [
                    'Missing mock return value',
                    'Uninitialized variable',
                    'API returned None unexpectedly'
                ],
                'suggested_fixes': [
                    'Add null check',
                    'Verify mock setup',
                    'Check API response handling'
                ]
            },
            'assertion_failure': {
                'pattern': r'AssertionError.*Expected:.*Received:',
                'category': 'AssertionFailure',
                'common_causes': [
                    'Implementation changed',
                    'Test expectation outdated',
                    'Data setup incorrect'
                ],
                'suggested_fixes': [
                    'Review recent code changes',
                    'Update test expectation',
                    'Verify test data setup'
                ]
            },
            'timeout': {
                'pattern': r'Timeout|exceeded.*ms',
                'category': 'Async/Timing',
                'common_causes': [
                    'Missing await',
                    'Slow operation',
                    'Deadlock'
                ],
                'suggested_fixes': [
                    'Add await keyword',
                    'Increase timeout',
                    'Check for deadlocks'
                ]
            },
            'import_error': {
                'pattern': r'ImportError|ModuleNotFoundError|Cannot find module',
                'category': 'ImportError',
                'common_causes': [
                    'Missing dependency',
                    'Wrong import path',
                    'Circular dependency'
                ],
                'suggested_fixes': [
                    'Install missing package',
                    'Fix import path',
                    'Resolve circular imports'
                ]
            }
        }

        # Match patterns
        for name, pattern_info in patterns.items():
            if re.search(pattern_info['pattern'], test_output):
                return {
                    'pattern': name,
                    'category': pattern_info['category'],
                    'causes': pattern_info['common_causes'],
                    'fixes': pattern_info['suggested_fixes'],
                    'confidence': self.calculate_confidence(test_output, pattern_info)
                }

        return {'pattern': 'unknown', 'confidence': 0.0}

    def calculate_confidence(self, output: str, pattern: Dict) -> float:
        """Calculate confidence score for pattern match"""
        score = 0.5  # Base confidence

        # Increase confidence if multiple indicators present
        if 'Expected' in output and 'Received' in output:
            score += 0.2

        if 'at line' in output or 'traceback' in output.lower():
            score += 0.1

        return min(score, 1.0)
```

## Log Correlation Techniques

### Cross-Service Log Correlation

```python
import re
from typing import List, Dict
from datetime import datetime

class LogCorrelator:
    """Correlate logs across services to find root cause"""

    def correlate_logs(
        self,
        test_failure_time: datetime,
        log_sources: List[str]
    ) -> Dict[str, List[str]]:
        """
        Correlate logs from multiple sources around failure time
        """
        window = 30  # seconds before/after failure

        correlated = {}

        for source in log_sources:
            logs = self.read_logs(source, test_failure_time, window)
            errors = self.extract_errors(logs)
            warnings = self.extract_warnings(logs)

            correlated[source] = {
                'errors': errors,
                'warnings': warnings,
                'timeline': self.build_timeline(logs)
            }

        # Find correlation patterns
        correlations = self.find_patterns(correlated)

        return correlations

    def extract_errors(self, logs: List[str]) -> List[Dict]:
        """Extract error messages from logs"""
        errors = []

        error_patterns = [
            r'ERROR:.*',
            r'Exception:.*',
            r'FATAL:.*',
            r'CRITICAL:.*'
        ]

        for log in logs:
            for pattern in error_patterns:
                match = re.search(pattern, log)
                if match:
                    errors.append({
                        'message': match.group(0),
                        'timestamp': self.extract_timestamp(log),
                        'context': self.extract_context(log)
                    })

        return errors

    def find_patterns(self, correlated: Dict) -> List[Dict]:
        """Find correlation patterns across services"""
        patterns = []

        # Example: Database connection errors followed by API errors
        for service1, data1 in correlated.items():
            for service2, data2 in correlated.items():
                if service1 != service2:
                    # Check if errors in service1 preceded service2
                    correlation = self.check_causation(
                        data1['errors'],
                        data2['errors']
                    )
                    if correlation:
                        patterns.append({
                            'source': service1,
                            'affected': service2,
                            'correlation': correlation
                        })

        return patterns
```

### Example Log Analysis

```python
# Example: Analyzing test failure logs

test_output = """
FAILED tests/test_api.py::test_user_creation - AssertionError

app.log:
2024-01-07 10:23:45 ERROR: Database connection failed
2024-01-07 10:23:45 ERROR: Connection pool exhausted
2024-01-07 10:23:46 ERROR: User creation failed: No database connection

test.log:
2024-01-07 10:23:46 FAIL: test_user_creation
Expected: User created successfully
Received: Database error
"""

analyzer = LogCorrelator()
analysis = analyzer.correlate_logs(
    test_failure_time=datetime(2024, 1, 7, 10, 23, 46),
    log_sources=['app.log', 'test.log']
)

# Output:
# {
#   'root_cause': 'Connection pool exhaustion',
#   'timeline': [
#     '10:23:45: Database connection failed (app)',
#     '10:23:45: Connection pool exhausted (app)',
#     '10:23:46: Test failed (test)'
#   ],
#   'recommendation': 'Increase connection pool size or fix connection leaks'
# }
```

## Flaky Test Detection

### Statistical Analysis

```python
class FlakeDetector:
    """Detect and analyze flaky tests"""

    def detect_flaky_tests(
        self,
        test_history: List[Dict],
        threshold: float = 0.95
    ) -> List[Dict]:
        """
        Detect flaky tests based on historical pass/fail rates
        """
        test_stats = {}

        # Calculate pass rate for each test
        for result in test_history:
            test_name = result['test_name']

            if test_name not in test_stats:
                test_stats[test_name] = {
                    'passes': 0,
                    'failures': 0,
                    'errors': []
                }

            if result['status'] == 'passed':
                test_stats[test_name]['passes'] += 1
            else:
                test_stats[test_name]['failures'] += 1
                test_stats[test_name]['errors'].append(result['error'])

        # Identify flaky tests
        flaky_tests = []

        for test_name, stats in test_stats.items():
            total = stats['passes'] + stats['failures']
            pass_rate = stats['passes'] / total

            # Flaky if: 0 < pass_rate < threshold
            if 0 < pass_rate < threshold:
                flaky_tests.append({
                    'test': test_name,
                    'pass_rate': pass_rate,
                    'total_runs': total,
                    'failures': stats['failures'],
                    'error_patterns': self.analyze_error_patterns(
                        stats['errors']
                    )
                })

        return sorted(flaky_tests, key=lambda x: x['pass_rate'])

    def analyze_error_patterns(self, errors: List[str]) -> Dict:
        """Find common patterns in flaky test errors"""
        patterns = {
            'timeout': 0,
            'race_condition': 0,
            'external_dependency': 0,
            'data_setup': 0
        }

        for error in errors:
            if 'timeout' in error.lower():
                patterns['timeout'] += 1
            if 'race' in error.lower() or 'concurrent' in error.lower():
                patterns['race_condition'] += 1
            if 'connection' in error.lower() or 'network' in error.lower():
                patterns['external_dependency'] += 1
            if 'already exists' in error.lower() or 'not found' in error.lower():
                patterns['data_setup'] += 1

        return {k: v for k, v in patterns.items() if v > 0}
```

### Flake Resolution Strategies

```python
flake_fixes = {
    'timeout': {
        'diagnosis': 'Test exceeds time limit intermittently',
        'causes': [
            'External API call without timeout',
            'Slow database query',
            'Missing await on async operation'
        ],
        'fixes': [
            'Add explicit timeout to external calls',
            'Mock external dependencies',
            'Add await keywords',
            'Increase test timeout as last resort'
        ]
    },
    'race_condition': {
        'diagnosis': 'Test depends on execution order or timing',
        'causes': [
            'Parallel test execution interference',
            'Shared state between tests',
            'Missing synchronization'
        ],
        'fixes': [
            'Isolate test data (unique IDs)',
            'Add proper cleanup in teardown',
            'Use transactions that rollback',
            'Disable parallel execution for specific tests'
        ]
    },
    'external_dependency': {
        'diagnosis': 'Test depends on external service',
        'causes': [
            'Network connectivity issues',
            'Third-party API down',
            'Database connection instability'
        ],
        'fixes': [
            'Mock external services',
            'Add retry logic with exponential backoff',
            'Use test containers for databases',
            'Implement circuit breaker pattern'
        ]
    },
    'data_setup': {
        'diagnosis': 'Test data conflicts or missing',
        'causes': [
            'Unique constraint violations',
            'Test data not cleaned up',
            'Race condition in data creation'
        ],
        'fixes': [
            'Use UUID/timestamp in test data',
            'Implement proper cleanup',
            'Use database transactions',
            'Isolate test database per test'
        ]
    }
}
```

## Common Failure Patterns

### Pattern 1: Null Pointer / Undefined Errors

**Symptoms**:
```
TypeError: Cannot read property 'name' of undefined
AttributeError: 'NoneType' object has no attribute 'email'
NullPointerException
```

**Debugging Steps**:

1. **Identify the null value source**:
```python
# Add debug logging
print(f"User object: {user}")
print(f"User type: {type(user)}")
print(f"User attributes: {dir(user) if user else 'None'}")
```

2. **Check mock setup**:
```javascript
// Missing return value
mockGetUser.mockReturnValue();  // ❌ Returns undefined

// Fixed
mockGetUser.mockReturnValue({ name: 'John', email: 'john@example.com' });  // ✅
```

3. **Verify API response**:
```python
# Check actual response
response = api_client.get_user(user_id)
assert response is not None, "API returned None"
assert hasattr(response, 'name'), "Response missing 'name' attribute"
```

**Common Fixes**:
```python
# Add null checks
user = get_user(user_id)
if user is None:
    raise ValueError(f"User {user_id} not found")

# Use optional chaining (JavaScript/TypeScript)
const name = user?.name ?? 'Unknown';

# Python with defaults
name = getattr(user, 'name', 'Unknown')
```

### Pattern 2: Assertion Failures

**Symptoms**:
```
Expected: 10, Received: 11
AssertionError: assert False
Expected array length: 5, Received: 3
```

**Debugging Steps**:

1. **Understand what changed**:
```bash
# Git blame to see recent changes
git blame path/to/test/file.test.js

# Find related commits
git log --oneline --grep="user" --since="1 week ago"

# See implementation changes
git diff HEAD~5 -- src/user.js
```

2. **Determine if test or code is wrong**:
```python
# Add detailed assertion messages
assert result == expected, \
    f"Result mismatch:\n" \
    f"Expected: {expected}\n" \
    f"Received: {result}\n" \
    f"Diff: {set(expected) - set(result)}"

# Check test intent
# Read test description and docstring
# Review original requirement/ticket
```

3. **Analyze the discrepancy**:
```javascript
// Expected vs Actual analysis
console.log('Expected:', JSON.stringify(expected, null, 2));
console.log('Actual:', JSON.stringify(actual, null, 2));

// Deep equality check
expect(actual).toEqual(expected);  // Deep comparison
```

**Common Fixes**:

```python
# Fix 1: Update test expectation (if implementation is correct)
# Before
assert total == 100

# After (tax was added to implementation)
assert total == 110  # 100 + 10% tax

# Fix 2: Fix implementation (if test is correct)
# Before
def calculate_total(items):
    return sum(item.price for item in items)

# After
def calculate_total(items):
    subtotal = sum(item.price for item in items)
    return subtotal * 1.1  # Add tax

# Fix 3: Update mock (if test setup is wrong)
# Before
mock_api.return_value = {'count': 10}

# After
mock_api.return_value = {'count': 10, 'total': 100}
```

### Pattern 3: Async/Timing Issues

**Symptoms**:
```
Timeout: Async callback was not invoked within 5000ms
UnhandledPromiseRejectionWarning
Error: Cannot perform action on unmounted component
```

**Debugging Steps**:

1. **Identify async operations**:
```javascript
// Find all async operations in test
const promises = [
    fetchUser(),  // ← Missing await?
    updateProfile(),  // ← Missing await?
    saveData()  // ← Missing await?
];
```

2. **Add proper waits**:
```javascript
// ❌ Bad: No wait
test('user loads', () => {
    const user = fetchUser(123);  // Returns Promise
    expect(user.name).toBe('John');  // Fails: user is Promise, not object
});

// ✅ Good: Await promise
test('user loads', async () => {
    const user = await fetchUser(123);
    expect(user.name).toBe('John');
});
```

3. **Check for race conditions**:
```python
# Add locks or synchronization
import threading

lock = threading.Lock()

with lock:
    # Critical section
    user = create_user()
    assign_role(user)
```

**Common Fixes**:

```javascript
// Fix 1: Add missing async/await
// Before
test('loads data', () => {
    const data = loadData();
    expect(data).toBeDefined();
});

// After
test('loads data', async () => {
    const data = await loadData();
    expect(data).toBeDefined();
});

// Fix 2: Use waitFor for UI updates
// Before
fireEvent.click(button);
expect(screen.getByText('Success')).toBeInTheDocument();

// After
fireEvent.click(button);
await waitFor(() => {
    expect(screen.getByText('Success')).toBeInTheDocument();
});

// Fix 3: Increase timeout for slow operations
// Before
test('slow operation', async () => {
    await slowFunction();
});

// After
test('slow operation', async () => {
    await slowFunction();
}, 10000);  // 10 second timeout
```

### Pattern 4: Import/Module Errors

**Symptoms**:
```
Cannot find module '../utils/helper'
ModuleNotFoundError: No module named 'mypackage'
ImportError: cannot import name 'calculate' from 'utils'
```

**Debugging Steps**:

1. **Verify file exists**:
```bash
# Check file location
find . -name "helper.js"
find . -name "helper.ts"

# List directory structure
tree src/
```

2. **Check import paths**:
```javascript
// Wrong: Incorrect relative path
import { helper } from '../utils/helper';  // From wrong directory

// Right: Correct path
import { helper } from '../../utils/helper';

// Better: Use absolute imports with path alias
import { helper } from '@/utils/helper';
```

3. **Verify package installation**:
```bash
# Check if package is installed
npm list package-name
pip show package-name

# Check package.json / requirements.txt
cat package.json | grep package-name
cat requirements.txt | grep package-name
```

**Common Fixes**:

```bash
# Fix 1: Install missing package
npm install --save-dev missing-package
uv uv pip install missing-package

# Fix 2: Fix import path
# Update relative path or configure path aliases

# tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}

# Fix 3: Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or in pytest.ini
[pytest]
pythonpath = src
```

## Binary Search Debugging

### Technique: Bisect Test Runs

```bash
# Find which commit introduced test failure
git bisect start
git bisect bad  # Current commit (tests fail)
git bisect good abc123  # Known good commit

# Git will checkout commits to test
# Run tests at each commit
npm test || pytest

# Mark as good or bad
git bisect good  # If tests pass
git bisect bad   # If tests fail

# Continue until finding exact commit
```

### Technique: Isolate Failing Test

```python
# Run single test
pytest tests/test_user.py::test_create_user -v

# Run with specific markers
pytest -m "not slow" -v

# Disable parallel execution
pytest -n 0

# Run in different order
pytest --randomly-seed=12345
```

## Distributed Tracing Integration

### Using OpenTelemetry in Tests

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

def test_user_creation_with_tracing():
    with tracer.start_as_current_span("test_user_creation"):
        with tracer.start_as_current_span("database_setup"):
            setup_database()

        with tracer.start_as_current_span("create_user"):
            user = create_user("John", "john@example.com")

        with tracer.start_as_current_span("verify_user"):
            assert user.name == "John"

# Trace output shows exact timing and operation flow
```

## Debugging Tools and Techniques

### pytest Debugging

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at start of test
pytest --trace

# Show local variables on failure
pytest -l

# Show full diff for assertions
pytest -vv

# Capture output
pytest -s  # Don't capture stdout

# Run last failed tests
pytest --lf

# Run failed first, then others
pytest --ff
```

### JavaScript/Jest Debugging

```javascript
// Debug single test
test.only('specific test', () => {
    debugger;  // Breakpoint
    const result = myFunction();
    expect(result).toBe(expected);
});

// Run with debugger
node --inspect-brk node_modules/.bin/jest --runInBand

// Show detailed error
npm test -- --verbose

// No coverage (faster)
npm test -- --coverage=false
```

### Go Testing Debugging

```bash
# Verbose output
go test -v ./...

# Run specific test
go test -run TestUserCreation

# With race detector
go test -race ./...

# Print test output
go test -v -args -test.v

# Coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

## Best Practices

1. **Always add reproduction steps** to bug reports
2. **Use detailed assertion messages** for faster debugging
3. **Enable debug logging** in tests when investigating
4. **Isolate tests** to avoid interference
5. **Use deterministic test data** (avoid random values)
6. **Run tests in same environment as CI**
7. **Track flaky tests** and fix them promptly
8. **Use tracing** for complex integration tests
9. **Keep test history** for pattern analysis
10. **Document known flaky tests and workarounds**
