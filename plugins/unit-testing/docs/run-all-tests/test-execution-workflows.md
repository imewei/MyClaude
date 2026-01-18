# Test Execution Workflows

Comprehensive guide for iterative execution patterns, parallel test strategies, coverage integration, CI/CD workflows, and performance optimization for test suites.

## Iterative Execution Patterns

### Sequential Fix-and-Retry Pattern

```python
def iterative_test_execution(
    max_iterations: int = 10,
    auto_fix: bool = False
) -> Dict[str, Any]:
    """
    Execute tests iteratively, fixing failures until all pass
    """
    iteration = 0
    previous_failures = float('inf')

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{max_iterations}")
        print(f"{'='*60}\n")

        # Run tests
        result = run_tests()

        # Check if all tests pass
        if result['failures'] == 0:
            print(" SUCCESS: All tests passing!")
            return {
                'status': 'success',
                'iterations': iteration,
                'final_state': result
            }

        # Check progress
        if result['failures'] >= previous_failures:
            print("⚠️ No progress or regression detected")
            if result['failures'] > previous_failures:
                print("❌ More failures than before, rolling back")
                return {
                    'status': 'regression',
                    'iterations': iteration
                }
            else:
                print(" Plateau detected, need deeper analysis")
                return {
                    'status': 'plateau',
                    'iterations': iteration,
                    'remaining_failures': result['failures']
                }

        # Analyze and fix
        if auto_fix:
            fixes_applied = analyze_and_fix(result)
            print(f" Applied {fixes_applied} fixes")

        previous_failures = result['failures']

    return {
        'status': 'max_iterations_reached',
        'iterations': iteration,
        'remaining_failures': previous_failures
    }
```

### Test Run Phases

**Phase 1: Discovery**
```python
def discover_tests():
    """Discover all test files and count tests"""
    frameworks = detect_frameworks()

    test_inventory = {}

    for framework in frameworks:
        test_files = glob(framework['patterns'])
        test_count = count_tests_in_files(test_files, framework)

        test_inventory[framework['name']] = {
            'files': test_files,
            'count': test_count,
            'command': framework['command']
        }

    return test_inventory
```

**Phase 2: Execution**
```python
def execute_tests(config: Dict) -> TestResult:
    """Execute tests with appropriate configuration"""

    # Build command
    cmd = build_test_command(
        framework=config['framework'],
        parallel=config.get('parallel', False),
        coverage=config.get('coverage', False),
        pattern=config.get('pattern', None)
    )

    # Execute
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=config.get('timeout', 3600)
    )
    duration = time.time() - start_time

    # Parse output
    parsed = parse_test_output(
        result.stdout,
        result.stderr,
        config['framework']
    )

    return TestResult(
        passed=parsed['passed'],
        failed=parsed['failed'],
        skipped=parsed['skipped'],
        duration=duration,
        output=result.stdout,
        exit_code=result.returncode
    )
```

**Phase 3: Analysis**
```python
def analyze_results(result: TestResult) -> Analysis:
    """Analyze test results and categorize failures"""

    failures = parse_failures(result.output)

    categorized = {
        'import_errors': [],
        'assertion_failures': [],
        'runtime_errors': [],
        'timeout_errors': [],
        'setup_errors': []
    }

    for failure in failures:
        category = categorize_failure(failure)
        categorized[category].append(failure)

    return Analysis(
        categories=categorized,
        total_failures=len(failures),
        suggested_fixes=generate_fix_suggestions(categorized)
    )
```

**Phase 4: Fixing**
```python
def apply_fixes(analysis: Analysis, auto_fix: bool = False) -> int:
    """Apply fixes based on analysis"""

    fixes_applied = 0

    for category, failures in analysis.categories.items():
        for failure in failures:
            fix_strategy = get_fix_strategy(category, failure)

            if auto_fix:
                success = apply_fix(fix_strategy)
                if success:
                    fixes_applied += 1
            else:
                print_fix_suggestion(fix_strategy)

    return fixes_applied
```

## Parallel Test Strategies

### Thread-Based Parallelization

```python
# pytest with xdist
pytest -n auto  # Auto-detect CPU count
pytest -n 4     # Use 4 workers

# Configuration in pytest.ini
[pytest]
addopts = -n auto
```

### Process-Based Parallelization

```javascript
// Jest parallel execution
{
  "jest": {
    "maxWorkers": "50%",  // Use 50% of CPU cores
    "testTimeout": 10000
  }
}

// Run tests
npm test -- --maxWorkers=4
```

### Test Sharding

```bash
# Playwright sharding
npx playwright test --shard=1/4  # Run 1st quarter
npx playwright test --shard=2/4  # Run 2nd quarter
npx playwright test --shard=3/4  # Run 3rd quarter
npx playwright test --shard=4/4  # Run 4th quarter

# GitHub Actions matrix
jobs:
  test:
    strategy:
      matrix:
        shard: [1, 2, 3, 4]
    steps:
      - run: npx playwright test --shard=${{ matrix.shard }}/4
```

### Parallel Execution Best Practices

```python
# 1. Isolate test data
@pytest.fixture
def user_data():
    """Each test gets unique data"""
    return {
        'email': f'test-{uuid.uuid4()}@example.com',
        'username': f'user-{uuid.uuid4()}'
    }

# 2. Use database transactions
@pytest.fixture
def db_session():
    """Rollback changes after test"""
    session = create_session()
    yield session
    session.rollback()
    session.close()

# 3. Avoid shared state
# ❌ Bad: Global state
global_cache = {}

def test_cache():
    global_cache['key'] = 'value'  # Affects other tests!

# ✅ Good: Isolated state
@pytest.fixture
def cache():
    return {}

def test_cache(cache):
    cache['key'] = 'value'  # Only affects this test
```

## Coverage Integration

### Python Coverage with pytest

```bash
# Install coverage
uv uv pip install pytest-cov

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Configuration in pyproject.toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__init__.py"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

### JavaScript Coverage with Jest

```javascript
// jest.config.js
module.exports = {
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  collectCoverageFrom: [
    'src/**/*.{js,ts}',
    '!src/**/*.test.{js,ts}',
    '!src/**/*.d.ts',
    '!src/**/index.ts'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  }
};
```

### Coverage Analysis Workflow

```python
def analyze_coverage_gaps(coverage_file: str) -> List[Dict]:
    """Identify uncovered code and generate tests"""

    coverage_data = read_coverage_report(coverage_file)

    gaps = []

    for file_path, data in coverage_data['files'].items():
        missing_lines = data.get('missing_lines', [])

        if missing_lines:
            # Read source code
            source = read_file(file_path)
            ast_tree = parse_ast(source)

            # Find uncovered functions
            uncovered_functions = find_functions_with_missing_lines(
                ast_tree,
                missing_lines
            )

            for func in uncovered_functions:
                gaps.append({
                    'file': file_path,
                    'function': func['name'],
                    'lines': func['missing_lines'],
                    'complexity': func['complexity'],
                    'priority': calculate_priority(func)
                })

    # Sort by priority
    gaps.sort(key=lambda x: x['priority'], reverse=True)

    return gaps
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [18, 20]
        shard: [1, 2, 3, 4]

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run tests (Shard ${{ matrix.shard }}/4)
        run: npm test -- --shard=${{ matrix.shard }}/4 --coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/coverage-final.json
          flags: shard-${{ matrix.shard }}

      - name: Upload test results
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-results-${{ matrix.shard }}
          path: test-results/

  coverage-report:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Download coverage reports
        uses: actions/download-artifact@v3

      - name: Merge coverage reports
        run: npx nyc merge coverage/ merged-coverage.json

      - name: Generate final report
        run: npx nyc report --reporter=lcov --reporter=text

      - name: Check coverage threshold
        run: |
          COVERAGE=$(npx nyc report --reporter=text-summary | grep "Lines" | awk '{print $3}' | sed 's/%//')
          if (( $(echo "$COVERAGE < 80" | bc -l) )); then
            echo "Coverage $COVERAGE% is below threshold 80%"
            exit 1
          fi
```

### GitLab CI Configuration

```yaml
# .gitlab-ci.yml
stages:
  - test
  - coverage

variables:
  POSTGRES_DB: test_db
  POSTGRES_USER: test_user
  POSTGRES_PASSWORD: test_pass

test:
  stage: test
  image: python:3.11
  services:
    - postgres:15
  parallel:
    matrix:
      - SHARD: [1, 2, 3, 4]
  script:
    - uv uv pip install -r requirements-dev.txt
    - pytest --shard=${SHARD}/4 --cov=src --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - .coverage
      - htmlcov/
    expire_in: 1 week

merge_coverage:
  stage: coverage
  script:
    - coverage combine
    - coverage report
    - coverage html
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    paths:
      - htmlcov/
```

### CircleCI Configuration

```yaml
# .circleci/config.yml
version: 2.1

executors:
  node-executor:
    docker:
      - image: cimg/node:20.0
    working_directory: ~/project

jobs:
  test:
    executor: node-executor
    parallelism: 4
    steps:
      - checkout
      - restore_cache:
          keys:
            - deps-{{ checksum "package-lock.json" }}
      - run: npm ci
      - save_cache:
          key: deps-{{ checksum "package-lock.json" }}
          paths:
            - node_modules
      - run:
          name: Run tests
          command: |
            TESTFILES=$(circleci tests glob "**/*.test.js" | circleci tests split --split-by=timings)
            npm test -- $TESTFILES --coverage
      - store_test_results:
          path: test-results
      - store_artifacts:
          path: coverage

workflows:
  test-workflow:
    jobs:
      - test
```

## Exit Criteria and Validation

### Success Criteria

```python
class ExitCriteria:
    """Define when test execution should stop"""

    def __init__(self, config: Dict):
        self.min_pass_rate = config.get('min_pass_rate', 1.0)  # 100%
        self.min_coverage = config.get('min_coverage', 0.8)  # 80%
        self.max_iterations = config.get('max_iterations', 10)
        self.allow_flaky = config.get('allow_flaky', False)

    def should_continue(
        self,
        iteration: int,
        result: TestResult
    ) -> Tuple[bool, str]:
        """Determine if execution should continue"""

        # Check max iterations
        if iteration >= self.max_iterations:
            return False, "Max iterations reached"

        # Check pass rate
        pass_rate = result.passed / (result.passed + result.failed)
        if pass_rate >= self.min_pass_rate:
            return False, "All tests passing"

        # Check for regression
        if hasattr(self, 'previous_failures'):
            if result.failed > self.previous_failures:
                return False, "Regression detected"

        # Check for plateau
        if hasattr(self, 'plateau_count'):
            if self.plateau_count >= 2:
                return False, "Plateau detected"

        self.previous_failures = result.failed

        return True, "Continue execution"
```

### Quality Gates

```python
def validate_quality_gates(result: TestResult, coverage: float) -> bool:
    """Validate all quality gates pass"""

    gates = {
        'test_pass_rate': {
            'actual': result.passed / (result.passed + result.failed),
            'threshold': 1.0,
            'required': True
        },
        'code_coverage': {
            'actual': coverage,
            'threshold': 0.80,
            'required': True
        },
        'no_skipped_tests': {
            'actual': result.skipped,
            'threshold': 0,
            'required': False
        },
        'max_duration': {
            'actual': result.duration,
            'threshold': 300,  # 5 minutes
            'required': False
        }
    }

    failed_gates = []

    for gate_name, gate_config in gates.items():
        if gate_config['required']:
            if gate_name == 'no_skipped_tests':
                passed = gate_config['actual'] <= gate_config['threshold']
            elif gate_name == 'max_duration':
                passed = gate_config['actual'] <= gate_config['threshold']
            else:
                passed = gate_config['actual'] >= gate_config['threshold']

            if not passed:
                failed_gates.append({
                    'gate': gate_name,
                    'expected': gate_config['threshold'],
                    'actual': gate_config['actual']
                })

    if failed_gates:
        print("❌ Quality gates failed:")
        for gate in failed_gates:
            print(f"  - {gate['gate']}: {gate['actual']} (required: {gate['expected']})")
        return False

    print(" All quality gates passed")
    return True
```

## Performance Optimization

### Test Execution Optimization

```python
# 1. Run fast tests first
pytest --durations=0  # Show test durations
pytest --durations=10  # Show 10 slowest tests

# Organize tests by speed
@pytest.mark.fast
def test_quick_validation():
    pass

@pytest.mark.slow
def test_complex_integration():
    pass

# Run fast tests first
pytest -m fast
pytest -m "not slow"
```

### Caching Strategies

```python
# Cache test fixtures
@pytest.fixture(scope="session")
def expensive_resource():
    """Created once per test session"""
    resource = create_expensive_resource()
    yield resource
    cleanup(resource)

# Cache test data
@pytest.fixture(scope="module")
def sample_data():
    """Created once per test module"""
    return load_sample_data()
```

### Mocking External Services

```python
# Mock slow external API calls
@pytest.fixture
def mock_api(monkeypatch):
    """Mock external API for faster tests"""

    def mock_get_user(user_id):
        return {'id': user_id, 'name': 'Mock User'}

    monkeypatch.setattr('api_client.get_user', mock_get_user)
```

### Database Optimization

```python
# Use in-memory database for tests
import pytest
from sqlalchemy import create_engine

@pytest.fixture(scope="session")
def db_engine():
    """Use SQLite in-memory for fast tests"""
    return create_engine('sqlite:///:memory:')

# Use transactions for isolation
@pytest.fixture
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()
```

## Test Reporting

### HTML Reports

```bash
# pytest-html
pytest --html=report.html --self-contained-html

# Jest HTML reporter
npm test -- --reporters=jest-html-reporter
```

### JUnit XML for CI

```bash
# pytest
pytest --junitxml=results.xml

# Jest
npm test -- --reporters=jest-junit

# Go
go test -v 2>&1 ./... | go-junit-report > results.xml
```

### Custom Reporting

```python
class CustomTestReporter:
    """Generate custom test reports"""

    def __init__(self):
        self.results = []

    def report_test(self, test_name: str, result: str, duration: float):
        self.results.append({
            'test': test_name,
            'result': result,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })

    def generate_report(self, output_file: str):
        report = {
            'summary': {
                'total': len(self.results),
                'passed': sum(1 for r in self.results if r['result'] == 'passed'),
                'failed': sum(1 for r in self.results if r['result'] == 'failed'),
                'total_duration': sum(r['duration'] for r in self.results)
            },
            'tests': self.results
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
```

## Best Practices

1. **Run tests in parallel** for faster feedback
2. **Use test sharding** for large test suites
3. **Cache dependencies** in CI/CD
4. **Mock external services** for speed and reliability
5. **Set appropriate timeouts** for different test types
6. **Track test execution time** and optimize slow tests
7. **Use incremental testing** (only run affected tests)
8. **Implement quality gates** before deployment
9. **Generate coverage reports** and track trends
10. **Archive test results** for historical analysis
