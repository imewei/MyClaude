---
version: 1.0.3
command: /run-all-tests
description: Iteratively run and fix all tests until zero failures with AI-driven root cause analysis across 3 execution modes
argument-hint: [test-path] [--fix] [--max-iterations=10] [--parallel] [--coverage]
execution_modes:
  quick:
    duration: "30min-1h"
    description: "Fast iteration for single test file or small suite"
    agents: ["test-automator"]
    scope: "Single test file or small test suite"
    iterations: "Max 3 iterations"
    debugging: "Basic error analysis"
    use_case: "Fixing specific test failures, rapid debugging"
  standard:
    duration: "2-4h"
    description: "Comprehensive iteration for full test suite"
    agents: ["test-automator", "debugger"]
    scope: "Full test suite for feature/module"
    iterations: "Max 10 iterations (default)"
    debugging: "AI-assisted RCA with debugger agent"
    coverage: "Enabled, target >80%"
    parallel: "Enabled"
    use_case: "Pre-commit validation, CI/CD integration"
  enterprise:
    duration: "1-2d"
    description: "Exhaustive iteration across entire codebase"
    agents: ["test-automator", "debugger", "code-quality"]
    scope: "Entire codebase across multiple languages"
    iterations: "Unlimited (until 100% pass)"
    debugging: "Deep AI-driven RCA, distributed tracing"
    coverage: "Comprehensive (>90%), mutation testing"
    performance: "Benchmark validation, flaky test detection"
    use_case: "Release validation, comprehensive QA audit"
workflow_type: "iterative"
interactive_mode: true
color: blue
allowed-tools: Bash(npm:*), Bash(yarn:*), Bash(pnpm:*), Bash(pytest:*), Bash(python:*), Bash(cargo:*), Bash(go:*), Bash(mvn:*), Bash(gradle:*), Bash(make:*), Bash(jest:*), Bash(vitest:*), Bash(grep:*), Bash(find:*)
agents:
  primary:
    - test-automator
  conditional:
    - agent: debugger
      trigger: argument "--fix" OR pattern "fix|debug|failure"
    - agent: code-quality
      trigger: argument "--coverage" OR pattern "coverage|quality"
    - agent: devops-security-engineer
      trigger: pattern "ci|github.*actions|docker" OR files ".github/|.gitlab-ci.yml"
    - agent: hpc-numerical-coordinator
      trigger: pattern "numpy|scipy|pytest.*numerical" OR files "*.ipynb"
  orchestrated: false
---

# Iterative Test Execution & Auto-Fix System

Systematic test execution with AI-driven failure analysis, iterative fixing, and automated root cause detection across multiple frameworks and languages.

## Context

The user needs test execution and fixing for: $ARGUMENTS

## Execution Mode Selection

<AskUserQuestion>
questions:
  - question: "Which execution mode best fits your testing needs?"
    header: "Test Mode"
    multiSelect: false
    options:
      - label: "Quick (30min-1h)"
        description: "Single test file or small suite. Max 3 iterations, basic error analysis. Use for rapid debugging and specific test fixes."

      - label: "Standard (2-4h)"
        description: "Full test suite with up to 10 iterations. AI-assisted RCA, coverage >80%, parallel execution. Use for pre-commit validation and CI/CD."

      - label: "Enterprise (1-2d)"
        description: "Entire codebase with unlimited iterations until 100% pass. Deep AI-driven RCA, >90% coverage, mutation testing, flaky test detection. Use for release validation."
</AskUserQuestion>

## Phase 0: Test Framework Discovery

**See comprehensive guide**: [Framework Detection Guide](../docs/run-all-tests/framework-detection-guide.md)

### Auto-Detection Strategy

```bash
# Detect all test frameworks in repository
frameworks=$(detect_frameworks)

# JavaScript/TypeScript: Check for Jest, Vitest, Mocha
if [ -f "package.json" ]; then
    if grep -q '"jest"' package.json; then
        TEST_CMD="npm test"
    elif grep -q '"vitest"' package.json; then
        TEST_CMD="npx vitest run"
    fi
fi

# Python: Check for pytest, unittest
if command -v pytest &> /dev/null; then
    TEST_CMD="pytest"
elif [ -d "tests" ]; then
    TEST_CMD="python -m unittest discover"
fi

# Rust: Check for Cargo.toml
if [ -f "Cargo.toml" ]; then
    TEST_CMD="cargo test"
fi

# Go: Check for go.mod
if [ -f "go.mod" ]; then
    TEST_CMD="go test ./..."
fi

# Java: Check for Maven or Gradle
if [ -f "pom.xml" ]; then
    TEST_CMD="mvn test"
elif [ -f "build.gradle" ]; then
    TEST_CMD="./gradlew test"
fi
```

### Test Count Analysis

```bash
# Count total tests
TOTAL_TESTS=$(count_tests "$TEST_CMD")

echo "Discovered $TOTAL_TESTS tests using $TEST_CMD"
```

## Phase 1: Baseline Test Run

### Initial Execution

Execute tests and capture baseline metrics:

```bash
# Run tests with full output capture
$TEST_CMD > baseline_output.log 2>&1
EXIT_CODE=$?

# Parse results
PASSED=$(parse_passed baseline_output.log)
FAILED=$(parse_failed baseline_output.log)
SKIPPED=$(parse_skipped baseline_output.log)

PASS_RATE=$(echo "scale=2; $PASSED * 100 / ($PASSED + $FAILED)" | bc)

echo "
Baseline Test Results
Total Tests:  $((PASSED + FAILED + SKIPPED))
  Passed:     $PASSED ($PASS_RATE%)
  Failed:     $FAILED
  Skipped:    $SKIPPED

Goal: Fix $FAILED failures → 100% pass rate
"
```

## Phase 2: AI-Driven Failure Analysis

**See comprehensive guide**: [Debugging Strategies](../docs/run-all-tests/debugging-strategies.md)

### Failure Categorization

Categorize failures into actionable groups:

1. **Import/Module Errors** (20% of failures)
   - Cannot find module
   - Missing dependencies
   - Circular imports

2. **Assertion Failures** (35% of failures)
   - Expected vs actual mismatches
   - Logic errors
   - Outdated expectations

3. **Runtime Errors** (25% of failures)
   - Null pointer exceptions
   - Type errors
   - Undefined references

4. **Async/Timing Issues** (10% of failures)
   - Timeouts
   - Promise rejections
   - Race conditions

5. **Setup/Teardown Errors** (5% of failures)
   - Fixture issues
   - Mock problems
   - Database setup failures

6. **Snapshot Mismatches** (3% of failures)
7. **Environment/Configuration** (2% of failures)

### Root Cause Analysis with Ultra-Think

```
For each failing test:

1. Test Intent Analysis
   - What is this test trying to verify?
   - Unit / Integration / E2E?

2. Failure Point Identification
   - Where exactly does it fail?
   - Line number and context?

3. Code Change Correlation
   - What changed recently?
   - Git blame analysis

4. Dependency Analysis
   - What does this test depend on?
   - Mocks, fixtures, data, external services?

5. Flakiness Detection
   - Run test 10 times
   - Pass rate < 100% → flaky test
```

## Phase 3: Iterative Fix Execution

**See comprehensive guide**: [Test Execution Workflows](../docs/run-all-tests/test-execution-workflows.md)

### Iteration Loop

```bash
ITERATION=0
MAX_ITERATIONS=${MAX_ITERATIONS:-10}
PREVIOUS_FAILURES=999999

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))

    echo "
    ITERATION $ITERATION/$MAX_ITERATIONS
    "

    # Run tests
    $TEST_CMD > iter${ITERATION}.log 2>&1

    # Parse results
    CURRENT_FAILURES=$(parse_failed iter${ITERATION}.log)

    # Check success
    if [ $CURRENT_FAILURES -eq 0 ]; then
        echo " SUCCESS: All tests passing!"
        break
    fi

    # Check progress
    if [ $CURRENT_FAILURES -ge $PREVIOUS_FAILURES ]; then
        echo "⚠️ No progress or regression detected"
        break
    fi

    # Analyze and fix
    if [ "$AUTO_FIX" = true ]; then
        analyze_and_fix iter${ITERATION}.log
        git add -A
        git commit -m "test: fix batch from iteration $ITERATION"
    fi

    PREVIOUS_FAILURES=$CURRENT_FAILURES
done
```

### Fix Strategies by Category

**Import Errors**:
```bash
# Install missing packages
npm install --save-dev missing-package
pip install missing-package

# Fix import paths
sed -i 's|../utils/helper|../../utils/helper|' test.js
```

**Assertion Failures**:
```bash
# Option 1: Fix implementation (if code is wrong)
# Option 2: Update test expectation (if test is wrong)
# Option 3: Update mock (if test setup is wrong)
```

**Async Issues**:
```javascript
// Add missing await
test('async test', async () => {
    const result = await asyncFunction();
    expect(result).toBeDefined();
});
```

**Snapshot Updates**:
```bash
# Update snapshots if changes are intentional
npm test -- -u
pytest --snapshot-update
```

## Phase 4: Coverage Analysis & Reporting

**See comprehensive guide**: [Multi-Language Testing](../docs/run-all-tests/multi-language-testing.md)

### Coverage Integration

```bash
# Python
pytest --cov=src --cov-report=html --cov-report=term

# JavaScript
npm test -- --coverage

# Rust
cargo tarpaulin --out Html

# Go
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Java
mvn test jacoco:report
```

### Quality Gates

```bash
# Check coverage threshold
if [ "$COVERAGE" -lt 80 ]; then
    echo "❌ Coverage $COVERAGE% below threshold 80%"
    exit 1
fi

# Check pass rate
if [ "$FAILED" -gt 0 ]; then
    echo "❌ $FAILED tests still failing"
    exit 1
fi

echo " All quality gates passed!"
```

## Phase 5: Final Report

### Success Report

```
Test Suite: 100% Pass Rate Achieved!

Summary
  Total Tests:        120
  Pass Rate:          100%
  Time Taken:         15 minutes
  Iterations:         4
  Auto-Fixed:         98 tests (82%)

Progress Timeline
  Iteration 0:  40/120  (33%)  80 failures
  Iteration 1:  60/120  (50%)  60 failures (-20)
  Iteration 2:  80/120  (67%)  40 failures (-20)
  Iteration 3:  95/120  (79%)  25 failures (-15)
  Iteration 4: 120/120 (100%)   0 failures (-25)

Fixes Applied
  Import errors:        28 tests
  Async fixes:          22 tests
  Assertion updates:    35 tests
  Snapshot updates:     12 tests
  Setup fixes:           8 tests
  Complex logic:        15 tests

Coverage Improvement
  Line Coverage:   78% → 82% (+4%)
  Branch Coverage: 65% → 70% (+5%)

Status:  ALL TESTS PASSING
```

## Execution Modes

### Quick Mode (30min-1h)

```bash
/run-all-tests src/utils --max-iterations=3

# Runs fast iteration on specific file/directory
# Best for: Debugging specific failures
```

### Standard Mode (2-4h)

```bash
/run-all-tests --fix --parallel --coverage

# Runs comprehensive iteration with auto-fix
# Best for: Pre-commit validation, CI/CD
```

### Enterprise Mode (1-2d)

```bash
/run-all-tests --fix --max-iterations=100 --coverage --performance

# Runs exhaustive iteration across entire codebase
# Best for: Release validation, QA audit
```

## Common Debugging Scenarios

**Scenario 1: Flaky Tests**
```bash
# Run test multiple times to detect flakiness
for i in {1..10}; do
    $TEST_CMD specific_test.py
done

# Fix: Add proper waits, isolation, deterministic data
```

**Scenario 2: Environment Issues**
```bash
# Ensure correct environment
export NODE_ENV=test
export TESTING=true

# Start test database
docker-compose up -d test-db
```

**Scenario 3: Parallel Execution Conflicts**
```bash
# Use unique test data per test
user_email = f"test-{uuid.uuid4()}@example.com"

# Use database transactions
@pytest.fixture
def db_session():
    session = create_session()
    yield session
    session.rollback()
```

## Exit Criteria

**Success Conditions**:
- 100% test pass rate
- 0 test failures
- 0 test errors
- Coverage >= threshold (default 80%)
- All tests stable (not flaky)

**Stop Conditions**:
- All tests passing → SUCCESS
- Max iterations reached → PARTIAL SUCCESS
- No progress for 2 iterations → PLATEAU
- Regression detected → ROLLBACK

## Safety Guarantees

- Non-destructive analysis by default
- Incremental commits per iteration
- Easy rollback support
- Validation before moving to next iteration
- Full transparency with detailed reporting
- Manual fallback guide when automation can't proceed

## External Documentation

Comprehensive guides available in `docs/run-all-tests/`:

1. **framework-detection-guide.md** - Auto-detection for Jest, pytest, cargo, go test, Maven, Gradle
2. **debugging-strategies.md** - AI-driven RCA, log correlation, flaky test detection
3. **test-execution-workflows.md** - Iterative patterns, parallel strategies, CI/CD integration
4. **multi-language-testing.md** - Cross-language patterns, monorepo strategies, framework comparison

## Now Execute

Begin comprehensive test suite analysis and iterative fixing process based on selected execution mode.

**Remember**:
- Safety first: validate every fix
- Progress incrementally: small commits
- Learn from failures: improve fix strategies
- Full transparency: detailed reporting
- Know when to stop: provide manual guide when needed

Let's get those tests green!
