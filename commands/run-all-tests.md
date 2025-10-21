---
description: Iteratively run all the tests in the codebase to fix the test failures, errors, and warnings until zero failures/errors and 100% pass rate
allowed-tools: Bash(npm:*), Bash(yarn:*), Bash(pnpm:*), Bash(pytest:*), Bash(python:*), Bash(cargo:*), Bash(go:*), Bash(mvn:*), Bash(gradle:*), Bash(make:*), Bash(jest:*), Bash(vitest:*), Bash(grep:*), Bash(find:*)
argument-hint: [test-path] [--fix] [--max-iterations=10] [--parallel] [--coverage]
color: blue
agents:
  primary:
    - code-quality
  conditional:
    - agent: devops-security-engineer
      trigger: pattern "ci|github.*actions|docker" OR files ".github/|.gitlab-ci.yml"
    - agent: hpc-numerical-coordinator
      trigger: pattern "numpy|scipy|pytest.*numerical" OR files "*.ipynb"
  orchestrated: false
---

# Iterative Test Execution & Auto-Fix System

## Phase 0: Test Framework Discovery & Environment Setup

### Repository Context
- Working directory: !`pwd`
- Git status: !`git status --short 2>/dev/null | head -10`
- Current branch: !`git branch --show-current 2>/dev/null`
- Uncommitted changes: !`git diff --stat 2>/dev/null | tail -1`

### Test Framework Detection

#### JavaScript/TypeScript Ecosystem
- Package.json: @package.json
- Jest config: !`find . -maxdepth 2 -name "jest.config.*" -o -name ".jestrc*" 2>/dev/null`
- Vitest config: !`find . -maxdepth 2 -name "vitest.config.*" 2>/dev/null`
- Mocha config: !`find . -maxdepth 2 -name ".mocharc.*" -o -name "mocha.opts" 2>/dev/null`
- Test scripts: !`grep -A 5 '"scripts"' package.json 2>/dev/null | grep test`

#### Python Ecosystem
- Pytest: !`which pytest 2>/dev/null || echo "Not found"`
- Unittest: !`find . -name "test_*.py" -o -name "*_test.py" 2>/dev/null | head -5`
- Tox config: !`find . -maxdepth 2 -name "tox.ini" 2>/dev/null`
- Pytest config: !`find . -maxdepth 2 -name "pytest.ini" -o -name "pyproject.toml" 2>/dev/null`
- Requirements: @requirements.txt
- Dev requirements: @requirements-dev.txt

#### Rust Ecosystem
- Cargo.toml: @Cargo.toml
- Test directories: !`find . -type d -name "tests" 2>/dev/null | head -5`
- Cargo test available: !`which cargo 2>/dev/null || echo "Not found"`

#### Go Ecosystem
- Go.mod: @go.mod
- Test files: !`find . -name "*_test.go" 2>/dev/null | head -10`
- Go test available: !`which go 2>/dev/null || echo "Not found"`

#### Java Ecosystem
- Maven: !`find . -maxdepth 2 -name "pom.xml" 2>/dev/null`
- Gradle: !`find . -maxdepth 2 -name "build.gradle" -o -name "build.gradle.kts" 2>/dev/null`
- JUnit tests: !`find . -path "*/src/test/*" -name "*.java" 2>/dev/null | head -5`

#### Other Frameworks
- Ruby (RSpec): !`find . -maxdepth 2 -name ".rspec" -o -name "spec_helper.rb" 2>/dev/null`
- PHP (PHPUnit): !`find . -maxdepth 2 -name "phpunit.xml" 2>/dev/null`
- C/C++ (GoogleTest): !`find . -name "*_test.cpp" -o -name "*_test.cc" 2>/dev/null | head -5`

### Test Count Analysis
- Total test files: !`find . -name "*.test.*" -o -name "*.spec.*" -o -name "*_test.*" -o -name "test_*" 2>/dev/null | grep -v node_modules | wc -l`
- JavaScript tests: !`find . -name "*.test.js" -o -name "*.test.ts" -o -name "*.spec.js" -o -name "*.spec.ts" 2>/dev/null | grep -v node_modules | wc -l`
- Python tests: !`find . -name "test_*.py" -o -name "*_test.py" 2>/dev/null | wc -l`
- Go tests: !`find . -name "*_test.go" 2>/dev/null | wc -l`
- Rust tests: !`grep -r "#\[test\]" --include="*.rs" 2>/dev/null | wc -l`

---

## Phase 1: Initial Test Run & Baseline Establishment

### Test Execution Strategy Detection

**Determine test command**:
```bash
# Priority order for JavaScript/TypeScript
1. npm test (if "test" script exists)
2. yarn test
3. pnpm test
4. npx jest
5. npx vitest
6. npx mocha

# For Python
1. pytest (if available)
2. python -m pytest
3. python -m unittest discover
4. tox

# For Rust
1. cargo test

# For Go
1. go test ./...

# For Java
1. mvn test
2. gradle test
```

### Initial Baseline Run

**Execute with full output capture**:
```bash
# Run tests and capture all output
TEST_OUTPUT=$(run_test_command 2>&1)
EXIT_CODE=$?

# Parse results
TOTAL_TESTS=$(extract_total_tests "$TEST_OUTPUT")
PASSED=$(extract_passed_tests "$TEST_OUTPUT")
FAILED=$(extract_failed_tests "$TEST_OUTPUT")
SKIPPED=$(extract_skipped_tests "$TEST_OUTPUT")
ERRORS=$(extract_error_tests "$TEST_OUTPUT")
WARNINGS=$(extract_warnings "$TEST_OUTPUT")

# Calculate metrics
PASS_RATE=$(calculate_pass_rate $PASSED $TOTAL_TESTS)
FAIL_RATE=$(calculate_fail_rate $FAILED $TOTAL_TESTS)
```

### Baseline Metrics

**Initial State**:
```
📊 Test Suite Baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Tests:     ${TOTAL_TESTS}
✅ Passed:       ${PASSED} (${PASS_RATE}%)
❌ Failed:       ${FAILED} (${FAIL_RATE}%)
⚠️  Errors:       ${ERRORS}
⏭️  Skipped:      ${SKIPPED}
⚠️  Warnings:     ${WARNINGS}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Goal: 100% pass rate (${TOTAL_TESTS}/${TOTAL_TESTS})
Gap:  ${FAILED + ERRORS} tests to fix
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Phase 2: Multi-Agent Test Failure Analysis

### Agent 1: Test Output Parser & Categorizer
**Mission**: Parse test output and categorize all failures

**Parsing Strategy by Framework**:

#### Jest/Vitest Output
```
Pattern: "● Test suite failed to run"
Pattern: "FAIL src/components/Button.test.tsx"
Pattern: "✕ renders correctly (5 ms)"
Pattern: "Expected: true, Received: false"
Pattern: "ReferenceError: X is not defined"
Pattern: "TypeError: Cannot read property 'X' of undefined"
```

#### Pytest Output
```
Pattern: "FAILED tests/test_api.py::test_user_creation"
Pattern: "AssertionError: assert 1 == 2"
Pattern: "fixture 'db' not found"
Pattern: "ModuleNotFoundError: No module named 'X'"
Pattern: "E       AttributeError: 'NoneType' object has no attribute 'X'"
```

#### Go Test Output
```
Pattern: "FAIL    github.com/user/repo/pkg"
Pattern: "panic: runtime error"
Pattern: "got 1, want 2"
Pattern: "undefined: X"
```

#### Cargo Test Output
```
Pattern: "test result: FAILED"
Pattern: "thread 'main' panicked at"
Pattern: "assertion failed: `(left == right)`"
Pattern: "error[E0425]: cannot find value `X`"
```

**Extraction Tasks**:
1. Extract test file path
2. Extract test name/description
3. Extract failure reason
4. Extract stack trace
5. Extract expected vs actual values
6. Extract error type
7. Extract line numbers

### Agent 2: Failure Categorization Engine
**Mission**: Classify failures into actionable categories

**Category Taxonomy**:

#### Category 1: Assertion Failures (35% of failures)
**Characteristics**:
- Expected value ≠ Actual value
- Test logic issue or implementation bug
- Clear comparison in output

**Subcategories**:
- Value mismatch: `expect(5).toBe(10)`
- Type mismatch: `expect(string).toBe(number)`
- Array/Object mismatch: `expect([1,2]).toEqual([1,2,3])`
- Boolean assertion: `expect(false).toBeTruthy()`
- Null/undefined: `expect(null).toBeDefined()`

**Example**:
```javascript
Expected: { name: 'John', age: 30 }
Received: { name: 'John', age: 25 }
```

#### Category 2: Import/Module Errors (20% of failures)
**Characteristics**:
- Test suite fails to load
- Missing dependencies
- Import path issues

**Subcategories**:
- Module not found: `Cannot find module 'X'`
- Circular dependencies: `Circular dependency detected`
- Import errors: `SyntaxError: Unexpected token import`
- Type import errors: `Cannot find type 'X'`

**Example**:
```
Error: Cannot find module '../utils/helper'
```

#### Category 3: Runtime Errors (25% of failures)
**Characteristics**:
- Code execution crashes
- Null pointer exceptions
- Type errors at runtime

**Subcategories**:
- Null/undefined access: `Cannot read property 'X' of undefined`
- Type errors: `X is not a function`
- Reference errors: `X is not defined`
- Range errors: `Maximum call stack size exceeded`

**Example**:
```
TypeError: Cannot read property 'map' of undefined
at UserList.render (UserList.tsx:25)
```

#### Category 4: Async/Timing Issues (10% of failures)
**Characteristics**:
- Timeouts
- Promise rejections
- Race conditions

**Subcategories**:
- Timeout: `Timeout - Async callback was not invoked`
- Unhandled rejection: `UnhandledPromiseRejectionWarning`
- Await missing: `Promise pending`
- Race condition: `Intermittent failure`

**Example**:
```
Timeout - Async callback was not invoked within 5000ms
```

#### Category 5: Setup/Teardown Errors (5% of failures)
**Characteristics**:
- beforeEach/afterEach failures
- Fixture issues
- Mock setup problems

**Subcategories**:
- Fixture not found: `fixture 'db' not found`
- Setup failed: `beforeEach hook failed`
- Teardown failed: `afterEach cleanup error`
- Mock issues: `Mock function not implemented`

#### Category 6: Snapshot Mismatches (3% of failures)
**Characteristics**:
- Snapshot tests failing
- UI regression tests
- Output comparison failures

**Example**:
```
Snapshot mismatch
- Expected:
+ Received:

- <div class="old-class">
+ <div class="new-class">
```

#### Category 7: Environment/Configuration (2% of failures)
**Characteristics**:
- Missing environment variables
- Config issues
- Path problems

**Example**:
```
Error: Environment variable DATABASE_URL not set
```

### Agent 3: Root Cause Analyzer
**Mission**: Determine why each test is failing using UltraThink intelligence

**Analysis Framework**:

#### For Each Failed Test:

**1. Test Intent Analysis**
```
Question: What is this test trying to verify?
- Unit test: Single function/component
- Integration test: Multiple components
- E2E test: Full user flow
- Regression test: Bug fix verification
```

**2. Failure Point Identification**
```
Question: Where exactly does it fail?
- File: tests/components/Button.test.tsx
- Line: 45
- Function: expect(onClick).toHaveBeenCalled()
- Context: After button click simulation
```

**3. Code Change Correlation**
```
Question: What changed recently?
- Git blame: Show last change to failing line
- Recent commits: Last 5 commits touching this file
- PR context: Is this a new test or existing test?
```

**4. Dependency Analysis**
```
Question: What does this test depend on?
- Mocks: Are mocks properly configured?
- Fixtures: Are fixtures available?
- Data: Is test data valid?
- External services: Any network calls?
```

**5. Flakiness Detection**
```
Question: Is this test flaky?
- Run test 10 times
- If pass rate < 100% → flaky test
- Identify non-deterministic behavior
```

### Agent 4: Solution Generator
**Mission**: Generate ranked fix strategies for each failure category

**Solution Templates by Category**:

#### For Assertion Failures:

**Option 1: Code Fix (if implementation wrong)**
```javascript
// Before
function calculateTotal(items) {
  return items.reduce((sum, item) => sum + item.price, 0);
}

// After (fix: forgot tax)
function calculateTotal(items) {
  const subtotal = items.reduce((sum, item) => sum + item.price, 0);
  return subtotal * 1.1; // Add 10% tax
}
```

**Option 2: Test Fix (if test expectation wrong)**
```javascript
// Before
expect(calculateTotal([{price: 10}])).toBe(10);

// After (fix: update expectation to include tax)
expect(calculateTotal([{price: 10}])).toBe(11);
```

**Option 3: Mock Update (if test setup wrong)**
```javascript
// Before
mockGetUser.mockReturnValue({ name: 'John' });

// After (fix: add missing field)
mockGetUser.mockReturnValue({ name: 'John', age: 30 });
```

#### For Import/Module Errors:

**Option 1: Install Missing Package**
```bash
npm install --save-dev missing-package
```

**Option 2: Fix Import Path**
```typescript
// Before
import { helper } from '../utils/helper';

// After
import { helper } from '../../utils/helper';
```

**Option 3: Add Type Definitions**
```bash
npm install --save-dev @types/package-name
```

#### For Runtime Errors:

**Option 1: Add Null Checks**
```typescript
// Before
const names = users.map(u => u.name);

// After
const names = users?.map(u => u.name) ?? [];
```

**Option 2: Fix Type Errors**
```typescript
// Before
const value = parseInt('123');
const doubled = value.map(x => x * 2); // Error: number has no map

// After
const value = parseInt('123');
const doubled = value * 2;
```

**Option 3: Add Error Handling**
```typescript
// Before
const data = JSON.parse(response);

// After
let data;
try {
  data = JSON.parse(response);
} catch (error) {
  data = null;
}
```

#### For Async/Timing Issues:

**Option 1: Add await**
```typescript
// Before
test('fetches user', () => {
  const user = fetchUser();
  expect(user.name).toBe('John');
});

// After
test('fetches user', async () => {
  const user = await fetchUser();
  expect(user.name).toBe('John');
});
```

**Option 2: Increase timeout**
```typescript
test('slow operation', async () => {
  // ...
}, 10000); // 10 second timeout
```

**Option 3: Use waitFor**
```typescript
// Before
expect(screen.getByText('Loaded')).toBeInTheDocument();

// After
await waitFor(() => {
  expect(screen.getByText('Loaded')).toBeInTheDocument();
});
```

#### For Snapshot Mismatches:

**Option 1: Update snapshots (if change is intentional)**
```bash
npm test -- -u
```

**Option 2: Revert code changes (if change is unintentional)**
```bash
git checkout -- src/components/Button.tsx
```

**Option 3: Fix dynamic values**
```javascript
// Before (contains timestamp that changes)
<div>{new Date().toISOString()}</div>

// After (use fixed date in tests)
<div>{props.timestamp || new Date().toISOString()}</div>
```

---

## Phase 3: UltraThink Intelligence Layer

### Deep Reasoning Framework for Test Failures

**1. Problem Space Understanding**

**Current State**:
- Failed tests: `${FAILED_COUNT}`
- Error types: `${ERROR_CATEGORIES}`
- Affected areas: `${AFFECTED_MODULES}`
- Complexity score: `${COMPLEXITY_1_TO_10}`

**Historical Context**:
```bash
# When did tests last pass?
git log --all --grep="all tests passing" --oneline -1

# What changed since then?
git diff ${LAST_GREEN_COMMIT}..HEAD --stat

# Were these tests ever green?
git log -S "test('failing test name')" --oneline
```

**Impact Assessment**:
- Is this blocking CI/CD? `${BLOCKING_PIPELINE}`
- Is this blocking feature development? `${BLOCKING_FEATURES}`
- How many developers affected? `${TEAM_SIZE}`
- Technical debt accumulation rate? `${DEBT_VELOCITY}`

**2. Multi-Perspective Analysis**

#### Developer Perspective
```
Question: Why are developers not fixing these?
- Time pressure: Sprint deadlines?
- Complexity: Too hard to understand?
- Knowledge gap: Unfamiliar with test framework?
- Tooling: Difficult to debug locally?
```

#### Quality Perspective
```
Question: What does this tell us about code quality?
- Test coverage inadequate?
- Tests not maintained alongside code?
- Missing integration tests?
- Tech debt accumulated?
```

#### Risk Perspective
```
Question: What's the risk of broken tests?
- Regression risk: Bugs shipping to production
- Developer confidence: Fear of breaking things
- Deployment risk: Can't safely deploy
- Technical debt: Increasing fix cost over time
```

**3. Solution Strategy Synthesis**

**Quick Win vs. Proper Fix Analysis**:

```
For each failing test, evaluate:

QUICK WIN (immediate fix):
- Confidence: Can we fix this in <5 min?
- Risk: Will it introduce other issues?
- Sustainability: Is this a proper fix or hack?

Example:
Test: "User login validates email"
Failure: "expect(validate('test')).toBe(false)"
Quick Win: Change to .toBe(true) [90 seconds]
Risk: HIGH - bypasses validation
Verdict: ❌ Don't do quick win, fix properly

PROPER FIX (sustainable solution):
- Root cause: Email validation regex broken
- Investigation time: 10 minutes
- Fix time: 5 minutes
- Test verification: 2 minutes
- Total: 17 minutes
Verdict: ✅ Do proper fix
```

**Batch Processing Strategy**:

```
Group failures by:
1. Same root cause (fix once, resolve many)
2. Same file (reduce context switching)
3. Same category (use same debugging tools)
4. Same developer area (domain knowledge)

Example batching:
Batch 1: All "Cannot find module" errors (15 tests)
  → Solution: Run npm install, fix imports
  → Expected resolution: 12/15 tests

Batch 2: All snapshot mismatches (8 tests)
  → Solution: Review changes, update snapshots
  → Expected resolution: 8/8 tests

Batch 3: All async timeout errors (5 tests)
  → Solution: Add proper await, increase timeouts
  → Expected resolution: 4/5 tests (1 needs investigation)
```

**4. Iterative Fix Planning**

**Iteration Strategy**:
```
Iteration 1: Low-hanging fruit (expected 40% reduction)
- Fix import errors
- Install missing dependencies
- Update snapshots
- Add missing awaits

Iteration 2: Assertion failures (expected 30% reduction)
- Fix obvious logic bugs
- Update test expectations
- Fix mock configurations

Iteration 3: Complex issues (expected 20% reduction)
- Fix timing issues
- Resolve race conditions
- Fix flaky tests

Iteration 4: Edge cases (expected 10% reduction)
- Investigate remaining failures
- Deep debugging required
- May need architecture changes
```

**5. Learning & Adaptation**

**Track Fix Effectiveness**:
```javascript
const fixHistory = {
  iteration1: {
    fixesAttempted: 25,
    fixesSuccessful: 22,
    successRate: 0.88,
    categories: {
      "import_errors": { attempted: 15, successful: 15, rate: 1.0 },
      "assertion_failures": { attempted: 10, successful: 7, rate: 0.7 }
    }
  }
};

// Adapt strategy based on success rate
if (successRate < 0.7) {
  strategy = "MORE_ANALYSIS_NEEDED";
} else {
  strategy = "CONTINUE_CURRENT_APPROACH";
}
```

---

## Phase 4: Iterative Fix Execution

### Iteration Loop

```bash
ITERATION=0
MAX_ITERATIONS=${MAX_ITERATIONS:-10}
PREVIOUS_FAILED_COUNT=999999

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
  ITERATION=$((ITERATION + 1))

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🔄 ITERATION $ITERATION / $MAX_ITERATIONS"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Step 1: Run tests
  run_tests > test_output_iter${ITERATION}.log 2>&1
  EXIT_CODE=$?

  # Step 2: Parse results
  parse_test_results test_output_iter${ITERATION}.log

  # Step 3: Check progress
  if [ $FAILED_COUNT -eq 0 ]; then
    echo "✅ SUCCESS! All tests passing!"
    break
  fi

  if [ $FAILED_COUNT -eq $PREVIOUS_FAILED_COUNT ]; then
    echo "⚠️  No progress made. Switching strategy..."
    STRATEGY="DEEP_ANALYSIS"
  fi

  if [ $FAILED_COUNT -gt $PREVIOUS_FAILED_COUNT ]; then
    echo "❌ Failures increased! Rolling back last fix..."
    git checkout HEAD -- .
    break
  fi

  # Step 4: Analyze failures
  analyze_failures test_output_iter${ITERATION}.log

  # Step 5: Generate fixes
  generate_fixes

  # Step 6: Apply fixes
  if [ "$AUTO_FIX" = true ]; then
    apply_fixes
  else
    echo "Generated fixes. Apply with --fix flag."
    break
  fi

  # Step 7: Update metrics
  PREVIOUS_FAILED_COUNT=$FAILED_COUNT

  # Step 8: Commit progress
  git add -A
  git commit -m "test: fix batch from iteration $ITERATION

- Fixed: $FIXED_THIS_ITERATION tests
- Remaining: $FAILED_COUNT failures
- Categories: ${FIX_CATEGORIES[@]}

Auto-fixed by run-all-tests command"

done
```

### Fix Application Strategy

**Safety-First Application**:

```bash
apply_fix() {
  local FIX_FILE=$1
  local FIX_TYPE=$2
  local CONFIDENCE=$3

  # Backup file
  cp "$FIX_FILE" "$FIX_FILE.backup"

  # Apply fix
  case $FIX_TYPE in
    "import_fix")
      apply_import_fix "$FIX_FILE"
      ;;
    "assertion_fix")
      apply_assertion_fix "$FIX_FILE"
      ;;
    "async_fix")
      apply_async_fix "$FIX_FILE"
      ;;
    "snapshot_update")
      update_snapshots
      ;;
    *)
      echo "Unknown fix type: $FIX_TYPE"
      return 1
      ;;
  esac

  # Validate fix
  run_single_test "$FIX_FILE"
  if [ $? -eq 0 ]; then
    echo "✅ Fix successful for $FIX_FILE"
    rm "$FIX_FILE.backup"
    return 0
  else
    echo "❌ Fix failed for $FIX_FILE, rolling back"
    mv "$FIX_FILE.backup" "$FIX_FILE"
    return 1
  fi
}
```

### Specific Fix Implementations

#### Fix 1: Import Errors
```bash
fix_import_errors() {
  # Extract missing module name
  MISSING_MODULE=$(grep "Cannot find module" $TEST_OUTPUT | \
    sed -n "s/.*Cannot find module '\(.*\)'.*/\1/p" | head -1)

  # Try to install
  if [[ $MISSING_MODULE == @* ]]; then
    # Scoped package
    npm install --save-dev "$MISSING_MODULE"
  else
    # Check if it's a path issue
    find . -name "${MISSING_MODULE}.ts" -o -name "${MISSING_MODULE}.js"
    # Suggest correct path
  fi
}
```

#### Fix 2: Assertion Failures
```bash
fix_assertion_failures() {
  local TEST_FILE=$1
  local LINE_NUMBER=$2
  local EXPECTED=$3
  local RECEIVED=$4

  # Analyze if code or test should change
  analyze_assertion_intent "$TEST_FILE" "$LINE_NUMBER"

  if [ "$INTENT" = "test_wrong" ]; then
    # Update test expectation
    sed -i "${LINE_NUMBER}s/${EXPECTED}/${RECEIVED}/" "$TEST_FILE"
  else
    # Fix implementation (requires deeper analysis)
    analyze_implementation_bug
  fi
}
```

#### Fix 3: Async Issues
```bash
fix_async_issues() {
  local TEST_FILE=$1

  # Find test functions without async
  grep -n "test('.*'," "$TEST_FILE" | while read -r line; do
    LINE_NUM=$(echo "$line" | cut -d: -f1)

    # Check if function uses await
    if grep -A 10 "test('.*'," "$TEST_FILE" | grep -q "await"; then
      # Add async keyword
      sed -i "${LINE_NUM}s/test(/test(async /" "$TEST_FILE"
    fi
  done

  # Find missing awaits
  grep -n "fetchUser()\|apiCall()\|database\." "$TEST_FILE" | while read -r line; do
    LINE_NUM=$(echo "$line" | cut -d: -f1)

    # Add await if missing
    if ! grep -q "await" <<< "$line"; then
      sed -i "${LINE_NUM}s/const /const await /" "$TEST_FILE"
    fi
  done
}
```

#### Fix 4: Snapshot Updates
```bash
fix_snapshots() {
  echo "📸 Updating snapshots..."

  # Show what will change
  npm test -- -u --dry-run

  # Ask for confirmation if not auto-fix
  if [ "$AUTO_FIX" != true ]; then
    read -p "Update snapshots? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      return 1
    fi
  fi

  # Update snapshots
  npm test -- -u
}
```

#### Fix 5: Missing Mocks
```bash
fix_missing_mocks() {
  local TEST_FILE=$1
  local MISSING_MOCK=$2

  # Add mock setup
  cat >> "$TEST_FILE" <<EOF

jest.mock('${MISSING_MOCK}', () => ({
  __esModule: true,
  default: jest.fn(() => ({})),
}));
EOF
}
```

---

## Phase 5: Progress Tracking & Reporting

### Real-Time Progress Dashboard

```
╔══════════════════════════════════════════════════════════════╗
║              TEST FIXING PROGRESS - ITERATION 3              ║
╠══════════════════════════════════════════════════════════════╣
║  Status: 🔄 In Progress                                      ║
║  Goal: 100% pass rate (120/120 tests)                       ║
╠══════════════════════════════════════════════════════════════╣
║  📊 Current Metrics:                                         ║
║    Total Tests:        120                                   ║
║    ✅ Passing:         95 (79%)  [▓▓▓▓▓▓▓▓░░░░] +15         ║
║    ❌ Failing:         25 (21%)  [░░░░░░░░▓▓▓▓] -15         ║
║    ⚠️  Errors:          8 (7%)   [░░░░░░░░░░▓▓] -3          ║
║    ⏭️  Skipped:         0 (0%)                               ║
╠══════════════════════════════════════════════════════════════╣
║  📈 Progress:                                                ║
║    Iteration 1:  60/120 ✅  (50%) → Fixed 20                ║
║    Iteration 2:  80/120 ✅  (67%) → Fixed 20                ║
║    Iteration 3:  95/120 ✅  (79%) → Fixed 15                ║
║    Remaining:    25 tests to fix                             ║
║    Velocity:     18 tests/iteration (avg)                   ║
║    ETA:          2 iterations (~10 minutes)                  ║
╠══════════════════════════════════════════════════════════════╣
║  🎯 Current Batch: Assertion Failures (10 tests)            ║
║    Status: Analyzing root causes...                          ║
║    Confidence: 85% (high)                                    ║
║    Expected fixes: 8-9 tests                                 ║
╠══════════════════════════════════════════════════════════════╣
║  📋 Fix Categories Applied:                                  ║
║    ✅ Import errors fixed:        15 tests                  ║
║    ✅ Snapshots updated:           8 tests                  ║
║    ✅ Async fixes applied:        12 tests                  ║
║    🔄 Assertion fixes in progress: 10 tests                 ║
║    ⏳ Pending analysis:           25 tests                  ║
╠══════════════════════════════════════════════════════════════╣
║  ⚡ Recent Fixes:                                            ║
║    ✅ Button.test.tsx (import fix)                          ║
║    ✅ UserList.test.tsx (snapshot update)                   ║
║    ✅ api.test.ts (async/await fix)                         ║
║    ✅ utils.test.js (assertion fix)                         ║
╚══════════════════════════════════════════════════════════════╝
```

### Detailed Iteration Report

After each iteration, generate:

```markdown
# Iteration 3 Report

## Summary
- **Tests Fixed**: 15
- **Tests Remaining**: 25
- **Success Rate**: 60% (15/25 attempted)
- **Time Taken**: 3 minutes 45 seconds

## Fixes Applied

### Category: Import Errors (5 tests)
✅ **Button.test.tsx:5** - Cannot find module 'react-icons'
   - Solution: `npm install --save-dev react-icons`
   - Result: FIXED
   - Time: 45s

✅ **Header.test.tsx:12** - Cannot find module '../utils/helper'
   - Solution: Updated path to '../../utils/helper'
   - Result: FIXED
   - Time: 15s

... (3 more)

### Category: Assertion Failures (7 tests)
✅ **calculateTotal.test.js:23** - Expected 10, received 11
   - Root Cause: Tax calculation added to function
   - Solution: Updated test expectation from 10 to 11
   - Result: FIXED
   - Time: 30s
   - Confidence: HIGH

⚠️ **validateEmail.test.js:45** - Expected true, received false
   - Root Cause: Regex changed to be more strict
   - Solution: Updated test input to valid email
   - Result: FIXED
   - Time: 2m
   - Confidence: MEDIUM

❌ **processOrder.test.js:67** - Expected {...}, received undefined
   - Root Cause: Unclear - needs investigation
   - Solution: PENDING
   - Time: 5m (timeout)
   - Confidence: LOW
   - Next Action: Manual debugging required

... (4 more)

### Category: Async Issues (3 tests)
✅ **fetchUser.test.ts:34** - Timeout exceeded
   - Solution: Added missing await keyword
   - Result: FIXED
   - Time: 20s

... (2 more)

## Failures Analysis

### Still Failing (10 tests)

**High Priority** (blocking others):
1. **database.test.js:12** - Connection refused
   - Category: Environment
   - Impact: Blocks 5 other tests
   - Recommendation: Start test database

**Medium Priority**:
2. **auth.test.js:56** - Token validation failed
   - Category: Logic error
   - Impact: 1 test
   - Recommendation: Debug token generation

... (8 more)

## Next Iteration Plan

**Target**: Fix 10 tests
**Strategy**: Focus on database tests first (unblocks 5 others)
**Expected Time**: 5 minutes

1. Start test database
2. Fix authentication tests
3. Address remaining assertion failures
```

---

## Phase 6: Intelligent Decision Making

### When to Stop Iteration

**Success Conditions**:
```bash
# Stop when goal reached
if [ $FAILED_COUNT -eq 0 ] && [ $ERROR_COUNT -eq 0 ]; then
  echo "✅ GOAL ACHIEVED: 100% pass rate!"
  exit 0
fi
```

**Plateau Detection**:
```bash
# Stop if no progress for 2 iterations
if [ $NO_PROGRESS_COUNT -ge 2 ]; then
  echo "⚠️  Plateau detected. Manual intervention needed."
  generate_manual_intervention_guide
  exit 1
fi
```

**Max Iterations Reached**:
```bash
if [ $ITERATION -ge $MAX_ITERATIONS ]; then
  echo "⏱️  Max iterations reached."
  echo "Progress: $INITIAL_FAILED → $FAILED_COUNT failures"
  echo "Improvement: $((100 - FAILED_COUNT * 100 / INITIAL_FAILED))%"
  generate_remaining_failures_report
  exit 1
fi
```

**Regression Detection**:
```bash
if [ $FAILED_COUNT -gt $PREVIOUS_FAILED_COUNT ]; then
  echo "❌ REGRESSION: Failures increased!"
  echo "Rolling back last changes..."
  git reset --hard HEAD~1
  exit 1
fi
```

### Manual Intervention Guide

When automatic fixing plateaus:

```markdown
# Manual Intervention Required

## Remaining Failures: 12 tests

### Category: Complex Logic Errors (8 tests)
These require understanding business logic:

1. **processPayment.test.js:89**
   ```
   Expected: Payment successful
   Received: Insufficient funds error
   ```
   **Analysis**: Mock account balance may be incorrect
   **Suggested Action**:
   - Review account setup in test
   - Verify payment processing logic
   - Check for recent changes in payment module

   **Code Context**:
   ```javascript
   // processPayment.test.js:85-95
   const account = createMockAccount({ balance: 100 });
   const result = processPayment(account, { amount: 150 });
   expect(result.success).toBe(true); // FAILS
   ```

   **Possible Fixes**:
   a) Increase mock balance: `balance: 200`
   b) Decrease payment amount: `amount: 50`
   c) Fix payment logic if it's the bug

2. **calculateDiscount.test.js:34**
   ...

### Category: Environment Setup (4 tests)
These require external resources:

1. **database.test.js** (all 4 tests)
   **Issue**: Cannot connect to test database
   **Required Actions**:
   ```bash
   # Start test database
   docker-compose up -d test-db

   # Run migrations
   npm run db:migrate:test

   # Seed test data
   npm run db:seed:test
   ```

### Recommended Next Steps
1. ☑️ Start test database (fixes 4 tests)
2. ☑️ Review payment logic with domain expert (fixes ~3 tests)
3. ☑️ Debug remaining 5 tests individually
4. ☑️ Consider if tests need updating for new requirements

### Commands to Debug Individual Tests
```bash
# Run single test file with debug
npm test -- processPayment.test.js --verbose

# Run with debugger
node --inspect-brk node_modules/.bin/jest processPayment.test.js

# Run with coverage
npm test -- processPayment.test.js --coverage
```
```

---

## Phase 7: Final Report Generation

### Success Report (100% Pass Rate Achieved)

```markdown
# 🎉 Test Suite: 100% Pass Rate Achieved!

## Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Total Tests**: 120
**Pass Rate**: 100% ✅
**Time Taken**: 15 minutes
**Iterations**: 4
**Auto-Fixed**: 98 tests (82%)
**Manual Intervention**: 22 tests (18%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Progress Timeline

```
Iteration 0 (Baseline):  40/120 ✅ (33%)  80 failures
Iteration 1 (Imports):   60/120 ✅ (50%)  60 failures (-20)
Iteration 2 (Async):     80/120 ✅ (67%)  40 failures (-20)
Iteration 3 (Assert):    95/120 ✅ (79%)  25 failures (-15)
Iteration 4 (Complex):  120/120 ✅ (100%)  0 failures (-25)
```

## Fixes Applied by Category

### 1. Import/Module Errors: 28 tests fixed
- Installed missing packages: 15 tests
- Fixed import paths: 10 tests
- Added type definitions: 3 tests

### 2. Async/Timing Issues: 22 tests fixed
- Added missing await: 15 tests
- Increased timeouts: 4 tests
- Fixed race conditions: 3 tests

### 3. Assertion Failures: 35 tests fixed
- Updated test expectations: 20 tests
- Fixed implementation bugs: 10 tests
- Updated mocks: 5 tests

### 4. Snapshot Mismatches: 12 tests fixed
- Updated snapshots: 12 tests

### 5. Setup/Environment: 8 tests fixed
- Started test database: 4 tests
- Added environment variables: 3 tests
- Fixed fixture setup: 1 test

### 6. Complex Logic: 15 tests fixed
- Debugged business logic: 8 tests
- Fixed edge cases: 5 tests
- Updated test data: 2 tests

## Code Changes

**Files Modified**: 45
**Lines Changed**: ~350
**Commits Created**: 5

### Key Commits:
1. `test: fix import errors and missing deps (20 tests)`
2. `test: add async/await to test functions (22 tests)`
3. `test: update assertions and snapshots (47 tests)`
4. `test: fix environment and setup issues (8 tests)`
5. `test: resolve complex logic failures (15 tests)`

## Performance Impact

**Test Execution Time**:
- Before: 2m 45s (with failures)
- After: 1m 30s (all passing)
- Improvement: 45% faster ⚡

**CI/CD Impact**:
- Build Status: ✅ PASSING
- Deployment: UNBLOCKED
- Developer Confidence: HIGH

## Quality Improvements

### Test Coverage
- Line Coverage: 78% → 82% (+4%)
- Branch Coverage: 65% → 70% (+5%)
- Function Coverage: 85% → 88% (+3%)

### Code Quality
- Tests are now maintained and passing
- Better async handling patterns
- Improved mock configurations
- Updated snapshots reflect current UI

## Lessons Learned

### Common Issues Found:
1. **Missing awaits** (18% of failures)
   - Added lint rule: `@typescript-eslint/no-floating-promises`

2. **Outdated snapshots** (10% of failures)
   - Added pre-commit hook to check snapshots

3. **Import path issues** (23% of failures)
   - Configured path aliases in tsconfig.json

### Preventive Measures Added:
- ✅ Pre-commit hook: Run affected tests
- ✅ CI: Fail on test failures (was allowing failures)
- ✅ Lint rules: Catch common async mistakes
- ✅ Documentation: Test writing guidelines

## Recommendations

### Short-term (This Week):
1. ✅ Keep tests green (don't let failures accumulate)
2. ✅ Review test quality guidelines with team
3. ✅ Add test coverage requirements to CI

### Medium-term (This Month):
1. 📈 Increase coverage to 85%
2. 🔧 Refactor complex tests for maintainability
3. 📚 Document testing patterns and best practices

### Long-term (This Quarter):
1. 🎯 Implement visual regression testing
2. 🚀 Add performance benchmarking tests
3. 🔐 Expand security testing coverage

## Resources Generated

- 📊 Test report: `test-results/final-report.html`
- 📈 Coverage report: `coverage/index.html`
- 📝 Fix history: `.test-fixes/history.json`
- 🔍 Manual intervention guide: `.test-fixes/manual-guide.md`

---

**Status**: ✅ ALL TESTS PASSING
**Next Run**: Scheduled for next commit
**Maintained By**: Automated test fixing system
```

---

## Your Task: Achieve 100% Test Pass Rate

**Arguments Received**: `$ARGUMENTS`

### Execution Plan:

**Step 1: Setup & Discovery**
```bash
# Detect test frameworks
detect_test_frameworks

# Count tests
count_total_tests

# Determine test command
TEST_COMMAND=$(determine_test_command)
```

**Step 2: Baseline Run**
```bash
# Run tests and capture output
$TEST_COMMAND > baseline_output.log 2>&1

# Parse results
parse_test_results baseline_output.log

# Report baseline
echo "Baseline: $PASSED/$TOTAL_TESTS passing ($PASS_RATE%)"
echo "Goal: Fix $FAILED_COUNT failures"
```

**Step 3: Iterative Fixing**
```bash
ITERATION=0
MAX_ITERATIONS=${MAX_ITERATIONS:-10}

while [ $ITERATION -lt $MAX_ITERATIONS ] && [ $FAILED_COUNT -gt 0 ]; do
  ITERATION=$((ITERATION + 1))

  # Analyze failures
  analyze_failures

  # Generate fixes
  generate_fixes

  # Apply fixes (if --fix flag)
  if [ "$AUTO_FIX" = true ]; then
    apply_fixes

    # Commit progress
    commit_fixes "Iteration $ITERATION"
  fi

  # Run tests again
  $TEST_COMMAND > iter${ITERATION}_output.log 2>&1

  # Check progress
  check_progress
done
```

**Step 4: Final Validation**
```bash
if [ $FAILED_COUNT -eq 0 ]; then
  echo "✅ SUCCESS: 100% pass rate achieved!"
  generate_success_report
else
  echo "⚠️ Partial success: $PASSED/$TOTAL tests passing"
  generate_intervention_guide
fi
```

**Step 5: Reporting**
```bash
# Generate comprehensive report
generate_final_report > test-fix-report.md

# Update statistics
update_test_statistics

# Create artifacts
save_test_artifacts
```

---

## Execution Modes

### 1. Analysis Only (Default)
```bash
/run-all-tests
# Runs tests, analyzes failures, provides fix suggestions
# Does NOT apply fixes automatically
```

### 2. Auto-Fix Mode
```bash
/run-all-tests --fix
# Automatically applies fixes iteratively until 100% or plateau
```

### 3. Targeted Testing
```bash
/run-all-tests src/components
# Only run tests in specific directory
```

### 4. Parallel Execution
```bash
/run-all-tests --parallel
# Run tests in parallel for faster execution
```

### 5. Coverage Mode
```bash
/run-all-tests --coverage
# Run with coverage reporting
```

### 6. Max Iterations
```bash
/run-all-tests --fix --max-iterations=5
# Limit number of fix iterations
```

---

## Success Criteria

✅ **100% test pass rate**
✅ **0 test failures**
✅ **0 test errors**
✅ **All tests stable (not flaky)**
✅ **Tests run successfully in CI**
✅ **Coverage maintained or improved**

---

## Safety Guarantees

✅ **Non-destructive analysis**: Default mode only analyzes, doesn't change code
✅ **Incremental commits**: Each iteration committed separately
✅ **Rollback support**: Easy to revert any iteration
✅ **Validation**: Every fix validated before moving to next
✅ **Progress tracking**: Full visibility into fix attempts
✅ **Manual fallback**: Clear guide when automation can't proceed

---

## Now Execute

Begin comprehensive test suite analysis and iterative fixing process. Run all tests, categorize failures, apply ultrathink intelligence, and achieve 100% pass rate.

**Remember**:
- Safety first: validate every fix
- Progress incrementally: small commits
- Learn from failures: improve fix strategies
- Full transparency: detailed reporting
- Know when to stop: provide manual guide when needed

Let's get those tests green! 🟢
