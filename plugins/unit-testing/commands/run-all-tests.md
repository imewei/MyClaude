---
version: "1.0.5"
command: /run-all-tests
description: Iteratively run and fix all tests until zero failures with AI-driven root cause analysis
argument-hint: [test-path] [--fix] [--max-iterations=10] [--parallel] [--coverage]
execution_modes:
  quick:
    duration: "30min-1h"
    scope: "Single test file or small suite"
    iterations: "Max 3"
  standard:
    duration: "2-4h"
    scope: "Full test suite"
    iterations: "Max 10"
    coverage: ">80%"
  enterprise:
    duration: "1-2d"
    scope: "Entire codebase"
    iterations: "Until 100% pass"
    coverage: ">90%"
workflow_type: "iterative"
color: blue
allowed-tools: Bash(npm:*), Bash(pytest:*), Bash(cargo:*), Bash(go:*), Bash(mvn:*), Bash(gradle:*)
agents:
  primary:
    - test-automator
  conditional:
    - agent: debugger
      trigger: argument "--fix" OR pattern "fix|debug|failure"
    - agent: code-quality
      trigger: argument "--coverage" OR pattern "coverage|quality"
    - agent: hpc-numerical-coordinator
      trigger: pattern "numpy|scipy|pytest.*numerical"
  orchestrated: false
---

# Iterative Test Execution & Auto-Fix

Systematic test execution with AI-driven failure analysis and iterative fixing.

## Target

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Iterations | Coverage | Use Case |
|------|----------|------------|----------|----------|
| Quick | 30min-1h | Max 3 | Basic | Specific test fixes |
| Standard | 2-4h | Max 10 | >80% | Pre-commit, CI/CD |
| Enterprise | 1-2d | Unlimited | >90% | Release validation |

**Options:** `--fix`, `--max-iterations=N`, `--parallel`, `--coverage`

---

## Phase 0: Framework Discovery

### Auto-Detection

| Language | Detection | Command |
|----------|-----------|---------|
| JavaScript | `package.json` has jest/vitest | `npm test` |
| Python | pytest installed | `pytest` |
| Rust | `Cargo.toml` exists | `cargo test` |
| Go | `go.mod` exists | `go test ./...` |
| Java | `pom.xml` / `build.gradle` | `mvn test` / `./gradlew test` |

---

## Phase 1: Baseline Test Run

Execute tests, capture baseline metrics:
- Total tests, passed, failed, skipped
- Pass rate percentage
- Identify failure count for goal

---

## Phase 2: Failure Analysis

### Failure Categories

| Category | Frequency | Fix Strategy |
|----------|-----------|--------------|
| Import/Module Errors | 20% | Install deps, fix paths |
| Assertion Failures | 35% | Fix code or update expectation |
| Runtime Errors | 25% | Fix null refs, type errors |
| Async/Timing Issues | 10% | Add awaits, proper waits |
| Setup/Teardown | 5% | Fix fixtures, mocks |
| Snapshot Mismatches | 3% | Update if intentional |
| Environment/Config | 2% | Set env vars, start services |

### Root Cause Analysis

For each failure:
1. What is test verifying?
2. Where exactly does it fail?
3. What changed recently? (git blame)
4. What are the dependencies?
5. Is it flaky? (run 10x to check)

---

## Phase 3: Iterative Fix Loop

### Workflow

```
WHILE failures > 0 AND iteration < max:
  1. Run tests
  2. Parse failures
  3. If zero failures → SUCCESS
  4. If no progress → STOP
  5. Analyze and fix highest-priority failures
  6. Commit fixes
  7. Repeat
```

### Fix Priority

1. Import errors (quick wins)
2. Setup/fixture errors (unblock other tests)
3. Assertion failures (largest category)
4. Async issues (need careful handling)
5. Complex logic (most time-consuming)

---

## Phase 4: Coverage & Quality Gates

### Coverage Commands

| Language | Command |
|----------|---------|
| Python | `pytest --cov=src --cov-report=html` |
| JavaScript | `npm test -- --coverage` |
| Rust | `cargo tarpaulin --out Html` |
| Go | `go test -coverprofile=coverage.out ./...` |

### Quality Gates

| Gate | Threshold |
|------|-----------|
| Pass Rate | 100% |
| Line Coverage | ≥80% (≥90% enterprise) |
| Failures | 0 |

---

## Phase 5: Final Report

```
Summary
  Total Tests:   120
  Pass Rate:     100%
  Iterations:    4
  Auto-Fixed:    98 tests

Fixes Applied
  Import errors:     28
  Async fixes:       22
  Assertions:        35
  Snapshots:         12
  Setup fixes:        8

Coverage: 78% → 82% (+4%)
Status: ✅ ALL TESTS PASSING
```

---

## Common Scenarios

| Scenario | Solution |
|----------|----------|
| Flaky tests | Run 10x to detect, add isolation |
| Environment issues | Set NODE_ENV=test, start test DB |
| Parallel conflicts | Use unique test data, DB transactions |

---

## Exit Criteria

| Condition | Result |
|-----------|--------|
| All tests passing | ✅ SUCCESS |
| Max iterations reached | ⚠️ PARTIAL |
| No progress for 2 iterations | ⏸️ PLATEAU |
| Regression detected | ❌ ROLLBACK |

---

## Safety Guarantees

- ✅ Non-destructive analysis by default
- ✅ Incremental commits per iteration
- ✅ Easy rollback support
- ✅ Validation before next iteration
- ✅ Manual fallback guide when needed

---

## Examples

```bash
# Quick: Single file
/run-all-tests src/utils --max-iterations=3

# Standard: Full suite with auto-fix
/run-all-tests --fix --parallel --coverage

# Enterprise: Complete validation
/run-all-tests --fix --max-iterations=100 --coverage
```
