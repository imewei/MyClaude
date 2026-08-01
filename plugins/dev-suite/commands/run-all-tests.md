---
name: run-all-tests
command: /run-all-tests
description: Iteratively run and fix all tests until zero failures with AI-driven RCA
argument-hint: "[test-path] [--fix] [--max-iterations=10] [--parallel] [--coverage]"
execution-modes: {quick: "30min-1h", standard: "2-4h", enterprise: "1-2d"}
workflow-type: iterative
allowed-tools: [Bash, Read, Edit, Task, Bash(uv:*), ScheduleWakeup]
---

# Iterative Test Execution & Auto-Fix

$ARGUMENTS

## Modes

| Mode | Time | Iterations | Coverage | Use |
|------|------|------------|----------|-----|
| Quick | 30min-1h | Max 3 | Basic | Specific test fixes |
| Standard | 2-4h | Max 10 | >80% | Pre-commit, CI/CD |
| Enterprise | 1-2d | Max 20 | >90% | Release validation |

No mode runs unbounded — rewriting an assertion or a snapshot to match
current (possibly broken) output makes the loop's own exit condition true
regardless of whether the change was correct, so an unlimited iteration
count is an unlimited number of chances to launder a bug into "passing".

Options: `--fix`, `--max-iterations=N`, `--parallel`, `--coverage`,
`--allow-suppression` (required to rewrite assertions/snapshots — see
Failure Analysis below)

## Framework Auto-Detection

| Lang | Detection | Command |
|------|-----------|---------|
| JavaScript | package.json has jest/vitest | `npm test` |
| Python | pytest installed | `pytest` |
| Rust | Cargo.toml | `cargo test` |
| Go | go.mod | `go test ./...` |
| Java | pom.xml/build.gradle | `mvn test`/`./gradlew test` |

## Workflow

1. **Baseline**: Total, passed, failed, skipped, pass rate
2. **Failure Analysis**:
   - Import/Module (20%): Install deps, fix paths
   - Assertion (35%): Fix the code by default. Rewriting the *expectation*
     instead requires `--allow-suppression` — it makes the test pass without
     proving the code is right, and is tracked separately from real fixes.
   - Runtime (25%): Fix null refs, types
   - Async/Timing (10%): Add awaits, proper waits
   - Setup/Teardown (5%): Fix fixtures, mocks
   - Snapshot (3%): Requires `--allow-suppression`. Never counted as a fix —
     tracked as a suppression-class edit that needs manual review of the diff.
   - Env/Config (2%): Set vars, start services

3. **RCA** per failure: What testing? Where fail? What changed (git blame)? Dependencies? Flaky (run 10x)?

4. **Iterative Fix Loop**:
```text
WHILE failures > 0 AND iteration < max:   # max is always finite — see Modes
  1. Run tests
  2. Parse failures
  3. If zero → SUCCESS
  4. If no progress → STOP
  5. Fix high-priority failures (suppression-class edits only with --allow-suppression)
  6. Commit fixes, recording which were suppression-class
  7. Repeat
```

5. **Fix Priority**: Import → Setup/fixture → Assertions → Async → Complex logic

6. **Coverage & Gates**:
   - Python: `pytest --cov=src --cov-report=html`
   - JS: `npm test -- --coverage`
   - Rust: `cargo tarpaulin --out Html`
   - Go: `go test -coverprofile=coverage.out ./...`
   - Gates: 100% pass, ≥80% line coverage (≥90% enterprise), 0 failures

## Exit Criteria

| Condition | Result |
|-----------|--------|
| All pass, no suppression-class edits | ✅ SUCCESS |
| All pass, suppression-class edits made | ✅ PASS (review diff) |
| Max iterations | ⚠️ PARTIAL |
| No progress (2 iterations) | ⏸️ PLATEAU |
| Regression | ❌ STOP — revert manually (`git revert`/`git reset`); no auto-rollback |

## Common Scenarios

| Scenario | Solution |
|----------|----------|
| Flaky | Run 10x to detect, add isolation |
| Environment | Set NODE_ENV=test, start test DB |
| Parallel conflicts | Unique data, DB transactions |

## Output

```text
Summary
  Total: 120
  Pass: 100%
  Iterations: 4
  Auto-fixed: 86
  Suppression-class edits: 12 (required --allow-suppression; review diff)

Fixes
  Import: 28
  Async: 22
  Assertions: 35
  Setup: 8

Suppression-class (not counted as fixes)
  Snapshots updated: 12

Coverage: 78% → 82% (+4%)
Status: ✅ PASS — 12 suppression-class edits made, review diff before trusting this run
```

If no suppression-class edits were made this run, `Status: ✅ ALL PASS` is
accurate as-is; the qualifier only appears when it's true.

## Safety

- ✅ Non-destructive without `--fix`: dry-run analysis only
- ✅ Incremental commits per iteration
- ✅ Suppression-class edits (assertion rewrites, snapshot updates) require
  explicit `--allow-suppression` and are never silently counted as fixes
- ✅ Tests re-run every iteration before continuing
- ⚠️ No automated rollback — commits are incremental so `git revert`/`git
  reset` work, but nothing here reverts automatically on regression
- ✅ Manual fallback guide
