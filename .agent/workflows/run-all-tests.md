---
description: Iteratively run and fix all tests until zero failures with AI-driven
  RCA
triggers:
- /run-all-tests
- workflow for run all tests
version: 1.0.7
command: /run-all-tests
argument-hint: '[test-path] [--fix] [--max-iterations=10] [--parallel] [--coverage]'
execution_modes:
  quick: 30min-1h
  standard: 2-4h
  enterprise: 1-2d
workflow_type: iterative
color: blue
allowed-tools: [Bash, Read, Edit, Task]
---


# Iterative Test Execution & Auto-Fix

$ARGUMENTS

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

## Modes

| Mode | Time | Iterations | Coverage | Use |
|------|------|------------|----------|-----|
| Quick | 30min-1h | Max 3 | Basic | Specific test fixes |
| Standard | 2-4h | Max 10 | >80% | Pre-commit, CI/CD |
| Enterprise | 1-2d | Unlimited | >90% | Release validation |

Options: `--fix`, `--max-iterations=N`, `--parallel`, `--coverage`

## Framework Auto-Detection

| Lang | Detection | Command | Parallel Flag |
|------|-----------|---------|---------------|
| JavaScript | package.json | `npm test` | `--maxWorkers=N` |
| Python | pytest | `pytest` | `-n auto` |
| Rust | Cargo.toml | `cargo test` | `--jobs N` |
| Go | go.mod | `go test ./...` | `-p N` |
| Java | pom.xml | `mvn test` | `-T 1C` |

## Workflow

1. **Baseline**: Total, passed, failed, skipped, pass rate

2. **Failure Analysis** (Parallel Execution):
   > **Orchestration Note**: Group failures and analyze RCA concurrently.
   - Import/Module (20%): Install deps, fix paths
   - Assertion (35%): Fix code or expectation
   - Runtime (25%): Fix null refs, types
   - Async/Timing (10%): Add awaits, proper waits
   - Setup/Teardown (5%): Fix fixtures, mocks
   - Snapshot (3%): Update if intentional
   - Env/Config (2%): Set vars, start services

3. **RCA** per failure: What testing? Where fail? What changed (git blame)? Dependencies? Flaky (run 10x)?

4. **Iterative Fix Loop**:
```
WHILE failures > 0 AND iteration < max:
  1. Run tests (Parallel)
  2. Parse failures
  3. If zero → SUCCESS
  4. If no progress → STOP
  5. Fix high-priority failures (Parallel Streams)
  6. Commit fixes
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
| All pass | ✅ SUCCESS |
| Max iterations | ⚠️ PARTIAL |
| No progress (2 iterations) | ⏸️ PLATEAU |
| Regression | ❌ ROLLBACK |

## Common Scenarios

| Scenario | Solution |
|----------|----------|
| Flaky | Run 10x to detect, add isolation |
| Environment | Set NODE_ENV=test, start test DB |
| Parallel conflicts | Unique data, DB transactions |

## Output

```
Summary
  Total: 120
  Pass: 100%
  Iterations: 4
  Auto-fixed: 98

Fixes
  Import: 28
  Async: 22
  Assertions: 35
  Snapshots: 12
  Setup: 8

Coverage: 78% → 82% (+4%)
Status: ✅ ALL PASS
```

## Safety

- ✅ Non-destructive by default
- ✅ Incremental commits per iteration
- ✅ Easy rollback
- ✅ Validation before next iteration
- ✅ Manual fallback guide
