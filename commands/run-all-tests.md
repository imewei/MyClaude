# Run All Tests

## Overview

Executes comprehensive test suite validation with automated failure resolution to ensure code quality and prevent regressions.

## Command

Use the test-runner subagent to run ALL tests in the application's test suite and systematically fix any failures until achieving 100% pass rate.

## Process Flow

<test_execution>
  <phase name="discovery">
    ACTION: Use test-runner subagent
    REQUEST: "Run the complete test suite with detailed failure analysis"
    SCOPE: All test categories (unit, integration, performance, validation)
  </phase>

  <phase name="analysis">
    WAIT: For comprehensive test-runner analysis and failure categorization
    EVALUATE: Test results, failure patterns, and root cause identification
    PRIORITIZE: Critical failures before non-critical issues
  </phase>

  <phase name="resolution">
    PROCESS: Systematically fix reported failures by category
    VALIDATE: Each fix doesn't introduce new regressions
    ITERATE: Until all tests achieve passing status
  </phase>
</test_execution>

## Requirements

- **Pass Rate**: 100% (all tests must pass)
- **Coverage**: Complete test suite execution
- **Quality**: No test skips or suppressions
- **Stability**: Consistent results across runs

## Execution Strategy

### 1. Initial Test Run
```
Priority: Run entire test suite first
Output: Comprehensive failure report with categorization
Focus: Identify failure patterns and dependencies
```

### 2. Systematic Resolution
```
Order: Fix tests in dependency order
Approach: Address root causes, not symptoms
Validation: Verify fixes don't break other tests
Documentation: Track changes and rationale
```

### 3. Final Verification
```
Confirmation: Re-run complete suite
Requirement: 100% pass rate achieved
Quality: All test categories passing consistently
```

## Failure Handling Protocol

<failure_categories>
  <critical priority="1">
    - Core functionality breaks
    - Security vulnerabilities
    - Data corruption risks
  </critical>

  <high priority="2">
    - Integration test failures
    - Performance regressions
    - API contract violations
  </high>

  <medium priority="3">
    - Unit test edge cases
    - Documentation inconsistencies
    - Non-critical warnings
  </medium>
</failure_categories>

## Success Criteria

- ✅ **Zero Test Failures**: All tests pass without exceptions
- ✅ **No Regressions**: Existing functionality preserved
- ✅ **Clean Output**: No errors, warnings, or deprecations
- ✅ **Performance**: Test suite completes within acceptable timeframe
- ✅ **Repeatability**: Consistent results across multiple runs

## Usage Instructions

1. Execute this command when:
   - Before major releases or deployments
   - After significant code changes
   - Before merging feature branches
   - During continuous integration validation

2. Expected workflow:
   - Command triggers comprehensive test execution
   - Test-runner provides detailed failure analysis
   - Systematic resolution of all identified issues
   - Final validation confirms 100% pass rate

3. Completion verification:
   - All test categories show green/passing status
   - No skipped or ignored tests remain
   - Performance benchmarks meet expectations
   - Code quality metrics satisfy standards