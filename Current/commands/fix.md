---
description: Systematically debug and fix errors, issues, and failures
allowed-tools: Bash(git:*), Bash(npm:*), Bash(pytest:*), Bash(cargo:*)
argument-hint: <error-or-issue> [--trace] [--test]
color: red
agents:
  primary:
    - code-quality
  conditional:
    - agent: systems-architect
      trigger: complexity > 15 OR pattern "architecture|design.*pattern|system.*design|scalability"
    - agent: neural-architecture-engineer
      trigger: pattern "torch|pytorch|tensorflow|keras|neural.*network"
    - agent: jax-pro
      trigger: pattern "jax|flax|@jit|@vmap|@pmap|grad\\("
    - agent: devops-security-engineer
      trigger: pattern "security|vulnerability|ci/cd|deploy.*|docker|kubernetes|container"
  orchestrated: false
---

# Error Resolution System

## Context

**Error/Issue**: $ARGUMENTS

- Recent activity: !`git log --oneline -5 2>/dev/null`
- Current branch: !`git branch --show-current 2>/dev/null`
- Uncommitted changes: !`git diff --stat 2>/dev/null | tail -1`
- Last test run: !`find . -name "*.test.*" -newer ~/.bash_history 2>/dev/null | wc -l` recent tests

## Your Task

**Systematic debugging methodology**:

### 1. Information Gathering
- Complete error message + stack trace
- Timing: when does it occur?
- Environment: dev/staging/prod?
- Reproducibility: always/intermittent?
- Recent changes correlation

### 2. Reproduce Reliably
Create minimal reproduction:
```python
# Minimal test case
def test_error():
    # Simplest code that reproduces issue
    result = function_with_bug(input)
    assert result == expected  # Should fail
```

### 3. Root Cause Analysis
**Common patterns**:
- **Null/undefined**: Add null checks, use optional chaining
- **Type mismatch**: Validate types, add type hints
- **Race condition**: Add synchronization, use locks
- **Resource leak**: Ensure cleanup (finally/defer/RAII)
- **Logic error**: Review algorithm, add assertions
- **Dependency issue**: Check versions, update/pin deps

### 4. Solution Implementation
**Fix patterns**:
```python
# Before: Crashes on None
result = data.process()

# After: Safe null handling
result = data.process() if data else default_value

# Before: Type error
value = int(user_input)

# After: Validated conversion
try:
    value = int(user_input)
except ValueError:
    return error_response("Invalid number")
```

### 5. Validation
```bash
# Run affected tests
npm test -- --grep "affected"
pytest tests/test_module.py -v

# Full regression
npm test
pytest

# Verify fix locally
# Review changed code
# Check for side effects
```

### 6. Prevention
- Add test for this bug
- Add input validation
- Improve error messages
- Document assumptions
- Add type hints/checks

## Execution

1. **Parse error** → identify error type
2. **Locate source** → file:line from stack trace
3. **Hypothesize causes** → list 3-5 possible causes
4. **Test hypotheses** → eliminate until root cause found
5. **Implement fix** → minimal change to fix
6. **Validate thoroughly** → tests pass, no regressions
7. **Document** → comment explaining why fix works

**Output**: Clear explanation of root cause, fix applied, and prevention measures added
