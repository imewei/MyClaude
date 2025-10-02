# Tutorial 02: Code Quality Improvement

> Master code quality automation with AI-powered analysis and fixes

**Duration**: 45 minutes | **Level**: Beginner | **Prerequisites**: Tutorial 01

---

## Learning Objectives

- Understand quality metrics
- Use auto-fix effectively
- Generate comprehensive tests
- Enforce quality gates
- Measure improvement

---

## Setup: Create Sample Project

```bash
mkdir quality-tutorial
cd quality-tutorial

# Create messy Python code
cat > app.py << 'EOF'
import os, sys, json

def process_data(data):
    result=[]
    for i in range(len(data)):
        if data[i]>0:
            result.append(data[i]*2)
    return result

def calculate(x,y,z):
    temp=x+y
    temp2=temp*z
    return temp2

class DataProcessor:
    def __init__(self,data):
        self.data=data
    def process(self):
        return process_data(self.data)

if __name__=="__main__":
    data=[1,2,3,4,5]
    processor=DataProcessor(data)
    print(processor.process())
EOF
```

---

## Step 1: Analyze Quality (10 min)

### Basic Quality Check

```bash
/check-code-quality --language=python app.py
```

**Issues found:**
- Multiple imports on one line
- Missing docstrings
- Non-Pythonic loops
- Inconsistent spacing
- Missing type hints
- Poor variable names
- No error handling

**Quality Score**: ~45/100

### Detailed Analysis

```bash
/check-code-quality --language=python --analysis=thorough --format=json app.py > quality-report.json
```

Review the detailed report:
```bash
cat quality-report.json
```

---

## Step 2: Auto-Fix Issues (10 min)

### Automatic Fixes

```bash
/check-code-quality --auto-fix app.py
```

**What gets fixed:**
- Import formatting
- Spacing and indentation
- Basic docstrings added
- Pythonic list comprehensions
- Type hints added
- Variable naming improved

**New Quality Score**: ~75/100

### Review Changes

```bash
# See what changed
git diff app.py

# Or compare manually
cat app.py
```

**Improved code:**
```python
import json
import os
import sys

def process_data(data: list) -> list:
    """Process data by doubling positive values."""
    return [item * 2 for item in data if item > 0]

def calculate(x: float, y: float, z: float) -> float:
    """Calculate (x + y) * z."""
    return (x + y) * z

class DataProcessor:
    """Process data using configured transformation."""

    def __init__(self, data: list):
        """Initialize with data."""
        self.data = data

    def process(self) -> list:
        """Process the data."""
        return process_data(self.data)

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    processor = DataProcessor(data)
    print(processor.process())
```

---

## Step 3: Generate Comprehensive Tests (10 min)

### Generate Unit Tests

```bash
/generate-tests --type=unit --coverage=90 app.py
```

**Generated tests:**
- `test_process_data_positive_values()`
- `test_process_data_negative_values()`
- `test_process_data_empty_list()`
- `test_calculate_basic()`
- `test_calculate_zero_values()`
- `test_data_processor_init()`
- `test_data_processor_process()`

### Run Tests

```bash
/run-all-tests --auto-fix --coverage
```

**Results:**
```
✓ 7 tests passed
✓ Coverage: 95%
```

### Add Edge Case Tests

```bash
# Generate additional edge case tests
/generate-tests --type=unit --focus=edge-cases app.py
```

---

## Step 4: Advanced Refactoring (10 min)

### Apply Modern Patterns

```bash
/refactor-clean --patterns=modern --implement app.py
```

**Improvements:**
- Context managers added
- Type annotations enhanced
- Error handling added
- Configuration externalized
- Logging added

### Security Refactoring

```bash
/refactor-clean --patterns=security --implement app.py
```

**Security improvements:**
- Input validation added
- Safe JSON parsing
- Exception handling
- Security best practices

---

## Step 5: Complete Quality Workflow (5 min)

### Run Complete Workflow

```bash
/multi-agent-optimize --mode=review --focus=quality --implement
```

**This executes:**
1. Deep quality analysis
2. Auto-fix all fixable issues
3. Generate comprehensive tests
4. Run tests with coverage
5. Security scan
6. Performance check
7. Documentation update
8. Final verification

**Results:**
```
Initial Quality Score: 45/100
Final Quality Score: 94/100

Improvements:
✓ Code style: 45 → 98
✓ Complexity: 50 → 90
✓ Type hints: 0 → 100
✓ Documentation: 30 → 95
✓ Security: 60 → 95
✓ Test coverage: 0 → 97%
```

---

## Step 6: Enforce Quality Gates

### Setup Quality Configuration

Create `.claude-commands.yml`:
```yaml
quality:
  min_score: 85
  min_coverage: 90
  strict_mode: true

  style:
    guide: pep8
    max_line_length: 100
    enforce_docstrings: true

  complexity:
    max_cyclomatic: 10
    max_cognitive: 15

  security:
    scan: true
    fail_on: ["high", "critical"]

  type_hints:
    required: true
    strict: false
```

### Setup Pre-Commit Hook

```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
/check-code-quality --auto-fix --validate
if [ $? -ne 0 ]; then
    echo "❌ Quality gate failed"
    exit 1
fi
echo "✓ Quality gate passed"
EOF

chmod +x .git/hooks/pre-commit
```

### Test Quality Gate

```bash
# Make a poor-quality change
echo "def bad_function(): pass" >> app.py

# Try to commit
git add .
git commit -m "test"

# Quality gate will block commit and auto-fix
```

---

## Practice Exercises

### Exercise 1: Fix Legacy Code

```python
# Create legacy code
cat > legacy.py << 'EOF'
def f(x):
    r=[]
    for i in x:
        if i%2==0:
            r.append(i)
    return r

def g(a,b):
    c=a
    for i in range(b):
        c+=a
    return c
EOF

# Task: Improve quality to 90+
# 1. Run quality check
# 2. Auto-fix
# 3. Generate tests
# 4. Verify with double-check
```

**Solution:**
```bash
/check-code-quality --auto-fix legacy.py
/generate-tests --coverage=90 legacy.py
/run-all-tests --auto-fix
/double-check "legacy.py has quality score ≥ 90"
```

### Exercise 2: Test-Driven Quality

```bash
# Generate tests first
/generate-tests --type=unit --coverage=95 legacy.py

# Then improve code to pass tests
/check-code-quality --auto-fix legacy.py
/run-all-tests --auto-fix

# Optimize
/optimize --implement legacy.py

# Verify
/double-check "all tests pass and quality ≥ 90"
```

---

## Key Concepts

### Quality Metrics

**Style (20%)**
- PEP 8 compliance
- Naming conventions
- Import organization
- Whitespace usage

**Complexity (20%)**
- Cyclomatic complexity
- Cognitive complexity
- Nesting depth
- Function length

**Type Hints (15%)**
- Coverage percentage
- Correctness
- Completeness

**Documentation (15%)**
- Docstring coverage
- Docstring quality
- Comment clarity

**Security (15%)**
- Vulnerability scan
- Best practices
- Input validation

**Maintainability (15%)**
- Code smells
- Duplication
- Modularity

### Auto-Fix Capabilities

**What can be auto-fixed:**
✓ Style violations
✓ Import organization
✓ Basic type hints
✓ Simple refactorings
✓ Documentation scaffolding
✓ Common patterns

**What requires manual review:**
✗ Complex algorithms
✗ Architecture changes
✗ Business logic
✗ Design patterns
✗ Performance optimization

---

## Advanced Techniques

### Custom Quality Rules

```yaml
# .claude-commands.yml
quality:
  custom_rules:
    - name: no-print-statements
      pattern: "print\\("
      severity: warning
      message: "Use logging instead"

    - name: require-type-hints
      check: type_hint_coverage
      threshold: 100
      severity: error
```

### Quality Monitoring

```bash
# Generate quality report
/check-code-quality --format=html --report > quality-report.html

# Track over time
echo "$(date),$(quality-score)" >> quality-history.csv

# Visualize trends
# Use quality-report.html
```

---

## CI/CD Integration

### Setup CI Quality Gate

```bash
/ci-setup --platform=github --type=enterprise --security
```

This creates `.github/workflows/quality.yml`:
```yaml
name: Quality Gate
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Quality Check
        run: /check-code-quality --auto-fix --validate
      - name: Tests
        run: /run-all-tests --coverage --min-coverage=90
      - name: Security Scan
        run: /check-code-quality --focus=security --fail-on=high
```

---

## Summary

### What You Learned

✓ Analyze code quality comprehensively
✓ Use auto-fix to improve code automatically
✓ Generate comprehensive test suites
✓ Enforce quality gates in workflow
✓ Integrate quality checks in CI/CD
✓ Measure and track quality improvements

### Quality Improvement Process

```
1. Analyze   → /check-code-quality
2. Fix       → --auto-fix
3. Test      → /generate-tests
4. Validate  → /run-all-tests
5. Verify    → /double-check
6. Commit    → /commit --validate
```

### Best Practices

1. **Run quality checks frequently** - Pre-commit hooks
2. **Aim for 85+ score** - Maintainable threshold
3. **90%+ test coverage** - Comprehensive testing
4. **Use workflows** - Automated quality improvements
5. **Track metrics** - Monitor improvements over time
6. **Enforce gates** - Prevent quality regression

---

## Next Steps

- **[Tutorial 03: Performance Optimization](tutorial-03-performance.md)**
- **[Tutorial 04: Workflows](tutorial-04-workflows.md)**
- **[User Guide: Code Quality Section](../docs/USER_GUIDE.md#code-quality)**

---

**Congratulations!** You've mastered code quality automation.

**Version**: 1.0.0 | **Duration**: 45 minutes | **Level**: Beginner