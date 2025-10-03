# Tutorial 01: Check and Improve Code Quality

**Duration:** 15 minutes
**Difficulty:** Beginner
**Prerequisites:** Basic Python knowledge, framework installed

## What You'll Learn

- How to check code quality
- Using agents for quality analysis
- Applying automatic fixes
- Verifying improvements

## Scenario

You have a Python project with some code quality issues. You want to identify and fix them automatically.

## Setup

```bash
# Create a sample project
mkdir quality-tutorial
cd quality-tutorial

# Create a sample Python file with issues
cat > example.py << 'EOF'
import os
import sys
import json
import requests
from typing import List

def calculate_total(items):
    total=0
    for item in items:
        total=total+item
    return total

def process_data(data, flag=True):
    result=[]
    for i in range(len(data)):
        if flag==True:
            result.append(data[i]*2)
    return result

class DataProcessor:
    def __init__(self,data):
        self.data=data

    def process(self):
        return [x*2 for x in self.data]

if __name__=="__main__":
    items=[1,2,3,4,5]
    print(calculate_total(items))
EOF
```

## Step 1: Initial Analysis

Run quality check in dry-run mode:

```bash
/check-code-quality --dry-run
```

**Expected Output:**

```
🔍 Analyzing codebase...

🤖 Agents selected (auto):
  - code-quality-master
  - systems-architect
  - documentation-architect

📊 Issues Found: 15

HIGH Priority (5):
  - 3 unused imports (os, sys, json)
  - 2 functions missing docstrings

MEDIUM Priority (7):
  - Inconsistent spacing around operators
  - Using 'flag==True' instead of 'flag'
  - Using range(len()) anti-pattern
  - Missing type hints on return values
  - Using 'total=total+item' instead of '+='

LOW Priority (3):
  - Line spacing issues
  - Naming convention suggestions
  - Missing blank lines

✅ All issues can be auto-fixed

⏱️  Duration: 1.2s
```

## Step 2: Apply Automatic Fixes

Now apply the fixes:

```bash
/check-code-quality --auto-fix
```

**What Happens:**
- Removes unused imports
- Fixes spacing and formatting
- Applies Python best practices
- Adds docstrings where possible
- Updates code patterns

## Step 3: Review Changes

Check what changed:

```bash
cat example.py
```

**Fixed Code:**

```python
from typing import List


def calculate_total(items: List[int]) -> int:
    """Calculate the sum of items in a list.

    Args:
        items: List of integers to sum

    Returns:
        The total sum of all items
    """
    total = 0
    for item in items:
        total += item
    return total


def process_data(data: List[int], flag: bool = True) -> List[int]:
    """Process data by doubling values when flag is True.

    Args:
        data: List of integers to process
        flag: Whether to process the data

    Returns:
        Processed list with doubled values
    """
    result = []
    if flag:
        result = [item * 2 for item in data]
    return result


class DataProcessor:
    """Process data by doubling values."""

    def __init__(self, data: List[int]):
        """Initialize processor with data.

        Args:
            data: List of integers to process
        """
        self.data = data

    def process(self) -> List[int]:
        """Process the data.

        Returns:
            List with doubled values
        """
        return [x * 2 for x in self.data]


if __name__ == "__main__":
    items = [1, 2, 3, 4, 5]
    print(calculate_total(items))
```

## Step 4: Advanced Analysis

Use more agents for deeper analysis:

```bash
/check-code-quality --agents=all --detailed
```

**Additional Insights:**

```
🤖 Agents: 23 (all with orchestration)

📊 Advanced Analysis:

Architecture (systems-architect):
  ✅ Good separation of concerns
  💡 Consider extracting DataProcessor to separate module

Performance (scientific-computing-master):
  ✅ Efficient implementations
  💡 Consider numpy for large arrays

Documentation (documentation-architect):
  ✅ All functions documented
  💡 Add module-level docstring
  💡 Add usage examples

Quality Score: 92/100 (Excellent)
  Before: 67/100
  Improvement: +25 points

⏱️  Duration: 3.8s
```

## Step 5: Generate Tests

Ensure quality with tests:

```bash
/generate-tests example.py --type=unit
```

**Generated Tests:**

```python
# tests/test_example.py
import pytest
from example import calculate_total, process_data, DataProcessor


class TestCalculateTotal:
    def test_empty_list(self):
        assert calculate_total([]) == 0

    def test_single_item(self):
        assert calculate_total([5]) == 5

    def test_multiple_items(self):
        assert calculate_total([1, 2, 3, 4, 5]) == 15

    def test_negative_numbers(self):
        assert calculate_total([-1, -2, -3]) == -6


class TestProcessData:
    def test_flag_true(self):
        assert process_data([1, 2, 3], True) == [2, 4, 6]

    def test_flag_false(self):
        assert process_data([1, 2, 3], False) == []

    def test_empty_list(self):
        assert process_data([], True) == []


class TestDataProcessor:
    def test_process(self):
        processor = DataProcessor([1, 2, 3])
        assert processor.process() == [2, 4, 6]

    def test_empty_data(self):
        processor = DataProcessor([])
        assert processor.process() == []
```

## Step 6: Run Tests

```bash
/run-all-tests --coverage
```

**Output:**

```
🧪 Running tests...

✅ All tests passed: 11/11

📈 Coverage: 100% (example.py)

⏱️  Duration: 0.8s
```

## Step 7: Complete Workflow

Put it all together:

```bash
# Complete quality improvement pipeline
/check-code-quality --auto-fix && \
/generate-tests --type=unit --coverage=95 && \
/run-all-tests --coverage && \
/update-docs --type=readme
```

## What You Learned

You now know how to:

✅ Check code quality with dry-run
✅ Apply automatic fixes safely
✅ Use different agent groups
✅ Generate tests for quality assurance
✅ Run tests with coverage
✅ Create a complete quality workflow

## Next Steps

- **[Tutorial 02: Optimization](tutorial-02-optimization.md)** - Improve performance
- **[Tutorial 03: Documentation](tutorial-03-documentation.md)** - Generate docs
- **[Tutorial 04: Testing](tutorial-04-testing.md)** - Advanced testing

## Common Issues

### Issue: "No issues found"

Your code is already high quality! Try:
```bash
/check-code-quality --detailed --agents=all
```

### Issue: "Some fixes require manual review"

Some complex issues can't be auto-fixed:
```bash
/check-code-quality --interactive
```

### Issue: "Tests failing after fixes"

Very rare, but if it happens:
```bash
# Framework creates automatic backup
git diff  # Review changes
git checkout -- .  # Revert if needed
```

## Practice Exercise

Create your own Python file with issues and fix them:

```python
# Create exercise.py with intentional issues
cat > exercise.py << 'EOF'
import time,datetime,os
def my_function(x,y,z):
    result=x+y+z
    return result
class MyClass:
    def __init__(self,value):
        self.value=value
EOF

# Run the full workflow
/check-code-quality --auto-fix && \
/generate-tests exercise.py && \
/run-all-tests
```

---

**Congratulations!** You've completed Tutorial 01. → [Tutorial 02](tutorial-02-optimization.md)