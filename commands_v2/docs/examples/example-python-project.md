# Example: Complete Python Project Workflow

Real-world example of using the Claude Code Command Executor on a Python data science project.

## Project Overview

**Project:** Data Analysis Library
**Language:** Python
**Size:** ~5000 lines, 42 files
**Framework:** NumPy, Pandas, scikit-learn
**Goal:** Improve quality, optimize performance, add tests, generate docs

## Initial State

```
data-analysis-lib/
├── src/
│   ├── algorithms/
│   │   ├── sorting.py (complex algorithms)
│   │   ├── search.py
│   │   └── optimize.py
│   ├── processing/
│   │   ├── cleaner.py
│   │   ├── transformer.py
│   │   └── validator.py
│   ├── models/
│   │   ├── linear.py
│   │   ├── ensemble.py
│   │   └── neural.py
│   └── utils/
│       ├── io.py
│       ├── math.py
│       └── viz.py
├── tests/ (minimal coverage)
├── docs/ (outdated)
└── requirements.txt
```

## Phase 1: Initial Assessment

### Step 1: Code Quality Check

```bash
cd data-analysis-lib
/check-code-quality --dry-run --report
```

**Results:**
```
📊 Code Quality Report:
  Files analyzed: 42
  Quality score: 68/100 (Needs Improvement)

Issues found: 87
  HIGH (23):
    - 15 unused imports
    - 8 functions without docstrings
  MEDIUM (42):
    - 25 complexity warnings (> cyclomatic complexity 10)
    - 12 naming violations
    - 5 missing type hints
  LOW (22):
    - 22 formatting issues

🤖 Agents used: 5 (auto-selected)
  - code-quality-master (lead)
  - scientific-computing-master (NumPy/Pandas analysis)
  - systems-architect (architecture review)
  - documentation-architect (doc analysis)
  - multi-agent-orchestrator (coordination)

⏱️  Duration: 3.8s
```

### Step 2: Performance Analysis

```bash
/optimize --profile --detailed src/
```

**Results:**
```
🔍 Performance Analysis:

Hot Spots:
  1. src/algorithms/sorting.py:45
     - Bubble sort O(n²) on large arrays
     - Est. improvement: 100x with merge sort or NumPy

  2. src/processing/transformer.py:112
     - Iterative pandas operations
     - Est. improvement: 10x with vectorization

  3. src/utils/io.py:78
     - Reading file line-by-line
     - Est. improvement: 5x with bulk read

🤖 Agents used: 8 (scientific agents auto-selected)
  - scientific-computing-master (lead)
  - jax-pro (numerical optimization)
  - neural-networks-master (ML model optimization)
  - data-professional (pandas optimization)

⏱️  Duration: 6.2s
```

## Phase 2: Quality Improvement

### Step 1: Fix Quality Issues

```bash
/check-code-quality --auto-fix
```

**Applied Fixes:**
- Removed 15 unused imports
- Fixed 22 formatting issues
- Added 8 basic docstrings
- Updated 12 naming violations
- Added type hints to simple functions

**Quality Score: 68 → 78** (+10 points)

### Step 2: Clean Codebase

```bash
/clean-codebase --imports --dead-code --backup --dry-run
```

**Preview Results:**
```
🧹 Cleanup Analysis:

Unused Imports: 23 found
  src/algorithms/sorting.py: 5
  src/processing/cleaner.py: 8
  ... (others)

Dead Code: 8 blocks
  src/utils/old_helpers.py: Entire file unused (234 lines)
  src/processing/transformer.py: Function 'deprecated_transform' (67 lines)

💾 Space Savings: 1,234 lines (24.7% of codebase)
```

**Apply Cleanup:**
```bash
/clean-codebase --imports --dead-code --backup
```

**Results:**
- Removed 23 unused imports
- Deleted 8 dead code blocks
- Saved 1,234 lines
- Created backup in ~/.claude/backups/

## Phase 3: Performance Optimization

### Step 1: Apply Optimizations

```bash
/optimize --implement --category=algorithm src/algorithms/
```

**Changes Applied:**

**sorting.py:**
```python
# Before
def sort_data(arr):
    # Bubble sort - O(n²)
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# After (optimized)
def sort_data(arr):
    """Sort array efficiently using NumPy.

    Args:
        arr: Input array to sort

    Returns:
        Sorted array
    """
    return np.sort(arr)  # O(n log n) - 100x faster for large arrays
```

### Step 2: Optimize Data Processing

```bash
/optimize --implement --category=memory src/processing/
```

**Changes Applied:**

**transformer.py:**
```python
# Before
def transform_column(df, column):
    result = []
    for i in range(len(df)):
        result.append(df[column].iloc[i] * 2)
    return pd.Series(result)

# After (vectorized)
def transform_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Transform column with vectorized operation.

    Args:
        df: Input dataframe
        column: Column name to transform

    Returns:
        Transformed series
    """
    return df[column] * 2  # 10x faster - vectorized
```

### Step 3: Validate Performance

```bash
/run-all-tests --benchmark --profile
```

**Performance Improvements:**
```
📊 Benchmark Results:

Before Optimization:
  test_sort_large_array: 2.34s
  test_transform_dataframe: 0.87s
  test_process_pipeline: 5.12s

After Optimization:
  test_sort_large_array: 0.02s (117x faster!)
  test_transform_dataframe: 0.08s (10.9x faster!)
  test_process_pipeline: 0.45s (11.4x faster!)

Overall Speedup: 25x average improvement
```

## Phase 4: Testing

### Step 1: Generate Unit Tests

```bash
/generate-tests src/ --type=unit --coverage=90
```

**Generated Tests:**
```
✅ Test Generation Complete:

Created 156 test cases:
  tests/algorithms/test_sorting.py - 24 tests
  tests/algorithms/test_search.py - 18 tests
  tests/processing/test_cleaner.py - 32 tests
  tests/processing/test_transformer.py - 28 tests
  tests/models/test_linear.py - 22 tests
  tests/models/test_ensemble.py - 18 tests
  tests/utils/test_io.py - 14 tests

Coverage: 92% (exceeds 90% target)
```

### Step 2: Generate Integration Tests

```bash
/generate-tests src/ --type=integration
```

### Step 3: Add Performance Tests

```bash
/generate-tests src/ --type=performance
```

### Step 4: Run All Tests

```bash
/run-all-tests --coverage --auto-fix
```

**Results:**
```
🧪 Test Results:
  Total: 198 tests
  Passed: 196 (99%)
  Failed: 2 (1%)
  Coverage: 92%

❌ Failed Tests:
  1. test_models/test_neural.py::test_convergence
  2. test_utils/test_io.py::test_large_file

🔧 Auto-fixing failures...
✅ Fixed: 2/2 tests

Final Results:
  Total: 198 tests
  Passed: 198 (100%)
  Coverage: 92%
```

## Phase 5: Documentation

### Step 1: Generate README

```bash
/update-docs --type=readme
```

**Generated README.md:**
- Project overview
- Installation instructions
- Quick start guide
- API overview
- Examples
- Contributing guidelines
- License

### Step 2: Generate API Documentation

```bash
/update-docs --type=api --format=markdown
```

**Generated API docs for all modules:**
```
docs/api/
├── algorithms.md
├── processing.md
├── models.md
└── utils.md
```

### Step 3: Add Usage Examples

```bash
/explain-code --level=basic --docs --format=examples src/
```

## Phase 6: CI/CD Setup

### Step 1: Setup GitHub Actions

```bash
/ci-setup --platform=github --type=enterprise --monitoring
```

**Created .github/workflows/ci.yml:**
```yaml
name: CI Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2

  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Code quality check
        run: /check-code-quality --format=json

  security:
    runs-on: ubuntu-latest
    steps:
      - name: Security scan
        run: bandit -r src/
```

## Phase 7: Final Verification

### Step 1: Run Complete Pipeline

```bash
# Complete workflow
/check-code-quality --detailed && \
/optimize --profile --report && \
/run-all-tests --coverage --benchmark && \
/update-docs --type=all
```

### Step 2: Final Quality Check

```bash
/check-code-quality --agents=all --orchestrate --detailed
```

**Final Results:**
```
📊 Final Quality Report:

Quality Score: 94/100 (Excellent!)
  Initial: 68/100
  Improvement: +26 points (38% better)

Code Metrics:
  Lines of code: 3,766 (down from 5,000 - 24.7% reduction)
  Test coverage: 92% (up from 12%)
  Documentation: 100% (up from 15%)
  Performance: 25x average speedup

Issues Resolved: 87/87 (100%)

🤖 All 23 agents agree: Production ready! ✅
```

## Summary of Changes

### Before
- Quality score: 68/100
- 5,000 lines
- Test coverage: 12%
- Performance: Baseline
- Documentation: Minimal
- Technical debt: High

### After
- Quality score: 94/100 (+38%)
- 3,766 lines (-24.7%)
- Test coverage: 92% (+667%)
- Performance: 25x speedup
- Documentation: Complete
- Technical debt: Low
- CI/CD: Configured

## Commands Used

```bash
# Assessment
/check-code-quality --dry-run --report
/optimize --profile --detailed src/

# Quality Improvement
/check-code-quality --auto-fix
/clean-codebase --imports --dead-code --backup

# Optimization
/optimize --implement --category=algorithm src/algorithms/
/optimize --implement --category=memory src/processing/

# Testing
/generate-tests src/ --type=unit --coverage=90
/generate-tests src/ --type=integration
/generate-tests src/ --type=performance
/run-all-tests --coverage --auto-fix

# Documentation
/update-docs --type=readme
/update-docs --type=api --format=markdown
/explain-code --level=basic --docs src/

# CI/CD
/ci-setup --platform=github --type=enterprise --monitoring

# Final Verification
/check-code-quality --agents=all --orchestrate --detailed
```

## Time Investment

- Initial assessment: 10 minutes
- Quality improvement: 15 minutes
- Performance optimization: 20 minutes
- Test generation: 15 minutes
- Documentation: 10 minutes
- CI/CD setup: 10 minutes
- Final verification: 10 minutes

**Total: 90 minutes** for complete transformation

## Key Takeaways

1. **Start with assessment** - Understand current state
2. **Use dry-run** - Always preview changes first
3. **Incremental improvement** - Quality → Performance → Testing → Docs
4. **Leverage agents** - Auto-selection works great
5. **Validate continuously** - Run tests after each phase
6. **Automate everything** - CI/CD ensures quality stays high

## Next Steps

Apply similar workflow to your projects:

1. **[Tutorial 01: Code Quality](../tutorials/tutorial-01-code-quality.md)**
2. **[Tutorial 02: Optimization](../tutorials/tutorial-02-optimization.md)**
3. **[Common Workflows](../getting-started/common-workflows.md)**

---

**Ready to transform your project?** → [Getting Started](../getting-started/README.md)