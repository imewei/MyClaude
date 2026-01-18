# Coverage Analysis Guide

Comprehensive guide for coverage metrics (line, branch, mutation), gap identification, test prioritization, reporting, visualization, and integration with pytest-cov and istanbul.

## Coverage Metrics

### Line Coverage

```python
# Line coverage: percentage of code lines executed during tests

# Example:
def calculate_discount(price, customer_type):
    if customer_type == 'premium':  # Line 1: Covered
        discount = 0.20              # Line 2: Covered
    elif customer_type == 'standard': # Line 3: Covered
        discount = 0.10              # Line 4: Covered
    else:
        discount = 0                 # Line 5: NOT COVERED (missing test)

    return price * (1 - discount)    # Line 6: Covered

# Line coverage = 5/6 = 83.3%
```

### Branch Coverage

```python
# Branch coverage: percentage of decision branches taken

# Same example:
# - if customer_type == 'premium': True branch ✓, False branch ✓
# - elif customer_type == 'standard': True branch ✓, False branch ✓  
# - else: NOT COVERED ✗

# Branch coverage = 4/5 = 80%
```

### Function Coverage

```python
# Function coverage: percentage of functions called

def function_a():  # Covered
    pass

def function_b():  # Covered  
    pass

def function_c():  # NOT COVERED
    pass

# Function coverage = 2/3 = 66.7%
```

### Statement Coverage

```python
# Statement coverage: percentage of statements executed
# Similar to line coverage but counts actual statements

def complex_function(x):
    a = 1; b = 2; c = 3  # 3 statements on one line
    if x > 0:
        return a + b     # 1 statement
    return c             # 1 statement

# All 5 statements must be covered for 100%
```

## pytest-cov Integration

### Basic Usage

```bash
# Run tests with coverage
pytest --cov=src

# With HTML report
pytest --cov=src --cov-report=html

# With terminal report
pytest --cov=src --cov-report=term

# With XML report (for CI)
pytest --cov=src --cov-report=xml

# Multiple report formats
pytest --cov=src --cov-report=html --cov-report=term --cov-report=xml
```

### Configuration

```ini
# pytest.ini
[pytest]
addopts = 
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

[coverage:run]
source = src
omit = 
    */tests/*
    */test_*.py
    */__init__.py
    */migrations/*

[coverage:report]
precision = 2
show_missing = True
skip_covered = False
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstract
```

### pyproject.toml Configuration

```toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/__init__.py"
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if TYPE_CHECKING:",
    "raise NotImplementedError"
]

[tool.coverage.html]
directory = "htmlcov"
```

### Advanced Coverage Features

```python
# Exclude specific lines from coverage
def debug_only_function():
    if DEBUG:  # pragma: no cover
        print("Debug info")


# Exclude entire function
def experimental_feature():  # pragma: no cover
    """Not tested yet"""
    pass


# Branch coverage for assertions
def validate_input(x):
    assert x > 0, "Must be positive"  # Both branches tested
    return x * 2
```

## JavaScript Coverage with Istanbul/NYC

### Jest Integration

```javascript
// jest.config.js
module.exports = {
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html', 'json'],
  collectCoverageFrom: [
    'src/**/*.{js,ts}',
    '!src/**/*.test.{js,ts}',
    '!src/**/*.d.ts',
    '!src/**/index.ts',
    '!src/**/*.stories.{js,ts}'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    },
    './src/utils/': {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90
    }
  }
};
```

### NYC Configuration

```json
{
  "nyc": {
    "all": true,
    "include": [
      "src/**/*.js"
    ],
    "exclude": [
      "**/*.test.js",
      "**/*.spec.js",
      "**/node_modules/**"
    ],
    "reporter": [
      "html",
      "text",
      "lcov"
    ],
    "check-coverage": true,
    "lines": 80,
    "functions": 80,
    "branches": 80,
    "statements": 80
  }
}
```

## Gap Identification

### Automated Gap Detection

```python
import coverage
import ast
from typing import List, Dict

class CoverageGapAnalyzer:
    """Identify and prioritize coverage gaps"""

    def analyze_gaps(self, coverage_file: str) -> List[Dict]:
        """Find uncovered code and prioritize tests"""

        # Load coverage data
        cov = coverage.Coverage(data_file=coverage_file)
        cov.load()

        gaps = []

        # Analyze each file
        for filename in cov.get_data().measured_files():
            analysis = cov.analysis2(filename)

            missing_lines = analysis[2]  # Lines not executed

            if missing_lines:
                # Parse source to understand missing code
                with open(filename) as f:
                    source = f.read()

                tree = ast.parse(source)

                # Find functions with missing coverage
                uncovered_functions = self.find_uncovered_functions(
                    tree,
                    missing_lines
                )

                for func in uncovered_functions:
                    gaps.append({
                        'file': filename,
                        'function': func['name'],
                        'missing_lines': func['missing_lines'],
                        'complexity': func['complexity'],
                        'priority': self.calculate_priority(func)
                    })

        # Sort by priority
        gaps.sort(key=lambda x: x['priority'], reverse=True)

        return gaps

    def find_uncovered_functions(
        self,
        tree: ast.Module,
        missing_lines: List[int]
    ) -> List[Dict]:
        """Find functions with missing coverage"""

        uncovered = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = set(range(
                    node.lineno,
                    node.end_lineno + 1
                ))

                missing_in_func = func_lines.intersection(missing_lines)

                if missing_in_func:
                    uncovered.append({
                        'name': node.name,
                        'missing_lines': list(missing_in_func),
                        'total_lines': len(func_lines),
                        'complexity': self.calculate_complexity(node)
                    })

        return uncovered

    def calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1

        return complexity

    def calculate_priority(self, func: Dict) -> float:
        """Calculate test priority score"""

        score = 0.0

        # Higher complexity = higher priority
        score += func['complexity'] * 10

        # More missing lines = higher priority
        coverage_pct = 1 - (len(func['missing_lines']) / func['total_lines'])
        score += (1 - coverage_pct) * 30

        # Public functions = higher priority
        if not func['name'].startswith('_'):
            score += 20

        return score
```

### Gap Visualization

```python
import matplotlib.pyplot as plt
import pandas as pd

def visualize_coverage_gaps(gaps: List[Dict]):
    """Create visualization of coverage gaps"""

    df = pd.DataFrame(gaps)

    # Bar chart of priority scores
    plt.figure(figsize=(12, 6))
    plt.barh(df['function'], df['priority'])
    plt.xlabel('Priority Score')
    plt.title('Coverage Gap Priority')
    plt.tight_layout()
    plt.savefig('coverage_gaps.png')

    # Scatter plot: complexity vs coverage
    plt.figure(figsize=(10, 6))
    coverage_pct = 1 - (df['missing_lines'].apply(len) / df['total_lines'])
    plt.scatter(df['complexity'], coverage_pct)
    plt.xlabel('Cyclomatic Complexity')
    plt.ylabel('Coverage %')
    plt.title('Complexity vs Coverage')
    plt.savefig('complexity_coverage.png')
```

## Test Prioritization

### Priority-Based Test Generation

```python
class PrioritizedTestGenerator:
    """Generate tests based on coverage gaps and priority"""

    def generate_tests(self, gaps: List[Dict]) -> List[str]:
        """Generate tests for highest priority gaps"""

        tests = []

        # Focus on top 20% of gaps
        high_priority_gaps = gaps[:max(1, len(gaps) // 5)]

        for gap in high_priority_gaps:
            test_code = self.generate_test_for_gap(gap)
            tests.append(test_code)

        return tests

    def generate_test_for_gap(self, gap: Dict) -> str:
        """Generate test targeting specific coverage gap"""

        if gap['complexity'] > 5:
            # High complexity: generate multiple test cases
            return self.generate_comprehensive_tests(gap)
        else:
            # Low complexity: simple test
            return self.generate_simple_test(gap)

    def generate_comprehensive_tests(self, gap: Dict) -> str:
        """Generate multiple tests for complex function"""

        return f"""
class Test{gap['function'].title()}:
    '''Comprehensive tests for {gap['function']}'''

    def test_{gap['function']}_happy_path(self):
        '''Test main execution path'''
        # Test implementation
        pass

    def test_{gap['function']}_edge_cases(self):
        '''Test edge cases on lines {gap['missing_lines']}'''
        # Test implementation
        pass

    @pytest.mark.parametrize("input,expected", [
        # Test cases
    ])
    def test_{gap['function']}_parametrized(self, input, expected):
        '''Parametrized test for multiple scenarios'''
        # Test implementation
        pass
"""
```

## Coverage Reporting

### HTML Reports

```bash
# Generate HTML report
pytest --cov=src --cov-report=html

# Open in browser
open htmlcov/index.html

# HTML report shows:
# - Overall coverage percentage
# - Per-file coverage
# - Line-by-line highlighting:
#   - Green: Covered
#   - Red: Not covered
#   - Yellow: Partially covered (branches)
```

### Terminal Reports

```bash
# Terminal report with missing lines
pytest --cov=src --cov-report=term-missing

# Output:
# Name                     Stmts   Miss  Cover   Missing
# ------------------------------------------------------
# src/utils.py                50     10    80%   23-28, 45
# src/api.py                 100      5    95%   67-68
# ------------------------------------------------------
# TOTAL                      150     15    90%
```

### JSON Reports

```python
# Generate JSON report for programmatic analysis
pytest --cov=src --cov-report=json

# Parse JSON report
import json

with open('coverage.json') as f:
    coverage_data = json.load(f)

# Access coverage data
total_coverage = coverage_data['totals']['percent_covered']
file_coverage = coverage_data['files']['src/utils.py']['summary']['percent_covered']
```

### CI Integration

```yaml
# .github/workflows/coverage.yml
name: Coverage

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          uv uv pip install -r requirements-dev.txt

      - name: Run tests with coverage
        run: |
          pytest --cov=src --cov-report=xml --cov-report=term

      - name: Upload to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

      - name: Check coverage threshold
        run: |
          pytest --cov=src --cov-fail-under=80
```

## Mutation Testing

### Python: mutmut

```bash
# Install mutmut
uv uv pip install mutmut

# Run mutation testing
mutmut run

# Show results
mutmut results

# Show specific mutations
mutmut show 1

# HTML report
mutmut html
```

### JavaScript: Stryker

```javascript
// stryker.conf.js
module.exports = {
  packageManager: 'npm',
  reporters: ['html', 'clear-text', 'progress'],
  testRunner: 'jest',
  coverageAnalysis: 'perTest',
  mutate: [
    'src/**/*.js',
    '!src/**/*.test.js'
  ],
  thresholds: {
    high: 80,
    low: 60,
    break: 50
  }
};
```

## Coverage Trends

### Tracking Coverage Over Time

```python
import json
from datetime import datetime

class CoverageTracker:
    """Track coverage trends over time"""

    def __init__(self, history_file='coverage_history.json'):
        self.history_file = history_file
        self.history = self.load_history()

    def load_history(self):
        try:
            with open(self.history_file) as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def add_coverage_data(self, coverage_pct: float):
        """Add coverage data point"""
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'coverage': coverage_pct
        })

        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_trend(self):
        """Plot coverage trend"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        timestamps = [datetime.fromisoformat(h['timestamp']) for h in self.history]
        coverage = [h['coverage'] for h in self.history]

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, coverage, marker='o')
        plt.xlabel('Date')
        plt.ylabel('Coverage %')
        plt.title('Test Coverage Trend')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('coverage_trend.png')
```

## Best Practices

1. **Set minimum coverage thresholds** (80% is common)
2. **Focus on critical code** first
3. **Use branch coverage**, not just line coverage
4. **Track coverage trends** over time
5. **Integrate coverage into CI/CD**
6. **Generate HTML reports** for visualization
7. **Prioritize uncovered complex code**
8. **Exclude generated code** from coverage
9. **Use mutation testing** to verify test quality
10. **Don't chase 100% coverage** blindly - focus on meaningful tests
