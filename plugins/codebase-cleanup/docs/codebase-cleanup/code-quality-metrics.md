# Code Quality Metrics

**Version**: 1.0.3
**Category**: codebase-cleanup
**Purpose**: Comprehensive metrics framework for measuring and tracking code quality

## Core Quality Metrics

### Cyclomatic Complexity

**Definition**: Number of independent paths through code

**Formula**: `M = E - N + 2P`
- E = number of edges in control flow graph
- N = number of nodes
- P = number of connected components (usually 1)

**Thresholds**:
- **1-10**: Simple, low risk
- **11-20**: Moderate complexity, acceptable
- **21-50**: High complexity, requires refactoring
- **50+**: Very high risk, immediate refactoring needed

**Calculation Example**:
```python
def calculate_complexity(function_node):
    """Calculate cyclomatic complexity for a function"""
    complexity = 1  # Start with 1

    # Count decision points
    decision_keywords = ['if', 'elif', 'while', 'for', 'and', 'or', 'except']

    for node in ast.walk(function_node):
        if isinstance(node, ast.If):
            complexity += 1
        elif isinstance(node, ast.While):
            complexity += 1
        elif isinstance(node, ast.For):
            complexity += 1
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1

    return complexity
```

**Example**:
```python
# Complexity = 5
def validate_user(user):
    if not user.email:  # +1
        raise ValueError("Email required")

    if not user.age:  # +1
        raise ValueError("Age required")

    if user.age < 18 or user.age > 120:  # +2 (or adds 1)
        raise ValueError("Invalid age")

    return True  # +1 (base)
```

### Code Duplication

**Definition**: Percentage of duplicated code blocks

**Threshold**:
- **< 3%**: Excellent
- **3-5%**: Good
- **5-10%**: Acceptable
- **> 10%**: Requires cleanup

**Detection Algorithm**:
```python
import hashlib
from collections import defaultdict

class DuplicationDetector:
    def __init__(self, min_lines=6):
        self.min_lines = min_lines
        self.hashes = defaultdict(list)

    def hash_code_block(self, lines):
        """Hash normalized code block"""
        # Normalize: remove whitespace, comments
        normalized = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                normalized.append(line)

        code = '\n'.join(normalized)
        return hashlib.md5(code.encode()).hexdigest()

    def find_duplicates(self, file_path):
        """Find duplicate code blocks"""
        with open(file_path) as f:
            lines = f.readlines()

        duplicates = []

        # Sliding window
        for i in range(len(lines) - self.min_lines + 1):
            block = lines[i:i + self.min_lines]
            block_hash = self.hash_code_block(block)

            self.hashes[block_hash].append({
                'file': file_path,
                'start_line': i + 1,
                'end_line': i + self.min_lines
            })

        # Find blocks with multiple occurrences
        for block_hash, occurrences in self.hashes.items():
            if len(occurrences) > 1:
                duplicates.append({
                    'hash': block_hash,
                    'occurrences': occurrences,
                    'count': len(occurrences)
                })

        return duplicates
```

### Test Coverage

**Definition**: Percentage of code executed by tests

**Thresholds**:
- **90-100%**: Excellent
- **80-89%**: Good
- **70-79%**: Acceptable
- **< 70%**: Needs improvement

**Measurement**:
```python
# Using coverage.py
import coverage

cov = coverage.Coverage()
cov.start()

# Run tests
import pytest
pytest.main(['tests/'])

cov.stop()
cov.save()

# Generate report
coverage_data = cov.get_data()
total_statements = 0
covered_statements = 0

for file_path in coverage_data.measured_files():
    analysis = coverage_data.analysis(file_path)
    total_statements += len(analysis.statements)
    covered_statements += len(analysis.executed)

coverage_percentage = (covered_statements / total_statements) * 100
print(f"Test coverage: {coverage_percentage:.2f}%")
```

### Maintainability Index

**Definition**: Composite metric from complexity, lines of code, and Halstead volume

**Formula**:
```
MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)
```
- V = Halstead Volume
- G = Cyclomatic Complexity
- LOC = Lines of Code

**Thresholds**:
- **85-100**: Highly maintainable
- **65-84**: Moderately maintainable
- **< 65**: Difficult to maintain

**Implementation**:
```python
import math

def calculate_maintainability_index(file_path):
    """Calculate maintainability index for a file"""
    with open(file_path) as f:
        source = f.read()

    # Count lines of code (excluding comments and blanks)
    loc = count_loc(source)

    # Calculate cyclomatic complexity
    complexity = calculate_cyclomatic_complexity(source)

    # Calculate Halstead volume
    volume = calculate_halstead_volume(source)

    # Calculate MI
    if loc == 0:
        return 100

    mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(loc)

    # Normalize to 0-100 scale
    mi = max(0, min(100, mi))

    return round(mi, 2)

def calculate_halstead_volume(source):
    """Calculate Halstead volume (simplified)"""
    import ast

    tree = ast.parse(source)

    operators = set()
    operands = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            operands.add(node.id)
        elif isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            operators.add(type(node).__name__)

    n1 = len(operators)  # Unique operators
    n2 = len(operands)   # Unique operands
    N1 = n1 * 2          # Total operators (simplified)
    N2 = n2 * 2          # Total operands (simplified)

    vocabulary = n1 + n2
    length = N1 + N2

    if vocabulary == 0:
        return 0

    volume = length * math.log2(vocabulary)
    return volume
```

## Quality Gates

### Pre-Commit Quality Gate

```yaml
pre_commit_checks:
  complexity:
    max_cyclomatic_complexity: 10
    fail_on_violation: true

  duplication:
    max_duplication_percentage: 5
    min_duplicate_lines: 6
    fail_on_violation: false  # Warning only

  coverage:
    min_coverage_percentage: 80
    fail_on_violation: true
    coverage_delta_threshold: -2  # Fail if coverage decreases > 2%

  linting:
    tools:
      - pylint
      - flake8
      - mypy
    fail_on_error: true
    fail_on_warning: false

  security:
    tools:
      - bandit
      - safety
    fail_on_high_severity: true
    fail_on_medium_severity: false
```

**Implementation**:
```python
class QualityGate:
    def __init__(self, config):
        self.config = config

    def check_all(self, changed_files):
        """Run all quality checks"""
        results = {
            'passed': True,
            'checks': []
        }

        # Complexity check
        complexity_result = self.check_complexity(changed_files)
        results['checks'].append(complexity_result)
        if not complexity_result['passed']:
            results['passed'] = False

        # Duplication check
        duplication_result = self.check_duplication(changed_files)
        results['checks'].append(duplication_result)
        if not duplication_result['passed']:
            results['passed'] = False

        # Coverage check
        coverage_result = self.check_coverage()
        results['checks'].append(coverage_result)
        if not coverage_result['passed']:
            results['passed'] = False

        return results

    def check_complexity(self, files):
        """Check cyclomatic complexity"""
        max_complexity = self.config['complexity']['max_cyclomatic_complexity']
        violations = []

        for file_path in files:
            complexity = calculate_file_complexity(file_path)
            if complexity > max_complexity:
                violations.append({
                    'file': file_path,
                    'complexity': complexity,
                    'threshold': max_complexity
                })

        return {
            'check': 'complexity',
            'passed': len(violations) == 0,
            'violations': violations
        }

    def check_coverage(self):
        """Check test coverage"""
        min_coverage = self.config['coverage']['min_coverage_percentage']

        current_coverage = get_current_coverage()

        return {
            'check': 'coverage',
            'passed': current_coverage >= min_coverage,
            'current': current_coverage,
            'threshold': min_coverage
        }
```

### Pull Request Quality Gate

```python
class PRQualityGate:
    def __init__(self, pr_number):
        self.pr_number = pr_number

    def evaluate(self):
        """Evaluate PR quality"""
        pr_data = self.fetch_pr_data()

        checks = {
            'size': self.check_pr_size(pr_data),
            'tests': self.check_tests_added(pr_data),
            'coverage': self.check_coverage_maintained(pr_data),
            'complexity': self.check_complexity_not_increased(pr_data),
            'documentation': self.check_documentation_updated(pr_data),
            'review': self.check_reviews_approved(pr_data)
        }

        passed = all(check['passed'] for check in checks.values())

        return {
            'passed': passed,
            'checks': checks,
            'recommendation': self.generate_recommendation(checks)
        }

    def check_pr_size(self, pr_data):
        """Check PR is not too large"""
        lines_changed = pr_data['additions'] + pr_data['deletions']

        return {
            'passed': lines_changed <= 500,
            'lines_changed': lines_changed,
            'threshold': 500,
            'message': 'PR is too large, consider breaking it up' if lines_changed > 500 else 'PR size is acceptable'
        }

    def check_tests_added(self, pr_data):
        """Check that tests were added for new code"""
        has_production_changes = any(
            not file_path.startswith('test')
            for file_path in pr_data['changed_files']
        )

        has_test_changes = any(
            file_path.startswith('test') or 'test' in file_path
            for file_path in pr_data['changed_files']
        )

        if not has_production_changes:
            # No production code changed, test requirement doesn't apply
            return {'passed': True, 'message': 'No production code changed'}

        return {
            'passed': has_test_changes,
            'message': 'Tests added' if has_test_changes else 'No tests added for production changes'
        }
```

## Metrics Dashboard

### Dashboard Schema

```typescript
interface MetricsDashboard {
    timestamp: string;
    project: string;
    metrics: {
        complexity: ComplexityMetrics;
        duplication: DuplicationMetrics;
        coverage: CoverageMetrics;
        maintainability: MaintainabilityMetrics;
        trends: TrendMetrics;
    };
}

interface ComplexityMetrics {
    average: number;
    median: number;
    max: number;
    filesAboveThreshold: number;
    distribution: {
        low: number;      // 1-10
        medium: number;   // 11-20
        high: number;     // 21-50
        veryHigh: number; // 50+
    };
}

interface DuplicationMetrics {
    percentage: number;
    duplicateBlocks: number;
    largestDuplicate: {
        lines: number;
        occurrences: number;
        files: string[];
    };
}

interface CoverageMetrics {
    overall: number;
    byType: {
        unit: number;
        integration: number;
        e2e: number;
    };
    uncoveredFiles: string[];
}

interface TrendMetrics {
    complexity: TrendData[];
    coverage: TrendData[];
    duplication: TrendData[];
}

interface TrendData {
    date: string;
    value: number;
}
```

### Visualization Example

```python
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta

class MetricsDashboard:
    def __init__(self, metrics_file):
        with open(metrics_file) as f:
            self.metrics = json.load(f)

    def plot_complexity_distribution(self):
        """Plot complexity distribution"""
        dist = self.metrics['complexity']['distribution']

        categories = ['Low\n(1-10)', 'Medium\n(11-20)', 'High\n(21-50)', 'Very High\n(50+)']
        values = [dist['low'], dist['medium'], dist['high'], dist['veryHigh']]

        colors = ['green', 'yellow', 'orange', 'red']

        plt.figure(figsize=(10, 6))
        plt.bar(categories, values, color=colors)
        plt.title('Cyclomatic Complexity Distribution')
        plt.ylabel('Number of Functions')
        plt.xlabel('Complexity Range')
        plt.savefig('complexity_distribution.png')
        plt.close()

    def plot_coverage_trend(self):
        """Plot coverage over time"""
        trend = self.metrics['trends']['coverage']

        dates = [datetime.fromisoformat(d['date']) for d in trend]
        values = [d['value'] for d in trend]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, marker='o', linewidth=2)
        plt.axhline(y=80, color='r', linestyle='--', label='Target (80%)')
        plt.title('Test Coverage Trend')
        plt.ylabel('Coverage %')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('coverage_trend.png')
        plt.close()

    def generate_html_report(self):
        """Generate HTML dashboard"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Quality Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-card {{
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 10px;
                    display: inline-block;
                    width: 250px;
                }}
                .metric-value {{
                    font-size: 48px;
                    font-weight: bold;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                }}
                .status-good {{ color: green; }}
                .status-warning {{ color: orange; }}
                .status-bad {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Code Quality Dashboard</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="metric-card">
                <div class="metric-label">Average Complexity</div>
                <div class="metric-value status-{self._get_complexity_status()}">
                    {self.metrics['complexity']['average']:.1f}
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Test Coverage</div>
                <div class="metric-value status-{self._get_coverage_status()}">
                    {self.metrics['coverage']['overall']:.1f}%
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Code Duplication</div>
                <div class="metric-value status-{self._get_duplication_status()}">
                    {self.metrics['duplication']['percentage']:.1f}%
                </div>
            </div>

            <h2>Complexity Distribution</h2>
            <img src="complexity_distribution.png" />

            <h2>Coverage Trend</h2>
            <img src="coverage_trend.png" />
        </body>
        </html>
        """

        with open('dashboard.html', 'w') as f:
            f.write(html)

    def _get_complexity_status(self):
        avg = self.metrics['complexity']['average']
        if avg <= 10:
            return 'good'
        elif avg <= 20:
            return 'warning'
        return 'bad'

    def _get_coverage_status(self):
        cov = self.metrics['coverage']['overall']
        if cov >= 80:
            return 'good'
        elif cov >= 70:
            return 'warning'
        return 'bad'

    def _get_duplication_status(self):
        dup = self.metrics['duplication']['percentage']
        if dup < 3:
            return 'good'
        elif dup < 5:
            return 'warning'
        return 'bad'
```

## Metrics Interpretation Guide

### When Complexity is High

**Root Causes**:
- Too many conditional branches
- Nested loops and conditions
- Long methods doing multiple things

**Actions**:
1. Extract methods to reduce nesting
2. Use guard clauses to reduce indentation
3. Replace conditionals with polymorphism
4. Apply strategy pattern for complex decision logic

### When Coverage is Low

**Root Causes**:
- Legacy code without tests
- Hard-to-test code (tight coupling)
- Missing edge case tests

**Actions**:
1. Add characterization tests for legacy code
2. Refactor for testability (dependency injection)
3. Focus on high-risk areas first
4. Use mutation testing to verify test quality

### When Duplication is High

**Root Causes**:
- Copy-paste programming
- Lack of abstraction
- Missed refactoring opportunities

**Actions**:
1. Extract shared logic to functions/classes
2. Use inheritance or composition
3. Create utility modules for common operations
4. Apply DRY principle systematically

## Target Metrics by Project Type

### Web Application
- Complexity: < 12 average
- Coverage: > 85%
- Duplication: < 3%
- Maintainability Index: > 75

### API Service
- Complexity: < 10 average
- Coverage: > 90%
- Duplication: < 2%
- Maintainability Index: > 80

### Data Pipeline
- Complexity: < 15 average
- Coverage: > 75%
- Duplication: < 5%
- Maintainability Index: > 70

### CLI Tool
- Complexity: < 12 average
- Coverage: > 80%
- Duplication: < 4%
- Maintainability Index: > 75
