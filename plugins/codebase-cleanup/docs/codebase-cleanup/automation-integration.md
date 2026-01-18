# Automation Integration

**Version**: 1.0.3
**Category**: codebase-cleanup
**Purpose**: CI/CD integration patterns, quality gates, and automated workflows for continuous code cleanup

## GitHub Actions Integration

### Comprehensive Quality Check Workflow

```yaml
name: Code Quality Check

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  quality-gate:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Full history for better analysis

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          uv uv pip install -r requirements.txt
          uv uv pip install pylint flake8 mypy bandit safety radon

      - name: Run linting
        run: |
          echo "::group::Pylint"
          pylint src/ --fail-under=8.0
          echo "::endgroup::"

          echo "::group::Flake8"
          flake8 src/ --max-complexity=10 --max-line-length=120
          echo "::endgroup::"

      - name: Run type checking
        run: |
          mypy src/ --strict

      - name: Check complexity
        run: |
          radon cc src/ -a -nb
          radon cc src/ -a -nb --total-average --min B

      - name: Security scan
        run: |
          bandit -r src/ -ll
          safety check --json

      - name: Run tests with coverage
        run: |
          pytest --cov=src --cov-report=xml --cov-report=term
          coverage report --fail-under=80

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

      - name: Check for code duplication
        run: |
          python scripts/check_duplication.py --threshold 5

      - name: Quality gate summary
        if: always()
        run: |
          python scripts/generate_quality_report.py > quality_report.md

      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('quality_report.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
```

### Dependency Security Scan

```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  pull_request:
    paths:
      - 'requirements.txt'
      - 'package.json'
      - 'Gemfile'

jobs:
  security-audit:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Python security scan
        if: hashFiles('requirements.txt') != ''
        run: |
          uv uv pip install safety
          safety check --json > security_report.json || true

      - name: NPM security scan
        if: hashFiles('package.json') != ''
        run: |
          npm audit --json > npm_audit.json || true

      - name: Analyze vulnerabilities
        run: |
          python scripts/analyze_vulnerabilities.py

      - name: Create security issue
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const vulns = JSON.parse(fs.readFileSync('critical_vulns.json', 'utf8'));

            if (vulns.length > 0) {
              github.rest.issues.create({
                owner: context.repo.owner,
                repo: context.repo.repo,
                title: `🚨 Security Alert: ${vulns.length} Critical Vulnerabilities`,
                body: vulns.map(v => `- ${v.package}: ${v.vulnerability}`).join('\n'),
                labels: ['security', 'critical']
              });
            }
```

### Automated Refactoring PR

```yaml
name: Auto-refactor

on:
  workflow_dispatch:
    inputs:
      target:
        description: 'Refactoring target (imports|format|complexity)'
        required: true
        default: 'imports'

jobs:
  auto-refactor:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install tools
        run: |
          uv uv pip install black isort autoflake

      - name: Run refactoring
        run: |
          case "${{ github.event.inputs.target }}" in
            imports)
              isort src/ tests/
              ;;
            format)
              black src/ tests/
              ;;
            complexity)
              python scripts/simplify_complex_functions.py
              ;;
          esac

      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v5
        with:
          commit-message: "refactor: automated ${{ github.event.inputs.target }} cleanup"
          title: "🤖 Automated refactoring: ${{ github.event.inputs.target }}"
          body: |
            ## Automated Refactoring

            Target: `${{ github.event.inputs.target }}`

            This PR was automatically generated to improve code quality.

            ### Changes
            - Applied automated refactoring tools
            - No functional changes expected

            ### Verification
            - [ ] All tests pass
            - [ ] No regressions identified
            - [ ] Code review completed

            🤖 Generated with Claude Code
          branch: auto-refactor-${{ github.event.inputs.target }}
          labels: |
            refactoring
            automated
```

## Pre-commit Hooks

### Configuration (.pre-commit-config.yaml)

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-complexity=10', '--max-line-length=120']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: local
    hooks:
      - id: complexity-check
        name: Check cyclomatic complexity
        entry: python scripts/check_complexity.py
        language: system
        files: \.py$
        pass_filenames: true

      - id: test-coverage
        name: Verify test coverage
        entry: python scripts/check_coverage.py
        language: system
        files: \.py$
        pass_filenames: false
```

### Custom Hook Scripts

**scripts/check_complexity.py**:
```python
#!/usr/bin/env python3
"""Pre-commit hook to check cyclomatic complexity"""

import sys
import ast
from pathlib import Path

MAX_COMPLEXITY = 10

def calculate_complexity(node):
    """Calculate cyclomatic complexity for a function"""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity

def check_file(file_path):
    """Check complexity of all functions in a file"""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            complexity = calculate_complexity(node)
            if complexity > MAX_COMPLEXITY:
                violations.append({
                    'function': node.name,
                    'line': node.lineno,
                    'complexity': complexity
                })

    return violations

def main():
    """Main entry point"""
    failed = False

    for file_path in sys.argv[1:]:
        if not file_path.endswith('.py'):
            continue

        violations = check_file(file_path)

        if violations:
            failed = True
            print(f"\n❌ {file_path}:")
            for v in violations:
                print(f"  Line {v['line']}: {v['function']}() has complexity {v['complexity']} (max: {MAX_COMPLEXITY})")

    if failed:
        print("\n💡 Tip: Break down complex functions into smaller, focused functions")
        sys.exit(1)

    print("✅ All functions have acceptable complexity")
    sys.exit(0)

if __name__ == '__main__':
    main()
```

**scripts/check_coverage.py**:
```python
#!/usr/bin/env python3
"""Pre-commit hook to verify test coverage doesn't decrease"""

import sys
import json
from pathlib import Path

COVERAGE_FILE = '.coverage_baseline.json'
MIN_COVERAGE = 80.0

def get_current_coverage():
    """Run tests and get current coverage"""
    import subprocess
    result = subprocess.run(
        ['pytest', '--cov=src', '--cov-report=json'],
        capture_output=True
    )

    with open('coverage.json') as f:
        data = json.load(f)

    return data['totals']['percent_covered']

def load_baseline():
    """Load baseline coverage"""
    if not Path(COVERAGE_FILE).exists():
        return None

    with open(COVERAGE_FILE) as f:
        return json.load(f)['coverage']

def save_baseline(coverage):
    """Save baseline coverage"""
    with open(COVERAGE_FILE, 'w') as f:
        json.dump({'coverage': coverage}, f)

def main():
    """Main entry point"""
    current = get_current_coverage()
    baseline = load_baseline()

    print(f"Current coverage: {current:.2f}%")

    if current < MIN_COVERAGE:
        print(f"❌ Coverage below minimum ({MIN_COVERAGE}%)")
        sys.exit(1)

    if baseline and current < baseline - 1.0:
        print(f"❌ Coverage decreased by {baseline - current:.2f}%")
        print(f"Previous: {baseline:.2f}%")
        sys.exit(1)

    # Update baseline if coverage improved
    if not baseline or current > baseline:
        save_baseline(current)
        if baseline:
            print(f"✅ Coverage improved by {current - baseline:.2f}%")

    print("✅ Coverage check passed")
    sys.exit(0)

if __name__ == '__main__':
    main()
```

## Automated PR Generation

### Dependency Update Bot

```python
#!/usr/bin/env python3
"""Automated dependency update PR generator"""

import subprocess
import json
from datetime import datetime
from typing import List, Dict

class DependencyUpdateBot:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def check_outdated_dependencies(self) -> List[Dict]:
        """Check for outdated dependencies"""
        result = subprocess.run(
            ['pip', 'list', '--outdated', '--format=json'],
            capture_output=True,
            text=True
        )

        return json.loads(result.stdout)

    def update_dependency(self, package: str, version: str) -> bool:
        """Update a single dependency"""
        try:
            subprocess.run(
                ['pip', 'install', f'{package}=={version}'],
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def run_tests(self) -> bool:
        """Run test suite"""
        result = subprocess.run(['pytest'], capture_output=True)
        return result.returncode == 0

    def create_update_pr(self, updates: List[Dict]) -> None:
        """Create PR for dependency updates"""
        # Create new branch
        branch_name = f"deps/auto-update-{datetime.now().strftime('%Y%m%d')}"
        subprocess.run(['git', 'checkout', '-b', branch_name])

        # Group updates by severity
        security_updates = [u for u in updates if u.get('has_vulnerability')]
        regular_updates = [u for u in updates if not u.get('has_vulnerability')]

        # Apply updates
        for update in security_updates + regular_updates:
            package = update['name']
            new_version = update['latest_version']

            if self.update_dependency(package, new_version):
                # Commit each update
                subprocess.run(['git', 'add', 'requirements.txt'])
                subprocess.run([
                    'git', 'commit', '-m',
                    f"build: update {package} to {new_version}"
                ])

                # Verify tests still pass
                if not self.run_tests():
                    # Revert if tests fail
                    subprocess.run(['git', 'reset', '--hard', 'HEAD~1'])
                    print(f"⚠️  Tests failed for {package}, skipping")

        # Generate PR body
        pr_body = self._generate_pr_body(security_updates, regular_updates)

        # Push and create PR
        subprocess.run(['git', 'push', 'origin', branch_name])
        subprocess.run([
            'gh', 'pr', 'create',
            '--title', f"chore: automated dependency updates {datetime.now().strftime('%Y-%m-%d')}",
            '--body', pr_body
        ])

    def _generate_pr_body(self, security: List, regular: List) -> str:
        """Generate PR description"""
        body = "## 🤖 Automated Dependency Updates\n\n"

        if security:
            body += "### 🔒 Security Updates\n\n"
            for update in security:
                body += f"- **{update['name']}**: {update['version']} → {update['latest_version']}\n"
                if update.get('cve'):
                    body += f"  - CVE: {update['cve']}\n"
            body += "\n"

        if regular:
            body += "### 📦 Regular Updates\n\n"
            for update in regular:
                body += f"- {update['name']}: {update['version']} → {update['latest_version']}\n"
            body += "\n"

        body += """
### ✅ Verification
- All tests pass
- No breaking changes detected
- Dependencies resolved successfully

### 🚀 Deployment
Can be merged and deployed immediately after approval.

---
🤖 Generated with Claude Code
"""
        return body

def main():
    bot = DependencyUpdateBot('.')
    outdated = bot.check_outdated_dependencies()

    if outdated:
        print(f"Found {len(outdated)} outdated dependencies")
        bot.create_update_pr(outdated)
    else:
        print("All dependencies are up to date")

if __name__ == '__main__':
    main()
```

## Quality Gates Configuration

### SonarQube Integration

```yaml
# sonar-project.properties
sonar.projectKey=my-project
sonar.projectName=My Project
sonar.projectVersion=1.0

sonar.sources=src
sonar.tests=tests
sonar.python.coverage.reportPaths=coverage.xml

# Quality Gate thresholds
sonar.qualitygate.wait=true

# Coverage
sonar.coverage.exclusions=**/*_test.py,**/tests/**

# Duplications
sonar.cpd.exclusions=**/*_test.py

# Complexity
sonar.python.complexity.threshold=10
```

### Quality Gate Script

```python
#!/usr/bin/env python3
"""Quality gate checker"""

import sys
import json
from typing import Dict, List

class QualityGate:
    THRESHOLDS = {
        'coverage': 80.0,
        'complexity': 10,
        'duplication': 5.0,
        'vulnerabilities': 0,
        'bugs': 0,
        'code_smells': 10
    }

    def __init__(self):
        self.results = {}

    def check_all(self) -> bool:
        """Run all quality checks"""
        all_passed = True

        all_passed &= self.check_coverage()
        all_passed &= self.check_complexity()
        all_passed &= self.check_duplication()
        all_passed &= self.check_security()

        self.print_summary()

        return all_passed

    def check_coverage(self) -> bool:
        """Check test coverage"""
        with open('coverage.json') as f:
            data = json.load(f)

        coverage = data['totals']['percent_covered']
        passed = coverage >= self.THRESHOLDS['coverage']

        self.results['coverage'] = {
            'passed': passed,
            'value': coverage,
            'threshold': self.THRESHOLDS['coverage']
        }

        return passed

    def check_complexity(self) -> bool:
        """Check cyclomatic complexity"""
        # Run radon and parse results
        import subprocess
        result = subprocess.run(
            ['radon', 'cc', 'src/', '-j'],
            capture_output=True,
            text=True
        )

        data = json.loads(result.stdout)

        max_complexity = 0
        for file_data in data.values():
            for item in file_data:
                max_complexity = max(max_complexity, item['complexity'])

        passed = max_complexity <= self.THRESHOLDS['complexity']

        self.results['complexity'] = {
            'passed': passed,
            'value': max_complexity,
            'threshold': self.THRESHOLDS['complexity']
        }

        return passed

    def print_summary(self) -> None:
        """Print quality gate summary"""
        print("\n" + "="*60)
        print("QUALITY GATE SUMMARY")
        print("="*60 + "\n")

        for check, result in self.results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{check.upper()}: {status}")
            print(f"  Value: {result['value']:.2f}")
            print(f"  Threshold: {result['threshold']:.2f}\n")

        overall = all(r['passed'] for r in self.results.values())
        print("="*60)
        print(f"OVERALL: {'✅ PASSED' if overall else '❌ FAILED'}")
        print("="*60 + "\n")

def main():
    gate = QualityGate()
    passed = gate.check_all()
    sys.exit(0 if passed else 1)

if __name__ == '__main__':
    main()
```

## Monitoring and Alerts

### Metrics Dashboard Integration

```python
#!/usr/bin/env python3
"""Send code quality metrics to monitoring dashboard"""

import requests
from datetime import datetime

class MetricsPublisher:
    def __init__(self, dashboard_url: str, api_key: str):
        self.dashboard_url = dashboard_url
        self.api_key = api_key

    def publish(self, metrics: dict) -> None:
        """Publish metrics to dashboard"""
        payload = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }

        response = requests.post(
            f"{self.dashboard_url}/api/metrics",
            json=payload,
            headers={'Authorization': f'Bearer {self.api_key}'}
        )

        response.raise_for_status()

def main():
    # Collect metrics
    metrics = {
        'coverage': get_coverage(),
        'complexity': get_avg_complexity(),
        'duplication': get_duplication_pct(),
        'tech_debt_hours': get_tech_debt_hours(),
        'vulnerabilities': get_vulnerability_count()
    }

    # Publish to dashboard
    publisher = MetricsPublisher(
        dashboard_url='https://metrics.example.com',
        api_key=os.getenv('METRICS_API_KEY')
    )

    publisher.publish(metrics)

if __name__ == '__main__':
    main()
```

## Best Practices

### Automation Guidelines

1. **Start Small**: Begin with simple checks, add complexity gradually
2. **Fast Feedback**: Keep CI checks under 5 minutes when possible
3. **Clear Messages**: Provide actionable error messages
4. **Don't Block**: Use warnings for non-critical issues
5. **Measure Impact**: Track metrics to prove automation value

### Common Pitfalls

- **Over-automation**: Not everything needs to be automated
- **Flaky Tests**: Fix flaky tests before enforcing in CI
- **Slow CI**: Optimize or parallelize slow checks
- **Alert Fatigue**: Only alert on actionable issues
- **Missing Context**: Provide links to documentation in failures
