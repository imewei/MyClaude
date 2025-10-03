# Validation Framework

Comprehensive real-world validation framework for the Claude Code Command Executor Framework.

## Overview

This validation framework tests the Claude Code system against real-world production codebases, ensuring it works reliably across different project sizes, languages, and domains.

## Key Features

- **22 Production Projects**: Curated validation against real open-source projects from 1K to 500K+ LOC
- **12 Validation Scenarios**: Comprehensive test scenarios covering all major use cases
- **Performance Benchmarking**: Baseline collection and regression detection
- **Quality Analysis**: Automated code quality measurement and improvement tracking
- **Multi-format Reports**: HTML, JSON, and Markdown reports with interactive dashboards
- **Continuous Validation**: Automated scheduled validation runs
- **Production Readiness**: Automated pre-deployment validation checklist

## Quick Start

### Run Basic Validation

```bash
# Run validation on all small projects
python validation/executor.py --size small

# Run specific scenario
python validation/executor.py --scenarios code_quality_improvement

# Run on specific projects
python validation/executor.py --projects fastapi flask
```

### View Results

```bash
# Start web dashboard
python validation/dashboard/dashboard.py

# Open in browser: http://localhost:5000
```

### Check Production Readiness

```bash
python validation/production_readiness.py
```

## Architecture

```
validation/
├── suite/                  # Test suite configuration
│   ├── validation_projects.yaml   # 22 curated projects
│   └── validation_scenarios.yaml  # 12 validation scenarios
│
├── executor.py            # Main validation orchestrator
│
├── metrics/               # Metrics collection
│   ├── metrics_collector.py      # Performance metrics
│   └── quality_analyzer.py        # Code quality analysis
│
├── benchmarks/            # Baseline & regression detection
│   ├── baseline_collector.py     # Baseline metrics storage
│   └── regression_detector.py    # Regression detection
│
├── scenarios/             # Scenario implementations
│   └── scenario_runner.py        # Scenario execution engine
│
├── reports/               # Report generation
│   └── report_generator.py       # Multi-format reports
│
├── feedback/              # User feedback system
│   └── feedback_collector.py     # Feedback collection
│
├── edge_cases/            # Edge case detection
│   └── edge_case_detector.py     # Edge case identification
│
├── dashboard/             # Web dashboard
│   └── dashboard.py              # Flask-based dashboard
│
├── continuous/            # Continuous validation
│   └── continuous_validator.py   # Scheduled validation
│
└── production_readiness.py       # Pre-deployment checks
```

## Validation Projects

The framework validates against 22 real-world open-source projects:

### Small Projects (1K-5K LOC)
- Click, Rich, Typer, httpx, attrs

### Medium Projects (10K-50K LOC)
- FastAPI, Flask, SQLAlchemy, pytest, requests

### Large Projects (50K-200K LOC)
- Django, Pandas, scikit-learn, NumPy, Transformers

### Enterprise Projects (200K+ LOC)
- Apache Airflow, Home Assistant, SaltStack

### Specialized
- Multi-language: JupyterLab, Streamlit
- Scientific: SciPy, Matplotlib
- Web Apps: Sentry, Celery
- CLI Tools: Black, Poetry

## Validation Scenarios

12 comprehensive scenarios test the complete system:

1. **Code Quality Improvement** - Run quality checks, auto-fix, validate
2. **Performance Optimization** - Identify and optimize bottlenecks
3. **Test Generation** - Generate comprehensive test suites (≥80% coverage)
4. **Documentation Generation** - Create complete project documentation
5. **Safe Refactoring** - Refactor code with comprehensive validation
6. **Codebase Cleanup** - Remove dead code, unused imports, duplicates
7. **Multi-Agent Analysis** - Run full 23-agent comprehensive analysis
8. **End-to-End Workflow** - Complete development cycle simulation
9. **CI/CD Setup** - Pipeline configuration and validation
10. **Security & Bug Fixing** - Find and fix security issues
11. **Large-Scale Refactoring** - Enterprise-scale refactoring
12. **Code Migration** - Modernization and migration workflows

## Metrics Collected

### Performance Metrics
- Execution time
- Memory usage (current & peak)
- CPU utilization
- Disk I/O (read/write)
- Cache hit rates
- Agent coordination efficiency

### Quality Metrics
- Code quality score (0-100)
- Complexity (cyclomatic)
- Maintainability index
- Test coverage percentage
- Documentation coverage
- Security issues (count & severity)
- Code smells
- Duplication percentage
- Lines of code

### Success Metrics
- Commands successful/failed
- Regression detection
- Improvement percentages
- Test pass rates

## Reports

The framework generates comprehensive reports in multiple formats:

### HTML Dashboard
Interactive web dashboard with:
- Executive summary
- Metrics visualizations
- Project/scenario drill-down
- Historical trends

### JSON Report
Machine-readable format for:
- CI/CD integration
- Automated analysis
- API consumption

### Markdown Report
Human-readable format for:
- Documentation
- GitHub integration
- Team sharing

## Continuous Validation

Run validation automatically on a schedule:

```bash
# Daily validation (default at 2 AM)
python validation/continuous/continuous_validator.py --interval daily

# Hourly validation
python validation/continuous/continuous_validator.py --interval hourly

# Weekly validation
python validation/continuous/continuous_validator.py --interval weekly
```

Features:
- Automatic baseline collection
- Regression detection and alerting
- Historical trend analysis
- Automated reporting

## Regression Detection

The framework automatically detects regressions:

- **Critical**: >50% performance degradation
- **High**: >25% degradation
- **Medium**: >10% degradation
- **Low**: >5% degradation

Fails validation if critical or high regressions detected.

## Production Readiness

Pre-deployment checklist validates:

✅ All tests pass
✅ Coverage ≥90%
✅ Performance meets targets
✅ Security scans clean
✅ Documentation complete
✅ Validation successful
✅ No critical bugs
✅ Dependencies up-to-date

Run: `python validation/production_readiness.py`

## Configuration

### Customize Projects

Edit `validation/suite/validation_projects.yaml`:

```yaml
validation_projects:
  small:
    - name: my_project
      repo: https://github.com/user/repo
      language: python
      loc: ~3000
      domain: CLI tool
      validation_priority: high
```

### Customize Scenarios

Edit `validation/suite/validation_scenarios.yaml`:

```yaml
validation_scenarios:
  my_scenario:
    name: "My Custom Scenario"
    priority: high
    steps:
      - action: analyze
        command: explain-code
        options: [--level=advanced]
```

## Advanced Usage

### Parallel Execution

```bash
# Run 5 validations in parallel
python validation/executor.py --parallel 5
```

### Custom Output Directory

```bash
python validation/executor.py --output-dir /path/to/reports
```

### Specific Report Formats

```bash
python validation/executor.py --formats html json
```

### Cache Management

```bash
# Clean old cached projects
python validation/executor.py --cleanup-cache
```

## Integration

### CI/CD Integration

```yaml
# .github/workflows/validation.yml
name: Validation
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Validation
        run: python validation/executor.py --size small
      - name: Upload Reports
        uses: actions/upload-artifact@v2
        with:
          name: validation-reports
          path: validation/reports/
```

### API Integration

```python
from validation.executor import ValidationExecutor

executor = ValidationExecutor(parallel_jobs=3)
results = executor.run_validation(
    project_filter={'fastapi', 'flask'},
    scenario_filter={'code_quality_improvement'}
)

for result in results:
    print(f"{result.project_name}: {'✓' if result.success else '✗'}")
```

## Troubleshooting

### Validation Fails

1. Check logs: `validation/logs/validation_*.log`
2. Run with single project: `--projects fastapi`
3. Check cache: `validation/cache/`

### Low Success Rate

1. Review regression report
2. Check baseline metrics
3. Validate against known-good projects first

### Performance Issues

1. Reduce parallel jobs: `--parallel 1`
2. Use size filter: `--size small`
3. Clean cache: `--cleanup-cache`

## Requirements

- Python 3.10+
- psutil
- pyyaml
- flask (for dashboard)
- schedule (for continuous validation)

Install: `pip install -r validation/requirements.txt`

## Contributing

To add new validation projects:
1. Edit `validation/suite/validation_projects.yaml`
2. Ensure project is publicly accessible
3. Test with: `python validation/executor.py --projects your_project`

## License

Same as parent project.

## Support

For issues, see: `/Users/b80985/.claude/commands/validation/logs/`