# Validation Framework - Quick Reference

## Installation

```bash
cd /Users/b80985/.claude/commands/validation
pip install -r requirements.txt
```

## Quick Commands

### Basic Validation
```bash
# Small projects (fastest)
python executor.py --size small

# Single project
python executor.py --projects fastapi

# Single scenario
python executor.py --scenarios code_quality_improvement

# Multiple
python executor.py --projects fastapi flask --scenarios code_quality_improvement performance_optimization
```

### Dashboard
```bash
# Start dashboard
python dashboard/dashboard.py

# Open http://localhost:5000
```

### Production Check
```bash
# Pre-deployment validation
python production_readiness.py
```

### Continuous
```bash
# Daily validation
python continuous/continuous_validator.py --interval daily
```

## Project Sizes

- **Small** (1K-5K LOC): Click, Rich, Typer, httpx, attrs
- **Medium** (10K-50K LOC): FastAPI, Flask, SQLAlchemy, pytest, requests
- **Large** (50K-200K LOC): Django, Pandas, scikit-learn, NumPy
- **Enterprise** (200K+ LOC): Airflow, Home Assistant, Salt

## Scenarios

1. `code_quality_improvement` - Quality analysis and fixes
2. `performance_optimization` - Performance improvements
3. `test_generation` - Test suite generation
4. `documentation_generation` - Doc generation
5. `safe_refactoring` - Safe refactoring
6. `codebase_cleanup` - Code cleanup
7. `multi_agent_analysis` - Full agent analysis
8. `end_to_end_workflow` - Complete workflow
9. `cicd_setup` - CI/CD setup
10. `security_bug_fixing` - Security fixes
11. `large_scale_refactoring` - Large refactoring
12. `code_migration` - Code migration

## Options

```bash
--size small|medium|large|enterprise    # Filter by size
--projects NAME [NAME ...]              # Specific projects
--scenarios NAME [NAME ...]             # Specific scenarios
--parallel N                            # Parallel jobs (default: 3)
--output-dir PATH                       # Output directory
--formats html json markdown            # Report formats
--cleanup-cache                         # Clean cache
```

## Success Criteria

- Quality improvement: ≥20%
- Performance improvement: ≥2x
- Test coverage: ≥80%
- Documentation: ≥70%
- Zero critical regressions

## Metrics

### Performance
- Execution time (seconds)
- Memory usage (MB)
- CPU utilization (%)
- Disk I/O (MB)
- Cache hit rate (%)

### Quality
- Overall score (0-100)
- Complexity score (0-100)
- Test coverage (%)
- Documentation coverage (%)
- Security score (0-100)

## Reports Location

```
validation/reports/YYYYMMDD_HHMMSS/
├── validation_report.html      # Interactive dashboard
├── validation_report.json      # Machine-readable
└── validation_report.md        # Human-readable
```

## Logs Location

```
validation/logs/
├── validation_YYYYMMDD_HHMMSS.log    # Main logs
└── continuous/                        # Continuous logs
```

## Data Location

```
validation/data/
├── baselines.db     # Baseline metrics
└── feedback.db      # User feedback
```

## Troubleshooting

### High Failure Rate
```bash
# Test single project
python executor.py --projects click --parallel 1

# Check logs
tail -f logs/validation_*.log
```

### Performance Issues
```bash
# Reduce parallelism
python executor.py --size small --parallel 1

# Clean cache
python executor.py --cleanup-cache
```

### View Results
```bash
# Latest HTML report
open validation/reports/*/validation_report.html

# Dashboard
python dashboard/dashboard.py
```

## API Usage

```python
from validation import ValidationExecutor

# Create executor
executor = ValidationExecutor(parallel_jobs=3)

# Run validation
results = executor.run_validation(
    project_filter={'fastapi'},
    scenario_filter={'code_quality_improvement'}
)

# Check results
for result in results:
    print(f"{result.project_name}: {'✓' if result.success else '✗'}")
    print(f"  Quality: {result.metrics.get('quality_score')}")
    print(f"  Duration: {result.duration_seconds:.1f}s")

# Generate reports
report_paths = executor.generate_report(formats=['html', 'json'])
```

## Adding Projects

Edit `suite/validation_projects.yaml`:

```yaml
validation_projects:
  small:  # or medium, large, enterprise
    - name: my_project
      repo: https://github.com/user/repo
      language: python
      loc: ~3000
      domain: cli_tool
      validation_priority: high
```

## Custom Scenarios

Edit `suite/validation_scenarios.yaml`:

```yaml
validation_scenarios:
  my_scenario:
    name: "My Custom Scenario"
    priority: high
    steps:
      - action: analyze
        command: check-code-quality
        options: [--language=python]
    success_criteria:
      quality_score: ">= 70"
```

## File Structure

```
validation/
├── executor.py              # Main orchestrator
├── production_readiness.py  # Pre-deployment checks
├── suite/                   # Projects & scenarios
├── metrics/                 # Metrics collection
├── benchmarks/              # Baselines & regression
├── reports/                 # Report generation
├── dashboard/               # Web dashboard
├── continuous/              # Continuous validation
├── data/                    # SQLite databases
├── logs/                    # Execution logs
└── docs/                    # Documentation
```

## Key Documentation

- **README.md** - Main documentation
- **VALIDATION_GUIDE.md** - Complete usage guide
- **METRICS_GUIDE.md** - Metrics reference
- **ADDING_PROJECTS.md** - Add projects guide
- **FRAMEWORK_SUMMARY.md** - Complete summary

## Support

- Logs: `validation/logs/`
- Issues: Check error messages in logs
- Documentation: See guides in validation/
- Help: `python executor.py --help`