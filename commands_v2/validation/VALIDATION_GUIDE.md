# Validation Guide

Complete guide to running and interpreting validation results.

## Running Validation

### Basic Validation Run

```bash
# Run all validations (takes ~4-6 hours)
python validation/executor.py

# Run small projects only (~30 minutes)
python validation/executor.py --size small

# Run medium projects (~1-2 hours)
python validation/executor.py --size medium

# Run specific scenario
python validation/executor.py --scenarios code_quality_improvement
```

### Targeted Validation

```bash
# Single project, single scenario
python validation/executor.py \
    --projects fastapi \
    --scenarios code_quality_improvement

# Multiple projects
python validation/executor.py \
    --projects fastapi flask django

# Multiple scenarios
python validation/executor.py \
    --scenarios code_quality_improvement performance_optimization
```

### Performance Options

```bash
# Increase parallelism (default: 3)
python validation/executor.py --parallel 5

# Single-threaded (for debugging)
python validation/executor.py --parallel 1
```

## Understanding Results

### Success Criteria

A validation passes if:
- All scenario steps complete successfully
- No critical errors occur
- Success criteria thresholds met:
  - Quality improvement ≥20%
  - Performance improvement ≥2x (for optimization scenarios)
  - Test coverage ≥80% (for test generation)
  - No regressions detected

### Report Interpretation

#### HTML Report

Open `validation/reports/YYYYMMDD_HHMMSS/validation_report.html`:

- **Executive Summary**: High-level metrics
- **Success Rate**: Percentage of validations passed
- **Duration**: Total and average execution time
- **Detailed Results**: Per-project/scenario breakdown

#### JSON Report

```json
{
  "timestamp": "2025-09-29T...",
  "summary": {
    "total": 120,
    "successful": 108,
    "failed": 12,
    "success_rate": 90.0
  },
  "results": [...]
}
```

Use for:
- Automated processing
- CI/CD integration
- Trend analysis

#### Markdown Report

Human-readable format with:
- Project-grouped results
- Error/warning details
- Metrics table

### Common Failure Reasons

1. **Clone Failure**
   - Network issues
   - Repository unavailable
   - Solution: Check internet connection, retry

2. **Timeout**
   - Project too large
   - Command hung
   - Solution: Increase timeout in config

3. **Dependency Issues**
   - Missing dependencies in validation project
   - Solution: Expected, move to next project

4. **Success Criteria Not Met**
   - Quality improvement <20%
   - Coverage <80%
   - Solution: Review thresholds in scenarios.yaml

## Analyzing Metrics

### Quality Metrics

- **Overall Score**: 0-100 composite score
  - 90-100: Excellent
  - 70-89: Good
  - 50-69: Fair
  - <50: Needs improvement

- **Complexity Score**: Lower is better
  - Complex functions >10 cyclomatic complexity
  - Target: <10% complex functions

- **Documentation Score**: Percentage of documented functions
  - Target: ≥70%

### Performance Metrics

- **Execution Time**: Time to complete scenario
  - Compare to baseline
  - Look for >2x improvements in optimization scenarios

- **Memory Usage**: Peak memory during execution
  - Monitor for memory leaks
  - Compare to baseline

- **Cache Hit Rate**: Efficiency of caching system
  - Target: >75%

## Baseline Management

### Collecting Baselines

First run automatically collects baselines:

```bash
python validation/executor.py --size small
```

Baselines stored in: `validation/data/baselines.db`

### Comparing to Baseline

Subsequent runs compare to baseline:
- Regression detected if performance degrades >10%
- Quality regression if score drops >5 points

### Resetting Baselines

```bash
rm validation/data/baselines.db
python validation/executor.py --size small
```

## Regression Analysis

### Severity Levels

- **Critical**: >50% degradation - Blocks deployment
- **High**: >25% degradation - Requires investigation
- **Medium**: >10% degradation - Monitor closely
- **Low**: >5% degradation - Document for tracking

### Investigating Regressions

1. Check regression report in logs
2. Review specific metrics:
   ```python
   from validation.benchmarks.regression_detector import RegressionDetector

   detector = RegressionDetector()
   regressions = detector.detect('fastapi', 'code_quality', current_metrics)
   ```

3. Compare baseline vs current:
   - Execution time
   - Memory usage
   - Quality scores

4. Root cause analysis:
   - Recent code changes
   - Dependency updates
   - Configuration changes

## Dashboard Usage

### Starting Dashboard

```bash
python validation/dashboard/dashboard.py

# Custom port
python validation/dashboard/dashboard.py --port 8080
```

Access at: `http://localhost:5000`

### Dashboard Features

- **Real-time Status**: Current validation state
- **Metrics Cards**: Key metrics at a glance
- **Results Table**: Detailed per-validation results
- **Auto-refresh**: Updates every 30 seconds

### API Endpoints

- `/api/summary` - Summary statistics
- `/api/results` - Detailed results
- `/api/trends` - Historical trends (if implemented)

## Continuous Validation

### Setup

```bash
# Start continuous validation
python validation/continuous/continuous_validator.py --interval daily
```

### Monitoring

Logs: `validation/logs/continuous/continuous_YYYYMMDD.log`

### Alerting

Alerts triggered when:
- Validation fails
- Critical regressions detected
- Success rate drops below 80%

Configure alerts in `continuous_validator.py`

## Best Practices

### Initial Validation

1. Start with small projects
2. Run each scenario individually
3. Establish baselines
4. Fix any failures before scaling up

### Regular Validation

1. Run daily on small/medium projects
2. Run weekly on large/enterprise projects
3. Monitor trends over time
4. Investigate regressions immediately

### Pre-release Validation

1. Run full validation suite
2. Check production readiness
3. Review all reports
4. Ensure zero critical regressions

## Troubleshooting

### High Failure Rate

**Symptoms**: <80% success rate

**Solutions**:
1. Check individual failures in logs
2. Verify baseline metrics are reasonable
3. Review success criteria thresholds
4. Test on known-good projects first

### Performance Issues

**Symptoms**: Validation takes too long

**Solutions**:
1. Reduce parallel jobs
2. Use size filters
3. Run subset of scenarios
4. Clean cache regularly

### Memory Issues

**Symptoms**: Out of memory errors

**Solutions**:
1. Reduce parallel jobs to 1-2
2. Run smaller projects first
3. Increase system memory
4. Run validation on dedicated machine

### Network Issues

**Symptoms**: Clone failures

**Solutions**:
1. Check internet connection
2. Use cached projects
3. Clone projects manually first
4. Increase clone timeout

## Advanced Topics

### Custom Scenarios

Create custom scenario in `validation_scenarios.yaml`:

```yaml
my_custom_scenario:
  name: "Custom Validation"
  priority: high
  estimated_duration_minutes: 15
  steps:
    - action: analyze
      command: check-code-quality
      options: [--language=python]

    - action: validate
      command: run-all-tests
      options: [--coverage]

  success_criteria:
    quality_score: ">= 70"
    coverage: ">= 80"
```

### Custom Metrics

Add custom metrics in `metrics_collector.py`:

```python
def collect_custom_metric(self, project_path: Path) -> float:
    # Your custom metric logic
    return metric_value
```

### Integration Testing

Use validation in integration tests:

```python
import unittest
from validation.executor import ValidationExecutor

class ValidationTests(unittest.TestCase):
    def test_fastapi_quality(self):
        executor = ValidationExecutor()
        results = executor.run_validation(
            project_filter={'fastapi'},
            scenario_filter={'code_quality_improvement'}
        )

        self.assertTrue(all(r.success for r in results))
```

## Support

- Logs: `validation/logs/`
- Database: `validation/data/`
- Cache: `validation/cache/`
- Reports: `validation/reports/`