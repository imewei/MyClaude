# Command Executor Framework - Testing Suite

Comprehensive integration testing suite for the 14-command executor framework with 23-agent orchestration system.

## Overview

This testing suite provides:

- **90%+ Code Coverage Target**
- **Integration Tests** - Full workflow testing
- **Unit Tests** - Component-level testing
- **Performance Benchmarks** - Cache, parallel execution, agent orchestration
- **Real-World Workflows** - Complete development scenarios
- **Safety Testing** - Backup, rollback, validation

## Quick Start

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=executors --cov-report=html

# Run specific test categories
pytest -m integration    # Integration tests
pytest -m unit          # Unit tests
pytest -m workflow      # Workflow tests
pytest -m performance   # Performance benchmarks

# Run fast tests only
pytest -m fast

# Run tests in parallel
pytest -n auto
```

## Test Structure

```
tests/
├── conftest.py                  # Shared fixtures and configuration
├── pytest.ini                   # Pytest configuration
├── tox.ini                      # Multi-environment testing
├── .coveragerc                  # Coverage configuration
│
├── integration/                 # Integration tests
│   ├── test_framework_integration.py        # Core framework
│   ├── test_agent_orchestration.py          # 23-agent system
│   ├── test_safety_features.py              # Backup/rollback
│   ├── test_performance_system.py           # Cache/parallel
│   ├── test_validation_engine.py            # Validation pipeline
│   ├── test_update_docs_integration.py      # Update-docs command
│   ├── test_optimize_integration.py         # Optimize command
│   ├── test_clean_codebase_integration.py   # Clean-codebase command
│   └── ... (14 command executors total)
│
├── unit/                        # Unit tests
│   ├── test_base_executor.py
│   ├── test_agent_system.py
│   ├── test_safety_manager.py
│   ├── test_cache_manager.py
│   └── test_validation_engine.py
│
├── workflows/                   # Real-world workflow tests
│   ├── test_complete_workflows.py           # Multi-command workflows
│   ├── test_development_workflow.py         # Dev cycle
│   ├── test_documentation_workflow.py       # Docs generation
│   └── test_refactoring_workflow.py         # Safe refactoring
│
├── performance/                 # Performance benchmarks
│   ├── test_performance_benchmarks.py       # Comprehensive benchmarks
│   ├── test_cache_performance.py            # Cache system
│   └── test_parallel_performance.py         # Parallel execution
│
└── fixtures/                    # Test data and mocks
    ├── sample_python_project/
    ├── sample_julia_project/
    ├── sample_jax_project/
    ├── mock_git_repo/
    └── quality_issues/
```

## Test Categories

### Integration Tests

Test complete execution flows:

```python
# Framework integration
pytest tests/integration/test_framework_integration.py

# Agent orchestration
pytest tests/integration/test_agent_orchestration.py

# Safety features
pytest tests/integration/test_safety_features.py
```

### Workflow Tests

Test real-world development workflows:

```python
# Complete development cycle
pytest tests/workflows/test_complete_workflows.py::TestDevelopmentWorkflow

# Documentation workflow
pytest tests/workflows/test_complete_workflows.py::TestDocumentationWorkflow

# Refactoring workflow
pytest tests/workflows/test_complete_workflows.py::TestRefactoringWorkflow
```

### Performance Benchmarks

Measure and verify performance targets:

```python
# Cache performance (target: 5-8x speedup)
pytest tests/performance/test_performance_benchmarks.py::TestCachePerformance

# Parallel execution (target: 3-5x speedup)
pytest tests/performance/test_performance_benchmarks.py::TestParallelExecutionPerformance

# Agent orchestration efficiency
pytest tests/performance/test_performance_benchmarks.py::TestAgentOrchestrationPerformance
```

## Test Fixtures

### Workspace Fixtures

- `temp_workspace` - Temporary workspace directory
- `mock_git_repo` - Mock git repository
- `sample_python_project` - Python project with tests
- `sample_julia_project` - Julia project
- `sample_jax_project` - JAX/ML project

### Component Fixtures

- `backup_system` - Configured backup system
- `dry_run_executor` - Dry-run executor
- `agent_orchestrator` - Agent orchestrator
- `cache_manager` - Cache manager
- `validation_engine` - Validation engine

### Mock Fixtures

- `mock_claude_client` - Mock Claude API client
- `mock_git_operations` - Mock git operations

## Coverage

### Coverage Targets

- **Overall Coverage**: 90%+
- **Core Framework**: 95%+
- **Agent System**: 90%+
- **Safety Manager**: 95%+
- **Performance System**: 85%+

### Generate Coverage Report

```bash
# HTML report
pytest --cov=executors --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=executors --cov-report=term-missing

# XML report (for CI/CD)
pytest --cov=executors --cov-report=xml
```

### Coverage Analysis

```bash
# Show uncovered lines
coverage report --show-missing

# Coverage by module
coverage report --sort=cover

# Detailed coverage
coverage html
```

## Performance Benchmarks

### Performance Targets

1. **Cache System**: 5-8x speedup for cached results
2. **Parallel Execution**: 3-5x speedup with 4 workers
3. **Agent Orchestration**: < 500ms for core 5-agent team
4. **Simple Command**: < 200ms execution time

### Running Benchmarks

```bash
# All benchmarks
pytest tests/performance/ --benchmark-only

# Cache benchmarks
pytest tests/performance/test_performance_benchmarks.py::TestCachePerformance

# Parallel benchmarks
pytest tests/performance/test_performance_benchmarks.py::TestParallelExecutionPerformance

# Generate benchmark report
pytest tests/performance/ --benchmark-autosave --benchmark-save=baseline
```

### Benchmark Results

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed baseline measurements.

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e .[test]
      - name: Run tests
        run: pytest --cov=executors --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Tox Multi-Environment Testing

```bash
# Test all environments
tox

# Specific environment
tox -e py311

# Linting
tox -e lint

# Type checking
tox -e type

# Coverage
tox -e coverage
```

## Writing Tests

### Test Structure

```python
import pytest
from pathlib import Path

@pytest.mark.integration
@pytest.mark.framework
class TestMyFeature:
    """Test my feature"""

    def test_basic_functionality(self, temp_workspace: Path):
        """Test basic functionality"""
        # Arrange
        setup_data()

        # Act
        result = execute_feature()

        # Assert
        assert result.success
        assert result.details["key"] == "value"

    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            execute_with_invalid_input()
```

### Using Fixtures

```python
def test_with_fixtures(
    temp_workspace: Path,
    backup_system: BackupSystem,
    agent_orchestrator: AgentOrchestrator
):
    """Test using multiple fixtures"""
    # Fixtures are automatically provided by pytest
    backup_id = backup_system.create_backup(temp_workspace, "test")
    agents = agent_orchestrator.select_agents([AgentType.CORE], context)

    assert backup_id is not None
    assert len(agents) > 0
```

### Markers

Use markers to categorize tests:

```python
@pytest.mark.unit           # Unit test
@pytest.mark.integration    # Integration test
@pytest.mark.workflow       # Workflow test
@pytest.mark.performance    # Performance test
@pytest.mark.slow           # Slow test
@pytest.mark.fast           # Fast test
```

## Troubleshooting

### Tests Failing

1. **Check dependencies**: `pip install -e .[test]`
2. **Clear cache**: `pytest --cache-clear`
3. **Verbose output**: `pytest -v`
4. **Show print statements**: `pytest -s`
5. **Debug mode**: `pytest --pdb`

### Coverage Issues

1. **Missing coverage**: Check `.coveragerc` exclude patterns
2. **Import errors**: Verify PYTHONPATH includes parent directory
3. **Parallel coverage**: Use `coverage combine` after parallel runs

### Performance Issues

1. **Slow tests**: Use `pytest --durations=10` to identify
2. **Run fast tests only**: `pytest -m fast`
3. **Parallel execution**: `pytest -n auto`

## Contributing Tests

See [CONTRIBUTING_TESTS.md](CONTRIBUTING_TESTS.md) for detailed guidelines on:

- Writing new tests
- Test naming conventions
- Coverage requirements
- Performance testing
- CI/CD integration

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [tox Documentation](https://tox.wiki/)

## License

Copyright 2025 Claude Code Framework. All rights reserved.