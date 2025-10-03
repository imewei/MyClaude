# Contributing Tests - Guidelines

Guidelines for writing tests for the Command Executor Framework.

## Testing Philosophy

### Core Principles

1. **Test Behavior, Not Implementation** - Focus on what the code does, not how it does it
2. **Comprehensive Coverage** - Aim for 90%+ coverage, 95%+ for critical components
3. **Fast Feedback** - Unit tests should be fast (< 100ms), integration tests < 5s
4. **Realistic Scenarios** - Workflow tests should mirror real-world usage
5. **Clear Intent** - Test names and structure should clearly communicate purpose

### Test Pyramid

```
          /\
         /  \   10% - Workflow Tests (E2E, slow, realistic)
        /    \
       /------\  30% - Integration Tests (component interaction)
      /        \
     /----------\ 60% - Unit Tests (fast, isolated, focused)
```

## Test Structure

### File Organization

```python
#!/usr/bin/env python3
"""
Module: test_<component>_<category>.py
Brief description of what this test module covers.

Test Classes:
- TestComponentName: Main functionality tests
- TestComponentEdgeCases: Edge case and error handling
- TestComponentIntegration: Integration with other components
"""

import pytest
from pathlib import Path
from typing import Dict, Any

# Import components to test
from executors.component import Component


@pytest.mark.unit
@pytest.mark.fast
class TestComponentName:
    """Test Component main functionality"""

    def test_basic_operation(self):
        """Test basic operation succeeds"""
        # Arrange
        component = Component()

        # Act
        result = component.operate()

        # Assert
        assert result.success
        assert result.value == expected_value

    def test_operation_with_options(self):
        """Test operation with various options"""
        # Test implementation
        pass
```

### Naming Conventions

#### Test Files

- `test_<component>_<category>.py` - e.g., `test_cache_manager_integration.py`
- Place in appropriate directory: `unit/`, `integration/`, `workflows/`, `performance/`

#### Test Classes

- `TestComponentName` - Main functionality
- `TestComponentEdgeCases` - Edge cases
- `TestComponentErrors` - Error handling
- `TestComponentIntegration` - Integration scenarios
- `TestComponentPerformance` - Performance tests

#### Test Methods

- `test_<action>_<expected_result>` - e.g., `test_cache_hit_returns_cached_data`
- Use descriptive names that explain the scenario
- Avoid generic names like `test_method1`

**Good Examples:**
```python
def test_backup_creation_succeeds_for_valid_directory(self):
def test_rollback_fails_gracefully_when_backup_not_found(self):
def test_parallel_execution_achieves_3x_speedup_with_4_workers(self):
```

**Bad Examples:**
```python
def test_backup(self):
def test_error(self):
def test_1(self):
```

## Test Types

### Unit Tests

Test individual components in isolation.

```python
@pytest.mark.unit
@pytest.mark.fast
class TestCacheManager:
    """Unit tests for CacheManager"""

    def test_cache_set_stores_value(self, temp_workspace: Path):
        """Test that set() stores value in cache"""
        cache = CacheManager(temp_workspace / "cache")

        cache.set("key", {"data": "value"}, level="default")

        # Verify file was created
        cache_file = temp_workspace / "cache" / "default" / "key.json"
        assert cache_file.exists()

    def test_cache_get_returns_none_for_missing_key(self, temp_workspace: Path):
        """Test that get() returns None for missing key"""
        cache = CacheManager(temp_workspace / "cache")

        result = cache.get("nonexistent", level="default")

        assert result is None
```

**Characteristics:**
- **Fast**: < 100ms per test
- **Isolated**: No external dependencies
- **Focused**: Test one thing
- **Deterministic**: Same input → same output

### Integration Tests

Test interaction between components.

```python
@pytest.mark.integration
@pytest.mark.framework
class TestExecutorIntegration:
    """Integration tests for executor pipeline"""

    def test_execution_with_backup_and_cache(
        self,
        temp_workspace: Path,
        backup_system: BackupSystem
    ):
        """Test execution with backup and caching enabled"""
        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        # Execute with backup
        result = executor.execute({
            "implement": True,
            "create_backup": True
        })

        assert result.success
        assert "backup_id" in executor.context.metadata

        # Verify backup was created
        backups = backup_system.list_backups()
        assert len(backups) > 0
```

**Characteristics:**
- **Moderate Speed**: < 5s per test
- **Multiple Components**: Test interaction
- **Realistic**: Use real implementations
- **State Management**: May require setup/teardown

### Workflow Tests

Test complete real-world scenarios.

```python
@pytest.mark.workflow
@pytest.mark.slow
class TestDevelopmentWorkflow:
    """Test complete development workflow"""

    def test_full_dev_cycle(self, sample_python_project: Path):
        """Test: quality check → optimize → test → commit"""
        # Step 1: Quality check
        quality_result = run_quality_check(sample_python_project)
        assert quality_result["success"]

        # Step 2: Optimize based on findings
        optimize_result = run_optimization(
            sample_python_project,
            issues=quality_result["issues"]
        )
        assert optimize_result["success"]

        # Step 3: Generate and run tests
        test_result = generate_and_run_tests(sample_python_project)
        assert test_result["all_passed"]

        # Step 4: Commit if all passed
        if test_result["all_passed"]:
            commit_result = commit_changes(sample_python_project)
            assert commit_result["success"]
```

**Characteristics:**
- **Slow**: May take several seconds
- **End-to-End**: Complete user journeys
- **Multiple Commands**: Chain operations
- **Realistic Data**: Use sample projects

### Performance Tests

Benchmark and verify performance targets.

```python
@pytest.mark.performance
@pytest.mark.benchmark
class TestCachePerformance:
    """Performance benchmarks for cache system"""

    def test_cache_achieves_5x_speedup(self, temp_workspace: Path):
        """Verify cache achieves target 5x speedup"""
        cache = CacheManager(temp_workspace / "cache")

        # Measure without cache
        start = time.time()
        for _ in range(100):
            expensive_operation()
        no_cache_time = time.time() - start

        # Measure with cache
        cache.set("result", expensive_operation())
        start = time.time()
        for _ in range(100):
            cache.get("result")
        cache_time = time.time() - start

        speedup = no_cache_time / cache_time
        assert speedup >= 5.0, f"Speedup {speedup:.2f}x < 5x target"

    def test_parallel_execution_scalability(self, benchmark):
        """Benchmark parallel execution scalability"""
        executor = ParallelExecutor(max_workers=4)

        def parallel_work():
            return executor.execute_parallel(task, items, max_workers=4)

        result = benchmark(parallel_work)
        assert result is not None
```

**Characteristics:**
- **Quantifiable**: Measure specific metrics
- **Targets**: Verify performance goals
- **Repeatable**: Consistent measurements
- **Baseline**: Compare against baselines

## Test Coverage

### Coverage Requirements

| Component | Target | Critical |
|-----------|--------|----------|
| Core Framework | 95% | Yes |
| Agent System | 90% | Yes |
| Safety Manager | 95% | Yes |
| Command Executors | 90% | No |
| Performance System | 85% | No |
| Utilities | 80% | No |

### Measuring Coverage

```bash
# Generate coverage report
pytest --cov=executors --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html

# Check coverage threshold
pytest --cov=executors --cov-fail-under=90
```

### What to Cover

**Must Cover:**
- ✅ All public methods and functions
- ✅ All error paths and exception handling
- ✅ Edge cases and boundary conditions
- ✅ Different configuration options
- ✅ Integration points between components

**Can Skip:**
- ❌ Private implementation details (unless critical)
- ❌ Third-party library code
- ❌ Simple getters/setters
- ❌ Abstract base classes (test via implementations)
- ❌ Code explicitly marked `# pragma: no cover`

### Improving Coverage

1. **Identify gaps**: `coverage report --show-missing`
2. **Focus on uncovered branches**: `pytest --cov-branch`
3. **Test error paths**: Explicitly test exception handling
4. **Test edge cases**: Boundary values, empty inputs, None values
5. **Integration scenarios**: Test component interaction

## Using Fixtures

### Built-in Fixtures

```python
def test_with_fixtures(
    temp_workspace: Path,              # Temporary directory
    sample_python_project: Path,       # Sample Python codebase
    backup_system: BackupSystem,       # Configured backup system
    cache_manager: CacheManager,       # Cache manager
    agent_orchestrator: AgentOrchestrator,  # Agent orchestrator
):
    """Test using multiple fixtures"""
    # Fixtures are automatically provided
    pass
```

### Creating Custom Fixtures

```python
# In conftest.py or test file

@pytest.fixture
def custom_executor(temp_workspace: Path) -> CustomExecutor:
    """Create configured custom executor"""
    executor = CustomExecutor()
    executor.work_dir = temp_workspace
    executor.setup()

    yield executor

    # Cleanup
    executor.cleanup()


# Use in tests
def test_with_custom_executor(custom_executor: CustomExecutor):
    result = custom_executor.execute({})
    assert result.success
```

### Fixture Scopes

```python
@pytest.fixture(scope="function")  # Default: recreated for each test
def per_test_fixture():
    return setup()

@pytest.fixture(scope="class")  # Shared across test class
def per_class_fixture():
    return setup()

@pytest.fixture(scope="module")  # Shared across test module
def per_module_fixture():
    return setup()

@pytest.fixture(scope="session")  # Shared across entire test session
def per_session_fixture():
    return setup()
```

## Best Practices

### 1. Arrange-Act-Assert Pattern

```python
def test_example(self):
    """Test example using AAA pattern"""
    # Arrange - Set up test conditions
    executor = Executor()
    input_data = {"key": "value"}

    # Act - Execute the operation
    result = executor.execute(input_data)

    # Assert - Verify expectations
    assert result.success
    assert result.output["key"] == "processed_value"
```

### 2. One Concept Per Test

```python
# Good - Tests one thing
def test_cache_hit_returns_cached_data(self):
    cache.set("key", "value")
    result = cache.get("key")
    assert result == "value"

# Bad - Tests multiple things
def test_cache_operations(self):
    cache.set("key", "value")
    assert cache.get("key") == "value"
    cache.invalidate("key")
    assert cache.get("key") is None
    cache.set("key2", "value2")
    assert cache.get("key2") == "value2"
```

### 3. Test Independence

```python
# Good - Independent tests
class TestCache:
    def test_set(self, cache_manager):
        cache_manager.set("key", "value")
        assert cache_manager.get("key") == "value"

    def test_get_missing_key(self, cache_manager):
        # Doesn't depend on test_set
        assert cache_manager.get("missing") is None

# Bad - Tests depend on order
class TestCache:
    def test_1_set(self):
        self.cache.set("key", "value")  # Shared state

    def test_2_get(self):
        assert self.cache.get("key") == "value"  # Depends on test_1
```

### 4. Descriptive Assertions

```python
# Good - Clear assertion messages
assert len(results) == 5, f"Expected 5 results, got {len(results)}"
assert speedup >= 3.0, f"Speedup {speedup:.2f}x below 3x target"

# Bad - No context when failing
assert len(results) == 5
assert speedup >= 3.0
```

### 5. Use Parametrize for Similar Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("test", "TEST"),
])
def test_uppercase(input, expected):
    """Test uppercase conversion"""
    assert input.upper() == expected
```

### 6. Mock External Dependencies

```python
def test_with_mock(mocker):
    """Test using mock for external API"""
    mock_api = mocker.patch('module.external_api.call')
    mock_api.return_value = {"status": "success"}

    result = function_that_calls_api()

    assert result.success
    mock_api.assert_called_once()
```

## Performance Testing

### Setting Targets

```python
# Define performance targets as constants
CACHE_SPEEDUP_TARGET = 5.0  # 5x speedup
PARALLEL_SPEEDUP_TARGET = 3.0  # 3x speedup
SIMPLE_COMMAND_TIME_TARGET = 0.2  # 200ms
```

### Measuring Performance

```python
def test_performance_target(self):
    """Test meets performance target"""
    import time

    # Measure baseline
    start = time.time()
    baseline_operation()
    baseline_time = time.time() - start

    # Measure optimized
    start = time.time()
    optimized_operation()
    optimized_time = time.time() - start

    # Verify improvement
    speedup = baseline_time / optimized_time
    assert speedup >= TARGET_SPEEDUP, (
        f"Speedup {speedup:.2f}x below target {TARGET_SPEEDUP}x"
    )
```

### Using pytest-benchmark

```python
def test_benchmark(benchmark):
    """Benchmark using pytest-benchmark"""
    result = benchmark(expensive_operation)
    assert result is not None

# Run benchmarks
# pytest --benchmark-only
# pytest --benchmark-compare=baseline
```

## CI/CD Integration

### Pre-commit Checks

```bash
# Run before committing
pytest -m "fast or unit"
pytest --cov=executors --cov-fail-under=90
```

### Pull Request Checks

```bash
# Run on PR
pytest -m "not slow"
pytest --cov=executors --cov-report=xml
```

### Full Suite

```bash
# Run before merge
pytest
pytest --cov=executors --cov-report=html
```

## Common Pitfalls

### ❌ Testing Implementation Details

```python
# Bad - Tests internal implementation
def test_cache_uses_json_files(self):
    cache.set("key", "value")
    cache_file = cache.cache_dir / "default" / "key.json"
    assert cache_file.exists()
```

### ✅ Testing Behavior

```python
# Good - Tests observable behavior
def test_cache_retrieves_stored_value(self):
    cache.set("key", "value")
    assert cache.get("key") == "value"
```

### ❌ Fragile Tests

```python
# Bad - Depends on timing, file system state
def test_creates_5_files(self):
    operation()
    assert len(list(Path(".").glob("*"))) == 5
```

### ✅ Robust Tests

```python
# Good - Uses controlled environment
def test_creates_expected_files(self, temp_workspace):
    operation(temp_workspace)
    expected_files = {"file1.txt", "file2.txt", "file3.txt"}
    actual_files = {f.name for f in temp_workspace.glob("*")}
    assert actual_files == expected_files
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Test-Driven Development](https://www.obeythetestinggoat.com/)

## Questions?

For questions about testing:
1. Check existing tests for examples
2. Review this guide
3. Ask in team discussions
4. Open an issue for clarification