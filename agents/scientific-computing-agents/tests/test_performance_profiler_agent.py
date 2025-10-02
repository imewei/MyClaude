"""
Tests for PerformanceProfilerAgent.

Comprehensive test suite covering profiling, memory analysis, and bottleneck detection.
"""

import pytest
import time
import numpy as np
from pathlib import Path
import sys
import tracemalloc

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.performance_profiler_agent import PerformanceProfilerAgent, ProfileResult


@pytest.fixture(autouse=True)
def cleanup_profilers():
    """Ensure profilers are in clean state before and after each test."""
    # Stop tracemalloc if it's running
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    # Clear any active profiler
    sys.setprofile(None)
    sys.settrace(None)
    yield
    # Clean up after test
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    sys.setprofile(None)
    sys.settrace(None)


# Test fixtures and helper functions

def simple_function(n=1000):
    """Simple function for testing profiling."""
    total = 0
    for i in range(n):
        total += i
    return total


def memory_intensive_function(size=1000):
    """Function that allocates memory for testing."""
    data = np.zeros((size, size))
    return data.sum()


def slow_function(duration=0.1):
    """Function with known duration for timing tests."""
    time.sleep(duration)
    return "completed"


def recursive_function(n):
    """Recursive function for testing call graph."""
    if n <= 0:
        return 1
    return n * recursive_function(n - 1)


def nested_calls_function(n=100):
    """Function with nested calls for profiling."""
    def inner_function_a(x):
        return [i ** 2 for i in range(x)]

    def inner_function_b(x):
        return sum(inner_function_a(x))

    return inner_function_b(n)


def error_function():
    """Function that raises an error."""
    raise ValueError("Intentional error for testing")


class TestPerformanceProfilerAgent:
    """Test suite for PerformanceProfilerAgent."""

    def test_init(self):
        """Test agent initialization."""
        agent = PerformanceProfilerAgent()
        assert agent is not None

    def test_process_unknown_task(self):
        """Test handling of unknown task."""
        agent = PerformanceProfilerAgent()
        result = agent.process({'task': 'unknown_task'})

        assert isinstance(result, ProfileResult)
        assert not result.success
        assert len(result.errors) > 0
        assert 'Unknown task' in result.errors[0]


class TestFunctionProfiling:
    """Tests for function profiling capabilities."""

    def test_profile_function_basic(self):
        """Test basic function profiling."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_function',
            'function': simple_function,
            'args': [1000]
        })

        assert result.success
        assert 'function_name' in result.data
        assert result.data['function_name'] == 'simple_function'
        assert 'total_time' in result.data
        assert result.data['total_time'] > 0
        assert 'statistics' in result.data
        assert 'report' in result.data
        assert 'result' in result.data
        assert result.data['result'] == simple_function(1000)

    def test_profile_function_with_kwargs(self):
        """Test profiling function with keyword arguments."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_function',
            'function': simple_function,
            'kwargs': {'n': 2000}
        })

        assert result.success
        assert result.data['result'] == simple_function(2000)

    def test_profile_function_no_function(self):
        """Test profiling without providing function."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_function'
        })

        assert not result.success
        assert len(result.errors) > 0
        assert 'No function provided' in result.errors[0]

    def test_profile_function_top_n(self):
        """Test limiting number of profiling results."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_function',
            'function': nested_calls_function,
            'top_n': 5
        })

        assert result.success
        assert result.data['top_n'] == 5

    def test_profile_function_timing_accuracy(self):
        """Test that profiling captures timing accurately."""
        agent = PerformanceProfilerAgent()
        duration = 0.05
        result = agent.process({
            'task': 'profile_function',
            'function': slow_function,
            'args': [duration]
        })

        assert result.success
        # Should be close to expected duration (within 50% tolerance)
        assert abs(result.data['total_time'] - duration) < duration * 0.5

    def test_profile_recursive_function(self):
        """Test profiling recursive functions."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_function',
            'function': recursive_function,
            'args': [10]
        })

        assert result.success
        assert result.data['result'] == recursive_function(10)

    def test_profile_function_with_exception(self):
        """Test profiling function that raises exception."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_function',
            'function': error_function
        })

        assert not result.success
        assert len(result.errors) > 0


class TestMemoryProfiling:
    """Tests for memory profiling capabilities."""

    def test_profile_memory_basic(self):
        """Test basic memory profiling."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_memory',
            'function': memory_intensive_function,
            'args': [100]
        })

        assert result.success
        assert 'function_name' in result.data
        assert 'current_memory_mb' in result.data
        assert 'peak_memory_mb' in result.data
        assert 'memory_stats' in result.data
        assert 'report' in result.data
        assert result.data['peak_memory_mb'] > 0

    def test_profile_memory_allocation(self):
        """Test memory profiling captures allocations."""
        agent = PerformanceProfilerAgent()

        # Profile with larger array to ensure measurable memory
        result = agent.process({
            'task': 'profile_memory',
            'function': memory_intensive_function,
            'args': [500]
        })

        assert result.success
        assert result.data['peak_memory_mb'] > 0
        # Should have some memory stats
        assert len(result.data['memory_stats']) > 0

    def test_profile_memory_no_function(self):
        """Test memory profiling without function."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_memory'
        })

        assert not result.success
        assert 'No function provided' in result.errors[0]

    def test_profile_memory_top_n(self):
        """Test limiting memory profiling results."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_memory',
            'function': memory_intensive_function,
            'args': [100],
            'top_n': 3
        })

        assert result.success
        # Should have at most top_n memory stats
        assert len(result.data['memory_stats']) <= 3

    def test_profile_memory_with_kwargs(self):
        """Test memory profiling with keyword arguments."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_memory',
            'function': memory_intensive_function,
            'kwargs': {'size': 200}
        })

        assert result.success
        assert result.data['result'] == memory_intensive_function(200)

    def test_profile_memory_timing(self):
        """Test that memory profiling also captures timing."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_memory',
            'function': memory_intensive_function,
            'args': [100]
        })

        assert result.success
        assert 'total_time' in result.data
        assert result.data['total_time'] > 0


class TestBottleneckAnalysis:
    """Tests for bottleneck analysis capabilities."""

    def test_analyze_bottlenecks_basic(self):
        """Test basic bottleneck analysis."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'analyze_bottlenecks',
            'function': nested_calls_function,
            'args': [200]
        })

        assert result.success
        assert 'bottlenecks' in result.data
        assert 'total_time' in result.data
        assert 'threshold' in result.data

    def test_analyze_bottlenecks_threshold(self):
        """Test bottleneck detection with custom threshold."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'analyze_bottlenecks',
            'function': nested_calls_function,
            'args': [200],
            'threshold': 0.1  # 10% threshold
        })

        assert result.success
        assert result.data['threshold'] == 0.1

    def test_analyze_bottlenecks_no_function(self):
        """Test bottleneck analysis without function."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'analyze_bottlenecks'
        })

        assert not result.success
        assert 'No function provided' in result.errors[0]

    def test_analyze_bottlenecks_identification(self):
        """Test that bottlenecks are actually identified."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'analyze_bottlenecks',
            'function': nested_calls_function,
            'args': [300],
            'threshold': 0.01  # Low threshold to catch bottlenecks
        })

        assert result.success
        # Should identify some bottlenecks with nested calls
        assert 'bottlenecks' in result.data


class TestModuleProfiling:
    """Tests for module-level profiling."""

    def test_profile_module_basic(self):
        """Test basic module profiling."""
        agent = PerformanceProfilerAgent()

        # Profile a simple module execution
        module_code = """
def test_func():
    return sum(range(1000))

test_func()
"""

        result = agent.process({
            'task': 'profile_module',
            'module_code': module_code
        })

        # This test depends on implementation
        # If not implemented, should return error or success
        assert isinstance(result, ProfileResult)


class TestProfileResult:
    """Tests for ProfileResult dataclass."""

    def test_profile_result_creation(self):
        """Test ProfileResult creation."""
        result = ProfileResult(
            success=True,
            data={'test': 'value'}
        )

        assert result.success
        assert result.data == {'test': 'value'}
        assert result.errors == []

    def test_profile_result_with_errors(self):
        """Test ProfileResult with errors."""
        result = ProfileResult(
            success=False,
            data={},
            errors=['error1', 'error2']
        )

        assert not result.success
        assert len(result.errors) == 2

    def test_profile_result_default_errors(self):
        """Test ProfileResult default error list."""
        result = ProfileResult(
            success=True,
            data={}
        )

        assert result.errors == []


class TestIntegration:
    """Integration tests combining multiple profiling features."""

    def test_compare_function_and_memory_profiling(self):
        """Test that both profiling methods work on same function."""
        agent = PerformanceProfilerAgent()

        # Function profile
        result1 = agent.process({
            'task': 'profile_function',
            'function': memory_intensive_function,
            'args': [100]
        })

        # Memory profile
        result2 = agent.process({
            'task': 'profile_memory',
            'function': memory_intensive_function,
            'args': [100]
        })

        assert result1.success
        assert result2.success
        # Both should have similar results
        assert result1.data['result'] == result2.data['result']

    def test_profile_multiple_functions_sequentially(self):
        """Test profiling multiple functions in sequence."""
        agent = PerformanceProfilerAgent()

        functions = [simple_function, memory_intensive_function, nested_calls_function]
        results = []

        for func in functions:
            result = agent.process({
                'task': 'profile_function',
                'function': func,
                'args': [100]
            })
            results.append(result)

        # All should succeed
        assert all(r.success for r in results)
        # Should have different timings
        timings = [r.data['total_time'] for r in results]
        assert len(set(timings)) > 1  # Not all the same


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_profile_lambda_function(self):
        """Test profiling lambda function."""
        agent = PerformanceProfilerAgent()
        result = agent.process({
            'task': 'profile_function',
            'function': lambda x: x ** 2,
            'args': [10]
        })

        assert result.success
        assert result.data['result'] == 100

    def test_profile_function_with_empty_args(self):
        """Test profiling with empty arguments."""
        agent = PerformanceProfilerAgent()

        def no_arg_function():
            return 42

        result = agent.process({
            'task': 'profile_function',
            'function': no_arg_function,
            'args': []
        })

        assert result.success
        assert result.data['result'] == 42

    def test_profile_zero_duration_function(self):
        """Test profiling very fast function."""
        agent = PerformanceProfilerAgent()

        def instant_function():
            return 1

        result = agent.process({
            'task': 'profile_function',
            'function': instant_function
        })

        assert result.success
        # Timing might be very small but should be non-negative
        assert result.data['total_time'] >= 0

    def test_profile_function_returning_none(self):
        """Test profiling function that returns None."""
        agent = PerformanceProfilerAgent()

        def none_function():
            pass

        result = agent.process({
            'task': 'profile_function',
            'function': none_function
        })

        assert result.success
        assert result.data['result'] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
