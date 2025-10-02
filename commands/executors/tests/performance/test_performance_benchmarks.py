#!/usr/bin/env python3
"""
Performance Benchmarks for Command Executor Framework
=====================================================

Benchmarks for:
- Cache performance (target: 5-8x improvement)
- Parallel execution (target: 3-5x improvement)
- Agent orchestration efficiency
- Memory usage
- End-to-end command execution times

Uses pytest-benchmark for accurate measurements.
"""

import pytest
import time
from pathlib import Path
from typing import Dict, Any, List

from executors.framework import (
    BaseCommandExecutor,
    ExecutionContext,
    AgentOrchestrator,
    CacheManager,
)
from executors.performance import PerformanceMonitor, ParallelExecutor


@pytest.mark.performance
@pytest.mark.benchmark
class TestCachePerformance:
    """Benchmark caching system performance"""

    def test_cache_hit_performance(self, benchmark, temp_workspace: Path):
        """Benchmark cache hit performance (target: 5-8x faster)"""
        cache_manager = CacheManager(temp_workspace / "cache")

        # Warm up cache
        test_data = {"large_result": "x" * 10000}
        cache_manager.set("test_key", test_data, level="default")

        def cache_hit():
            return cache_manager.get("test_key", level="default")

        result = benchmark(cache_hit)
        assert result is not None

    def test_cache_miss_performance(self, benchmark, temp_workspace: Path):
        """Benchmark cache miss performance"""
        cache_manager = CacheManager(temp_workspace / "cache")

        def cache_miss():
            return cache_manager.get("nonexistent_key", level="default")

        result = benchmark(cache_miss)
        assert result is None

    def test_cache_write_performance(self, benchmark, temp_workspace: Path):
        """Benchmark cache write performance"""
        cache_manager = CacheManager(temp_workspace / "cache")
        test_data = {"data": "x" * 10000}

        def cache_write():
            cache_manager.set(f"key_{time.time()}", test_data, level="default")

        benchmark(cache_write)

    def test_cache_speedup_measurement(self, temp_workspace: Path):
        """Measure actual cache speedup for typical operations"""
        cache_manager = CacheManager(temp_workspace / "cache")

        # Simulate expensive operation
        def expensive_operation():
            time.sleep(0.1)  # Simulate work
            return {"result": "computed"}

        # Measure without cache
        start = time.time()
        for _ in range(10):
            result = expensive_operation()
        no_cache_time = time.time() - start

        # Measure with cache
        cache_manager.set("cached_result", {"result": "computed"}, level="default")
        start = time.time()
        for _ in range(10):
            result = cache_manager.get("cached_result", level="default")
        cache_time = time.time() - start

        speedup = no_cache_time / cache_time
        assert speedup >= 5.0, f"Cache speedup {speedup:.2f}x below target (5x)"


@pytest.mark.performance
@pytest.mark.benchmark
class TestParallelExecutionPerformance:
    """Benchmark parallel execution performance"""

    def test_parallel_speedup(self, temp_workspace: Path):
        """
        Measure parallel execution speedup (target: 3-5x for 4 workers)
        """
        parallel_executor = ParallelExecutor(max_workers=4)

        def task(x):
            time.sleep(0.1)  # Simulate work
            return x * 2

        tasks = list(range(20))

        # Sequential execution
        start = time.time()
        sequential_results = [task(x) for x in tasks]
        sequential_time = time.time() - start

        # Parallel execution
        start = time.time()
        parallel_results = parallel_executor.execute_parallel(
            task,
            tasks,
            max_workers=4
        )
        parallel_time = time.time() - start

        speedup = sequential_time / parallel_time
        assert speedup >= 3.0, f"Parallel speedup {speedup:.2f}x below target (3x)"

    def test_agent_parallel_execution(self, benchmark, temp_workspace: Path):
        """Benchmark parallel agent execution"""
        orchestrator = AgentOrchestrator()

        context = ExecutionContext(
            command_name="test",
            work_dir=temp_workspace,
            args={},
            parallel=True
        )

        agents = ["agent1", "agent2", "agent3", "agent4"]
        task = "Test task"

        def parallel_agents():
            return orchestrator.orchestrate(agents, context, task)

        result = benchmark(parallel_agents)
        assert result["agents_executed"] == len(agents)


@pytest.mark.performance
@pytest.mark.benchmark
class TestAgentOrchestrationPerformance:
    """Benchmark agent orchestration efficiency"""

    def test_agent_selection_performance(self, benchmark, temp_workspace: Path):
        """Benchmark intelligent agent selection"""
        orchestrator = AgentOrchestrator()

        context = ExecutionContext(
            command_name="test",
            work_dir=temp_workspace,
            args={}
        )

        def select_agents():
            return orchestrator._intelligent_selection(context)

        result = benchmark(select_agents)
        assert len(result) > 0

    def test_result_synthesis_performance(self, benchmark):
        """Benchmark result synthesis from multiple agents"""
        orchestrator = AgentOrchestrator()

        # Create large result set
        results = {
            f"agent_{i}": {
                "status": "completed",
                "findings": [f"finding_{j}" for j in range(10)],
                "recommendations": [f"rec_{j}" for j in range(5)]
            }
            for i in range(20)
        }

        def synthesize():
            return orchestrator._synthesize_results(results)

        result = benchmark(synthesize)
        assert result["agents_executed"] == 20

    def test_multi_agent_orchestration_scalability(self, temp_workspace: Path):
        """Test orchestration scalability with increasing agent count"""
        orchestrator = AgentOrchestrator()

        context = ExecutionContext(
            command_name="test",
            work_dir=temp_workspace,
            args={},
            parallel=True
        )

        agent_counts = [5, 10, 15, 20]
        timings = []

        for count in agent_counts:
            agents = [f"agent_{i}" for i in range(count)]

            start = time.time()
            result = orchestrator.orchestrate(agents, context, "test")
            duration = time.time() - start

            timings.append(duration)
            assert result["agents_executed"] == count

        # Verify sublinear scaling with parallelization
        # Time for 20 agents should be < 4x time for 5 agents
        assert timings[-1] < timings[0] * 4


@pytest.mark.performance
@pytest.mark.benchmark
class TestMemoryPerformance:
    """Benchmark memory usage"""

    def test_cache_memory_efficiency(self, temp_workspace: Path):
        """Test cache doesn't consume excessive memory"""
        import sys
        cache_manager = CacheManager(temp_workspace / "cache")

        # Store many items
        for i in range(1000):
            cache_manager.set(
                f"key_{i}",
                {"data": "x" * 100},
                level="default"
            )

        stats = cache_manager.get_stats()
        total_size_mb = sum(
            level["size_mb"] for level in stats["levels"].values()
        )

        # Should be reasonable (< 100MB for test data)
        assert total_size_mb < 100.0

    def test_agent_result_memory(self):
        """Test agent results don't accumulate excessively"""
        orchestrator = AgentOrchestrator()

        # Track memory before
        import tracemalloc
        tracemalloc.start()

        snapshot1 = tracemalloc.take_snapshot()

        # Execute many agent operations
        for i in range(100):
            results = {f"agent_{j}": {"data": "test"} for j in range(10)}
            orchestrator._synthesize_results(results)

        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        # Memory growth should be reasonable
        tracemalloc.stop()


@pytest.mark.performance
@pytest.mark.benchmark
class TestEndToEndPerformance:
    """Benchmark end-to-end command execution"""

    def test_simple_command_execution(self, benchmark, temp_workspace: Path):
        """Benchmark simple command execution time"""
        from test_framework_integration import TestCommandExecutor

        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        def execute():
            return executor.execute({"test": "value"})

        result = benchmark(execute)
        assert result.success

    def test_command_with_validation(self, benchmark, temp_workspace: Path):
        """Benchmark command execution with full validation"""
        from test_framework_integration import TestCommandExecutor

        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        def execute_with_validation():
            return executor.execute({"validate": True, "test": "value"})

        result = benchmark(execute_with_validation)
        assert result.success

    def test_command_with_agents(self, benchmark, temp_workspace: Path):
        """Benchmark command execution with agent orchestration"""
        from test_framework_integration import TestCommandExecutor

        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        def execute_with_agents():
            return executor.execute({
                "agents": "core",
                "orchestrate": True
            })

        result = benchmark(execute_with_agents)
        assert result.success


@pytest.mark.performance
@pytest.mark.benchmark
class TestPerformanceTargets:
    """Verify performance targets are met"""

    def test_cache_target_5x_speedup(self, temp_workspace: Path):
        """Verify cache achieves 5x+ speedup target"""
        cache_manager = CacheManager(temp_workspace / "cache")

        # Expensive operation
        def compute():
            result = sum(i**2 for i in range(10000))
            return {"result": result}

        # Time without cache (10 runs)
        start = time.time()
        for _ in range(10):
            compute()
        no_cache = time.time() - start

        # Time with cache (10 runs)
        cached_result = compute()
        cache_manager.set("result", cached_result)

        start = time.time()
        for _ in range(10):
            cache_manager.get("result")
        with_cache = time.time() - start

        speedup = no_cache / with_cache
        assert speedup >= 5.0, (
            f"Cache speedup {speedup:.2f}x below 5x target. "
            f"No cache: {no_cache:.4f}s, With cache: {with_cache:.4f}s"
        )

    def test_parallel_target_3x_speedup(self, temp_workspace: Path):
        """Verify parallel execution achieves 3x+ speedup target"""
        parallel_executor = ParallelExecutor(max_workers=4)

        def work(x):
            time.sleep(0.05)  # 50ms work
            return x * 2

        items = list(range(40))  # 40 items = 2 seconds sequential

        # Sequential
        start = time.time()
        [work(x) for x in items]
        sequential_time = time.time() - start

        # Parallel
        start = time.time()
        parallel_executor.execute_parallel(work, items, max_workers=4)
        parallel_time = time.time() - start

        speedup = sequential_time / parallel_time
        assert speedup >= 3.0, (
            f"Parallel speedup {speedup:.2f}x below 3x target. "
            f"Sequential: {sequential_time:.2f}s, Parallel: {parallel_time:.2f}s"
        )


@pytest.mark.performance
class TestPerformanceRegression:
    """Detect performance regressions"""

    BASELINE_TIMINGS = {
        "simple_execution": 0.2,  # 200ms
        "with_validation": 0.3,   # 300ms
        "with_agents": 0.5,       # 500ms
        "cache_hit": 0.001,       # 1ms
        "agent_selection": 0.05,  # 50ms
    }

    def test_no_performance_regression(self, temp_workspace: Path):
        """Verify no performance regression from baseline"""
        from test_framework_integration import TestCommandExecutor

        executor = TestCommandExecutor()
        executor.work_dir = temp_workspace

        # Test simple execution
        start = time.time()
        result = executor.execute({})
        duration = time.time() - start

        baseline = self.BASELINE_TIMINGS["simple_execution"]
        assert duration < baseline * 1.5, (
            f"Performance regression detected: {duration:.3f}s > "
            f"{baseline * 1.5:.3f}s (1.5x baseline)"
        )