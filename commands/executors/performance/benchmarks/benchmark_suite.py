#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark Suite
==========================================

Benchmarks all performance-critical components with validation.

Features:
- Cache performance benchmarks
- Parallel execution benchmarks
- Agent orchestration benchmarks
- Memory and I/O benchmarks
- Regression detection
- Performance baselines

Author: Claude Code Framework
Version: 2.0
"""

import sys
import os
import time
import json
import logging
import tempfile
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from performance.adaptive.auto_tuner import AutoTuner
from performance.cache.cache_tuner import CacheTuner, CacheMetrics
from performance.parallel.worker_pool_optimizer import WorkerPoolOptimizer, TaskType
from performance import ParallelExecutor, MultiLevelCache, ExecutionMode, WorkerTask


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    name: str
    duration_seconds: float
    throughput: float  # ops/sec
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    suite_name: str
    timestamp: datetime
    system_info: Dict[str, Any]
    results: List[BenchmarkResult] = field(default_factory=list)
    baseline: Optional[Dict[str, Any]] = None

    def add_result(self, result: BenchmarkResult):
        """Add benchmark result"""
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get results summary"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        return {
            "suite": self.suite_name,
            "timestamp": self.timestamp.isoformat(),
            "total_benchmarks": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "avg_throughput": statistics.mean([r.throughput for r in successful])
            if successful else 0.0,
            "results": [
                {
                    "name": r.name,
                    "duration": r.duration_seconds,
                    "throughput": r.throughput,
                    "success": r.success
                }
                for r in self.results
            ]
        }

    def compare_to_baseline(self) -> Dict[str, Any]:
        """Compare results to baseline"""
        if not self.baseline:
            return {"message": "No baseline available"}

        comparisons = []
        for result in self.results:
            baseline_result = self.baseline.get(result.name)
            if baseline_result:
                speedup = baseline_result["duration"] / result.duration_seconds
                comparisons.append({
                    "benchmark": result.name,
                    "current_duration": result.duration_seconds,
                    "baseline_duration": baseline_result["duration"],
                    "speedup": speedup,
                    "regression": speedup < 0.9  # >10% slower
                })

        return {
            "comparisons": comparisons,
            "regressions": [c for c in comparisons if c["regression"]]
        }


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmark suite.

    Benchmarks:
    1. Cache performance (hit rates, latency)
    2. Parallel execution (speedup, efficiency)
    3. Agent orchestration (coordination overhead)
    4. Memory usage (allocation, GC)
    5. I/O performance (throughput, latency)
    6. End-to-end command execution
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (
            Path.home() / ".claude" / "performance" / "benchmarks"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.suite = BenchmarkSuite(
            suite_name="Performance Benchmark Suite",
            timestamp=datetime.now(),
            system_info=self._get_system_info()
        )

        # Load baseline
        self._load_baseline()

    def run_all(self) -> BenchmarkSuite:
        """Run all benchmarks"""
        self.logger.info("Starting comprehensive benchmark suite")

        # Run benchmark categories
        self.benchmark_cache()
        self.benchmark_parallel_execution()
        self.benchmark_worker_optimization()
        self.benchmark_auto_tuning()
        self.benchmark_memory()
        self.benchmark_io()

        # Save results
        self._save_results()

        # Generate report
        self._generate_report()

        self.logger.info("Benchmark suite completed")
        return self.suite

    def benchmark_cache(self):
        """Benchmark cache performance"""
        self.logger.info("Benchmarking cache performance")

        # Test 1: Cache write throughput
        result = self._benchmark(
            "cache_write_throughput",
            self._test_cache_write_throughput,
            iterations=1000
        )
        self.suite.add_result(result)

        # Test 2: Cache read throughput (hits)
        result = self._benchmark(
            "cache_read_throughput_hits",
            self._test_cache_read_throughput_hits,
            iterations=1000
        )
        self.suite.add_result(result)

        # Test 3: Cache miss handling
        result = self._benchmark(
            "cache_miss_handling",
            self._test_cache_miss_handling,
            iterations=500
        )
        self.suite.add_result(result)

    def benchmark_parallel_execution(self):
        """Benchmark parallel execution"""
        self.logger.info("Benchmarking parallel execution")

        # Test 1: Thread pool speedup
        result = self._benchmark(
            "thread_pool_speedup",
            self._test_thread_pool_speedup,
            iterations=1
        )
        self.suite.add_result(result)

        # Test 2: Load balancing
        result = self._benchmark(
            "load_balancing",
            self._test_load_balancing,
            iterations=1
        )
        self.suite.add_result(result)

    def benchmark_worker_optimization(self):
        """Benchmark worker pool optimization"""
        self.logger.info("Benchmarking worker optimization")

        result = self._benchmark(
            "worker_optimization",
            self._test_worker_optimization,
            iterations=10
        )
        self.suite.add_result(result)

    def benchmark_auto_tuning(self):
        """Benchmark auto-tuning"""
        self.logger.info("Benchmarking auto-tuning")

        result = self._benchmark(
            "auto_tuning",
            self._test_auto_tuning,
            iterations=5
        )
        self.suite.add_result(result)

    def benchmark_memory(self):
        """Benchmark memory usage"""
        self.logger.info("Benchmarking memory performance")

        result = self._benchmark(
            "memory_allocation",
            self._test_memory_allocation,
            iterations=100
        )
        self.suite.add_result(result)

    def benchmark_io(self):
        """Benchmark I/O performance"""
        self.logger.info("Benchmarking I/O performance")

        result = self._benchmark(
            "io_throughput",
            self._test_io_throughput,
            iterations=100
        )
        self.suite.add_result(result)

    # ========================================================================
    # Individual Test Methods
    # ========================================================================

    def _test_cache_write_throughput(self) -> Dict[str, Any]:
        """Test cache write throughput"""
        cache = MultiLevelCache(max_memory_mb=100)

        start = time.time()
        for i in range(1000):
            cache.set(f"key_{i}", {"data": f"value_{i}"}, category="test")
        duration = time.time() - start

        return {
            "operations": 1000,
            "throughput": 1000 / duration,
            "duration": duration
        }

    def _test_cache_read_throughput_hits(self) -> Dict[str, Any]:
        """Test cache read throughput (hits)"""
        cache = MultiLevelCache(max_memory_mb=100)

        # Populate cache
        for i in range(100):
            cache.set(f"key_{i}", {"data": f"value_{i}"}, category="test")

        start = time.time()
        for i in range(1000):
            cache.get(f"key_{i % 100}", category="test")
        duration = time.time() - start

        stats = cache.get_stats()

        return {
            "operations": 1000,
            "throughput": 1000 / duration,
            "duration": duration,
            "hit_rate": stats["hit_rate"]
        }

    def _test_cache_miss_handling(self) -> Dict[str, Any]:
        """Test cache miss handling"""
        cache = MultiLevelCache(max_memory_mb=100)

        start = time.time()
        for i in range(500):
            cache.get(f"missing_key_{i}", category="test")
        duration = time.time() - start

        return {
            "operations": 500,
            "throughput": 500 / duration,
            "duration": duration
        }

    def _test_thread_pool_speedup(self) -> Dict[str, Any]:
        """Test thread pool speedup"""

        def sample_task(x):
            time.sleep(0.01)
            return x * 2

        # Sequential baseline
        start = time.time()
        for i in range(50):
            sample_task(i)
        sequential_duration = time.time() - start

        # Parallel execution
        executor = ParallelExecutor(
            mode=ExecutionMode.PARALLEL_THREAD,
            max_workers=8
        )

        tasks = [
            WorkerTask(
                task_id=f"task_{i}",
                function=sample_task,
                args=(i,)
            )
            for i in range(50)
        ]

        start = time.time()
        results = executor.execute_parallel(tasks)
        parallel_duration = time.time() - start

        speedup = sequential_duration / parallel_duration

        return {
            "operations": 50,
            "sequential_duration": sequential_duration,
            "parallel_duration": parallel_duration,
            "speedup": speedup,
            "throughput": 50 / parallel_duration
        }

    def _test_load_balancing(self) -> Dict[str, Any]:
        """Test load balancing"""

        def variable_task(duration):
            time.sleep(duration)
            return duration

        executor = ParallelExecutor(
            mode=ExecutionMode.PARALLEL_THREAD,
            max_workers=4
        )

        # Variable duration tasks
        durations = [0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01, 0.04]
        tasks = [
            WorkerTask(
                task_id=f"task_{i}",
                function=variable_task,
                args=(d,)
            )
            for i, d in enumerate(durations)
        ]

        start = time.time()
        results = executor.execute_parallel(tasks)
        duration = time.time() - start

        return {
            "operations": len(tasks),
            "duration": duration,
            "throughput": len(tasks) / duration
        }

    def _test_worker_optimization(self) -> Dict[str, Any]:
        """Test worker pool optimization"""
        optimizer = WorkerPoolOptimizer()

        start = time.time()
        config = optimizer.optimize(task_type=TaskType.CPU_BOUND)
        duration = time.time() - start

        return {
            "duration": duration,
            "config": config.to_dict()
        }

    def _test_auto_tuning(self) -> Dict[str, Any]:
        """Test auto-tuning"""
        tuner = AutoTuner()

        start = time.time()
        config = tuner.tune()
        duration = time.time() - start

        return {
            "duration": duration,
            "config": config.to_dict()
        }

    def _test_memory_allocation(self) -> Dict[str, Any]:
        """Test memory allocation"""
        allocations = []

        start = time.time()
        for i in range(100):
            allocations.append([0] * 10000)  # Allocate 10K integers
        duration = time.time() - start

        # Cleanup
        allocations.clear()

        return {
            "operations": 100,
            "duration": duration,
            "throughput": 100 / duration
        }

    def _test_io_throughput(self) -> Dict[str, Any]:
        """Test I/O throughput"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write test
            start = time.time()
            for i in range(100):
                (tmppath / f"file_{i}.txt").write_text("test data" * 100)
            write_duration = time.time() - start

            # Read test
            start = time.time()
            for i in range(100):
                (tmppath / f"file_{i}.txt").read_text()
            read_duration = time.time() - start

        return {
            "operations": 200,
            "write_throughput": 100 / write_duration,
            "read_throughput": 100 / read_duration,
            "duration": write_duration + read_duration
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _benchmark(
        self,
        name: str,
        test_func: Callable,
        iterations: int = 1
    ) -> BenchmarkResult:
        """Run a benchmark with multiple iterations"""
        self.logger.info(f"Running benchmark: {name}")

        durations = []
        errors = []
        metrics_list = []

        for i in range(iterations):
            try:
                start = time.time()
                metrics = test_func()
                duration = time.time() - start

                durations.append(duration)
                metrics_list.append(metrics)

            except Exception as e:
                self.logger.error(f"Benchmark {name} iteration {i} failed: {e}")
                errors.append(str(e))

        if not durations:
            return BenchmarkResult(
                name=name,
                duration_seconds=0.0,
                throughput=0.0,
                success=False,
                errors=errors
            )

        # Aggregate metrics
        avg_duration = statistics.mean(durations)
        throughput = metrics_list[0].get("throughput", 1.0 / avg_duration)

        aggregated_metrics = {
            "iterations": iterations,
            "avg_duration": avg_duration,
            "min_duration": min(durations),
            "max_duration": max(durations),
            "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0.0
        }

        # Add test-specific metrics
        if metrics_list:
            for key in metrics_list[0]:
                if key not in ["duration", "throughput", "operations"]:
                    values = [m.get(key) for m in metrics_list if key in m]
                    if values and isinstance(values[0], (int, float)):
                        aggregated_metrics[key] = statistics.mean(values)

        return BenchmarkResult(
            name=name,
            duration_seconds=avg_duration,
            throughput=throughput,
            success=True,
            metrics=aggregated_metrics,
            errors=errors
        )

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        import psutil

        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": platform.python_version()
        }

    def _load_baseline(self):
        """Load baseline results"""
        baseline_file = self.output_dir / "baseline.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    self.suite.baseline = {
                        r["name"]: r for r in baseline_data.get("results", [])
                    }
                self.logger.info("Loaded baseline results")
            except Exception as e:
                self.logger.error(f"Failed to load baseline: {e}")

    def _save_results(self):
        """Save benchmark results"""
        results_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        summary = self.suite.get_summary()

        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Results saved to {results_file}")

        # Update baseline if significantly better
        if self.suite.baseline:
            comparison = self.suite.compare_to_baseline()
            if not comparison.get("regressions"):
                baseline_file = self.output_dir / "baseline.json"
                with open(baseline_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                self.logger.info("Updated baseline with improved results")

    def _generate_report(self):
        """Generate benchmark report"""
        report_file = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE BENCHMARK REPORT")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {self.suite.timestamp}")
        lines.append(f"System: {self.suite.system_info['platform']}")
        lines.append(f"CPU Cores: {self.suite.system_info['cpu_count']}")
        lines.append(f"Memory: {self.suite.system_info['total_memory_gb']:.1f} GB")
        lines.append("")

        lines.append("RESULTS")
        lines.append("-" * 80)
        for result in self.suite.results:
            status = "PASS" if result.success else "FAIL"
            lines.append(f"{result.name:40} {status:6} {result.duration_seconds:8.3f}s "
                        f"{result.throughput:10.1f} ops/s")

        # Comparison to baseline
        if self.suite.baseline:
            lines.append("")
            lines.append("COMPARISON TO BASELINE")
            lines.append("-" * 80)
            comparison = self.suite.compare_to_baseline()
            for comp in comparison.get("comparisons", []):
                speedup = comp["speedup"]
                status = "REGRESSION" if comp["regression"] else "OK"
                lines.append(f"{comp['benchmark']:40} {status:12} {speedup:6.2f}x")

        report = "\n".join(lines)
        report_file.write_text(report)
        print(report)

        self.logger.info(f"Report saved to {report_file}")


def main():
    """Run benchmark suite"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Performance Benchmark Suite")
    print("=" * 80)

    suite = PerformanceBenchmarkSuite()
    results = suite.run_all()

    print(f"\nCompleted {len(results.results)} benchmarks")
    print(f"Successful: {len([r for r in results.results if r.success])}")
    print(f"Failed: {len([r for r in results.results if not r.success])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())