"""
Performance Profiler Agent - Advanced profiling and bottleneck analysis.

This agent provides line-by-line profiling, memory profiling, and bottleneck
identification for scientific computing code.
"""

import sys
import io
import cProfile
import pstats
from typing import Any, Dict, Optional, List
from pathlib import Path
import tracemalloc
import time
from dataclasses import dataclass


@dataclass
class ProfileResult:
    """Simple result container for profiling operations."""
    success: bool
    data: Dict[str, Any]
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class PerformanceProfilerAgent:
    """
    Agent for profiling performance and identifying bottlenecks.

    Capabilities:
    - Function-level profiling with cProfile
    - Line-by-line profiling (when line_profiler available)
    - Memory profiling with tracemalloc
    - Bottleneck identification
    - Performance report generation
    """

    def process(self, data: Dict[str, Any]) -> ProfileResult:
        """
        Profile code or function performance.

        Args:
            data: Dictionary containing:
                - task: str, one of ['profile_function', 'profile_memory',
                        'analyze_bottlenecks', 'profile_module']
                - function: callable (for profile_function)
                - args: list (function arguments)
                - kwargs: dict (function keyword arguments)
                - module_path: str (for profile_module)
                - top_n: int (number of top results to show)

        Returns:
            AgentResult with profiling data and reports
        """
        task = data.get("task", "profile_function")

        if task == "profile_function":
            return self._profile_function(data)
        elif task == "profile_memory":
            return self._profile_memory(data)
        elif task == "analyze_bottlenecks":
            return self._analyze_bottlenecks(data)
        elif task == "profile_module":
            return self._profile_module(data)
        else:
            return ProfileResult(
                success=False,
                data={},
                errors=[f"Unknown task: {task}"]
            )

    def _profile_function(self, data: Dict[str, Any]) -> ProfileResult:
        """
        Profile a function's execution time using cProfile.

        Args:
            data: Dictionary with 'function', 'args', 'kwargs', 'top_n'

        Returns:
            AgentResult with profiling statistics
        """
        func = data.get("function")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})
        top_n = data.get("top_n", 20)

        if func is None:
            return ProfileResult(
                success=False,
                data={},
                errors=["No function provided"]
            )

        # Save existing profiler state (for pytest coverage compatibility)
        old_profile = sys.getprofile()

        # Ensure clean profiler state for tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        # Profile the function
        profiler = cProfile.Profile()
        start_time = time.perf_counter()

        try:
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()

            # Restore previous profiler state (for pytest coverage)
            sys.setprofile(old_profile)

            elapsed_time = time.perf_counter() - start_time

            # Capture statistics
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats(top_n)

            # Parse statistics
            stats_data = self._parse_pstats(stats, top_n)

            # Generate report
            report = self._generate_profile_report(
                func.__name__,
                elapsed_time,
                stats_data,
                stats_stream.getvalue()
            )

            return ProfileResult(
                success=True,
                data={
                    'function_name': func.__name__,
                    'total_time': elapsed_time,
                    'statistics': stats_data,
                    'report': report,
                    'result': result,
                    'profiling_method': 'cProfile',
                    'top_n': top_n
                }
            )

        except Exception as e:
            # Ensure profiler is disabled before restoring
            try:
                profiler.disable()
            except:
                pass  # May already be disabled
            # Restore previous profiler state even on error
            sys.setprofile(old_profile)
            return ProfileResult(
                success=False,
                data={},
                errors=[f"Profiling failed: {str(e)}"]
            )

    def _profile_memory(self, data: Dict[str, Any]) -> ProfileResult:
        """
        Profile memory usage of a function using tracemalloc.

        Args:
            data: Dictionary with 'function', 'args', 'kwargs', 'top_n'

        Returns:
            AgentResult with memory profiling data
        """
        func = data.get("function")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})
        top_n = data.get("top_n", 10)

        if func is None:
            return ProfileResult(
                success=False,
                data={},
                errors=["No function provided"]
            )

        # Profile memory
        # Check if tracemalloc is already running
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        try:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time

            snapshot_after = tracemalloc.take_snapshot()
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            # Only stop if we started it
            if not was_tracing:
                tracemalloc.stop()

            # Analyze memory differences
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

            # Format memory statistics
            memory_data = []
            for stat in top_stats[:top_n]:
                memory_data.append({
                    'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size_diff_mb': stat.size_diff / 1024 / 1024,
                    'size_mb': stat.size / 1024 / 1024,
                    'count_diff': stat.count_diff,
                    'count': stat.count
                })

            # Generate report
            report = self._generate_memory_report(
                func.__name__,
                elapsed_time,
                current_memory / 1024 / 1024,
                peak_memory / 1024 / 1024,
                memory_data
            )

            return ProfileResult(
                success=True,
                data={
                    'function_name': func.__name__,
                    'total_time': elapsed_time,
                    'current_memory_mb': current_memory / 1024 / 1024,
                    'peak_memory_mb': peak_memory / 1024 / 1024,
                    'memory_stats': memory_data,
                    'report': report,
                    'result': result,
                    'profiling_method': 'tracemalloc',
                    'top_n': top_n
                }
            )

        except Exception as e:
            # Only stop if we started it
            if not was_tracing:
                tracemalloc.stop()
            return ProfileResult(
                success=False,
                data={},
                errors=[f"Memory profiling failed: {str(e)}"]
            )

    def _analyze_bottlenecks(self, data: Dict[str, Any]) -> ProfileResult:
        """
        Analyze profiling data to identify performance bottlenecks.

        Args:
            data: Dictionary with 'function', 'args', 'kwargs', 'threshold'

        Returns:
            AgentResult with bottleneck analysis
        """
        func = data.get("function")
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})
        threshold = data.get("threshold", 0.05)  # 5% of total time

        if func is None:
            return ProfileResult(
                success=False,
                data={},
                errors=["No function provided"]
            )

        # Save existing profiler state (for pytest coverage compatibility)
        old_profile = sys.getprofile()

        # Ensure clean profiler state for tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        # Profile with cProfile
        profiler = cProfile.Profile()
        start_time = time.perf_counter()

        try:
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()

            # Restore previous profiler state (for pytest coverage)
            sys.setprofile(old_profile)

            total_time = time.perf_counter() - start_time

            # Analyze statistics
            stats = pstats.Stats(profiler)
            stats.strip_dirs()

            # Identify bottlenecks
            bottlenecks = []
            for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
                filename, line, func_name = func_info
                percentage = (ct / total_time * 100) if total_time > 0 else 0

                if percentage >= threshold * 100:
                    bottlenecks.append({
                        'function': func_name,
                        'file': filename,
                        'line': line,
                        'cumulative_time': ct,
                        'percentage': percentage,
                        'calls': nc
                    })

            # Sort by cumulative time
            bottlenecks.sort(key=lambda x: x['cumulative_time'], reverse=True)

            # Generate report
            report = self._generate_bottleneck_report(
                func.__name__,
                total_time,
                threshold,
                bottlenecks
            )

            return ProfileResult(
                success=True,
                data={
                    'function_name': func.__name__,
                    'total_time': total_time,
                    'bottlenecks': bottlenecks,
                    'threshold': threshold,
                    'report': report,
                    'result': result,
                    'analysis_method': 'bottleneck_detection'
                }
            )

        except Exception as e:
            # Ensure profiler is disabled before restoring
            try:
                profiler.disable()
            except:
                pass  # May already be disabled
            # Restore previous profiler state even on error
            sys.setprofile(old_profile)
            return ProfileResult(
                success=False,
                data={},
                errors=[f"Bottleneck analysis failed: {str(e)}"]
            )

    def _profile_module(self, data: Dict[str, Any]) -> ProfileResult:
        """
        Profile an entire module execution.

        Args:
            data: Dictionary with 'module_path', 'top_n'

        Returns:
            AgentResult with module profiling data
        """
        module_path = data.get("module_path")
        top_n = data.get("top_n", 20)

        if not module_path:
            return ProfileResult(
                success=False,
                data={},
                errors=["No module_path provided"]
            )

        # Check if file exists
        if not Path(module_path).exists():
            return ProfileResult(
                success=False,
                data={},
                errors=[f"Module file not found: {module_path}"]
            )

        try:
            # Profile module execution
            profiler = cProfile.Profile()
            start_time = time.perf_counter()

            with open(module_path) as f:
                code = compile(f.read(), module_path, 'exec')
                profiler.runcall(exec, code)

            elapsed_time = time.perf_counter() - start_time

            # Get statistics
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats(top_n)

            stats_data = self._parse_pstats(stats, top_n)

            report = self._generate_profile_report(
                f"Module: {Path(module_path).name}",
                elapsed_time,
                stats_data,
                stats_stream.getvalue()
            )

            return ProfileResult(
                success=True,
                data={
                    'module_path': module_path,
                    'total_time': elapsed_time,
                    'statistics': stats_data,
                    'report': report,
                    'profiling_method': 'cProfile',
                    'top_n': top_n
                }
            )

        except Exception as e:
            return ProfileResult(
                success=False,
                data={},
                errors=[f"Module profiling failed: {str(e)}"]
            )

    def _parse_pstats(self, stats: pstats.Stats, top_n: int) -> List[Dict[str, Any]]:
        """Parse pstats.Stats object into structured data."""
        stats_data = []

        for func_info, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:top_n]:
            filename, line, func_name = func_info
            stats_data.append({
                'function': func_name,
                'file': filename,
                'line': line,
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / nc if nc > 0 else 0
            })

        return stats_data

    def _generate_profile_report(
        self,
        name: str,
        total_time: float,
        stats_data: List[Dict[str, Any]],
        raw_stats: str
    ) -> str:
        """Generate human-readable profiling report."""
        lines = [
            "=" * 70,
            f"Performance Profile: {name}",
            "=" * 70,
            f"Total Execution Time: {total_time:.4f} s",
            "",
            "Top Functions by Cumulative Time:",
            "-" * 70,
        ]

        for i, stat in enumerate(stats_data[:10], 1):
            lines.append(
                f"{i:2d}. {stat['function']:<30} {stat['cumulative_time']:8.4f}s ({stat['calls']:6d} calls)"
            )

        lines.extend([
            "-" * 70,
            "",
            "Detailed Statistics:",
            raw_stats,
            "=" * 70
        ])

        return "\n".join(lines)

    def _generate_memory_report(
        self,
        func_name: str,
        total_time: float,
        current_mb: float,
        peak_mb: float,
        memory_data: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable memory profiling report."""
        lines = [
            "=" * 70,
            f"Memory Profile: {func_name}",
            "=" * 70,
            f"Execution Time: {total_time:.4f} s",
            f"Current Memory: {current_mb:.2f} MB",
            f"Peak Memory: {peak_mb:.2f} MB",
            "",
            "Top Memory Allocations:",
            "-" * 70,
        ]

        for i, stat in enumerate(memory_data, 1):
            lines.append(
                f"{i:2d}. {stat['size_mb']:8.2f} MB (+{stat['size_diff_mb']:7.2f} MB) - "
                f"{stat['count']:6d} blocks (+{stat['count_diff']:5d})"
            )
            lines.append(f"    {stat['file']}")

        lines.extend([
            "-" * 70,
            "=" * 70
        ])

        return "\n".join(lines)

    def _generate_bottleneck_report(
        self,
        func_name: str,
        total_time: float,
        threshold: float,
        bottlenecks: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable bottleneck analysis report."""
        lines = [
            "=" * 70,
            f"Bottleneck Analysis: {func_name}",
            "=" * 70,
            f"Total Time: {total_time:.4f} s",
            f"Threshold: {threshold * 100:.1f}% of total time",
            f"Bottlenecks Found: {len(bottlenecks)}",
            "",
            "Performance Bottlenecks:",
            "-" * 70,
        ]

        if not bottlenecks:
            lines.append("No significant bottlenecks detected.")
        else:
            for i, bn in enumerate(bottlenecks, 1):
                lines.extend([
                    f"{i}. {bn['function']} - {bn['percentage']:.1f}% of total time",
                    f"   Cumulative Time: {bn['cumulative_time']:.4f} s",
                    f"   Calls: {bn['calls']}",
                    f"   Location: {bn['file']}:{bn['line']}",
                    ""
                ])

        lines.extend([
            "-" * 70,
            "=" * 70
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    def example_function(n):
        """Example function with intentional performance characteristics."""
        # Intentional bottleneck 1: nested loops
        result = 0
        for i in range(n):
            for j in range(n):
                result += i * j

        # Intentional bottleneck 2: large list creation
        data = [x**2 for x in range(n * 100)]

        return result

    # Create profiler agent
    profiler = PerformanceProfilerAgent()

    # Profile function
    print("1. Function Profiling:")
    result = profiler.process({
        'task': 'profile_function',
        'function': example_function,
        'args': [100],
        'top_n': 10
    })
    print(result.data['report'])

    # Memory profiling
    print("\n2. Memory Profiling:")
    result = profiler.process({
        'task': 'profile_memory',
        'function': example_function,
        'args': [100],
        'top_n': 5
    })
    print(result.data['report'])

    # Bottleneck analysis
    print("\n3. Bottleneck Analysis:")
    result = profiler.process({
        'task': 'analyze_bottlenecks',
        'function': example_function,
        'args': [100],
        'threshold': 0.05
    })
    print(result.data['report'])
