"""Performance Profiling for Optimal Control.

This module provides profiling utilities for performance analysis:
1. Time profiling (cProfile integration)
2. Memory profiling (memory_profiler integration)
3. Custom profiling decorators
4. Profile report generation

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, Any, Callable, Optional
from pathlib import Path
from functools import wraps
import time
import json

# Try to import profiling libraries
try:
    import cProfile
    import pstats
    from pstats import SortKey
    CPROFILE_AVAILABLE = True
except ImportError:
    CPROFILE_AVAILABLE = False

try:
    from memory_profiler import profile as memory_profile_decorator
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


# =============================================================================
# Time Profiling
# =============================================================================

def profile_solver(
    solver_func: Callable,
    *args,
    output_file: Optional[Path] = None,
    sort_by: str = 'cumulative',
    top_n: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """Profile solver function with cProfile.

    Args:
        solver_func: Solver function to profile
        *args: Arguments for solver
        output_file: Path to save profile stats
        sort_by: Sort key ('cumulative', 'time', 'calls')
        top_n: Number of top functions to display
        **kwargs: Keyword arguments for solver

    Returns:
        Dictionary with profiling results and solver output
    """
    if not CPROFILE_AVAILABLE:
        raise ImportError("cProfile not available")

    # Create profiler
    profiler = cProfile.Profile()

    # Profile execution
    profiler.enable()
    start_time = time.time()

    result = solver_func(*args, **kwargs)

    end_time = time.time()
    profiler.disable()

    # Get stats
    stats = pstats.Stats(profiler)

    # Sort and filter
    sort_keys = {
        'cumulative': SortKey.CUMULATIVE,
        'time': SortKey.TIME,
        'calls': SortKey.CALLS
    }
    stats.sort_stats(sort_keys.get(sort_by, SortKey.CUMULATIVE))

    # Save if requested
    if output_file:
        stats.dump_stats(str(output_file))
        print(f"Profile saved to {output_file}")

    # Print top functions
    print(f"\nTop {top_n} functions by {sort_by}:")
    stats.print_stats(top_n)

    return {
        'result': result,
        'total_time': end_time - start_time,
        'stats': stats
    }


def profile_training(
    train_func: Callable,
    n_steps: int,
    output_file: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """Profile training function.

    Args:
        train_func: Training function
        n_steps: Number of training steps
        output_file: Path to save profile
        **kwargs: Additional training arguments

    Returns:
        Dictionary with profiling results
    """
    return profile_solver(
        train_func,
        n_steps=n_steps,
        output_file=output_file,
        **kwargs
    )


# =============================================================================
# Memory Profiling
# =============================================================================

def memory_profile(func: Callable) -> Callable:
    """Decorator for memory profiling.

    Args:
        func: Function to profile

    Returns:
        Wrapped function

    Example:
        @memory_profile
        def my_solver(...):
            ...
    """
    if MEMORY_PROFILER_AVAILABLE:
        return memory_profile_decorator(func)
    else:
        # Return unmodified if memory_profiler not available
        print("Warning: memory_profiler not available, profiling disabled")
        return func


# =============================================================================
# Custom Profiling Decorators
# =============================================================================

class TimingProfiler:
    """Simple timing profiler using decorators.

    Tracks execution time of decorated functions.
    """

    def __init__(self):
        """Initialize profiler."""
        self.timings: Dict[str, List[float]] = {}

    def profile(self, func: Callable) -> Callable:
        """Decorator for timing profiling.

        Args:
            func: Function to profile

        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            func_name = func.__name__
            if func_name not in self.timings:
                self.timings[func_name] = []

            self.timings[func_name].append(elapsed)

            return result

        return wrapper

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary.

        Returns:
            Dictionary mapping function names to timing statistics
        """
        import numpy as np

        summary = {}

        for func_name, times in self.timings.items():
            summary[func_name] = {
                'count': len(times),
                'total': sum(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'min': min(times),
                'max': max(times)
            }

        return summary

    def print_summary(self):
        """Print timing summary."""
        summary = self.get_summary()

        print("\n" + "="*70)
        print("Timing Profile Summary")
        print("="*70)

        for func_name, stats in summary.items():
            print(f"\n{func_name}:")
            print(f"  Calls:  {stats['count']}")
            print(f"  Total:  {stats['total']:.3f}s")
            print(f"  Mean:   {stats['mean']:.3f}s")
            print(f"  Std:    {stats['std']:.3f}s")
            print(f"  Range:  [{stats['min']:.3f}s, {stats['max']:.3f}s]")

    def reset(self):
        """Reset all timings."""
        self.timings.clear()


# Global profiler instance
_global_profiler = TimingProfiler()


def timed(func: Callable) -> Callable:
    """Decorator for simple timing (uses global profiler).

    Args:
        func: Function to time

    Returns:
        Wrapped function

    Example:
        @timed
        def my_function(...):
            ...

        # Later:
        from visualization.profiling import _global_profiler
        _global_profiler.print_summary()
    """
    return _global_profiler.profile(func)


# =============================================================================
# Profile Report Generation
# =============================================================================

def create_profile_report(
    profile_data: Dict[str, Any],
    output_path: Path,
    format: str = 'json'
):
    """Create profile report from profiling data.

    Args:
        profile_data: Dictionary with profiling data
        output_path: Path to save report
        format: Report format ('json', 'html', 'txt')
    """
    output_path = Path(output_path)

    if format == 'json':
        # Convert to JSON-serializable format
        report = _make_json_serializable(profile_data)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"JSON report saved to {output_path}")

    elif format == 'html':
        # Create HTML report
        html = _create_html_report(profile_data)

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"HTML report saved to {output_path}")

    elif format == 'txt':
        # Create text report
        text = _create_text_report(profile_data)

        with open(output_path, 'w') as f:
            f.write(text)

        print(f"Text report saved to {output_path}")

    else:
        raise ValueError(f"Unknown format: {format}")


def _make_json_serializable(data: Any) -> Any:
    """Convert data to JSON-serializable format.

    Args:
        data: Input data

    Returns:
        JSON-serializable version
    """
    if isinstance(data, dict):
        return {k: _make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_make_json_serializable(item) for item in data]
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        return str(data)


def _create_html_report(profile_data: Dict[str, Any]) -> str:
    """Create HTML profile report.

    Args:
        profile_data: Profiling data

    Returns:
        HTML string
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Profile Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Performance Profile Report</h1>
    """

    # Add sections based on available data
    if 'total_time' in profile_data:
        html += f"<p><strong>Total Time:</strong> {profile_data['total_time']:.3f}s</p>"

    # Add timing summary if available
    if 'timing_summary' in profile_data:
        html += "<h2>Timing Summary</h2><table>"
        html += "<tr><th>Function</th><th>Calls</th><th>Total</th><th>Mean</th><th>Std</th></tr>"

        for func, stats in profile_data['timing_summary'].items():
            html += f"<tr><td>{func}</td>"
            html += f"<td>{stats['count']}</td>"
            html += f"<td>{stats['total']:.3f}s</td>"
            html += f"<td>{stats['mean']:.3f}s</td>"
            html += f"<td>{stats['std']:.3f}s</td></tr>"

        html += "</table>"

    html += """
    </body>
    </html>
    """

    return html


def _create_text_report(profile_data: Dict[str, Any]) -> str:
    """Create text profile report.

    Args:
        profile_data: Profiling data

    Returns:
        Text string
    """
    lines = []
    lines.append("="*70)
    lines.append("Performance Profile Report")
    lines.append("="*70)
    lines.append("")

    if 'total_time' in profile_data:
        lines.append(f"Total Time: {profile_data['total_time']:.3f}s")
        lines.append("")

    if 'timing_summary' in profile_data:
        lines.append("Timing Summary:")
        lines.append("-"*70)

        for func, stats in profile_data['timing_summary'].items():
            lines.append(f"\n{func}:")
            lines.append(f"  Calls:  {stats['count']}")
            lines.append(f"  Total:  {stats['total']:.3f}s")
            lines.append(f"  Mean:   {stats['mean']:.3f}s")
            lines.append(f"  Std:    {stats['std']:.3f}s")

    return "\n".join(lines)


# =============================================================================
# Profiling Context Manager
# =============================================================================

class ProfileContext:
    """Context manager for profiling code blocks.

    Example:
        with ProfileContext() as prof:
            # Code to profile
            solver.solve(...)

        prof.print_summary()
    """

    def __init__(self, name: str = "ProfiledCode"):
        """Initialize profile context.

        Args:
            name: Name for this profiling session
        """
        self.name = name
        self.profiler = TimingProfiler()
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Enter context."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.end_time = time.time()

    def get_elapsed_time(self) -> float:
        """Get elapsed time.

        Returns:
            Elapsed time in seconds
        """
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    def print_summary(self):
        """Print summary."""
        elapsed = self.get_elapsed_time()
        print(f"\n{self.name}: {elapsed:.3f}s")


# =============================================================================
# Comparative Profiling
# =============================================================================

def compare_implementations(
    implementations: Dict[str, Callable],
    *args,
    n_runs: int = 10,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """Compare multiple implementations.

    Args:
        implementations: Dict mapping names to functions
        *args: Arguments for functions
        n_runs: Number of runs per implementation
        **kwargs: Keyword arguments for functions

    Returns:
        Dictionary with timing statistics for each implementation
    """
    import numpy as np

    results = {}

    for name, func in implementations.items():
        times = []

        for _ in range(n_runs):
            start = time.time()
            func(*args, **kwargs)
            elapsed = time.time() - start
            times.append(elapsed)

        results[name] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'runs': n_runs
        }

    # Print comparison
    print("\n" + "="*70)
    print("Implementation Comparison")
    print("="*70)

    for name, stats in results.items():
        print(f"\n{name}:")
        print(f"  Mean: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
        print(f"  Range: [{stats['min']:.3f}s, {stats['max']:.3f}s]")

    # Find fastest
    fastest = min(results.items(), key=lambda x: x[1]['mean'])
    print(f"\nFastest: {fastest[0]} ({fastest[1]['mean']:.3f}s)")

    return results
