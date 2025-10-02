"""
Phase 4 Week 37-38: Performance Benchmarking Suite

This package provides comprehensive benchmarking tools for the nonequilibrium
physics optimal control framework. It establishes performance baselines and
validates scalability across different problem sizes and hardware configurations.

Benchmark Categories:
- Standard Problems: LQR, MPC, neural optimal control
- Scalability Studies: Strong and weak scaling
- GPU Performance: CPU vs GPU comparison
- Distributed Execution: Parallel speedup analysis

Author: Nonequilibrium Physics Agents
Date: 2025-10-01
"""

__version__ = "1.0.0"

from typing import Dict, List, Any

# Benchmark results storage
BENCHMARK_RESULTS: Dict[str, Any] = {}

def register_benchmark(name: str, result: Dict[str, Any]) -> None:
    """Register benchmark result for reporting.

    Parameters
    ----------
    name : str
        Benchmark name
    result : Dict[str, Any]
        Benchmark results
    """
    BENCHMARK_RESULTS[name] = result

def get_benchmark_results() -> Dict[str, Any]:
    """Get all registered benchmark results.

    Returns
    -------
    Dict[str, Any]
        All benchmark results
    """
    return BENCHMARK_RESULTS.copy()

def clear_benchmark_results() -> None:
    """Clear all registered benchmark results."""
    BENCHMARK_RESULTS.clear()
