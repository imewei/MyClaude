#!/usr/bin/env python3
"""
Metrics Collector - Collects comprehensive performance and quality metrics.

This module collects various metrics during validation runs including:
- Execution time
- Memory usage
- CPU utilization
- Disk I/O
- Cache hit rates
- Agent coordination efficiency
"""

import os
import sys
import psutil
import resource
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import subprocess


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during execution."""
    execution_time_seconds: float
    cpu_percent: float
    memory_mb: float
    memory_peak_mb: float
    disk_read_mb: float
    disk_write_mb: float
    cache_hit_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityMetrics:
    """Code quality metrics."""
    quality_score: float
    complexity_average: float
    complexity_max: int
    maintainability_index: float
    test_coverage_percent: float
    documentation_coverage_percent: float
    security_issues_count: int
    code_smells_count: int
    duplication_percent: float
    lines_of_code: int
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Collects various metrics during validation."""

    def __init__(self):
        """Initialize metrics collector."""
        self.process = psutil.Process()
        self.start_time: Optional[float] = None
        self.start_io: Optional[psutil._common.pio] = None
        self.metrics_history: List[Dict[str, Any]] = []

    def start_collection(self) -> None:
        """Start collecting metrics."""
        self.start_time = time.time()
        try:
            self.start_io = self.process.io_counters()
        except (AttributeError, psutil.AccessDenied):
            self.start_io = None

    def collect_performance(self) -> PerformanceMetrics:
        """Collect performance metrics.

        Returns:
            PerformanceMetrics object
        """
        if self.start_time is None:
            self.start_collection()

        # Execution time
        execution_time = time.time() - self.start_time if self.start_time else 0

        # CPU usage
        cpu_percent = self.process.cpu_percent(interval=0.1)

        # Memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        # Peak memory (from resource module)
        try:
            if os.name == 'posix':
                peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # On macOS, ru_maxrss is in bytes; on Linux, it's in KB
                if sys.platform == 'darwin':
                    memory_peak_mb = peak_memory_kb / (1024 * 1024)
                else:
                    memory_peak_mb = peak_memory_kb / 1024
            else:
                memory_peak_mb = memory_mb
        except Exception:
            memory_peak_mb = memory_mb

        # Disk I/O
        disk_read_mb = 0.0
        disk_write_mb = 0.0

        try:
            current_io = self.process.io_counters()
            if self.start_io:
                disk_read_mb = (current_io.read_bytes - self.start_io.read_bytes) / (1024 * 1024)
                disk_write_mb = (current_io.write_bytes - self.start_io.write_bytes) / (1024 * 1024)
        except (AttributeError, psutil.AccessDenied):
            pass

        # Cache hit rate (placeholder - would be collected from actual cache system)
        cache_hit_rate = 0.75  # Simulated

        return PerformanceMetrics(
            execution_time_seconds=execution_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_peak_mb=memory_peak_mb,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            cache_hit_rate=cache_hit_rate
        )

    def collect_quality(self, project_path: Path) -> QualityMetrics:
        """Collect code quality metrics for a project.

        Args:
            project_path: Path to project

        Returns:
            QualityMetrics object
        """
        # Count lines of code
        loc = self._count_lines_of_code(project_path)

        # Analyze complexity (placeholder - would use radon or similar)
        complexity_avg = 5.0  # Simulated
        complexity_max = 15  # Simulated

        # Calculate quality score (0-100)
        quality_score = 75.0  # Simulated

        # Maintainability index
        maintainability = 65.0  # Simulated

        # Test coverage (would run coverage tool)
        test_coverage = 0.0  # Simulated

        # Documentation coverage
        doc_coverage = self._calculate_doc_coverage(project_path)

        # Security issues (would run bandit or similar)
        security_issues = 0  # Simulated

        # Code smells (would run pylint or similar)
        code_smells = 5  # Simulated

        # Duplication (would run jscpd or similar)
        duplication = 2.5  # Simulated

        return QualityMetrics(
            quality_score=quality_score,
            complexity_average=complexity_avg,
            complexity_max=complexity_max,
            maintainability_index=maintainability,
            test_coverage_percent=test_coverage,
            documentation_coverage_percent=doc_coverage,
            security_issues_count=security_issues,
            code_smells_count=code_smells,
            duplication_percent=duplication,
            lines_of_code=loc
        )

    def _count_lines_of_code(self, project_path: Path) -> int:
        """Count lines of code in project.

        Args:
            project_path: Path to project

        Returns:
            Total lines of code
        """
        total_lines = 0
        extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.jl'}

        for root, _, files in os.walk(project_path):
            # Skip common non-source directories
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', 'venv', '.venv']):
                continue

            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines += sum(1 for line in f if line.strip())
                    except Exception:
                        continue

        return total_lines

    def _calculate_doc_coverage(self, project_path: Path) -> float:
        """Calculate documentation coverage.

        Args:
            project_path: Path to project

        Returns:
            Documentation coverage percentage
        """
        total_functions = 0
        documented_functions = 0

        for root, _, files in os.walk(project_path):
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', 'venv']):
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        in_docstring = False
                        for i, line in enumerate(lines):
                            stripped = line.strip()

                            # Check for function definition
                            if stripped.startswith('def '):
                                total_functions += 1

                                # Check if next non-empty line is docstring
                                for j in range(i + 1, min(i + 5, len(lines))):
                                    next_line = lines[j].strip()
                                    if next_line.startswith('"""') or next_line.startswith("'''"):
                                        documented_functions += 1
                                        break
                                    elif next_line and not next_line.startswith('#'):
                                        break

                    except Exception:
                        continue

        if total_functions == 0:
            return 0.0

        return (documented_functions / total_functions) * 100

    def collect(
        self,
        project_path: Path,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Collect specified metrics.

        Args:
            project_path: Path to project
            metric_names: List of metric names to collect (None = all)

        Returns:
            Dictionary of collected metrics
        """
        metrics = {}

        collect_all = metric_names is None

        # Performance metrics
        if collect_all or any(m in metric_names for m in [
            'execution_time', 'memory_usage', 'cpu_utilization',
            'disk_io', 'cache_hit_rate'
        ]):
            perf = self.collect_performance()
            metrics.update({
                'execution_time': perf.execution_time_seconds,
                'memory_usage': perf.memory_mb,
                'memory_peak': perf.memory_peak_mb,
                'cpu_utilization': perf.cpu_percent,
                'disk_read_mb': perf.disk_read_mb,
                'disk_write_mb': perf.disk_write_mb,
                'cache_hit_rate': perf.cache_hit_rate
            })

        # Quality metrics
        if collect_all or any(m in metric_names for m in [
            'quality_score', 'complexity', 'maintainability',
            'test_coverage', 'documentation_completeness'
        ]):
            quality = self.collect_quality(project_path)
            metrics.update({
                'quality_score': quality.quality_score,
                'complexity_average': quality.complexity_average,
                'complexity_max': quality.complexity_max,
                'maintainability_index': quality.maintainability_index,
                'test_coverage': quality.test_coverage_percent,
                'documentation_completeness': quality.documentation_coverage_percent,
                'security_issues_found': quality.security_issues_count,
                'code_smells': quality.code_smells_count,
                'duplication_percent': quality.duplication_percent,
                'lines_of_code': quality.lines_of_code
            })

        # Store in history
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'project_path': str(project_path),
            'metrics': metrics
        })

        return metrics

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics.

        Returns:
            Summary dictionary
        """
        if not self.metrics_history:
            return {}

        # Calculate averages and aggregates
        total_metrics = len(self.metrics_history)

        summary = {
            'total_collections': total_metrics,
            'first_collection': self.metrics_history[0]['timestamp'],
            'last_collection': self.metrics_history[-1]['timestamp'],
        }

        # Average numeric metrics
        numeric_metrics = [
            'execution_time', 'memory_usage', 'cpu_utilization',
            'quality_score', 'test_coverage', 'documentation_completeness'
        ]

        for metric in numeric_metrics:
            values = [
                m['metrics'].get(metric)
                for m in self.metrics_history
                if metric in m['metrics']
            ]
            if values:
                summary[f'{metric}_avg'] = sum(values) / len(values)
                summary[f'{metric}_min'] = min(values)
                summary[f'{metric}_max'] = max(values)

        return summary

    def export_metrics(self, output_path: Path) -> None:
        """Export collected metrics to JSON file.

        Args:
            output_path: Path to output file
        """
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'summary': self.get_metrics_summary(),
            'history': self.metrics_history
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


# For standalone testing
if __name__ == "__main__":
    import sys

    collector = MetricsCollector()
    collector.start_collection()

    # Simulate some work
    time.sleep(1)

    # Collect metrics
    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
    else:
        project_path = Path.cwd()

    metrics = collector.collect(project_path)

    print("Collected Metrics:")
    print("-" * 50)
    for key, value in metrics.items():
        print(f"{key:30s}: {value}")

    print("\nMetrics Summary:")
    print("-" * 50)
    summary = collector.get_metrics_summary()
    for key, value in summary.items():
        print(f"{key:30s}: {value}")