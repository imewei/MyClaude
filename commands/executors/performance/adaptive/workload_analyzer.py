#!/usr/bin/env python3
"""
Workload Analyzer
=================

Analyzes workload patterns to optimize performance configuration.

Features:
- File access pattern analysis
- I/O vs CPU vs Memory profiling
- Temporal pattern detection
- Predictive workload modeling

Author: Claude Code Framework
Version: 2.0
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics


@dataclass
class FileAccessPattern:
    """File access pattern information"""
    path: Path
    access_count: int = 0
    last_access: Optional[datetime] = None
    avg_access_interval_seconds: float = 0.0
    size_bytes: int = 0
    read_count: int = 0
    write_count: int = 0


@dataclass
class WorkloadMetrics:
    """Workload performance metrics"""
    total_operations: int = 0
    io_operations: int = 0
    cpu_operations: int = 0
    memory_operations: int = 0
    avg_operation_duration_ms: float = 0.0
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class WorkloadAnalysis:
    """Complete workload analysis"""
    workload_type: str  # io_bound, cpu_bound, memory_bound, balanced
    intensity: float  # 0.0 to 1.0
    patterns: List[str] = field(default_factory=list)
    hot_files: List[Path] = field(default_factory=list)
    metrics: WorkloadMetrics = field(default_factory=WorkloadMetrics)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workload_type": self.workload_type,
            "intensity": self.intensity,
            "patterns": self.patterns,
            "hot_files": [str(f) for f in self.hot_files],
            "metrics": {
                "total_operations": self.metrics.total_operations,
                "io_operations": self.metrics.io_operations,
                "cpu_operations": self.metrics.cpu_operations,
                "memory_operations": self.metrics.memory_operations,
                "avg_duration_ms": self.metrics.avg_operation_duration_ms,
                "peak_memory_mb": self.metrics.peak_memory_mb,
                "peak_cpu_percent": self.metrics.peak_cpu_percent,
                "cache_hit_rate": self.metrics.cache_hit_rate
            },
            "recommendations": self.recommendations
        }


class WorkloadAnalyzer:
    """
    Analyzes workload characteristics for performance optimization.

    Tracks:
    - File access patterns
    - I/O vs CPU vs Memory usage
    - Hot files and directories
    - Temporal access patterns
    - Cache effectiveness
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.logger = logging.getLogger(self.__class__.__name__)

        # Access tracking
        self.file_patterns: Dict[Path, FileAccessPattern] = {}
        self.access_history: deque = deque(maxlen=window_size)

        # Operation tracking
        self.operation_types: defaultdict = defaultdict(int)
        self.operation_durations: List[float] = []

        # Resource tracking
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []

        # Cache tracking
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # Start time
        self.start_time = datetime.now()

    def record_file_access(
        self,
        path: Path,
        operation: str = "read",
        duration_ms: float = 0.0
    ):
        """
        Record file access.

        Args:
            path: File path
            operation: Operation type (read, write)
            duration_ms: Operation duration in milliseconds
        """
        now = datetime.now()

        # Update pattern
        if path not in self.file_patterns:
            try:
                size = path.stat().st_size if path.exists() else 0
            except:
                size = 0

            self.file_patterns[path] = FileAccessPattern(
                path=path,
                size_bytes=size
            )

        pattern = self.file_patterns[path]
        pattern.access_count += 1

        if operation == "read":
            pattern.read_count += 1
        elif operation == "write":
            pattern.write_count += 1

        # Update timing
        if pattern.last_access:
            interval = (now - pattern.last_access).total_seconds()
            if pattern.avg_access_interval_seconds > 0:
                pattern.avg_access_interval_seconds = (
                    pattern.avg_access_interval_seconds * 0.9 + interval * 0.1
                )
            else:
                pattern.avg_access_interval_seconds = interval

        pattern.last_access = now

        # Record in history
        self.access_history.append({
            "timestamp": now,
            "path": path,
            "operation": operation,
            "duration_ms": duration_ms
        })

        # Track operation type
        self.operation_types[operation] += 1
        if duration_ms > 0:
            self.operation_durations.append(duration_ms)

    def record_operation(
        self,
        operation_type: str,
        duration_ms: float = 0.0,
        memory_mb: Optional[float] = None,
        cpu_percent: Optional[float] = None
    ):
        """
        Record general operation.

        Args:
            operation_type: Type of operation
            duration_ms: Duration in milliseconds
            memory_mb: Memory usage in MB
            cpu_percent: CPU usage percentage
        """
        self.operation_types[operation_type] += 1

        if duration_ms > 0:
            self.operation_durations.append(duration_ms)

        if memory_mb is not None:
            self.memory_samples.append(memory_mb)

        if cpu_percent is not None:
            self.cpu_samples.append(cpu_percent)

    def record_cache_access(self, hit: bool):
        """Record cache access"""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def analyze(self) -> WorkloadAnalysis:
        """
        Analyze workload and generate recommendations.

        Returns:
            Workload analysis with recommendations
        """
        self.logger.info("Analyzing workload patterns")

        # Determine workload type
        workload_type = self._determine_workload_type()

        # Calculate intensity
        intensity = self._calculate_intensity()

        # Identify patterns
        patterns = self._identify_patterns()

        # Find hot files
        hot_files = self._find_hot_files()

        # Collect metrics
        metrics = self._collect_metrics()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            workload_type, intensity, patterns, metrics
        )

        return WorkloadAnalysis(
            workload_type=workload_type,
            intensity=intensity,
            patterns=patterns,
            hot_files=hot_files,
            metrics=metrics,
            recommendations=recommendations
        )

    def _determine_workload_type(self) -> str:
        """Determine primary workload type"""
        total_ops = sum(self.operation_types.values())

        if total_ops == 0:
            return "unknown"

        # Count operation types
        io_ops = (
            self.operation_types.get("read", 0) +
            self.operation_types.get("write", 0) +
            self.operation_types.get("file_access", 0)
        )

        cpu_ops = (
            self.operation_types.get("compute", 0) +
            self.operation_types.get("analysis", 0) +
            self.operation_types.get("transform", 0)
        )

        memory_ops = (
            self.operation_types.get("cache", 0) +
            self.operation_types.get("allocate", 0)
        )

        # Determine dominant type
        io_ratio = io_ops / total_ops
        cpu_ratio = cpu_ops / total_ops
        memory_ratio = memory_ops / total_ops

        if io_ratio > 0.5:
            return "io_bound"
        elif cpu_ratio > 0.5:
            return "cpu_bound"
        elif memory_ratio > 0.5:
            return "memory_bound"
        else:
            return "balanced"

    def _calculate_intensity(self) -> float:
        """Calculate workload intensity (0.0 to 1.0)"""
        if not self.access_history:
            return 0.0

        # Time-based intensity
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed == 0:
            return 0.0

        ops_per_second = len(self.access_history) / elapsed

        # Normalize to 0-1 (100 ops/sec = 1.0)
        time_intensity = min(ops_per_second / 100.0, 1.0)

        # Resource-based intensity
        resource_intensity = 0.0
        if self.cpu_samples:
            avg_cpu = statistics.mean(self.cpu_samples)
            resource_intensity = max(resource_intensity, avg_cpu / 100.0)

        if self.memory_samples:
            avg_memory = statistics.mean(self.memory_samples)
            resource_intensity = max(resource_intensity, avg_memory / 1024.0)

        # Combined intensity
        return (time_intensity + resource_intensity) / 2.0

    def _identify_patterns(self) -> List[str]:
        """Identify access patterns"""
        patterns = []

        if not self.file_patterns:
            return patterns

        # Sequential vs random access
        sequential_count = 0
        total_accesses = len(self.access_history)

        if total_accesses > 1:
            for i in range(1, min(100, total_accesses)):
                curr = self.access_history[i]
                prev = self.access_history[i-1]

                # Check if same or nearby file
                if curr["path"] == prev["path"]:
                    sequential_count += 1

            if sequential_count / min(100, total_accesses) > 0.7:
                patterns.append("sequential_access")
            else:
                patterns.append("random_access")

        # Hot file pattern
        access_counts = [p.access_count for p in self.file_patterns.values()]
        if access_counts:
            max_access = max(access_counts)
            avg_access = statistics.mean(access_counts)

            if max_access > avg_access * 5:
                patterns.append("hot_file_concentration")

        # Repeated access pattern
        repeated = sum(1 for p in self.file_patterns.values() if p.access_count > 2)
        if repeated / len(self.file_patterns) > 0.5:
            patterns.append("repeated_access")

        # Read vs write heavy
        total_reads = sum(p.read_count for p in self.file_patterns.values())
        total_writes = sum(p.write_count for p in self.file_patterns.values())

        if total_reads > 0 or total_writes > 0:
            read_ratio = total_reads / (total_reads + total_writes)
            if read_ratio > 0.8:
                patterns.append("read_heavy")
            elif read_ratio < 0.2:
                patterns.append("write_heavy")
            else:
                patterns.append("balanced_io")

        return patterns

    def _find_hot_files(self, top_n: int = 10) -> List[Path]:
        """Find most frequently accessed files"""
        sorted_files = sorted(
            self.file_patterns.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )

        return [path for path, _ in sorted_files[:top_n]]

    def _collect_metrics(self) -> WorkloadMetrics:
        """Collect workload metrics"""
        metrics = WorkloadMetrics()

        # Operation counts
        metrics.total_operations = sum(self.operation_types.values())
        metrics.io_operations = (
            self.operation_types.get("read", 0) +
            self.operation_types.get("write", 0)
        )
        metrics.cpu_operations = self.operation_types.get("compute", 0)
        metrics.memory_operations = self.operation_types.get("cache", 0)

        # Duration
        if self.operation_durations:
            metrics.avg_operation_duration_ms = statistics.mean(
                self.operation_durations
            )

        # Resource peaks
        if self.memory_samples:
            metrics.peak_memory_mb = max(self.memory_samples)

        if self.cpu_samples:
            metrics.peak_cpu_percent = max(self.cpu_samples)

        # Cache hit rate
        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops > 0:
            metrics.cache_hit_rate = (self.cache_hits / total_cache_ops) * 100

        return metrics

    def _generate_recommendations(
        self,
        workload_type: str,
        intensity: float,
        patterns: List[str],
        metrics: WorkloadMetrics
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Workload-specific recommendations
        if workload_type == "io_bound":
            recommendations.append(
                "I/O-bound workload detected: Increase thread pool size"
            )
            recommendations.append(
                "Enable streaming and buffering for large files"
            )

        elif workload_type == "cpu_bound":
            recommendations.append(
                "CPU-bound workload detected: Use process pool for parallelism"
            )
            recommendations.append(
                "Consider batch processing to reduce overhead"
            )

        elif workload_type == "memory_bound":
            recommendations.append(
                "Memory-bound workload: Enable lazy loading"
            )
            recommendations.append(
                "Reduce cache sizes and enable streaming"
            )

        # Pattern-based recommendations
        if "sequential_access" in patterns:
            recommendations.append(
                "Sequential access detected: Enable prefetching"
            )

        if "hot_file_concentration" in patterns:
            recommendations.append(
                "Hot files detected: Increase L1 cache for frequently accessed files"
            )

        if "repeated_access" in patterns:
            recommendations.append(
                "Repeated access pattern: Increase cache TTL"
            )

        # Cache recommendations
        if metrics.cache_hit_rate < 50:
            recommendations.append(
                f"Low cache hit rate ({metrics.cache_hit_rate:.1f}%): Increase cache size"
            )

        # Intensity recommendations
        if intensity > 0.8:
            recommendations.append(
                "High intensity workload: Consider load balancing"
            )

        return recommendations

    def get_summary(self) -> Dict[str, Any]:
        """Get workload summary"""
        return {
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_files_accessed": len(self.file_patterns),
            "total_operations": sum(self.operation_types.values()),
            "operation_breakdown": dict(self.operation_types),
            "cache_hit_rate": (
                (self.cache_hits / (self.cache_hits + self.cache_misses) * 100)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            ),
            "hot_files": len(self._find_hot_files())
        }


def main():
    """Test workload analyzer"""
    import random

    logging.basicConfig(level=logging.INFO)

    analyzer = WorkloadAnalyzer()

    # Simulate workload
    test_files = [Path(f"file_{i}.py") for i in range(20)]

    print("Simulating workload...")
    for _ in range(100):
        # Simulate hot file concentration
        if random.random() < 0.7:
            file = test_files[0]  # Hot file
        else:
            file = random.choice(test_files)

        analyzer.record_file_access(file, "read", duration_ms=random.uniform(1, 50))
        analyzer.record_cache_access(hit=random.random() < 0.6)

    # Analyze
    analysis = analyzer.analyze()

    print("\nWorkload Analysis:")
    print("=" * 60)
    print(f"Workload Type: {analysis.workload_type}")
    print(f"Intensity: {analysis.intensity:.2f}")
    print(f"Patterns: {', '.join(analysis.patterns)}")
    print(f"\nCache Hit Rate: {analysis.metrics.cache_hit_rate:.1f}%")
    print(f"\nRecommendations:")
    for rec in analysis.recommendations:
        print(f"  - {rec}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())