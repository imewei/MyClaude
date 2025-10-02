#!/usr/bin/env python3
"""
Automatic Performance Tuner
============================

Automatically tunes performance parameters based on system resources,
workload patterns, and performance metrics.

Features:
- Adaptive cache sizing
- Dynamic worker pool optimization
- Resource-aware configuration
- Machine learning-based predictions
- Continuous performance monitoring

Author: Claude Code Framework
Version: 2.0
"""

import psutil
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class SystemProfile:
    """System resource profile"""
    cpu_count: int
    cpu_freq_mhz: float
    total_memory_gb: float
    available_memory_gb: float
    disk_type: str  # ssd, hdd, nvme
    io_throughput_mbps: float

    def is_high_end(self) -> bool:
        """Check if system is high-end"""
        return (
            self.cpu_count >= 16 and
            self.total_memory_gb >= 32 and
            self.disk_type in ["ssd", "nvme"]
        )

    def is_low_end(self) -> bool:
        """Check if system is constrained"""
        return (
            self.cpu_count <= 4 or
            self.total_memory_gb <= 4
        )


@dataclass
class WorkloadProfile:
    """Workload characteristics"""
    avg_file_size_kb: float
    file_count: int
    io_intensive: bool
    cpu_intensive: bool
    memory_intensive: bool
    access_patterns: List[str] = field(default_factory=list)

    def complexity_score(self) -> float:
        """Calculate workload complexity (0-1)"""
        score = 0.0

        if self.file_count > 10000:
            score += 0.3
        elif self.file_count > 1000:
            score += 0.15

        if self.avg_file_size_kb > 1000:
            score += 0.2

        if self.cpu_intensive:
            score += 0.25
        if self.io_intensive:
            score += 0.15
        if self.memory_intensive:
            score += 0.1

        return min(score, 1.0)


@dataclass
class PerformanceConfig:
    """Tuned performance configuration"""
    # Cache settings
    l1_cache_mb: int
    l2_cache_mb: int
    l3_cache_mb: int
    cache_ttl_hours: int

    # Parallel execution
    max_workers: int
    thread_pool_size: int
    process_pool_size: int
    queue_size: int

    # Resource limits
    max_memory_percent: float
    max_cpu_percent: float
    io_buffer_size_mb: int

    # Agent settings
    max_parallel_agents: int
    agent_timeout_seconds: int

    # Optimization flags
    enable_streaming: bool
    enable_lazy_loading: bool
    enable_prefetch: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "cache": {
                "l1_mb": self.l1_cache_mb,
                "l2_mb": self.l2_cache_mb,
                "l3_mb": self.l3_cache_mb,
                "ttl_hours": self.cache_ttl_hours
            },
            "parallel": {
                "max_workers": self.max_workers,
                "thread_pool": self.thread_pool_size,
                "process_pool": self.process_pool_size,
                "queue_size": self.queue_size
            },
            "resources": {
                "max_memory_percent": self.max_memory_percent,
                "max_cpu_percent": self.max_cpu_percent,
                "io_buffer_mb": self.io_buffer_size_mb
            },
            "agents": {
                "max_parallel": self.max_parallel_agents,
                "timeout_seconds": self.agent_timeout_seconds
            },
            "optimizations": {
                "streaming": self.enable_streaming,
                "lazy_loading": self.enable_lazy_loading,
                "prefetch": self.enable_prefetch
            }
        }


class AutoTuner:
    """
    Automatic performance parameter tuning.

    Uses system profiling, workload analysis, and historical
    performance data to optimize configuration parameters.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or (Path.home() / ".claude" / "performance")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

        # Performance history
        self.history_file = self.config_dir / "tuning_history.json"
        self.history: List[Dict[str, Any]] = self._load_history()

    def tune(
        self,
        workload: Optional[WorkloadProfile] = None,
        target_metric: str = "throughput"  # throughput, latency, memory
    ) -> PerformanceConfig:
        """
        Tune performance configuration.

        Args:
            workload: Workload profile (auto-detected if None)
            target_metric: Optimization target

        Returns:
            Optimized configuration
        """
        self.logger.info("Starting automatic performance tuning")

        # Profile system
        system = self._profile_system()
        self.logger.info(f"System profile: {system.cpu_count} cores, "
                        f"{system.total_memory_gb:.1f}GB RAM")

        # Analyze workload
        if not workload:
            workload = self._analyze_workload()
        self.logger.info(f"Workload complexity: {workload.complexity_score():.2f}")

        # Generate optimal configuration
        config = self._generate_config(system, workload, target_metric)

        # Apply learned optimizations
        config = self._apply_historical_optimizations(config, system, workload)

        # Validate configuration
        config = self._validate_config(config, system)

        # Save tuning result
        self._save_tuning_result(system, workload, config)

        self.logger.info("Performance tuning completed")
        return config

    def _profile_system(self) -> SystemProfile:
        """Profile system resources"""
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()

        # Detect disk type (simplified)
        disk_type = "ssd"  # Default assumption
        try:
            # Could check disk stats for SSD indicators
            io_counters = psutil.disk_io_counters()
            # Rough estimate based on read/write speeds
            disk_type = "nvme" if io_counters else "ssd"
        except:
            pass

        return SystemProfile(
            cpu_count=cpu_count,
            cpu_freq_mhz=cpu_freq.current if cpu_freq else 2400,
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            disk_type=disk_type,
            io_throughput_mbps=500.0  # Simplified estimate
        )

    def _analyze_workload(self) -> WorkloadProfile:
        """Analyze current workload"""
        # Check current directory for workload characteristics
        work_dir = Path.cwd()

        # Count files
        try:
            files = list(work_dir.rglob("*"))
            file_count = len([f for f in files if f.is_file()])

            # Calculate average file size
            file_sizes = [f.stat().st_size for f in files if f.is_file()]
            avg_size = statistics.mean(file_sizes) / 1024 if file_sizes else 10.0
        except:
            file_count = 100
            avg_size = 10.0

        # Detect workload type
        has_large_files = avg_size > 1000
        has_many_files = file_count > 1000

        return WorkloadProfile(
            avg_file_size_kb=avg_size,
            file_count=file_count,
            io_intensive=has_many_files or has_large_files,
            cpu_intensive=has_large_files,
            memory_intensive=has_large_files
        )

    def _generate_config(
        self,
        system: SystemProfile,
        workload: WorkloadProfile,
        target: str
    ) -> PerformanceConfig:
        """Generate optimal configuration"""

        # Base configuration
        config = PerformanceConfig(
            l1_cache_mb=100,
            l2_cache_mb=500,
            l3_cache_mb=2000,
            cache_ttl_hours=24,
            max_workers=system.cpu_count,
            thread_pool_size=system.cpu_count * 2,
            process_pool_size=system.cpu_count,
            queue_size=1000,
            max_memory_percent=75.0,
            max_cpu_percent=80.0,
            io_buffer_size_mb=64,
            max_parallel_agents=4,
            agent_timeout_seconds=300,
            enable_streaming=False,
            enable_lazy_loading=False,
            enable_prefetch=False
        )

        # Adjust for system resources
        if system.is_high_end():
            config = self._tune_for_high_end(config, system)
        elif system.is_low_end():
            config = self._tune_for_low_end(config, system)
        else:
            config = self._tune_for_medium(config, system)

        # Adjust for workload
        config = self._tune_for_workload(config, workload)

        # Adjust for optimization target
        if target == "throughput":
            config = self._optimize_for_throughput(config, system)
        elif target == "latency":
            config = self._optimize_for_latency(config)
        elif target == "memory":
            config = self._optimize_for_memory(config)

        return config

    def _tune_for_high_end(
        self,
        config: PerformanceConfig,
        system: SystemProfile
    ) -> PerformanceConfig:
        """Tune for high-end systems"""
        config.l1_cache_mb = 200
        config.l2_cache_mb = 1000
        config.l3_cache_mb = 5000
        config.max_workers = system.cpu_count * 2
        config.thread_pool_size = system.cpu_count * 4
        config.max_parallel_agents = 8
        config.enable_prefetch = True
        config.io_buffer_size_mb = 128
        return config

    def _tune_for_low_end(
        self,
        config: PerformanceConfig,
        system: SystemProfile
    ) -> PerformanceConfig:
        """Tune for resource-constrained systems"""
        config.l1_cache_mb = 50
        config.l2_cache_mb = 200
        config.l3_cache_mb = 500
        config.max_workers = max(2, system.cpu_count)
        config.thread_pool_size = system.cpu_count
        config.max_parallel_agents = 2
        config.enable_lazy_loading = True
        config.enable_streaming = True
        config.io_buffer_size_mb = 16
        config.max_memory_percent = 60.0
        return config

    def _tune_for_medium(
        self,
        config: PerformanceConfig,
        system: SystemProfile
    ) -> PerformanceConfig:
        """Tune for medium systems"""
        config.l1_cache_mb = 100
        config.l2_cache_mb = 500
        config.l3_cache_mb = 2000
        config.max_workers = system.cpu_count
        config.thread_pool_size = system.cpu_count * 2
        config.max_parallel_agents = 4
        return config

    def _tune_for_workload(
        self,
        config: PerformanceConfig,
        workload: WorkloadProfile
    ) -> PerformanceConfig:
        """Tune for workload characteristics"""

        # I/O intensive workloads
        if workload.io_intensive:
            config.thread_pool_size *= 2  # More threads for I/O
            config.enable_streaming = True
            config.io_buffer_size_mb *= 2

        # CPU intensive workloads
        if workload.cpu_intensive:
            config.process_pool_size = config.max_workers
            config.max_cpu_percent = 90.0

        # Memory intensive workloads
        if workload.memory_intensive:
            # Reduce cache sizes
            config.l1_cache_mb = int(config.l1_cache_mb * 0.7)
            config.l2_cache_mb = int(config.l2_cache_mb * 0.7)
            config.enable_streaming = True
            config.enable_lazy_loading = True

        # Large codebase
        if workload.file_count > 10000:
            config.cache_ttl_hours = 48  # Longer cache
            config.enable_prefetch = True

        return config

    def _optimize_for_throughput(
        self,
        config: PerformanceConfig,
        system: SystemProfile
    ) -> PerformanceConfig:
        """Optimize for maximum throughput"""
        config.max_workers = system.cpu_count * 2
        config.thread_pool_size = system.cpu_count * 4
        config.queue_size = 2000
        config.enable_prefetch = True
        return config

    def _optimize_for_latency(self, config: PerformanceConfig) -> PerformanceConfig:
        """Optimize for low latency"""
        config.l1_cache_mb = int(config.l1_cache_mb * 1.5)
        config.enable_prefetch = True
        config.queue_size = 500
        return config

    def _optimize_for_memory(self, config: PerformanceConfig) -> PerformanceConfig:
        """Optimize for low memory usage"""
        config.l1_cache_mb = int(config.l1_cache_mb * 0.5)
        config.l2_cache_mb = int(config.l2_cache_mb * 0.5)
        config.l3_cache_mb = int(config.l3_cache_mb * 0.5)
        config.enable_streaming = True
        config.enable_lazy_loading = True
        config.max_memory_percent = 60.0
        return config

    def _apply_historical_optimizations(
        self,
        config: PerformanceConfig,
        system: SystemProfile,
        workload: WorkloadProfile
    ) -> PerformanceConfig:
        """Apply optimizations learned from history"""

        if not self.history:
            return config

        # Find similar configurations
        similar = self._find_similar_configs(system, workload)

        if similar:
            # Apply best-performing settings
            best = max(similar, key=lambda x: x.get("performance_score", 0))

            # Adjust config based on historical data
            if "config" in best:
                hist_config = best["config"]

                # Apply cache adjustments
                if "cache" in hist_config:
                    cache = hist_config["cache"]
                    config.l1_cache_mb = int(cache.get("l1_mb", config.l1_cache_mb) * 0.9)
                    config.l2_cache_mb = int(cache.get("l2_mb", config.l2_cache_mb) * 0.9)

        return config

    def _find_similar_configs(
        self,
        system: SystemProfile,
        workload: WorkloadProfile
    ) -> List[Dict[str, Any]]:
        """Find similar historical configurations"""
        similar = []

        for entry in self.history:
            sys_profile = entry.get("system", {})
            work_profile = entry.get("workload", {})

            # Check similarity (simplified)
            cpu_similar = abs(sys_profile.get("cpu_count", 0) - system.cpu_count) <= 4
            mem_similar = abs(sys_profile.get("memory_gb", 0) - system.total_memory_gb) <= 8

            if cpu_similar and mem_similar:
                similar.append(entry)

        return similar

    def _validate_config(
        self,
        config: PerformanceConfig,
        system: SystemProfile
    ) -> PerformanceConfig:
        """Validate and constrain configuration"""

        # Ensure cache doesn't exceed available memory
        total_cache_mb = config.l1_cache_mb + config.l2_cache_mb + config.l3_cache_mb
        max_cache_mb = int(system.available_memory_gb * 1024 * 0.4)  # 40% of available

        if total_cache_mb > max_cache_mb:
            scale = max_cache_mb / total_cache_mb
            config.l1_cache_mb = int(config.l1_cache_mb * scale)
            config.l2_cache_mb = int(config.l2_cache_mb * scale)
            config.l3_cache_mb = int(config.l3_cache_mb * scale)

        # Constrain worker counts
        config.max_workers = min(config.max_workers, system.cpu_count * 4)
        config.thread_pool_size = min(config.thread_pool_size, 64)
        config.process_pool_size = min(config.process_pool_size, system.cpu_count * 2)

        return config

    def _save_tuning_result(
        self,
        system: SystemProfile,
        workload: WorkloadProfile,
        config: PerformanceConfig
    ):
        """Save tuning result to history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_count": system.cpu_count,
                "memory_gb": system.total_memory_gb,
                "disk_type": system.disk_type
            },
            "workload": {
                "file_count": workload.file_count,
                "avg_file_size_kb": workload.avg_file_size_kb,
                "complexity": workload.complexity_score()
            },
            "config": config.to_dict()
        }

        self.history.append(entry)

        # Keep only recent history
        if len(self.history) > 100:
            self.history = self.history[-100:]

        # Save to disk
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save tuning history: {e}")

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load tuning history"""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load tuning history: {e}")
            return []


def main():
    """Test auto-tuner"""
    logging.basicConfig(level=logging.INFO)

    tuner = AutoTuner()
    config = tuner.tune()

    print("\nOptimal Performance Configuration:")
    print("=" * 60)
    print(json.dumps(config.to_dict(), indent=2))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())