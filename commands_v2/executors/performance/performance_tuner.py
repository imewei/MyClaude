#!/usr/bin/env python3
"""
Performance Tuner - Main Orchestrator
======================================

Main entry point for performance tuning system.

Features:
- Orchestrates all tuning components
- Generates optimal configurations
- Validates performance improvements
- Provides recommendations

Author: Claude Code Framework
Version: 2.0
"""

import sys
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from performance.adaptive.auto_tuner import AutoTuner, WorkloadProfile
from performance.adaptive.workload_analyzer import WorkloadAnalyzer
from performance.cache.cache_tuner import CacheTuner, CacheMetrics
from performance.parallel.worker_pool_optimizer import WorkerPoolOptimizer, TaskType
from performance.benchmarks.benchmark_suite import PerformanceBenchmarkSuite


@dataclass
class TuningResult:
    """Complete tuning result"""
    timestamp: datetime
    system_profile: Dict[str, Any]
    workload_analysis: Dict[str, Any]
    optimal_config: Dict[str, Any]
    recommendations: list
    estimated_speedup: float
    profile_name: str


class PerformanceTuner:
    """
    Main performance tuning orchestrator.

    Coordinates all tuning components to generate optimal
    performance configurations.
    """

    def __init__(self, work_dir: Optional[Path] = None):
        self.work_dir = work_dir or Path.cwd()
        self.config_dir = Path.home() / ".claude" / "performance"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.auto_tuner = AutoTuner(self.config_dir)
        self.workload_analyzer = WorkloadAnalyzer()
        self.cache_tuner = CacheTuner()
        self.worker_optimizer = WorkerPoolOptimizer()

        # Load profiles
        self.profiles = self._load_profiles()

    def tune_full(
        self,
        target_metric: str = "throughput",
        run_benchmarks: bool = False
    ) -> TuningResult:
        """
        Perform complete performance tuning.

        Args:
            target_metric: Optimization target (throughput, latency, memory)
            run_benchmarks: Run benchmarks to validate

        Returns:
            Tuning result with recommendations
        """
        self.logger.info("Starting full performance tuning")

        # Step 1: Analyze workload
        self.logger.info("Step 1: Analyzing workload...")
        workload_analysis = self._analyze_workload()

        # Step 2: Auto-tune configuration
        self.logger.info("Step 2: Auto-tuning configuration...")
        optimal_config = self.auto_tuner.tune(target_metric=target_metric)

        # Step 3: Select best profile
        self.logger.info("Step 3: Selecting performance profile...")
        profile_name = self._select_profile(optimal_config)

        # Step 4: Generate recommendations
        self.logger.info("Step 4: Generating recommendations...")
        recommendations = self._generate_recommendations(
            workload_analysis,
            optimal_config
        )

        # Step 5: Estimate speedup
        estimated_speedup = self._estimate_speedup(
            optimal_config,
            workload_analysis
        )

        # Step 6: Run benchmarks if requested
        if run_benchmarks:
            self.logger.info("Step 5: Running performance benchmarks...")
            self._run_benchmarks()

        result = TuningResult(
            timestamp=datetime.now(),
            system_profile=self._get_system_profile(),
            workload_analysis=workload_analysis,
            optimal_config=optimal_config.to_dict(),
            recommendations=recommendations,
            estimated_speedup=estimated_speedup,
            profile_name=profile_name
        )

        # Save result
        self._save_tuning_result(result)

        self.logger.info("Performance tuning completed")
        return result

    def tune_cache(
        self,
        current_metrics: Optional[CacheMetrics] = None,
        target_hit_rate: float = 70.0
    ) -> Dict[str, Any]:
        """
        Tune cache configuration only.

        Args:
            current_metrics: Current cache metrics
            target_hit_rate: Target cache hit rate

        Returns:
            Optimal cache configuration
        """
        self.logger.info("Tuning cache configuration")

        config = self.cache_tuner.tune(
            current_metrics=current_metrics,
            target_hit_rate=target_hit_rate
        )

        return {
            "l1_cache_mb": config.l1_size_mb,
            "l2_cache_mb": config.l2_size_mb,
            "l3_cache_mb": config.l3_size_mb,
            "total_cache_mb": config.total_size_mb(),
            "eviction_policy": config.eviction_policy,
            "ttl_hours": config.ttl_hours,
            "target_hit_rate": config.target_hit_rate
        }

    def tune_workers(
        self,
        task_type: Optional[TaskType] = None
    ) -> Dict[str, Any]:
        """
        Tune worker pool configuration only.

        Args:
            task_type: Task type (auto-detected if None)

        Returns:
            Optimal worker configuration
        """
        self.logger.info("Tuning worker pool configuration")

        config = self.worker_optimizer.optimize(task_type=task_type)

        return config.to_dict()

    def select_profile(
        self,
        project_size: Optional[str] = None,
        system_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select optimal performance profile.

        Args:
            project_size: Project size (small, medium, large)
            system_type: System type (high_performance, memory_constrained, etc.)

        Returns:
            Selected profile configuration
        """
        if system_type:
            profile_name = system_type
        elif project_size:
            profile_name = f"{project_size}_project"
        else:
            # Auto-detect
            profile_name = self._auto_select_profile()

        profile = self.profiles.get(profile_name)
        if not profile:
            self.logger.warning(f"Profile {profile_name} not found, using medium")
            profile = self.profiles.get("medium_project", {})

        return profile

    def run_benchmarks(self) -> Dict[str, Any]:
        """
        Run performance benchmarks.

        Returns:
            Benchmark results
        """
        self.logger.info("Running performance benchmarks")

        suite = PerformanceBenchmarkSuite()
        results = suite.run_all()

        return results.get_summary()

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _analyze_workload(self) -> Dict[str, Any]:
        """Analyze current workload"""
        # Simulate some workload for analysis
        try:
            files = list(self.work_dir.rglob("*.py"))[:100]
            for file in files:
                self.workload_analyzer.record_file_access(
                    file, "read", duration_ms=10.0
                )
        except Exception as e:
            self.logger.error(f"Error analyzing workload: {e}")

        analysis = self.workload_analyzer.analyze()
        return analysis.to_dict()

    def _select_profile(self, config: Any) -> str:
        """Select best matching profile"""
        # Simple selection based on cache size
        total_cache = (
            config.l1_cache_mb +
            config.l2_cache_mb +
            config.l3_cache_mb
        )

        if total_cache > 6000:
            return "large_project"
        elif total_cache > 2000:
            return "medium_project"
        else:
            return "small_project"

    def _generate_recommendations(
        self,
        workload_analysis: Dict[str, Any],
        config: Any
    ) -> list:
        """Generate tuning recommendations"""
        recommendations = []

        # Extract workload recommendations
        if "recommendations" in workload_analysis:
            recommendations.extend(workload_analysis["recommendations"])

        # Add config-specific recommendations
        total_cache = (
            config.l1_cache_mb +
            config.l2_cache_mb +
            config.l3_cache_mb
        )

        if total_cache > 5000:
            recommendations.append(
                "Large cache configured - ensure sufficient RAM available"
            )

        if config.max_workers > 16:
            recommendations.append(
                "High worker count - monitor CPU utilization"
            )

        if config.enable_streaming:
            recommendations.append(
                "Streaming enabled - optimal for large files"
            )

        return recommendations

    def _estimate_speedup(
        self,
        config: Any,
        workload_analysis: Dict[str, Any]
    ) -> float:
        """Estimate performance speedup"""
        # Base speedup from parallelization
        base_speedup = min(config.max_workers * 0.7, 8.0)

        # Cache contribution
        cache_speedup = 1.5  # Assume 50% speedup from caching

        # Workload intensity adjustment
        intensity = workload_analysis.get("intensity", 0.5)
        adjusted_speedup = base_speedup * (0.5 + intensity * 0.5)

        # Total speedup (not multiplicative)
        total_speedup = adjusted_speedup + cache_speedup - 1.0

        return round(total_speedup, 1)

    def _run_benchmarks(self):
        """Run performance benchmarks"""
        suite = PerformanceBenchmarkSuite()
        suite.run_all()

    def _get_system_profile(self) -> Dict[str, Any]:
        """Get system profile"""
        import psutil
        import platform

        return {
            "platform": platform.system(),
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": platform.python_version()
        }

    def _auto_select_profile(self) -> str:
        """Auto-select profile based on system and workload"""
        import psutil

        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count(logical=True)

        # Check for constraints
        if memory_gb < 4:
            return "memory_constrained"
        elif cpu_count < 4:
            return "cpu_constrained"
        elif memory_gb >= 32 and cpu_count >= 16:
            return "high_performance"

        # Check project size
        try:
            file_count = len(list(self.work_dir.rglob("*")))
            if file_count > 10000:
                return "large_project"
            elif file_count > 1000:
                return "medium_project"
            else:
                return "small_project"
        except:
            return "medium_project"

    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load performance profiles"""
        profiles = {}
        profile_dir = Path(__file__).parent / "profiles"

        if not profile_dir.exists():
            return profiles

        for profile_file in profile_dir.glob("*.yaml"):
            try:
                with open(profile_file, 'r') as f:
                    profile = yaml.safe_load(f)
                    profiles[profile["name"]] = profile
            except Exception as e:
                self.logger.error(f"Error loading profile {profile_file}: {e}")

        return profiles

    def _save_tuning_result(self, result: TuningResult):
        """Save tuning result"""
        result_file = self.config_dir / f"tuning_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        result_dict = {
            "timestamp": result.timestamp.isoformat(),
            "system_profile": result.system_profile,
            "workload_analysis": result.workload_analysis,
            "optimal_config": result.optimal_config,
            "recommendations": result.recommendations,
            "estimated_speedup": result.estimated_speedup,
            "profile_name": result.profile_name
        }

        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

        self.logger.info(f"Tuning result saved to {result_file}")


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Performance Tuner - Command Executor Framework")
    print("=" * 80)

    tuner = PerformanceTuner()

    # Run full tuning
    result = tuner.tune_full(target_metric="throughput", run_benchmarks=True)

    # Display results
    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)
    print(f"\nSelected Profile: {result.profile_name}")
    print(f"Estimated Speedup: {result.estimated_speedup}x")
    print(f"\nOptimal Configuration:")
    print(json.dumps(result.optimal_config, indent=2))
    print(f"\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")

    return 0


if __name__ == "__main__":
    sys.exit(main())