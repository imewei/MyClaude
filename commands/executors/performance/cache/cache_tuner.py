#!/usr/bin/env python3
"""
Cache Tuner
===========

Optimizes cache sizes and policies for maximum hit rates.

Features:
- Optimal cache size calculation
- Hit rate prediction
- Eviction policy optimization
- Memory-aware sizing

Author: Claude Code Framework
Version: 2.0
"""

import logging
import psutil
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import statistics


@dataclass
class CacheConfig:
    """Cache configuration"""
    l1_size_mb: int
    l2_size_mb: int
    l3_size_mb: int
    eviction_policy: str  # lru, lfu, arc
    target_hit_rate: float
    ttl_hours: int

    def total_size_mb(self) -> int:
        """Get total cache size"""
        return self.l1_size_mb + self.l2_size_mb + self.l3_size_mb


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    misses: int = 0
    evictions: int = 0
    avg_object_size_kb: float = 0.0

    def total_hits(self) -> int:
        """Total cache hits"""
        return self.l1_hits + self.l2_hits + self.l3_hits

    def total_requests(self) -> int:
        """Total cache requests"""
        return self.total_hits() + self.misses

    def hit_rate(self) -> float:
        """Cache hit rate percentage"""
        total = self.total_requests()
        return (self.total_hits() / total * 100) if total > 0 else 0.0


class CacheTuner:
    """
    Optimizes cache configuration for performance.

    Uses workload analysis and system resources to calculate
    optimal cache sizes and policies.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def tune(
        self,
        current_metrics: Optional[CacheMetrics] = None,
        target_hit_rate: float = 70.0,
        available_memory_gb: Optional[float] = None
    ) -> CacheConfig:
        """
        Tune cache configuration.

        Args:
            current_metrics: Current cache metrics
            target_hit_rate: Target hit rate percentage
            available_memory_gb: Available memory for cache

        Returns:
            Optimized cache configuration
        """
        self.logger.info(f"Tuning cache for {target_hit_rate}% hit rate")

        # Get available memory
        if available_memory_gb is None:
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)

        # Calculate optimal sizes
        l1, l2, l3 = self._calculate_optimal_sizes(
            available_memory_gb,
            current_metrics
        )

        # Select eviction policy
        policy = self._select_eviction_policy(current_metrics)

        # Calculate TTL
        ttl = self._calculate_optimal_ttl(current_metrics)

        config = CacheConfig(
            l1_size_mb=l1,
            l2_size_mb=l2,
            l3_size_mb=l3,
            eviction_policy=policy,
            target_hit_rate=target_hit_rate,
            ttl_hours=ttl
        )

        self.logger.info(
            f"Optimal cache config: L1={l1}MB, L2={l2}MB, L3={l3}MB, "
            f"policy={policy}, TTL={ttl}h"
        )

        return config

    def _calculate_optimal_sizes(
        self,
        available_memory_gb: float,
        metrics: Optional[CacheMetrics]
    ) -> Tuple[int, int, int]:
        """Calculate optimal cache sizes"""

        # Allocate 20-40% of available memory to cache
        total_cache_mb = int(available_memory_gb * 1024 * 0.3)

        # Distribute across levels (L1:L2:L3 = 1:4:10 ratio)
        l1_size = max(50, int(total_cache_mb * 0.1))
        l2_size = max(200, int(total_cache_mb * 0.3))
        l3_size = max(500, int(total_cache_mb * 0.6))

        # Adjust based on metrics
        if metrics:
            # If L1 hit rate is high, increase L1
            if metrics.l1_hits > metrics.l2_hits + metrics.l3_hits:
                l1_size = int(l1_size * 1.5)
                l2_size = int(l2_size * 0.9)

            # If L2 hit rate is high, increase L2
            elif metrics.l2_hits > metrics.l1_hits + metrics.l3_hits:
                l2_size = int(l2_size * 1.3)
                l3_size = int(l3_size * 0.9)

        # Ensure minimum sizes
        l1_size = max(50, min(l1_size, 200))
        l2_size = max(200, min(l2_size, 1000))
        l3_size = max(500, min(l3_size, 5000))

        return l1_size, l2_size, l3_size

    def _select_eviction_policy(
        self,
        metrics: Optional[CacheMetrics]
    ) -> str:
        """Select optimal eviction policy"""

        # Default to LRU (works well for most workloads)
        policy = "lru"

        if metrics and metrics.total_requests() > 1000:
            # If many evictions, consider ARC (adaptive)
            if metrics.evictions > metrics.total_requests() * 0.3:
                policy = "arc"

        return policy

    def _calculate_optimal_ttl(
        self,
        metrics: Optional[CacheMetrics]
    ) -> int:
        """Calculate optimal TTL in hours"""

        # Default TTL
        ttl = 24

        if metrics:
            # If high hit rate, can use longer TTL
            if metrics.hit_rate() > 80:
                ttl = 48
            # If low hit rate, use shorter TTL
            elif metrics.hit_rate() < 40:
                ttl = 6

        return ttl

    def estimate_hit_rate(
        self,
        config: CacheConfig,
        working_set_size_mb: int
    ) -> float:
        """
        Estimate cache hit rate for a configuration.

        Args:
            config: Cache configuration
            working_set_size_mb: Working set size

        Returns:
            Estimated hit rate percentage
        """
        total_cache = config.total_size_mb()

        # Simple model: hit rate proportional to cache/working set ratio
        ratio = total_cache / working_set_size_mb if working_set_size_mb > 0 else 1.0

        # Logarithmic relationship
        if ratio >= 1.0:
            hit_rate = 95.0  # Nearly all hits if cache >= working set
        elif ratio >= 0.5:
            hit_rate = 80.0 + (ratio - 0.5) * 30.0
        elif ratio >= 0.2:
            hit_rate = 60.0 + (ratio - 0.2) * 66.7
        else:
            hit_rate = ratio * 300.0  # Linear for small ratios

        return min(hit_rate, 95.0)  # Cap at 95%

    def recommend_adjustments(
        self,
        current_config: CacheConfig,
        current_metrics: CacheMetrics
    ) -> Dict[str, Any]:
        """
        Recommend cache adjustments.

        Args:
            current_config: Current configuration
            current_metrics: Current metrics

        Returns:
            Recommendations dictionary
        """
        recommendations = []

        current_hit_rate = current_metrics.hit_rate()
        target = current_config.target_hit_rate

        # Hit rate recommendations
        if current_hit_rate < target - 10:
            recommendations.append({
                "type": "increase_size",
                "reason": f"Hit rate {current_hit_rate:.1f}% below target {target}%",
                "action": "Increase cache size by 50%"
            })

        elif current_hit_rate > target + 20:
            recommendations.append({
                "type": "decrease_size",
                "reason": f"Hit rate {current_hit_rate:.1f}% exceeds target by 20%",
                "action": "Can reduce cache size by 30% to free memory"
            })

        # Eviction recommendations
        if current_metrics.evictions > current_metrics.total_requests() * 0.5:
            recommendations.append({
                "type": "high_evictions",
                "reason": "High eviction rate indicates cache thrashing",
                "action": "Increase cache size or adjust eviction policy"
            })

        # Level balance recommendations
        total_hits = current_metrics.total_hits()
        if total_hits > 0:
            l1_ratio = current_metrics.l1_hits / total_hits
            l2_ratio = current_metrics.l2_hits / total_hits

            if l1_ratio < 0.5 and current_config.l1_size_mb < 200:
                recommendations.append({
                    "type": "balance_levels",
                    "reason": "L1 hit ratio is low",
                    "action": "Increase L1 cache size"
                })

        return {
            "current_hit_rate": current_hit_rate,
            "target_hit_rate": target,
            "gap": target - current_hit_rate,
            "recommendations": recommendations
        }


def main():
    """Test cache tuner"""
    logging.basicConfig(level=logging.INFO)

    tuner = CacheTuner()

    # Test with sample metrics
    metrics = CacheMetrics(
        l1_hits=500,
        l2_hits=300,
        l3_hits=100,
        misses=200,
        evictions=50,
        avg_object_size_kb=10.0
    )

    print(f"Current hit rate: {metrics.hit_rate():.1f}%")

    # Tune
    config = tuner.tune(current_metrics=metrics, target_hit_rate=75.0)

    print(f"\nOptimal Configuration:")
    print(f"  L1: {config.l1_size_mb}MB")
    print(f"  L2: {config.l2_size_mb}MB")
    print(f"  L3: {config.l3_size_mb}MB")
    print(f"  Total: {config.total_size_mb()}MB")
    print(f"  Policy: {config.eviction_policy}")
    print(f"  TTL: {config.ttl_hours}h")

    # Get recommendations
    recommendations = tuner.recommend_adjustments(config, metrics)
    print(f"\nRecommendations:")
    for rec in recommendations["recommendations"]:
        print(f"  - {rec['action']}")
        print(f"    Reason: {rec['reason']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())