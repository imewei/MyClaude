#!/usr/bin/env python3
"""
Performance Monitoring for Python Development Plugin

Tracks and analyzes:
1. Agent response times
2. Cache hit rates
3. Model selection distribution
4. Performance degradation alerts
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import statistics


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    timestamp: float
    agent: str
    model: str
    query_complexity: str
    response_time_ms: float
    cache_hit: bool
    token_count: Optional[int] = None


class PerformanceMonitor:
    """Monitor and track plugin performance metrics"""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize performance monitor

        Args:
            data_dir: Directory for performance data (default: plugin/.cache/performance)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / ".cache" / "performance"

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.data_dir / "metrics.jsonl"
        self.alerts_file = self.data_dir / "alerts.json"
        self.summary_file = self.data_dir / "summary.json"

        # Performance thresholds
        self.thresholds = {
            'haiku_max_ms': 300,      # Max acceptable for haiku
            'sonnet_max_ms': 1000,    # Max acceptable for sonnet
            'cache_hit_min': 0.30,    # Min 30% cache hit rate
            'degradation_pct': 20,     # Alert if 20% slower than baseline
        }

    def log_metric(
        self,
        agent: str,
        model: str,
        query_complexity: str,
        response_time_ms: float,
        cache_hit: bool = False,
        token_count: Optional[int] = None
    ):
        """Log a performance metric"""
        metric = PerformanceMetric(
            timestamp=time.time(),
            agent=agent,
            model=model,
            query_complexity=query_complexity,
            response_time_ms=response_time_ms,
            cache_hit=cache_hit,
            token_count=token_count
        )

        # Append to metrics file (JSONL format)
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metric)) + '\n')

        # Check for performance issues
        self._check_thresholds(metric)

    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric violates performance thresholds"""
        alerts = []

        # Response time checks
        if metric.model == 'haiku' and metric.response_time_ms > self.thresholds['haiku_max_ms']:
            alerts.append({
                'severity': 'WARNING',
                'type': 'slow_response',
                'message': f"Haiku response took {metric.response_time_ms:.0f}ms (threshold: {self.thresholds['haiku_max_ms']}ms)",
                'timestamp': metric.timestamp,
                'agent': metric.agent
            })

        if metric.model == 'sonnet' and metric.response_time_ms > self.thresholds['sonnet_max_ms']:
            alerts.append({
                'severity': 'WARNING',
                'type': 'slow_response',
                'message': f"Sonnet response took {metric.response_time_ms:.0f}ms (threshold: {self.thresholds['sonnet_max_ms']}ms)",
                'timestamp': metric.timestamp,
                'agent': metric.agent
            })

        # Log alerts
        if alerts:
            self._log_alerts(alerts)

    def _log_alerts(self, alerts: List[Dict]):
        """Log performance alerts"""
        existing_alerts = []
        if self.alerts_file.exists():
            with open(self.alerts_file, 'r') as f:
                existing_alerts = json.load(f)

        existing_alerts.extend(alerts)

        # Keep only last 100 alerts
        existing_alerts = existing_alerts[-100:]

        with open(self.alerts_file, 'w') as f:
            json.dump(existing_alerts, f, indent=2)

    def get_metrics(
        self,
        hours: int = 24,
        agent: Optional[str] = None
    ) -> List[PerformanceMetric]:
        """
        Get metrics from last N hours

        Args:
            hours: Number of hours to look back
            agent: Filter by agent name (optional)

        Returns:
            List of performance metrics
        """
        if not self.metrics_file.exists():
            return []

        cutoff_time = time.time() - (hours * 3600)
        metrics = []

        with open(self.metrics_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                metric = PerformanceMetric(**data)

                if metric.timestamp < cutoff_time:
                    continue

                if agent and metric.agent != agent:
                    continue

                metrics.append(metric)

        return metrics

    def analyze_performance(self, hours: int = 24) -> Dict:
        """
        Analyze performance over time window

        Returns:
            Dictionary with performance analysis
        """
        metrics = self.get_metrics(hours=hours)

        if not metrics:
            return {
                'status': 'NO_DATA',
                'message': f'No metrics found in last {hours} hours'
            }

        # Calculate statistics by model
        by_model = {}
        for model in ['haiku', 'sonnet']:
            model_metrics = [m for m in metrics if m.model == model]
            if model_metrics:
                response_times = [m.response_time_ms for m in model_metrics]
                by_model[model] = {
                    'count': len(model_metrics),
                    'avg_ms': statistics.mean(response_times),
                    'median_ms': statistics.median(response_times),
                    'p95_ms': self._percentile(response_times, 0.95),
                    'p99_ms': self._percentile(response_times, 0.99),
                    'min_ms': min(response_times),
                    'max_ms': max(response_times)
                }

        # Cache statistics
        cache_hits = sum(1 for m in metrics if m.cache_hit)
        cache_hit_rate = cache_hits / len(metrics) if metrics else 0

        # Agent distribution
        agent_counts = {}
        for metric in metrics:
            agent_counts[metric.agent] = agent_counts.get(metric.agent, 0) + 1

        # Complexity distribution
        complexity_counts = {}
        for metric in metrics:
            complexity_counts[metric.query_complexity] = \
                complexity_counts.get(metric.query_complexity, 0) + 1

        return {
            'status': 'OK',
            'time_window_hours': hours,
            'total_queries': len(metrics),
            'by_model': by_model,
            'cache': {
                'hits': cache_hits,
                'total': len(metrics),
                'hit_rate': cache_hit_rate,
                'status': 'GOOD' if cache_hit_rate >= self.thresholds['cache_hit_min'] else 'LOW'
            },
            'by_agent': agent_counts,
            'by_complexity': complexity_counts,
            'timestamp': datetime.now().isoformat()
        }

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        if not self.alerts_file.exists():
            return []

        with open(self.alerts_file, 'r') as f:
            all_alerts = json.load(f)

        cutoff_time = time.time() - (hours * 3600)
        return [
            alert for alert in all_alerts
            if alert['timestamp'] > cutoff_time
        ]

    def generate_summary(self) -> Dict:
        """Generate performance summary"""
        # Last 24 hours
        analysis_24h = self.analyze_performance(hours=24)

        # Last 7 days
        analysis_7d = self.analyze_performance(hours=24 * 7)

        # Recent alerts
        recent_alerts = self.get_alerts(hours=24)

        summary = {
            'generated_at': datetime.now().isoformat(),
            'last_24_hours': analysis_24h,
            'last_7_days': analysis_7d,
            'recent_alerts': {
                'count': len(recent_alerts),
                'by_severity': self._count_by_severity(recent_alerts)
            },
            'health_status': self._calculate_health_status(analysis_24h, recent_alerts)
        }

        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def _count_by_severity(self, alerts: List[Dict]) -> Dict[str, int]:
        """Count alerts by severity"""
        counts = {'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}
        for alert in alerts:
            severity = alert.get('severity', 'WARNING')
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    def _calculate_health_status(self, analysis: Dict, alerts: List[Dict]) -> str:
        """Calculate overall health status"""
        if analysis['status'] == 'NO_DATA':
            return 'UNKNOWN'

        issues = []

        # Check cache hit rate
        if analysis['cache']['status'] == 'LOW':
            issues.append('Low cache hit rate')

        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.get('severity') == 'CRITICAL']
        if critical_alerts:
            issues.append(f'{len(critical_alerts)} critical alerts')

        # Check response times
        for model, stats in analysis.get('by_model', {}).items():
            threshold = self.thresholds[f'{model}_max_ms']
            if stats['p95_ms'] > threshold * 1.2:  # 20% over threshold
                issues.append(f'{model} p95 latency high')

        if not issues:
            return 'HEALTHY'
        elif len(issues) == 1:
            return 'DEGRADED'
        else:
            return 'UNHEALTHY'


def demo():
    """Demonstration of performance monitoring"""
    print("=" * 80)
    print("Performance Monitoring Demo")
    print("=" * 80)

    monitor = PerformanceMonitor()

    # Simulate some metrics
    print("\n--- Simulating Performance Metrics ---")
    scenarios = [
        ("fastapi-pro", "haiku", "simple", 180, False),
        ("fastapi-pro", "haiku", "simple", 150, True),   # Cache hit
        ("python-pro", "sonnet", "complex", 750, False),
        ("django-pro", "haiku", "simple", 200, False),
        ("python-pro", "sonnet", "medium", 650, False),
        ("fastapi-pro", "haiku", "simple", 140, True),   # Cache hit
        ("python-pro", "sonnet", "complex", 850, False),
        ("django-pro", "haiku", "simple", 190, True),    # Cache hit
    ]

    for agent, model, complexity, response_time, cache_hit in scenarios:
        monitor.log_metric(agent, model, complexity, response_time, cache_hit)
        status = "CACHE HIT" if cache_hit else "MISS"
        print(f"✓ Logged: {agent} ({model}) - {response_time}ms [{status}]")

    # Analyze performance
    print("\n--- Performance Analysis (24h) ---")
    analysis = monitor.analyze_performance(hours=24)

    print(f"Total Queries: {analysis['total_queries']}")
    print(f"Cache Hit Rate: {analysis['cache']['hit_rate']:.1%} ({analysis['cache']['status']})")

    print("\n--- By Model ---")
    for model, stats in analysis['by_model'].items():
        print(f"\n{model.upper()}:")
        print(f"  Count: {stats['count']}")
        print(f"  Avg: {stats['avg_ms']:.0f}ms")
        print(f"  Median: {stats['median_ms']:.0f}ms")
        print(f"  P95: {stats['p95_ms']:.0f}ms")
        print(f"  P99: {stats['p99_ms']:.0f}ms")

    print("\n--- By Agent ---")
    for agent, count in analysis['by_agent'].items():
        print(f"  {agent}: {count} queries")

    print("\n--- By Complexity ---")
    for complexity, count in analysis['by_complexity'].items():
        print(f"  {complexity}: {count} queries")

    # Generate summary
    print("\n--- Generating Summary ---")
    summary = monitor.generate_summary()
    print(f"Health Status: {summary['health_status']}")
    print(f"Recent Alerts: {summary['recent_alerts']['count']}")

    print("\n✓ Demo complete")
    print(f"\nPerformance data saved to: {monitor.data_dir}")


if __name__ == "__main__":
    demo()
