"""
Monitoring and Metrics System

Comprehensive monitoring for MCP integration system with metrics collection,
alerting, and performance tracking.

Features:
- Real-time metrics collection
- Performance monitoring
- Alert system
- Cost tracking
- Dashboard data export

Example:
    >>> monitor = await Monitor.create()
    >>>
    >>> # Track MCP call
    >>> with monitor.track_mcp_call("context7", "library_api"):
    ...     result = await context7.fetch(...)
    >>>
    >>> # Get metrics
    >>> metrics = monitor.get_metrics()
    >>> # {'avg_latency_ms': 145, 'error_rate': 0.02, 'cost_usd': 1.25}
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from contextlib import asynccontextmanager
import json


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """
    Single metric measurement.

    Attributes:
        name: Metric name
        value: Metric value
        timestamp: Measurement timestamp
        tags: Additional tags for filtering
        unit: Measurement unit
    """
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Alert:
    """
    System alert.

    Attributes:
        severity: Alert severity
        message: Alert message
        timestamp: Alert timestamp
        component: Component that triggered alert
        metric: Related metric name
        threshold: Threshold that was breached
        current_value: Current metric value
    """
    severity: AlertSeverity
    message: str
    timestamp: float
    component: str
    metric: str = ""
    threshold: Optional[float] = None
    current_value: Optional[float] = None


class Monitor:
    """
    Monitoring and metrics system for MCP integration.

    Tracks performance, errors, costs, and provides alerting.

    Features:
    - Real-time metrics collection
    - Performance tracking (latency, throughput)
    - Error rate monitoring
    - Cost tracking
    - Alert system
    - Dashboard export
    """

    def __init__(
        self,
        enable_alerts: bool = True,
        metrics_retention_minutes: int = 60,
        alert_cooldown_seconds: int = 300,
    ):
        """
        Initialize monitor.

        Args:
            enable_alerts: Enable alert system
            metrics_retention_minutes: How long to retain metrics
            alert_cooldown_seconds: Cooldown between duplicate alerts
        """
        self.enable_alerts = enable_alerts
        self.metrics_retention_minutes = metrics_retention_minutes
        self.alert_cooldown_seconds = alert_cooldown_seconds

        # Metrics storage (time-series data)
        self.metrics: deque = deque(maxlen=10000)

        # Aggregated metrics
        self.aggregated: Dict[str, Any] = defaultdict(lambda: {
            'count': 0,
            'sum': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'avg': 0.0,
        })

        # MCP-specific metrics
        self.mcp_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'calls': 0,
            'errors': 0,
            'total_latency_ms': 0,
            'avg_latency_ms': 0,
            'p95_latency_ms': 0,
            'cost_usd': 0.0,
        })

        # Alerts
        self.alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.last_alert_time: Dict[str, float] = {}

        # Alert thresholds
        self.thresholds = {
            'latency_ms': {'warning': 500, 'error': 1000},
            'error_rate': {'warning': 0.05, 'error': 0.1},
            'cache_hit_rate': {'warning': 0.7, 'error': 0.5},
            'mcp_load_time_ms': {'warning': 300, 'error': 500},
        }

        # Cost tracking (per MCP)
        self.cost_per_call = {
            'context7': 0.001,  # $0.001 per call
            'github': 0.0005,
            'playwright': 0.002,
            'memory-bank': 0.0001,
            'serena': 0.0,  # Local, no cost
        }

        # Background cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None

    @classmethod
    async def create(cls, **kwargs) -> "Monitor":
        """
        Create monitor and start background tasks.

        Args:
            **kwargs: Monitor configuration

        Returns:
            Initialized Monitor instance
        """
        monitor = cls(**kwargs)
        await monitor.start()
        return monitor

    async def start(self) -> None:
        """Start background monitoring tasks."""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background monitoring tasks."""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    @asynccontextmanager
    async def track_mcp_call(
        self,
        mcp_name: str,
        operation: str = "fetch"
    ):
        """
        Context manager to track MCP call.

        Args:
            mcp_name: MCP name
            operation: Operation type

        Example:
            >>> async with monitor.track_mcp_call("context7", "library_api"):
            ...     result = await context7.fetch(...)
        """
        start_time = time.time()
        error_occurred = False

        try:
            yield
        except Exception as e:
            error_occurred = True
            self.record_error(mcp_name, str(e))
            raise
        finally:
            latency_ms = int((time.time() - start_time) * 1000)

            # Record metrics
            self.record_mcp_call(
                mcp_name=mcp_name,
                latency_ms=latency_ms,
                operation=operation,
                error=error_occurred
            )

    def record_mcp_call(
        self,
        mcp_name: str,
        latency_ms: int,
        operation: str = "fetch",
        error: bool = False
    ) -> None:
        """
        Record MCP call metrics.

        Args:
            mcp_name: MCP name
            latency_ms: Call latency in milliseconds
            operation: Operation type
            error: Whether an error occurred
        """
        # Record metric
        self.record_metric(
            name=f"mcp.{mcp_name}.latency",
            value=latency_ms,
            tags={'operation': operation, 'mcp': mcp_name},
            unit="ms"
        )

        # Update MCP-specific metrics
        stats = self.mcp_metrics[mcp_name]
        stats['calls'] += 1
        stats['total_latency_ms'] += latency_ms
        stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['calls']

        if error:
            stats['errors'] += 1
            self.record_metric(
                name=f"mcp.{mcp_name}.error",
                value=1,
                tags={'operation': operation, 'mcp': mcp_name}
            )

        # Calculate cost
        cost = self.cost_per_call.get(mcp_name, 0.0)
        stats['cost_usd'] += cost
        self.record_metric(
            name=f"mcp.{mcp_name}.cost",
            value=cost,
            tags={'mcp': mcp_name},
            unit="usd"
        )

        # Check thresholds
        self._check_mcp_thresholds(mcp_name, latency_ms, stats)

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: str = ""
    ) -> None:
        """
        Record a metric.

        Args:
            name: Metric name
            value: Metric value
            tags: Additional tags
            unit: Measurement unit
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit
        )

        self.metrics.append(metric)

        # Update aggregated metrics
        agg = self.aggregated[name]
        agg['count'] += 1
        agg['sum'] += value
        agg['min'] = min(agg['min'], value)
        agg['max'] = max(agg['max'], value)
        agg['avg'] = agg['sum'] / agg['count']

    def record_error(
        self,
        component: str,
        error_message: str,
        severity: AlertSeverity = AlertSeverity.ERROR
    ) -> None:
        """
        Record an error.

        Args:
            component: Component where error occurred
            error_message: Error message
            severity: Error severity
        """
        self.record_metric(
            name=f"{component}.error",
            value=1,
            tags={'component': component}
        )

        # Create alert
        if self.enable_alerts:
            self.create_alert(
                severity=severity,
                message=f"{component} error: {error_message}",
                component=component
            )

    def create_alert(
        self,
        severity: AlertSeverity,
        message: str,
        component: str,
        metric: str = "",
        threshold: Optional[float] = None,
        current_value: Optional[float] = None
    ) -> None:
        """
        Create an alert.

        Args:
            severity: Alert severity
            message: Alert message
            component: Component that triggered alert
            metric: Related metric
            threshold: Threshold breached
            current_value: Current value
        """
        # Check cooldown
        alert_key = f"{component}:{metric}:{severity.value}"
        last_time = self.last_alert_time.get(alert_key, 0)

        if time.time() - last_time < self.alert_cooldown_seconds:
            return  # Skip duplicate alert

        alert = Alert(
            severity=severity,
            message=message,
            timestamp=time.time(),
            component=component,
            metric=metric,
            threshold=threshold,
            current_value=current_value
        )

        self.alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_time[alert_key] = time.time()

        # Log alert
        print(f"[{severity.value.upper()}] {message}")

    def _check_mcp_thresholds(
        self,
        mcp_name: str,
        latency_ms: int,
        stats: Dict[str, Any]
    ) -> None:
        """Check MCP metrics against thresholds."""
        if not self.enable_alerts:
            return

        # Check latency
        if latency_ms > self.thresholds['latency_ms']['error']:
            self.create_alert(
                severity=AlertSeverity.ERROR,
                message=f"High latency for {mcp_name}: {latency_ms}ms",
                component=mcp_name,
                metric='latency_ms',
                threshold=self.thresholds['latency_ms']['error'],
                current_value=latency_ms
            )
        elif latency_ms > self.thresholds['latency_ms']['warning']:
            self.create_alert(
                severity=AlertSeverity.WARNING,
                message=f"Elevated latency for {mcp_name}: {latency_ms}ms",
                component=mcp_name,
                metric='latency_ms',
                threshold=self.thresholds['latency_ms']['warning'],
                current_value=latency_ms
            )

        # Check error rate
        if stats['calls'] > 10:  # Need minimum calls for meaningful rate
            error_rate = stats['errors'] / stats['calls']
            if error_rate > self.thresholds['error_rate']['error']:
                self.create_alert(
                    severity=AlertSeverity.ERROR,
                    message=f"High error rate for {mcp_name}: {error_rate:.1%}",
                    component=mcp_name,
                    metric='error_rate',
                    threshold=self.thresholds['error_rate']['error'],
                    current_value=error_rate
                )
            elif error_rate > self.thresholds['error_rate']['warning']:
                self.create_alert(
                    severity=AlertSeverity.WARNING,
                    message=f"Elevated error rate for {mcp_name}: {error_rate:.1%}",
                    component=mcp_name,
                    metric='error_rate',
                    threshold=self.thresholds['error_rate']['warning'],
                    current_value=error_rate
                )

    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        time_window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get metrics.

        Args:
            metric_name: Specific metric name or None for all
            time_window_minutes: Time window for metrics

        Returns:
            Metrics dictionary
        """
        if metric_name:
            return self.aggregated.get(metric_name, {})

        # Return all metrics summary
        cutoff_time = (
            time.time() - (time_window_minutes * 60)
            if time_window_minutes else 0
        )

        recent_metrics = [
            m for m in self.metrics
            if m.timestamp >= cutoff_time
        ] if time_window_minutes else list(self.metrics)

        return {
            'total_metrics': len(recent_metrics),
            'aggregated': dict(self.aggregated),
            'mcp_metrics': dict(self.mcp_metrics),
        }

    def get_mcp_metrics(self, mcp_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get MCP-specific metrics.

        Args:
            mcp_name: Specific MCP or None for all

        Returns:
            MCP metrics
        """
        if mcp_name:
            return self.mcp_metrics.get(mcp_name, {})

        return dict(self.mcp_metrics)

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        component: Optional[str] = None
    ) -> List[Alert]:
        """
        Get alerts.

        Args:
            severity: Filter by severity
            component: Filter by component

        Returns:
            List of alerts
        """
        alerts = self.alerts.copy()

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if component:
            alerts = [a for a in alerts if a.component == component]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get dashboard data for visualization.

        Returns:
            Dashboard data with all metrics
        """
        # Calculate overall stats
        total_calls = sum(m['calls'] for m in self.mcp_metrics.values())
        total_errors = sum(m['errors'] for m in self.mcp_metrics.values())
        total_cost = sum(m['cost_usd'] for m in self.mcp_metrics.values())

        error_rate = total_errors / total_calls if total_calls > 0 else 0.0

        # MCP breakdown
        mcp_breakdown = [
            {
                'name': mcp_name,
                'calls': stats['calls'],
                'avg_latency_ms': stats['avg_latency_ms'],
                'error_rate': stats['errors'] / stats['calls'] if stats['calls'] > 0 else 0,
                'cost_usd': stats['cost_usd'],
            }
            for mcp_name, stats in self.mcp_metrics.items()
        ]

        # Recent alerts
        recent_alerts = [
            {
                'severity': a.severity.value,
                'message': a.message,
                'timestamp': a.timestamp,
                'component': a.component,
            }
            for a in self.get_alerts()[:10]  # Last 10 alerts
        ]

        return {
            'summary': {
                'total_calls': total_calls,
                'total_errors': total_errors,
                'error_rate': error_rate,
                'total_cost_usd': total_cost,
                'active_alerts': len(self.alerts),
            },
            'mcp_breakdown': mcp_breakdown,
            'recent_alerts': recent_alerts,
            'timestamp': time.time(),
        }

    def export_metrics(
        self,
        filepath: str,
        format: str = "json"
    ) -> None:
        """
        Export metrics to file.

        Args:
            filepath: Output file path
            format: Export format ('json', 'csv')
        """
        data = self.get_dashboard_data()

        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "csv":
            # CSV export (simplified)
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['MCP', 'Calls', 'Avg Latency (ms)', 'Error Rate', 'Cost (USD)'])
                for mcp in data['mcp_breakdown']:
                    writer.writerow([
                        mcp['name'],
                        mcp['calls'],
                        mcp['avg_latency_ms'],
                        mcp['error_rate'],
                        mcp['cost_usd'],
                    ])

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                # Clean old metrics
                cutoff_time = time.time() - (self.metrics_retention_minutes * 60)
                while self.metrics and self.metrics[0].timestamp < cutoff_time:
                    self.metrics.popleft()

                # Clear resolved alerts (older than 1 hour)
                alert_cutoff = time.time() - 3600
                self.alerts = [
                    a for a in self.alerts
                    if a.timestamp >= alert_cutoff
                ]

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cleanup error: {e}")

    def clear_alerts(self) -> None:
        """Clear all active alerts."""
        self.alerts.clear()

    def reset(self) -> None:
        """Reset all metrics and alerts."""
        self.metrics.clear()
        self.aggregated.clear()
        self.mcp_metrics.clear()
        self.alerts.clear()
        self.alert_history.clear()
        self.last_alert_time.clear()
