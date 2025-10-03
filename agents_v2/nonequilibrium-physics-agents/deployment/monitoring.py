"""Health Monitoring and Metrics Collection.

Provides monitoring infrastructure for production deployments:
- System metrics (CPU, memory, GPU)
- Application metrics (request rates, latency, solver performance)
- Health checks and readiness probes
- Log aggregation
- Alert management

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import psutil
import json
from pathlib import Path
from collections import defaultdict, deque
import logging

# Optional GPU monitoring
try:
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class HealthStatus:
    """Health check status."""
    healthy: bool
    checks: Dict[str, bool]
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """Alert configuration and state."""
    name: str
    condition: Callable[[float], bool]
    message: str
    severity: str = "warning"  # 'info', 'warning', 'error', 'critical'
    cooldown: timedelta = timedelta(minutes=5)
    last_triggered: Optional[datetime] = None
    triggered_count: int = 0


class MetricsCollector:
    """Collect and store metrics."""

    def __init__(self, retention_period: timedelta = timedelta(hours=24)):
        """Initialize metrics collector.

        Args:
            retention_period: How long to retain metrics
        """
        self.retention_period = retention_period
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.start_time = datetime.now()

        # Initialize GPU monitoring if available
        if GPU_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"Initialized GPU monitoring for {self.gpu_count} GPUs")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.gpu_count = 0
        else:
            self.gpu_count = 0

    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None,
               unit: str = "") -> None:
        """Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for the metric
            unit: Unit of measurement
        """
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            unit=unit
        )
        self.metrics[name].append(metric)

    def get_latest(self, name: str) -> Optional[MetricPoint]:
        """Get latest value for metric.

        Args:
            name: Metric name

        Returns:
            Latest metric point or None
        """
        if name not in self.metrics or not self.metrics[name]:
            return None
        return self.metrics[name][-1]

    def get_range(self, name: str, start: datetime, end: datetime) -> List[MetricPoint]:
        """Get metrics within time range.

        Args:
            name: Metric name
            start: Start time
            end: End time

        Returns:
            List of metric points in range
        """
        if name not in self.metrics:
            return []

        return [
            m for m in self.metrics[name]
            if start <= m.timestamp <= end
        ]

    def get_statistics(self, name: str, window: timedelta = timedelta(minutes=5)) -> Dict[str, float]:
        """Get statistics for metric over time window.

        Args:
            name: Metric name
            window: Time window

        Returns:
            Dictionary with min, max, mean, p50, p95, p99
        """
        end = datetime.now()
        start = end - window
        points = self.get_range(name, start, end)

        if not points:
            return {}

        values = sorted([p.value for p in points])
        n = len(values)

        return {
            "min": values[0],
            "max": values[-1],
            "mean": sum(values) / n,
            "p50": values[int(n * 0.5)],
            "p95": values[int(n * 0.95)] if n > 1 else values[0],
            "p99": values[int(n * 0.99)] if n > 1 else values[0],
            "count": n
        }

    def cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff = datetime.now() - self.retention_period

        for name in self.metrics:
            while self.metrics[name] and self.metrics[name][0].timestamp < cutoff:
                self.metrics[name].popleft()


class SystemMonitor:
    """Monitor system resources."""

    def __init__(self, collector: MetricsCollector):
        """Initialize system monitor.

        Args:
            collector: Metrics collector
        """
        self.collector = collector

    def collect_cpu_metrics(self) -> None:
        """Collect CPU metrics."""
        # Overall CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.collector.record("system.cpu.usage_percent", cpu_percent, unit="%")

        # Per-core usage
        per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        for i, usage in enumerate(per_cpu):
            self.collector.record(
                "system.cpu.core_usage_percent",
                usage,
                labels={"core": str(i)},
                unit="%"
            )

        # Load average
        load_avg = psutil.getloadavg()
        self.collector.record("system.cpu.load_avg_1m", load_avg[0])
        self.collector.record("system.cpu.load_avg_5m", load_avg[1])
        self.collector.record("system.cpu.load_avg_15m", load_avg[2])

    def collect_memory_metrics(self) -> None:
        """Collect memory metrics."""
        mem = psutil.virtual_memory()

        self.collector.record("system.memory.total_bytes", mem.total, unit="bytes")
        self.collector.record("system.memory.available_bytes", mem.available, unit="bytes")
        self.collector.record("system.memory.used_bytes", mem.used, unit="bytes")
        self.collector.record("system.memory.usage_percent", mem.percent, unit="%")

        # Swap memory
        swap = psutil.swap_memory()
        self.collector.record("system.swap.total_bytes", swap.total, unit="bytes")
        self.collector.record("system.swap.used_bytes", swap.used, unit="bytes")
        self.collector.record("system.swap.usage_percent", swap.percent, unit="%")

    def collect_disk_metrics(self) -> None:
        """Collect disk metrics."""
        disk = psutil.disk_usage('/')

        self.collector.record("system.disk.total_bytes", disk.total, unit="bytes")
        self.collector.record("system.disk.used_bytes", disk.used, unit="bytes")
        self.collector.record("system.disk.free_bytes", disk.free, unit="bytes")
        self.collector.record("system.disk.usage_percent", disk.percent, unit="%")

        # Disk I/O
        io = psutil.disk_io_counters()
        if io:
            self.collector.record("system.disk.read_bytes", io.read_bytes, unit="bytes")
            self.collector.record("system.disk.write_bytes", io.write_bytes, unit="bytes")
            self.collector.record("system.disk.read_count", io.read_count)
            self.collector.record("system.disk.write_count", io.write_count)

    def collect_network_metrics(self) -> None:
        """Collect network metrics."""
        net = psutil.net_io_counters()

        self.collector.record("system.network.bytes_sent", net.bytes_sent, unit="bytes")
        self.collector.record("system.network.bytes_recv", net.bytes_recv, unit="bytes")
        self.collector.record("system.network.packets_sent", net.packets_sent)
        self.collector.record("system.network.packets_recv", net.packets_recv)
        self.collector.record("system.network.errin", net.errin)
        self.collector.record("system.network.errout", net.errout)

    def collect_gpu_metrics(self) -> None:
        """Collect GPU metrics."""
        if not GPU_AVAILABLE or self.collector.gpu_count == 0:
            return

        try:
            for i in range(self.collector.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.collector.record(
                    "system.gpu.usage_percent",
                    util.gpu,
                    labels={"gpu": str(i)},
                    unit="%"
                )
                self.collector.record(
                    "system.gpu.memory_usage_percent",
                    util.memory,
                    labels={"gpu": str(i)},
                    unit="%"
                )

                # Memory info
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.collector.record(
                    "system.gpu.memory_total_bytes",
                    mem.total,
                    labels={"gpu": str(i)},
                    unit="bytes"
                )
                self.collector.record(
                    "system.gpu.memory_used_bytes",
                    mem.used,
                    labels={"gpu": str(i)},
                    unit="bytes"
                )

                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                self.collector.record(
                    "system.gpu.temperature_celsius",
                    temp,
                    labels={"gpu": str(i)},
                    unit="°C"
                )

                # Power usage
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                self.collector.record(
                    "system.gpu.power_watts",
                    power,
                    labels={"gpu": str(i)},
                    unit="W"
                )
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")

    def collect_all(self) -> None:
        """Collect all system metrics."""
        self.collect_cpu_metrics()
        self.collect_memory_metrics()
        self.collect_disk_metrics()
        self.collect_network_metrics()
        self.collect_gpu_metrics()


class ApplicationMonitor:
    """Monitor application-specific metrics."""

    def __init__(self, collector: MetricsCollector):
        """Initialize application monitor.

        Args:
            collector: Metrics collector
        """
        self.collector = collector
        self.request_start_times: Dict[str, float] = {}

    def record_request_start(self, request_id: str) -> None:
        """Record start of request.

        Args:
            request_id: Request identifier
        """
        self.request_start_times[request_id] = time.time()

    def record_request_end(self, request_id: str, endpoint: str, status_code: int) -> None:
        """Record end of request.

        Args:
            request_id: Request identifier
            endpoint: API endpoint
            status_code: HTTP status code
        """
        if request_id not in self.request_start_times:
            return

        duration = time.time() - self.request_start_times[request_id]
        del self.request_start_times[request_id]

        labels = {
            "endpoint": endpoint,
            "status_code": str(status_code)
        }

        self.collector.record("app.request.duration_seconds", duration, labels=labels, unit="s")
        self.collector.record("app.request.count", 1, labels=labels)

    def record_solver_execution(self, solver_type: str, duration: float,
                               success: bool, iterations: Optional[int] = None) -> None:
        """Record solver execution metrics.

        Args:
            solver_type: Type of solver
            duration: Execution duration in seconds
            success: Whether solver succeeded
            iterations: Number of iterations (if applicable)
        """
        labels = {
            "solver_type": solver_type,
            "success": str(success)
        }

        self.collector.record("app.solver.duration_seconds", duration, labels=labels, unit="s")
        self.collector.record("app.solver.count", 1, labels=labels)

        if iterations is not None:
            self.collector.record(
                "app.solver.iterations",
                iterations,
                labels={"solver_type": solver_type}
            )

    def record_job_status(self, status: str) -> None:
        """Record job status change.

        Args:
            status: Job status (pending, running, completed, failed)
        """
        self.collector.record(
            "app.job.status_count",
            1,
            labels={"status": status}
        )


class HealthChecker:
    """Perform health checks."""

    def __init__(self, collector: MetricsCollector):
        """Initialize health checker.

        Args:
            collector: Metrics collector
        """
        self.collector = collector
        self.checks: Dict[str, Callable[[], bool]] = {}

    def register_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check.

        Args:
            name: Check name
            check_func: Function that returns True if healthy
        """
        self.checks[name] = check_func

    def check_cpu_usage(self, threshold: float = 90.0) -> bool:
        """Check CPU usage is below threshold.

        Args:
            threshold: CPU usage threshold percentage

        Returns:
            True if CPU usage is acceptable
        """
        cpu_metric = self.collector.get_latest("system.cpu.usage_percent")
        if cpu_metric is None:
            return True  # No data yet, assume healthy
        return cpu_metric.value < threshold

    def check_memory_usage(self, threshold: float = 90.0) -> bool:
        """Check memory usage is below threshold.

        Args:
            threshold: Memory usage threshold percentage

        Returns:
            True if memory usage is acceptable
        """
        mem_metric = self.collector.get_latest("system.memory.usage_percent")
        if mem_metric is None:
            return True
        return mem_metric.value < threshold

    def check_disk_usage(self, threshold: float = 85.0) -> bool:
        """Check disk usage is below threshold.

        Args:
            threshold: Disk usage threshold percentage

        Returns:
            True if disk usage is acceptable
        """
        disk_metric = self.collector.get_latest("system.disk.usage_percent")
        if disk_metric is None:
            return True
        return disk_metric.value < threshold

    def run_all_checks(self) -> HealthStatus:
        """Run all registered health checks.

        Returns:
            Health status with results of all checks
        """
        results = {}
        details = {}

        # Run built-in checks
        results["cpu"] = self.check_cpu_usage()
        results["memory"] = self.check_memory_usage()
        results["disk"] = self.check_disk_usage()

        # Run custom checks
        for name, check_func in self.checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                results[name] = False
                details[name] = f"Check failed: {e}"

        # Add resource details
        cpu_metric = self.collector.get_latest("system.cpu.usage_percent")
        mem_metric = self.collector.get_latest("system.memory.usage_percent")
        disk_metric = self.collector.get_latest("system.disk.usage_percent")

        if cpu_metric:
            details["cpu_usage"] = f"{cpu_metric.value:.1f}%"
        if mem_metric:
            details["memory_usage"] = f"{mem_metric.value:.1f}%"
        if disk_metric:
            details["disk_usage"] = f"{disk_metric.value:.1f}%"

        healthy = all(results.values())

        return HealthStatus(
            healthy=healthy,
            checks=results,
            details=details
        )


class AlertManager:
    """Manage alerts based on metrics."""

    def __init__(self, collector: MetricsCollector):
        """Initialize alert manager.

        Args:
            collector: Metrics collector
        """
        self.collector = collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []

    def register_alert(self, alert: Alert) -> None:
        """Register an alert.

        Args:
            alert: Alert configuration
        """
        self.alerts[alert.name] = alert

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register alert handler.

        Args:
            handler: Function to call when alert triggers
        """
        self.alert_handlers.append(handler)

    def check_alerts(self) -> List[Alert]:
        """Check all alerts and trigger handlers.

        Returns:
            List of triggered alerts
        """
        triggered = []

        for name, alert in self.alerts.items():
            # Check if in cooldown
            if alert.last_triggered:
                if datetime.now() - alert.last_triggered < alert.cooldown:
                    continue

            # Get latest metric value
            metric = self.collector.get_latest(name)
            if metric is None:
                continue

            # Check condition
            if alert.condition(metric.value):
                alert.last_triggered = datetime.now()
                alert.triggered_count += 1
                triggered.append(alert)

                # Call handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler failed: {e}")

        return triggered


class MonitoringService:
    """Complete monitoring service."""

    def __init__(self, metrics_file: Optional[Path] = None):
        """Initialize monitoring service.

        Args:
            metrics_file: Optional file to persist metrics
        """
        self.collector = MetricsCollector()
        self.system_monitor = SystemMonitor(self.collector)
        self.app_monitor = ApplicationMonitor(self.collector)
        self.health_checker = HealthChecker(self.collector)
        self.alert_manager = AlertManager(self.collector)
        self.metrics_file = metrics_file

        # Register default alerts
        self._register_default_alerts()

    def _register_default_alerts(self) -> None:
        """Register default system alerts."""
        # High CPU alert
        self.alert_manager.register_alert(Alert(
            name="system.cpu.usage_percent",
            condition=lambda x: x > 90,
            message="CPU usage above 90%",
            severity="warning"
        ))

        # High memory alert
        self.alert_manager.register_alert(Alert(
            name="system.memory.usage_percent",
            condition=lambda x: x > 90,
            message="Memory usage above 90%",
            severity="warning"
        ))

        # Critical memory alert
        self.alert_manager.register_alert(Alert(
            name="system.memory.usage_percent",
            condition=lambda x: x > 95,
            message="Memory usage critically high (>95%)",
            severity="critical"
        ))

    def collect_metrics(self) -> None:
        """Collect all metrics."""
        self.system_monitor.collect_all()
        self.collector.cleanup_old_metrics()

    def get_health_status(self) -> HealthStatus:
        """Get current health status.

        Returns:
            Health status
        """
        return self.health_checker.run_all_checks()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Dictionary with metric summaries
        """
        summary = {}

        for metric_name in self.collector.metrics.keys():
            stats = self.collector.get_statistics(metric_name)
            if stats:
                summary[metric_name] = stats

        return summary

    def export_metrics(self, output_file: Optional[Path] = None) -> None:
        """Export metrics to file.

        Args:
            output_file: Output file path (defaults to metrics_file)
        """
        output_file = output_file or self.metrics_file
        if not output_file:
            return

        # Export current metrics
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_metrics_summary(),
            "health": {
                "healthy": self.get_health_status().healthy,
                "checks": self.get_health_status().checks
            }
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)


def create_monitoring_service(
    metrics_file: Optional[Path] = None,
    alert_handler: Optional[Callable[[Alert], None]] = None
) -> MonitoringService:
    """Create and configure monitoring service.

    Args:
        metrics_file: Optional file to persist metrics
        alert_handler: Optional custom alert handler

    Returns:
        Configured monitoring service
    """
    service = MonitoringService(metrics_file=metrics_file)

    if alert_handler:
        service.alert_manager.register_handler(alert_handler)
    else:
        # Default alert handler - log to console
        def default_handler(alert: Alert):
            logger.warning(
                f"Alert triggered: {alert.name} - {alert.message} "
                f"(severity: {alert.severity})"
            )
        service.alert_manager.register_handler(default_handler)

    return service
