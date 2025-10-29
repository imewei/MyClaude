---
name: observability-sre-practices
description: Production observability with Datadog, Prometheus, Grafana, OpenTelemetry, ELK stack, SRE practices (SLO/SLI, incident management), Golden Signals monitoring, and automated alerting for distributed systems
tools: Read, Write, Bash, kubectl, prometheus, grafana
integration: Use for implementing observability, monitoring, and SRE best practices
---

# Observability and SRE Practices Mastery

Complete framework for implementing production-grade observability, monitoring, and site reliability engineering practices with modern tools and AI-enhanced alerting.

## When to Use This Skill

- **Observability setup**: OpenTelemetry, Prometheus, Grafana, Datadog, New Relic
- **SLO/SLI definition**: Service level objectives and indicators for reliability
- **Incident management**: On-call, post-mortems, PagerDuty integration
- **Golden Signals**: Latency, traffic, errors, saturation monitoring
- **Distributed tracing**: Request flow tracking across microservices
- **Log aggregation**: ELK stack, Loki, Splunk for centralized logging

## Core Observability Patterns

### 1. OpenTelemetry Implementation

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
import logging

def setup_observability(service_name: str, environment: str):
    """
    Complete OpenTelemetry setup for traces, metrics, and logs.

    Parameters
    ----------
    service_name : str
        Name of the service
    environment : str
        Environment (dev, staging, prod)
    """
    # Resource attributes for all telemetry
    resource = Resource.create({
        "service.name": service_name,
        "service.environment": environment,
        "service.version": "1.0.0"
    })

    # Traces
    trace_provider = TracerProvider(resource=resource)
    otlp_trace_exporter = OTLPSpanExporter(
        endpoint="http://otel-collector:4317",
        insecure=True
    )
    trace_provider.add_span_processor(
        BatchSpanProcessor(otlp_trace_exporter)
    )
    trace.set_tracer_provider(trace_provider)

    # Metrics
    otlp_metric_exporter = OTLPMetricExporter(
        endpoint="http://otel-collector:4317",
        insecure=True
    )
    metric_reader = PeriodicExportingMetricReader(
        otlp_metric_exporter,
        export_interval_millis=60000  # Export every 60 seconds
    )
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader]
    )
    metrics.set_meter_provider(meter_provider)

    # Logs (structured logging)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    return trace.get_tracer(__name__), metrics.get_meter(__name__)

# Usage in application
tracer, meter = setup_observability("payment-service", "production")

# Create custom metrics
request_counter = meter.create_counter(
    "http.server.requests",
    description="Total HTTP requests",
    unit="1"
)

request_duration = meter.create_histogram(
    "http.server.duration",
    description="HTTP request duration",
    unit="ms"
)

# Instrumented endpoint
@tracer.start_as_current_span("process_payment")
def process_payment(amount: float, currency: str):
    """Example instrumented function."""
    import time
    start = time.time()

    try:
        # Business logic
        result = charge_payment(amount, currency)

        # Record metrics
        request_counter.add(1, {"status": "success", "currency": currency})
        duration_ms = (time.time() - start) * 1000
        request_duration.record(duration_ms, {"status": "success"})

        return result

    except Exception as e:
        request_counter.add(1, {"status": "error", "error_type": type(e).__name__})

        # Add error details to span
        span = trace.get_current_span()
        span.set_attribute("error", True)
        span.set_attribute("error.type", type(e).__name__)
        span.set_attribute("error.message", str(e))

        raise
```

### 2. Prometheus Metrics and Alerting

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import REGISTRY, start_http_server
import time

# Define metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

error_rate = Gauge(
    'error_rate',
    'Current error rate',
    ['service']
)

# Instrumentation decorator
def prometheus_metrics(method: str, endpoint: str):
    """Decorator for automatic Prometheus instrumentation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "200"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "500"
                raise
            finally:
                duration = time.time() - start_time
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status
                ).inc()
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)

        return wrapper
    return decorator

# Usage
@prometheus_metrics("POST", "/api/payment")
def create_payment():
    pass

# Start metrics server
start_http_server(9090)
```

#### Prometheus Alerting Rules

```yaml
# prometheus_alerts.yml
groups:
  - name: sre_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          /
          sum(rate(http_requests_total[5m])) by (service)
          > 0.05
        for: 5m
        labels:
          severity: critical
          team: sre
        annotations:
          summary: "High error rate detected for {{ $labels.service }}"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
          runbook: "https://wiki.company.com/runbooks/high-error-rate"

      # High latency (P95)
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint)
          ) > 1.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency above 1s for {{ $labels.endpoint }}"
          description: "Current P95 latency: {{ $value | humanizeDuration }}"

      # Memory usage
      - alert: HighMemoryUsage
        expr: |
          (process_resident_memory_bytes / process_virtual_memory_max_bytes) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is at {{ $value | humanizePercentage }}"

      # SLO burn rate
      - alert: SLOBurnRateTooFast
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[1h]))
            /
            sum(rate(http_requests_total[1h]))
          ) < 0.995
        for: 1h
        labels:
          severity: critical
          slo: availability
        annotations:
          summary: "SLO burn rate too fast - 99.5% availability at risk"
          description: "Current availability: {{ $value | humanizePercentage }}"
```

### 3. SLO/SLI Framework

```python
from dataclasses import dataclass
from enum import Enum
from typing import List
import datetime

class SLIType(Enum):
    AVAILABILITY = "availability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"

@dataclass
class SLI:
    """Service Level Indicator."""
    name: str
    type: SLIType
    query: str  # Prometheus query
    target: float  # Target value (e.g., 0.999 for 99.9%)
    window: str  # Time window (e.g., "30d")

@dataclass
class SLO:
    """Service Level Objective."""
    name: str
    description: str
    slis: List[SLI]
    error_budget: float  # e.g., 0.001 for 99.9% target = 0.1% error budget

class SLOMonitor:
    """Monitor SLOs and calculate error budget burn."""

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url

    def define_availability_slo(self) -> SLO:
        """Define availability SLO."""
        return SLO(
            name="API Availability",
            description="99.9% of requests return non-5xx status",
            slis=[
                SLI(
                    name="Success Rate",
                    type=SLIType.AVAILABILITY,
                    query="""
                        sum(rate(http_requests_total{status!~"5.."}[30d]))
                        /
                        sum(rate(http_requests_total[30d]))
                    """,
                    target=0.999,
                    window="30d"
                )
            ],
            error_budget=0.001  # 0.1% error budget
        )

    def define_latency_slo(self) -> SLO:
        """Define latency SLO."""
        return SLO(
            name="API Latency",
            description="99% of requests complete within 500ms",
            slis=[
                SLI(
                    name="P99 Latency",
                    type=SLIType.LATENCY,
                    query="""
                        histogram_quantile(0.99,
                            sum(rate(http_request_duration_seconds_bucket[30d])) by (le)
                        )
                    """,
                    target=0.5,  # 500ms
                    window="30d"
                )
            ],
            error_budget=0.01  # 1% can be slower
        )

    def calculate_error_budget(self, slo: SLO) -> dict:
        """
        Calculate remaining error budget.

        Returns
        -------
        dict
            Error budget status with burn rate
        """
        # Query Prometheus for actual SLI value
        actual_sli = self.query_prometheus(slo.slis[0].query)

        # Calculate error budget consumption
        target = slo.slis[0].target
        actual = actual_sli['value']

        if slo.slis[0].type == SLIType.AVAILABILITY:
            # For availability: error budget = (target - actual) / (1 - target)
            error_budget_consumed = (target - actual) / (1 - target)
        else:
            # For latency: error budget = (actual - target) / target
            error_budget_consumed = max(0, (actual - target) / target)

        remaining = 1 - error_budget_consumed

        # Calculate burn rate (how fast we're consuming error budget)
        # Burn rate = (errors in last hour) / (total error budget for 30 days)
        burn_rate = self.calculate_burn_rate(slo)

        return {
            'slo_name': slo.name,
            'target': target,
            'actual': actual,
            'error_budget_remaining': remaining,
            'error_budget_consumed': error_budget_consumed,
            'burn_rate': burn_rate,
            'status': 'HEALTHY' if remaining > 0.2 else 'AT_RISK'
        }

    def query_prometheus(self, query: str) -> dict:
        """Query Prometheus (mock implementation)."""
        # In production, use requests.get(f"{self.prometheus_url}/api/v1/query")
        return {'value': 0.9985}  # Example: 99.85% availability

    def calculate_burn_rate(self, slo: SLO) -> float:
        """Calculate error budget burn rate."""
        # Query error rate for last 1 hour
        hourly_query = slo.slis[0].query.replace("[30d]", "[1h]")
        hourly_sli = self.query_prometheus(hourly_query)

        # Compare to error budget
        # Burn rate > 1 means consuming error budget faster than planned
        target = slo.slis[0].target
        actual = hourly_sli['value']

        # Normalize to 30-day window
        hourly_error = max(0, target - actual)
        monthly_error_budget = 1 - target

        burn_rate = (hourly_error * 24 * 30) / monthly_error_budget

        return burn_rate

# Usage
monitor = SLOMonitor(prometheus_url="http://prometheus:9090")

availability_slo = monitor.define_availability_slo()
budget_status = monitor.calculate_error_budget(availability_slo)

print(f"SLO: {budget_status['slo_name']}")
print(f"Target: {budget_status['target']:.2%}")
print(f"Actual: {budget_status['actual']:.2%}")
print(f"Error Budget Remaining: {budget_status['error_budget_remaining']:.2%}")
print(f"Burn Rate: {budget_status['burn_rate']:.2f}x")
print(f"Status: {budget_status['status']}")
```

### 4. Incident Management

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

class IncidentSeverity(Enum):
    SEV1 = "Critical - Complete outage"
    SEV2 = "High - Major functionality impaired"
    SEV3 = "Medium - Minor functionality impaired"
    SEV4 = "Low - Cosmetic issue"

@dataclass
class Incident:
    """Incident tracking."""
    id: str
    title: str
    severity: IncidentSeverity
    description: str
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    root_cause: Optional[str] = None
    impact: Optional[str] = None
    timeline: List[dict] = None
    action_items: List[str] = None

class IncidentManager:
    """Manage incidents from detection to post-mortem."""

    def create_incident(
        self,
        title: str,
        severity: IncidentSeverity,
        description: str
    ) -> Incident:
        """Create new incident."""
        incident = Incident(
            id=f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            title=title,
            severity=severity,
            description=description,
            detected_at=datetime.now(),
            timeline=[{
                'timestamp': datetime.now(),
                'event': 'Incident detected',
                'details': description
            }]
        )

        # Notify on-call engineer
        self.notify_oncall(incident)

        # Create incident channel
        self.create_slack_channel(incident)

        return incident

    def add_timeline_event(self, incident: Incident, event: str, details: str):
        """Add event to incident timeline."""
        incident.timeline.append({
            'timestamp': datetime.now(),
            'event': event,
            'details': details
        })

    def resolve_incident(
        self,
        incident: Incident,
        root_cause: str,
        resolution: str
    ):
        """Mark incident as resolved."""
        incident.resolved_at = datetime.now()
        incident.root_cause = root_cause

        self.add_timeline_event(
            incident,
            'Incident resolved',
            resolution
        )

        # Calculate MTTR
        mttr = (incident.resolved_at - incident.detected_at).total_seconds() / 60
        print(f"MTTR: {mttr:.2f} minutes")

    def generate_postmortem(self, incident: Incident) -> str:
        """Generate post-mortem report."""
        return f"""
# Post-Mortem: {incident.title}

## Incident Summary
- **ID**: {incident.id}
- **Severity**: {incident.severity.value}
- **Detected**: {incident.detected_at}
- **Resolved**: {incident.resolved_at}
- **Duration**: {(incident.resolved_at - incident.detected_at).total_seconds() / 60:.2f} minutes

## Impact
{incident.impact or 'To be determined'}

## Root Cause
{incident.root_cause or 'Under investigation'}

## Timeline
{self._format_timeline(incident.timeline)}

## Action Items
{self._format_action_items(incident.action_items or [])}

## Lessons Learned
1. What went well?
2. What could be improved?
3. What are we doing to prevent this in the future?
"""

    def _format_timeline(self, timeline: List[dict]) -> str:
        """Format timeline for report."""
        return "\n".join([
            f"- **{event['timestamp'].strftime('%H:%M:%S')}**: {event['event']} - {event['details']}"
            for event in timeline
        ])

    def _format_action_items(self, items: List[str]) -> str:
        """Format action items."""
        return "\n".join([f"- [ ] {item}" for item in items])

    def notify_oncall(self, incident: Incident):
        """Notify on-call engineer (mock)."""
        print(f"📟 Paging on-call for {incident.severity.value}")

    def create_slack_channel(self, incident: Incident):
        """Create incident Slack channel (mock)."""
        channel_name = f"incident-{incident.id.lower()}"
        print(f"💬 Created Slack channel: #{channel_name}")

# Usage
manager = IncidentManager()

# Detect incident
incident = manager.create_incident(
    title="API returning 500 errors",
    severity=IncidentSeverity.SEV1,
    description="User-facing API experiencing high error rates"
)

# Investigation updates
manager.add_timeline_event(
    incident,
    "Investigation started",
    "Checking database connections"
)

manager.add_timeline_event(
    incident,
    "Root cause identified",
    "Database connection pool exhausted"
)

# Resolve
manager.resolve_incident(
    incident,
    root_cause="Database connection pool size too small for traffic spike",
    resolution="Increased pool size from 10 to 50 connections"
)

# Generate post-mortem
postmortem = manager.generate_postmortem(incident)
print(postmortem)
```

### 5. Golden Signals Monitoring

```python
class GoldenSignalsMonitor:
    """Monitor the Four Golden Signals."""

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url

    def check_latency(self, service: str, threshold_ms: float = 500) -> dict:
        """
        Check latency (how long it takes to serve a request).

        Parameters
        ----------
        service : str
            Service name
        threshold_ms : float
            Latency threshold in milliseconds

        Returns
        -------
        dict
            Latency metrics and alert status
        """
        query = f"""
        histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le)
        ) * 1000
        """

        p99_latency = self.query_prometheus(query)

        return {
            'signal': 'Latency',
            'p99_ms': p99_latency,
            'threshold_ms': threshold_ms,
            'alert': p99_latency > threshold_ms,
            'severity': 'CRITICAL' if p99_latency > threshold_ms * 2 else 'WARNING'
        }

    def check_traffic(self, service: str, baseline_rps: float = 100) -> dict:
        """
        Check traffic (how much demand is being placed on your system).

        Parameters
        ----------
        service : str
            Service name
        baseline_rps : float
            Baseline requests per second for anomaly detection

        Returns
        -------
        dict
            Traffic metrics and anomaly detection
        """
        query = f'sum(rate(http_requests_total{{service="{service}"}}[5m]))'
        current_rps = self.query_prometheus(query)

        # Check for traffic anomalies (>2x or <0.5x baseline)
        is_anomaly = current_rps > baseline_rps * 2 or current_rps < baseline_rps * 0.5

        return {
            'signal': 'Traffic',
            'current_rps': current_rps,
            'baseline_rps': baseline_rps,
            'anomaly': is_anomaly,
            'severity': 'WARNING' if is_anomaly else 'OK'
        }

    def check_errors(self, service: str, threshold_percent: float = 1.0) -> dict:
        """
        Check errors (the rate of requests that fail).

        Parameters
        ----------
        service : str
            Service name
        threshold_percent : float
            Error rate threshold as percentage

        Returns
        -------
        dict
            Error rate metrics and alert status
        """
        query = f"""
        sum(rate(http_requests_total{{service="{service}",status=~"5.."}}[5m]))
        /
        sum(rate(http_requests_total{{service="{service}"}}[5m]))
        * 100
        """

        error_rate = self.query_prometheus(query)

        return {
            'signal': 'Errors',
            'error_rate_percent': error_rate,
            'threshold_percent': threshold_percent,
            'alert': error_rate > threshold_percent,
            'severity': 'CRITICAL' if error_rate > threshold_percent * 5 else 'WARNING'
        }

    def check_saturation(self, service: str) -> dict:
        """
        Check saturation (how "full" your service is).

        Monitors CPU, memory, disk, network utilization.

        Parameters
        ----------
        service : str
            Service name

        Returns
        -------
        dict
            Saturation metrics for key resources
        """
        metrics = {}

        # CPU
        cpu_query = f'avg(rate(process_cpu_seconds_total{{service="{service}"}}[5m])) * 100'
        metrics['cpu_percent'] = self.query_prometheus(cpu_query)

        # Memory
        mem_query = f"""
        sum(process_resident_memory_bytes{{service="{service}"}})
        /
        sum(process_virtual_memory_max_bytes{{service="{service}"}})
        * 100
        """
        metrics['memory_percent'] = self.query_prometheus(mem_query)

        # Alert if any resource > 80%
        critical_resources = [k for k, v in metrics.items() if v > 80]

        return {
            'signal': 'Saturation',
            'metrics': metrics,
            'critical_resources': critical_resources,
            'alert': len(critical_resources) > 0,
            'severity': 'CRITICAL' if any(v > 90 for v in metrics.values()) else 'WARNING'
        }

    def query_prometheus(self, query: str) -> float:
        """Query Prometheus (mock)."""
        # In production: requests.get(f"{self.prometheus_url}/api/v1/query", params={'query': query})
        import random
        return random.uniform(50, 200)

    def check_all_signals(self, service: str) -> list:
        """Check all four golden signals."""
        return [
            self.check_latency(service),
            self.check_traffic(service),
            self.check_errors(service),
            self.check_saturation(service)
        ]

# Usage
monitor = GoldenSignalsMonitor("http://prometheus:9090")
signals = monitor.check_all_signals("payment-service")

for signal in signals:
    status = "🔴 ALERT" if signal.get('alert') else "✅ OK"
    print(f"{status} {signal['signal']}: {signal.get('severity', 'OK')}")
```

## Best Practices

1. **Instrumentation First**: Add observability before deploying to production
2. **Structured Logging**: Use JSON logging with consistent fields
3. **Distributed Tracing**: Trace requests across all services
4. **Actionable Alerts**: Every alert should have a runbook
5. **SLO-Based Alerting**: Alert on SLO burn rate, not arbitrary thresholds
6. **Automated Remediation**: Auto-scale, restart, rollback when safe
7. **Regular Fire Drills**: Practice incident response quarterly
8. **Blame-Free Post-Mortems**: Focus on systems, not people

This skill provides comprehensive observability and SRE practices for production systems!
