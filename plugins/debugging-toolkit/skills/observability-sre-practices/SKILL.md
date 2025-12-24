---
name: observability-sre-practices
version: "1.0.5"
maturity: "5-Expert"
specialization: SRE and Observability
description: Production-grade observability, monitoring, and SRE practices with OpenTelemetry, Prometheus, Grafana, and incident management. Use when implementing tracing/metrics/logs, defining SLOs/SLIs, setting up alerts with AlertManager, implementing Golden Signals monitoring, conducting incident response and post-mortems, or building error budgets.
---

# Observability and SRE Practices

Production-grade monitoring, observability, and site reliability engineering.

---

## Tool Selection

| Tool | Purpose | Use Case |
|------|---------|----------|
| OpenTelemetry | Instrumentation | Traces, metrics, logs |
| Prometheus | Metrics storage | Time-series collection |
| Grafana | Visualization | Dashboards, alerts |
| Jaeger/Zipkin | Distributed tracing | Request flow analysis |
| ELK/Loki | Log aggregation | Search, analysis |
| PagerDuty | Incident management | On-call, escalation |

---

## OpenTelemetry Setup

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

def setup_observability(service_name: str, environment: str):
    resource = Resource.create({
        "service.name": service_name,
        "service.environment": environment,
    })

    # Traces
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
    )
    trace.set_tracer_provider(trace_provider)

    return trace.get_tracer(__name__), metrics.get_meter(__name__)

# Instrumented function
tracer, meter = setup_observability("payment-service", "production")

@tracer.start_as_current_span("process_payment")
def process_payment(amount: float):
    span = trace.get_current_span()
    span.set_attribute("payment.amount", amount)
    # Business logic
```

---

## Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

http_requests_total = Counter(
    'http_requests_total', 'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds', 'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# Decorator for instrumentation
def prometheus_metrics(method: str, endpoint: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            status = "200"
            try:
                return func(*args, **kwargs)
            except Exception:
                status = "500"
                raise
            finally:
                http_requests_total.labels(method, endpoint, status).inc()
                http_request_duration.labels(method, endpoint).observe(time.time() - start)
        return wrapper
    return decorator
```

---

## Alert Rules

```yaml
# prometheus_alerts.yml
groups:
  - name: sre_alerts
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          / sum(rate(http_requests_total[5m])) by (service) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate for {{ $labels.service }}"
          runbook: "https://wiki.company.com/runbooks/high-error-rate"

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
```

---

## SLO/SLI Framework

```python
@dataclass
class SLI:
    name: str
    query: str  # Prometheus query
    target: float  # e.g., 0.999 for 99.9%
    window: str  # e.g., "30d"

@dataclass
class SLO:
    name: str
    slis: list[SLI]
    error_budget: float  # e.g., 0.001 for 0.1%

# Example SLOs
availability_slo = SLO(
    name="API Availability",
    slis=[SLI(
        name="Success Rate",
        query='sum(rate(http_requests_total{status!~"5.."}[30d])) / sum(rate(http_requests_total[30d]))',
        target=0.999,
        window="30d"
    )],
    error_budget=0.001
)

def calculate_error_budget(slo: SLO, actual: float) -> dict:
    target = slo.slis[0].target
    consumed = (target - actual) / (1 - target)
    return {
        'target': target,
        'actual': actual,
        'remaining': 1 - consumed,
        'status': 'HEALTHY' if (1 - consumed) > 0.2 else 'AT_RISK'
    }
```

---

## Golden Signals

| Signal | What to Monitor | Alert Threshold |
|--------|-----------------|-----------------|
| Latency | P50, P95, P99 response time | P99 > 500ms |
| Traffic | Requests per second | Anomaly (>2x or <0.5x baseline) |
| Errors | 5xx rate, error types | >1% error rate |
| Saturation | CPU, memory, disk, connections | >80% utilization |

```python
class GoldenSignalsMonitor:
    def check_latency(self, service: str, threshold_ms: float = 500) -> dict:
        query = f'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le)) * 1000'
        p99 = self.query_prometheus(query)
        return {'p99_ms': p99, 'alert': p99 > threshold_ms}

    def check_errors(self, service: str, threshold_pct: float = 1.0) -> dict:
        query = f'sum(rate(http_requests_total{{service="{service}",status=~"5.."}}[5m])) / sum(rate(http_requests_total{{service="{service}"}}[5m])) * 100'
        error_rate = self.query_prometheus(query)
        return {'error_rate_pct': error_rate, 'alert': error_rate > threshold_pct}
```

---

## Incident Management

```python
class IncidentSeverity(Enum):
    SEV1 = "Critical - Complete outage"
    SEV2 = "High - Major functionality impaired"
    SEV3 = "Medium - Minor functionality impaired"
    SEV4 = "Low - Cosmetic issue"

@dataclass
class Incident:
    id: str
    title: str
    severity: IncidentSeverity
    detected_at: datetime
    resolved_at: datetime | None = None
    root_cause: str | None = None
    timeline: list[dict] = None

class IncidentManager:
    def create_incident(self, title: str, severity: IncidentSeverity) -> Incident:
        incident = Incident(
            id=f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            title=title,
            severity=severity,
            detected_at=datetime.now(),
            timeline=[]
        )
        self.notify_oncall(incident)
        return incident

    def generate_postmortem(self, incident: Incident) -> str:
        mttr = (incident.resolved_at - incident.detected_at).total_seconds() / 60
        return f"""
# Post-Mortem: {incident.title}
- Severity: {incident.severity.value}
- Duration: {mttr:.2f} minutes
- Root Cause: {incident.root_cause}

## Action Items
- [ ] Improve monitoring
- [ ] Add automated remediation
"""
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Instrument first | Add observability before production deploy |
| Structured logging | JSON format with correlation IDs |
| SLO-based alerting | Alert on burn rate, not thresholds |
| Actionable alerts | Every alert has a runbook |
| Blameless post-mortems | Focus on systems, not people |
| Error budgets | Balance velocity vs reliability |

---

## Checklist

- [ ] OpenTelemetry traces, metrics, logs configured
- [ ] Prometheus scraping all services
- [ ] Grafana dashboards for Golden Signals
- [ ] SLOs defined with error budgets
- [ ] AlertManager rules with runbooks
- [ ] On-call rotation configured
- [ ] Post-mortem template ready
- [ ] Structured logging with correlation IDs

---

**Version**: 1.0.5
