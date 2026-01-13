---
name: observability-sre-practices
version: "1.0.7"
maturity: "5-Expert"
specialization: SRE and Observability
description: Production observability, monitoring, SRE with OpenTelemetry, Prometheus, Grafana, incident management. Use for tracing/metrics/logs, SLOs/SLIs, alerts with AlertManager, Golden Signals, incident response, post-mortems, error budgets.
---

# Observability and SRE

## Tool Selection

| Tool | Purpose | Use |
|------|---------|-----|
| OpenTelemetry | Instrumentation | Traces, metrics, logs |
| Prometheus | Metrics | Time-series collection |
| Grafana | Visualization | Dashboards, alerts |
| Jaeger/Zipkin | Tracing | Request flow |
| ELK/Loki | Logs | Search, analysis |
| PagerDuty | Incidents | On-call, escalation |

## OpenTelemetry

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

def setup(service, env):
    resource = Resource.create({"service.name":service,"service.environment":env})
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel:4317")))
    trace.set_tracer_provider(tp)
    return trace.get_tracer(__name__), metrics.get_meter(__name__)

tracer, meter = setup("payment-service", "production")

@tracer.start_as_current_span("process_payment")
def process_payment(amount):
    span = trace.get_current_span()
    span.set_attribute("payment.amount", amount)
```

## Prometheus

```python
from prometheus_client import Counter, Histogram, start_http_server

http_requests = Counter('http_requests_total','Total HTTP',['method','endpoint','status'])
http_duration = Histogram('http_request_duration_seconds','Latency',['method','endpoint'],
                         buckets=[0.01,0.05,0.1,0.5,1.0,5.0])

def prom_metrics(method, endpoint):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time; start = time.time(); status = "200"
            try: return func(*args, **kwargs)
            except: status = "500"; raise
            finally:
                http_requests.labels(method, endpoint, status).inc()
                http_duration.labels(method, endpoint).observe(time.time()-start)
        return wrapper
    return decorator
```

## Alerts

```yaml
# prometheus_alerts.yml
groups:
  - name: sre
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          / sum(rate(http_requests_total[5m])) by (service) > 0.05
        for: 5m
        labels: {severity: critical}
        annotations:
          summary: "High error for {{ $labels.service }}"
          runbook: "https://wiki/runbooks/high-error"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,sum(rate(http_request_duration_seconds_bucket[5m])) by (le,endpoint)) > 1.0
        for: 10m
        labels: {severity: warning}
        annotations: {summary: "P95 >1s for {{ $labels.endpoint }}"}
```

## SLO/SLI

```python
@dataclass
class SLI:
    name: str; query: str; target: float; window: str

@dataclass
class SLO:
    name: str; slis: list[SLI]; error_budget: float

availability_slo = SLO(
    name="API Availability",
    slis=[SLI(name="Success Rate",
              query='sum(rate(http_requests_total{status!~"5.."}[30d]))/sum(rate(http_requests_total[30d]))',
              target=0.999, window="30d")],
    error_budget=0.001
)

def calc_error_budget(slo, actual):
    target = slo.slis[0].target
    consumed = (target - actual) / (1 - target)
    return {'target':target, 'actual':actual, 'remaining':1-consumed,
            'status':'HEALTHY' if 1-consumed>0.2 else 'AT_RISK'}
```

## Golden Signals

| Signal | Monitor | Alert |
|--------|---------|-------|
| Latency | P50/P95/P99 | P99 >500ms |
| Traffic | RPS | Anomaly (>2x or <0.5x) |
| Errors | 5xx rate, types | >1% error rate |
| Saturation | CPU, memory, disk, conns | >80% |

```python
class GoldenSignals:
    def latency(self, svc, threshold_ms=500):
        q = f'histogram_quantile(0.99,sum(rate(http_request_duration_seconds_bucket{{service="{svc}"}}[5m])) by (le))*1000'
        p99 = self.query_prometheus(q)
        return {'p99_ms':p99, 'alert':p99>threshold_ms}

    def errors(self, svc, threshold_pct=1.0):
        q = f'sum(rate(http_requests_total{{service="{svc}",status=~"5.."}}[5m]))/sum(rate(http_requests_total{{service="{svc}"}}[5m]))*100'
        rate = self.query_prometheus(q)
        return {'error_rate_pct':rate, 'alert':rate>threshold_pct}
```

## Incident Management

```python
class IncidentSeverity(Enum):
    SEV1 = "Critical - Complete outage"
    SEV2 = "High - Major impaired"
    SEV3 = "Medium - Minor impaired"
    SEV4 = "Low - Cosmetic"

@dataclass
class Incident:
    id: str; title: str; severity: IncidentSeverity; detected_at: datetime
    resolved_at: datetime | None = None; root_cause: str | None = None; timeline: list = None

class IncidentManager:
    def create(self, title, severity):
        inc = Incident(id=f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                      title=title, severity=severity, detected_at=datetime.now(), timeline=[])
        self.notify_oncall(inc)
        return inc

    def postmortem(self, inc):
        mttr = (inc.resolved_at - inc.detected_at).seconds / 60
        return f"""# Post-Mortem: {inc.title}
- Severity: {inc.severity.value}
- Duration: {mttr:.2f}min
- Root Cause: {inc.root_cause}
## Action Items
- [ ] Improve monitoring
- [ ] Add automation"""
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Instrument first | Before production deploy |
| Structured logging | JSON with correlation IDs |
| SLO alerting | Alert on burn rate |
| Actionable alerts | Every alert has runbook |
| Blameless postmortems | Focus on systems |
| Error budgets | Balance velocity vs reliability |

## Checklist

- [ ] OpenTelemetry traces, metrics, logs
- [ ] Prometheus scraping all services
- [ ] Grafana dashboards for Golden Signals
- [ ] SLOs with error budgets
- [ ] AlertManager rules with runbooks
- [ ] On-call rotation
- [ ] Post-mortem template
- [ ] Structured logging with correlation IDs
