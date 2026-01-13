---
name: distributed-tracing
version: "1.0.7"
maturity: "5-Expert"
specialization: Request Flow Visibility
description: Implement distributed tracing with OpenTelemetry, Jaeger, and Tempo including instrumentation, context propagation, sampling strategies, and trace analysis. Use when debugging latency issues, understanding service dependencies, or tracing error propagation across microservices.
---

# Distributed Tracing

Track requests across distributed systems for latency and dependency analysis.

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| Trace | End-to-end request journey |
| Span | Single operation within trace |
| Context | Metadata propagated between services |
| Tags | Key-value pairs for filtering |

```
Trace (Request ID: abc123)
  ↓
Span (frontend) [100ms]
  ↓
Span (api-gateway) [80ms]
  ├→ Span (auth-service) [10ms]
  └→ Span (user-service) [60ms]
      └→ Span (database) [40ms]
```

---

## OpenTelemetry Instrumentation

### Python (Flask)

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(
    JaegerExporter(agent_host_name="jaeger", agent_port=6831)
))
trace.set_tracer_provider(provider)

app = Flask(__name__)
FlaskInstrumentor().instrument_app(app)

@app.route('/api/users')
def get_users():
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("get_users") as span:
        span.set_attribute("user.count", 100)
        return fetch_users()
```

### Node.js (Express)

```javascript
const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');
const { registerInstrumentations } = require('@opentelemetry/instrumentation');

const provider = new NodeTracerProvider();
provider.addSpanProcessor(new BatchSpanProcessor(new JaegerExporter({
  endpoint: 'http://jaeger:14268/api/traces'
})));
provider.register();

registerInstrumentations({
  instrumentations: [new HttpInstrumentation(), new ExpressInstrumentation()]
});
```

---

## Context Propagation

```python
# HTTP Header
traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01

# Propagate in requests
from opentelemetry.propagate import inject
headers = {}
inject(headers)
response = requests.get('http://downstream/api', headers=headers)
```

---

## Jaeger Deployment

### Docker Compose

```yaml
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "6831:6831/udp"  # Thrift compact
      - "14268:14268"  # HTTP collector
```

### Kubernetes

```yaml
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger
spec:
  strategy: production
  storage:
    type: elasticsearch
```

---

## Sampling Strategies

| Strategy | Config | Use Case |
|----------|--------|----------|
| Probabilistic | `param: 0.01` | Sample 1% |
| Rate Limiting | `param: 100` | Max 100/sec |
| Adaptive | Parent-based | Follow parent decision |

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
sampler = TraceIdRatioBased(0.01)  # 1% sampling
```

---

## Trace Analysis

### Finding Slow Requests

```
service=my-service duration > 1s
```

### Finding Errors

```
service=my-service error=true tags.http.status_code >= 500
```

---

## Log Correlation

```python
import logging
from opentelemetry import trace

def process_request():
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, '032x')
    logger.info("Processing request", extra={"trace_id": trace_id})
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Sample appropriately | 1-10% in production |
| Add meaningful tags | user_id, request_id |
| Propagate context | All service boundaries |
| Log exceptions | Record errors in spans |
| Monitor overhead | <1% CPU impact |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No traces | Check collector endpoint, sampling |
| High latency | Reduce sampling, use batch processor |
| Missing spans | Verify context propagation |

---

## Checklist

- [ ] OpenTelemetry instrumentation configured
- [ ] Context propagation in HTTP/gRPC
- [ ] Sampling strategy appropriate for production
- [ ] Jaeger/Tempo collector deployed
- [ ] Log correlation with trace IDs
- [ ] Service dependency graph visible

---

**Version**: 1.0.5
