# Observability Integration Guide

> Comprehensive guide for APM integration, distributed tracing, logging best practices, monitoring dashboards, and production-safe debugging techniques

## Table of Contents

1. [APM Integration](#apm-integration)
2. [Distributed Tracing Setup](#distributed-tracing-setup)
3. [Logging Best Practices](#logging-best-practices)
4. [Monitoring Dashboards and Alerting](#monitoring-dashboards-and-alerting)
5. [Production-Safe Debugging Techniques](#production-safe-debugging-techniques)
6. [Observability Tool Comparison Matrix](#observability-tool-comparison-matrix)

---

## APM Integration

### Datadog APM

**Installation:**

```python
# requirements.txt
ddtrace>=1.0.2

# Initialize in application
from ddtrace import tracer, patch_all

# Patch all supported libraries
patch_all()

# Configure Datadog
tracer.configure(
    hostname='datadog-agent',
    port=8126,
    priority_sampling=True,
)
```

**Instrumentation:**

```python
from ddtrace import tracer

@tracer.wrap('payment.process', service='payment-api')
def process_payment(payment_id):
    """Process payment with automatic tracing."""
    with tracer.trace('database.query', resource='fetch_payment'):
        payment = db.get_payment(payment_id)

    with tracer.trace('external.api', resource='charge_payment'):
        result = payment_gateway.charge(payment)

    return result

# Add custom tags
span = tracer.current_span()
span.set_tag('payment.amount', payment.amount)
span.set_tag('payment.currency', payment.currency)
span.set_tag('user.id', payment.user_id)
```

**Error Tracking:**

```python
from ddtrace import tracer

def handle_payment_error(error, payment_id):
    """Track errors with context."""
    span = tracer.current_span()
    if span:
        span.set_exc_info(*sys.exc_info())
        span.set_tag('error.type', type(error).__name__)
        span.set_tag('error.payment_id', payment_id)
    raise
```

**Performance Profiling:**

```python
# Enable continuous profiler
import ddtrace.profiling.auto

# Or manual profiling
from ddtrace.profiling import Profiler

profiler = Profiler()
profiler.start()

# Your application code

profiler.stop()
```

---

### New Relic APM

**Installation:**

```python
# requirements.txt
newrelic>=9.0.0

# Initialize with config file
newrelic-admin run-program gunicorn app:app

# Or programmatically
import newrelic.agent
newrelic.agent.initialize('newrelic.ini')
```

**Configuration:**

```ini
# newrelic.ini
[newrelic]
license_key = YOUR_LICENSE_KEY
app_name = Payment API (Production)

monitor_mode = true
log_level = info

distributed_tracing.enabled = true
span_events.enabled = true

transaction_tracer.enabled = true
transaction_tracer.transaction_threshold = 0.5
transaction_tracer.record_sql = obfuscated
transaction_tracer.stack_trace_threshold = 0.5

error_collector.enabled = true
error_collector.ignore_status_codes = 404
```

**Custom Instrumentation:**

```python
import newrelic.agent

@newrelic.agent.function_trace()
def process_payment(payment_id):
    """Traced function."""
    return payment_service.process(payment_id)

# Add custom attributes
newrelic.agent.add_custom_attribute('payment_id', payment_id)
newrelic.agent.add_custom_attribute('user_tier', 'premium')

# Record custom events
newrelic.agent.record_custom_event('PaymentProcessed', {
    'payment_id': payment_id,
    'amount': payment.amount,
    'currency': payment.currency,
    'processing_time_ms': elapsed_time
})

# Background tasks
@newrelic.agent.background_task(name='process_refunds')
def process_refunds_batch():
    """Background task tracking."""
    pass
```

---

### Prometheus + Grafana

**Installation:**

```python
# requirements.txt
prometheus-client>=0.19.0

# Initialize metrics
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
payment_requests_total = Counter(
    'payment_requests_total',
    'Total payment requests',
    ['method', 'status']
)

payment_duration_seconds = Histogram(
    'payment_duration_seconds',
    'Payment processing duration',
    ['method'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

active_payments = Gauge(
    'active_payments',
    'Number of payments currently processing'
)

# Start metrics server
start_http_server(8000)
```

**Instrumentation:**

```python
import time
from functools import wraps

def track_metrics(method_name):
    """Decorator to track metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            active_payments.inc()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                payment_requests_total.labels(
                    method=method_name,
                    status='success'
                ).inc()
                return result

            except Exception as e:
                payment_requests_total.labels(
                    method=method_name,
                    status='error'
                ).inc()
                raise

            finally:
                duration = time.time() - start_time
                payment_duration_seconds.labels(
                    method=method_name
                ).observe(duration)
                active_payments.dec()

        return wrapper
    return decorator

@track_metrics('process_payment')
def process_payment(payment_id):
    """Process payment with metrics tracking."""
    return payment_service.process(payment_id)
```

**Prometheus Configuration:**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'payment-api'
    static_configs:
      - targets: ['payment-api:8000']
    metrics_path: '/metrics'

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

---

### AWS X-Ray

**Installation:**

```python
# requirements.txt
aws-xray-sdk>=2.12.0

# Initialize
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.ext.flask.middleware import XRayMiddleware

app = Flask(__name__)
XRayMiddleware(app, xray_recorder)
```

**Instrumentation:**

```python
from aws_xray_sdk.core import xray_recorder

@xray_recorder.capture('process_payment')
def process_payment(payment_id):
    """Traced with X-Ray."""
    # Add metadata
    xray_recorder.current_subsegment().put_metadata('payment_id', payment_id)
    xray_recorder.current_subsegment().put_annotation('user_tier', 'premium')

    # Capture exceptions
    try:
        result = payment_service.process(payment_id)
    except Exception as e:
        xray_recorder.current_subsegment().add_exception(e, sys.exc_info())
        raise

    return result

# SQL queries (automatic with patching)
from aws_xray_sdk.core import patch_all
patch_all()

# Manual subsegments
with xray_recorder.in_subsegment('external_api_call') as subsegment:
    response = requests.post(PAYMENT_GATEWAY_URL, data=payload)
    subsegment.put_http_meta('response_code', response.status_code)
```

---

## Distributed Tracing Setup

### OpenTelemetry (Unified Standard)

**Installation:**

```python
# requirements.txt
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-instrumentation-flask>=0.41b0
opentelemetry-instrumentation-requests>=0.41b0
opentelemetry-exporter-jaeger>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
```

**Configuration:**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Initialize tracer provider
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure exporter
otlp_exporter = OTLPSpanExporter(
    endpoint="otel-collector:4317",
    insecure=True
)

# Add span processor
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

# Auto-instrument frameworks
FlaskInstrumentor().instrument_app(app)
RequestsInstrumentor().instrument()
```

**Manual Instrumentation:**

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

def process_payment(payment_id):
    """Process payment with OpenTelemetry tracing."""
    with tracer.start_as_current_span("process_payment") as span:
        # Add attributes
        span.set_attribute("payment.id", payment_id)
        span.set_attribute("payment.processor", "stripe")

        try:
            # Business logic
            payment = fetch_payment(payment_id)
            span.set_attribute("payment.amount", payment.amount)
            span.set_attribute("payment.currency", payment.currency)

            result = charge_payment(payment)

            # Mark success
            span.set_status(Status(StatusCode.OK))
            return result

        except Exception as e:
            # Record exception
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

def fetch_payment(payment_id):
    """Fetch payment with nested span."""
    with tracer.start_as_current_span("fetch_payment") as span:
        span.set_attribute("db.system", "postgresql")
        span.set_attribute("db.statement", "SELECT * FROM payments WHERE id = ?")
        return db.query(payment_id)
```

**Context Propagation:**

```python
from opentelemetry.propagate import inject, extract

# Inject trace context into outgoing HTTP request
headers = {}
inject(headers)
requests.post(EXTERNAL_API_URL, headers=headers, data=payload)

# Extract trace context from incoming request
context = extract(request.headers)
with tracer.start_as_current_span("handle_request", context=context):
    # Handle request with parent trace context
    pass
```

---

### Jaeger

**Docker Setup:**

```bash
# Start Jaeger all-in-one
docker run -d --name jaeger \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 14250:14250 \
  -p 9411:9411 \
  jaegertracing/all-in-one:latest

# Access UI: http://localhost:16686
```

**Python Integration:**

```python
from jaeger_client import Config

def init_jaeger_tracer(service_name='payment-api'):
    """Initialize Jaeger tracer."""
    config = Config(
        config={
            'sampler': {
                'type': 'const',
                'param': 1,  # Sample 100% of traces
            },
            'logging': True,
            'reporter_batch_size': 1,
        },
        service_name=service_name,
        validate=True,
    )

    return config.initialize_tracer()

tracer = init_jaeger_tracer()

# Use tracer
with tracer.start_span('process_payment') as span:
    span.set_tag('payment.id', payment_id)
    span.log_kv({'event': 'payment_started', 'timestamp': time.time()})

    # Nested span
    with tracer.start_span('charge_card', child_of=span) as child_span:
        result = payment_gateway.charge(payment_id)

    span.log_kv({'event': 'payment_completed', 'result': result})
```

---

### Zipkin

**Docker Setup:**

```bash
# Start Zipkin
docker run -d -p 9411:9411 openzipkin/zipkin

# Access UI: http://localhost:9411
```

**Python Integration:**

```python
from py_zipkin.zipkin import zipkin_span, ZipkinAttrs
from py_zipkin.transport import SimpleHTTPTransport

def http_transport(encoded_span):
    """Send spans to Zipkin."""
    requests.post(
        'http://localhost:9411/api/v2/spans',
        data=encoded_span,
        headers={'Content-Type': 'application/json'}
    )

@zipkin_span(service_name='payment-api', span_name='process_payment')
def process_payment(payment_id):
    """Traced function."""
    return payment_service.process(payment_id)

# In Flask app
@app.route('/api/payment', methods=['POST'])
@zipkin_span(
    service_name='payment-api',
    span_name='payment_endpoint',
    transport_handler=http_transport,
    port=5000,
    sample_rate=100.0  # 100% sampling
)
def payment_endpoint():
    payment_id = request.json.get('payment_id')
    return process_payment(payment_id)
```

---

## Logging Best Practices

### Structured Logging

**Python (structlog):**

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
def process_payment(payment_id, amount, currency):
    """Process payment with structured logging."""
    log = logger.bind(
        payment_id=payment_id,
        amount=amount,
        currency=currency,
        service='payment-api',
        environment='production'
    )

    log.info("payment_processing_started")

    try:
        result = payment_gateway.charge(payment_id, amount)
        log.info(
            "payment_processing_completed",
            transaction_id=result.transaction_id,
            status=result.status,
            processing_time_ms=result.duration_ms
        )
        return result

    except PaymentError as e:
        log.error(
            "payment_processing_failed",
            error_type=type(e).__name__,
            error_message=str(e),
            error_code=e.code
        )
        raise
```

**Output Example:**

```json
{
  "event": "payment_processing_started",
  "payment_id": "pay_abc123",
  "amount": 99.99,
  "currency": "USD",
  "service": "payment-api",
  "environment": "production",
  "timestamp": "2025-01-15T14:32:45.123456Z",
  "level": "info",
  "logger": "payment_service"
}
```

---

### Log Aggregation (ELK Stack)

**Logstash Configuration:**

```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [service] == "payment-api" {
    json {
      source => "message"
    }

    # Extract trace_id for correlation
    mutate {
      add_field => {
        "[@metadata][trace_id]" => "%{trace_id}"
      }
    }

    # Parse timestamps
    date {
      match => [ "timestamp", "ISO8601" ]
      target => "@timestamp"
    }

    # Grok parsing for non-JSON logs
    grok {
      match => {
        "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}"
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "payment-api-logs-%{+YYYY.MM.dd}"
  }
}
```

**Filebeat Configuration:**

```yaml
# filebeat.yml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/payment-api/*.log
    fields:
      service: payment-api
      environment: production
    json.keys_under_root: true
    json.add_error_key: true

output.logstash:
  hosts: ["logstash:5044"]

processors:
  - add_host_metadata: ~
  - add_cloud_metadata: ~
  - add_docker_metadata: ~
```

---

### Log Levels and When to Use Them

```python
import logging

# DEBUG: Detailed information for diagnosing problems
logger.debug(
    "Database query executed",
    query=query,
    parameters=parameters,
    execution_time_ms=elapsed_time
)

# INFO: Confirmation that things are working as expected
logger.info(
    "Payment processed successfully",
    payment_id=payment_id,
    amount=amount,
    transaction_id=transaction_id
)

# WARNING: Unexpected behavior that doesn't prevent operation
logger.warning(
    "Payment processing slow",
    payment_id=payment_id,
    processing_time_ms=processing_time,
    threshold_ms=1000
)

# ERROR: Error that prevented specific operation
logger.error(
    "Payment processing failed",
    payment_id=payment_id,
    error_type=type(error).__name__,
    error_message=str(error),
    exc_info=True  # Include stack trace
)

# CRITICAL: Serious error affecting entire system
logger.critical(
    "Database connection pool exhausted",
    active_connections=pool.active_count,
    max_connections=pool.max_size,
    waiting_requests=pool.waiting_count
)
```

---

## Monitoring Dashboards and Alerting

### Grafana Dashboard Configuration

**Dashboard JSON Example:**

```json
{
  "dashboard": {
    "title": "Payment API - Production Monitoring",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(payment_requests_total[5m])",
            "legendFormat": "{{ method }} - {{ status }}"
          }
        ]
      },
      {
        "id": 2,
        "title": "Error Rate %",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(payment_requests_total{status=\"error\"}[5m])) / sum(rate(payment_requests_total[5m])) * 100",
            "legendFormat": "Error Rate"
          }
        ],
        "alert": {
          "name": "High Error Rate",
          "conditions": [
            {
              "evaluator": {
                "type": "gt",
                "params": [1.0]
              },
              "operator": { "type": "and" },
              "query": { "params": ["A", "5m", "now"] },
              "type": "query"
            }
          ],
          "for": "2m",
          "frequency": "1m",
          "handler": 1,
          "message": "Payment API error rate exceeds 1%",
          "noDataState": "no_data",
          "notifications": [
            {"uid": "pagerduty-integration"}
          ]
        }
      },
      {
        "id": 3,
        "title": "P95 Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(payment_duration_seconds_bucket[5m]))",
            "legendFormat": "P95 Latency"
          }
        ]
      },
      {
        "id": 4,
        "title": "Active Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "db_connection_pool_active",
            "legendFormat": "Active Connections"
          },
          {
            "expr": "db_connection_pool_max",
            "legendFormat": "Max Connections"
          }
        ]
      }
    ]
  }
}
```

---

### Alert Rules (Prometheus)

```yaml
# alerts.yml
groups:
  - name: payment-api
    interval: 30s
    rules:
      # High error rate alert
      - alert: HighErrorRate
        expr: |
          (sum(rate(payment_requests_total{status="error"}[5m]))
          / sum(rate(payment_requests_total[5m]))) > 0.01
        for: 2m
        labels:
          severity: critical
          service: payment-api
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 1%)"
          runbook: "https://wiki.company.com/runbooks/high-error-rate"

      # High latency alert
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            rate(payment_duration_seconds_bucket[5m])
          ) > 0.5
        for: 5m
        labels:
          severity: warning
          service: payment-api
        annotations:
          summary: "P95 latency high"
          description: "P95 latency is {{ $value }}s (threshold: 0.5s)"

      # Database connection pool alert
      - alert: DatabaseConnectionPoolHighUtilization
        expr: |
          (db_connection_pool_active / db_connection_pool_max) > 0.8
        for: 2m
        labels:
          severity: warning
          service: payment-api
        annotations:
          summary: "Database connection pool utilization high"
          description: "Pool is {{ $value | humanizePercentage }} full"

      # Memory usage alert
      - alert: HighMemoryUsage
        expr: |
          (process_resident_memory_bytes / 1024 / 1024 / 1024) > 3.5
        for: 5m
        labels:
          severity: warning
          service: payment-api
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanize }}GB (threshold: 3.5GB)"

      # Service availability alert
      - alert: ServiceDown
        expr: up{job="payment-api"} == 0
        for: 1m
        labels:
          severity: critical
          service: payment-api
        annotations:
          summary: "Payment API is down"
          description: "Payment API instance {{ $labels.instance }} is unreachable"
          runbook: "https://wiki.company.com/runbooks/service-down"
```

---

## Production-Safe Debugging Techniques

### Feature Flags

**LaunchDarkly Integration:**

```python
import ldclient
from ldclient.config import Config

# Initialize client
ldclient.set_config(Config("YOUR_SDK_KEY"))
ld_client = ldclient.get()

def process_payment(payment_id, user_id):
    """Process payment with feature flag."""
    user = {"key": user_id, "custom": {"tier": "premium"}}

    # Check feature flag
    use_new_processor = ld_client.variation(
        "new-payment-processor",
        user,
        False  # Default value
    )

    if use_new_processor:
        logger.info("Using new payment processor", feature_flag=True)
        return new_payment_processor.process(payment_id)
    else:
        logger.info("Using legacy payment processor", feature_flag=False)
        return legacy_payment_processor.process(payment_id)
```

**Custom Feature Flag Implementation:**

```python
class FeatureFlags:
    """Simple feature flag implementation."""

    def __init__(self, config_backend):
        self.config = config_backend
        self.cache = {}
        self.cache_ttl = 60  # seconds

    def is_enabled(self, flag_name, user_id=None, context=None):
        """Check if feature flag is enabled."""
        flag_config = self._get_flag_config(flag_name)

        if not flag_config:
            return False

        # Global rollout percentage
        if 'rollout_percent' in flag_config:
            user_hash = hash(f"{flag_name}:{user_id}")
            if (user_hash % 100) < flag_config['rollout_percent']:
                return True

        # User whitelist
        if user_id and user_id in flag_config.get('whitelist', []):
            return True

        # Context-based rules
        if context and self._evaluate_rules(flag_config.get('rules', []), context):
            return True

        return flag_config.get('default', False)

    def _get_flag_config(self, flag_name):
        """Get flag configuration with caching."""
        if flag_name in self.cache:
            cached_value, cached_time = self.cache[flag_name]
            if time.time() - cached_time < self.cache_ttl:
                return cached_value

        config = self.config.get_flag(flag_name)
        self.cache[flag_name] = (config, time.time())
        return config
```

---

### Dark Launches

**Implementation:**

```python
def process_payment_with_dark_launch(payment_id):
    """Process payment with dark launch of new implementation."""

    # Always run primary implementation
    try:
        primary_result = legacy_payment_processor.process(payment_id)
    except Exception as e:
        logger.error("Primary processor failed", error=str(e))
        raise

    # Dark launch: run new implementation in parallel
    if feature_flags.is_enabled('dark-launch-new-processor', payment_id):
        try:
            # Run new implementation asynchronously
            executor.submit(
                _dark_launch_new_processor,
                payment_id,
                primary_result
            )
        except Exception as e:
            # Don't fail request if dark launch fails
            logger.warning("Dark launch failed", error=str(e))

    # Always return primary result
    return primary_result

def _dark_launch_new_processor(payment_id, expected_result):
    """Execute new processor and compare results."""
    try:
        new_result = new_payment_processor.process(payment_id)

        # Compare results
        if new_result != expected_result:
            logger.warning(
                "Dark launch result mismatch",
                payment_id=payment_id,
                primary_result=expected_result,
                new_result=new_result
            )
        else:
            logger.info(
                "Dark launch result matches",
                payment_id=payment_id
            )

    except Exception as e:
        logger.error(
            "Dark launch exception",
            payment_id=payment_id,
            error=str(e),
            exc_info=True
        )
```

---

### Dynamic Log Level Adjustment

**Implementation:**

```python
import logging
from flask import Flask, request

app = Flask(__name__)

# Global log level registry
log_levels = {
    'payment_service': logging.INFO,
    'database': logging.WARNING,
}

def get_logger(name):
    """Get logger with dynamic level."""
    logger = logging.getLogger(name)
    logger.setLevel(log_levels.get(name, logging.INFO))
    return logger

@app.route('/admin/loglevel', methods=['POST'])
@require_admin_auth
def set_log_level():
    """Dynamically adjust log level at runtime."""
    logger_name = request.json['logger']
    level = request.json['level'].upper()

    if level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        return {'error': 'Invalid log level'}, 400

    # Update log level
    log_levels[logger_name] = getattr(logging, level)
    logging.getLogger(logger_name).setLevel(log_levels[logger_name])

    logger.info(
        "Log level changed",
        logger=logger_name,
        new_level=level,
        admin_user=request.user.email
    )

    return {'status': 'success', 'logger': logger_name, 'level': level}
```

**Usage:**

```bash
# Temporarily enable DEBUG logging for payment_service
curl -X POST https://api.example.com/admin/loglevel \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"logger": "payment_service", "level": "DEBUG"}'

# Collect debug logs for 5 minutes, then reset
sleep 300

curl -X POST https://api.example.com/admin/loglevel \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"logger": "payment_service", "level": "INFO"}'
```

---

### Conditional Debugging

**Implementation:**

```python
def process_payment(payment_id):
    """Process payment with conditional debugging."""

    # Check if debugging enabled for this payment
    debug_enabled = (
        feature_flags.is_enabled('debug-payment-processor') or
        payment_id in get_debug_payment_ids() or
        request.headers.get('X-Debug-Mode') == 'true'
    )

    if debug_enabled:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled for payment", payment_id=payment_id)

    try:
        if debug_enabled:
            logger.debug("Fetching payment from database", payment_id=payment_id)

        payment = db.get_payment(payment_id)

        if debug_enabled:
            logger.debug("Payment fetched", payment=payment.to_dict())

        result = charge_payment(payment)

        if debug_enabled:
            logger.debug("Payment charged", result=result.to_dict())

        return result

    finally:
        if debug_enabled:
            logger.setLevel(logging.INFO)
```

---

### Sampling and Profiling

**Implementation:**

```python
import cProfile
import pstats
from io import StringIO

def profile_payment_processing():
    """Profile payment processing with sampling."""

    # Sample 1% of requests
    if random.random() > 0.01:
        return process_payment_normal()

    # Profile this request
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        result = process_payment_normal()
    finally:
        profiler.disable()

        # Generate profile report
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions

        # Log profile data
        logger.info(
            "Payment processing profile",
            profile_data=s.getvalue(),
            payment_id=payment_id
        )

    return result
```

---

## Observability Tool Comparison Matrix

| Feature | Datadog | New Relic | Prometheus + Grafana | AWS X-Ray | OpenTelemetry |
|---------|---------|-----------|---------------------|-----------|---------------|
| **APM** | ✅ Built-in | ✅ Built-in | ❌ Requires custom metrics | ✅ Built-in | ✅ With exporters |
| **Distributed Tracing** | ✅ Native | ✅ Native | ⚠️ Via Tempo | ✅ Native | ✅ Standard |
| **Log Aggregation** | ✅ Native | ✅ Native | ⚠️ Via Loki | ⚠️ Via CloudWatch | ⚠️ External tools |
| **Custom Metrics** | ✅ StatsD/DogStatsD | ✅ Custom events | ✅ Prometheus format | ⚠️ Limited | ✅ Via exporters |
| **Alerting** | ✅ Advanced | ✅ Advanced | ✅ Prometheus AlertManager | ✅ CloudWatch Alarms | ❌ External tools |
| **Dashboards** | ✅ Built-in | ✅ Built-in | ✅ Grafana | ⚠️ CloudWatch Dashboards | ❌ External tools |
| **Cost** | 💰💰💰 High | 💰💰💰 High | 💰 Open-source (infra costs) | 💰💰 AWS-based | 💰 Open-source |
| **Cloud Vendor Lock-in** | ❌ No | ❌ No | ❌ No | ✅ AWS only | ❌ No |
| **Ease of Setup** | ⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐ Easy | ⭐⭐ Moderate | ⭐⭐⭐ AWS Easy | ⭐⭐⭐ Moderate |
| **Language Support** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good | ⭐⭐⭐ AWS-focused | ⭐⭐⭐⭐⭐ Excellent |
| **Community** | ⭐⭐⭐ Good | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ Limited | ⭐⭐⭐⭐ Growing |
| **Best For** | All-in-one SaaS | All-in-one SaaS | Cost-conscious teams | AWS-native apps | Vendor-neutral standard |

---

## Quick Reference Guide

### When to Use Each Observability Tool

| Scenario | Recommended Tool |
|----------|------------------|
| Full-stack SaaS observability | Datadog or New Relic |
| Cost-conscious startup | Prometheus + Grafana |
| AWS-native application | AWS X-Ray |
| Vendor-neutral strategy | OpenTelemetry |
| Microservices tracing | OpenTelemetry + Jaeger |
| Kubernetes monitoring | Prometheus + Grafana |
| Production debugging | Feature Flags + Dynamic Logging |

### Observability Maturity Levels

**Level 1: Basic Monitoring**
- Application metrics (request rate, latency, errors)
- Infrastructure metrics (CPU, memory, disk)
- Basic alerting on thresholds
- Manual log inspection

**Level 2: Advanced Monitoring**
- Distributed tracing
- Structured logging
- Log aggregation
- Custom business metrics
- Automated alerting with runbooks

**Level 3: Observability Excellence**
- Full distributed tracing across all services
- Correlation between logs, metrics, and traces
- Automated anomaly detection
- Production-safe debugging tools
- SLI/SLO-based alerting
- Continuous profiling

---

## Integration Checklist

### Production Observability Readiness

- [ ] APM installed and configured
- [ ] Distributed tracing instrumented
- [ ] Structured logging implemented
- [ ] Log aggregation configured
- [ ] Key metrics exported (request rate, latency, errors)
- [ ] Dashboards created for all services
- [ ] Alerts configured with runbooks
- [ ] Feature flags implemented
- [ ] Dynamic log level adjustment available
- [ ] Production-safe debugging techniques in place
- [ ] SLI/SLO definitions created
- [ ] Incident response procedures documented
