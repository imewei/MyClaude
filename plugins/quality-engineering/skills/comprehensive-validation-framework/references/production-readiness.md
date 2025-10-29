# Production Readiness Reference

Comprehensive checklist for deploying systems to production.

## Pre-Launch Checklist

### Configuration

- [ ] No hardcoded configuration values
- [ ] Environment-specific configs separated (dev/staging/prod)
- [ ] Secrets managed securely (vault, AWS Secrets Manager)
- [ ] Configuration validation on startup
- [ ] Graceful handling of missing config

### Security

- [ ] All security scans pass
- [ ] Dependencies up to date
- [ ] No exposed secrets in code/logs
- [ ] Authentication/authorization tested
- [ ] Rate limiting configured
- [ ] Security headers set (CSP, HSTS, etc.)

### Observability

- [ ] Logging implemented at appropriate levels
- [ ] Structured logging (JSON format)
- [ ] Metrics/telemetry instrumented
- [ ] Distributed tracing configured
- [ ] Health check endpoint
- [ ] Readiness/liveness probes (Kubernetes)

### Reliability

- [ ] Error handling comprehensive
- [ ] Circuit breakers configured
- [ ] Timeouts set on all external calls
- [ ] Retries with exponential backoff
- [ ] Connection pooling configured
- [ ] Resource cleanup (connections, file handles)

### Performance

- [ ] Load testing completed
- [ ] Database indexes verified
- [ ] Caching strategy implemented
- [ ] CDN configured (if applicable)
- [ ] Bundle sizes optimized (frontend)

### Data

- [ ] Database migrations tested
- [ ] Backup strategy defined
- [ ] Data retention policy implemented
- [ ] PII handling complies with regulations (GDPR, CCPA)

### Deployment

- [ ] CI/CD pipeline working
- [ ] Smoke tests pass after deployment
- [ ] Rollback plan documented and tested
- [ ] Feature flags for gradual rollout
- [ ] Deployment automation tested

### Documentation

- [ ] API documentation complete
- [ ] Runbooks for common operations
- [ ] Incident response procedures
- [ ] Architecture diagrams
- [ ] Deployment guide

---

## Health Check Implementation

### HTTP Health Endpoint

**Python/FastAPI**:
```python
from fastapi import FastAPI, Response

@app.get("/health")
def health_check():
    """Basic liveness check."""
    return {"status": "healthy"}

@app.get("/ready")
async def readiness_check():
    """Check dependencies are ready."""
    checks = {
        "database": await check_database(),
        "cache": await check_redis(),
        "queue": await check_message_queue()
    }

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503

    return Response(
        content=json.dumps({"status": "ready" if all_healthy else "not_ready", "checks": checks}),
        status_code=status_code
    )
```

**Node.js/Express**:
```javascript
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'healthy' });
});

app.get('/ready', async (req, res) => {
  const checks = {
    database: await checkDatabase(),
    cache: await checkRedis()
  };

  const allHealthy = Object.values(checks).every(v => v);
  res.status(allHealthy ? 200 : 503).json({
    status: allHealthy ? 'ready' : 'not_ready',
    checks
  });
});
```

### Kubernetes Probes

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: app
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

## Structured Logging

### Python

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "user_login",
    user_id=user.id,
    username=user.username,
    ip_address=request.remote_addr,
    user_agent=request.headers.get('User-Agent')
)
```

### JavaScript

```javascript
const winston = require('winston');

const logger = winston.createLogger({
  format: winston.format.json(),
  transports: [new winston.transports.Console()]
});

logger.info({
  event: 'user_login',
  userId: user.id,
  username: user.username,
  ipAddress: req.ip
});
```

### Log Levels

- **DEBUG**: Detailed diagnostic info
- **INFO**: Confirmation that things are working
- **WARNING**: Unexpected event, but system continues
- **ERROR**: Error that prevented specific operation
- **CRITICAL**: System failure, immediate attention needed

**What to log**:
- Authentication attempts (success/failure)
- Authorization failures
- Input validation failures
- Performance issues
- External API calls
- Database errors

**What NOT to log**:
- Passwords or secrets
- Credit card numbers
- PII (unless encrypted)
- Full request bodies

---

## Metrics and Monitoring

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration'
)

# Business metrics
active_users = Gauge('active_users', 'Number of active users')
orders_total = Counter('orders_total', 'Total orders processed')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    with request_duration.time():
        response = await call_next(request)

    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    return response
```

### Datadog Example

```python
from datadog import statsd

# Increment counter
statsd.increment('page.views', tags=['page:home'])

# Timing
with statsd.timed('database.query.duration', tags=['query:select_users']):
    users = db.query(User).all()

# Gauge
statsd.gauge('queue.size', len(queue), tags=['queue:default'])
```

---

## Error Tracking

### Sentry Integration

**Python**:
```python
import sentry_sdk

sentry_sdk.init(
    dsn="https://your-dsn@sentry.io/project",
    environment="production",
    release="myapp@1.0.0",
    traces_sample_rate=0.1
)

# Automatic error capture
try:
    risky_operation()
except Exception as e:
    sentry_sdk.capture_exception(e)

# Add context
with sentry_sdk.configure_scope() as scope:
    scope.set_user({"id": user.id, "email": user.email})
    scope.set_tag("transaction_type", "checkout")
```

**JavaScript**:
```javascript
import * as Sentry from "@sentry/node";

Sentry.init({
  dsn: "https://your-dsn@sentry.io/project",
  environment: "production",
  tracesSampleRate: 0.1
});

// Automatic error capture
app.use(Sentry.Handlers.errorHandler());

// Manual capture
try {
  riskyOperation();
} catch (error) {
  Sentry.captureException(error);
}
```

---

## Graceful Shutdown

### Python

```python
import signal
import sys

def graceful_shutdown(signum, frame):
    logger.info("Shutdown signal received, cleaning up...")

    # Close database connections
    db.close()

    # Finish processing current requests
    app.shutdown()

    logger.info("Shutdown complete")
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)
```

### Node.js

```javascript
process.on('SIGTERM', async () => {
  console.log('Shutdown signal received, cleaning up...');

  // Stop accepting new connections
  server.close(() => {
    console.log('HTTP server closed');
  });

  // Close database connections
  await db.close();

  // Finish processing
  await queue.drain();

  process.exit(0);
});
```

---

## Circuit Breaker Pattern

### Python (circuitbreaker)

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_external_api():
    response = requests.get('https://api.example.com/data', timeout=5)
    response.raise_for_status()
    return response.json()

try:
    data = call_external_api()
except CircuitBreakerError:
    # Circuit is open, use fallback
    data = get_cached_data()
```

### JavaScript (opossum)

```javascript
const CircuitBreaker = require('opossum');

const options = {
  timeout: 3000,
  errorThresholdPercentage: 50,
  resetTimeout: 30000
};

const breaker = new CircuitBreaker(fetchData, options);

breaker.fallback(() => getCachedData());

breaker.on('open', () => console.log('Circuit breaker opened'));
breaker.on('halfOpen', () => console.log('Circuit breaker half-open'));
breaker.on('close', () => console.log('Circuit breaker closed'));

const data = await breaker.fire();
```

---

## Deployment Strategies

### Blue-Green Deployment

```
Blue (v1.0) ← 100% traffic
Green (v1.1) ← 0% traffic

Deploy to Green → Run smoke tests → Switch traffic → Keep Blue for rollback
```

### Canary Deployment

```
v1.0 ← 95% traffic
v1.1 ← 5% traffic (canary)

Monitor metrics → Gradually increase → 100% on v1.1
```

### Feature Flags

```python
from unleash import UnleashClient

client = UnleashClient(
    url="https://unleash.example.com/api",
    app_name="myapp"
)

if client.is_enabled("new_checkout_flow", context={"userId": user.id}):
    return new_checkout()
else:
    return old_checkout()
```

---

## Runbook Template

```markdown
# Service Name Runbook

## Overview
Brief description of the service and its purpose.

## Architecture
- Dependencies: Database, Redis, External APIs
- Scaling: Horizontal (3 instances minimum)

## Common Operations

### Restart Service
```bash
kubectl rollout restart deployment/myapp
```

### View Logs
```bash
kubectl logs -f deployment/myapp --tail=100
```

### Check Health
```bash
curl https://api.example.com/health
```

## Common Issues

### Issue: High CPU Usage

**Symptoms**: CPU >80%, slow response times

**Diagnosis**:
1. Check metrics: `kubectl top pods`
2. Review logs for errors

**Resolution**:
- Scale up: `kubectl scale deployment/myapp --replicas=5`
- If persists, investigate memory leaks

### Issue: Database Connection Failures

**Symptoms**: "Connection refused" errors

**Diagnosis**:
1. Check database health
2. Verify connection pool settings

**Resolution**:
- Restart database proxy
- Increase connection pool size

## Incident Response

1. **Acknowledge**: Update status page
2. **Diagnose**: Check logs, metrics, traces
3. **Mitigate**: Roll back or hotfix
4. **Resolve**: Deploy permanent fix
5. **Post-Mortem**: Document lessons learned

## Contacts

- On-call: PagerDuty rotation
- Escalation: team-leads@example.com
```

---

## References

- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)
- [12 Factor App](https://12factor.net/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
