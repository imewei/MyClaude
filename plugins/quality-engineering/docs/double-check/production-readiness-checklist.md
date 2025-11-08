# Production Readiness Checklist

Comprehensive checklist for validating applications are production-ready with proper configuration management, observability, and deployment strategies.

## Overview

Before deploying to production, ensure your application meets enterprise standards for:
- Configuration management and secrets
- Observability (logging, metrics, tracing)
- Infrastructure as Code validation
- Deployment strategies and rollback plans
- Health checks and monitoring
- Disaster recovery and business continuity

---

## 1. Configuration Management

### Environment-Based Configuration

#### Separation of Concerns

**✅ Best Practice**:
```python
# config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str
    database_pool_size: int = 10

    # Redis
    redis_url: str
    redis_max_connections: int = 50

    # API Keys (loaded from secrets)
    stripe_api_key: str
    sendgrid_api_key: str

    # Feature Flags
    enable_new_feature: bool = False

    # Environment
    environment: str = "production"
    debug: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

**❌ Anti-pattern**:
```python
# Hard-coded configuration
DATABASE_URL = "postgresql://localhost/mydb"
API_KEY = "sk_live_abc123"  # Secret in code!
DEBUG = True  # Debug mode in production!
```

### Environment Variables

**Required environment variables**:
```bash
# .env.example (committed to git)
DATABASE_URL=postgresql://user:pass@host:5432/dbname
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-here
STRIPE_API_KEY=sk_live_...
SENDGRID_API_KEY=SG...

# Optional with defaults
LOG_LEVEL=INFO
MAX_WORKERS=4
ENABLE_PROFILING=false
```

**Validation on startup**:
```python
import sys

def validate_config():
    required = [
        'DATABASE_URL',
        'SECRET_KEY',
        'STRIPE_API_KEY',
    ]

    missing = [var for var in required if not os.getenv(var)]

    if missing:
        print(f"ERROR: Missing required environment variables: {missing}")
        sys.exit(1)

validate_config()
```

### Secrets Management

#### AWS Secrets Manager

```python
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-east-1')

    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response['SecretString'])
    except ClientError as e:
        raise Exception(f"Failed to retrieve secret: {e}")

# Usage
secrets = get_secret("prod/myapp/database")
DATABASE_URL = secrets['url']
DATABASE_PASSWORD = secrets['password']
```

#### HashiCorp Vault

```python
import hvac

client = hvac.Client(url='https://vault.example.com')
client.token = os.getenv('VAULT_TOKEN')

# Read secret
secret = client.secrets.kv.v2.read_secret_version(
    path='myapp/database',
    mount_point='secret'
)

DATABASE_URL = secret['data']['data']['url']
```

#### Kubernetes Secrets

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
stringData:
  database-url: postgresql://user:pass@host:5432/db
  stripe-api-key: sk_live_...
```

```python
# Access in pod
DATABASE_URL = os.getenv('DATABASE_URL')
# Mounted from secret via env or volume
```

### Configuration Checklist

- [ ] No secrets in code or git repository
- [ ] Environment variables documented in `.env.example`
- [ ] Configuration validated on startup
- [ ] Different configs for dev/staging/production
- [ ] Secrets rotated regularly (90-day max)
- [ ] Secrets accessed via secure vault (not env vars in prod)
- [ ] Feature flags for gradual rollouts
- [ ] Configuration changes don't require code deploy

---

## 2. Observability

### Logging Best Practices

#### Structured Logging

**Python (structlog)**:
```python
import structlog

logger = structlog.get_logger()

# Structured log entries
logger.info(
    "user_login",
    user_id=user.id,
    email=user.email,
    ip_address=request.remote_addr,
    user_agent=request.user_agent.string
)

# Automatic context
logger.bind(request_id=request_id)
logger.info("processing_payment", amount=100.00, currency="USD")
```

**JavaScript (winston)**:
```javascript
const winston = require('winston');

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  defaultMeta: { service: 'user-service' },
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
  ],
});

logger.info('User login', {
  userId: user.id,
  email: user.email,
  ipAddress: req.ip,
});
```

#### Log Levels

```python
# DEBUG: Detailed information for diagnosing problems
logger.debug("Cache miss for key: user_profile_{user_id}")

# INFO: Confirmation that things are working
logger.info("User {user_id} successfully logged in")

# WARNING: Something unexpected happened, but app continues
logger.warning("API rate limit approaching", usage=95, limit=100)

# ERROR: Serious problem, operation failed
logger.error("Failed to process payment", error=str(e), user_id=user.id)

# CRITICAL: System is unusable
logger.critical("Database connection pool exhausted")
```

#### What to Log

**✅ Log**:
- User authentication events
- Authorization failures
- API requests/responses (sanitized)
- Database queries (slow queries)
- External API calls
- Background job status
- Errors and exceptions with context
- Performance metrics

**❌ Don't Log**:
- Passwords or API keys
- Credit card numbers
- Social Security Numbers
- Personally Identifiable Information (PII) in plain text
- Full request/response bodies by default

### Metrics Collection

#### Prometheus Metrics (Python)

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# Request counter
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Request duration histogram
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Active connections gauge
active_connections = Gauge(
    'active_database_connections',
    'Number of active database connections'
)

# Usage
@app.route('/api/users')
def get_users():
    with http_request_duration_seconds.labels('GET', '/api/users').time():
        users = User.query.all()
        http_requests_total.labels('GET', '/api/users', 200).inc()
        return users
```

#### Key Metrics to Track

**Application Metrics**:
- Request rate (requests/second)
- Error rate (errors/second, error percentage)
- Request duration (p50, p95, p99)
- Queue depth
- Background job success/failure rate

**System Metrics**:
- CPU utilization (%)
- Memory usage (MB, %)
- Disk I/O (reads/writes per second)
- Network I/O (bytes in/out)
- Open file descriptors

**Business Metrics**:
- User signups
- Order completions
- Revenue per minute
- Active users
- Feature usage

### Distributed Tracing

#### OpenTelemetry (Python)

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Export to Jaeger/Tempo
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317")
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

# Instrument code
@app.route('/api/users/<user_id>')
def get_user(user_id):
    with tracer.start_as_current_span("get_user") as span:
        span.set_attribute("user.id", user_id)

        with tracer.start_as_current_span("database_query"):
            user = User.query.get(user_id)

        with tracer.start_as_current_span("fetch_user_stats"):
            stats = fetch_stats(user_id)

        return {"user": user, "stats": stats}
```

### Observability Checklist

- [ ] Structured logging in JSON format
- [ ] Log aggregation (ELK, Splunk, CloudWatch)
- [ ] Metrics collection (Prometheus, DataDog)
- [ ] Distributed tracing (Jaeger, Zipkin)
- [ ] Error tracking (Sentry, Rollbar)
- [ ] Real-time alerting configured
- [ ] Log retention policy defined (30-90 days)
- [ ] PII sanitization in logs

---

## 3. Infrastructure as Code

### Terraform Validation

```bash
# Format check
terraform fmt -check -recursive

# Validate syntax
terraform validate

# Security scanning
tfsec .

# Plan and review
terraform plan -out=tfplan

# Cost estimation
terraform cost tfplan
```

**Best practices**:
```hcl
# Use variables
variable "environment" {
  type        = string
  description = "Environment name (dev/staging/prod)"
}

# Remote state
terraform {
  backend "s3" {
    bucket = "myapp-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
    dynamodb_table = "terraform-lock"
  }
}

# Modules for reusability
module "vpc" {
  source = "./modules/vpc"
  environment = var.environment
}

# Resource tagging
resource "aws_instance" "app" {
  tags = {
    Name        = "myapp-${var.environment}"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
```

### Kubernetes Manifest Validation

```bash
# Validate syntax
kubectl apply --dry-run=client -f k8s/

# Validate against cluster
kubectl apply --dry-run=server -f k8s/

# Lint with kubeval
kubeval k8s/*.yaml

# Security scanning
kubesec scan k8s/deployment.yaml

# Policy enforcement
conftest test k8s/ --policy policy/
```

### Infrastructure Checklist

- [ ] Infrastructure as Code for all resources
- [ ] Remote state with locking (Terraform)
- [ ] Environment parity (dev matches prod)
- [ ] Automated validation in CI/CD
- [ ] Security scanning (tfsec, checkov)
- [ ] Cost monitoring and budgets
- [ ] Disaster recovery plan tested
- [ ] Infrastructure changes reviewed before apply

---

## 4. Deployment Strategies

### Blue-Green Deployment

```yaml
# Blue environment (current production)
apiVersion: v1
kind: Service
metadata:
  name: myapp-blue
spec:
  selector:
    app: myapp
    version: v1.0.0
  ports:
    - port: 80

---
# Green environment (new version)
apiVersion: v1
kind: Service
metadata:
  name: myapp-green
spec:
  selector:
    app: myapp
    version: v1.1.0
  ports:
    - port: 80

---
# Main service points to blue
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
    version: v1.0.0  # Switch to v1.1.0 to cutover
```

**Cutover process**:
1. Deploy green environment (v1.1.0)
2. Test green environment internally
3. Switch traffic to green
4. Monitor for errors
5. Rollback to blue if issues
6. Decommission blue after validation

### Canary Deployment

```yaml
# Istio VirtualService for canary
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: myapp
spec:
  hosts:
    - myapp.example.com
  http:
    - match:
        - headers:
            cookie:
              regex: ".*canary=true.*"
      route:
        - destination:
            host: myapp
            subset: v2
    - route:
        - destination:
            host: myapp
            subset: v1
          weight: 95
        - destination:
            host: myapp
            subset: v2
          weight: 5  # 5% traffic to canary
```

**Canary stages**:
1. Deploy v2 with 5% traffic
2. Monitor metrics for 1 hour
3. Increase to 25% if healthy
4. Monitor for 2 hours
5. Increase to 50%
6. Increase to 100% or rollback

### Rolling Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # Max 2 extra pods during update
      maxUnavailable: 1  # Max 1 pod down during update
  template:
    spec:
      containers:
        - name: myapp
          image: myapp:v1.1.0
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
```

### Rollback Procedures

**Kubernetes rollback**:
```bash
# Check deployment history
kubectl rollout history deployment/myapp

# Rollback to previous version
kubectl rollout undo deployment/myapp

# Rollback to specific revision
kubectl rollout undo deployment/myapp --to-revision=3

# Monitor rollback
kubectl rollout status deployment/myapp
```

**Database rollback**:
```python
# Use reversible migrations
# Alembic (Python)
alembic downgrade -1

# Django
python manage.py migrate myapp 0001_previous_migration
```

### Deployment Checklist

- [ ] Automated deployment pipeline (CI/CD)
- [ ] Deployment strategy documented (blue-green/canary/rolling)
- [ ] Rollback plan tested and documented
- [ ] Database migrations are reversible
- [ ] Feature flags for risky changes
- [ ] Smoke tests run post-deployment
- [ ] Monitoring alerts active during deployment
- [ ] Deployment time window defined (off-peak hours)
- [ ] Rollback decision criteria defined

---

## 5. Health Checks and Monitoring

### Health Check Endpoints

**Liveness Probe** (Is the app running?):
```python
@app.route('/health/live')
def liveness():
    return {"status": "alive"}, 200
```

**Readiness Probe** (Can the app serve traffic?):
```python
@app.route('/health/ready')
def readiness():
    # Check dependencies
    checks = {
        "database": check_database(),
        "redis": check_redis(),
        "external_api": check_external_api(),
    }

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503

    return {
        "status": "ready" if all_healthy else "not ready",
        "checks": checks
    }, status_code

def check_database():
    try:
        db.session.execute("SELECT 1")
        return True
    except Exception:
        return False
```

**Startup Probe** (Has the app finished starting?):
```python
@app.route('/health/startup')
def startup():
    # Check if initialization complete
    if not app_initialized:
        return {"status": "starting"}, 503

    return {"status": "started"}, 200
```

### Kubernetes Probe Configuration

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: myapp
      image: myapp:latest
      livenessProbe:
        httpGet:
          path: /health/live
          port: 8080
        initialDelaySeconds: 30
        periodSeconds: 10
        timeoutSeconds: 5
        failureThreshold: 3

      readinessProbe:
        httpGet:
          path: /health/ready
          port: 8080
        initialDelaySeconds: 10
        periodSeconds: 5
        timeoutSeconds: 3
        failureThreshold: 2

      startupProbe:
        httpGet:
          path: /health/startup
          port: 8080
        initialDelaySeconds: 0
        periodSeconds: 10
        timeoutSeconds: 3
        failureThreshold: 30  # 5 minutes max startup time
```

### Alerting Rules

**Prometheus AlertManager**:
```yaml
groups:
  - name: app_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) /
          sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # Slow response time
      - alert: SlowResponseTime
        expr: |
          histogram_quantile(0.95,
            rate(http_request_duration_seconds_bucket[5m])
          ) > 1.0
        for: 10m
        annotations:
          summary: "Slow response time"
          description: "P95 latency is {{ $value }}s"

      # Service down
      - alert: ServiceDown
        expr: up{job="myapp"} == 0
        for: 2m
        annotations:
          summary: "Service is down"
          description: "{{ $labels.instance }} is unreachable"
```

### Health Check Checklist

- [ ] Liveness probe configured (restart on failure)
- [ ] Readiness probe configured (remove from load balancer)
- [ ] Startup probe for slow-starting apps
- [ ] Health checks validate all critical dependencies
- [ ] Alerts configured for critical metrics
- [ ] On-call rotation defined
- [ ] Runbooks for common incidents
- [ ] Alert fatigue minimized (no flapping alerts)

---

## 6. Disaster Recovery Planning

### Backup Strategy

**Database backups**:
```bash
# Automated daily backups
0 2 * * * pg_dump myapp_production | gzip > /backups/myapp_$(date +\%Y\%m\%d).sql.gz

# Retention policy (keep 30 days)
find /backups -name "myapp_*.sql.gz" -mtime +30 -delete
```

**Application state backups**:
- Configuration files
- User-uploaded files (S3 versioning)
- Redis snapshots (if used for state)

**Backup checklist**:
- [ ] Automated daily backups
- [ ] Backups stored in different region/zone
- [ ] Backup restoration tested monthly
- [ ] Point-in-time recovery available (PITR)
- [ ] Backup encryption at rest
- [ ] Backup retention policy defined

### Recovery Time Objective (RTO)

**RTO targets** (time to restore service):
- Critical systems: < 1 hour
- Important systems: < 4 hours
- Non-critical systems: < 24 hours

**Recovery procedures**:
```bash
# 1. Restore database from backup
gunzip < /backups/myapp_20240101.sql.gz | psql myapp_production

# 2. Deploy last known good version
kubectl set image deployment/myapp myapp=myapp:v1.0.0

# 3. Restore application state
aws s3 sync s3://backups/redis-state/ /data/redis/

# 4. Validate health checks
curl http://myapp.example.com/health/ready
```

### Recovery Point Objective (RPO)

**RPO targets** (acceptable data loss):
- Critical data: < 5 minutes (continuous replication)
- Important data: < 1 hour (frequent backups)
- Non-critical data: < 24 hours (daily backups)

**Implementation**:
- Database replication to standby (continuous)
- Transaction log shipping (every 5 minutes)
- Full backups (daily)
- Incremental backups (hourly)

### Disaster Recovery Checklist

- [ ] RTO and RPO defined for all systems
- [ ] Disaster recovery plan documented
- [ ] DR plan tested quarterly
- [ ] Automated failover for critical systems
- [ ] Multi-region deployment for HA
- [ ] Data replication across regions
- [ ] Emergency contact list maintained
- [ ] Post-mortem process defined

---

## Production Readiness Final Checklist

### Before First Production Deployment

**Infrastructure**:
- [ ] Production environment provisioned
- [ ] Load balancer configured
- [ ] SSL/TLS certificates installed
- [ ] DNS configured
- [ ] CDN configured (if needed)
- [ ] Auto-scaling enabled

**Security**:
- [ ] Security scanning passed
- [ ] Penetration testing completed
- [ ] Access controls configured (RBAC)
- [ ] Secrets in vault (not env vars)
- [ ] HTTPS enforced
- [ ] Rate limiting enabled
- [ ] DDoS protection configured

**Data**:
- [ ] Database migrations tested
- [ ] Data migration plan (if needed)
- [ ] Backup/restore tested
- [ ] Database indexes created
- [ ] Connection pooling configured

**Observability**:
- [ ] Logging configured
- [ ] Metrics collection enabled
- [ ] Tracing configured
- [ ] Error tracking configured
- [ ] Alerts configured
- [ ] Dashboards created
- [ ] On-call rotation defined

**Performance**:
- [ ] Load testing completed
- [ ] Performance benchmarks met
- [ ] Caching configured
- [ ] Static assets optimized
- [ ] Database queries optimized

**Operations**:
- [ ] Deployment runbook created
- [ ] Rollback procedure tested
- [ ] Health checks configured
- [ ] Monitoring dashboard created
- [ ] Incident response plan defined
- [ ] Post-deployment verification plan

### Post-Deployment Validation

**Immediate (0-30 minutes)**:
- [ ] Health checks passing
- [ ] Error rate normal (<1%)
- [ ] Response times normal (p95 < SLO)
- [ ] No critical alerts firing
- [ ] Sample transactions successful

**Short-term (1-24 hours)**:
- [ ] Resource utilization normal
- [ ] No memory leaks detected
- [ ] All background jobs running
- [ ] Database connections stable
- [ ] External integrations working

**Long-term (1-7 days)**:
- [ ] Business metrics normal
- [ ] User feedback positive
- [ ] No recurring errors
- [ ] Performance stable under load
- [ ] Cost within budget

---

## Continuous Improvement

1. **Conduct post-mortems** for all incidents
2. **Review metrics** weekly
3. **Update runbooks** based on incidents
4. **Test disaster recovery** quarterly
5. **Review and update alerts** monthly
6. **Optimize slow queries** continuously
7. **Patch dependencies** regularly
8. **Capacity planning** quarterly

---

This checklist ensures your application is ready for production with proper configuration, observability, and resilience.
