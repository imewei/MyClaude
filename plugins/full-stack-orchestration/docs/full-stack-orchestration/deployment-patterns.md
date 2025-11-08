# Deployment Patterns

> **Reference**: CI/CD pipelines, container orchestration, feature flags, and progressive delivery strategies

## CI/CD Pipeline Patterns

### GitHub Actions Full-Stack Pipeline
```yaml
name: Full-Stack CI/CD
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: npm test && pytest
      
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build containers
        run: docker-compose build
      - name: Push to registry
        run: docker push myapp:${{github.sha}}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: kubectl apply -f k8s/
```

## Container Orchestration

### Docker Compose for Local Development
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://user:pass@db:5432/myapp
    depends_on:
      - db
      - redis

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      API_URL: http://backend:8000

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: api
        image: myapp/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Feature Flags

### LaunchDarkly Integration
```typescript
import * as LaunchDarkly from 'launchdarkly-node-server-sdk';

const client = LaunchDarkly.init(process.env.LAUNCHDARKLY_SDK_KEY);

async function isFeatureEnabled(featureName: string, user: User): boolean {
  await client.waitForInitialization();
  return await client.variation(featureName, {
    key: user.id,
    email: user.email
  }, false);
}
```

## Progressive Delivery

### Canary Deployment with Argo Rollouts
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: backend-rollout
spec:
  replicas: 5
  strategy:
    canary:
      steps:
      - setWeight: 20
      - pause: {duration: 5m}
      - setWeight: 40
      - pause: {duration: 5m}
      - setWeight: 60
      - pause: {duration: 5m}
      - setWeight: 80
      - pause: {duration: 5m}
```

## Monitoring & Observability

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, generate_latest

http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
http_request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    http_request_duration.observe(duration)
    
    return response
```

### OpenTelemetry Tracing
```typescript
import { trace } from '@opentelemetry/api';

const tracer = trace.getTracer('app-name');

async function fetchUserData(userId: string) {
  const span = tracer.startSpan('fetchUserData');
  span.setAttribute('user.id', userId);
  
  try {
    const user = await database.getUser(userId);
    span.setStatus({ code: SpanStatusCode.OK });
    return user;
  } catch (error) {
    span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
    throw error;
  } finally {
    span.end();
  }
}
```

## Rollback Procedures

### Quick Rollback Steps
1. Identify deployment version: `kubectl rollout history deployment/backend`
2. Rollback to previous: `kubectl rollout undo deployment/backend`
3. Rollback to specific: `kubectl rollout undo deployment/backend --to-revision=3`
4. Verify health: `kubectl get pods -w`
5. Check metrics dashboard
6. Monitor error rates for 15 minutes

### Automated Rollback with Flagger
```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: backend
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  service:
    port: 8000
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500
      interval: 1m
```

## Blue-Green Deployment

### Traffic Switching Strategy
```bash
# Deploy green environment
kubectl apply -f k8s/deployment-green.yaml

# Wait for health checks
kubectl wait --for=condition=available deployment/backend-green --timeout=300s

# Run smoke tests
./scripts/smoke-test.sh https://green.example.com

# Switch traffic (update service selector)
kubectl patch service backend -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor for 30 minutes
# If issues: kubectl patch service backend -p '{"spec":{"selector":{"version":"blue"}}}'

# Decommission blue after validation
kubectl delete deployment backend-blue
```
