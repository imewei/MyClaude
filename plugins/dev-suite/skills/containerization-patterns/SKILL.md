---
name: containerization-patterns
description: Build and optimize Docker containers and Kubernetes deployments including multi-stage builds, image security, pod design, Helm charts, and container orchestration. Use when writing Dockerfiles, creating K8s manifests, optimizing image size, or designing container architectures.
---

# Containerization Patterns

## Expert Agent

For container architecture, Kubernetes deployments, and orchestration design, delegate to:

- **`devops-architect`**: Designs cloud-native platform architecture with container orchestration and IaC.
  - *Location*: `plugins/dev-suite/agents/devops-architect.md`


## Dockerfile Best Practices

### Multi-Stage Build (Python)

```dockerfile
# Stage 1: Build
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
ENV PATH="/app/.venv/bin:$PATH"
USER 1000:1000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "src.main"]
```

### Image Size Optimization

| Technique | Impact |
|-----------|--------|
| Alpine/slim base images | 50-80% reduction |
| Multi-stage builds | Remove build dependencies |
| `.dockerignore` | Exclude unnecessary files |
| Layer ordering | Cache dependencies first |
| `--no-install-recommends` | Skip optional packages |

### .dockerignore

```
.git
.env
node_modules
__pycache__
*.pyc
.pytest_cache
.mypy_cache
docker-compose*.yml
README.md
docs/
tests/
```


## Image Security

- [ ] Use specific version tags, never `latest`
- [ ] Run as non-root user (`USER 1000:1000`)
- [ ] Scan with `trivy image <name>` or `docker scout`
- [ ] No secrets in build args or layers
- [ ] Use `COPY` not `ADD` (unless extracting archives)
- [ ] Pin package versions in `RUN apt-get install`


## Kubernetes Manifests

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  labels:
    app: api-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
        - name: api-server
          image: registry.example.com/api-server:1.2.3
          ports:
            - containerPort: 8000
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 512Mi
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
          env:
            - name: DB_HOST
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: host
```

## Helm Chart Structure

```
charts/api-server/
  Chart.yaml
  values.yaml
  templates/
    deployment.yaml
    service.yaml
    ingress.yaml
    _helpers.tpl
```

### values.yaml Pattern

```yaml
replicaCount: 3
image:
  repository: registry.example.com/api-server
  tag: "1.2.3"
  pullPolicy: IfNotPresent
resources:
  requests: { cpu: 100m, memory: 128Mi }
  limits: { cpu: 500m, memory: 512Mi }
```


## Resource Limits Guidelines

| Workload Type | CPU Request | Memory Request | CPU Limit | Memory Limit |
|---------------|-------------|----------------|-----------|--------------|
| API server | 100m | 128Mi | 500m | 512Mi |
| Worker | 250m | 256Mi | 1000m | 1Gi |
| Batch job | 500m | 512Mi | 2000m | 2Gi |


## Health Check Patterns

| Probe | Purpose | Failure Action |
|-------|---------|---------------|
| Liveness | Is process alive? | Restart container |
| Readiness | Can it serve traffic? | Remove from service |
| Startup | Has it finished init? | Delay other probes |


## Deployment Checklist

- [ ] Multi-stage Dockerfile with minimal runtime image
- [ ] Non-root user configured
- [ ] Image scanned for vulnerabilities
- [ ] Resource requests and limits set
- [ ] Liveness and readiness probes configured
- [ ] Secrets injected via K8s secrets or external vault
- [ ] Pod disruption budget defined for HA
- [ ] Horizontal pod autoscaler configured
