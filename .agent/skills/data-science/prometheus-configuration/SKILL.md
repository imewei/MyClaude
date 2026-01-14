---
name: prometheus-configuration
version: "1.0.7"
maturity: "5-Expert"
specialization: Metric Collection
description: Configure Prometheus for metric collection, alerting, and monitoring with scrape configs, recording rules, alert rules, and service discovery. Use when setting up Prometheus servers, creating alert rules, or implementing Kubernetes monitoring.
---

# Prometheus Configuration

Metric collection, alerting, and monitoring infrastructure setup.

---

## Installation

### Kubernetes (Helm)

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set prometheus.prometheusSpec.retention=30d
```

### Docker Compose

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
```

---

## Core Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'production'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - /etc/prometheus/rules/*.yml

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

---

## Scrape Configurations

### Static Targets

```yaml
- job_name: 'node-exporter'
  static_configs:
    - targets: ['node1:9100', 'node2:9100']
      labels:
        env: 'production'
```

### Kubernetes Pods

```yaml
- job_name: 'kubernetes-pods'
  kubernetes_sd_configs:
    - role: pod
  relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
      action: replace
      regex: ([^:]+)(?::\d+)?;(\d+)
      replacement: $1:$2
      target_label: __address__
    - source_labels: [__meta_kubernetes_namespace]
      target_label: namespace
```

### File-based Discovery

```yaml
- job_name: 'file-sd'
  file_sd_configs:
    - files: ['/etc/prometheus/targets/*.json']
      refresh_interval: 5m
```

---

## Recording Rules

Pre-compute expensive queries:

```yaml
groups:
  - name: api_metrics
    interval: 15s
    rules:
      - record: job:http_requests:rate5m
        expr: sum by (job) (rate(http_requests_total[5m]))

      - record: job:http_requests_error_rate:percentage
        expr: |
          sum by (job) (rate(http_requests_total{status=~"5.."}[5m]))
          / sum by (job) (rate(http_requests_total[5m])) * 100

      - record: job:http_request_duration:p95
        expr: histogram_quantile(0.95, sum by (job, le) (rate(http_request_duration_seconds_bucket[5m])))

  - name: resource_metrics
    rules:
      - record: instance:node_cpu:utilization
        expr: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

      - record: instance:node_memory:utilization
        expr: 100 - ((node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)
```

---

## Alert Rules

```yaml
groups:
  - name: availability
    rules:
      - alert: ServiceDown
        expr: up{job="my-app"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.instance }} is down"

      - alert: HighErrorRate
        expr: job:http_requests_error_rate:percentage > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Error rate {{ $value }}% for {{ $labels.job }}"

      - alert: HighLatency
        expr: job:http_request_duration:p95 > 1
        for: 5m
        labels:
          severity: warning

  - name: resources
    rules:
      - alert: HighCPUUsage
        expr: instance:node_cpu:utilization > 80
        for: 5m
        labels:
          severity: warning

      - alert: DiskSpaceLow
        expr: instance:node_disk:utilization > 90
        for: 5m
        labels:
          severity: critical
```

---

## Validation

```bash
# Validate configuration
promtool check config prometheus.yml

# Validate rules
promtool check rules /etc/prometheus/rules/*.yml

# Test query
promtool query instant http://localhost:9090 'up'
```

---

## API Endpoints

```bash
# Check targets
curl http://localhost:9090/api/v1/targets

# Check configuration
curl http://localhost:9090/api/v1/status/config

# Test query
curl 'http://localhost:9090/api/v1/query?query=up'
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Naming convention | prefix_name_unit (e.g., http_requests_total) |
| Scrape interval | 15-60s typical |
| Recording rules | Pre-compute expensive queries |
| High availability | Multiple Prometheus instances |
| Retention | Based on storage capacity |
| Relabeling | Clean up metrics at scrape time |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| High cardinality | Limit label values |
| Slow queries | Use recording rules |
| Missing targets | Check service discovery |
| Alert fatigue | Tune thresholds, add for: duration |
| Storage full | Set retention, use remote storage |

---

## Checklist

- [ ] Global scrape/evaluation interval configured
- [ ] Alertmanager integration configured
- [ ] Recording rules for expensive queries
- [ ] Alert rules with appropriate thresholds
- [ ] Service discovery for dynamic targets
- [ ] Configuration validated with promtool
- [ ] Retention and storage configured

---

**Version**: 1.0.5
