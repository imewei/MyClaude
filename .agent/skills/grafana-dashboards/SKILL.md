---
name: grafana-dashboards
version: "1.0.7"
maturity: "5-Expert"
specialization: Metric Visualization
description: Create production Grafana dashboards with panels, variables, alerts, and templates using RED/USE methods. Use when building API monitoring, infrastructure, database, or SLO dashboards with Prometheus data sources.
---

# Grafana Dashboards

Production-ready metric visualization with Prometheus integration.

---

## Dashboard Design Methods

| Method | Focus | Metrics |
|--------|-------|---------|
| RED (Services) | Request behavior | Rate, Errors, Duration |
| USE (Resources) | Resource health | Utilization, Saturation, Errors |
| Golden Signals | User experience | Latency, Traffic, Errors, Saturation |

```
Dashboard Hierarchy:
┌─ Critical Metrics (Stat panels, big numbers)
├─ Key Trends (Time series graphs)
└─ Details (Tables, heatmaps)
```

---

## Panel Types

### Stat Panel

```json
{
  "type": "stat",
  "title": "Error Rate %",
  "targets": [{"expr": "(sum(rate(http_errors_total[5m]))/sum(rate(http_requests_total[5m])))*100"}],
  "fieldConfig": {
    "defaults": {
      "thresholds": {
        "steps": [
          {"value": 0, "color": "green"},
          {"value": 1, "color": "yellow"},
          {"value": 5, "color": "red"}
        ]
      }
    }
  }
}
```

### Time Series

```json
{
  "type": "timeseries",
  "title": "Request Rate",
  "targets": [{
    "expr": "sum(rate(http_requests_total[5m])) by (service)",
    "legendFormat": "{{service}}"
  }],
  "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
}
```

### Heatmap (Latency Distribution)

```json
{
  "type": "heatmap",
  "title": "Latency Heatmap",
  "targets": [{
    "expr": "sum(rate(http_request_duration_seconds_bucket[5m])) by (le)",
    "format": "heatmap"
  }]
}
```

---

## Variables (Templating)

```json
{
  "templating": {
    "list": [
      {
        "name": "namespace",
        "type": "query",
        "query": "label_values(kube_pod_info, namespace)",
        "refresh": 1
      },
      {
        "name": "service",
        "type": "query",
        "query": "label_values(kube_service_info{namespace=\"$namespace\"}, service)",
        "multi": true
      }
    ]
  }
}
```

**Usage in queries**: `sum(rate(http_requests_total{namespace="$namespace", service=~"$service"}[5m]))`

---

## Dashboard Alerts

```json
{
  "alert": {
    "name": "High Error Rate",
    "conditions": [{
      "evaluator": {"params": [5], "type": "gt"},
      "query": {"params": ["A", "5m", "now"]},
      "reducer": {"type": "avg"}
    }],
    "for": "5m",
    "frequency": "1m",
    "notifications": [{"uid": "slack-channel"}]
  }
}
```

---

## Common Dashboards

### API Monitoring (RED)

| Panel | Query |
|-------|-------|
| Request Rate | `sum(rate(http_requests_total[5m])) by (service)` |
| Error Rate % | `(sum(rate(http_requests_total{status=~"5.."}[5m]))/sum(rate(http_requests_total[5m])))*100` |
| P95 Latency | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))` |

### Infrastructure (USE)

| Panel | Query |
|-------|-------|
| CPU Utilization | `100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m]))*100)` |
| Memory Usage | `(1 - node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes)*100` |
| Disk I/O | `rate(node_disk_io_time_seconds_total[5m])` |

---

## Provisioning

### File-based (YAML)

```yaml
apiVersion: 1
providers:
  - name: 'default'
    folder: 'General'
    type: file
    options:
      path: /etc/grafana/dashboards
```

### Terraform

```hcl
resource "grafana_dashboard" "api" {
  config_json = file("${path.module}/dashboards/api.json")
  folder      = grafana_folder.monitoring.id
}
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Hierarchy | Critical stats at top, details below |
| Variables | Enable dynamic filtering |
| Consistent naming | prefix_metric_unit |
| Time ranges | Default 6h, allow customization |
| Thresholds | Color coding for quick assessment |
| Descriptions | Add panel descriptions |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Too many panels | Focus on actionable metrics |
| Wrong refresh rate | 30s-1m for real-time, 5m for trends |
| Missing variables | Add namespace/service filters |
| No thresholds | Define green/yellow/red levels |
| Hardcoded values | Use variables for flexibility |

---

## Checklist

- [ ] Dashboard follows RED or USE method
- [ ] Variables for namespace/service filtering
- [ ] Thresholds configured for key panels
- [ ] Alerts on critical metrics
- [ ] Panel descriptions added
- [ ] Provisioning as code configured

---

**Version**: 1.0.5
