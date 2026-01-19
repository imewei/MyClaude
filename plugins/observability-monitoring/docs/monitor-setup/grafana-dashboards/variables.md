# Dashboard Variables & Templating

Template variables enable dynamic, reusable dashboards.

## Variable Types

### Query Variable

```json
{
  "type": "query",
  "name": "service",
  "label": "Service",
  "datasource": "${datasource}",
  "query": "label_values(http_requests_total, job)",
  "refresh": 1,
  "regex": "",
  "multi": true,
  "includeAll": true,
  "allValue": ".*",
  "sort": 1
}
```

### Custom Variable

```json
{
  "type": "custom",
  "name": "environment",
  "label": "Environment",
  "query": "production,staging,development",
  "multi": false,
  "includeAll": false,
  "current": {
    "text": "production",
    "value": "production"
  }
}
```

### Interval Variable

```json
{
  "type": "interval",
  "name": "interval",
  "label": "Interval",
  "auto": true,
  "auto_count": 30,
  "auto_min": "10s",
  "query": "10s,30s,1m,5m,10m,30m,1h",
  "refresh": 2
}
```

### Datasource Variable

```json
{
  "type": "datasource",
  "name": "datasource",
  "label": "Data Source",
  "query": "prometheus",
  "regex": "",
  "multi": false,
  "includeAll": false
}
```

### Constant Variable

```json
{
  "type": "constant",
  "name": "cluster",
  "label": "Cluster",
  "query": "us-west-2",
  "hide": 2
}
```

## Advanced Variable Usage

### Chained Variables

```json
{
  "templating": {
    "list": [
      {
        "type": "query",
        "name": "cluster",
        "query": "label_values(kube_pod_info, cluster)",
        "refresh": 1
      },
      {
        "type": "query",
        "name": "namespace",
        "query": "label_values(kube_pod_info{cluster=\"$cluster\"}, namespace)",
        "refresh": 1
      },
      {
        "type": "query",
        "name": "pod",
        "query": "label_values(kube_pod_info{cluster=\"$cluster\",namespace=\"$namespace\"}, pod)",
        "refresh": 1
      }
    ]
  }
}
```

### Multi-Value with All

```json
{
  "type": "query",
  "name": "service",
  "label": "Service",
  "query": "label_values(http_requests_total, job)",
  "multi": true,
  "includeAll": true,
  "allValue": ".*"
}
```

Usage in query: `{job=~"$service"}`

### Text Box Variable

```json
{
  "type": "textbox",
  "name": "threshold",
  "label": "Alert Threshold",
  "query": "0.95"
}
```
