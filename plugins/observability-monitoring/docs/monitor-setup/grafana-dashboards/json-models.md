# Complete Dashboard Examples

## Microservices Dashboard

```json
{
  "dashboard": {
    "uid": "microservices-overview",
    "title": "Microservices Overview",
    "tags": ["microservices", "production"],
    "timezone": "browser",
    "schemaVersion": 38,
    "refresh": "30s",
    "time": {
      "from": "now-3h",
      "to": "now"
    },
    "templating": {
      "list": [
        {
          "type": "query",
          "name": "environment",
          "label": "Environment",
          "datasource": "Prometheus",
          "query": "label_values(http_requests_total, environment)",
          "multi": false,
          "includeAll": false
        },
        {
          "type": "query",
          "name": "service",
          "label": "Service",
          "datasource": "Prometheus",
          "query": "label_values(http_requests_total{environment=\"$environment\"}, job)",
          "multi": true,
          "includeAll": true
        }
      ]
    },
    "panels": [
      {
        "type": "row",
        "title": "Service Health Overview",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}
      },
      {
        "type": "stat",
        "title": "Services Up",
        "gridPos": {"h": 4, "w": 4, "x": 0, "y": 1},
        "targets": [
          {
            "expr": "count(up{job=~\"$service\",environment=\"$environment\"} == 1)",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "color": {"mode": "thresholds"},
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "red"},
                {"value": 1, "color": "green"}
              ]
            }
          }
        }
      },
      {
        "type": "stat",
        "title": "Total Request Rate",
        "gridPos": {"h": 4, "w": 4, "x": 4, "y": 1},
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=~\"$service\",environment=\"$environment\"}[5m]))",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "decimals": 2
          }
        },
        "options": {
          "colorMode": "value",
          "graphMode": "area"
        }
      },
      {
        "type": "stat",
        "title": "Overall Error Rate",
        "gridPos": {"h": 4, "w": 4, "x": 8, "y": 1},
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=~\"$service\",environment=\"$environment\",status=~\"5..\"}[5m])) / sum(rate(http_requests_total{job=~\"$service\",environment=\"$environment\"}[5m])) * 100",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "decimals": 3,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"},
                {"value": 1, "color": "yellow"},
                {"value": 5, "color": "red"}
              ]
            }
          }
        },
        "options": {
          "colorMode": "background"
        }
      },
      {
        "type": "stat",
        "title": "p99 Latency",
        "gridPos": {"h": 4, "w": 4, "x": 12, "y": 1},
        "targets": [
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job=~\"$service\",environment=\"$environment\"}[5m])) by (le))",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "decimals": 3,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"},
                {"value": 0.5, "color": "yellow"},
                {"value": 1, "color": "red"}
              ]
            }
          }
        },
        "options": {
          "colorMode": "background",
          "graphMode": "area"
        }
      },
      {
        "type": "table",
        "title": "Service Status Matrix",
        "gridPos": {"h": 4, "w": 8, "x": 16, "y": 1},
        "targets": [
          {
            "expr": "up{job=~\"$service\",environment=\"$environment\"}",
            "format": "table",
            "instant": true,
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "overrides": [
            {
              "matcher": {"id": "byName", "options": "Value"},
              "properties": [
                {"id": "displayName", "value": "Status"},
                {
                  "id": "custom.displayMode",
                  "value": "color-background"
                },
                {
                  "id": "mappings",
                  "value": [
                    {"type": "value", "options": {"1": {"text": "UP", "color": "green"}}},
                    {"type": "value", "options": {"0": {"text": "DOWN", "color": "red"}}}
                  ]
                }
              ]
            }
          ]
        },
        "transformations": [
          {
            "id": "organize",
            "options": {
              "excludeByName": {"Time": true, "__name__": true, "environment": false}
            }
          }
        ]
      },
      {
        "type": "row",
        "title": "Request Metrics",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 5}
      },
      {
        "type": "timeseries",
        "title": "Request Rate by Service",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6},
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=~\"$service\",environment=\"$environment\"}[5m])) by (job)",
            "legendFormat": "{{job}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2,
              "fillOpacity": 10,
              "stacking": {"mode": "normal"}
            }
          }
        },
        "options": {
          "legend": {
            "displayMode": "table",
            "placement": "right",
            "calcs": ["mean", "max", "last"]
          }
        }
      },
      {
        "type": "timeseries",
        "title": "Request Duration Percentiles",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6},
        "targets": [
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job=~\"$service\",environment=\"$environment\"}[5m])) by (le, job))",
            "legendFormat": "{{job}} p99",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job=~\"$service\",environment=\"$environment\"}[5m])) by (le, job))",
            "legendFormat": "{{job}} p95",
            "refId": "B"
          },
          {
            "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{job=~\"$service\",environment=\"$environment\"}[5m])) by (le, job))",
            "legendFormat": "{{job}} p50",
            "refId": "C"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2
            }
          }
        }
      },
      {
        "type": "row",
        "title": "Service Dependencies",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 14}
      },
      {
        "type": "timeseries",
        "title": "Database Connection Pool",
        "gridPos": {"h": 6, "w": 8, "x": 0, "y": 15},
        "targets": [
          {
            "expr": "db_pool_active_connections{job=~\"$service\",environment=\"$environment\"}",
            "legendFormat": "{{job}} active",
            "refId": "A"
          },
          {
            "expr": "db_pool_max_connections{job=~\"$service\",environment=\"$environment\"}",
            "legendFormat": "{{job}} max",
            "refId": "B"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2
            }
          }
        }
      },
      {
        "type": "timeseries",
        "title": "Cache Hit Rate",
        "gridPos": {"h": 6, "w": 8, "x": 8, "y": 15},
        "targets": [
          {
            "expr": "sum(rate(cache_hits_total{job=~\"$service\",environment=\"$environment\"}[5m])) by (job) / (sum(rate(cache_hits_total{job=~\"$service\",environment=\"$environment\"}[5m])) by (job) + sum(rate(cache_misses_total{job=~\"$service\",environment=\"$environment\"}[5m])) by (job)) * 100",
            "legendFormat": "{{job}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2,
              "fillOpacity": 20
            }
          }
        }
      },
      {
        "type": "timeseries",
        "title": "Message Queue Depth",
        "gridPos": {"h": 6, "w": 8, "x": 16, "y": 15},
        "targets": [
          {
            "expr": "queue_depth{job=~\"$service\",environment=\"$environment\"}",
            "legendFormat": "{{job}} {{queue}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2
            },
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"},
                {"value": 1000, "color": "yellow"},
                {"value": 5000, "color": "red"}
              ]
            }
          }
        }
      }
    ]
  }
}
```

## Kubernetes Cluster Dashboard

```json
{
  "dashboard": {
    "uid": "k8s-cluster",
    "title": "Kubernetes Cluster Overview",
    "tags": ["kubernetes", "infrastructure"],
    "timezone": "browser",
    "schemaVersion": 38,
    "refresh": "30s",
    "templating": {
      "list": [
        {
          "type": "query",
          "name": "cluster",
          "query": "label_values(kube_pod_info, cluster)",
          "datasource": "Prometheus"
        },
        {
          "type": "query",
          "name": "namespace",
          "query": "label_values(kube_pod_info{cluster=\"$cluster\"}, namespace)",
          "multi": true,
          "includeAll": true
        }
      ]
    },
    "panels": [
      {
        "type": "stat",
        "title": "Total Pods",
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "count(kube_pod_info{cluster=\"$cluster\",namespace=~\"$namespace\"})",
            "refId": "A"
          }
        ]
      },
      {
        "type": "stat",
        "title": "Running Pods",
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0},
        "targets": [
          {
            "expr": "count(kube_pod_status_phase{cluster=\"$cluster\",namespace=~\"$namespace\",phase=\"Running\"})",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "fixed", "fixedColor": "green"}
          }
        }
      },
      {
        "type": "stat",
        "title": "Failed Pods",
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "count(kube_pod_status_phase{cluster=\"$cluster\",namespace=~\"$namespace\",phase=\"Failed\"})",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"},
                {"value": 1, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "type": "stat",
        "title": "Node Count",
        "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0},
        "targets": [
          {
            "expr": "count(kube_node_info{cluster=\"$cluster\"})",
            "refId": "A"
          }
        ]
      },
      {
        "type": "timeseries",
        "title": "CPU Usage by Namespace",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
        "targets": [
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total{cluster=\"$cluster\",namespace=~\"$namespace\",container!=\"\"}[5m])) by (namespace)",
            "legendFormat": "{{namespace}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "cores",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2,
              "fillOpacity": 10,
              "stacking": {"mode": "normal"}
            }
          }
        }
      },
      {
        "type": "timeseries",
        "title": "Memory Usage by Namespace",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
        "targets": [
          {
            "expr": "sum(container_memory_working_set_bytes{cluster=\"$cluster\",namespace=~\"$namespace\",container!=\"\"}) by (namespace)",
            "legendFormat": "{{namespace}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "bytes",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2,
              "fillOpacity": 10,
              "stacking": {"mode": "normal"}
            }
          }
        }
      }
    ]
  }
}
```
