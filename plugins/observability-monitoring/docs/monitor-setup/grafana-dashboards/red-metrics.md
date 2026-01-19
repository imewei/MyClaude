# RED Metrics Dashboards

RED metrics focus on: Rate, Errors, Duration - ideal for request-driven systems.

## RED Metrics Dashboard

```json
{
  "dashboard": {
    "uid": "red-metrics",
    "title": "RED Metrics - Request-Driven Services",
    "tags": ["red-metrics", "microservices"],
    "timezone": "browser",
    "schemaVersion": 38,
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "templating": {
      "list": [
        {
          "type": "datasource",
          "name": "datasource",
          "label": "Data Source",
          "query": "prometheus"
        },
        {
          "type": "query",
          "name": "namespace",
          "label": "Namespace",
          "datasource": "${datasource}",
          "query": "label_values(http_requests_total, namespace)",
          "refresh": 1,
          "multi": true,
          "includeAll": true
        },
        {
          "type": "query",
          "name": "service",
          "label": "Service",
          "datasource": "${datasource}",
          "query": "label_values(http_requests_total{namespace=~\"$namespace\"}, job)",
          "refresh": 1,
          "multi": true,
          "includeAll": true
        }
      ]
    },
    "panels": [
      {
        "type": "row",
        "title": "Rate",
        "gridPos": {
          "h": 1,
          "w": 24,
          "x": 0,
          "y": 0
        }
      },
      {
        "type": "stat",
        "title": "Request Rate (QPS)",
        "gridPos": {
          "h": 4,
          "w": 4,
          "x": 0,
          "y": 1
        },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=~\"$service\",namespace=~\"$namespace\"}[5m]))",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "decimals": 2,
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"}
              ]
            }
          }
        },
        "options": {
          "colorMode": "value",
          "graphMode": "area",
          "textMode": "value_and_name"
        }
      },
      {
        "type": "timeseries",
        "title": "Request Rate by Service",
        "gridPos": {
          "h": 8,
          "w": 20,
          "x": 4,
          "y": 1
        },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=~\"$service\",namespace=~\"$namespace\"}[5m])) by (job)",
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
              "gradientMode": "opacity"
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
        "type": "stat",
        "title": "Total Requests (1h)",
        "gridPos": {
          "h": 4,
          "w": 4,
          "x": 0,
          "y": 5
        },
        "targets": [
          {
            "expr": "sum(increase(http_requests_total{job=~\"$service\",namespace=~\"$namespace\"}[1h]))",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "decimals": 0
          }
        },
        "options": {
          "colorMode": "value",
          "graphMode": "area"
        }
      },
      {
        "type": "row",
        "title": "Errors",
        "gridPos": {
          "h": 1,
          "w": 24,
          "x": 0,
          "y": 9
        }
      },
      {
        "type": "stat",
        "title": "Error Rate (%)",
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 0,
          "y": 10
        },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=~\"$service\",namespace=~\"$namespace\",status=~\"5..\"}[5m])) / sum(rate(http_requests_total{job=~\"$service\",namespace=~\"$namespace\"}[5m])) * 100",
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
                {"value": 0.1, "color": "yellow"},
                {"value": 1, "color": "orange"},
                {"value": 5, "color": "red"}
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
        "type": "stat",
        "title": "4xx Rate (%)",
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 6,
          "y": 10
        },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=~\"$service\",namespace=~\"$namespace\",status=~\"4..\"}[5m])) / sum(rate(http_requests_total{job=~\"$service\",namespace=~\"$namespace\"}[5m])) * 100",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "decimals": 2,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"},
                {"value": 5, "color": "yellow"},
                {"value": 15, "color": "red"}
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
        "type": "timeseries",
        "title": "Error Rate by Status Code",
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 10
        },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=~\"$service\",namespace=~\"$namespace\",status=~\"[45]..\" }[5m])) by (status)",
            "legendFormat": "{{status}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2,
              "fillOpacity": 15,
              "stacking": {
                "mode": "normal"
              }
            }
          },
          "overrides": [
            {
              "matcher": {"id": "byRegexp", "options": "/5../"},
              "properties": [
                {
                  "id": "color",
                  "value": {"mode": "fixed", "fixedColor": "red"}
                }
              ]
            },
            {
              "matcher": {"id": "byRegexp", "options": "/4../"},
              "properties": [
                {
                  "id": "color",
                  "value": {"mode": "fixed", "fixedColor": "yellow"}
                }
              ]
            }
          ]
        }
      },
      {
        "type": "table",
        "title": "Error Breakdown",
        "gridPos": {
          "h": 4,
          "w": 12,
          "x": 0,
          "y": 14
        },
        "targets": [
          {
            "expr": "topk(20, sum by (job, status, handler) (rate(http_requests_total{job=~\"$service\",namespace=~\"$namespace\",status=~\"[45]..\"}[5m])))",
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
                {"id": "unit", "value": "reqps"},
                {"id": "displayName", "value": "Error Rate"},
                {"id": "custom.displayMode", "value": "color-background"}
              ]
            }
          ]
        },
        "transformations": [
          {
            "id": "organize",
            "options": {
              "excludeByName": {"Time": true}
            }
          }
        ]
      },
      {
        "type": "row",
        "title": "Duration",
        "gridPos": {
          "h": 1,
          "w": 24,
          "x": 0,
          "y": 18
        }
      },
      {
        "type": "stat",
        "title": "p99 Latency",
        "gridPos": {
          "h": 4,
          "w": 4,
          "x": 0,
          "y": 19
        },
        "targets": [
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job=~\"$service\",namespace=~\"$namespace\"}[5m])) by (le))",
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
                {"value": 1, "color": "orange"},
                {"value": 2, "color": "red"}
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
        "type": "stat",
        "title": "p95 Latency",
        "gridPos": {
          "h": 4,
          "w": 4,
          "x": 4,
          "y": 19
        },
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job=~\"$service\",namespace=~\"$namespace\"}[5m])) by (le))",
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
                {"value": 0.3, "color": "yellow"},
                {"value": 0.7, "color": "orange"},
                {"value": 1.5, "color": "red"}
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
        "type": "stat",
        "title": "p50 Latency",
        "gridPos": {
          "h": 4,
          "w": 4,
          "x": 8,
          "y": 19
        },
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{job=~\"$service\",namespace=~\"$namespace\"}[5m])) by (le))",
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
                {"value": 0.2, "color": "yellow"},
                {"value": 0.5, "color": "red"}
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
        "type": "timeseries",
        "title": "Latency Percentiles",
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 19
        },
        "targets": [
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job=~\"$service\",namespace=~\"$namespace\"}[5m])) by (le, job))",
            "legendFormat": "{{job}} p99",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job=~\"$service\",namespace=~\"$namespace\"}[5m])) by (le, job))",
            "legendFormat": "{{job}} p95",
            "refId": "B"
          },
          {
            "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{job=~\"$service\",namespace=~\"$namespace\"}[5m])) by (le, job))",
            "legendFormat": "{{job}} p50",
            "refId": "C"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2,
              "fillOpacity": 10
            }
          }
        }
      },
      {
        "type": "heatmap",
        "title": "Duration Heatmap",
        "gridPos": {
          "h": 4,
          "w": 12,
          "x": 0,
          "y": 23
        },
        "targets": [
          {
            "expr": "sum(increase(http_request_duration_seconds_bucket{job=~\"$service\",namespace=~\"$namespace\"}[5m])) by (le)",
            "format": "heatmap",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```
