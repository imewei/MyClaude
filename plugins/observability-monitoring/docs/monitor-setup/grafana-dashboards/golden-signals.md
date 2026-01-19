# Golden Signals Dashboards

The Four Golden Signals (from Google's SRE book): Latency, Traffic, Errors, and Saturation.

## Complete Golden Signals Dashboard

```json
{
  "dashboard": {
    "uid": "golden-signals",
    "title": "Golden Signals - Service Overview",
    "tags": ["golden-signals", "sre"],
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
          "name": "service",
          "label": "Service",
          "datasource": "${datasource}",
          "query": "label_values(http_requests_total, job)",
          "refresh": 1,
          "multi": false,
          "includeAll": false
        },
        {
          "type": "interval",
          "name": "interval",
          "label": "Interval",
          "auto": true,
          "auto_count": 30,
          "auto_min": "10s",
          "query": "10s,30s,1m,5m,10m,30m,1h"
        }
      ]
    },
    "panels": [
      {
        "type": "row",
        "title": "Latency",
        "gridPos": {
          "h": 1,
          "w": 24,
          "x": 0,
          "y": 0
        },
        "collapsed": false
      },
      {
        "type": "timeseries",
        "title": "Request Latency (p50, p95, p99)",
        "gridPos": {
          "h": 8,
          "w": 16,
          "x": 0,
          "y": 1
        },
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{job=\"$service\"}[$interval])) by (le))",
            "legendFormat": "p50",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job=\"$service\"}[$interval])) by (le))",
            "legendFormat": "p95",
            "refId": "B"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job=\"$service\"}[$interval])) by (le))",
            "legendFormat": "p99",
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
            },
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
          "legend": {
            "displayMode": "table",
            "placement": "right",
            "calcs": ["mean", "max", "last"]
          }
        }
      },
      {
        "type": "stat",
        "title": "p99 Latency",
        "gridPos": {
          "h": 4,
          "w": 4,
          "x": 16,
          "y": 1
        },
        "targets": [
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job=\"$service\"}[$interval])) by (le))",
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
        "type": "stat",
        "title": "p95 Latency",
        "gridPos": {
          "h": 4,
          "w": 4,
          "x": 20,
          "y": 1
        },
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job=\"$service\"}[$interval])) by (le))",
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
                {"value": 0.7, "color": "red"}
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
        "type": "heatmap",
        "title": "Latency Distribution",
        "gridPos": {
          "h": 4,
          "w": 8,
          "x": 16,
          "y": 5
        },
        "targets": [
          {
            "expr": "sum(increase(http_request_duration_seconds_bucket{job=\"$service\"}[$interval])) by (le)",
            "format": "heatmap",
            "refId": "A"
          }
        ],
        "options": {
          "calculate": false,
          "color": {
            "mode": "scheme",
            "scheme": "Spectral"
          },
          "yAxis": {
            "unit": "s"
          }
        }
      },
      {
        "type": "row",
        "title": "Traffic",
        "gridPos": {
          "h": 1,
          "w": 24,
          "x": 0,
          "y": 9
        },
        "collapsed": false
      },
      {
        "type": "timeseries",
        "title": "Request Rate",
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 10
        },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=\"$service\"}[$interval]))",
            "legendFormat": "Total",
            "refId": "A"
          },
          {
            "expr": "sum(rate(http_requests_total{job=\"$service\"}[$interval])) by (method)",
            "legendFormat": "{{method}}",
            "refId": "B"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 1,
              "fillOpacity": 10,
              "stacking": {
                "mode": "normal"
              }
            }
          }
        }
      },
      {
        "type": "timeseries",
        "title": "Request Rate by Endpoint",
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 10
        },
        "targets": [
          {
            "expr": "topk(10, sum(rate(http_requests_total{job=\"$service\"}[$interval])) by (handler))",
            "legendFormat": "{{handler}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 1
            }
          }
        }
      },
      {
        "type": "row",
        "title": "Errors",
        "gridPos": {
          "h": 1,
          "w": 24,
          "x": 0,
          "y": 18
        },
        "collapsed": false
      },
      {
        "type": "stat",
        "title": "Error Rate",
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 0,
          "y": 19
        },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=\"$service\",status=~\"5..\"}[$interval])) / sum(rate(http_requests_total{job=\"$service\"}[$interval])) * 100",
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
                {"value": 1, "color": "yellow"},
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
        "title": "Total Errors (5m)",
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 6,
          "y": 19
        },
        "targets": [
          {
            "expr": "sum(increase(http_requests_total{job=\"$service\",status=~\"5..\"}[5m]))",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "decimals": 0,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"},
                {"value": 10, "color": "yellow"},
                {"value": 100, "color": "red"}
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
        "title": "Error Rate Over Time",
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 19
        },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=\"$service\",status=~\"5..\"}[$interval])) by (status)",
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
              "fillOpacity": 10
            },
            "color": {
              "mode": "palette-classic"
            }
          }
        }
      },
      {
        "type": "table",
        "title": "Top Errors by Endpoint",
        "gridPos": {
          "h": 4,
          "w": 12,
          "x": 0,
          "y": 23
        },
        "targets": [
          {
            "expr": "topk(10, sum by (handler, status) (rate(http_requests_total{job=\"$service\",status=~\"5..\"}[$interval])))",
            "format": "table",
            "instant": true,
            "refId": "A"
          }
        ],
        "transformations": [
          {
            "id": "organize",
            "options": {
              "excludeByName": {"Time": true},
              "renameByName": {
                "Value": "Error Rate"
              }
            }
          }
        ]
      },
      {
        "type": "row",
        "title": "Saturation",
        "gridPos": {
          "h": 1,
          "w": 24,
          "x": 0,
          "y": 27
        },
        "collapsed": false
      },
      {
        "type": "gauge",
        "title": "CPU Usage",
        "gridPos": {
          "h": 6,
          "w": 6,
          "x": 0,
          "y": 28
        },
        "targets": [
          {
            "expr": "avg(rate(process_cpu_seconds_total{job=\"$service\"}[$interval])) * 100",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"},
                {"value": 70, "color": "yellow"},
                {"value": 85, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "type": "gauge",
        "title": "Memory Usage",
        "gridPos": {
          "h": 6,
          "w": 6,
          "x": 6,
          "y": 28
        },
        "targets": [
          {
            "expr": "process_resident_memory_bytes{job=\"$service\"} / process_virtual_memory_max_bytes{job=\"$service\"} * 100",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"},
                {"value": 75, "color": "yellow"},
                {"value": 90, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "type": "timeseries",
        "title": "Thread Pool Saturation",
        "gridPos": {
          "h": 6,
          "w": 12,
          "x": 12,
          "y": 28
        },
        "targets": [
          {
            "expr": "threadpool_active_threads{job=\"$service\"} / threadpool_max_threads{job=\"$service\"} * 100",
            "legendFormat": "{{pool}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"},
                {"value": 70, "color": "yellow"},
                {"value": 90, "color": "red"}
              ]
            }
          }
        }
      }
    ]
  }
}
```
