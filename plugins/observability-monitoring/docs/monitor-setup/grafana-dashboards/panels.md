# Grafana Panels & Queries

## Panel Types

### Time Series (Graph) Panel

Modern replacement for the legacy Graph panel with improved performance.

```json
{
  "type": "timeseries",
  "title": "Request Rate",
  "gridPos": {
    "h": 8,
    "w": 12,
    "x": 0,
    "y": 0
  },
  "targets": [
    {
      "expr": "sum(rate(http_requests_total{job=\"$service\"}[5m])) by (status)",
      "legendFormat": "{{status}}",
      "refId": "A"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "custom": {
        "drawStyle": "line",
        "lineInterpolation": "linear",
        "lineWidth": 1,
        "fillOpacity": 10,
        "gradientMode": "none",
        "spanNulls": false,
        "showPoints": "never",
        "pointSize": 5,
        "stacking": {
          "mode": "none",
          "group": "A"
        },
        "axisPlacement": "auto",
        "axisLabel": "",
        "scaleDistribution": {
          "type": "linear"
        }
      },
      "color": {
        "mode": "palette-classic"
      },
      "unit": "reqps",
      "decimals": 2
    },
    "overrides": [
      {
        "matcher": {
          "id": "byName",
          "options": "5xx"
        },
        "properties": [
          {
            "id": "color",
            "value": {
              "mode": "fixed",
              "fixedColor": "red"
            }
          }
        ]
      }
    ]
  },
  "options": {
    "tooltip": {
      "mode": "multi",
      "sort": "desc"
    },
    "legend": {
      "displayMode": "table",
      "placement": "right",
      "calcs": ["mean", "max", "last"]
    }
  }
}
```

### Stat Panel

Single value visualization with sparkline and thresholds.

```json
{
  "type": "stat",
  "title": "Success Rate",
  "gridPos": {
    "h": 4,
    "w": 6,
    "x": 0,
    "y": 0
  },
  "targets": [
    {
      "expr": "sum(rate(http_requests_total{job=\"$service\",status=~\"2..\"}[5m])) / sum(rate(http_requests_total{job=\"$service\"}[5m])) * 100",
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
          {
            "value": null,
            "color": "red"
          },
          {
            "value": 95,
            "color": "yellow"
          },
          {
            "value": 99,
            "color": "green"
          }
        ]
      },
      "mappings": []
    }
  },
  "options": {
    "reduceOptions": {
      "values": false,
      "calcs": ["lastNotNull"]
    },
    "orientation": "auto",
    "textMode": "value_and_name",
    "colorMode": "background",
    "graphMode": "area",
    "justifyMode": "auto"
  }
}
```

### Gauge Panel

Radial or linear gauge for percentage values.

```json
{
  "type": "gauge",
  "title": "CPU Usage",
  "gridPos": {
    "h": 6,
    "w": 6,
    "x": 0,
    "y": 0
  },
  "targets": [
    {
      "expr": "100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\",instance=\"$instance\"}[5m])) * 100)",
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
          {
            "value": null,
            "color": "green"
          },
          {
            "value": 70,
            "color": "yellow"
          },
          {
            "value": 85,
            "color": "red"
          }
        ]
      }
    }
  },
  "options": {
    "orientation": "auto",
    "showThresholdLabels": true,
    "showThresholdMarkers": true
  }
}
```

### Heatmap Panel

Distribution visualization for latency percentiles.

```json
{
  "type": "heatmap",
  "title": "Request Duration Heatmap",
  "gridPos": {
    "h": 8,
    "w": 12,
    "x": 0,
    "y": 0
  },
  "targets": [
    {
      "expr": "sum(increase(http_request_duration_seconds_bucket{job=\"$service\"}[5m])) by (le)",
      "format": "heatmap",
      "legendFormat": "{{le}}",
      "refId": "A"
    }
  ],
  "options": {
    "calculate": false,
    "cellGap": 2,
    "cellRadius": 0,
    "color": {
      "exponent": 0.5,
      "fill": "dark-orange",
      "mode": "scheme",
      "reverse": false,
      "scale": "exponential",
      "scheme": "Spectral",
      "steps": 128
    },
    "exemplars": {
      "color": "rgba(255,0,255,0.7)"
    },
    "filterValues": {
      "le": 1e-9
    },
    "legend": {
      "show": true
    },
    "rowsFrame": {
      "layout": "auto"
    },
    "tooltip": {
      "show": true,
      "yHistogram": true
    },
    "yAxis": {
      "axisPlacement": "left",
      "reverse": false,
      "unit": "s"
    }
  }
}
```

### Table Panel

Tabular data with sorting, filtering, and cell rendering.

```json
{
  "type": "table",
  "title": "Service Endpoints",
  "gridPos": {
    "h": 8,
    "w": 12,
    "x": 0,
    "y": 0
  },
  "targets": [
    {
      "expr": "topk(10, sum by (handler, method) (rate(http_requests_total{job=\"$service\"}[5m])))",
      "format": "table",
      "instant": true,
      "refId": "A"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "custom": {
        "align": "auto",
        "displayMode": "auto"
      },
      "mappings": [],
      "thresholds": {
        "mode": "absolute",
        "steps": [
          {
            "value": null,
            "color": "green"
          }
        ]
      }
    },
    "overrides": [
      {
        "matcher": {
          "id": "byName",
          "options": "Value"
        },
        "properties": [
          {
            "id": "unit",
            "value": "reqps"
          },
          {
            "id": "displayName",
            "value": "Request Rate"
          },
          {
            "id": "custom.displayMode",
            "value": "gradient-gauge"
          }
        ]
      }
    ]
  },
  "options": {
    "showHeader": true,
    "sortBy": [
      {
        "displayName": "Request Rate",
        "desc": true
      }
    ],
    "footer": {
      "show": false,
      "reducer": ["sum"],
      "fields": ""
    }
  },
  "transformations": [
    {
      "id": "organize",
      "options": {
        "excludeByName": {
          "Time": true
        },
        "indexByName": {},
        "renameByName": {}
      }
    }
  ]
}
```

### Logs Panel

Log stream visualization with filtering and search.

```json
{
  "type": "logs",
  "title": "Application Logs",
  "gridPos": {
    "h": 10,
    "w": 24,
    "x": 0,
    "y": 0
  },
  "targets": [
    {
      "expr": "{job=\"$service\", level=~\"$log_level\"} |= \"$search_query\"",
      "refId": "A"
    }
  ],
  "options": {
    "showTime": true,
    "showLabels": true,
    "showCommonLabels": false,
    "wrapLogMessage": true,
    "prettifyLogMessage": false,
    "enableLogDetails": true,
    "dedupStrategy": "none",
    "sortOrder": "Descending"
  }
}
```

---

## Query Best Practices

### Efficient PromQL

```promql
# Good: Aggregation before rate
sum(rate(metric[5m])) by (label)

# Bad: Rate before aggregation
rate(sum(metric)[5m]) by (label)

# Good: Use recording rules for complex queries
service:error_rate:5m

# Good: Appropriate range selectors
rate(metric[5m])  # For 5m interval
rate(metric[1m])  # For 1m interval
```

### Template Variable Usage

```promql
# Use regex matching for multi-select
metric{job=~"$service"}

# Use pipe for OR conditions
metric{status=~"$status"}  # where $status = "200|201|204"
```

## Testing Dashboards

### 1. Query Validation
- Test queries in Prometheus/Explore first
- Verify template variable substitution
- Check for null/empty results

### 2. Performance Testing
- Monitor dashboard load time
- Check query execution time
- Optimize slow panels

### 3. Visual Testing
- Test with different time ranges
- Verify threshold colors
- Check responsive layout
