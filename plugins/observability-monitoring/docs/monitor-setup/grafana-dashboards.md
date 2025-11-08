# Grafana Dashboards: Complete Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-07
**Grafana Version:** 10.x / 11.x

## Overview

This guide provides comprehensive coverage of Grafana dashboard design, configuration, and best practices for modern observability. Learn to build production-grade dashboards for monitoring distributed systems using Golden Signals, RED metrics, and USE metrics methodologies.

## Table of Contents

1. [Dashboard Architecture](#dashboard-architecture)
2. [Panel Types](#panel-types)
3. [Golden Signals Dashboards](#golden-signals-dashboards)
4. [RED Metrics Dashboards](#red-metrics-dashboards)
5. [USE Metrics Dashboards](#use-metrics-dashboards)
6. [Template Variables](#template-variables)
7. [Dashboard Provisioning](#dashboard-provisioning)
8. [Alerting Integration](#alerting-integration)
9. [Complete Dashboard Examples](#complete-dashboard-examples)
10. [Best Practices](#best-practices)

---

## Dashboard Architecture

### Dashboard JSON Structure

Every Grafana dashboard is defined by a JSON structure with key components:

```json
{
  "dashboard": {
    "id": null,
    "uid": "service-overview",
    "title": "Service Overview",
    "tags": ["service", "production"],
    "timezone": "browser",
    "schemaVersion": 38,
    "version": 1,
    "refresh": "30s",
    "time": {
      "from": "now-6h",
      "to": "now"
    },
    "timepicker": {
      "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"],
      "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d"]
    },
    "templating": {
      "list": []
    },
    "annotations": {
      "list": []
    },
    "panels": [],
    "editable": true,
    "fiscalYearStartMonth": 0,
    "graphTooltip": 1,
    "links": []
  },
  "overwrite": true
}
```

### Key Components

**Dashboard Metadata:**
- `uid`: Unique identifier for URL and provisioning
- `title`: Display name
- `tags`: Categorization for search and filtering
- `version`: Auto-incremented on save

**Time Configuration:**
- `time.from` / `time.to`: Default time range
- `refresh`: Auto-refresh interval
- `timezone`: Display timezone (browser, utc, or specific timezone)

**Templating:**
- Variables for dynamic filtering
- Enables multi-service, multi-environment dashboards

**Panels:**
- Visualization components
- Queries, transformations, and display settings

---

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

## Golden Signals Dashboards

The Four Golden Signals (from Google's SRE book): Latency, Traffic, Errors, and Saturation.

### Complete Golden Signals Dashboard

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

---

## RED Metrics Dashboards

RED metrics focus on: Rate, Errors, Duration - ideal for request-driven systems.

### RED Metrics Dashboard

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

---

## USE Metrics Dashboards

USE metrics for resource monitoring: Utilization, Saturation, Errors.

### USE Metrics Dashboard

```json
{
  "dashboard": {
    "uid": "use-metrics",
    "title": "USE Metrics - Resource Monitoring",
    "tags": ["use-metrics", "resources", "infrastructure"],
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
          "type": "query",
          "name": "instance",
          "label": "Instance",
          "datasource": "Prometheus",
          "query": "label_values(node_cpu_seconds_total, instance)",
          "refresh": 1,
          "multi": true,
          "includeAll": true
        }
      ]
    },
    "panels": [
      {
        "type": "row",
        "title": "CPU - Utilization, Saturation, Errors",
        "gridPos": {
          "h": 1,
          "w": 24,
          "x": 0,
          "y": 0
        }
      },
      {
        "type": "gauge",
        "title": "CPU Utilization",
        "description": "Percentage of CPU time spent in non-idle states",
        "gridPos": {
          "h": 6,
          "w": 6,
          "x": 0,
          "y": 1
        },
        "targets": [
          {
            "expr": "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\",instance=~\"$instance\"}[5m])) * 100)",
            "legendFormat": "{{instance}}",
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
        "type": "timeseries",
        "title": "CPU Saturation (Load Average)",
        "description": "CPU load average - saturation when > number of cores",
        "gridPos": {
          "h": 6,
          "w": 9,
          "x": 6,
          "y": 1
        },
        "targets": [
          {
            "expr": "node_load1{instance=~\"$instance\"}",
            "legendFormat": "{{instance}} 1m",
            "refId": "A"
          },
          {
            "expr": "node_load5{instance=~\"$instance\"}",
            "legendFormat": "{{instance}} 5m",
            "refId": "B"
          },
          {
            "expr": "count by (instance) (node_cpu_seconds_total{mode=\"idle\",instance=~\"$instance\"})",
            "legendFormat": "{{instance}} cores",
            "refId": "C"
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
        "title": "CPU Context Switches (Saturation)",
        "description": "High context switches indicate CPU saturation",
        "gridPos": {
          "h": 6,
          "w": 9,
          "x": 15,
          "y": 1
        },
        "targets": [
          {
            "expr": "rate(node_context_switches_total{instance=~\"$instance\"}[5m])",
            "legendFormat": "{{instance}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ops",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 1
            }
          }
        }
      },
      {
        "type": "row",
        "title": "Memory - Utilization, Saturation, Errors",
        "gridPos": {
          "h": 1,
          "w": 24,
          "x": 0,
          "y": 7
        }
      },
      {
        "type": "gauge",
        "title": "Memory Utilization",
        "description": "Percentage of memory in use",
        "gridPos": {
          "h": 6,
          "w": 6,
          "x": 0,
          "y": 8
        },
        "targets": [
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes{instance=~\"$instance\"} / node_memory_MemTotal_bytes{instance=~\"$instance\"})) * 100",
            "legendFormat": "{{instance}}",
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
        "title": "Memory Saturation (Swap Usage)",
        "description": "Swap usage indicates memory saturation",
        "gridPos": {
          "h": 6,
          "w": 9,
          "x": 6,
          "y": 8
        },
        "targets": [
          {
            "expr": "(1 - (node_memory_SwapFree_bytes{instance=~\"$instance\"} / node_memory_SwapTotal_bytes{instance=~\"$instance\"})) * 100",
            "legendFormat": "{{instance}} swap used",
            "refId": "A"
          },
          {
            "expr": "rate(node_vmstat_pswpin{instance=~\"$instance\"}[5m])",
            "legendFormat": "{{instance}} swap in",
            "refId": "B"
          },
          {
            "expr": "rate(node_vmstat_pswpout{instance=~\"$instance\"}[5m])",
            "legendFormat": "{{instance}} swap out",
            "refId": "C"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2
            }
          }
        }
      },
      {
        "type": "stat",
        "title": "Page Faults (Errors)",
        "description": "Major page faults indicate memory pressure",
        "gridPos": {
          "h": 6,
          "w": 9,
          "x": 15,
          "y": 8
        },
        "targets": [
          {
            "expr": "rate(node_vmstat_pgmajfault{instance=~\"$instance\"}[5m])",
            "legendFormat": "{{instance}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ops",
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"},
                {"value": 10, "color": "yellow"},
                {"value": 50, "color": "red"}
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
        "type": "row",
        "title": "Disk - Utilization, Saturation, Errors",
        "gridPos": {
          "h": 1,
          "w": 24,
          "x": 0,
          "y": 14
        }
      },
      {
        "type": "gauge",
        "title": "Disk Space Utilization",
        "description": "Percentage of disk space used",
        "gridPos": {
          "h": 6,
          "w": 6,
          "x": 0,
          "y": 15
        },
        "targets": [
          {
            "expr": "(1 - (node_filesystem_avail_bytes{instance=~\"$instance\",fstype!~\"tmpfs|fuse.*\"} / node_filesystem_size_bytes{instance=~\"$instance\",fstype!~\"tmpfs|fuse.*\"})) * 100",
            "legendFormat": "{{instance}} {{mountpoint}}",
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
                {"value": 80, "color": "yellow"},
                {"value": 90, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "type": "timeseries",
        "title": "Disk I/O Saturation (Queue Depth)",
        "description": "Disk queue length indicates I/O saturation",
        "gridPos": {
          "h": 6,
          "w": 9,
          "x": 6,
          "y": 15
        },
        "targets": [
          {
            "expr": "rate(node_disk_io_time_weighted_seconds_total{instance=~\"$instance\"}[5m])",
            "legendFormat": "{{instance}} {{device}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2
            },
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "green"},
                {"value": 1, "color": "yellow"},
                {"value": 5, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "type": "timeseries",
        "title": "Disk I/O Errors",
        "description": "Read/write errors on disk",
        "gridPos": {
          "h": 6,
          "w": 9,
          "x": 15,
          "y": 15
        },
        "targets": [
          {
            "expr": "rate(node_disk_read_errors_total{instance=~\"$instance\"}[5m])",
            "legendFormat": "{{instance}} {{device}} read errors",
            "refId": "A"
          },
          {
            "expr": "rate(node_disk_write_errors_total{instance=~\"$instance\"}[5m])",
            "legendFormat": "{{instance}} {{device}} write errors",
            "refId": "B"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ops",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2
            }
          }
        }
      },
      {
        "type": "row",
        "title": "Network - Utilization, Saturation, Errors",
        "gridPos": {
          "h": 1,
          "w": 24,
          "x": 0,
          "y": 21
        }
      },
      {
        "type": "timeseries",
        "title": "Network Utilization (Throughput)",
        "description": "Network bandwidth usage",
        "gridPos": {
          "h": 6,
          "w": 12,
          "x": 0,
          "y": 22
        },
        "targets": [
          {
            "expr": "rate(node_network_receive_bytes_total{instance=~\"$instance\",device!~\"lo\"}[5m]) * 8",
            "legendFormat": "{{instance}} {{device}} rx",
            "refId": "A"
          },
          {
            "expr": "rate(node_network_transmit_bytes_total{instance=~\"$instance\",device!~\"lo\"}[5m]) * 8",
            "legendFormat": "{{instance}} {{device}} tx",
            "refId": "B"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "bps",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2,
              "fillOpacity": 10
            }
          }
        }
      },
      {
        "type": "timeseries",
        "title": "Network Saturation (Drops)",
        "description": "Dropped packets indicate network saturation",
        "gridPos": {
          "h": 6,
          "w": 6,
          "x": 12,
          "y": 22
        },
        "targets": [
          {
            "expr": "rate(node_network_receive_drop_total{instance=~\"$instance\",device!~\"lo\"}[5m])",
            "legendFormat": "{{instance}} {{device}} rx drops",
            "refId": "A"
          },
          {
            "expr": "rate(node_network_transmit_drop_total{instance=~\"$instance\",device!~\"lo\"}[5m])",
            "legendFormat": "{{instance}} {{device}} tx drops",
            "refId": "B"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "pps",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2
            }
          }
        }
      },
      {
        "type": "timeseries",
        "title": "Network Errors",
        "description": "Network transmission errors",
        "gridPos": {
          "h": 6,
          "w": 6,
          "x": 18,
          "y": 22
        },
        "targets": [
          {
            "expr": "rate(node_network_receive_errs_total{instance=~\"$instance\",device!~\"lo\"}[5m])",
            "legendFormat": "{{instance}} {{device}} rx errors",
            "refId": "A"
          },
          {
            "expr": "rate(node_network_transmit_errs_total{instance=~\"$instance\",device!~\"lo\"}[5m])",
            "legendFormat": "{{instance}} {{device}} tx errors",
            "refId": "B"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "pps",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2
            }
          }
        }
      }
    ]
  }
}
```

---

## Template Variables

Template variables enable dynamic, reusable dashboards.

### Variable Types

**Query Variable:**
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

**Custom Variable:**
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

**Interval Variable:**
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

**Datasource Variable:**
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

**Constant Variable:**
```json
{
  "type": "constant",
  "name": "cluster",
  "label": "Cluster",
  "query": "us-west-2",
  "hide": 2
}
```

### Advanced Variable Usage

**Chained Variables:**
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

**Multi-Value with All:**
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

**Text Box Variable:**
```json
{
  "type": "textbox",
  "name": "threshold",
  "label": "Alert Threshold",
  "query": "0.95"
}
```

---

## Dashboard Provisioning

Automate dashboard deployment using provisioning.

### Provisioning Configuration

**provisioning/dashboards/default.yml:**
```yaml
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: 'Services'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/dashboards
      foldersFromFilesStructure: true
```

### Dashboard File Structure

```
/etc/grafana/dashboards/
├── services/
│   ├── api-gateway.json
│   ├── user-service.json
│   └── payment-service.json
├── infrastructure/
│   ├── kubernetes.json
│   ├── nodes.json
│   └── network.json
└── business/
    ├── sales.json
    └── conversions.json
```

### ConfigMap for Kubernetes

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: monitoring
data:
  service-overview.json: |
    {
      "dashboard": {
        "uid": "service-overview",
        "title": "Service Overview",
        "panels": [...]
      }
    }
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-provider
  namespace: monitoring
data:
  dashboards.yml: |
    apiVersion: 1
    providers:
      - name: 'default'
        orgId: 1
        folder: ''
        type: file
        disableDeletion: false
        options:
          path: /etc/grafana/dashboards
```

### Terraform Provisioning

```hcl
resource "grafana_dashboard" "service_overview" {
  config_json = file("${path.module}/dashboards/service-overview.json")
  folder      = grafana_folder.services.id
  overwrite   = true
}

resource "grafana_folder" "services" {
  title = "Services"
}
```

---

## Alerting Integration

Configure alerts directly from dashboard panels.

### Alert Rule Configuration

```json
{
  "type": "timeseries",
  "title": "Error Rate",
  "targets": [
    {
      "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100",
      "refId": "A"
    }
  ],
  "alert": {
    "name": "High Error Rate",
    "message": "Error rate is above threshold",
    "conditions": [
      {
        "evaluator": {
          "params": [5],
          "type": "gt"
        },
        "operator": {
          "type": "and"
        },
        "query": {
          "params": ["A", "5m", "now"]
        },
        "reducer": {
          "params": [],
          "type": "avg"
        },
        "type": "query"
      }
    ],
    "executionErrorState": "alerting",
    "for": "5m",
    "frequency": "1m",
    "handler": 1,
    "noDataState": "no_data",
    "notifications": [
      {
        "uid": "slack-alerts"
      }
    ]
  }
}
```

### Grafana 9+ Unified Alerting

**Alert Rule:**
```json
{
  "uid": "high-error-rate",
  "title": "High Error Rate",
  "condition": "C",
  "data": [
    {
      "refId": "A",
      "queryType": "range",
      "relativeTimeRange": {
        "from": 300,
        "to": 0
      },
      "datasourceUid": "prometheus-uid",
      "model": {
        "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100",
        "refId": "A"
      }
    },
    {
      "refId": "C",
      "queryType": "",
      "relativeTimeRange": {
        "from": 0,
        "to": 0
      },
      "datasourceUid": "-100",
      "model": {
        "conditions": [
          {
            "evaluator": {
              "params": [5],
              "type": "gt"
            },
            "operator": {
              "type": "and"
            },
            "query": {
              "params": ["A"]
            },
            "reducer": {
              "params": [],
              "type": "avg"
            },
            "type": "query"
          }
        ],
        "refId": "C"
      }
    }
  ],
  "noDataState": "NoData",
  "execErrState": "Alerting",
  "for": "5m",
  "annotations": {
    "description": "Error rate has exceeded 5% for the last 5 minutes",
    "runbook_url": "https://wiki.example.com/runbooks/high-error-rate",
    "summary": "Service {{$labels.job}} has high error rate"
  },
  "labels": {
    "severity": "critical",
    "team": "platform"
  }
}
```

### Alert Notification Templates

```json
{
  "name": "Slack Critical Alerts",
  "type": "slack",
  "settings": {
    "url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    "recipient": "#alerts-critical",
    "username": "Grafana",
    "icon_emoji": ":grafana:",
    "mentionChannel": "here",
    "message": "{{ template \"slack.default.message\" . }}"
  }
}
```

---

## Complete Dashboard Examples

### Microservices Dashboard

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

### Kubernetes Cluster Dashboard

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

---

## Best Practices

### Dashboard Design Principles

**1. Focus on User Needs:**
- Design for your audience (developers, SREs, business stakeholders)
- Show actionable metrics
- Prioritize most important information at the top

**2. Layout and Organization:**
- Use rows to group related panels
- Consistent panel sizing (use 24-column grid)
- Top: High-level stats and KPIs
- Middle: Time series graphs
- Bottom: Detailed tables and logs

**3. Performance Optimization:**
- Limit panels per dashboard (< 30 for best performance)
- Use appropriate query intervals
- Leverage template variables to reduce queries
- Avoid wildcard queries when possible

**4. Color and Thresholds:**
- Use semantic colors (green = good, yellow = warning, red = critical)
- Set meaningful thresholds based on SLOs
- Consistent color scheme across dashboards

**5. Naming Conventions:**
- Clear, descriptive panel titles
- Consistent legend formatting
- Use units in panel titles when helpful

### Query Best Practices

**Efficient PromQL:**
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

**Template Variable Usage:**
```promql
# Use regex matching for multi-select
metric{job=~"$service"}

# Use pipe for OR conditions
metric{status=~"$status"}  # where $status = "200|201|204"
```

### Dashboard Versioning

**Git Repository Structure:**
```
dashboards/
├── README.md
├── production/
│   ├── services/
│   │   ├── api-gateway.json
│   │   └── user-service.json
│   └── infrastructure/
│       └── kubernetes.json
├── staging/
│   └── ...
└── scripts/
    ├── deploy.sh
    └── validate.sh
```

**Validation Script:**
```bash
#!/bin/bash
# validate.sh

for dashboard in dashboards/**/*.json; do
  echo "Validating $dashboard"
  jq empty "$dashboard" || exit 1

  # Check required fields
  jq -e '.dashboard.uid' "$dashboard" > /dev/null || {
    echo "Missing UID in $dashboard"
    exit 1
  }
done

echo "All dashboards valid"
```

### Documentation Standards

**Dashboard Description:**
- Add description to dashboard JSON
- Document template variables
- Include links to runbooks
- Add annotations for deployments and incidents

**Panel Descriptions:**
```json
{
  "type": "timeseries",
  "title": "Request Latency",
  "description": "p99 request latency across all services. Alert fires if > 1s for 5 minutes. See runbook: https://wiki.example.com/latency",
  "targets": [...]
}
```

### Security Considerations

**1. Data Source Permissions:**
- Use datasource variables
- Restrict access via Grafana RBAC
- Separate dashboards for different environments

**2. Variable Injection Protection:**
- Avoid user-input variables in sensitive queries
- Use constant variables for critical values
- Validate regex patterns

**3. Dashboard Access Control:**
```json
{
  "dashboard": {
    "uid": "prod-services",
    "title": "Production Services",
    "tags": ["production", "restricted"]
  },
  "folderId": 5,
  "folderUid": "prod-folder",
  "overwrite": true
}
```

### Testing Dashboards

**1. Query Validation:**
- Test queries in Prometheus/Explore first
- Verify template variable substitution
- Check for null/empty results

**2. Performance Testing:**
- Monitor dashboard load time
- Check query execution time
- Optimize slow panels

**3. Visual Testing:**
- Test with different time ranges
- Verify threshold colors
- Check responsive layout

### Maintenance and Updates

**Regular Reviews:**
- Quarterly dashboard audits
- Remove unused panels
- Update queries for schema changes
- Align with current SLOs/SLIs

**Change Management:**
- Version control all dashboards
- Code review for changes
- Test in staging before production
- Document breaking changes

---

## Conclusion

This comprehensive guide covers Grafana dashboard design from fundamentals to advanced patterns. Key takeaways:

- **Structure**: Use consistent JSON structure with proper metadata
- **Panels**: Choose appropriate visualization types for your data
- **Methodologies**: Implement Golden Signals, RED, or USE metrics
- **Variables**: Enable dynamic, reusable dashboards
- **Provisioning**: Automate deployment with GitOps
- **Alerting**: Integrate alerts directly from panels
- **Best Practices**: Focus on actionable metrics and user needs

**Next Steps:**
1. Start with a simple dashboard template
2. Customize panels for your metrics
3. Add template variables for flexibility
4. Implement provisioning for automation
5. Configure alerts for critical metrics
6. Iterate based on user feedback

**Additional Resources:**
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Documentation](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [RED Method](https://www.weave.works/blog/the-red-method-key-metrics-for-microservices-architecture/)
- [USE Method](http://www.brendangregg.com/usemethod.html)

---

**Document Information:**
- **Author**: Claude Code Plugin - Observability & Monitoring
- **Version**: 1.0.0
- **Last Updated**: 2025-11-07
- **License**: MIT
