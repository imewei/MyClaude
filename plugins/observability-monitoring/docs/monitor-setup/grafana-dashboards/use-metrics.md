# USE Metrics Dashboards

USE metrics for resource monitoring: Utilization, Saturation, Errors.

## USE Metrics Dashboard

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
