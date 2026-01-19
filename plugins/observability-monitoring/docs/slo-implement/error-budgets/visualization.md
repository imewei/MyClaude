# Error Budget Visualization

## Grafana Dashboard JSON

```python
def create_error_budget_dashboard(
    service_name: str,
    slo_target: float = 99.9
) -> dict:
    """
    Generate complete Grafana dashboard for error budget monitoring.

    Args:
        service_name: Name of the service
        slo_target: SLO target percentage

    Returns:
        Grafana dashboard JSON
    """
    return {
        "dashboard": {
            "title": f"Error Budget - {service_name}",
            "tags": ["slo", "error-budget", service_name],
            "timezone": "browser",
            "refresh": "30s",
            "time": {
                "from": "now-7d",
                "to": "now"
            },
            "panels": [
                # Row 1: Summary Stats
                {
                    "id": 1,
                    "title": "Budget Remaining",
                    "type": "gauge",
                    "gridPos": {"x": 0, "y": 0, "w": 6, "h": 8},
                    "targets": [{
                        "expr": f'''
                        100 * (
                            1 - (
                                (100 - service:success_rate_30d{{service="{service_name}"}}) /
                                (100 - {slo_target})
                            )
                        )
                        ''',
                        "refId": "A"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "min": 0,
                            "max": 100,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "red"},
                                    {"value": 10, "color": "orange"},
                                    {"value": 25, "color": "yellow"},
                                    {"value": 50, "color": "green"}
                                ]
                            }
                        }
                    }
                },
                {
                    "id": 2,
                    "title": "Current Burn Rate",
                    "type": "stat",
                    "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
                    "targets": [{
                        "expr": f'service:burn_rate_1h{{service="{service_name}"}}',
                        "refId": "A"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "short",
                            "decimals": 1,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "green"},
                                    {"value": 1, "color": "yellow"},
                                    {"value": 3, "color": "orange"},
                                    {"value": 6, "color": "red"}
                                ]
                            }
                        }
                    },
                    "options": {
                        "textMode": "value_and_name"
                    }
                },
                {
                    "id": 3,
                    "title": "SLO Compliance (30d)",
                    "type": "stat",
                    "gridPos": {"x": 12, "y": 0, "w": 6, "h": 4},
                    "targets": [{
                        "expr": f'service:success_rate_30d{{service="{service_name}"}}',
                        "refId": "A"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "decimals": 3,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "red"},
                                    {"value": slo_target - 0.1, "color": "orange"},
                                    {"value": slo_target, "color": "green"}
                                ]
                            }
                        }
                    }
                },
                {
                    "id": 4,
                    "title": "Time to Exhaustion",
                    "type": "stat",
                    "gridPos": {"x": 18, "y": 0, "w": 6, "h": 4},
                    "targets": [{
                        "expr": f'''
                        30 * (
                            100 * (
                                1 - (
                                    (100 - service:success_rate_30d{{service="{service_name}"}}) /
                                    (100 - {slo_target})
                                )
                            ) / 100
                        ) / service:burn_rate_1h{{service="{service_name}"}}
                        ''',
                        "refId": "A"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "d",
                            "decimals": 1,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "red"},
                                    {"value": 3, "color": "orange"},
                                    {"value": 7, "color": "yellow"},
                                    {"value": 15, "color": "green"}
                                ]
                            }
                        }
                    }
                },

                # Row 2: Burn Rate Trend
                {
                    "id": 10,
                    "title": "Burn Rate Over Time",
                    "type": "graph",
                    "gridPos": {"x": 0, "y": 8, "w": 24, "h": 8},
                    "targets": [
                        {
                            "expr": f'service:burn_rate_5m{{service="{service_name}"}}',
                            "legendFormat": "5min",
                            "refId": "A"
                        },
                        {
                            "expr": f'service:burn_rate_1h{{service="{service_name}"}}',
                            "legendFormat": "1hour",
                            "refId": "B"
                        },
                        {
                            "expr": f'service:burn_rate_6h{{service="{service_name}"}}',
                            "legendFormat": "6hour",
                            "refId": "C"
                        },
                        {
                            "expr": f'service:burn_rate_24h{{service="{service_name}"}}',
                            "legendFormat": "24hour",
                            "refId": "D"
                        }
                    ],
                    "yaxes": [
                        {
                            "format": "short",
                            "label": "Burn Rate (x)",
                            "min": 0,
                            "logBase": 1
                        }
                    ],
                    "seriesOverrides": [
                        {
                            "alias": "5min",
                            "linewidth": 1
                        },
                        {
                            "alias": "1hour",
                            "linewidth": 2
                        }
                    ],
                    "thresholds": [
                        {
                            "value": 1,
                            "op": "gt",
                            "fill": False,
                            "line": True,
                            "colorMode": "custom",
                            "lineColor": "rgba(255, 255, 0, 0.7)"
                        },
                        {
                            "value": 3,
                            "op": "gt",
                            "colorMode": "custom",
                            "lineColor": "rgba(255, 165, 0, 0.7)"
                        },
                        {
                            "value": 14.4,
                            "op": "gt",
                            "colorMode": "custom",
                            "lineColor": "rgba(255, 0, 0, 0.7)"
                        }
                    ]
                },

                # Row 3: Budget Consumption Trend
                {
                    "id": 20,
                    "title": "Error Budget Consumption (30-day rolling)",
                    "type": "graph",
                    "gridPos": {"x": 0, "y": 16, "w": 24, "h": 8},
                    "targets": [{
                        "expr": f'''
                        100 * (
                            (100 - service:success_rate_30d{{service="{service_name}"}}) /
                            (100 - {slo_target})
                        )
                        ''',
                        "legendFormat": "Budget Consumed %",
                        "refId": "A"
                    }],
                    "yaxes": [
                        {
                            "format": "percent",
                            "label": "Budget Consumed",
                            "min": 0,
                            "max": 100
                        }
                    ],
                    "thresholds": [
                        {
                            "value": 50,
                            "colorMode": "custom",
                            "fill": True,
                            "op": "gt",
                            "fillColor": "rgba(255, 255, 0, 0.1)"
                        },
                        {
                            "value": 75,
                            "colorMode": "custom",
                            "fill": True,
                            "op": "gt",
                            "fillColor": "rgba(255, 165, 0, 0.2)"
                        },
                        {
                            "value": 90,
                            "colorMode": "custom",
                            "fill": True,
                            "op": "gt",
                            "fillColor": "rgba(255, 0, 0, 0.3)"
                        }
                    ]
                },

                # Row 4: Error Rate Details
                {
                    "id": 30,
                    "title": "Error Rate by Window",
                    "type": "graph",
                    "gridPos": {"x": 0, "y": 24, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": f'service:error_rate_5m{{service="{service_name}"}}',
                            "legendFormat": "5min",
                            "refId": "A"
                        },
                        {
                            "expr": f'service:error_rate_1h{{service="{service_name}"}}',
                            "legendFormat": "1hour",
                            "refId": "B"
                        },
                        {
                            "expr": f'service:error_rate_24h{{service="{service_name}"}}',
                            "legendFormat": "24hour",
                            "refId": "C"
                        }
                    ],
                    "yaxes": [
                        {
                            "format": "percent",
                            "label": "Error Rate",
                            "min": 0
                        }
                    ]
                },
                {
                    "id": 31,
                    "title": "Request Rate",
                    "type": "graph",
                    "gridPos": {"x": 12, "y": 24, "w": 12, "h": 8},
                    "targets": [{
                        "expr": f'sum(rate(http_requests_total{{service="{service_name}"}}[5m]))',
                        "legendFormat": "Requests/sec",
                        "refId": "A"
                    }],
                    "yaxes": [
                        {
                            "format": "reqps",
                            "label": "Requests/sec",
                            "min": 0
                        }
                    ]
                }
            ],
            "templating": {
                "list": [
                    {
                        "name": "service",
                        "type": "constant",
                        "current": {
                            "value": service_name
                        }
                    },
                    {
                        "name": "slo_target",
                        "type": "constant",
                        "current": {
                            "value": str(slo_target)
                        }
                    }
                ]
            },
            "annotations": {
                "list": [
                    {
                        "datasource": "Prometheus",
                        "enable": True,
                        "expr": f'ALERTS{{alertname=~"ErrorBudget.*", service="{service_name}"}}',
                        "iconColor": "red",
                        "name": "Error Budget Alerts",
                        "tagKeys": "severity",
                        "textFormat": "{{alertname}}",
                        "titleFormat": "Alert"
                    },
                    {
                        "datasource": "Prometheus",
                        "enable": True,
                        "expr": f'changes(service:success_rate_30d{{service="{service_name}"}}[5m]) != 0',
                        "iconColor": "blue",
                        "name": "Deployments",
                        "titleFormat": "Deploy"
                    }
                ]
            }
        }
    }
```
