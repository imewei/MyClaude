# Alerting Integration

Configure alerts directly from dashboard panels.

## Alert Rule Configuration

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

## Grafana 9+ Unified Alerting

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

## Alert Notification Templates

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
