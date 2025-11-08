# SLO Monitoring - Production-Ready Implementation Guide

This guide provides comprehensive Prometheus recording rules, alert configurations, and Grafana dashboard integration for implementing production-grade Service Level Objective (SLO) monitoring with multi-window multi-burn-rate alerting.

## Overview

Effective SLO monitoring requires:
- **Recording rules** for efficient metric aggregation
- **Multi-window calculations** for different time horizons
- **Burn rate alerts** for proactive error budget management
- **Dashboard integration** for visualization and analysis

## Table of Contents

1. [Prometheus Recording Rules](#prometheus-recording-rules)
2. [Multi-Window Success Rate Calculations](#multi-window-success-rate-calculations)
3. [Latency Percentile Tracking](#latency-percentile-tracking)
4. [Burn Rate Recording Rules](#burn-rate-recording-rules)
5. [Multi-Window Multi-Burn-Rate Alerts](#multi-window-multi-burn-rate-alerts)
6. [Complete Prometheus Configuration](#complete-prometheus-configuration)
7. [Alert Rule Configurations](#alert-rule-configurations)
8. [Grafana Dashboard Integration](#grafana-dashboard-integration)

## Prometheus Recording Rules

Recording rules pre-compute frequently used queries and store them as new time series. This improves query performance and enables consistent metric calculations across dashboards and alerts.

### Basic Recording Rules Structure

```yaml
# prometheus_recording_rules.yml
groups:
  - name: slo_recording_rules
    interval: 30s
    rules:
      # Recording rules will be defined here
```

### Request Rate Recording Rules

Track request rates across different dimensions:

```yaml
      # Total request rate per service
      - record: service:request_rate:5m
        expr: |
          sum(rate(http_requests_total[5m])) by (service)

      # Request rate by method and route
      - record: service:request_rate:by_endpoint:5m
        expr: |
          sum(rate(http_requests_total[5m])) by (service, method, route)

      # Request rate by status code
      - record: service:request_rate:by_status:5m
        expr: |
          sum(rate(http_requests_total[5m])) by (service, status)

      # Request rate excluding health checks
      - record: service:request_rate:excluding_health:5m
        expr: |
          sum(rate(http_requests_total{route!~"/health|/healthz|/ready|/metrics"}[5m])) by (service)
```

## Multi-Window Success Rate Calculations

Calculate success rates across multiple time windows to detect issues at different scales:

```yaml
      # 5-minute success rate (fast detection)
      - record: service:success_rate:5m
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[5m])) by (service)
            /
            sum(rate(http_requests_total[5m])) by (service)
          ) * 100

      # 30-minute success rate (short-term trend)
      - record: service:success_rate:30m
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[30m])) by (service)
            /
            sum(rate(http_requests_total[30m])) by (service)
          ) * 100

      # 1-hour success rate (medium-term stability)
      - record: service:success_rate:1h
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[1h])) by (service)
            /
            sum(rate(http_requests_total[1h])) by (service)
          ) * 100

      # 6-hour success rate (shift-level performance)
      - record: service:success_rate:6h
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[6h])) by (service)
            /
            sum(rate(http_requests_total[6h])) by (service)
          ) * 100

      # 24-hour success rate (daily performance)
      - record: service:success_rate:24h
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[24h])) by (service)
            /
            sum(rate(http_requests_total[24h])) by (service)
          ) * 100

      # 30-day success rate (SLO compliance window)
      - record: service:success_rate:30d
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[30d])) by (service)
            /
            sum(rate(http_requests_total[30d])) by (service)
          ) * 100
```

### Success Rate by Endpoint

Track success rates for specific endpoints to identify problematic routes:

```yaml
      # Success rate per endpoint (5m window)
      - record: service:success_rate:by_endpoint:5m
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[5m])) by (service, route)
            /
            sum(rate(http_requests_total[5m])) by (service, route)
          ) * 100

      # Success rate per endpoint (1h window)
      - record: service:success_rate:by_endpoint:1h
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[1h])) by (service, route)
            /
            sum(rate(http_requests_total[1h])) by (service, route)
          ) * 100
```

## Latency Percentile Tracking

Track latency percentiles using histogram metrics for comprehensive latency monitoring:

### P50 Latency (Median)

```yaml
      # P50 latency - 5 minute window
      - record: service:latency:p50:5m
        expr: |
          histogram_quantile(0.50,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      # P50 latency - 30 minute window
      - record: service:latency:p50:30m
        expr: |
          histogram_quantile(0.50,
            sum(rate(http_request_duration_seconds_bucket[30m])) by (service, le)
          )

      # P50 latency - 1 hour window
      - record: service:latency:p50:1h
        expr: |
          histogram_quantile(0.50,
            sum(rate(http_request_duration_seconds_bucket[1h])) by (service, le)
          )

      # P50 latency - 24 hour window
      - record: service:latency:p50:24h
        expr: |
          histogram_quantile(0.50,
            sum(rate(http_request_duration_seconds_bucket[24h])) by (service, le)
          )
```

### P95 Latency

```yaml
      # P95 latency - 5 minute window (fast detection)
      - record: service:latency:p95:5m
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      # P95 latency - 30 minute window
      - record: service:latency:p95:30m
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[30m])) by (service, le)
          )

      # P95 latency - 1 hour window
      - record: service:latency:p95:1h
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[1h])) by (service, le)
          )

      # P95 latency - 6 hour window
      - record: service:latency:p95:6h
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[6h])) by (service, le)
          )

      # P95 latency - 24 hour window
      - record: service:latency:p95:24h
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[24h])) by (service, le)
          )
```

### P99 Latency

```yaml
      # P99 latency - 5 minute window
      - record: service:latency:p99:5m
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      # P99 latency - 30 minute window
      - record: service:latency:p99:30m
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[30m])) by (service, le)
          )

      # P99 latency - 1 hour window
      - record: service:latency:p99:1h
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[1h])) by (service, le)
          )

      # P99 latency - 6 hour window
      - record: service:latency:p99:6h
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[6h])) by (service, le)
          )

      # P99 latency - 24 hour window
      - record: service:latency:p99:24h
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[24h])) by (service, le)
          )
```

### P99.9 Latency (Long Tail)

```yaml
      # P99.9 latency - 5 minute window
      - record: service:latency:p999:5m
        expr: |
          histogram_quantile(0.999,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      # P99.9 latency - 30 minute window
      - record: service:latency:p999:30m
        expr: |
          histogram_quantile(0.999,
            sum(rate(http_request_duration_seconds_bucket[30m])) by (service, le)
          )

      # P99.9 latency - 1 hour window
      - record: service:latency:p999:1h
        expr: |
          histogram_quantile(0.999,
            sum(rate(http_request_duration_seconds_bucket[1h])) by (service, le)
          )

      # P99.9 latency - 6 hour window
      - record: service:latency:p999:6h
        expr: |
          histogram_quantile(0.999,
            sum(rate(http_request_duration_seconds_bucket[6h])) by (service, le)
          )

      # P99.9 latency - 24 hour window
      - record: service:latency:p999:24h
        expr: |
          histogram_quantile(0.999,
            sum(rate(http_request_duration_seconds_bucket[24h])) by (service, le)
          )
```

### Latency by Endpoint

Track latency percentiles per endpoint:

```yaml
      # P95 latency by endpoint
      - record: service:latency:p95:by_endpoint:5m
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, route, le)
          )

      # P99 latency by endpoint
      - record: service:latency:p99:by_endpoint:5m
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, route, le)
          )
```

## Burn Rate Recording Rules

Error budget burn rate indicates how fast you're consuming your error budget. A burn rate of 1 means you're consuming budget at exactly the rate needed to exhaust it by the end of the SLO window.

### Burn Rate Calculation Formula

For a 99.9% availability SLO (0.1% error budget):
- Burn rate = (actual error rate) / (error budget rate)
- Burn rate = (1 - success_rate) / (1 - SLO_target)

### Multi-Window Burn Rates

```yaml
      # 5-minute burn rate (fastest detection)
      - record: service:error_budget_burn_rate:5m
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[5m])) by (service)
              /
              sum(rate(http_requests_total[5m])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"

      # 30-minute burn rate
      - record: service:error_budget_burn_rate:30m
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[30m])) by (service)
              /
              sum(rate(http_requests_total[30m])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"

      # 1-hour burn rate
      - record: service:error_budget_burn_rate:1h
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[1h])) by (service)
              /
              sum(rate(http_requests_total[1h])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"

      # 6-hour burn rate
      - record: service:error_budget_burn_rate:6h
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[6h])) by (service)
              /
              sum(rate(http_requests_total[6h])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"

      # 24-hour burn rate
      - record: service:error_budget_burn_rate:24h
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[24h])) by (service)
              /
              sum(rate(http_requests_total[24h])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"

      # 3-day burn rate
      - record: service:error_budget_burn_rate:3d
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[3d])) by (service)
              /
              sum(rate(http_requests_total[3d])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"
```

### Burn Rate for Different SLO Targets

For services with different SLO targets (99.5%, 99.9%, 99.95%):

```yaml
      # 99.5% SLO burn rate (1h window)
      - record: service:error_budget_burn_rate:1h:slo_995
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[1h])) by (service)
              /
              sum(rate(http_requests_total[1h])) by (service)
            )
          ) / (1 - 0.995)
        labels:
          slo_target: "99.5"

      # 99.95% SLO burn rate (1h window)
      - record: service:error_budget_burn_rate:1h:slo_9995
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[1h])) by (service)
              /
              sum(rate(http_requests_total[1h])) by (service)
            )
          ) / (1 - 0.9995)
        labels:
          slo_target: "99.95"
```

### Error Budget Remaining

Calculate remaining error budget:

```yaml
      # Error budget remaining (percentage)
      - record: service:error_budget_remaining:30d
        expr: |
          100 * (
            1 - (
              (1 - service:success_rate:30d / 100) /
              (1 - 0.999)
            )
          )
        labels:
          slo_target: "99.9"

      # Error budget consumed (minutes)
      - record: service:error_budget_consumed_minutes:30d
        expr: |
          (
            (1 - service:success_rate:30d / 100) /
            (1 - 0.999)
          ) * (30 * 24 * 60 * (1 - 0.999))
        labels:
          slo_target: "99.9"
```

## Multi-Window Multi-Burn-Rate Alerts

Implement Google's multi-window multi-burn-rate alerting methodology for precise and actionable alerts.

### Alert Methodology

**Fast Burn Alerts** (2% budget in 1 hour):
- Burn rate: 14.4x
- Short window: 5 minutes
- Long window: 1 hour
- Action: Page on-call

**Slow Burn Alerts** (10% budget in 6 hours):
- Burn rate: 3x
- Short window: 30 minutes
- Long window: 6 hours
- Action: Create ticket

### Fast Burn Alert Rules

```yaml
# prometheus_alert_rules.yml
groups:
  - name: slo_fast_burn_alerts
    interval: 30s
    rules:
      # Critical: Fast burn - exhausts 2% budget in 1 hour
      - alert: SLOErrorBudgetFastBurn
        expr: |
          (
            service:error_budget_burn_rate:5m{service="api"} > 14.4
            AND
            service:error_budget_burn_rate:1h{service="api"} > 14.4
          )
        for: 2m
        labels:
          severity: critical
          team: platform
          slo: availability
          burn_type: fast
        annotations:
          summary: "Critical: Fast error budget burn for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} is burning error budget at 14.4x rate.

            Current 5m burn rate: {{ printf "%.2f" (index (query "service:error_budget_burn_rate:5m{service='api'}") 0).Value }}x
            Current 1h burn rate: {{ printf "%.2f" $value }}x

            At this rate, 2% of monthly error budget will be consumed in 1 hour.

            Impact:
            - Monthly budget: 43.2 minutes (99.9% SLO)
            - Burning: 0.864 minutes per hour
            - Time to exhaust budget: ~50 hours if sustained

            Action Required:
            - Investigate immediately
            - Check for ongoing incidents
            - Review recent deployments
            - Examine error logs and metrics
          runbook: https://runbooks.example.com/slo-fast-burn
          dashboard: https://grafana.example.com/d/slo-dashboard/{{ $labels.service }}
          graph: https://prometheus.example.com/graph?g0.expr=service:error_budget_burn_rate:5m{service="{{ $labels.service }}"}

      # Critical: Very fast burn - multiple services
      - alert: SLOErrorBudgetVeryFastBurn
        expr: |
          (
            service:error_budget_burn_rate:5m > 28.8
            AND
            service:error_budget_burn_rate:1h > 28.8
          )
        for: 1m
        labels:
          severity: critical
          team: sre
          slo: availability
          burn_type: very_fast
        annotations:
          summary: "CRITICAL: Very fast error budget burn - {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} is burning error budget at {{ printf "%.1f" $value }}x rate!

            This will exhaust 4% of monthly budget in 1 hour.
            At this rate, entire budget exhausted in ~25 hours.

            IMMEDIATE ACTION REQUIRED - Potential major incident
          runbook: https://runbooks.example.com/slo-very-fast-burn
```

### Slow Burn Alert Rules

```yaml
      # Warning: Slow burn - exhausts 10% budget in 6 hours
      - alert: SLOErrorBudgetSlowBurn
        expr: |
          (
            service:error_budget_burn_rate:30m{service="api"} > 3
            AND
            service:error_budget_burn_rate:6h{service="api"} > 3
          )
        for: 15m
        labels:
          severity: warning
          team: platform
          slo: availability
          burn_type: slow
        annotations:
          summary: "Warning: Slow error budget burn for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} is burning error budget at 3x rate.

            Current 30m burn rate: {{ printf "%.2f" (index (query "service:error_budget_burn_rate:30m{service='api'}") 0).Value }}x
            Current 6h burn rate: {{ printf "%.2f" $value }}x

            At this rate, 10% of monthly error budget will be consumed in 6 hours.

            Impact:
            - Monthly budget: 43.2 minutes (99.9% SLO)
            - Burning: 0.216 minutes per hour
            - Time to exhaust budget: ~200 hours if sustained

            Action Required:
            - Investigate within 1 hour
            - Create incident ticket
            - Review error trends
            - Plan remediation
          runbook: https://runbooks.example.com/slo-slow-burn
          dashboard: https://grafana.example.com/d/slo-dashboard/{{ $labels.service }}

      # Info: Moderate burn - tracking only
      - alert: SLOErrorBudgetModerateBurn
        expr: |
          (
            service:error_budget_burn_rate:1h > 2
            AND
            service:error_budget_burn_rate:6h > 2
          )
        for: 30m
        labels:
          severity: info
          team: platform
          slo: availability
          burn_type: moderate
        annotations:
          summary: "Info: Moderate error budget burn for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} is burning error budget at {{ printf "%.2f" $value }}x rate.

            This is above normal but below alerting threshold.
            Monitor for trends and investigate if sustained.
          runbook: https://runbooks.example.com/slo-moderate-burn
```

### Latency SLO Alerts

```yaml
      # Latency SLO fast burn
      - alert: SLOLatencyFastBurn
        expr: |
          (
            (
              sum(rate(http_request_duration_seconds_bucket{le="0.5"}[5m])) by (service)
              /
              sum(rate(http_request_duration_seconds_count[5m])) by (service)
            ) < 0.95
            AND
            (
              sum(rate(http_request_duration_seconds_bucket{le="0.5"}[1h])) by (service)
              /
              sum(rate(http_request_duration_seconds_count[1h])) by (service)
            ) < 0.95
          )
        for: 2m
        labels:
          severity: critical
          team: platform
          slo: latency
          burn_type: fast
        annotations:
          summary: "Critical: Latency SLO fast burn for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} latency SLO is being violated.

            Target: 95% of requests < 500ms
            Current (5m): {{ printf "%.2f" (mul $value 100) }}% of requests < 500ms

            Current P95 latency: {{ printf "%.0f" (mul (index (query "service:latency:p95:5m{service='" $labels.service "'}") 0).Value 1000) }}ms

            Action Required:
            - Investigate latency issues immediately
            - Check for slow database queries
            - Review recent deployments
            - Examine application performance
          runbook: https://runbooks.example.com/slo-latency-burn
```

### Budget Exhaustion Alerts

```yaml
      # Error budget nearly exhausted
      - alert: SLOErrorBudgetNearExhaustion
        expr: |
          service:error_budget_remaining:30d < 10
        for: 5m
        labels:
          severity: warning
          team: platform
          slo: availability
        annotations:
          summary: "Warning: Error budget nearly exhausted for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} has only {{ printf "%.1f" $value }}% error budget remaining.

            Current status:
            - SLO target: 99.9%
            - Current performance: {{ printf "%.3f" (index (query "service:success_rate:30d{service='" $labels.service "'}") 0).Value }}%
            - Budget remaining: {{ printf "%.1f" $value }}%

            Actions:
            - Freeze non-critical deployments
            - Focus on reliability improvements
            - Defer risky changes
            - Review incident trends
          runbook: https://runbooks.example.com/slo-budget-exhaustion

      # Error budget exhausted
      - alert: SLOErrorBudgetExhausted
        expr: |
          service:error_budget_remaining:30d <= 0
        for: 1m
        labels:
          severity: critical
          team: platform
          slo: availability
        annotations:
          summary: "CRITICAL: Error budget exhausted for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} has exhausted its error budget!

            Current SLO compliance: {{ printf "%.3f" (index (query "service:success_rate:30d{service='" $labels.service "'}") 0).Value }}%
            SLO target: 99.9%
            Budget deficit: {{ printf "%.1f" (mul $value -1) }}%

            REQUIRED ACTIONS:
            - STOP all non-critical deployments
            - Escalate to engineering leadership
            - Convene incident review
            - Create recovery plan
            - Focus exclusively on reliability
          runbook: https://runbooks.example.com/slo-budget-exhausted
```

## Complete Prometheus Configuration

### Full Recording Rules File

```yaml
# /etc/prometheus/rules/slo_recording_rules.yml
groups:
  - name: slo_recording_rules
    interval: 30s
    rules:
      # ============================================
      # Request Rate Metrics
      # ============================================

      - record: service:request_rate:5m
        expr: |
          sum(rate(http_requests_total[5m])) by (service)

      - record: service:request_rate:by_endpoint:5m
        expr: |
          sum(rate(http_requests_total[5m])) by (service, method, route)

      - record: service:request_rate:by_status:5m
        expr: |
          sum(rate(http_requests_total[5m])) by (service, status)

      # ============================================
      # Success Rate Metrics (Multi-Window)
      # ============================================

      - record: service:success_rate:5m
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[5m])) by (service)
            /
            sum(rate(http_requests_total[5m])) by (service)
          ) * 100

      - record: service:success_rate:30m
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[30m])) by (service)
            /
            sum(rate(http_requests_total[30m])) by (service)
          ) * 100

      - record: service:success_rate:1h
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[1h])) by (service)
            /
            sum(rate(http_requests_total[1h])) by (service)
          ) * 100

      - record: service:success_rate:6h
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[6h])) by (service)
            /
            sum(rate(http_requests_total[6h])) by (service)
          ) * 100

      - record: service:success_rate:24h
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[24h])) by (service)
            /
            sum(rate(http_requests_total[24h])) by (service)
          ) * 100

      - record: service:success_rate:30d
        expr: |
          (
            sum(rate(http_requests_total{status!~"5.."}[30d])) by (service)
            /
            sum(rate(http_requests_total[30d])) by (service)
          ) * 100

      # ============================================
      # Latency Percentiles (P50)
      # ============================================

      - record: service:latency:p50:5m
        expr: |
          histogram_quantile(0.50,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:latency:p50:30m
        expr: |
          histogram_quantile(0.50,
            sum(rate(http_request_duration_seconds_bucket[30m])) by (service, le)
          )

      - record: service:latency:p50:1h
        expr: |
          histogram_quantile(0.50,
            sum(rate(http_request_duration_seconds_bucket[1h])) by (service, le)
          )

      - record: service:latency:p50:24h
        expr: |
          histogram_quantile(0.50,
            sum(rate(http_request_duration_seconds_bucket[24h])) by (service, le)
          )

      # ============================================
      # Latency Percentiles (P95)
      # ============================================

      - record: service:latency:p95:5m
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:latency:p95:30m
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[30m])) by (service, le)
          )

      - record: service:latency:p95:1h
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[1h])) by (service, le)
          )

      - record: service:latency:p95:6h
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[6h])) by (service, le)
          )

      - record: service:latency:p95:24h
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[24h])) by (service, le)
          )

      # ============================================
      # Latency Percentiles (P99)
      # ============================================

      - record: service:latency:p99:5m
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:latency:p99:30m
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[30m])) by (service, le)
          )

      - record: service:latency:p99:1h
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[1h])) by (service, le)
          )

      - record: service:latency:p99:6h
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[6h])) by (service, le)
          )

      - record: service:latency:p99:24h
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[24h])) by (service, le)
          )

      # ============================================
      # Latency Percentiles (P99.9)
      # ============================================

      - record: service:latency:p999:5m
        expr: |
          histogram_quantile(0.999,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:latency:p999:30m
        expr: |
          histogram_quantile(0.999,
            sum(rate(http_request_duration_seconds_bucket[30m])) by (service, le)
          )

      - record: service:latency:p999:1h
        expr: |
          histogram_quantile(0.999,
            sum(rate(http_request_duration_seconds_bucket[1h])) by (service, le)
          )

      - record: service:latency:p999:6h
        expr: |
          histogram_quantile(0.999,
            sum(rate(http_request_duration_seconds_bucket[6h])) by (service, le)
          )

      - record: service:latency:p999:24h
        expr: |
          histogram_quantile(0.999,
            sum(rate(http_request_duration_seconds_bucket[24h])) by (service, le)
          )

      # ============================================
      # Error Budget Burn Rate (Multi-Window)
      # ============================================

      - record: service:error_budget_burn_rate:5m
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[5m])) by (service)
              /
              sum(rate(http_requests_total[5m])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"

      - record: service:error_budget_burn_rate:30m
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[30m])) by (service)
              /
              sum(rate(http_requests_total[30m])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"

      - record: service:error_budget_burn_rate:1h
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[1h])) by (service)
              /
              sum(rate(http_requests_total[1h])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"

      - record: service:error_budget_burn_rate:6h
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[6h])) by (service)
              /
              sum(rate(http_requests_total[6h])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"

      - record: service:error_budget_burn_rate:24h
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[24h])) by (service)
              /
              sum(rate(http_requests_total[24h])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"

      - record: service:error_budget_burn_rate:3d
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[3d])) by (service)
              /
              sum(rate(http_requests_total[3d])) by (service)
            )
          ) / (1 - 0.999)
        labels:
          slo_target: "99.9"

      # ============================================
      # Error Budget Status
      # ============================================

      - record: service:error_budget_remaining:30d
        expr: |
          100 * (
            1 - (
              (1 - service:success_rate:30d / 100) /
              (1 - 0.999)
            )
          )
        labels:
          slo_target: "99.9"

      - record: service:error_budget_consumed_minutes:30d
        expr: |
          (
            (1 - service:success_rate:30d / 100) /
            (1 - 0.999)
          ) * (30 * 24 * 60 * (1 - 0.999))
        labels:
          slo_target: "99.9"
```

### Prometheus Configuration

```yaml
# /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 30s
  external_labels:
    cluster: production
    environment: prod

rule_files:
  - /etc/prometheus/rules/slo_recording_rules.yml
  - /etc/prometheus/rules/slo_alert_rules.yml

scrape_configs:
  - job_name: 'api-service'
    static_configs:
      - targets: ['api:8080']
        labels:
          service: api

  - job_name: 'web-service'
    static_configs:
      - targets: ['web:3000']
        labels:
          service: web

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

## Alert Rule Configurations

### Complete Alert Rules File

```yaml
# /etc/prometheus/rules/slo_alert_rules.yml
groups:
  - name: slo_fast_burn_alerts
    interval: 30s
    rules:
      - alert: SLOErrorBudgetFastBurn
        expr: |
          (
            service:error_budget_burn_rate:5m > 14.4
            AND
            service:error_budget_burn_rate:1h > 14.4
          )
        for: 2m
        labels:
          severity: critical
          team: platform
          slo: availability
          burn_type: fast
          priority: P0
        annotations:
          summary: "Critical: Fast error budget burn for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} is burning error budget at 14.4x normal rate.
            This will exhaust 2% of monthly budget in 1 hour.

            Immediate investigation required.
          runbook: https://runbooks.example.com/slo-fast-burn
          dashboard: https://grafana.example.com/d/slo/{{ $labels.service }}

  - name: slo_slow_burn_alerts
    interval: 30s
    rules:
      - alert: SLOErrorBudgetSlowBurn
        expr: |
          (
            service:error_budget_burn_rate:30m > 3
            AND
            service:error_budget_burn_rate:6h > 3
          )
        for: 15m
        labels:
          severity: warning
          team: platform
          slo: availability
          burn_type: slow
          priority: P1
        annotations:
          summary: "Warning: Slow error budget burn for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} is burning error budget at 3x normal rate.
            This will exhaust 10% of monthly budget in 6 hours.

            Investigation required within 1 hour.
          runbook: https://runbooks.example.com/slo-slow-burn
          dashboard: https://grafana.example.com/d/slo/{{ $labels.service }}

  - name: slo_budget_exhaustion
    interval: 1m
    rules:
      - alert: SLOErrorBudgetNearExhaustion
        expr: service:error_budget_remaining:30d < 10
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "Error budget nearly exhausted: {{ $labels.service }}"
          description: "Only {{ printf \"%.1f\" $value }}% budget remaining"

      - alert: SLOErrorBudgetExhausted
        expr: service:error_budget_remaining:30d <= 0
        for: 1m
        labels:
          severity: critical
          team: leadership
        annotations:
          summary: "CRITICAL: Error budget exhausted for {{ $labels.service }}"
          description: "Freeze all non-critical deployments"
```

## Grafana Dashboard Integration

### Dashboard JSON Configuration

```json
{
  "dashboard": {
    "title": "Service SLO Dashboard",
    "tags": ["slo", "reliability", "sre"],
    "timezone": "browser",
    "schemaVersion": 30,
    "version": 1,
    "refresh": "30s",

    "templating": {
      "list": [
        {
          "name": "service",
          "type": "query",
          "datasource": "Prometheus",
          "query": "label_values(http_requests_total, service)",
          "refresh": 1,
          "includeAll": false,
          "multi": false
        },
        {
          "name": "slo_target",
          "type": "custom",
          "options": [
            {"text": "99.9%", "value": "0.999"},
            {"text": "99.95%", "value": "0.9995"},
            {"text": "99.5%", "value": "0.995"}
          ],
          "current": {"text": "99.9%", "value": "0.999"}
        }
      ]
    },

    "panels": [
      {
        "id": 1,
        "title": "SLO Compliance (30-day rolling)",
        "type": "stat",
        "gridPos": {"h": 6, "w": 6, "x": 0, "y": 0},
        "targets": [{
          "expr": "service:success_rate:30d{service=\"$service\"}",
          "legendFormat": "30-day SLO",
          "refId": "A"
        }],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "red", "value": null},
                {"color": "yellow", "value": 99.5},
                {"color": "green", "value": 99.9}
              ]
            },
            "unit": "percent",
            "decimals": 3,
            "mappings": []
          }
        },
        "options": {
          "colorMode": "background",
          "graphMode": "area",
          "justifyMode": "auto",
          "orientation": "auto",
          "reduceOptions": {
            "values": false,
            "calcs": ["lastNotNull"],
            "fields": ""
          },
          "textMode": "value_and_name"
        }
      },

      {
        "id": 2,
        "title": "Error Budget Remaining",
        "type": "gauge",
        "gridPos": {"h": 6, "w": 6, "x": 6, "y": 0},
        "targets": [{
          "expr": "service:error_budget_remaining:30d{service=\"$service\"}",
          "legendFormat": "Budget Remaining",
          "refId": "A"
        }],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 100,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "red", "value": null},
                {"color": "orange", "value": 10},
                {"color": "yellow", "value": 25},
                {"color": "green", "value": 50}
              ]
            },
            "unit": "percent"
          }
        },
        "options": {
          "showThresholdLabels": true,
          "showThresholdMarkers": true
        }
      },

      {
        "id": 3,
        "title": "Current Burn Rate (1h)",
        "type": "stat",
        "gridPos": {"h": 6, "w": 6, "x": 12, "y": 0},
        "targets": [{
          "expr": "service:error_budget_burn_rate:1h{service=\"$service\"}",
          "legendFormat": "Burn Rate",
          "refId": "A"
        }],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 2},
                {"color": "orange", "value": 5},
                {"color": "red", "value": 10}
              ]
            },
            "unit": "short",
            "decimals": 2,
            "mappings": []
          }
        },
        "options": {
          "colorMode": "background",
          "graphMode": "area",
          "textMode": "value_and_name"
        }
      },

      {
        "id": 4,
        "title": "Request Rate",
        "type": "stat",
        "gridPos": {"h": 6, "w": 6, "x": 18, "y": 0},
        "targets": [{
          "expr": "service:request_rate:5m{service=\"$service\"}",
          "legendFormat": "Requests/sec",
          "refId": "A"
        }],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "decimals": 0
          }
        },
        "options": {
          "colorMode": "value",
          "graphMode": "area",
          "textMode": "value_and_name"
        }
      },

      {
        "id": 5,
        "title": "Success Rate Trend (Multi-Window)",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6},
        "targets": [
          {
            "expr": "service:success_rate:5m{service=\"$service\"}",
            "legendFormat": "5m success rate",
            "refId": "A"
          },
          {
            "expr": "service:success_rate:1h{service=\"$service\"}",
            "legendFormat": "1h success rate",
            "refId": "B"
          },
          {
            "expr": "service:success_rate:24h{service=\"$service\"}",
            "legendFormat": "24h success rate",
            "refId": "C"
          },
          {
            "expr": "service:success_rate:30d{service=\"$service\"}",
            "legendFormat": "30d success rate",
            "refId": "D"
          }
        ],
        "yaxes": [
          {
            "format": "percent",
            "label": "Success Rate",
            "min": "99",
            "max": "100"
          },
          {
            "format": "short",
            "show": false
          }
        ],
        "thresholds": [
          {
            "value": 99.9,
            "colorMode": "critical",
            "op": "lt",
            "fill": true,
            "line": true,
            "yaxis": "left"
          }
        ]
      },

      {
        "id": 6,
        "title": "Burn Rate Trend (Multi-Window)",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6},
        "targets": [
          {
            "expr": "service:error_budget_burn_rate:5m{service=\"$service\"}",
            "legendFormat": "5m burn rate",
            "refId": "A"
          },
          {
            "expr": "service:error_budget_burn_rate:1h{service=\"$service\"}",
            "legendFormat": "1h burn rate",
            "refId": "B"
          },
          {
            "expr": "service:error_budget_burn_rate:6h{service=\"$service\"}",
            "legendFormat": "6h burn rate",
            "refId": "C"
          },
          {
            "expr": "service:error_budget_burn_rate:24h{service=\"$service\"}",
            "legendFormat": "24h burn rate",
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
        "thresholds": [
          {
            "value": 1,
            "colorMode": "custom",
            "op": "gt",
            "fill": false,
            "line": true,
            "lineColor": "rgba(255, 255, 0, 0.5)",
            "yaxis": "left"
          },
          {
            "value": 14.4,
            "colorMode": "critical",
            "op": "gt",
            "fill": true,
            "line": true,
            "yaxis": "left"
          }
        ]
      },

      {
        "id": 7,
        "title": "Latency Percentiles (5m window)",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 14},
        "targets": [
          {
            "expr": "service:latency:p50:5m{service=\"$service\"} * 1000",
            "legendFormat": "P50",
            "refId": "A"
          },
          {
            "expr": "service:latency:p95:5m{service=\"$service\"} * 1000",
            "legendFormat": "P95",
            "refId": "B"
          },
          {
            "expr": "service:latency:p99:5m{service=\"$service\"} * 1000",
            "legendFormat": "P99",
            "refId": "C"
          },
          {
            "expr": "service:latency:p999:5m{service=\"$service\"} * 1000",
            "legendFormat": "P99.9",
            "refId": "D"
          }
        ],
        "yaxes": [
          {
            "format": "ms",
            "label": "Latency",
            "min": 0,
            "logBase": 1
          }
        ],
        "thresholds": [
          {
            "value": 500,
            "colorMode": "custom",
            "op": "gt",
            "fill": false,
            "line": true,
            "lineColor": "rgba(255, 165, 0, 0.7)",
            "yaxis": "left"
          },
          {
            "value": 1000,
            "colorMode": "critical",
            "op": "gt",
            "fill": false,
            "line": true,
            "yaxis": "left"
          }
        ]
      },

      {
        "id": 8,
        "title": "Error Budget Projection",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 14},
        "targets": [
          {
            "expr": "service:error_budget_remaining:30d{service=\"$service\"}",
            "legendFormat": "Budget Remaining (%)",
            "refId": "A"
          },
          {
            "expr": "service:error_budget_consumed_minutes:30d{service=\"$service\"}",
            "legendFormat": "Budget Consumed (minutes)",
            "refId": "B"
          }
        ],
        "yaxes": [
          {
            "format": "percent",
            "label": "Budget Remaining",
            "min": 0,
            "max": 100
          },
          {
            "format": "m",
            "label": "Minutes Consumed",
            "min": 0
          }
        ],
        "seriesOverrides": [
          {
            "alias": "Budget Consumed (minutes)",
            "yaxis": 2
          }
        ]
      },

      {
        "id": 9,
        "title": "Request Rate by Status Code",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 22},
        "targets": [{
          "expr": "service:request_rate:by_status:5m{service=\"$service\"}",
          "legendFormat": "{{ status }}",
          "refId": "A"
        }],
        "yaxes": [
          {
            "format": "reqps",
            "label": "Requests/sec"
          }
        ],
        "stack": true
      },

      {
        "id": 10,
        "title": "Top 10 Slowest Endpoints (P95)",
        "type": "table",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 22},
        "targets": [{
          "expr": "topk(10, service:latency:p95:by_endpoint:5m{service=\"$service\"} * 1000)",
          "legendFormat": "{{ route }}",
          "refId": "A",
          "instant": true,
          "format": "table"
        }],
        "transformations": [
          {
            "id": "organize",
            "options": {
              "excludeByName": {
                "Time": true,
                "service": true,
                "method": false,
                "route": false,
                "Value": false
              },
              "renameByName": {
                "route": "Endpoint",
                "method": "Method",
                "Value": "P95 Latency (ms)"
              }
            }
          }
        ]
      }
    ]
  }
}
```

### Python Script to Create Dashboard

```python
#!/usr/bin/env python3
"""
create_slo_dashboard.py - Create Grafana SLO dashboard programmatically
"""

import json
import requests
from typing import Dict, List, Any

class GrafanaSLODashboard:
    def __init__(self, grafana_url: str, api_key: str):
        self.grafana_url = grafana_url
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def create_slo_dashboard(self, service_name: str) -> Dict[str, Any]:
        """Create comprehensive SLO dashboard for a service"""
        dashboard = {
            "dashboard": {
                "title": f"SLO Dashboard - {service_name}",
                "tags": ["slo", "reliability", service_name],
                "timezone": "browser",
                "refresh": "30s",
                "panels": self._create_panels(service_name),
                "templating": self._create_templating(service_name)
            },
            "overwrite": True
        }

        response = requests.post(
            f"{self.grafana_url}/api/dashboards/db",
            headers=self.headers,
            json=dashboard
        )

        return response.json()

    def _create_panels(self, service: str) -> List[Dict[str, Any]]:
        """Create all dashboard panels"""
        return [
            self._slo_compliance_panel(service, 1),
            self._error_budget_gauge(service, 2),
            self._burn_rate_stat(service, 3),
            self._request_rate_stat(service, 4),
            self._success_rate_graph(service, 5),
            self._burn_rate_graph(service, 6),
            self._latency_graph(service, 7),
            self._budget_projection_graph(service, 8)
        ]

    def _create_templating(self, service: str) -> Dict[str, Any]:
        """Create dashboard variables"""
        return {
            "list": [
                {
                    "name": "service",
                    "type": "constant",
                    "current": {"value": service}
                },
                {
                    "name": "slo_target",
                    "type": "custom",
                    "options": [
                        {"text": "99.9%", "value": "0.999"},
                        {"text": "99.95%", "value": "0.9995"}
                    ],
                    "current": {"text": "99.9%", "value": "0.999"}
                }
            ]
        }

    def _slo_compliance_panel(self, service: str, panel_id: int) -> Dict[str, Any]:
        """SLO compliance stat panel"""
        return {
            "id": panel_id,
            "title": "SLO Compliance (30-day)",
            "type": "stat",
            "gridPos": {"h": 6, "w": 6, "x": 0, "y": 0},
            "targets": [{
                "expr": f'service:success_rate:30d{{service="{service}"}}',
                "legendFormat": "30-day SLO"
            }],
            "fieldConfig": {
                "defaults": {
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 99.5},
                            {"color": "green", "value": 99.9}
                        ]
                    },
                    "unit": "percent",
                    "decimals": 3
                }
            }
        }

if __name__ == "__main__":
    # Example usage
    dashboard_creator = GrafanaSLODashboard(
        grafana_url="http://grafana:3000",
        api_key="your-api-key"
    )

    result = dashboard_creator.create_slo_dashboard("api-service")
    print(f"Dashboard created: {result}")
```

## Implementation Checklist

### Recording Rules Setup

- [ ] Deploy recording rules to Prometheus
- [ ] Verify recording rules are evaluating correctly
- [ ] Check recording rule performance and resource usage
- [ ] Validate metric cardinality is within acceptable limits
- [ ] Test recording rules with sample data

### Alert Configuration

- [ ] Deploy alert rules to Prometheus
- [ ] Configure Alertmanager routing
- [ ] Set up PagerDuty/Slack integration
- [ ] Test fast burn alerts with synthetic incidents
- [ ] Test slow burn alerts with gradual degradation
- [ ] Verify alert annotations and runbook links
- [ ] Configure alert severity levels and escalation

### Dashboard Integration

- [ ] Import dashboard JSON to Grafana
- [ ] Configure data source connections
- [ ] Test dashboard with live data
- [ ] Set up dashboard permissions
- [ ] Create dashboard links in runbooks
- [ ] Configure dashboard refresh intervals
- [ ] Set up dashboard alerts (optional)

### Validation

- [ ] Verify SLO calculations match expected values
- [ ] Test with historical incident data
- [ ] Validate burn rate calculations
- [ ] Check latency percentile accuracy
- [ ] Verify error budget calculations
- [ ] Test multi-window alert logic

### Documentation

- [ ] Document SLO targets and rationale
- [ ] Create runbooks for each alert type
- [ ] Document escalation procedures
- [ ] Create SLO review process
- [ ] Document error budget policy
- [ ] Train team on SLO monitoring

## Best Practices

### Recording Rule Design

1. **Use appropriate intervals**: 30s for SLO rules, 1m for less critical metrics
2. **Limit cardinality**: Avoid high-cardinality labels in recording rules
3. **Pre-aggregate intelligently**: Balance between query performance and storage
4. **Use consistent naming**: Follow `level:metric:operations` convention

### Alert Tuning

1. **Start conservative**: Begin with higher thresholds and tune based on experience
2. **Use multi-window**: Prevent false positives with short and long window validation
3. **Set appropriate for durations**: Balance between fast detection and noise reduction
4. **Provide context**: Include actionable information in alert annotations

### Dashboard Design

1. **Show multiple time windows**: Enable pattern recognition across scales
2. **Use appropriate visualizations**: Stats for current state, graphs for trends
3. **Set meaningful thresholds**: Visual indicators at SLO boundaries
4. **Optimize refresh rates**: Balance between freshness and load

### Operational Excellence

1. **Regular SLO reviews**: Weekly review of SLO performance and burn rates
2. **Incident correlation**: Link SLO violations to incidents for analysis
3. **Continuous improvement**: Iterate on SLO definitions based on learning
4. **Team alignment**: Ensure all stakeholders understand SLO implications
5. **Error budget policy**: Establish clear guidelines for budget usage

## Troubleshooting

### Common Issues

**Recording rules not updating:**
- Check Prometheus logs for evaluation errors
- Verify rule syntax with `promtool check rules`
- Ensure sufficient Prometheus resources

**Alerts not firing:**
- Verify alert expression returns data
- Check alert state in Prometheus UI
- Review Alertmanager configuration
- Validate routing rules

**Dashboard showing no data:**
- Verify data source configuration
- Check metric names match recording rules
- Ensure time range includes data
- Review Prometheus query logs

**Incorrect burn rate calculations:**
- Verify SLO target in recording rule labels
- Check for missing data points
- Validate rate() window sizes
- Review metric retention period

## References

- Google SRE Book: SLO Implementation
- Multi-Window Multi-Burn-Rate Alerts (Google)
- Prometheus Recording Rules Best Practices
- Grafana Dashboard Design Guidelines
- Error Budget Policy Templates
