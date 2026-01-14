---
name: slo-implementation
version: "1.0.7"
maturity: "5-Expert"
specialization: SRE Reliability
description: Define SLIs, SLOs, error budgets, and burn rate alerting following SRE best practices. Use when establishing reliability targets, implementing error budget policies, creating SLO dashboards, or designing multi-window burn rate alerts.
---

# SLO Implementation

Service Level Objectives with error budgets and burn rate alerting.

---

## SLI/SLO/SLA Hierarchy

```
SLA (Service Level Agreement) → Contract with customers
  ↓
SLO (Service Level Objective) → Internal reliability target
  ↓
SLI (Service Level Indicator) → Actual measurement
```

---

## Common SLI Types

| Type | Formula | PromQL |
|------|---------|--------|
| Availability | Success / Total | `sum(rate(http_requests_total{status!~"5.."}[28d])) / sum(rate(http_requests_total[28d]))` |
| Latency | Fast / Total | `sum(rate(http_request_duration_seconds_bucket{le="0.5"}[28d])) / sum(rate(http_request_duration_seconds_count[28d]))` |
| Durability | Writes OK / Total | `sum(storage_writes_successful_total) / sum(storage_writes_total)` |

---

## Availability Targets

| SLO % | Downtime/Month | Downtime/Year |
|-------|----------------|---------------|
| 99% | 7.2 hours | 3.65 days |
| 99.9% | 43.2 minutes | 8.76 hours |
| 99.95% | 21.6 minutes | 4.38 hours |
| 99.99% | 4.32 minutes | 52.56 minutes |

---

## Error Budget

**Formula**: `Error Budget = 1 - SLO Target`

**Example (99.9% SLO)**:
- Error Budget: 0.1% = 43.2 min/month
- Current Error: 0.05% = 21.6 min/month
- Remaining Budget: 50%

```yaml
error_budget_policy:
  - remaining: 100% → Normal development
  - remaining: 50%  → Postpone risky changes
  - remaining: 10%  → Freeze non-critical
  - remaining: 0%   → Feature freeze, focus reliability
```

---

## SLO Recording Rules

```yaml
groups:
  - name: sli_rules
    rules:
      - record: sli:http_availability:ratio
        expr: |
          sum(rate(http_requests_total{status!~"5.."}[28d]))
          / sum(rate(http_requests_total[28d]))

      - record: slo:http_availability:error_budget_remaining
        expr: (sli:http_availability:ratio - 0.999) / (1 - 0.999) * 100

      - record: slo:http_availability:burn_rate_5m
        expr: |
          (1 - sum(rate(http_requests_total{status!~"5.."}[5m]))
          / sum(rate(http_requests_total[5m]))) / (1 - 0.999)
```

---

## Multi-Window Burn Rate Alerts

| Window | Burn Rate | Budget Consumed | Severity |
|--------|-----------|-----------------|----------|
| 1h + 5m | 14.4x | 2% in 1 hour | Critical |
| 6h + 30m | 6x | 5% in 6 hours | Warning |

```yaml
- alert: SLOErrorBudgetBurnFast
  expr: |
    slo:http_availability:burn_rate_1h > 14.4
    and slo:http_availability:burn_rate_5m > 14.4
  for: 2m
  labels: { severity: critical }

- alert: SLOErrorBudgetBurnSlow
  expr: |
    slo:http_availability:burn_rate_6h > 6
    and slo:http_availability:burn_rate_30m > 6
  for: 15m
  labels: { severity: warning }
```

---

## SLO Dashboard

```
┌─────────────────────────────────┐
│ SLO Compliance: 99.95%          │
│ (Target: 99.9%) ✓               │
├─────────────────────────────────┤
│ Error Budget: ████████░░ 65%    │
├─────────────────────────────────┤
│ SLI Trend (28 days)             │
├─────────────────────────────────┤
│ Burn Rate Analysis              │
└─────────────────────────────────┘
```

**Queries**:
- Current: `sli:http_availability:ratio * 100`
- Budget: `slo:http_availability:error_budget_remaining`
- Days left: `(budget_remaining/100) * 28 / burn_rate`

---

## Review Cadence

| Frequency | Focus |
|-----------|-------|
| Weekly | Compliance, budget status, trends |
| Monthly | Achievement, incident impact, adjustments |
| Quarterly | Relevance, target changes, process improvements |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Start user-facing | Most visible services first |
| Multiple SLIs | Availability + latency + others |
| Achievable targets | Don't aim for 100% |
| Multi-window alerts | Reduces false positives |
| Regular review | Adjust SLOs based on data |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| SLO too aggressive | Start conservative, tighten over time |
| Single-window alerts | Use multi-window burn rate |
| Ignoring error budget | Make budget visible to teams |
| No policy | Define clear budget exhaustion actions |

---

## Checklist

- [ ] SLIs defined (availability, latency)
- [ ] SLO targets set realistically
- [ ] Recording rules implemented
- [ ] Multi-window burn rate alerts
- [ ] Error budget dashboard
- [ ] Budget policy documented
- [ ] Review cadence established

---

**Version**: 1.0.5
