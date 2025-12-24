---
version: "1.0.6"
command: /monitor-setup
description: Set up Prometheus, Grafana, and distributed tracing observability stack
execution_modes:
  quick: "1-2 days"
  standard: "1 week"
  enterprise: "2-3 weeks"
workflow_type: "hybrid"
interactive_mode: true
---

# Monitoring and Observability Setup

Set up comprehensive monitoring covering metrics, logs, and traces.

## Requirements

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope | Agents |
|------|----------|-------|--------|
| Quick | 1-2 days | Basic Prometheus + Grafana + alerts | observability-engineer |
| Standard (default) | 1 week | + Tracing, logs, IaC, multi-window alerts | + performance-engineer |
| Enterprise | 2-3 weeks | + Multi-cluster, SLO tracking, security monitoring | + database-optimizer, network-engineer |

---

## External Documentation

| Document | Content | Lines |
|----------|---------|-------|
| [Prometheus Setup](../docs/monitor-setup/prometheus-setup.md) | Config, service discovery, recording rules | ~1,220 |
| [Grafana Dashboards](../docs/monitor-setup/grafana-dashboards.md) | Panels, RED/USE metrics, provisioning | ~1,470 |
| [Distributed Tracing](../docs/monitor-setup/distributed-tracing.md) | OpenTelemetry, Jaeger, sampling | ~1,368 |
| [Log Aggregation](../docs/monitor-setup/log-aggregation.md) | Fluentd, Elasticsearch, correlation | ~1,200 |
| [Alerting Strategies](../docs/monitor-setup/alerting-strategies.md) | Alertmanager, routing, runbooks | ~1,327 |
| [Infrastructure as Code](../docs/monitor-setup/infrastructure-code.md) | Terraform, Helm, Docker Compose | ~1,200 |

**Total:** ~13,000 lines

---

## Phase 1: Planning

| Step | Quick | Standard | Enterprise |
|------|-------|----------|------------|
| Duration | 2h | 4h | 1 day |
| Infrastructure assessment | Basic | Full | Multi-cluster |
| Stack design | Minimal | Complete | Federated |
| Requirements gathering | Essential | SLO/SLA | Compliance |

---

## Phase 2: Core Setup

| Component | Quick | Standard | Enterprise |
|-----------|-------|----------|------------|
| Duration | 4h | 2 days | 1 week |
| Prometheus | Basic scraping | + Recording rules, federation | + HA, long-term storage |
| Grafana | Golden signals dashboard | + RED/USE dashboards | + Multi-org, provisioning |
| Instrumentation | Key metrics | Comprehensive | Full coverage |

---

## Phase 3: Advanced Integration (Standard+)

| Component | Standard | Enterprise |
|-----------|----------|------------|
| Distributed Tracing | OpenTelemetry + Jaeger | + Tail-based sampling |
| Log Aggregation | Fluentd + Elasticsearch | + Correlation by trace ID |
| Advanced Monitoring | - | Database, network, security |

---

## Phase 4: Validation & Optimization

| Task | Quick | Standard | Enterprise |
|------|-------|----------|------------|
| Duration | 2h | 1 day | 3 days |
| Dashboards | Service health | Team + exec dashboards | Full hierarchy |
| Alerts | 2 critical | Multi-window burn rate | + Runbook automation |
| Testing | Basic validation | Chaos testing | Full observability validation |

---

## Dashboard Types

| Dashboard | Metrics | Use Case |
|-----------|---------|----------|
| Golden Signals | Latency, traffic, errors, saturation | Overview |
| RED | Rate, errors, duration | Service health |
| USE | Utilization, saturation, errors | Resource health |

---

## Alert Strategy

| Type | Threshold | Action |
|------|-----------|--------|
| Critical | >5% error rate, >2s latency | Page on-call |
| Warning | >1% error rate, >500ms latency | Create ticket |
| Info | Approaching threshold | Dashboard notification |

---

## Output Deliverables

| Mode | Deliverables |
|------|--------------|
| Quick | Prometheus config, Grafana dashboard, 2+ alerts |
| Standard | + Tracing, logs, IaC (Terraform/Helm), full dashboards |
| Enterprise | + Federation, metrics-logs-traces correlation, SLO tracking |

---

## Success Criteria

### Quick Mode
- [ ] Prometheus collecting from target service
- [ ] Grafana showing golden signals
- [ ] 2+ critical alerts configured
- [ ] Team can view and respond

### Standard Mode
- [ ] All Quick criteria
- [ ] Tracing >80% request coverage
- [ ] Logs searchable in Elasticsearch
- [ ] Multi-window burn rate alerting
- [ ] Infrastructure as code deployed
- [ ] RED/USE dashboards for all services

### Enterprise Mode
- [ ] All Standard criteria
- [ ] Multi-cluster federation operational
- [ ] Metrics/logs/traces correlated by trace ID
- [ ] SLO error budgets tracked
- [ ] 30%+ cost optimization achieved
- [ ] Security monitoring integrated
