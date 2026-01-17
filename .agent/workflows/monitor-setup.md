---
description: Set up Prometheus, Grafana, and distributed tracing observability stack
triggers:
- /monitor-setup
- set up prometheus, grafana,
allowed-tools: [Bash, Write, Read, Task]
version: 1.0.0
---



# Monitoring and Observability Setup

$ARGUMENTS

## External Docs

[Prometheus Setup](../../plugins/observability-monitoring/docs/monitor-setup/prometheus-setup.md) (~1,220 lines), [Grafana Dashboards](../../plugins/observability-monitoring/docs/monitor-setup/grafana-dashboards.md) (~1,470), [Distributed Tracing](../../plugins/observability-monitoring/docs/monitor-setup/distributed-tracing.md) (~1,368), [Log Aggregation](../../plugins/observability-monitoring/docs/monitor-setup/log-aggregation.md) (~1,200), [Alerting Strategies](../../plugins/observability-monitoring/docs/monitor-setup/alerting-strategies.md) (~1,327), [Infrastructure Code](../../plugins/observability-monitoring/docs/monitor-setup/infrastructure-code.md) (~1,200)

## Phase 1: Planning

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 2h | Essential requirements, basic stack |
| Standard | 4h | Full assessment, SLO/SLA |
| Enterprise | 1d | Multi-cluster, compliance |

## Phase 2: Core Setup

| Component | Quick | Standard | Enterprise |
|-----------|-------|----------|------------|
| Duration | 4h | 2d | 1w |
| Prometheus | Basic scraping | + Recording rules, federation | + HA, long-term storage |
| Grafana | Golden signals | + RED/USE dashboards | + Multi-org, provisioning |
| Instrumentation | Key metrics | Comprehensive | Full coverage |

## Phase 3: Advanced (Standard+)

| Component | Standard | Enterprise |
|-----------|----------|------------|
| Tracing | OpenTelemetry + Jaeger | + Tail-based sampling |
| Logs | Fluentd + Elasticsearch | + Trace ID correlation |
| Advanced | - | DB, network, security monitoring |

## Phase 4: Validation

| Mode | Duration | Deliverables |
|------|----------|--------------|
| Quick | 2h | Service health, 2+ critical alerts |
| Standard | 1d | + Team/exec dashboards, multi-window alerts, chaos testing |
| Enterprise | 3d | + Full hierarchy, runbook automation, validation |

## Dashboards

- Golden Signals: Latency, traffic, errors, saturation
- RED: Rate, errors, duration (service health)
- USE: Utilization, saturation, errors (resources)

## Alerts

- Critical: >5% error, >2s latency → Page
- Warning: >1% error, >500ms latency → Ticket
- Info: Approaching threshold → Dashboard

## Success Criteria

**Quick:**
- Prometheus collecting, Grafana showing golden signals, 2+ alerts, team ready

**Standard:**
- All Quick + >80% trace coverage, searchable logs, multi-window burn rate, IaC deployed, RED/USE dashboards

**Enterprise:**
- All Standard + multi-cluster federation, metrics/logs/traces correlated, SLO tracking, 30%+ cost optimization, security integrated
