---
name: observability
version: "1.0.0"
description: Implement comprehensive monitoring, logging, and tracing using Prometheus, Grafana, and ELK/Loki. Master SLIs/SLOs, error budgets, and distributed tracing for distributed systems.
---

# Observability & Monitoring

Expert guide for maintaining system reliability through data-driven insights and proactive alerting.

## 1. Reliability Framework (SLO/SLI)

### Definitions
- **SLI (Indicator)**: The specific metric (e.g., 99th percentile latency).
- **SLO (Objective)**: The target value (e.g., Latency < 200ms for 99.9% of requests).
- **Error Budget**: The amount of "unreliability" allowed ($1 - SLO$).

### Budget Policy
- **100% Budget**: Normal feature development.
- **<10% Budget**: Freeze non-critical releases.
- **0% Budget**: Focus exclusively on reliability and technical debt.

## 2. Monitoring Stack

### Metrics (Prometheus/Grafana)
- **The Four Golden Signals**: Latency, Traffic, Errors, Saturation.
- **Alerting**: Use multi-window burn rate alerts to reduce false positives.

### Tracing & Logging
- **Distributed Tracing**: Use OpenTelemetry to track requests across microservices.
- **Structured Logging**: Log in JSON format with unique Request-IDs for correlation.

## 3. Infrastructure & Scientific Ops

- **Resource Monitoring**: Track CPU, GPU, and Memory utilization for heavy workloads.
- **Workflow Monitoring**: Use Airflow or Prefect dashboards to monitor complex scientific pipelines.

## 4. Observability Checklist

- [ ] **Golden Signals**: Are all four signals monitored for critical services?
- [ ] **Dashboards**: Are dashboards organized by audience (executive vs. technical)?
- [ ] **Alerting**: Are alerts actionable and linked to runbooks?
- [ ] **Tracing**: Is trace propagation configured across all service boundaries?
