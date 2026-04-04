---
name: observability-and-sre
description: Meta-orchestrator for observability and SRE practices. Routes to Prometheus, Grafana, distributed tracing, SLO implementation, and observability skills. Use when configuring Prometheus, building Grafana dashboards, implementing distributed tracing, defining SLOs, or setting up monitoring.
---

# Observability and SRE

Orchestrator for observability instrumentation and SRE practice implementation. Routes to the appropriate specialized skill based on the signal type, tooling, or reliability concern.

## Expert Agent

- **`sre-expert`**: Specialist for reliability engineering, SLO design, and observability stack architecture.
  - *Location*: `plugins/dev-suite/agents/sre-expert.md`
  - *Capabilities*: SLI/SLO/SLA definition, alerting strategy, distributed tracing, metrics pipelines, and incident response.

## Core Skills

### [Observability & SRE Practices](../observability-sre-practices/SKILL.md)
Observability pillars (metrics, logs, traces), on-call runbooks, and error budget policy.

### [Prometheus Configuration](../prometheus-configuration/SKILL.md)
Scrape configs, recording rules, alerting rules, and remote write setup.

### [Grafana Dashboards](../grafana-dashboards/SKILL.md)
Dashboard design, panel queries, templating variables, and alert notification channels.

### [Distributed Tracing](../distributed-tracing/SKILL.md)
OpenTelemetry instrumentation, trace context propagation, and Jaeger/Tempo backends.

### [SLO Implementation](../slo-implementation/SKILL.md)
SLI definition, error budget calculation, burn rate alerts, and SLO reporting.

## Routing Decision Tree

```
What is the observability concern?
|
+-- Overall strategy / runbooks / error budgets?
|   --> observability-sre-practices
|
+-- Prometheus scrape / rules / alertmanager?
|   --> prometheus-configuration
|
+-- Grafana panels / dashboards / alerts?
|   --> grafana-dashboards
|
+-- Request tracing / spans / context propagation?
|   --> distributed-tracing
|
+-- SLI definition / burn rate / error budget?
    --> slo-implementation
```

## Routing Table

| Trigger                                   | Sub-skill                    |
|-------------------------------------------|------------------------------|
| Runbook, on-call, error budget policy     | observability-sre-practices  |
| prometheus.yml, scrape, PromQL, rules     | prometheus-configuration     |
| Grafana, dashboard, panel, Loki query     | grafana-dashboards           |
| OpenTelemetry, spans, Jaeger, Tempo       | distributed-tracing          |
| SLO, SLI, burn rate, availability target | slo-implementation           |

## Checklist

- [ ] Identify the observability signal type (metrics / logs / traces) before routing
- [ ] Confirm SLIs are measurable from the user's perspective, not internal proxies
- [ ] Verify alerting rules have clear ownership and runbook links
- [ ] Check that distributed tracing covers all service-to-service boundaries
- [ ] Validate Grafana dashboards are reviewed by on-call engineers before production
- [ ] Ensure error budgets trigger team-wide reviews when consumed beyond threshold
