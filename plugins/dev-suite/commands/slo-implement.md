---
name: slo-implement
command: /slo-implement
description: Interactive workflow to implement SLO/SLA monitoring, error budgets, and burn rate alerting for a service
argument-hint: "[service-name] [--mode=quick|standard|enterprise]"
execution-modes:
  quick: "2-3d: 1 service, 1-2 SLIs, basic budget, fast burn alert"
  standard: "1-2w: 3-5 services, multi-SLI, multi-burn alerts, reporting"
  enterprise: "3-4w: + SLO-as-code, governance, automation"
workflow-type: "hybrid"
interactive-mode: true
allowed-tools: [Bash, Read, Write, Edit, Task, Glob, Grep]
---

# SLO Implementation

$ARGUMENTS

## Topics covered

SLO framework design, SLI measurement, error-budget tracking, SLO monitoring/alerting, reporting, automation (SLO-as-code), governance.

## Phase 1: Analysis & Design

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 4h | Single service, 1-2 SLIs, basic targets |
| Standard | 2d | 3-5 services, multiple SLIs, historical validation |
| Enterprise | 1w | All critical, comprehensive, progressive roadmap |

## Phase 2: SLI Implementation

Metric instrumentation → Recording rules (5m, 1h, 24h, 30d) → Error budget tracking (consumption, burn rates, projections) → Validation

## Phase 3: Monitoring & Alerting

**Burn Rate Alerts:**
- Fast burn (14.4x, 2% in 1h) → Page (All modes)
- Slow burn (3x, 10% in 6h) → Ticket (Standard+)
- Budget exhaustion (<7d projected) → Plan work (Enterprise)

**Dashboards:** SLO summary, error budget gauge, burn rate viz, multi-service overview (Enterprise)

## Phase 4: Governance & Automation

**Standard:** Monthly reports, release decisions, weekly reviews, budget policies

**Enterprise:** SLO-as-code (YAML + GitOps), auto-generation for new services, progressive roadmap (99.0→99.95), quarterly planning, stakeholder reporting, toil budgets

## Success Criteria

**Quick:** 1 service framework, 1-2 SLIs, budget tracked, fast burn alert, basic dashboard

**Standard:** All Quick + 3-5 services, multi-window alerts, monthly reports, release framework, weekly reviews

**Enterprise:** All Standard + SLO-as-code, auto-generation, progressive roadmap, governance, stakeholder automation, toil integration, maturity assessment

## Examples

```text
/slo-implement checkout-api --mode=quick
/slo-implement payments-service --mode=enterprise
```
