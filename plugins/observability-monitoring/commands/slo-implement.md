---
version: 1.0.5
command: /slo-implement
description: Implement SLO/SLA monitoring, error budgets, and burn rate alerting
execution_modes:
  quick: "2-3 days"
  standard: "1-2 weeks"
  enterprise: "3-4 weeks"
workflow_type: "hybrid"
interactive_mode: true
---

# SLO Implementation Guide

Implement SLOs to establish reliability targets and enable data-driven reliability decisions.

## Requirements

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope | Agents |
|------|----------|-------|--------|
| Quick | 2-3 days | 1 service, 1-2 SLIs, basic error budget, fast burn alert | observability-engineer |
| Standard (default) | 1-2 weeks | 3-5 services, multi-SLI, multi-burn-rate alerts, reporting | + performance-engineer |
| Enterprise | 3-4 weeks | + SLO-as-code, governance, automation, stakeholder reports | + database-optimizer, network-engineer |

---

## External Documentation

| Document | Content | Lines |
|----------|---------|-------|
| [SLO Framework](../docs/slo-implement/slo-framework.md) | Service tiers, SLI selection, targets | ~1,680 |
| [SLI Measurement](../docs/slo-implement/sli-measurement.md) | SLI types, Prometheus queries | ~1,538 |
| [Error Budgets](../docs/slo-implement/error-budgets.md) | Burn rate, consumption, projections | ~1,500 |
| [SLO Monitoring](../docs/slo-implement/slo-monitoring.md) | Recording rules, alerts | ~1,545 |
| [SLO Reporting](../docs/slo-implement/slo-reporting.md) | Monthly reports, stakeholder comms | ~1,450 |
| [SLO Automation](../docs/slo-implement/slo-automation.md) | SLO-as-code, GitOps | ~1,450 |
| [SLO Governance](../docs/slo-implement/slo-governance.md) | Culture, reviews, planning | ~1,420 |

**Total:** ~17,583 lines

---

## Phase 1: Analysis & Design

| Step | Quick | Standard | Enterprise |
|------|-------|----------|------------|
| Duration | 4h | 2 days | 1 week |
| Service tier analysis | Single | 3-5 services | All critical |
| SLI selection | 1-2 | Multiple per service | Comprehensive |
| SLO target setting | Basic | Historical validation | Progressive roadmap |

---

## Phase 2: SLI Implementation

| Step | Description |
|------|-------------|
| Metric instrumentation | Prometheus metrics in services |
| Recording rules | Multi-window calculations (5m, 1h, 24h, 30d) |
| Error budget tracking | Consumption, burn rates, projections |
| Validation | SLI accuracy vs ground truth |

---

## Phase 3: Monitoring & Alerting

### Burn Rate Alerts

| Alert Type | Rate | Budget Impact | Action |
|------------|------|---------------|--------|
| Fast burn (All) | 14.4x | 2% in 1 hour | Page on-call |
| Slow burn (Standard+) | 3x | 10% in 6 hours | Create ticket |
| Budget exhaustion (Enterprise) | Projected <7 days | - | Plan reliability work |

### Dashboard Components
- SLO summary (current status, trends)
- Error budget gauge and timeline
- Burn rate visualization
- Multi-service overview (Enterprise)

---

## Phase 4: Governance & Automation

### Standard Mode
- Monthly SLO reports
- Release decision framework
- Weekly review process
- Error budget policies

### Enterprise Mode
- SLO-as-code (YAML + GitOps)
- Automated generation for new services
- Progressive SLO roadmap (99.0 → 99.95)
- Quarterly planning cycle
- Stakeholder reporting automation
- Toil budget calculations

---

## Output Deliverables

| Mode | Deliverables |
|------|--------------|
| Quick | Framework doc, SLIs, error budget, fast burn alert, basic dashboard |
| Standard | + Multi-burn-rate alerts, reporting, decision framework, review process |
| Enterprise | + SLO-as-code, governance, automation, stakeholder reports, maturity plan |

---

## Success Criteria

### Quick Mode
- [ ] SLO framework for 1 service
- [ ] 1-2 SLIs measuring user experience
- [ ] Error budget calculated and tracked
- [ ] Fast burn alert tested
- [ ] Basic dashboard operational

### Standard Mode
- [ ] All Quick criteria
- [ ] 3-5 services with comprehensive SLOs
- [ ] Multi-window multi-burn-rate alerting
- [ ] Monthly reports automated
- [ ] Release decision framework documented
- [ ] Weekly review process established

### Enterprise Mode
- [ ] All Standard criteria
- [ ] SLO-as-code with GitOps
- [ ] Automated SLO generation
- [ ] Progressive roadmap defined
- [ ] Governance framework operational
- [ ] Stakeholder reporting automated
- [ ] Toil budget integrated
- [ ] Maturity assessment complete
