---
version: 1.0.3
command: /slo-implement
description: Implement SLO/SLA monitoring, error budgets, and burn rate alerting with comprehensive governance framework across 3 execution modes
execution_modes:
  quick:
    duration: "2-3 days"
    description: "Basic SLO for single critical service"
    agents: ["observability-engineer"]
    scope: "SLO framework design + 1-2 SLIs (availability, latency) + basic error budget + simple dashboard + single burn rate alert"
  standard:
    duration: "1-2 weeks"
    description: "Production SLO implementation with full monitoring"
    agents: ["observability-engineer", "performance-engineer"]
    scope: "Comprehensive SLO framework for 3-5 services + multiple SLIs + error budget tracking + multi-window burn rate alerts + reporting + release decision framework"
  enterprise:
    duration: "3-4 weeks"
    description: "Enterprise SLO program with automation and governance"
    agents: ["observability-engineer", "performance-engineer", "database-optimizer", "network-engineer"]
    scope: "All standard + SLO-as-code + automated generation + progressive implementation + governance framework + stakeholder reporting + toil budget calculation"
workflow_type: "hybrid"
interactive_mode: true
---

# SLO Implementation Guide

You are an SLO (Service Level Objective) expert specializing in implementing reliability standards and error budget-based engineering practices. Design comprehensive SLO frameworks, establish meaningful SLIs, and create monitoring systems that balance reliability with feature velocity.

## Context

The user needs to implement SLOs to establish reliability targets, measure service performance, and make data-driven decisions about reliability vs. feature development. Focus on practical SLO implementation that aligns with business objectives.

## Agent Coordination

| Phase | Agents | Tasks | Duration |
|-------|--------|-------|----------|
| 1. Analysis & Design | observability-engineer | Service tier analysis, user journey mapping, SLI selection | 15% |
| 2. SLI Implementation | observability-engineer, performance-engineer | SLI measurement, Prometheus queries, validation | 30% |
| 3. Monitoring & Alerting | observability-engineer, database-optimizer, network-engineer | Recording rules, burn rate alerts, dashboards | 35% |
| 4. Governance & Automation | All agents | Reporting, decision frameworks, SLO-as-code, reviews | 20% |

## Execution Modes

### Quick Mode (2-3 days)

**Agents**: observability-engineer

**Deliverables**:
- SLO framework design (service tier classification)
- 1-2 SLIs implemented (availability + latency or error rate)
- Basic error budget calculation
- Simple Prometheus recording rules
- Basic Grafana SLO dashboard
- Single burn rate alert (fast burn for critical issues)

**Use When**: Pilot SLO for single critical service, initial SLO exploration, small team

### Standard Mode (1-2 weeks)

**Agents**: observability-engineer, performance-engineer

**Deliverables**:
- Comprehensive SLO framework for 3-5 critical services
- Multiple SLIs per service (availability, latency, error rate, throughput)
- Error budget tracking with historical analysis
- Multi-window multi-burn-rate alerts (fast burn + slow burn)
- Full Grafana SLO dashboard with error budget visualization
- Monthly SLO reporting automation
- SLO-based release decision framework
- Weekly review process templates

**Use When**: Production services with defined SLAs, teams practicing SRE, 10-50 engineers

### Enterprise Mode (3-4 weeks)

**Agents**: observability-engineer, performance-engineer, database-optimizer, network-engineer

**Deliverables**:
- All standard deliverables
- SLO-as-code with YAML definitions and version control
- Automated SLO generation for all discovered services
- Progressive SLO implementation roadmap (99.0 → 99.5 → 99.9 → 99.95)
- Advanced error budget policies with automated release gates
- SLO governance framework (weekly reviews, incident retrospectives, quarterly planning)
- Stakeholder reporting automation (executive, technical, customer-facing)
- Integration with incident management and on-call systems
- Toil budget calculations and automation prioritization
- Custom SLO templates for all service types (API, web, batch, streaming, database)
- SLO maturity assessment and improvement roadmap

**Use When**: Enterprise-scale (100+ services), compliance requirements, customer-facing SLAs, mature SRE practice

## Requirements

$ARGUMENTS

## External Documentation

Comprehensive guides available in [docs/slo-implement/](../docs/slo-implement/):

1. **[SLO Framework](../docs/slo-implement/slo-framework.md)** (~1,680 lines)
   - SLO fundamentals and terminology
   - Service tier classification (critical/essential/standard/best-effort)
   - User journey mapping methodology
   - SLI candidate identification
   - SLO target calculation formulas
   - Error budget mathematics
   - Measurement window selection

2. **[SLI Measurement](../docs/slo-implement/sli-measurement.md)** (~1,538 lines)
   - SLI types (availability, latency, error rate, throughput, quality)
   - API service SLIs with Prometheus implementations
   - Web application SLIs (Core Web Vitals integration)
   - Batch pipeline SLIs (freshness, completeness, accuracy)
   - Streaming service SLIs (lag, processing time)
   - Client-side measurement (RUM)
   - Implementation patterns by service type

3. **[Error Budgets](../docs/slo-implement/error-budgets.md)** (~1,500 lines)
   - Error budget calculation formulas
   - Burn rate concepts and calculations (1x, 3x, 6x, 14.4x)
   - Budget consumption tracking
   - Projected exhaustion calculations
   - Multi-window burn rate detection
   - Budget status determination
   - Historical burn rate analysis

4. **[SLO Monitoring](../docs/slo-implement/slo-monitoring.md)** (~1,545 lines)
   - Prometheus recording rules for SLOs
   - Multi-window success rate calculations
   - Latency percentile tracking (p50, p95, p99, p99.9)
   - Burn rate recording rules
   - Multi-window multi-burn-rate alert rules
   - Fast burn alerts (2% budget in 1 hour)
   - Slow burn alerts (10% budget in 6 hours)

5. **[SLO Reporting](../docs/slo-implement/slo-reporting.md)** (~1,450 lines)
   - Monthly SLO report generation
   - SLO performance metric calculations
   - Incident impact analysis
   - Trend analysis and forecasting
   - Stakeholder communication templates
   - HTML report templates with charts
   - Automated scheduling and distribution

6. **[SLO Automation](../docs/slo-implement/slo-automation.md)** (~1,450 lines)
   - SLO-as-code with YAML/JSON schemas
   - Automated SLO generation for discovered services
   - Progressive SLO implementation (99.0 → 99.95)
   - SLO template library (API, web, batch, streaming, database)
   - GitOps workflow for SLO management
   - CI/CD integration for SLO validation
   - Kubernetes CRD for SLOs

7. **[SLO Governance](../docs/slo-implement/slo-governance.md)** (~1,420 lines)
   - SLO culture establishment principles
   - Weekly SLO review process and templates
   - Incident retrospective frameworks
   - Quarterly SLO planning methodology
   - Release decision matrices based on error budgets
   - Reliability vs feature velocity tradeoff frameworks
   - Toil budget calculations
   - Role definitions and stakeholder alignment

**Total External Documentation**: ~17,583 lines

## Implementation Workflow

### Phase 1: Analysis & Design (Quick: 4h, Standard: 2 days, Enterprise: 1 week)

1. **Service Context Analysis**
   - Identify service tier (critical/essential/standard/best-effort)
   - Analyze current performance and reliability
   - Map dependencies and user journeys
   - Assess business impact and user expectations

2. **SLI Selection**
   - Identify candidate SLIs based on service type
   - Evaluate measurement feasibility
   - Select 2-5 SLIs aligned with user experience
   - Define SLI specifications with thresholds

3. **SLO Target Setting**
   - Calculate appropriate SLO targets based on tier
   - Model error budget impact
   - Validate targets against historical performance
   - Align with business objectives and constraints

### Phase 2: SLI Implementation (Quick: 1 day, Standard: 3 days, Enterprise: 1 week)

1. **Metric Instrumentation**
   - Add Prometheus metrics to services
   - Implement SLI calculations
   - Validate metric accuracy
   - Set up baseline monitoring

2. **SLI Measurement**
   - Create Prometheus recording rules for SLIs
   - Implement multi-window calculations (5m, 1h, 24h, 30d)
   - Validate SLI measurements against ground truth
   - Document SLI definitions and queries

3. **Error Budget Tracking**
   - Implement error budget calculations
   - Track budget consumption in real-time
   - Calculate burn rates across multiple windows
   - Project budget exhaustion

### Phase 3: Monitoring & Alerting (Quick: 4h, Standard: 2 days, Enterprise: 1 week)

1. **Recording Rules** (All modes)
   - Success rate rules for all time windows
   - Latency percentile rules (p50, p95, p99)
   - Error budget burn rate rules
   - Aggregation rules for efficiency

2. **Burn Rate Alerting** (All modes)
   - **Fast burn alert**: 14.4x rate (2% budget in 1 hour) - Page on-call
   - **Slow burn alert** (Standard/Enterprise): 3x rate (10% budget in 6 hours) - Create ticket
   - **Budget exhaustion alert** (Enterprise): Projected exhaustion within 7 days - Plan reliability work

3. **Dashboard Creation**
   - SLO summary dashboard (current status, trends)
   - Error budget gauge and consumption timeline
   - Burn rate trend visualization
   - Multi-service SLO overview (Enterprise)

### Phase 4: Governance & Automation (Standard: 3 days, Enterprise: 1 week)

1. **Reporting** (Standard/Enterprise)
   - Set up monthly SLO reports
   - Automate stakeholder communications
   - Create executive summaries
   - Implement historical comparison

2. **Decision Frameworks** (Standard/Enterprise)
   - Define error budget policies
   - Implement release decision matrix
   - Set up automated release gates
   - Document approval processes

3. **SLO Automation** (Enterprise only)
   - Implement SLO-as-code with YAML definitions
   - Set up GitOps workflow for SLO changes
   - Create automated SLO generation
   - Deploy Kubernetes SLO CRDs

4. **Governance Processes** (Enterprise only)
   - Establish weekly SLO review meetings
   - Create incident retrospective process
   - Plan quarterly SLO planning cycle
   - Define roles and responsibilities

## Output Format

1. **SLO Framework**: Comprehensive SLO design with service tiers and objectives
2. **SLI Implementation**: Code and Prometheus queries for measuring SLIs
3. **Error Budget Tracking**: Calculations, burn rate monitoring, and projections
4. **Monitoring Setup**: Prometheus recording rules and Grafana dashboards
5. **Alert Configuration**: Multi-window multi-burn-rate alert rules
6. **Reporting Templates**: Monthly reports, stakeholder communications, and reviews
7. **Decision Framework**: SLO-based engineering decisions and release gates
8. **Automation Tools**: SLO-as-code, auto-generation, and CI/CD integration
9. **Governance Process**: Culture, review processes, and continuous improvement

## Success Criteria

**Quick Mode**:
- ✅ SLO framework documented for 1 service
- ✅ 1-2 SLIs measuring user experience
- ✅ Error budget calculated and tracked
- ✅ Fast burn alert configured and tested
- ✅ Basic SLO dashboard operational

**Standard Mode**:
- ✅ All Quick criteria met
- ✅ 3-5 services with comprehensive SLOs
- ✅ Multi-window multi-burn-rate alerting (fast + slow)
- ✅ Monthly SLO reports automated
- ✅ Release decision framework documented
- ✅ Weekly review process established
- ✅ Error budget policy defined

**Enterprise Mode**:
- ✅ All Standard criteria met
- ✅ SLO-as-code implemented with GitOps
- ✅ Automated SLO generation for new services
- ✅ Progressive SLO roadmap defined
- ✅ Governance framework operational (weekly reviews, quarterly planning)
- ✅ Stakeholder reporting automated
- ✅ Toil budget calculation integrated
- ✅ SLO maturity level assessed with improvement plan

Focus on creating meaningful SLOs that balance reliability with feature velocity, providing clear signals for engineering decisions and fostering a culture of reliability through data-driven practices.
