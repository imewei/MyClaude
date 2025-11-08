---
version: 1.0.3
command: /monitor-setup
description: Set up comprehensive monitoring and observability stack with Prometheus, Grafana, and distributed tracing across 3 execution modes
execution_modes:
  quick:
    duration: "1-2 days"
    description: "Basic observability for single service"
    agents: ["observability-engineer"]
    scope: "Basic Prometheus metrics + simple Grafana dashboard + application instrumentation + basic alerting"
  standard:
    duration: "1 week"
    description: "Production-ready observability stack"
    agents: ["observability-engineer", "performance-engineer"]
    scope: "Full Prometheus + comprehensive Grafana + distributed tracing + log aggregation + multi-window alerting + infrastructure-as-code"
  enterprise:
    duration: "2-3 weeks"
    description: "Enterprise observability with multi-cluster and SLO tracking"
    agents: ["observability-engineer", "performance-engineer", "database-optimizer", "network-engineer"]
    scope: "All standard + multi-cluster federation + advanced correlation (metrics+logs+traces) + SLO/error budget tracking + cost optimization + security monitoring"
workflow_type: "hybrid"
interactive_mode: true
---

# Monitoring and Observability Setup

You are a monitoring and observability expert specializing in implementing comprehensive monitoring solutions. Set up metrics collection, distributed tracing, log aggregation, and create insightful dashboards that provide full visibility into system health and performance.

## Context

The user needs to implement or improve monitoring and observability. Focus on the three pillars of observability (metrics, logs, traces), setting up monitoring infrastructure, creating actionable dashboards, and establishing effective alerting strategies.

## Agent Coordination

| Phase | Agents | Tasks | Duration |
|-------|--------|-------|----------|
| 1. Planning & Assessment | observability-engineer | Infrastructure assessment, stack design, requirements gathering | 10% |
| 2. Core Setup | observability-engineer, performance-engineer | Prometheus deployment, Grafana setup, basic dashboards | 40% |
| 3. Advanced Integration | observability-engineer, database-optimizer, network-engineer | Distributed tracing, log aggregation, specialized monitoring | 30% |
| 4. Validation & Optimization | All agents | Dashboard creation, alert testing, performance tuning | 20% |

## Execution Modes

### Quick Mode (1-2 days)

**Agents**: observability-engineer

**Deliverables**:
- Basic Prometheus configuration with service discovery
- Simple Grafana dashboard (golden signals: latency, traffic, errors, saturation)
- Application instrumentation for key metrics
- Basic alerting for availability and high error rate

**Use When**: Need quick visibility into single service, proof-of-concept, startup MVP

### Standard Mode (1 week)

**Agents**: observability-engineer, performance-engineer

**Deliverables**:
- Full Prometheus stack with recording rules and federation
- Comprehensive Grafana dashboards (RED/USE metrics, service health)
- Distributed tracing with OpenTelemetry and Jaeger
- Structured logging with Fluentd/Elasticsearch
- Multi-window alerting with Alertmanager
- Infrastructure-as-code deployment (Terraform/Helm)

**Use When**: Production deployment, multi-service architecture, team of 5-20 engineers

### Enterprise Mode (2-3 weeks)

**Agents**: observability-engineer, performance-engineer, database-optimizer, network-engineer

**Deliverables**:
- All standard deliverables
- Multi-cluster monitoring federation
- Advanced correlation (metrics + logs + traces with trace IDs)
- SLO/SLA monitoring with error budget tracking
- Custom business metrics and KPI dashboards
- Runbook automation and incident response integration
- Cost optimization (metric cardinality reduction, sampling strategies)
- Security monitoring integration (audit logs, access patterns)
- Database query performance monitoring
- Network latency and throughput analysis

**Use When**: Large-scale systems (50+ services), compliance requirements, enterprise SLAs

## Requirements

$ARGUMENTS

## External Documentation

Comprehensive guides available in [docs/monitor-setup/](../docs/monitor-setup/):

1. **[Prometheus Setup](../docs/monitor-setup/prometheus-setup.md)** (~1,220 lines)
   - Global configuration and scrape configs
   - Service discovery (Kubernetes, Consul, EC2, file-based)
   - Recording rules (RED metrics, USE metrics, aggregations)
   - Alerting configuration and exporters
   - Storage, retention, and federation
   - Best practices and optimization

2. **[Grafana Dashboards](../docs/monitor-setup/grafana-dashboards.md)** (~1,470 lines)
   - Dashboard JSON structure and panel types
   - Golden Signals dashboards (latency, traffic, errors, saturation)
   - RED metrics dashboards (rate, errors, duration)
   - USE metrics dashboards (utilization, saturation, errors)
   - Template variables for multi-service views
   - Dashboard provisioning and automation

3. **[Distributed Tracing](../docs/monitor-setup/distributed-tracing.md)** (~1,368 lines)
   - OpenTelemetry SDK setup (Node.js, Python, Go, Java)
   - Trace context propagation (W3C, B3, Jaeger)
   - Jaeger and Tempo deployment
   - Trace sampling strategies
   - Trace correlation with logs and metrics

4. **[Log Aggregation](../docs/monitor-setup/log-aggregation.md)** (~1,200 lines)
   - Fluentd/Fluent Bit configuration
   - Elasticsearch index templates and ILM policies
   - Kibana dashboard setup
   - Structured logging libraries
   - Log correlation with trace IDs
   - Retention and archival strategies

5. **[Alerting Strategies](../docs/monitor-setup/alerting-strategies.md)** (~1,327 lines)
   - Alertmanager configuration and routing
   - Notification channels (Slack, PagerDuty, email)
   - Alert rule design patterns
   - Multi-window multi-burn-rate alerting
   - Runbook automation
   - Alert fatigue prevention

6. **[Infrastructure as Code](../docs/monitor-setup/infrastructure-code.md)** (~1,200 lines)
   - Terraform modules for Prometheus, Grafana, Jaeger
   - Helm charts for Kubernetes deployment
   - Docker Compose for local development
   - Multi-cloud integration (AWS, Azure, GCP)
   - Backup and disaster recovery

**Total External Documentation**: ~13,000 lines

## Implementation Workflow

### Phase 1: Planning (Quick: 2h, Standard: 4h, Enterprise: 1 day)

1. **Infrastructure Assessment**
   - Current monitoring capabilities
   - Service architecture and dependencies
   - Scale requirements (requests/sec, services, hosts)
   - Existing telemetry and instrumentation

2. **Stack Design**
   - Select monitoring components based on mode
   - Design metric collection strategy
   - Plan dashboard hierarchy
   - Define alert thresholds and escalation

3. **Requirements Gathering**
   - SLO/SLA requirements
   - Compliance and retention needs
   - Team skills and training requirements
   - Budget and resource constraints

### Phase 2: Core Setup (Quick: 4h, Standard: 2 days, Enterprise: 1 week)

1. **Prometheus Deployment**
   - Install Prometheus server (Docker/Kubernetes/bare metal)
   - Configure scrape targets and service discovery
   - Set up recording rules for performance
   - Deploy exporters (node, blackbox, custom)

2. **Grafana Configuration**
   - Install and configure Grafana
   - Add Prometheus datasource
   - Create core dashboards (golden signals, service health)
   - Set up user authentication and permissions

3. **Application Instrumentation**
   - Add Prometheus client libraries
   - Instrument HTTP handlers, databases, external calls
   - Export custom business metrics
   - Validate metric collection

### Phase 3: Advanced Integration (Standard: 2 days, Enterprise: 1 week)

1. **Distributed Tracing** (Standard/Enterprise)
   - Deploy Jaeger or Tempo
   - Instrument services with OpenTelemetry
   - Configure trace sampling (head-based, tail-based)
   - Integrate traces with logs and metrics

2. **Log Aggregation** (Standard/Enterprise)
   - Deploy Fluentd/Fluent Bit collectors
   - Configure Elasticsearch cluster
   - Set up Kibana for log exploration
   - Implement structured logging
   - Configure log retention and archival

3. **Advanced Monitoring** (Enterprise only)
   - Database query performance monitoring
   - Network flow analysis
   - Multi-cluster federation
   - Custom business KPI dashboards

### Phase 4: Validation & Optimization (Quick: 2h, Standard: 1 day, Enterprise: 3 days)

1. **Dashboard Creation**
   - Create service-specific dashboards
   - Build team dashboards with relevant metrics
   - Set up executive/business dashboards
   - Configure dashboard variables and filters

2. **Alert Configuration**
   - Implement symptom-based alerts
   - Configure Alertmanager routing
   - Set up notification channels
   - Test alert escalation flows

3. **Testing & Validation**
   - Verify metric collection accuracy
   - Test alert firing and notifications
   - Validate dashboard performance
   - Conduct chaos testing to verify observability

4. **Documentation & Training**
   - Document runbooks and procedures
   - Train team on dashboards and alerts
   - Create troubleshooting guides
   - Establish on-call procedures

## Output Format

1. **Infrastructure Assessment**: Current monitoring capabilities and gaps
2. **Monitoring Architecture**: Complete stack design with components and data flows
3. **Implementation Plan**: Step-by-step deployment guide with timelines
4. **Metric Definitions**: Comprehensive catalog of collected metrics
5. **Dashboard Templates**: Production-ready Grafana dashboards
6. **Alert Runbooks**: Detailed alert response procedures with automation
7. **SLO Definitions**: Service level objectives and error budgets (Enterprise)
8. **Integration Guide**: Service instrumentation instructions with code examples

## Success Criteria

**Quick Mode**:
- ✅ Prometheus collecting metrics from target service
- ✅ Grafana dashboard showing golden signals
- ✅ At least 2 critical alerts configured
- ✅ Team can view metrics and respond to alerts

**Standard Mode**:
- ✅ All Quick criteria met
- ✅ Distributed tracing capturing >80% of requests
- ✅ Logs aggregated and searchable in Elasticsearch
- ✅ Multi-window burn rate alerting implemented
- ✅ Infrastructure deployed via code (repeatable)
- ✅ RED/USE metrics dashboards for all services

**Enterprise Mode**:
- ✅ All Standard criteria met
- ✅ Multi-cluster federation operational
- ✅ Metrics, logs, and traces correlated by trace ID
- ✅ SLO error budgets tracked and reported
- ✅ Cost optimization achieving 30%+ metric reduction
- ✅ Security monitoring integrated
- ✅ Database and network specialized monitoring active

Focus on creating a monitoring system that provides actionable insights, reduces MTTR (mean time to recovery), and enables proactive issue detection through comprehensive observability coverage.
