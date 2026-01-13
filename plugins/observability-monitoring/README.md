# Observability Monitoring

Observability, monitoring, logging, distributed tracing, and SLO/SLA management for production systems with Prometheus, Grafana, and modern observability platforms

**Version:** 1.0.7 | **Category:** devops | **License:** MIT

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/observability-monitoring.html) | [CHANGELOG](CHANGELOG.md)

---

## 🚀 What's New in v1.0.7

### Command Optimization with Execution Modes

Both `/monitor-setup` and `/slo-implement` commands now feature:
- **3 Execution Modes** (quick/standard/enterprise) for flexible implementation
- **Comprehensive External Documentation** (~30,583 lines across 13 guides)
- **Multi-Agent Orchestration** with phase-based workflows
- **63.8% Command File Reduction** while preserving all content

#### `/monitor-setup` - 3 Execution Modes

| Mode | Duration | Agents | Scope |
|------|----------|--------|-------|
| **Quick** | 1-2 days | 1 agent | Basic Prometheus + Grafana + instrumentation |
| **Standard** | 1 week | 2 agents | Full stack + tracing + logging + IaC |
| **Enterprise** | 2-3 weeks | 4 agents | All standard + multi-cluster + SLO tracking + cost optimization |

**External Documentation** (6 files - ~13,000 lines):
- Prometheus Setup (~1,220 lines) - Configuration, service discovery, recording rules, federation
- Grafana Dashboards (~1,470 lines) - Golden Signals, RED/USE metrics, provisioning
- Distributed Tracing (~1,368 lines) - OpenTelemetry, Jaeger, Tempo, sampling
- Log Aggregation (~1,200 lines) - Fluentd, Elasticsearch, structured logging
- Alerting Strategies (~1,327 lines) - Multi-window burn rate, runbooks, fatigue prevention
- Infrastructure as Code (~1,200 lines) - Terraform, Helm, multi-cloud deployment

#### `/slo-implement` - 3 Execution Modes

| Mode | Duration | Agents | Scope |
|------|----------|--------|-------|
| **Quick** | 2-3 days | 1 agent | Basic SLO framework + 1-2 SLIs + dashboard |
| **Standard** | 1-2 weeks | 2 agents | 3-5 services + burn rate alerts + reporting |
| **Enterprise** | 3-4 weeks | 4 agents | All standard + SLO-as-code + governance + automation |

**External Documentation** (7 files - ~17,583 lines):
- SLO Framework (~1,680 lines) - Service tiers, user journey mapping, target calculations
- SLI Measurement (~1,538 lines) - API/web/batch/streaming SLIs, Core Web Vitals
- Error Budgets (~1,500 lines) - Burn rate calculations, multi-window detection
- SLO Monitoring (~1,545 lines) - Recording rules, fast/slow burn alerts
- SLO Reporting (~1,450 lines) - Monthly reports, trend analysis, stakeholder templates
- SLO Automation (~1,450 lines) - SLO-as-code, GitOps workflows, Kubernetes CRDs
- SLO Governance (~1,420 lines) - Culture, reviews, release decisions, toil budgets

---

## Agents (4)

### Observability Engineer

**Status:** active

Expert in observability, monitoring, distributed tracing, and production system visibility

### Performance Engineer

**Status:** active

Specialist in performance monitoring, optimization, and system reliability

### Database Optimizer

**Status:** active

Expert in database performance monitoring and query optimization

### Network Engineer

**Status:** active

Specialist in network monitoring, latency optimization, and distributed system observability

## Commands (2)

### `/monitor-setup`

**Status:** active

Set up comprehensive monitoring and observability stack with Prometheus and Grafana

### `/slo-implement`

**Status:** active

Implement SLO/SLA monitoring and alerting for production services

## Skills (5)

### Prometheus Configuration

Set up and configure Prometheus for comprehensive metric collection, storage, alerting, and monitoring with scrape configs, recording rules, and alert rules for infrastructure and applications.

**Key capabilities:**
- Prometheus installation and configuration (prometheus.yml)
- Static and dynamic service discovery (Kubernetes, Consul, DNS)
- Recording rules for pre-aggregated metrics
- Alert rules with PromQL queries
- Relabeling configurations and federation
- Integration with Grafana and Alertmanager

### Grafana Dashboards

Create and manage production-ready Grafana dashboards with panels, variables, alerts, and templates for real-time visualization of system and application metrics.

**Key capabilities:**
- Dashboard JSON creation and management
- Panel types: graphs, stats, tables, heatmaps, gauges
- Variables and templating for dynamic filtering
- RED method (Rate, Errors, Duration) dashboards
- USE method (Utilization, Saturation, Errors) dashboards
- Dashboard provisioning with Terraform and Ansible

### Distributed Tracing

Implement distributed tracing with Jaeger, Tempo, and OpenTelemetry to track requests across microservices and identify performance bottlenecks.

**Key capabilities:**
- OpenTelemetry instrumentation (Python, Node.js, Go, Java)
- Jaeger and Tempo deployment configurations
- Trace context propagation (W3C Trace Context, baggage)
- Sampling strategies (probabilistic, rate-limiting, adaptive)
- Integration with logging for correlation
- Service dependency visualization

### SLO Implementation

Define and implement Service Level Indicators (SLIs), Service Level Objectives (SLOs), error budgets, and burn rate alerting following SRE best practices.

**Key capabilities:**
- SLI definitions (availability, latency, durability)
- SLO target calculation and error budget tracking
- Multi-window burn rate alerts for fast/slow violations
- Error budget policies and quarterly reviews
- SLO dashboards in Grafana
- PromQL queries for SLI/SLO metrics

### Airflow Scientific Workflows

Design and implement Apache Airflow DAGs for scientific data pipelines, workflow orchestration, and computational task automation.

**Key capabilities:**
- Time-series data processing pipelines
- Distributed simulation orchestration
- PostgreSQL and TimescaleDB integration
- Data quality validation and gating logic
- Multi-dimensional array processing
- JAX and scientific computing integration


## What's New in v1.0.7

This release implements **Opus 4.5 optimization** with enhanced token efficiency and standardized documentation.

### Key Improvements

- **Format Standardization**: All components now include consistent YAML frontmatter with version, maturity, specialization, and description fields
- **Token Efficiency**: 40-50% line reduction through tables over prose, minimal code examples, and structured sections
- **Enhanced Discoverability**: Clear "Use when..." trigger phrases for better Claude Code activation
- **Actionable Checklists**: Task-oriented guidance for common workflows
- **Cross-Reference Tables**: Quick-reference format for delegation and integration patterns


## Quick Start

To use this plugin:

1. Ensure Claude Code is installed
2. Enable the `observability-monitoring` plugin
3. Activate an agent (e.g., `@Observability Engineer`)
4. Try a command (e.g., `/monitor-setup`)
5. Skills will automatically activate when working on relevant files

## Example Workflows

### Setting Up Monitoring Stack

```bash
# Use the monitor-setup command to get started
/monitor-setup

# Claude Code will guide you through:
# - Prometheus installation and configuration
# - Grafana dashboard setup
# - Alert rule configuration
# - Integration with existing infrastructure
```

### Implementing SLOs

```bash
# Use the slo-implement command
/slo-implement

# Claude Code will help you:
# - Define SLIs for your services
# - Set appropriate SLO targets
# - Implement error budget tracking
# - Create burn rate alerts
```

## Integration

See the full documentation for integration patterns and compatible plugins.

## Documentation

For comprehensive documentation, see: [Plugin Documentation](https://myclaude.readthedocs.io/en/latest/plugins/observability-monitoring.html)

To build documentation locally:

```bash
cd docs/
make html
```
