# Observability Monitoring

Observability, monitoring, logging, distributed tracing, and SLO/SLA management for production systems with Prometheus, Grafana, and modern observability platforms

**Version:** 1.0.1 | **Category:** devops | **License:** MIT

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/observability-monitoring.html)

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

## What's New in v1.0.1

### Enhanced Skill Discoverability

All 5 skills have been comprehensively improved for better Claude Code integration:

- **Detailed Use Cases**: Each skill now includes 10-15 specific scenarios where it should be used
- **File Type References**: Skills automatically activate when working with relevant files (prometheus.yml, dashboard JSON, Airflow DAGs)
- **Proactive Usage**: Claude Code can now better identify when to use these skills during coding tasks
- **Comprehensive Examples**: All skills maintain extensive code examples and production-ready patterns

### Example Use Cases

**When you work on Prometheus configuration:**
```yaml
# Claude Code will automatically use the prometheus-configuration skill
# when you edit prometheus.yml files or create recording/alert rules
```

**When you design Grafana dashboards:**
```json
// Claude Code will use the grafana-dashboards skill when working
// with dashboard JSON files or creating visualization panels
```

**When you implement distributed tracing:**
```python
# Claude Code will use the distributed-tracing skill when you
# instrument applications with OpenTelemetry or configure Jaeger
```

**When you define SLOs:**
```yaml
# Claude Code will use the slo-implementation skill when you
# create SLI/SLO definitions or implement error budget tracking
```

**When you build Airflow workflows:**
```python
# Claude Code will use the airflow-scientific-workflows skill
# when you create DAG files or scientific data pipelines
```

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
