# Grafana Dashboards: Complete Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-07
**Grafana Version:** 10.x / 11.x

## Overview

This guide provides comprehensive coverage of Grafana dashboard design, configuration, and best practices for modern observability. The content has been split into modular sections for better maintainability.

## Documentation Sections

1. **[Dashboard Strategy & Architecture](grafana-dashboards/overview.md)**
   - JSON Structure
   - Key Components
   - Design Principles & Best Practices
   - Security Considerations

2. **[Panel Types & Configuration](grafana-dashboards/panels.md)**
   - Time Series (Graph)
   - Stat, Gauge, Heatmap
   - Table, Logs Panels
   - Query Best Practices

3. **[Template Variables](grafana-dashboards/variables.md)**
   - Variable Types (Query, Custom, Interval, etc.)
   - Advanced Usage (Chaining, Multi-value)

4. **[Methodologies](grafana-dashboards/json-models.md)**
   - [Golden Signals](grafana-dashboards/golden-signals.md) - Latency, Traffic, Errors, Saturation
   - [RED Metrics](grafana-dashboards/red-metrics.md) - Rate, Errors, Duration
   - [USE Metrics](grafana-dashboards/use-metrics.md) - Utilization, Saturation, Errors

5. **[Provisioning & Management](grafana-dashboards/provisioning.md)**
   - Provisioning Configuration
   - GitOps & File Structure
   - Kubernetes & Terraform examples
   - Dashboard Versioning

6. **[Alerting Integration](grafana-dashboards/alerting.md)**
   - Alert Rule Configuration
   - Unified Alerting (Grafana 9+)
   - Notification Templates

7. **[Complete JSON Models](grafana-dashboards/json-models.md)**
   - Microservices Dashboard
   - Kubernetes Cluster Dashboard
