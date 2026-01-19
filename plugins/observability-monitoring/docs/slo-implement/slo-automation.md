# SLO Automation Framework

Complete guide for automating Service Level Objective (SLO) management with SLO-as-Code, progressive implementation strategies, and GitOps workflows.

## Documentation Modules

This documentation has been split into focused modules for better maintainability and readability:

### [Concepts](./slo-automation/concept.md)
- **SLO-as-Code Overview**: Understanding the declarative approach to SLOs.
- **Benefits**: Version control, code review, and automation benefits.
- **Architecture**: How the SLO pipeline works.
- **Summary**: Key takeaways of the framework.

### [Implementation Guide](./slo-automation/implementation.md)
- **Automated SLO Generation**: How to generate SLOs automatically.
- **GitOps Workflow**: Managing SLOs via Git.
- **CI/CD Integration**: integrating SLO checks into pipelines.
- **Python Automation Tools**: Tooling support.
- **Service Discovery**: Automatically finding services to monitor.
- **End-to-End Example**: A complete walkthrough.

### [Configuration Reference](./slo-automation/configuration.md)
- **Schema Definitions**: YAML/JSON schemas for SLOs.
- **Template Library**: Reusable SLO templates.
- **Kubernetes CRD**: Custom Resource Definitions for K8s.

### [Best Practices & Strategies](./slo-automation/best-practices.md)
- **Progressive Implementation**: rolling out SLOs in phases.
- **Migration Strategies**: Moving from manual to automated SLOs.
- **Checklists**: Ensuring successful adoption.
