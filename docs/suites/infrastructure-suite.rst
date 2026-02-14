Infrastructure & Ops Suite
==========================

Consolidated suite for CI/CD automation, observability monitoring, and Git PR workflows. Includes Agent Teams task management for Claude Opus 4.6.

Agents
------

.. agent:: automation-engineer
   :description: Expert in automating software delivery pipelines and optimizing Git collaboration workflows.
   :model: sonnet
   :version: 2.2.0

.. agent:: devops-architect
   :description: Expert in multi-cloud architecture (AWS/Azure/GCP), Kubernetes orchestration, and Infrastructure as Code (Terraform).
   :model: sonnet
   :version: 2.2.0

.. agent:: sre-expert
   :description: Expert in system reliability, observability (monitoring, logging, tracing), and incident response.
   :model: sonnet
   :version: 2.2.0

Commands
--------

.. command:: /code-analyze
   :description: Semantic code analysis using Serena MCP for symbol navigation.

.. command:: /commit
   :description: Intelligent git commit with automated analysis and quality validation.

.. command:: /fix-commit-errors
   :description: Automatically analyzes GitHub Actions failures and applies solutions.

.. command:: /github-assist
   :description: GitHub operations using GitHub MCP for issues, PRs, and repos.

.. command:: /merge-all
   :description: Merge all local branches into main and clean up.

.. command:: /monitor-setup
   :description: Set up Prometheus, Grafana, and distributed tracing stack.

.. command:: /onboard
   :description: Orchestrate complete onboarding for new team members.

.. command:: /slo-implement
   :description: Implement SLO/SLA monitoring, error budgets, and alerting.

.. command:: /workflow-automate
   :description: Automated CI/CD workflow generation for GitHub Actions and GitLab CI.

Skills
------

.. skill:: airflow-scientific-workflows
   :description: Design Apache Airflow DAGs for scientific data pipelines, batch computations, distributed simulations, and time-series data ingestion with PostgreSQL/TimescaleDB integration. Use when orchestrating experimental workflows or coordinating scientific computations.
   :version: 2.2.0

.. skill:: deployment-pipeline-design
   :description: Design multi-stage CI/CD pipelines with approval gates, security checks, and progressive delivery (rolling, blue-green, canary, feature flags). Use when architecting deployment workflows, implementing GitOps, or establishing multi-environment promotion strategies.
   :version: 2.2.0

.. skill:: distributed-tracing
   :description: Implement distributed tracing with OpenTelemetry, Jaeger, and Tempo including instrumentation, context propagation, sampling strategies, and trace analysis. Use when debugging latency issues, understanding service dependencies, or tracing error propagation across microservices.
   :version: 2.2.0

.. skill:: git-workflow
   :description: Master advanced Git workflows for collaborative development. Covers interactive rebasing, cherry-picking, bisecting for bug discovery, and managing pull requests.
   :version: 2.2.0

.. skill:: github-actions-templates
   :description: Create production GitHub Actions workflows for testing, building, and deploying. Use when setting up CI pipelines, Docker builds, Kubernetes deployments, matrix builds, security scans, or reusable workflows.
   :version: 2.2.0

.. skill:: gitlab-ci-patterns
   :description: Build GitLab CI/CD pipelines with multi-stage workflows, caching, Docker builds, Kubernetes deployments, and security scanning. Use when creating .gitlab-ci.yml pipelines, setting up runners, implementing Terraform/IaC, or configuring GitOps workflows.
   :version: 2.2.0

.. skill:: grafana-dashboards
   :description: Create production Grafana dashboards with panels, variables, alerts, and templates using RED/USE methods. Use when building API monitoring, infrastructure, database, or SLO dashboards with Prometheus data sources.
   :version: 2.2.0

.. skill:: iterative-error-resolution
   :description: Iterative CI/CD error resolution with pattern recognition, automated fixes, and learning from outcomes. Use when debugging GitHub Actions, fixing dependency/build/test failures, or implementing automated error resolution loops.
   :version: 2.2.0

.. skill:: prometheus-configuration
   :description: Configure Prometheus for metric collection, alerting, and monitoring with scrape configs, recording rules, alert rules, and service discovery. Use when setting up Prometheus servers, creating alert rules, or implementing Kubernetes monitoring.
   :version: 2.2.0

.. skill:: secrets-management
   :description: Implement secrets management with HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, or platform-native solutions with encryption, rotation, and access control. Use when storing API keys, database passwords, TLS certificates, or implementing secret rotation.
   :version: 2.2.0

.. skill:: security-ci-template
   :description: Security scanning and lock file validation templates for CI/CD pipelines. Use when implementing SAST/DAST scanning, dependency vulnerability checks, lock file validation, or automated security gates in GitHub Actions or GitLab CI.
   :version: 2.2.0

.. skill:: slo-implementation
   :description: Define SLIs, SLOs, error budgets, and burn rate alerting following SRE best practices. Use when establishing reliability targets, implementing error budget policies, creating SLO dashboards, or designing multi-window burn rate alerts.
   :version: 2.2.0

