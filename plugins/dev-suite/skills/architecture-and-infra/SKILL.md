---
name: architecture-and-infra
description: Meta-orchestrator for software architecture and infrastructure. Routes to clean architecture, microservices, monorepo, systems/CLI, containers, and cloud patterns. Use when designing system architecture, implementing microservices, managing monorepos, building CLI tools, containerizing services, or deploying to cloud.
---

# Architecture and Infrastructure

Orchestrator for software architecture and infrastructure design. Routes to the appropriate specialized skill based on the structural concern, deployment target, or toolchain requirement.

## Expert Agent

- **`software-architect`**: Specialist for system decomposition, infrastructure design, and cross-cutting architectural concerns.
  - *Location*: `plugins/dev-suite/agents/software-architect.md`
  - *Capabilities*: Layered architecture, service mesh, monorepo strategy, container orchestration, and cloud-native patterns.

## Core Skills

### [Architecture Patterns](../architecture-patterns/SKILL.md)
Clean architecture, hexagonal design, domain-driven design, and CQRS/event sourcing.

### [Microservices Patterns](../microservices-patterns/SKILL.md)
Service decomposition, inter-service communication, sagas, and resilience patterns.

### [Monorepo Management](../monorepo-management/SKILL.md)
Nx, Turborepo, and pnpm workspaces for large-scale multi-package repositories.

### [Systems & CLI Engineering](../systems-cli-engineering/SKILL.md)
CLI design, systems programming, IPC, and low-level performance optimization.

### [Containerization Patterns](../containerization-patterns/SKILL.md)
Docker multi-stage builds, compose orchestration, and image optimization.

### [Cloud Provider Patterns](../cloud-provider-patterns/SKILL.md)
AWS, GCP, and Azure infrastructure-as-code, managed services, and cost optimization.

### [Modernization & Migration](../modernization-migration/SKILL.md)
Strangler Fig, framework migration playbooks, and database schema evolution.

## Routing Decision Tree

```
What is the architectural concern?
|
+-- Layering, boundaries, DDD, CQRS?
|   --> dev-suite:architecture-patterns
|
+-- Service decomposition / distributed transactions?
|   --> dev-suite:microservices-patterns
|
+-- Multi-package repo / build caching / workspaces?
|   --> dev-suite:monorepo-management
|
+-- CLI tools / systems programming / IPC?
|   --> dev-suite:systems-cli-engineering
|
+-- Docker / compose / image builds?
|   --> dev-suite:containerization-patterns
|
+-- Cloud IaC / managed services / multi-region?
|   --> dev-suite:cloud-provider-patterns
|
+-- Legacy modernization / Strangler Fig / framework migration?
|   --> dev-suite:modernization-migration
|
+-- None of the above / concern is ambiguous or spans multiple areas?
    --> Delegate to software-architect for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Routing Table

| Trigger                              | Sub-skill                    |
|--------------------------------------|------------------------------|
| Layers, ports/adapters, DDD, CQRS    | dev-suite:architecture-patterns        |
| Services, sagas, circuit breaker     | dev-suite:microservices-patterns       |
| Nx, Turborepo, pnpm workspaces       | dev-suite:monorepo-management          |
| CLI, argparse, IPC, syscalls         | dev-suite:systems-cli-engineering      |
| Dockerfile, compose, OCI images      | dev-suite:containerization-patterns    |
| Terraform, CDK, CloudFormation, IaC  | dev-suite:cloud-provider-patterns      |
| Strangler Fig, legacy migration, 2to3 | dev-suite:modernization-migration     |

## Checklist

- [ ] Identify the primary concern (structure vs deployment vs tooling) before routing
- [ ] Confirm bounded contexts are defined before applying microservices patterns
- [ ] Verify monorepo tooling matches team size and build frequency
- [ ] Check container images use multi-stage builds to minimize attack surface
- [ ] Validate cloud IaC is version-controlled and reviewed before apply
- [ ] Ensure architectural decisions are documented as ADRs
