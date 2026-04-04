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

## Routing Decision Tree

```
What is the architectural concern?
|
+-- Layering, boundaries, DDD, CQRS?
|   --> architecture-patterns
|
+-- Service decomposition / distributed transactions?
|   --> microservices-patterns
|
+-- Multi-package repo / build caching / workspaces?
|   --> monorepo-management
|
+-- CLI tools / systems programming / IPC?
|   --> systems-cli-engineering
|
+-- Docker / compose / image builds?
|   --> containerization-patterns
|
+-- Cloud IaC / managed services / multi-region?
    --> cloud-provider-patterns
```

## Routing Table

| Trigger                              | Sub-skill                    |
|--------------------------------------|------------------------------|
| Layers, ports/adapters, DDD, CQRS    | architecture-patterns        |
| Services, sagas, circuit breaker     | microservices-patterns       |
| Nx, Turborepo, pnpm workspaces       | monorepo-management          |
| CLI, argparse, IPC, syscalls         | systems-cli-engineering      |
| Dockerfile, compose, OCI images      | containerization-patterns    |
| Terraform, CDK, CloudFormation, IaC  | cloud-provider-patterns      |

## Checklist

- [ ] Identify the primary concern (structure vs deployment vs tooling) before routing
- [ ] Confirm bounded contexts are defined before applying microservices patterns
- [ ] Verify monorepo tooling matches team size and build frequency
- [ ] Check container images use multi-stage builds to minimize attack surface
- [ ] Validate cloud IaC is version-controlled and reviewed before apply
- [ ] Ensure architectural decisions are documented as ADRs
