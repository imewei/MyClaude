---
name: backend-patterns
description: Meta-orchestrator for backend development patterns. Routes to Node.js, async Python, API design, GraphQL, WebSocket, and message queue skills. Use when building REST APIs, Node.js services, async Python backends, GraphQL endpoints, WebSocket connections, or message queue integrations.
---

> **Loading a routing target.** `dev-suite:*`, `research-suite:*`, and `science-suite:*` targets below
> are slash-only (`disable-model-invocation: true`) — the Skill tool will not fire them. Load one by
> reading its file with the Read tool at `${CLAUDE_PLUGIN_ROOT}/skills/<name>/SKILL.md` — Claude Code
> substitutes that variable with this plugin's install directory. A target in a sibling suite is at
> the same relative path under that suite's own root. Targets from other plugins (`superpowers:*`,
> `ecc:*`, …) are unaffected and invoke normally.

# Backend Patterns

Orchestrator for backend development across Node.js and Python ecosystems. Routes to the appropriate specialized skill based on the server technology, protocol, or messaging pattern required.

## Expert Agent

- **`software-architect`**: Specialist for system design, API contracts, and distributed backend architecture.
  - *Location*: `${CLAUDE_PLUGIN_ROOT}/agents/software-architect.md`
  - *Capabilities*: Service decomposition, API versioning, protocol selection, and scalability design.

## Core Skills

### [API Design Principles](../api-design-principles/SKILL.md)
REST resource modeling, versioning strategies, pagination, and contract-first design.

### [Message Queue Patterns](../message-queue-patterns/SKILL.md)
Producer/consumer workflows, dead-letter queues, and at-least-once delivery guarantees.

### [Error Handling Patterns](../error-handling-patterns/SKILL.md)
Exception hierarchies, Result types, retry with exponential backoff, circuit breakers, and structured error responses.

## Routing Decision Tree

```
What is the backend concern?
|
+-- REST API design / versioning / contract?
|   --> dev-suite:api-design-principles
|
+-- Async messaging / queues / events?
|   --> dev-suite:message-queue-patterns
|
+-- Exception design / retry / circuit breaker / error responses?
|   --> dev-suite:error-handling-patterns
|
+-- None of the above / concern is ambiguous or spans multiple areas?
    --> Delegate to software-architect for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Routing Table

| Trigger                        | Sub-skill                    |
|--------------------------------|------------------------------|
| REST, OpenAPI, versioning      | dev-suite:api-design-principles        |
| RabbitMQ, Kafka, SQS, queues  | dev-suite:message-queue-patterns       |
| try/except, retry, circuit breaker | dev-suite:error-handling-patterns  |

## Checklist

- [ ] Identify the runtime (Node.js vs Python) before selecting a sub-skill
- [ ] Confirm the communication protocol (REST / GraphQL / WebSocket / queue)
- [ ] Verify authentication and authorization are addressed in API design
- [ ] Check that async patterns handle backpressure and cancellation
- [ ] Validate message queue consumers implement idempotency
- [ ] Ensure error responses follow a consistent schema across all endpoints
