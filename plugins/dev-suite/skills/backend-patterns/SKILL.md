---
name: backend-patterns
description: Meta-orchestrator for backend development patterns. Routes to Node.js, async Python, API design, GraphQL, WebSocket, and message queue skills. Use when building REST APIs, Node.js services, async Python backends, GraphQL endpoints, WebSocket connections, or message queue integrations.
---

# Backend Patterns

Orchestrator for backend development across Node.js and Python ecosystems. Routes to the appropriate specialized skill based on the server technology, protocol, or messaging pattern required.

## Expert Agent

- **`software-architect`**: Specialist for system design, API contracts, and distributed backend architecture.
  - *Location*: `plugins/dev-suite/agents/software-architect.md`
  - *Capabilities*: Service decomposition, API versioning, protocol selection, and scalability design.

## Core Skills

### [Node.js Backend Patterns](../nodejs-backend-patterns/SKILL.md)
Express/Fastify servers, middleware chains, and Node.js runtime patterns. For language-agnostic REST API design (versioning, pagination, contracts), see `api-design-principles`.

### [Async Python Patterns](../async-python-patterns/SKILL.md)
FastAPI, asyncio concurrency, and async I/O optimization for Python backends.

### [API Design Principles](../api-design-principles/SKILL.md)
REST resource modeling, versioning strategies, pagination, and contract-first design.

### [GraphQL Patterns](../graphql-patterns/SKILL.md)
Schema design, resolvers, DataLoader batching, and federation.

### [WebSocket Patterns](../websocket-patterns/SKILL.md)
Real-time bidirectional communication, connection lifecycle, and pub/sub over WebSockets.

### [Message Queue Patterns](../message-queue-patterns/SKILL.md)
Producer/consumer workflows, dead-letter queues, and at-least-once delivery guarantees.

## Routing Decision Tree

```
What is the backend concern?
|
+-- Node.js server / middleware / runtime performance?
|   --> nodejs-backend-patterns
|
+-- Python async service / FastAPI / asyncio?
|   --> async-python-patterns
|
+-- REST API design / versioning / contract?
|   --> api-design-principles
|
+-- GraphQL schema / resolvers / federation?
|   --> graphql-patterns
|
+-- Real-time connections / live updates?
|   --> websocket-patterns
|
+-- Async messaging / queues / events?
    --> message-queue-patterns
```

## Routing Table

| Trigger                        | Sub-skill                    |
|--------------------------------|------------------------------|
| Express, Fastify, Node streams | nodejs-backend-patterns      |
| FastAPI, asyncio, aiohttp      | async-python-patterns        |
| REST, OpenAPI, versioning      | api-design-principles        |
| GraphQL, Apollo, federation    | graphql-patterns             |
| WebSocket, SSE, real-time      | websocket-patterns           |
| RabbitMQ, Kafka, SQS, queues  | message-queue-patterns       |

## Checklist

- [ ] Identify the runtime (Node.js vs Python) before selecting a sub-skill
- [ ] Confirm the communication protocol (REST / GraphQL / WebSocket / queue)
- [ ] Verify authentication and authorization are addressed in API design
- [ ] Check that async patterns handle backpressure and cancellation
- [ ] Validate message queue consumers implement idempotency
- [ ] Ensure error responses follow a consistent schema across all endpoints
