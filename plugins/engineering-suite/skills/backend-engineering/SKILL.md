---
name: backend-engineering
version: "1.0.0"
description: Master backend development including API design (REST, GraphQL, gRPC), database optimization, microservices architecture, and authentication patterns.
---

# Backend Engineering

Expert guide for building scalable, secure, and maintainable server-side applications.

## 1. API Design & Implementation

### REST vs GraphQL vs gRPC
- **REST**: Best for CRUD, public APIs, and standard HTTP semantics. Use plural nouns (e.g., `/users`) and standard status codes (200, 201, 404, 500).
- **GraphQL**: Best for complex, relational data and mobile apps. Use DataLoaders to prevent N+1 query problems.
- **gRPC**: Best for high-performance internal microservices using Protocol Buffers.

### Pagination & Versioning
- **Pagination**: Always paginate collections using offset-based (simplicity) or cursor-based (scalability) strategies.
- **Versioning**: Plan for versioning from day one (e.g., `/api/v1/...`).

## 2. Data & Persistence

### SQL Optimization
- **Indexing**: Use B-Tree indexes for equality and range queries. Ensure composite indexes match query column order.
- **Query Analysis**: Use `EXPLAIN ANALYZE` to identify slow scans and missing indexes.

### Distributed Systems
- **Microservices**: Implement patterns like Saga (distributed transactions), CQRS (command-query separation), and Event Sourcing.
- **Caching**: Use Redis or Memcached for frequently accessed, slow-to-calculate data.

## 3. Security & Auth

- **Authentication**: Implement JWT (stateless) or Session-based (stateful) auth. Use Argon2 or bcrypt for password hashing.
- **Authorization**: Use RBAC (Role-Based) or ABAC (Attribute-Based) access control.

## 4. Backend Checklist

- [ ] **API Standards**: Are HTTP methods and status codes used correctly?
- [ ] **Performance**: Are N+1 queries eliminated? Are slow queries indexed?
- [ ] **Security**: Is input validation applied to all user-provided data?
- [ ] **Resilience**: Are timeouts, retries, and circuit breakers implemented?
- [ ] **Monitoring**: Are critical paths instrumented with logging and metrics?
