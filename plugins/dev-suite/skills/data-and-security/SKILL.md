---
name: data-and-security
description: Meta-orchestrator for data layer and security patterns. Routes to database, SQL, caching, search, authentication, and secrets management skills. Use when designing database schemas, optimizing SQL queries, implementing caching, building search, adding authentication, or managing secrets.
---

# Data and Security

Orchestrator for data layer design and application security. Routes to the appropriate specialized skill based on the storage type, query concern, or security domain.

## Expert Agent

- **`software-architect`**: Specialist for data modeling, storage selection, and security architecture.
  - *Location*: `plugins/dev-suite/agents/software-architect.md`
  - *Capabilities*: Schema design, query optimization, caching topology, auth protocol selection, and secrets lifecycle management.

## Core Skills

### [Database Patterns](../database-patterns/SKILL.md)
Relational and NoSQL schema design, migrations, indexing strategy, and connection pooling.

### [SQL Optimization Patterns](../sql-optimization-patterns/SKILL.md)
Query plans, index tuning, N+1 elimination, and partitioning for high-throughput workloads.

### [Caching Patterns](../caching-patterns/SKILL.md)
Cache-aside, write-through, TTL strategy, and Redis/Memcached configuration.

### [Search Patterns](../search-patterns/SKILL.md)
Elasticsearch/OpenSearch index design, query DSL, relevance tuning, and vector search.

### [Auth Implementation Patterns](../auth-implementation-patterns/SKILL.md)
OAuth2, OIDC, JWT, session management, and RBAC/ABAC authorization models.

### [Secrets Management](../secrets-management/SKILL.md)
Vault, AWS Secrets Manager, environment injection, and secret rotation strategies.

## Routing Decision Tree

```
What is the data or security concern?
|
+-- Schema design / migrations / connection pools?
|   --> database-patterns
|
+-- Slow queries / index tuning / N+1?
|   --> sql-optimization-patterns
|
+-- Cache invalidation / TTL / Redis config?
|   --> caching-patterns
|
+-- Full-text or vector search / relevance?
|   --> search-patterns
|
+-- Login / OAuth2 / JWT / RBAC?
|   --> auth-implementation-patterns
|
+-- Secrets injection / rotation / Vault?
    --> secrets-management
```

## Routing Table

| Trigger                                  | Sub-skill                      |
|------------------------------------------|--------------------------------|
| ORM, migrations, indexes, pooling        | database-patterns              |
| EXPLAIN, slow query, N+1, partition      | sql-optimization-patterns      |
| Redis, TTL, cache-aside, invalidation    | caching-patterns               |
| Elasticsearch, vector search, DSL        | search-patterns                |
| OAuth2, OIDC, JWT, RBAC, session         | auth-implementation-patterns   |
| Vault, AWS SM, .env, secret rotation     | secrets-management             |

## Checklist

- [ ] Identify the storage type (relational / NoSQL / cache / search) before routing
- [ ] Verify SQL queries have execution plan analysis before deploying to production
- [ ] Confirm cache TTLs are set based on data freshness requirements, not defaults
- [ ] Check that auth tokens have appropriate expiry and rotation policies
- [ ] Validate secrets are never committed to version control or logged
- [ ] Ensure search indexes are tested for recall and precision before going live
