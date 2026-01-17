---
name: sql-optimization-patterns
version: "1.0.7"
description: Master SQL optimization with EXPLAIN analysis, indexing strategies (B-Tree, GIN, partial, covering), N+1 elimination, pagination (cursor-based), aggregate optimization, materialized views, and partitioning. Use when debugging slow queries, designing schemas, or reducing database load.
---

# SQL Optimization Patterns

Transform slow queries into lightning-fast operations.

## Index Types

| Type | Use Case | Example |
|------|----------|---------|
| B-Tree | Equality, range queries | `WHERE email = ...` |
| Hash | Equality only | `WHERE id = ...` |
| GIN | Full-text, JSONB, arrays | `WHERE metadata @> ...` |
| BRIN | Very large tables, ordered data | Time-series data |
| Partial | Subset of rows | `WHERE status = 'active'` |
| Covering | Include additional columns | Avoid table access |

```sql
-- Composite index (order matters!)
CREATE INDEX idx_orders_user_status ON orders(user_id, status);

-- Partial index
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';

-- Covering index
CREATE INDEX idx_users_email ON users(email) INCLUDE (name, created_at);

-- GIN for JSONB
CREATE INDEX idx_metadata ON events USING GIN(metadata);
```

## EXPLAIN Analysis

```sql
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT * FROM users WHERE email = 'user@example.com';
```

| Scan Type | Meaning | Action |
|-----------|---------|--------|
| Seq Scan | Full table scan | Add index |
| Index Scan | Using index | Good |
| Index Only Scan | Index only, no table | Best |
| Nested Loop | Small dataset join | OK for small tables |
| Hash Join | Large dataset join | Good |

## N+1 Query Elimination

```python
# ❌ Bad: N+1 queries
users = db.query("SELECT * FROM users LIMIT 10")
for user in users:
    orders = db.query("SELECT * FROM orders WHERE user_id = ?", user.id)

# ✅ Good: JOIN or batch
SELECT u.*, o.* FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.id IN (1, 2, 3);

# Or batch load
SELECT * FROM orders WHERE user_id IN (1, 2, 3, 4, 5);
```

## Pagination

```sql
-- ❌ Bad: OFFSET on large tables
SELECT * FROM users ORDER BY created_at DESC LIMIT 20 OFFSET 100000;

-- ✅ Good: Cursor-based
SELECT * FROM users
WHERE created_at < '2024-01-15 10:30:00'
ORDER BY created_at DESC
LIMIT 20;

-- With composite cursor
WHERE (created_at, id) < ('2024-01-15', 12345)
ORDER BY created_at DESC, id DESC;
```

## Aggregate Optimization

```sql
-- Approximate count (fast)
SELECT reltuples::bigint FROM pg_class WHERE relname = 'orders';

-- Filter before grouping
SELECT user_id, COUNT(*) FROM orders
WHERE status = 'completed'  -- Filter first
GROUP BY user_id
HAVING COUNT(*) > 10;
```

## Correlated Subquery → JOIN

```sql
-- ❌ Bad: Correlated subquery (runs for each row)
SELECT u.name, (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id)
FROM users u;

-- ✅ Good: JOIN with aggregation
SELECT u.name, COUNT(o.id)
FROM users u LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.id, u.name;
```

## Materialized Views

```sql
CREATE MATERIALIZED VIEW user_order_summary AS
SELECT u.id, u.name, COUNT(o.id) as total_orders, SUM(o.total) as total_spent
FROM users u LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name;

CREATE INDEX idx_summary_spent ON user_order_summary(total_spent DESC);

REFRESH MATERIALIZED VIEW CONCURRENTLY user_order_summary;
```

## Partitioning

```sql
CREATE TABLE orders (
    id SERIAL, user_id INT, total DECIMAL, created_at TIMESTAMP
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2024_q1 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
```

## Monitoring

```sql
-- Slow queries
SELECT query, calls, mean_time FROM pg_stat_statements
ORDER BY mean_time DESC LIMIT 10;

-- Missing indexes (high seq_scan)
SELECT tablename, seq_scan, idx_scan
FROM pg_stat_user_tables WHERE seq_scan > idx_scan;

-- Unused indexes
SELECT indexname, idx_scan FROM pg_stat_user_indexes WHERE idx_scan = 0;
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Index selectively | Too many indexes slow writes |
| Monitor queries | Use slow query logs |
| Update statistics | Run ANALYZE regularly |
| Cursor pagination | Avoid OFFSET on large tables |
| Batch operations | Reduce round trips |

## Common Pitfalls

| Pitfall | Problem |
|---------|---------|
| Over-indexing | Slow INSERT/UPDATE |
| LIKE with leading wildcard | `LIKE '%abc'` can't use index |
| Function in WHERE | Prevents index unless functional index |
| Implicit type conversion | Prevents index usage |
| SELECT * | Fetches unnecessary columns |

## Checklist

- [ ] EXPLAIN ANALYZE on slow queries
- [ ] Appropriate indexes for query patterns
- [ ] No N+1 queries (check ORM)
- [ ] Cursor-based pagination
- [ ] Statistics up to date (VACUUM ANALYZE)
- [ ] Materialized views for expensive aggregates
