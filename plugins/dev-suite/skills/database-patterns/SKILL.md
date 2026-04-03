---
name: database-patterns
description: Master database design patterns including ORM usage (SQLAlchemy, Prisma, TypeORM), schema migrations (Alembic, Flyway), query optimization, connection pooling, and data modeling. Use when designing database schemas, writing migrations, optimizing queries, or integrating ORMs.
---

# Database Patterns

## Expert Agent

For database design, schema modeling, and query optimization, delegate to:

- **`software-architect`**: Designs data models, schema strategies, and ORM integration patterns.
  - *Location*: `plugins/dev-suite/agents/software-architect.md`


## Schema Design Principles

| Normal Form | Rule | When to Denormalize |
|-------------|------|---------------------|
| 1NF | Atomic values, no repeating groups | Never skip |
| 2NF | No partial dependencies | Rarely skip |
| 3NF | No transitive dependencies | Read-heavy analytics |
| BCNF | Every determinant is a candidate key | Complex joins hurt perf |

### Naming Conventions

- Tables: plural snake_case (`user_accounts`)
- Columns: singular snake_case (`created_at`)
- Foreign keys: `<referenced_table_singular>_id`
- Indexes: `idx_<table>_<columns>`
- Constraints: `chk_<table>_<rule>`, `uq_<table>_<columns>`


## ORM Patterns

### SQLAlchemy (Python)

```python
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine
from sqlalchemy.orm import relationship, Session, DeclarativeBase

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    orders = relationship("Order", back_populates="user", lazy="selectin")

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="orders")
```

## Migration Strategies

### Alembic (Python/SQLAlchemy)

```bash
alembic init migrations
alembic revision --autogenerate -m "add_users_table"
alembic upgrade head
alembic downgrade -1
```

### Migration Safety Checklist

- [ ] Migration is backward-compatible with current app version
- [ ] Large table ALTERs use `pt-online-schema-change` or equivalent
- [ ] New columns have defaults or are nullable
- [ ] Index creation uses `CONCURRENTLY` (PostgreSQL)
- [ ] Data migrations are separate from schema migrations
- [ ] Rollback migration is tested


## Query Optimization

### N+1 Prevention

```python
# BAD: N+1 queries
users = session.query(User).all()
for user in users:
    print(user.orders)  # Triggers a query per user

# GOOD: Eager loading
users = session.query(User).options(selectinload(User.orders)).all()
for user in users:
    print(user.orders)  # Already loaded
```

### Index Strategy

| Query Pattern | Index Type | Example |
|---------------|-----------|---------|
| Equality lookup | B-Tree | `WHERE status = 'active'` |
| Range scan | B-Tree | `WHERE created_at > '2024-01-01'` |
| Full-text search | GIN/GiST | `WHERE body @@ to_tsquery('search')` |
| JSON field | GIN | `WHERE metadata->>'key' = 'val'` |
| Partial match | Partial index | `WHERE status = 'active'` on subset |

### EXPLAIN Analysis

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT u.name, COUNT(o.id)
FROM users u
JOIN orders o ON o.user_id = u.id
WHERE u.created_at > '2024-01-01'
GROUP BY u.name;
```

Watch for: `Seq Scan` on large tables, `Nested Loop` with high row counts, `Sort` without index.


## Connection Pooling

### Configuration Guidelines

| Setting | Development | Production |
|---------|------------|------------|
| Pool size | 5 | 20-50 |
| Max overflow | 5 | 10 |
| Pool timeout | 30s | 10s |
| Pool recycle | 3600s | 1800s |

```python
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@host/db",
    pool_size=20,
    max_overflow=10,
    pool_timeout=10,
    pool_recycle=1800,
    pool_pre_ping=True,  # Verify connections before use
)
```


## Design Checklist

- [ ] Schema normalized to 3NF minimum
- [ ] Foreign keys and constraints enforced at DB level
- [ ] Indexes cover top query patterns
- [ ] Connection pooling configured for environment
- [ ] Migrations are reversible and tested
- [ ] N+1 queries eliminated with eager loading
- [ ] Transactions scoped to logical units of work
- [ ] Sensitive data encrypted at rest
