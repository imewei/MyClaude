---
name: api-design-principles
version: "1.0.7"
description: Master REST and GraphQL API design including resource-oriented architecture, HTTP semantics, pagination (cursor/offset), versioning strategies, error handling, HATEOAS, DataLoader patterns, and documentation. Use when designing new APIs, implementing pagination, handling errors, or establishing API standards.
---

# API Design Principles

REST and GraphQL API design for intuitive, scalable, and maintainable APIs.

## Paradigm Selection

| Paradigm | Best For | Key Characteristics |
|----------|----------|---------------------|
| REST | CRUD operations, caching, simplicity | Resource-oriented, HTTP semantics |
| GraphQL | Complex queries, mobile apps | Single endpoint, client-driven |
| gRPC | Internal services, high performance | Binary protocol, streaming |

## REST Resource Design

```python
# ✅ Good: Resource-oriented
GET    /api/users              # List users
POST   /api/users              # Create user
GET    /api/users/{id}         # Get user
PUT    /api/users/{id}         # Replace user
PATCH  /api/users/{id}         # Update fields
DELETE /api/users/{id}         # Delete user
GET    /api/users/{id}/orders  # Nested resource

# ❌ Bad: Action-oriented
POST   /api/createUser
POST   /api/getUserById
```

## HTTP Methods

| Method | Purpose | Idempotent | Safe |
|--------|---------|------------|------|
| GET | Retrieve resources | Yes | Yes |
| POST | Create new resource | No | No |
| PUT | Replace entire resource | Yes | No |
| PATCH | Partial update | No | No |
| DELETE | Remove resource | Yes | No |

## Pagination

### Offset-Based
```python
@app.get("/api/users")
async def list_users(page: int = 1, page_size: int = 20):
    offset = (page - 1) * page_size
    users = await fetch_users(limit=page_size, offset=offset)
    total = await count_users()
    return {
        "items": users,
        "total": total,
        "page": page,
        "pages": (total + page_size - 1) // page_size
    }
```

### Cursor-Based (GraphQL Relay)
```graphql
type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
}

type PageInfo {
  hasNextPage: Boolean!
  endCursor: String
}
```

## Error Handling

| Status | Meaning | Example |
|--------|---------|---------|
| 200 | Success | GET succeeded |
| 201 | Created | POST succeeded |
| 204 | No Content | DELETE succeeded |
| 400 | Bad Request | Invalid input |
| 401 | Unauthorized | Missing/invalid token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 422 | Unprocessable | Validation failed |
| 500 | Server Error | Internal failure |

### Consistent Error Response
```python
class ErrorResponse(BaseModel):
    error: str      # Error type
    message: str    # Human-readable message
    details: dict   # Additional context

def raise_not_found(resource: str, id: str):
    raise HTTPException(
        status_code=404,
        detail={"error": "NotFound", "message": f"{resource} not found", "details": {"id": id}}
    )
```

## GraphQL Schema Design

```graphql
type User {
  id: ID!
  email: String!
  orders(first: Int = 20, after: String): OrderConnection!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
}

type CreateUserPayload {
  user: User
  errors: [Error!]  # Errors in payload, not exceptions
}
```

## DataLoader (N+1 Prevention)

```python
from aiodataloader import DataLoader

class UserLoader(DataLoader):
    async def batch_load_fn(self, user_ids: list[str]) -> list[dict]:
        users = await fetch_users_by_ids(user_ids)
        user_map = {u["id"]: u for u in users}
        return [user_map.get(uid) for uid in user_ids]

# In resolver
@user_type.field("orders")
async def resolve_orders(user: dict, info):
    return await info.context["loaders"]["orders_by_user"].load(user["id"])
```

## API Versioning

| Strategy | Example | Pros/Cons |
|----------|---------|-----------|
| URL Path | `/api/v1/users` | Clear, cacheable |
| Header | `Accept: application/vnd.api+json; version=1` | Clean URLs |
| Query Param | `/api/users?version=1` | Easy testing |

## Best Practices

| Practice | REST | GraphQL |
|----------|------|---------|
| Naming | Plural nouns (`/users`) | camelCase fields |
| Pagination | Always paginate collections | Cursor-based (Relay) |
| Errors | HTTP status codes | Errors in payload |
| Versioning | Plan from day one | Use `@deprecated` |
| Documentation | OpenAPI/Swagger | Schema introspection |

## Common Pitfalls

| Pitfall | Problem |
|---------|---------|
| Breaking changes | No versioning strategy |
| N+1 queries | No DataLoader in GraphQL |
| Inconsistent errors | Different formats per endpoint |
| Over-fetching | Fixed response shapes |
| Poor documentation | Undocumented APIs frustrate devs |

## Checklist

- [ ] Resources named as plural nouns
- [ ] Correct HTTP methods and status codes
- [ ] Pagination on all collections
- [ ] Versioning strategy defined
- [ ] Consistent error response format
- [ ] Rate limiting implemented
- [ ] DataLoader for GraphQL relationships
- [ ] OpenAPI/GraphQL schema documented
