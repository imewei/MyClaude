---
name: graphql-patterns
description: Design GraphQL APIs with schema-first development, resolvers, federation, subscriptions, and performance optimization including DataLoader and query complexity analysis. Use when building GraphQL servers, designing schemas, implementing federation, or optimizing GraphQL performance.
---

# GraphQL Patterns

## Expert Agent

For GraphQL API design, schema architecture, and federation strategies, delegate to:

- **`software-architect`**: Designs API architectures with schema-first development and service integration.
  - *Location*: `plugins/dev-suite/agents/software-architect.md`


## Schema Design

### Schema-First Approach

```graphql
type Query {
  user(id: ID!): User
  users(filter: UserFilter, pagination: PaginationInput): UserConnection!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(id: ID!, input: UpdateUserInput!): UpdateUserPayload!
}

type User {
  id: ID!
  name: String!
  email: String!
  orders(first: Int, after: String): OrderConnection!
  createdAt: DateTime!
}

input CreateUserInput {
  name: String!
  email: String!
}

type CreateUserPayload {
  user: User
  errors: [UserError!]!
}

type UserError {
  field: String!
  message: String!
}
```

## Resolver Patterns (Node.js)

```typescript
const resolvers = {
  Query: {
    user: async (_parent: unknown, args: { id: string }, context: Context) => {
      return context.dataSources.userAPI.getUser(args.id);
    },
    users: async (_parent: unknown, args: UserFilterArgs, context: Context) => {
      return context.dataSources.userAPI.getUsers(args.filter, args.pagination);
    },
  },
  User: {
    orders: async (parent: User, args: PaginationArgs, context: Context) => {
      return context.loaders.ordersByUserId.load(parent.id);
    },
  },
  Mutation: {
    createUser: async (_parent: unknown, args: { input: CreateUserInput }, context: Context) => {
      const user = await context.dataSources.userAPI.createUser(args.input);
      return { user, errors: [] };
    },
  },
};
```


## N+1 Prevention with DataLoader

```typescript
import DataLoader from "dataloader";

function createLoaders(db: Database) {
  return {
    ordersByUserId: new DataLoader(async (userIds: readonly string[]) => {
      const orders = await db.orders.findMany({
        where: { userId: { in: [...userIds] } },
      });
      const grouped = new Map<string, Order[]>();
      for (const order of orders) {
        const existing = grouped.get(order.userId) || [];
        existing.push(order);
        grouped.set(order.userId, existing);
      }
      return userIds.map((id) => grouped.get(id) || []);
    }),
  };
}
```

### DataLoader Rules

- Create new loaders per request (no cross-request caching)
- Batch function must return results in same order as keys
- Use `.prime()` to prepopulate cache from mutations


## Federation (Apollo Federation v2)

```graphql
# Users subgraph
type User @key(fields: "id") {
  id: ID!
  name: String!
  email: String!
}

# Orders subgraph
type User @key(fields: "id") {
  id: ID!
  orders: [Order!]!
}

type Order @key(fields: "id") {
  id: ID!
  total: Float!
  user: User!
}
```

## Security

### Query Depth Limiting

```typescript
import depthLimit from "graphql-depth-limit";

const server = new ApolloServer({
  schema,
  validationRules: [depthLimit(7)],
});
```

### Query Complexity Analysis

```typescript
import { createComplexityRule, simpleEstimator, fieldExtensionsEstimator } from "graphql-query-complexity";

const complexityRule = createComplexityRule({
  maximumComplexity: 1000,
  estimators: [
    fieldExtensionsEstimator(),
    simpleEstimator({ defaultComplexity: 1 }),
  ],
  onComplete: (complexity: number) => {
    console.log("Query complexity:", complexity);
  },
});
```

### Security Checklist

- [ ] Query depth limited (max 7-10 levels)
- [ ] Query complexity analysis enabled (max 1000)
- [ ] Introspection disabled in production
- [ ] Rate limiting per client/IP
- [ ] Input validation on all mutation arguments
- [ ] Authentication via context, not resolvers


## Design Checklist

- [ ] Schema-first design with clear type ownership
- [ ] DataLoader used for all nested field resolvers
- [ ] Cursor-based pagination for list fields
- [ ] Mutation payloads include typed errors
- [ ] Depth and complexity limits configured
- [ ] Federation boundaries align with team ownership
- [ ] Subscriptions use filtered topics
- [ ] Schema versioning via deprecation, not breaking changes
