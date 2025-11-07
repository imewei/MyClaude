# Development Methodology Guides

Comprehensive guide to implementing different development methodologies in the feature development workflow.

## Table of Contents
- [Traditional Development](#traditional-development)
- [Test-Driven Development (TDD)](#test-driven-development-tdd)
- [Behavior-Driven Development (BDD)](#behavior-driven-development-bdd)
- [Domain-Driven Design (DDD)](#domain-driven-design-ddd)
- [Methodology Selection Guide](#methodology-selection-guide)

---

## Traditional Development

### Overview
Sequential development approach where implementation precedes testing. Best for rapid prototyping, MVP development, and projects with evolving requirements.

### Workflow
1. **Requirements Analysis** → 2. **Architecture Design** → 3. **Implementation** → 4. **Testing** → 5. **Deployment**

### Phase Adaptations

**Phase 1: Discovery**
- Standard requirements gathering
- Flexible architecture design allowing for changes
- Risk assessment focused on timeline and scope creep

**Phase 2: Implementation**
- Focus on rapid feature delivery
- Implement core functionality first
- Add error handling and edge cases iteratively
- Use feature flags to ship incomplete features safely

**Phase 3: Testing**
- Manual testing during development
- Automated tests written after implementation
- Focus on integration and E2E tests
- Acceptance testing with stakeholders

**Phase 4: Deployment**
- Gradual rollout with monitoring
- Quick iteration based on user feedback
- Post-deployment bug fixes

### Best Practices
- ✅ Maintain clean commit history for easier debugging
- ✅ Write tests for critical business logic even if after implementation
- ✅ Use feature flags to separate deployment from release
- ✅ Document decisions and technical debt as you go
- ⚠️ Be mindful of accumulating technical debt
- ⚠️ Plan refactoring cycles

### When to Use
- MVP and proof-of-concept development
- Tight deadlines with evolving requirements
- Projects where user feedback drives features
- Teams new to test-first methodologies

---

## Test-Driven Development (TDD)

### Overview
Test-first development approach following the red-green-refactor cycle. Ensures high test coverage and drives clean, testable design.

### Red-Green-Refactor Cycle
1. **Red**: Write a failing test for the next small piece of functionality
2. **Green**: Write the minimum code to make the test pass
3. **Refactor**: Improve code quality while keeping tests green

### Phase Adaptations

**Phase 1: Discovery**
- Define acceptance criteria as testable specifications
- Break down features into small, testable units
- Identify test boundaries and dependencies

**Phase 2: Implementation**
```
For each feature component:
  1. Write unit test (red)
  2. Implement minimum code (green)
  3. Refactor for quality
  4. Commit
  Repeat
```

**Backend TDD Example**:
```typescript
// 1. RED - Write failing test
describe('UserService', () => {
  it('should create user with hashed password', async () => {
    const userData = { email: 'test@example.com', password: 'secret123' };
    const user = await userService.create(userData);

    expect(user.email).toBe('test@example.com');
    expect(user.password).not.toBe('secret123'); // Should be hashed
    expect(await bcrypt.compare('secret123', user.password)).toBe(true);
  });
});

// 2. GREEN - Implement minimum code
class UserService {
  async create(userData: CreateUserDto): Promise<User> {
    const hashedPassword = await bcrypt.hash(userData.password, 10);
    return this.repository.save({
      ...userData,
      password: hashedPassword
    });
  }
}

// 3. REFACTOR - Improve code quality
class UserService {
  constructor(
    private repository: UserRepository,
    private passwordHasher: PasswordHasher
  ) {}

  async create(userData: CreateUserDto): Promise<User> {
    const hashedPassword = await this.passwordHasher.hash(userData.password);
    return this.repository.save({
      ...userData,
      password: hashedPassword
    });
  }
}
```

**Frontend TDD Example**:
```typescript
// 1. RED - Write failing test
describe('LoginForm', () => {
  it('should disable submit button when form is invalid', () => {
    const { getByRole } = render(<LoginForm />);
    const submitButton = getByRole('button', { name: /login/i });

    expect(submitButton).toBeDisabled();
  });
});

// 2. GREEN - Implement minimum code
const LoginForm = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const isValid = email.includes('@') && password.length >= 8;

  return (
    <form>
      <input type="email" value={email} onChange={e => setEmail(e.target.value)} />
      <input type="password" value={password} onChange={e => setPassword(e.target.value)} />
      <button disabled={!isValid}>Login</button>
    </form>
  );
};

// 3. REFACTOR - Extract validation logic
const useLoginValidation = (email: string, password: string) => {
  return useMemo(() => ({
    isValid: email.includes('@') && password.length >= 8,
    errors: {
      email: !email.includes('@') ? 'Invalid email' : null,
      password: password.length < 8 ? 'Password must be at least 8 characters' : null
    }
  }), [email, password]);
};
```

**Phase 3: Testing**
- Tests already written during implementation
- Focus on integration and E2E tests
- Maintain high code coverage (typically >90%)

**Phase 4: Deployment**
- High confidence in deployments due to comprehensive tests
- Regression risks minimized

### Best Practices
- ✅ Write the smallest test that fails
- ✅ Write only enough code to pass the test
- ✅ Refactor fearlessly with tests as safety net
- ✅ Test behavior, not implementation details
- ✅ Use dependency injection for testability
- ⚠️ Don't skip the refactor step
- ⚠️ Avoid testing framework internals

### When to Use
- Greenfield projects with clear requirements
- Critical business logic requiring high reliability
- Teams experienced with TDD practices
- Long-lived codebases requiring maintainability

---

## Behavior-Driven Development (BDD)

### Overview
Collaborative approach using natural language specifications (Gherkin) to define feature behavior. Bridges the gap between business stakeholders and technical teams.

### Gherkin Syntax
```gherkin
Feature: User Authentication
  As a user
  I want to log in securely
  So that I can access my account

  Scenario: Successful login with valid credentials
    Given I am on the login page
    When I enter valid credentials
    And I click the login button
    Then I should be redirected to the dashboard
    And I should see a welcome message

  Scenario: Failed login with invalid credentials
    Given I am on the login page
    When I enter invalid credentials
    And I click the login button
    Then I should see an error message
    And I should remain on the login page
```

### Phase Adaptations

**Phase 1: Discovery**
- Write feature files collaboratively with stakeholders
- Define scenarios covering happy paths and edge cases
- Use examples to clarify requirements
- Create shared understanding through ubiquitous language

**Phase 2: Implementation**
```
For each scenario:
  1. Write step definitions (pending)
  2. Implement application code
  3. Verify scenarios pass
  4. Refactor
  Repeat
```

**Step Definition Example** (Cucumber/Jest):
```typescript
// features/step_definitions/auth.steps.ts
import { Given, When, Then } from '@cucumber/cucumber';
import { render, screen, userEvent } from '@testing-library/react';

Given('I am on the login page', async function() {
  this.component = render(<LoginPage />);
});

When('I enter valid credentials', async function() {
  await userEvent.type(screen.getByLabelText(/email/i), 'user@example.com');
  await userEvent.type(screen.getByLabelText(/password/i), 'ValidPassword123');
});

When('I click the login button', async function() {
  await userEvent.click(screen.getByRole('button', { name: /login/i }));
});

Then('I should be redirected to the dashboard', async function() {
  await waitFor(() => {
    expect(window.location.pathname).toBe('/dashboard');
  });
});
```

**Phase 3: Testing**
- Scenarios serve as acceptance tests
- Additional unit tests for complex logic
- Scenarios become living documentation

**Phase 4: Deployment**
- Feature files communicate what's being deployed
- Non-technical stakeholders can review scenarios

### Best Practices
- ✅ Write scenarios from user perspective
- ✅ Use domain language, not technical jargon
- ✅ Keep scenarios focused and independent
- ✅ Involve stakeholders in scenario writing
- ✅ Maintain scenarios as requirements change
- ⚠️ Avoid over-specifying UI interactions
- ⚠️ Don't duplicate scenarios with different data

### When to Use
- Complex domain logic requiring stakeholder collaboration
- Regulated industries needing traceability
- Distributed teams needing shared understanding
- Projects where documentation is critical

---

## Domain-Driven Design (DDD)

### Overview
Strategic design approach organizing code around business domains. Uses bounded contexts, aggregates, entities, and value objects to model complex business logic.

### Core Concepts

**Bounded Context**: Explicit boundary within which a domain model is defined
**Aggregate**: Cluster of domain objects treated as a single unit
**Entity**: Object with distinct identity that persists over time
**Value Object**: Immutable object defined by its attributes
**Domain Event**: Something that happened in the domain that domain experts care about

### Phase Adaptations

**Phase 1: Discovery**
- **Event Storming**: Collaborative workshop to discover domain events
- **Context Mapping**: Identify bounded contexts and their relationships
- **Ubiquitous Language**: Define shared vocabulary
- **Aggregate Design**: Identify transactional boundaries

**Event Storming Process**:
```
1. Domain Events (orange): UserRegistered, PaymentProcessed, OrderShipped
2. Commands (blue): RegisterUser, ProcessPayment, ShipOrder
3. Aggregates (yellow): User, Payment, Order
4. Policies (purple): When PaymentProcessed → ShipOrder
5. Read Models (green): OrderSummary, UserProfile
```

**Phase 2: Implementation**

**Aggregate Example**:
```typescript
// domain/order/Order.ts
export class Order {
  private constructor(
    private readonly id: OrderId,
    private customerId: CustomerId,
    private items: OrderItem[],
    private status: OrderStatus,
    private total: Money
  ) {}

  static create(customerId: CustomerId, items: OrderItem[]): Order {
    // Invariant: Order must have at least one item
    if (items.length === 0) {
      throw new DomainError('Order must contain at least one item');
    }

    const total = items.reduce((sum, item) => sum.add(item.price), Money.zero());
    const order = new Order(
      OrderId.generate(),
      customerId,
      items,
      OrderStatus.Pending,
      total
    );

    // Domain event
    order.addDomainEvent(new OrderCreated(order.id, customerId, total));
    return order;
  }

  confirm(paymentId: PaymentId): void {
    // Invariant: Can only confirm pending orders
    if (!this.status.isPending()) {
      throw new DomainError('Only pending orders can be confirmed');
    }

    this.status = OrderStatus.Confirmed;
    this.addDomainEvent(new OrderConfirmed(this.id, paymentId));
  }

  // Aggregate root enforces invariants
  addItem(item: OrderItem): void {
    if (!this.status.isPending()) {
      throw new DomainError('Cannot modify confirmed order');
    }

    this.items.push(item);
    this.total = this.total.add(item.price);
    this.addDomainEvent(new OrderItemAdded(this.id, item.id));
  }
}
```

**Repository Pattern**:
```typescript
// domain/order/OrderRepository.ts
export interface OrderRepository {
  findById(id: OrderId): Promise<Order | null>;
  save(order: Order): Promise<void>;
  findByCustomer(customerId: CustomerId): Promise<Order[]>;
}

// infrastructure/persistence/PostgresOrderRepository.ts
export class PostgresOrderRepository implements OrderRepository {
  async save(order: Order): Promise<void> {
    // Persist aggregate
    await this.db.transaction(async (trx) => {
      await this.saveOrderData(order, trx);
      await this.publishDomainEvents(order.domainEvents, trx);
    });
  }
}
```

**Application Service** (orchestrates use cases):
```typescript
// application/CreateOrderService.ts
export class CreateOrderService {
  constructor(
    private orderRepository: OrderRepository,
    private productRepository: ProductRepository,
    private eventBus: DomainEventBus
  ) {}

  async execute(command: CreateOrderCommand): Promise<OrderId> {
    // Load necessary aggregates
    const products = await this.productRepository.findByIds(command.productIds);

    // Domain logic in aggregate
    const items = products.map(p => OrderItem.create(p.id, p.price, command.quantities[p.id]));
    const order = Order.create(command.customerId, items);

    // Persist
    await this.orderRepository.save(order);

    // Publish events
    await this.eventBus.publishAll(order.domainEvents);

    return order.id;
  }
}
```

**Phase 3: Testing**
- Unit test aggregates and domain logic
- Integration test repositories and application services
- Test domain event handlers
- Verify invariants are enforced

**Phase 4: Deployment**
- Deploy bounded contexts independently if microservices
- Monitor domain events for business insights
- Use event sourcing for audit trail (optional)

### Best Practices
- ✅ Protect aggregate invariants
- ✅ Use value objects for concepts without identity
- ✅ Keep aggregates small and focused
- ✅ Use domain events for cross-aggregate communication
- ✅ Model the domain, not the database
- ✅ Collaborate with domain experts continuously
- ⚠️ Don't create anemic domain models (all logic in services)
- ⚠️ Avoid large, god-like aggregates

### When to Use
- Complex business domains with intricate rules
- Applications expected to evolve over years
- Microservices architectures
- Teams with dedicated domain experts
- Event-driven systems

---

## Methodology Selection Guide

### Decision Matrix

| Criteria | Traditional | TDD | BDD | DDD |
|----------|------------|-----|-----|-----|
| **Project Complexity** | Low-Medium | Medium-High | Medium-High | High |
| **Requirements Clarity** | Evolving | Clear | Needs Collaboration | Complex Domain |
| **Team Experience** | Any | TDD Background | BDD Tools | DDD Expertise |
| **Time to Market** | Fastest | Medium | Medium | Slowest |
| **Test Coverage** | Variable | Very High | High | High |
| **Stakeholder Involvement** | Low | Low | High | Very High |
| **Long-term Maintainability** | Medium | High | High | Very High |
| **Best for** | MVPs, Prototypes | Critical Systems | Complex Features | Enterprise Apps |

### Hybrid Approaches

**TDD + DDD**:
- Write tests for aggregate behavior
- Test-drive domain logic
- Use TDD for infrastructure layer

**BDD + TDD**:
- BDD scenarios for acceptance criteria
- TDD for implementation details
- Scenarios verify behavior, unit tests verify logic

**All Methodologies**:
- Use appropriate methodology per layer
- DDD for domain layer
- TDD for business logic
- BDD for user-facing features
- Traditional for infrastructure/glue code

### Questions to Ask

1. **How well understood is the domain?**
   - Clear and simple → Traditional or TDD
   - Complex but known → DDD
   - Needs discovery → BDD or Event Storming (DDD)

2. **How critical is correctness?**
   - Must be correct → TDD or BDD
   - Can iterate → Traditional

3. **How involved are stakeholders?**
   - Highly involved → BDD or DDD
   - Limited involvement → TDD or Traditional

4. **What's the team's skill level?**
   - Junior team → Traditional (with mentoring)
   - Experienced team → TDD, BDD, or DDD

5. **How long will this codebase live?**
   - Short-term (< 1 year) → Traditional
   - Long-term (5+ years) → DDD + TDD

---

## References

- **TDD**: "Test Driven Development: By Example" by Kent Beck
- **BDD**: "The Cucumber Book" by Matt Wynne and Aslak Hellesøy
- **DDD**: "Domain-Driven Design" by Eric Evans, "Implementing Domain-Driven Design" by Vaughn Vernon
- **Event Storming**: Alberto Brandolini's Event Storming method
