---
name: architecture-patterns
version: "1.0.7"
description: Master Clean Architecture, Hexagonal Architecture, and DDD patterns including entities, value objects, aggregates, repositories, and domain events. Use when designing scalable backends, refactoring monoliths, or implementing domain-driven design.
---

# Architecture Patterns

Clean Architecture, Hexagonal Architecture, and Domain-Driven Design for maintainable systems.

## Pattern Comparison

| Pattern | Core Idea | Best For |
|---------|-----------|----------|
| Clean Architecture | Dependencies point inward | Framework-independent business logic |
| Hexagonal | Ports & Adapters | Swappable infrastructure |
| DDD | Bounded contexts, ubiquitous language | Complex business domains |

## Clean Architecture Layers

```
┌─────────────────────────────────────┐
│         Frameworks (outer)          │  ← UI, DB, external services
├─────────────────────────────────────┤
│         Adapters                    │  ← Controllers, repositories
├─────────────────────────────────────┤
│         Use Cases                   │  ← Application logic
├─────────────────────────────────────┤
│         Entities (inner)            │  ← Business rules
└─────────────────────────────────────┘
Dependencies flow INWARD only
```

## Directory Structure

```
app/
├── domain/           # Entities, value objects, interfaces
├── use_cases/        # Application business rules
├── adapters/         # Repository implementations, controllers
└── infrastructure/   # Database, config, external services
```

## Clean Architecture Example

```python
# domain/entities/user.py
@dataclass
class User:
    id: str
    email: str
    is_active: bool = True

    def can_place_order(self) -> bool:
        return self.is_active

# domain/interfaces/user_repository.py
class IUserRepository(ABC):
    @abstractmethod
    async def find_by_id(self, user_id: str) -> Optional[User]: pass
    @abstractmethod
    async def save(self, user: User) -> User: pass

# use_cases/create_user.py
class CreateUserUseCase:
    def __init__(self, user_repository: IUserRepository):
        self.repo = user_repository

    async def execute(self, request: CreateUserRequest) -> CreateUserResponse:
        existing = await self.repo.find_by_email(request.email)
        if existing:
            return CreateUserResponse(success=False, error="Email exists")
        user = User(id=str(uuid4()), email=request.email)
        await self.repo.save(user)
        return CreateUserResponse(success=True, user=user)

# adapters/repositories/postgres_user_repository.py
class PostgresUserRepository(IUserRepository):
    async def find_by_id(self, user_id: str) -> Optional[User]:
        row = await self.pool.fetchrow("SELECT * FROM users WHERE id=$1", user_id)
        return self._to_entity(row) if row else None
```

## Hexagonal Architecture

```python
# Domain Core (hexagon center)
class OrderService:
    def __init__(self, order_repo: OrderRepositoryPort, payment: PaymentGatewayPort):
        self.orders = order_repo
        self.payments = payment

    async def place_order(self, order: Order) -> OrderResult:
        payment = await self.payments.charge(order.total, order.customer_id)
        if not payment.success:
            return OrderResult(success=False)
        await self.orders.save(order)
        return OrderResult(success=True)

# Ports (interfaces)
class PaymentGatewayPort(ABC):
    @abstractmethod
    async def charge(self, amount: Money, customer: str) -> PaymentResult: pass

# Adapters (implementations)
class StripePaymentAdapter(PaymentGatewayPort):
    async def charge(self, amount: Money, customer: str) -> PaymentResult:
        charge = stripe.Charge.create(amount=amount.cents, customer=customer)
        return PaymentResult(success=True, transaction_id=charge.id)

class MockPaymentAdapter(PaymentGatewayPort):
    async def charge(self, amount: Money, customer: str) -> PaymentResult:
        return PaymentResult(success=True, transaction_id="mock-123")
```

## DDD Tactical Patterns

```python
# Value Object (immutable)
@dataclass(frozen=True)
class Money:
    amount: int  # cents
    currency: str

    def add(self, other: "Money") -> "Money":
        assert self.currency == other.currency
        return Money(self.amount + other.amount, self.currency)

# Entity (with identity)
class Order:
    def __init__(self, id: str, customer: Customer):
        self.id = id
        self.items: List[OrderItem] = []
        self.status = OrderStatus.PENDING
        self._events: List[DomainEvent] = []

    def submit(self):
        if not self.items:
            raise ValueError("Cannot submit empty order")
        self.status = OrderStatus.SUBMITTED
        self._events.append(OrderSubmittedEvent(self.id))

# Aggregate (consistency boundary)
class Customer:
    def add_address(self, address: Address):
        if len(self._addresses) >= 5:
            raise ValueError("Maximum 5 addresses")
        self._addresses.append(address)

# Domain Event
@dataclass
class OrderSubmittedEvent:
    order_id: str
    occurred_at: datetime = field(default_factory=datetime.now)
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Dependency Rule | Dependencies always point inward |
| Interface Segregation | Small, focused interfaces |
| Business Logic in Domain | Keep frameworks out of core |
| Test Independence | Core testable without infrastructure |
| Rich Domain Models | Behavior with data |

## Common Pitfalls

| Pitfall | Problem |
|---------|---------|
| Anemic Domain | Entities with only data, no behavior |
| Framework Coupling | Business logic depends on frameworks |
| Fat Controllers | Business logic in controllers |
| Repository Leakage | Exposing ORM objects |
| Over-Engineering | Clean architecture for simple CRUD |

## Checklist

- [ ] Dependencies flow inward only
- [ ] Business rules in domain layer
- [ ] Repositories return domain entities (not ORM models)
- [ ] Use cases orchestrate business logic
- [ ] Adapters handle external concerns
- [ ] Core testable without frameworks
