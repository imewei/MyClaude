---
name: microservices-patterns
version: "1.0.7"
description: Design microservices with proper service boundaries, event-driven communication (Kafka/RabbitMQ), resilience patterns (circuit breakers, retries, bulkheads), Saga pattern for distributed transactions, API Gateway, service discovery, database-per-service, CQRS, and distributed tracing. Use when decomposing monoliths or building distributed systems.
---

# Microservices Patterns

Service boundaries, communication, data management, and resilience patterns.

## Decomposition Strategies

| Strategy | Approach | Example |
|----------|----------|---------|
| Business Capability | Organize by business function | OrderService, PaymentService |
| DDD Subdomain | Bounded contexts | Core, Supporting, Generic |
| Strangler Fig | Gradual extraction | Proxy routes→old/new |

## Communication Patterns

| Type | Mechanism | Use Case |
|------|-----------|----------|
| Synchronous | REST, gRPC, GraphQL | Real-time response needed |
| Asynchronous | Kafka, RabbitMQ, SQS | Decoupled, event-driven |

### Event-Driven
```python
@dataclass
class DomainEvent:
    event_id: str
    event_type: str
    aggregate_id: str
    occurred_at: datetime
    data: dict

class EventBus:
    async def publish(self, event: DomainEvent):
        await self.producer.send_and_wait(
            event.event_type,
            value=asdict(event),
            key=event.aggregate_id.encode()
        )

    async def subscribe(self, topic: str, handler: callable):
        async for message in self.consumer:
            await handler(message.value)
```

## API Gateway

```python
class APIGateway:
    @circuit(failure_threshold=5, recovery_timeout=30)
    async def call_service(self, service_url: str, path: str, **kwargs):
        response = await self.http_client.request("GET", f"{service_url}{path}", **kwargs)
        response.raise_for_status()
        return response.json()

    async def aggregate(self, order_id: str):
        """Aggregate from multiple services"""
        order, payment, inventory = await asyncio.gather(
            self.call_order_service(f"/orders/{order_id}"),
            self.call_payment_service(f"/payments/order/{order_id}"),
            self.call_inventory_service(f"/reservations/order/{order_id}"),
            return_exceptions=True
        )
        return {"order": order, "payment": payment, "inventory": inventory}
```

## Saga Pattern

```python
class SagaStep:
    def __init__(self, name: str, action: Callable, compensation: Callable):
        self.name = name
        self.action = action
        self.compensation = compensation

class OrderFulfillmentSaga:
    def __init__(self):
        self.steps = [
            SagaStep("create_order", self.create_order, self.cancel_order),
            SagaStep("reserve_inventory", self.reserve_inventory, self.release_inventory),
            SagaStep("process_payment", self.process_payment, self.refund_payment),
            SagaStep("confirm_order", self.confirm_order, self.cancel_confirmation)
        ]

    async def execute(self, order_data: dict):
        completed = []
        context = {"order_data": order_data}

        for step in self.steps:
            result = await step.action(context)
            if not result.success:
                await self.compensate(completed, context)
                return SagaResult(status="FAILED", error=result.error)
            completed.append(step)
            context.update(result.data)

        return SagaResult(status="COMPLETED", data=context)

    async def compensate(self, completed: list, context: dict):
        for step in reversed(completed):
            await step.compensation(context)
```

## Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED"
        self.opened_at = None

    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if datetime.now() - self.opened_at > timedelta(seconds=self.recovery_timeout):
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"

    def _on_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.opened_at = datetime.now()
```

## Service Client with Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class ServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=5.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def get(self, path: str, **kwargs):
        response = await self.client.get(f"{self.base_url}{path}", **kwargs)
        response.raise_for_status()
        return response.json()
```

## Data Management

| Pattern | Description |
|---------|-------------|
| Database per Service | Each service owns its data |
| Event Sourcing | Store events, derive state |
| CQRS | Separate read/write models |
| Eventual Consistency | Accept async propagation |

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Service boundaries | Align with business capabilities |
| Database per service | No shared databases |
| API contracts | Versioned, backward compatible |
| Async when possible | Events over direct calls |
| Circuit breakers | Fail fast on service failures |
| Distributed tracing | Track requests across services |
| Health checks | Liveness and readiness probes |

## Common Pitfalls

| Pitfall | Problem |
|---------|---------|
| Distributed monolith | Tightly coupled services |
| Chatty services | Too many inter-service calls |
| Shared databases | Tight coupling through data |
| No circuit breakers | Cascade failures |
| Synchronous everything | Poor resilience |
| No compensation logic | Can't undo failed transactions |

## Checklist

- [ ] Services aligned with business capabilities
- [ ] Database per service
- [ ] Event-driven for cross-service updates
- [ ] Circuit breakers on external calls
- [ ] Retry with exponential backoff
- [ ] Saga pattern for distributed transactions
- [ ] Health checks implemented
- [ ] Distributed tracing enabled
