---
name: message-queue-patterns
description: Implement message queue architectures with Kafka, RabbitMQ, and SQS including event-driven design, message ordering, dead letter queues, consumer groups, and exactly-once processing. Use when designing async communication, event sourcing, or pub/sub systems.
---

# Message Queue Patterns

## Expert Agent

For event-driven architecture, message queue design, and async communication patterns, delegate to:

- **`software-architect`**: Designs event-driven systems with service boundaries and integration patterns.
  - *Location*: `plugins/dev-suite/agents/software-architect.md`


## Queue vs Topic Comparison

| Feature | Queue (Point-to-Point) | Topic (Pub/Sub) |
|---------|----------------------|-----------------|
| Consumers | Single consumer per message | Multiple subscribers |
| Use case | Task distribution | Event broadcasting |
| Ordering | Per-queue FIFO possible | Per-partition (Kafka) |
| Example | SQS, RabbitMQ queue | Kafka topic, SNS, RabbitMQ exchange |


## Technology Selection

| Requirement | Kafka | RabbitMQ | SQS |
|-------------|-------|----------|-----|
| Throughput | Millions/sec | Thousands/sec | Thousands/sec |
| Ordering | Per-partition | Per-queue | FIFO queues |
| Replay | Yes (retention) | No | No |
| Managed option | MSK, Confluent | CloudAMQP | Native AWS |
| Complexity | High | Medium | Low |


## Kafka Patterns

### Producer

```python
from confluent_kafka import Producer

config = {
    "bootstrap.servers": "localhost:9092",
    "acks": "all",
    "retries": 3,
    "enable.idempotence": True,
}

producer = Producer(config)

def publish_event(topic: str, key: str, value: bytes) -> None:
    producer.produce(topic, key=key.encode(), value=value, callback=delivery_report)
    producer.flush()

def delivery_report(err, msg):
    if err:
        logging.error("Delivery failed: %s", err)
    else:
        logging.info("Delivered to %s [%d]", msg.topic(), msg.partition())
```

### Consumer Group

```python
from confluent_kafka import Consumer

config = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "order-processing",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
}

consumer = Consumer(config)
consumer.subscribe(["orders"])

try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            logging.error("Consumer error: %s", msg.error())
            continue
        process_message(msg.value())
        consumer.commit(message=msg)
finally:
    consumer.close()
```


## Dead Letter Queue (DLQ)

### Pattern

```
Main Queue --> Consumer --> Success
                |
                v (after N retries)
             DLQ --> Alert + Manual Review
```

### SQS DLQ Configuration

```json
{
  "RedrivePolicy": {
    "deadLetterTargetArn": "arn:aws:sqs:us-east-1:123456789:orders-dlq",
    "maxReceiveCount": 3
  }
}
```

### DLQ Handling Rules

- [ ] Set max retry count (typically 3-5)
- [ ] Log full message payload and error on DLQ entry
- [ ] Alert on DLQ depth exceeding threshold
- [ ] Build tooling to inspect and replay DLQ messages
- [ ] Never silently discard failed messages


## Idempotency

### Idempotent Consumer Pattern

```python
import hashlib

def process_message(msg: dict, db_session) -> None:
    idempotency_key = msg.get("idempotency_key") or compute_key(msg)

    existing = db_session.query(ProcessedMessage).filter_by(key=idempotency_key).first()
    if existing:
        logging.info("Duplicate message, skipping: %s", idempotency_key)
        return

    execute_business_logic(msg)

    db_session.add(ProcessedMessage(key=idempotency_key))
    db_session.commit()

def compute_key(msg: dict) -> str:
    return hashlib.sha256(json.dumps(msg, sort_keys=True).encode()).hexdigest()
```


## Event Schema Design

### Schema Evolution Rules

- Add fields with defaults (backward-compatible)
- Never remove or rename required fields
- Use a schema registry (Confluent, AWS Glue) for validation
- Version schemas explicitly in the event type


## Backpressure Strategies

| Strategy | When to Use |
|----------|------------|
| Rate limiting | Protect downstream services |
| Buffering | Absorb temporary spikes |
| Dropping | Non-critical telemetry data |
| Scaling consumers | Sustained high load |
| Circuit breaker | Downstream failures |


## Design Checklist

- [ ] Queue vs topic chosen based on consumer pattern
- [ ] Idempotency enforced at consumer level
- [ ] DLQ configured with alerting
- [ ] Message schema versioned and validated
- [ ] Consumer groups sized for throughput needs
- [ ] Monitoring on queue depth, consumer lag, error rate
- [ ] Backpressure strategy defined for overload scenarios
- [ ] Ordering guarantees match business requirements
