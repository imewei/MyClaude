# Strangler Fig Pattern Playbook

**Version:** 1.0.3 | **Category:** framework-migration | **Type:** Implementation Guide

Detailed implementation guide for the Strangler Fig pattern - the gold standard for incremental legacy system modernization.

---

## Pattern Overview

**Named After**: The strangler fig tree that gradually grows around and replaces its host tree.

**Principle**: Incrementally replace legacy system components with modern implementations while both systems run in parallel.

**Key Benefits**:
- Zero-downtime migration
- Incremental risk mitigation
- Continuous business operations
- Instant rollback capability

---

## Implementation Phases

### Phase 1: Routing Layer Setup

**Purpose**: Create traffic routing mechanism to direct requests to legacy or modern system.

**Options**:
1. **API Gateway** (Kong, AWS API Gateway, Azure API Management)
2. **Load Balancer** (NGINX, HAProxy)
3. **Service Mesh** (Istio, Linkerd)
4. **Application-Level Routing** (feature flags, middleware)

**Example - NGINX Routing**:
```nginx
upstream legacy_backend {
    server legacy-app:3000;
}

upstream modern_backend {
    server modern-app:3000;
}

server {
    location /api/users {
        # Route to modern implementation
        proxy_pass http://modern_backend;
    }

    location / {
        # Route to legacy by default
        proxy_pass http://legacy_backend;
    }
}
```

### Phase 2: Feature Flag System

**Purpose**: Control rollout percentage and enable instant rollback.

**Implementation**:
```javascript
// Feature flag service
const featureFlags = {
  'modern-checkout-flow': {
    enabled: true,
    rolloutPercentage: 10,  // 10% of users
    userSegments: ['beta-testers']
  }
};

// Usage in application
function getCheckoutComponent() {
  const useModern = featureFlags.isEnabled('modern-checkout-flow', user.id);
  return useModern ? <ModernCheckout /> : <LegacyCheckout />;
}
```

### Phase 3: Dual-Write Pattern

**Purpose**: Maintain data consistency between legacy and modern databases during transition.

**Example**:
```python
def create_order(order_data):
    # Write to legacy database
    legacy_order = legacy_db.orders.create(order_data)

    # Also write to modern database (dual-write)
    try:
        modern_order = modern_db.orders.create(
            transform_to_modern_schema(order_data)
        )
    except Exception as e:
        logger.error(f"Dual-write failed: {e}")
        # Don't fail request - legacy is source of truth

    return legacy_order
```

### Phase 4: Data Migration

**Purpose**: Backfill historical data from legacy to modern system.

**Strategy**:
```python
# Batch migration script
def migrate_historical_data(batch_size=1000):
    offset = 0
    while True:
        # Read batch from legacy
        legacy_records = legacy_db.query(
            "SELECT * FROM orders LIMIT %s OFFSET %s",
            (batch_size, offset)
        )

        if not legacy_records:
            break

        # Transform and write to modern
        modern_records = [
            transform_to_modern_schema(record)
            for record in legacy_records
        ]
        modern_db.bulk_insert(modern_records)

        offset += batch_size
        logger.info(f"Migrated {offset} records")
```

### Phase 5: Read Migration

**Purpose**: Shift read operations to modern system.

**Gradual Approach**:
1. Start with 10% read traffic to modern
2. Compare responses (modern vs legacy)
3. Log discrepancies for investigation
4. Gradually increase percentage
5. 100% reads from modern when validated

### Phase 6: Write Migration

**Purpose**: Shift write operations to modern system.

**Dual-Write to Single-Write**:
```python
# Phase 6a: Modern becomes primary
def create_order(order_data):
    # Write to modern first
    modern_order = modern_db.orders.create(order_data)

    # Shadow write to legacy for safety
    try:
        legacy_db.orders.create(
            transform_to_legacy_schema(order_data)
        )
    except Exception as e:
        logger.warning(f"Legacy shadow write failed: {e}")

    return modern_order
```

### Phase 7: Legacy Decommissioning

**Purpose**: Remove legacy system after validation.

**Checklist**:
- [ ] 30+ days with 0% traffic to legacy
- [ ] All features migrated
- [ ] Data validation complete
- [ ] Monitoring confirms modern system stable
- [ ] Rollback plan documented (just in case)

---

## Monitoring & Observability

### Key Metrics

**Traffic Distribution**:
- Legacy requests/sec
- Modern requests/sec
- Routing decision latency

**Correctness**:
- Response comparison (legacy vs modern)
- Data consistency checks
- Error rate by system

**Performance**:
- Latency (p50, p95, p99) per system
- Throughput
- Resource utilization

### Dashboard Example

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strangler Fig Migration Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Traffic Split:
  Legacy: ████░░░░░░ 40%
  Modern: ██████████ 60%

Error Rates:
  Legacy: 0.5%
  Modern: 0.3% ✅

Latency (p95):
  Legacy: 850ms
  Modern: 320ms ✅ (-62%)

Features Migrated: 12/18 (67%)
  ✅ User authentication
  ✅ Product catalog
  ✅ Shopping cart
  ⏳ Checkout (10% rollout)
  ⏳ Order history
  ❌ Admin dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Rollback Strategies

### Instant Rollback (Feature Flag)

```javascript
// Emergency: disable modern implementation
featureFlags.set('modern-checkout-flow', {
  enabled: false,
  rolloutPercentage: 0
});

// Takes effect in <1 second
```

### Gradual Rollback (Traffic Shift)

```nginx
# Reduce traffic to modern system
location /api/checkout {
    # 90% to legacy, 10% to modern
    split_clients "${remote_addr}" $backend {
        10%   modern;
        *     legacy;
    }

    proxy_pass http://$backend_backend;
}
```

---

**For complete legacy modernization workflow**, see `/legacy-modernize` command.
