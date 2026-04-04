---
name: modernization-migration
description: Reference patterns for modernization strategies including Strangler Fig, framework migration playbooks, and database schema evolution. Provides domain knowledge for the /modernize command. Use when planning migration approaches, evaluating modernization trade-offs, or implementing Strangler Fig patterns.
---

# Modernization & Migration

## Expert Agent

For legacy modernization strategy, framework migration, and incremental system evolution, delegate to:

- **`software-architect`**: Plans Strangler Fig migrations, architecture refactoring, and technology modernization.
  - *Location*: `plugins/dev-suite/agents/software-architect.md`

Expert guide for safely evolving legacy systems and adopting modern technologies.

## 1. Strategy & Patterns

- **Strangler Fig**: Gradually replace legacy functionality with new services behind an API gateway.
- **Anti-Corruption Layer**: Build a translation layer between old and new systems to prevent legacy patterns from leaking.

## 2. Migration Execution

- **Frameworks**: Patterns for migrating from Angular to React, or monolith to microservices.
- **Databases**: Use blue-green deployments or expand-contract patterns for zero-downtime schema changes.

## 3. Strangler Fig Implementation

```
                    ┌─────────────┐
   Requests ───────►│  API Gateway │
                    │  / Proxy     │
                    └──┬──────┬───┘
                       │      │
              ┌────────▼──┐ ┌─▼────────┐
              │  Legacy    │ │  New      │
              │  System    │ │  Service  │
              │ (shrinking)│ │ (growing) │
              └────────────┘ └──────────┘
```

| Phase | Action | Risk Level |
|-------|--------|------------|
| 1. Identify | Map legacy features to bounded contexts | Low |
| 2. Intercept | Route traffic through proxy/gateway | Low |
| 3. Implement | Build new service for one feature | Medium |
| 4. Redirect | Switch traffic for that feature to new service | Medium |
| 5. Retire | Remove legacy code for migrated feature | Low |
| 6. Repeat | Continue until legacy system is fully replaced | Cumulative |

## 4. Migration Strategies

### Incremental (Recommended)
Migrate one module at a time behind feature flags. Run old and new in parallel with traffic splitting. Compare outputs before full cutover.

### Parallel Run
Execute both old and new systems simultaneously. Compare results to verify correctness. Use for critical financial or data-integrity paths.

```python
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MigrationResult:
    legacy_result: dict
    new_result: dict
    matches: bool

def parallel_run(request):
    legacy = legacy_system.process(request)
    new = new_system.process(request)
    result = MigrationResult(
        legacy_result=legacy,
        new_result=new,
        matches=(legacy == new),
    )
    if not result.matches:
        logger.warning("Migration mismatch", extra={"diff": diff(legacy, new)})
    return legacy  # Return legacy until confidence threshold met
```

## 5. Compatibility Layers

- **API Adapter**: Translate legacy API contracts to new service interfaces.
- **Database View**: Create views that map legacy schema to new schema during transition.
- **Event Bridge**: Emit events from both systems; consumers read from whichever is authoritative.
- **Feature Flags**: Gate new code paths per-tenant or per-region for gradual rollout.

## 6. Rollback Patterns

| Strategy | Recovery Time | Data Risk |
|----------|---------------|-----------|
| Feature flag toggle | Seconds | None |
| Blue-green DNS switch | Minutes | None if stateless |
| Database expand-contract | Minutes | Low (backward-compatible schema) |
| Full revert deploy | 10-30 minutes | Medium (check data written during migration) |

## 7. Testing Migration

- **Contract tests**: Verify old and new APIs produce identical responses for same inputs.
- **Shadow traffic**: Replay production traffic against new service without serving responses.
- **Data validation**: Checksum migrated records against source; sample audit for semantic correctness.
- **Load testing**: Benchmark new service against legacy performance baselines before cutover.

## 8. Migration Checklist

- [ ] **Rollback**: Verified plan to revert the migration at any stage
- [ ] **Validation**: Automated tests verify feature parity between old and new
- [ ] **Performance**: New system load-tested against legacy benchmarks
- [ ] **Data Integrity**: Checksums or audits ensure data consistency during transfer
- [ ] **Compatibility layer**: Adapter translates between old and new interfaces
- [ ] **Feature flags**: Migration gated for gradual rollout per-tenant or per-region
- [ ] **Parallel run**: Critical paths compared between old and new before cutover
- [ ] **Monitoring**: Alerts configured for error rate spikes during migration
- [ ] **Documentation**: Migration runbook covers each phase with rollback steps
