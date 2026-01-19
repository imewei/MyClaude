---
name: modernization-migration
version: "1.0.0"
description: Strategy and patterns for legacy modernization, framework migrations, and database schema evolution.
---

# Modernization & Migration

Expert guide for safely evolving legacy systems and adopting modern technologies.

## 1. Strategy & Patterns

- **Strangler Fig**: Gradually replace legacy functionality with new services behind an API gateway.
- **Anti-Corruption Layer**: Build a translation layer between old and new systems to prevent legacy patterns from leaking.

## 2. Migration Execution

- **Frameworks**: Patterns for migrating from Angular to React, or monolith to microservices.
- **Databases**: Use blue-green deployments or expand-contract patterns for zero-downtime schema changes.

## 3. Migration Checklist

- [ ] **Rollback**: Is there a verified plan to revert the migration at any stage?
- [ ] **Validation**: Are there automated tests to verify feature parity between old and new?
- [ ] **Performance**: Has the new system been load-tested against legacy benchmarks?
- [ ] **Data Integrity**: Are there checksums or audits to ensure data remains consistent during transfer?
