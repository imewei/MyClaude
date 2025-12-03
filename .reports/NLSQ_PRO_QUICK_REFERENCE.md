# NLSQ-Pro Quick Reference Guide

## 5 Optimized Agents - Quick Overview

### 1️⃣ architect-review (v1.1.0, 88%)
**When to Use**: System architecture, design patterns, microservices, scalability analysis
**Key Target**: 92% pattern compliance, 10x growth runway, 99.9% SLA feasible
**Core Question**: "Does this enable business growth without compromising security/scalability?"
**Avoid**: Implementation details, infrastructure provisioning, code style issues
**File**: `plugins/framework-migration/agents/architect-review.md`

---

### 2️⃣ legacy-modernizer (v1.1.0, 83%)
**When to Use**: Framework migrations, technical debt reduction, backward compatibility
**Key Target**: 100% backward compatibility, 80%+ test coverage, 2-week value delivery
**Core Question**: "Can we modernize safely without breaking anything?"
**Avoid**: New feature development, performance optimization (standalone), architecture redesign (solo)
**File**: `plugins/framework-migration/agents/legacy-modernizer.md`

---

### 3️⃣ code-reviewer (v1.2.0, 89%)
**When to Use**: Pull request review, security vulnerabilities, performance analysis, reliability
**Key Target**: 0 critical vulnerabilities, <5% latency increase, ≥80% test coverage
**Core Question**: "Would I merge this knowing it serves prod in 5 minutes?"
**Avoid**: Writing code, fixing bugs directly, penetration testing, architecture design
**File**: `plugins/git-pr-workflows/agents/code-reviewer.md`

---

### 4️⃣ hpc-numerical-coordinator (v1.1.0, 87%)
**When to Use**: Numerical algorithms, HPC workflows, GPU acceleration, Julia/SciML vs Python
**Key Target**: 98% numerical accuracy, >80% performance efficiency, 100% reproducibility
**Core Question**: "Is this numerically correct, verifiable, and reproducible?"
**Avoid**: Molecular dynamics, statistical physics, JAX-specific optimization, domain-specific physics
**File**: `plugins/hpc-computing/agents/hpc-numerical-coordinator.md`

---

### 5️⃣ data-engineer (v1.1.0, 86%)
**When to Use**: Data pipelines, ETL/ELT, data quality, storage optimization, cost efficiency
**Key Target**: 99%+ quality pass rate, 100% idempotency, ±20% cost variance
**Core Question**: "Would I trust this data for critical business decisions?"
**Avoid**: ML model development, feature engineering logic, analytics/BI, data warehouse design
**File**: `plugins/machine-learning/agents/data-engineer.md`

---

## Key Improvements

| Metric | architect-review | legacy-modernizer | code-reviewer | hpc | data-engineer |
|--------|----------|----------|-----------|-----|------|
| Version | 1.0.3→1.1 | 1.0.3→1.1 | 1.1.1→1.2 | 1.0.1→1.1 | 1.0.3→1.1 |
| Maturity | +13% | +13% | +5% | +5% | +16% |

**Average Improvement**: +10.4% per agent

---

## Summary

All 5 agents now have:
- ✅ Version bumps (1.0.x → 1.1.x/1.2.x)
- ✅ Pre-Response Validation (5 checks + 5 gates each)
- ✅ When to Invoke clarity (USE/DO NOT USE + Decision Tree)
- ✅ Enhanced Constitutional AI (core question + 4 principles with metrics)
- ✅ 4 anti-patterns per principle to prevent
- ✅ 3 success metrics per principle to measure

**Status**: Optimization complete with nlsq-pro template pattern.
