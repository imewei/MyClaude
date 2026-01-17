---
description: Workflow for refactor-clean
triggers:
- /refactor-clean
- workflow for refactor clean
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



# Refactor and Clean Code

Analyze and refactor code to improve quality, maintainability, and performance.

## Target

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| `--quick` | 5-10 min | Immediate fixes (rename, constants, dead code) |
| standard (default) | 15-30 min | Full code smell analysis, SOLID violations, method extraction |
| `--comprehensive` | 30-90 min | Deep architectural analysis, design patterns, metrics |

---

## Phase 1: Code Analysis

### Code Smells

| Smell | Threshold |
|-------|-----------|
| Long methods | >20 lines |
| Large classes | >200 lines |
| Duplicate code | Any significant blocks |
| Complex conditionals | Nested >3 levels |
| Magic numbers | Hardcoded values |
| Poor naming | Non-descriptive identifiers |

### SOLID Violations

| Principle | Violation Indicator |
|-----------|---------------------|
| **SRP** | Class/function with multiple responsibilities |
| **OCP** | Modification required to add features |
| **LSP** | Subclasses not substitutable for base |
| **ISP** | Clients depend on unused interfaces |
| **DIP** | High-level modules depend on low-level |

---

## Phase 2: Refactoring Strategy

### Quick Fixes (Quick Mode)

| Fix | Action |
|-----|--------|
| Magic numbers | Extract to named constants |
| Poor naming | Rename to descriptive names |
| Dead code | Delete unused imports, variables, functions |
| Boolean expressions | Simplify `== True` → direct check |

### Method Extraction (Standard Mode)
- Break 50+ line methods into focused methods
- Each method should do one thing
- Clear, descriptive method names

### Class Decomposition (Comprehensive Mode)
- Extract responsibilities to separate classes
- Create interfaces for dependencies
- Use dependency injection
- Favor composition over inheritance

---

## Phase 3: Apply SOLID Principles

| Principle | Refactoring Action |
|-----------|-------------------|
| **SRP** | Extract mixed responsibilities → separate classes (e.g., UserManager → UserValidator + UserRepository + EmailService) |
| **OCP** | Replace conditionals with polymorphism, use Strategy pattern |
| **LSP** | Use interfaces over deep inheritance hierarchies |
| **ISP** | Split fat interfaces into focused ones |
| **DIP** | Inject abstractions, not concretions |

---

## Phase 4: Design Patterns (Comprehensive)

| Pattern | Use Case |
|---------|----------|
| Factory | Complex object creation |
| Strategy | Algorithm variants |
| Observer | Event handling |
| Repository | Data access |
| Decorator | Extending behavior |

---

## Phase 5: Verification

### Safety Checklist

**Before:**
- [ ] All tests pass
- [ ] Git clean or changes committed
- [ ] Code behavior understood

**During:**
- [ ] One refactoring at a time
- [ ] Tests after each change
- [ ] Commit after each success

**After:**
- [ ] Full test suite passes
- [ ] No performance regression
- [ ] Documentation updated

```bash
npm test              # Full tests
npm run type-check    # Type safety
npm run lint          # Linting
npm run build         # Build verification
```

---

## Metrics (Comprehensive Mode)

| Metric | Poor | Target | Excellent |
|--------|------|--------|-----------|
| Cyclomatic Complexity | >20 | <10 | <5 |
| Code Duplication | >10% | <5% | <2% |
| Test Coverage | <60% | >80% | >90% |
| Maintainability Index | <40 | >60 | >80 |

---

## Common Scenarios

| Scenario | Strategy |
|----------|----------|
| **Legacy Monolith (500+ lines)** | Identify responsibilities → Extract classes → Create interfaces → Apply DI |
| **Spaghetti Code (nested if/else)** | Guard clauses → Extract conditions → Replace with polymorphism |
| **God Object** | Apply SRP → Extract Repository, Service, Validator classes |

---

## When NOT to Refactor

- Code about to be deleted/replaced
- Under tight deadline (defer to backlog)
- No test coverage and can't add tests
- Performance-critical hot path (profile first)

---

## Output

1. **Analysis Summary** - Smells, violations, metrics
2. **Refactoring Plan** - Prioritized changes with effort/risk
3. **Refactored Code** - Complete working implementation
4. **Verification Steps** - Commands to validate

**Reference:** See external docs for detailed examples and patterns.
