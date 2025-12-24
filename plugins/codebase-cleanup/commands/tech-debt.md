---
version: "1.0.5"
category: codebase-cleanup
purpose: Analyze, prioritize, and create remediation plans for technical debt
execution_time:
  quick: 5-10 minutes
  standard: 15-25 minutes
  comprehensive: 30-60 minutes
external_docs:
  - technical-debt-framework.md
  - code-quality-metrics.md
  - refactoring-patterns.md
  - automation-integration.md
---

# Technical Debt Analysis and Remediation

Identify, quantify, prioritize, and systematically reduce technical debt with measurable ROI.

## Target

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| `--quick` | 5-10 min | Top 10 high-impact items, basic prioritization |
| standard (default) | 15-25 min | Full inventory, multi-dimensional scoring, roadmap |
| `--comprehensive` | 30-60 min | Deep architectural analysis, quarterly plan, ROI calculations |

---

## Phase 1: Debt Inventory

### Debt Categories

| Category | Items to Scan |
|----------|---------------|
| **Code Debt** | Duplicated code, high complexity (>10), circular dependencies, dead code |
| **Architecture Debt** | Missing abstractions, violated boundaries, monoliths, outdated tech |
| **Testing Debt** | Coverage gaps (<70%), flaky tests, missing integration tests |
| **Documentation Debt** | Missing API docs, undocumented logic, no architecture diagrams |
| **Infrastructure Debt** | Manual deployments, missing monitoring, no rollback procedures |

---

## Phase 2: Scoring and Prioritization

### Scoring Formula

```
Score = Severity × Impact × (1 + Interest × Months) × ModuleCount
ROI = (Impact × Affected_Modules × Team_Size) / Estimated_Hours
```

### Priority Tiers

| Tier | Score | Action |
|------|-------|--------|
| P0 | >60 | Immediate action |
| P1 | 40-60 | Next sprint |
| P2 | 20-40 | Next quarter |
| P3 | <20 | Backlog |

### Impact Levels

| Level | Description |
|-------|-------------|
| Critical | Security vulnerabilities, data loss risk |
| High | Performance degradation, frequent outages |
| Medium | Developer frustration, slow delivery |
| Low | Code style, minor inefficiencies |

---

## Phase 3: Automated Detection (Comprehensive)

| Check | Command |
|-------|---------|
| Complexity | `radon cc src/ -nb --min B` |
| Coverage | `pytest --cov=src --cov-report=term` |
| Duplication | `jscpd src/ --min-lines 6` |
| Outdated deps | `npm audit` / `pip list --outdated` |
| Security | `bandit -r src/` / `safety check` |

---

## Phase 4: Metrics Dashboard

| Metric | Current | Target | Trend |
|--------|---------|--------|-------|
| Cyclomatic Complexity | avg | <10 | |
| Code Duplication | % | <5% | |
| Test Coverage (unit/integ) | %/% | 80%/60% | |
| Outdated Major Deps | count | 0 | |
| Security Vulnerabilities | count | 0 | |

**Trend Analysis:**
- Monthly accrual rate: X hours/month
- Monthly burn-down rate: Y hours/month
- Net change: (Y-X) hours/month
- Target: >+10 hours/month

---

## Phase 5: Remediation Roadmap

### Quick Wins (High ROI, Low Effort)

| Item | Effort | Monthly Savings | ROI |
|------|--------|-----------------|-----|
| Extract duplicate validation | 8h | 20h | 250% |
| Add error monitoring | 4h | 15h | 375% |
| Automate deployment | 12h | 40h | 333% |

### Medium-Term (Month 1-3)

| Item | Effort | Benefits |
|------|--------|----------|
| Refactor God class | 60h | 30h/month savings |
| Upgrade legacy framework | 80h | +30% performance |

### Long-Term (Quarter 2-4)

| Item | Effort | Benefits |
|------|--------|----------|
| Proper architecture | 200h | 50% coupling reduction |
| Comprehensive tests | 300h | 70% bug reduction |

---

## Phase 6: Implementation Strategy

### Incremental Refactoring (Strangler Fig)
1. Add facade over legacy code
2. Implement new service alongside
3. Gradual migration with feature flag
4. Remove legacy when fully migrated

### Team Allocation
- Dedicated: 15-20% of sprint capacity
- Tech lead: Architecture decisions
- Senior dev: Complex refactoring
- Dev: Testing and documentation

---

## Phase 7: Prevention Strategy

### Quality Gates

| Stage | Checks |
|-------|--------|
| Pre-commit | Complexity <10, duplication <5%, coverage ≥80% new code |
| CI Pipeline | No critical vulnerabilities, no perf regression >10% |
| Code Review | Two approvals, tests included, docs updated |

### Debt Budget
- Monthly increase limit: <2%
- Quarterly reduction target: 5%
- Tracking: SonarQube, Dependabot, CodeCov

---

## Output

### Report Structure

1. **Executive Summary**
   - Total debt hours/items
   - Priority distribution (P0/P1/P2/P3)
   - Top 5 critical items with ROI

2. **Debt Inventory**
   - Categorized list with scores
   - Impact and effort estimates

3. **Reduction Roadmap**
   - Quarter-by-quarter plan
   - Sprint breakdown

4. **Quick Wins**
   - Immediate high-ROI items

5. **Prevention Plan**
   - Quality gates
   - Monitoring setup

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Debt score reduction | -5%/month |
| Bug rate | -20% |
| Deployment frequency | +50% |
| Lead time | -30% |
| Test coverage | +10% |

---

## Best Practices

1. **Quantify**: Measure debt in hours and dollars
2. **Prioritize**: Focus on high-ROI items
3. **Track**: Monitor accrual vs burn-down
4. **Allocate**: 15-20% capacity consistently
5. **Prevent**: Enforce quality gates
6. **Celebrate**: Acknowledge progress

**Reference:** See external docs for detailed algorithms and templates.
