---
description: Workflow for tech-debt
triggers:
- /tech-debt
- workflow for tech debt
allowed-tools: [Bash, Read, Task]
version: 1.0.0
---



# Technical Debt Analysis

$ARGUMENTS

## Modes

| Mode | Time | Scope |
|------|------|-------|
| Quick | 5-10min | Top 10 high-impact, basic prioritization |
| Standard | 15-25min | Full inventory, scoring, roadmap |
| Comprehensive | 30-60min | Deep architecture, quarterly plan, ROI |

## Debt Categories

| Category | Items |
|----------|-------|
| Code | Duplication, complexity >10, circular deps, dead code |
| Architecture | Missing abstractions, violated boundaries, monoliths |
| Testing | Coverage <70%, flaky tests, missing integration |
| Documentation | Missing API docs, undocumented logic, no diagrams |
| Infrastructure | Manual deployments, no monitoring, no rollback |

## Scoring

```
Score = Severity × Impact × (1 + Interest × Months) × ModuleCount
ROI = (Impact × Affected × Team) / Hours
```

| Tier | Score | Action |
|------|-------|--------|
| P0 | >60 | Immediate |
| P1 | 40-60 | Next sprint |
| P2 | 20-40 | Next quarter |
| P3 | <20 | Backlog |

## Automated Detection

| Check | Command |
|-------|---------|
| Complexity | `radon cc src/ -nb --min B` |
| Coverage | `pytest --cov=src` |
| Duplication | `jscpd src/ --min-lines 6` |
| Outdated | `npm audit` / `pip list --outdated` |
| Security | `bandit -r src/` / `safety check` |

## Metrics

| Metric | Current | Target | Trend |
|--------|---------|--------|-------|
| Complexity | avg | <10 | |
| Duplication | % | <5% | |
| Coverage | %/% | 80%/60% | |
| Outdated Deps | count | 0 | |
| Vulns | count | 0 | |

**Trend**: Monthly accrual X h/mo, burn-down Y h/mo, net (Y-X), target >+10 h/mo

## Roadmap

### Quick Wins (High ROI, Low Effort)
| Item | Effort | Monthly Savings | ROI |
|------|--------|-----------------|-----|
| Extract duplicate validation | 8h | 20h | 250% |
| Add error monitoring | 4h | 15h | 375% |
| Automate deployment | 12h | 40h | 333% |

### Medium-Term (1-3mo)
| Item | Effort | Benefits |
|------|--------|----------|
| Refactor God class | 60h | 30h/mo savings |
| Upgrade framework | 80h | +30% perf |

### Long-Term (Q2-Q4)
| Item | Effort | Benefits |
|------|--------|----------|
| Proper architecture | 200h | 50% coupling ↓ |
| Comprehensive tests | 300h | 70% bugs ↓ |

## Strategy

### Strangler Fig
1. Add facade over legacy
2. Implement new service alongside
3. Gradual migration with flag
4. Remove legacy when done

### Team Allocation
- 15-20% sprint capacity
- Tech lead: Architecture
- Senior: Complex refactoring
- Dev: Tests, docs

## Prevention

### Quality Gates

| Stage | Checks |
|-------|--------|
| Pre-commit | Complexity <10, duplication <5%, coverage ≥80% new |
| CI | No critical vulns, no perf regression >10% |
| Review | 2 approvals, tests, docs updated |

### Debt Budget
- Monthly increase: <2%
- Quarterly reduction: 5%
- Track: SonarQube, Dependabot, CodeCov

## Output

1. **Summary**: Total debt h/items, priority dist, top 5 with ROI
2. **Inventory**: Categorized list with scores, impact, effort
3. **Roadmap**: Quarter plan, sprint breakdown
4. **Quick Wins**: Immediate high-ROI
5. **Prevention**: Gates, monitoring

## Metrics

| Metric | Target |
|--------|--------|
| Debt score | -5%/mo |
| Bug rate | -20% |
| Deploy frequency | +50% |
| Lead time | -30% |
| Coverage | +10% |

## Best Practices

1. Quantify (hours, dollars)
2. Prioritize (high-ROI)
3. Track (accrual vs burn-down)
4. Allocate (15-20% capacity)
5. Prevent (quality gates)
6. Celebrate (progress)
