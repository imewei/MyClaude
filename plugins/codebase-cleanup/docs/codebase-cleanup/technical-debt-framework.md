# Technical Debt Framework

**Version**: 1.0.3
**Category**: codebase-cleanup
**Purpose**: Systematic approach to identifying, measuring, prioritizing, and reducing technical debt

## Debt Classification System

### Types of Technical Debt

**Code Debt**:
- Poor code quality, code smells
- High complexity, low maintainability
- Lack of tests or documentation
- Inconsistent coding standards

**Architecture Debt**:
- Monolithic architecture that should be microservices
- Tight coupling between modules
- Missing abstraction layers
- Violated architectural principles

**Infrastructure Debt**:
- Outdated dependencies
- Unsupported platforms or frameworks
- Missing CI/CD automation
- Poor deployment processes

**Documentation Debt**:
- Missing or outdated documentation
- Undocumented APIs
- No architectural diagrams
- Lack of onboarding guides

**Test Debt**:
- Low test coverage
- Slow or flaky tests
- Missing integration/e2e tests
- No test automation

## Debt Scoring Algorithm

### Debt Score Calculation

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class DebtSeverity(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1

class DebtImpact(Enum):
    BLOCKS_DEVELOPMENT = 5
    SLOWS_DEVELOPMENT = 4
    OCCASIONAL_FRICTION = 3
    MINOR_INCONVENIENCE = 2
    NEGLIGIBLE = 1

@dataclass
class TechnicalDebt:
    id: str
    title: str
    description: str
    severity: DebtSeverity
    impact: DebtImpact
    affected_modules: List[str]
    estimated_hours: int
    created_date: str
    interest_rate: float  # How much slower development gets per month

    def calculate_score(self, months_old: int = 0) -> float:
        """
        Calculate debt priority score

        Formula:
        Score = Severity × Impact × (1 + Interest × Months) × ModuleCount
        """
        base_score = self.severity.value * self.impact.value

        # Debt accumulates interest over time
        time_multiplier = 1 + (self.interest_rate * months_old)

        # More affected modules = higher priority
        module_multiplier = len(self.affected_modules)

        return base_score * time_multiplier * module_multiplier

    def calculate_roi(self, team_size: int = 1) -> float:
        """
        Calculate ROI of fixing this debt

        ROI = (Impact × Affected Modules × Team Size) / Estimated Hours
        """
        benefit = self.impact.value * len(self.affected_modules) * team_size
        cost = self.estimated_hours

        return benefit / cost if cost > 0 else 0

    def get_priority_tier(self) -> str:
        """Determine priority tier based on score"""
        score = self.calculate_score()

        if score >= 60:
            return "P0 - Critical"
        elif score >= 40:
            return "P1 - High"
        elif score >= 20:
            return "P2 - Medium"
        else:
            return "P3 - Low"
```

### Example Debt Items

```python
debt_items = [
    TechnicalDebt(
        id="DEBT-001",
        title="Legacy authentication system using MD5",
        description="Authentication module uses MD5 hashing instead of bcrypt",
        severity=DebtSeverity.CRITICAL,
        impact=DebtImpact.BLOCKS_DEVELOPMENT,
        affected_modules=["auth", "user-service", "api-gateway"],
        estimated_hours=16,
        created_date="2023-06-01",
        interest_rate=0.15  # 15% slower per month
    ),
    TechnicalDebt(
        id="DEBT-002",
        title="Missing test coverage in payment module",
        description="Payment processing has only 45% test coverage",
        severity=DebtSeverity.HIGH,
        impact=DebtImpact.SLOWS_DEVELOPMENT,
        affected_modules=["payments"],
        estimated_hours=24,
        created_date="2023-09-15",
        interest_rate=0.10
    ),
    TechnicalDebt(
        id="DEBT-003",
        title="Inconsistent error handling patterns",
        description="Different modules use different error handling approaches",
        severity=DebtSeverity.MEDIUM,
        impact=DebtImpact.OCCASIONAL_FRICTION,
        affected_modules=["api", "services", "workers"],
        estimated_hours=12,
        created_date="2023-11-01",
        interest_rate=0.05
    )
]

# Calculate priorities
from datetime import datetime

for debt in debt_items:
    created = datetime.fromisoformat(debt.created_date)
    months_old = (datetime.now() - created).days // 30

    score = debt.calculate_score(months_old)
    roi = debt.calculate_roi(team_size=5)
    tier = debt.get_priority_tier()

    print(f"{debt.id}: {debt.title}")
    print(f"  Score: {score:.2f}")
    print(f"  ROI: {roi:.2f}")
    print(f"  Priority: {tier}")
    print()
```

## Debt Inventory Management

### Debt Registry Schema

```typescript
interface DebtRegistry {
    version: string;
    lastUpdated: string;
    totalDebtHours: number;
    debtByCategory: {
        code: DebtItem[];
        architecture: DebtItem[];
        infrastructure: DebtItem[];
        documentation: DebtItem[];
        test: DebtItem[];
    };
    metrics: DebtMetrics;
}

interface DebtItem {
    id: string;
    title: string;
    description: string;
    category: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    impact: 1 | 2 | 3 | 4 | 5;
    affectedModules: string[];
    estimatedHours: number;
    createdDate: string;
    resolvedDate?: string;
    assignedTo?: string;
    relatedIssues: string[];
    tags: string[];
}

interface DebtMetrics {
    totalItems: number;
    totalHours: number;
    averageAge: number;
    debtByPriority: {
        p0: number;
        p1: number;
        p2: number;
        p3: number;
    };
    monthlyTrend: TrendData[];
}
```

### Automated Debt Detection

```python
class DebtDetector:
    """Automatically detect technical debt in codebase"""

    def __init__(self, project_root: str):
        self.project_root = project_root

    def detect_all(self) -> List[TechnicalDebt]:
        """Run all debt detection checks"""
        detected_debt = []

        detected_debt.extend(self.detect_complexity_debt())
        detected_debt.extend(self.detect_test_coverage_debt())
        detected_debt.extend(self.detect_outdated_dependencies())
        detected_debt.extend(self.detect_code_duplication())
        detected_debt.extend(self.detect_security_vulnerabilities())

        return detected_debt

    def detect_complexity_debt(self) -> List[TechnicalDebt]:
        """Detect high complexity functions"""
        debt_items = []

        for file_path in self.get_source_files():
            complexity_data = analyze_complexity(file_path)

            for func, complexity in complexity_data.items():
                if complexity > 20:
                    debt_items.append(TechnicalDebt(
                        id=f"COMPLEXITY-{hash(file_path + func)}",
                        title=f"High complexity in {func}",
                        description=f"Function has cyclomatic complexity of {complexity}",
                        severity=DebtSeverity.HIGH if complexity > 30 else DebtSeverity.MEDIUM,
                        impact=DebtImpact.SLOWS_DEVELOPMENT,
                        affected_modules=[self.get_module_name(file_path)],
                        estimated_hours=2 + (complexity // 10),
                        created_date=datetime.now().isoformat(),
                        interest_rate=0.08
                    ))

        return debt_items

    def detect_test_coverage_debt(self) -> List[TechnicalDebt]:
        """Detect low test coverage areas"""
        debt_items = []

        coverage_data = get_coverage_report()

        for file_path, coverage_pct in coverage_data.items():
            if coverage_pct < 70:
                debt_items.append(TechnicalDebt(
                    id=f"COVERAGE-{hash(file_path)}",
                    title=f"Low test coverage in {file_path}",
                    description=f"Only {coverage_pct}% test coverage",
                    severity=DebtSeverity.CRITICAL if coverage_pct < 50 else DebtSeverity.HIGH,
                    impact=DebtImpact.BLOCKS_DEVELOPMENT,
                    affected_modules=[self.get_module_name(file_path)],
                    estimated_hours=int((70 - coverage_pct) * 0.5),
                    created_date=datetime.now().isoformat(),
                    interest_rate=0.12
                ))

        return debt_items

    def detect_outdated_dependencies(self) -> List[TechnicalDebt]:
        """Detect outdated or vulnerable dependencies"""
        debt_items = []

        dependencies = get_dependencies()

        for dep in dependencies:
            if dep.has_security_vulnerability():
                debt_items.append(TechnicalDebt(
                    id=f"VULN-{dep.name}",
                    title=f"Security vulnerability in {dep.name}",
                    description=f"CVE: {dep.cve_id}, upgrade to {dep.safe_version}",
                    severity=DebtSeverity.CRITICAL,
                    impact=DebtImpact.BLOCKS_DEVELOPMENT,
                    affected_modules=dep.affected_modules,
                    estimated_hours=4,
                    created_date=datetime.now().isoformat(),
                    interest_rate=0.20
                ))

        return debt_items
```

## Debt Reduction Roadmap

### Sprint Planning Template

```markdown
# Technical Debt Sprint Plan

**Sprint**: Q1 2024 - Week 1-2
**Team Capacity**: 80 hours (2 engineers × 2 weeks × 20 hours/week)
**Debt Allocation**: 20% of sprint (16 hours)

## Selected Debt Items

### P0 - Critical (8 hours allocated)
- [ ] **DEBT-001**: Legacy authentication using MD5 (8 hours)
  - **Impact**: Security risk, blocks new auth features
  - **Approach**: Migrate to bcrypt, add migration script
  - **Success Criteria**: All passwords use bcrypt, tests pass

### P1 - High (8 hours allocated)
- [ ] **DEBT-004**: Database N+1 queries in order endpoint (4 hours)
  - **Impact**: Performance degradation at scale
  - **Approach**: Add eager loading, optimize queries
  - **Success Criteria**: Endpoint response time < 200ms

- [ ] **DEBT-005**: Missing API documentation (4 hours)
  - **Impact**: Slows integration, support burden
  - **Approach**: Add OpenAPI spec, generate docs
  - **Success Criteria**: All endpoints documented

## Deferred to Next Sprint
- DEBT-002: Test coverage (24 hours - too large)
- DEBT-003: Error handling patterns (12 hours - lower priority)

## Risk Mitigation
- Authentication migration has rollback plan
- Deploy to staging first, monitor for 48 hours
- Feature flag for gradual rollout
```

### Quarterly Roadmap

```python
class DebtReductionRoadmap:
    def __init__(self, debt_items: List[TechnicalDebt]):
        self.debt_items = debt_items

    def plan_quarter(self, team_capacity_hours: int, debt_allocation: float = 0.2) -> Dict:
        """
        Plan debt reduction for a quarter

        Args:
            team_capacity_hours: Total team hours for quarter
            debt_allocation: Percentage of time allocated to debt (default 20%)
        """
        available_hours = team_capacity_hours * debt_allocation

        # Sort by priority score
        sorted_debt = sorted(
            self.debt_items,
            key=lambda d: d.calculate_score(),
            reverse=True
        )

        # Pack debt items into sprints
        sprints = []
        current_sprint = []
        current_hours = 0
        sprint_capacity = available_hours / 6  # Assuming 6 sprints per quarter

        for debt in sorted_debt:
            if current_hours + debt.estimated_hours <= sprint_capacity:
                current_sprint.append(debt)
                current_hours += debt.estimated_hours
            else:
                if current_sprint:
                    sprints.append({
                        'items': current_sprint,
                        'total_hours': current_hours
                    })
                current_sprint = [debt]
                current_hours = debt.estimated_hours

        if current_sprint:
            sprints.append({
                'items': current_sprint,
                'total_hours': current_hours
            })

        return {
            'total_capacity': available_hours,
            'total_planned_hours': sum(s['total_hours'] for s in sprints),
            'sprints': sprints,
            'deferred_items': sorted_debt[len([i for s in sprints for i in s['items']]):],
            'completion_percentage': self._calculate_completion_percentage(sprints)
        }

    def _calculate_completion_percentage(self, sprints: List) -> float:
        """Calculate what % of total debt will be resolved"""
        planned_items = [item for sprint in sprints for item in sprint['items']]
        planned_score = sum(item.calculate_score() for item in planned_items)
        total_score = sum(item.calculate_score() for item in self.debt_items)

        return (planned_score / total_score) * 100 if total_score > 0 else 0
```

## Debt Monitoring

### Tracking Metrics

```python
class DebtMetricsTracker:
    """Track debt metrics over time"""

    def __init__(self, registry_path: str):
        self.registry_path = registry_path

    def calculate_current_metrics(self) -> Dict:
        """Calculate current debt metrics"""
        debt_items = self.load_debt_items()

        return {
            'total_items': len(debt_items),
            'total_hours': sum(item.estimated_hours for item in debt_items),
            'average_age_days': self._calculate_average_age(debt_items),
            'by_priority': self._group_by_priority(debt_items),
            'by_category': self._group_by_category(debt_items),
            'monthly_accrual': self._calculate_monthly_accrual(debt_items),
            'burn_down_rate': self._calculate_burn_down_rate()
        }

    def _calculate_average_age(self, debt_items: List[TechnicalDebt]) -> int:
        """Calculate average age of debt items in days"""
        if not debt_items:
            return 0

        total_age = 0
        for item in debt_items:
            created = datetime.fromisoformat(item.created_date)
            age_days = (datetime.now() - created).days
            total_age += age_days

        return total_age // len(debt_items)

    def _calculate_monthly_accrual(self, debt_items: List[TechnicalDebt]) -> float:
        """Calculate how much new debt is added per month"""
        # Get items created in last 3 months
        three_months_ago = datetime.now() - timedelta(days=90)

        recent_items = [
            item for item in debt_items
            if datetime.fromisoformat(item.created_date) >= three_months_ago
        ]

        total_hours = sum(item.estimated_hours for item in recent_items)
        return total_hours / 3  # Average per month

    def _calculate_burn_down_rate(self) -> float:
        """Calculate how much debt is being resolved per month"""
        # Get items resolved in last 3 months
        three_months_ago = datetime.now() - timedelta(days=90)

        resolved_items = [
            item for item in self.load_all_items()
            if item.resolved_date and datetime.fromisoformat(item.resolved_date) >= three_months_ago
        ]

        total_hours = sum(item.estimated_hours for item in resolved_items)
        return total_hours / 3  # Average per month

    def generate_trend_report(self) -> str:
        """Generate debt trend report"""
        metrics = self.calculate_current_metrics()

        monthly_accrual = metrics['monthly_accrual']
        burn_down_rate = metrics['burn_down_rate']
        net_change = burn_down_rate - monthly_accrual

        status = "improving" if net_change > 0 else "worsening"

        return f"""
Technical Debt Trend Report
===========================

Current State:
- Total Debt Items: {metrics['total_items']}
- Total Debt Hours: {metrics['total_hours']}
- Average Age: {metrics['average_age_days']} days

Monthly Rates:
- New Debt Accrual: {monthly_accrual:.1f} hours/month
- Debt Resolution: {burn_down_rate:.1f} hours/month
- Net Change: {net_change:+.1f} hours/month

Trend: Debt is {status} at {abs(net_change):.1f} hours/month

Recommendation:
{'Maintain current pace' if net_change > 5 else 'Increase debt allocation by ' + str(int(abs(net_change) / burn_down_rate * 20)) + '%'}
        """.strip()
```

## Best Practices

### Debt Prevention

1. **Code Review Checklist**:
   - [ ] No high complexity functions introduced
   - [ ] Test coverage maintained or improved
   - [ ] No new code duplication
   - [ ] Dependencies up to date
   - [ ] Documentation updated

2. **Definition of Done**:
   - Feature code complete
   - Tests written and passing
   - Documentation updated
   - Code reviewed and approved
   - No new technical debt introduced

3. **Continuous Monitoring**:
   - Daily automated debt detection
   - Weekly debt review meetings
   - Monthly debt reduction sprints
   - Quarterly debt audits

### Debt Resolution

1. **Start Small**: Pick quick wins (< 4 hours) first
2. **Measure Impact**: Track before/after metrics
3. **Document Learnings**: Capture what worked
4. **Prevent Recurrence**: Add quality gates
5. **Celebrate Progress**: Acknowledge team efforts
