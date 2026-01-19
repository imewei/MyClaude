# Error Budget Examples

## Burn Rate Example Scenarios

```python
def demonstrate_burn_rate_scenarios():
    """Examples of burn rate calculations in different scenarios."""

    scenarios = [
        {
            'name': 'Normal Operation',
            'slo_target': 99.9,
            'actual_success_rate': 99.95,
            'actual_error_rate': 0.05,
            'expected_analysis': 'Burn rate = 0.5 (consuming half the budget, very healthy)'
        },
        {
            'name': 'On-Track',
            'slo_target': 99.9,
            'actual_success_rate': 99.9,
            'actual_error_rate': 0.1,
            'expected_analysis': 'Burn rate = 1.0 (consuming exactly as budgeted)'
        },
        {
            'name': 'Moderate Issue',
            'slo_target': 99.9,
            'actual_success_rate': 99.8,
            'actual_error_rate': 0.2,
            'expected_analysis': 'Burn rate = 2.0 (consuming budget 2x faster)'
        },
        {
            'name': 'Severe Incident',
            'slo_target': 99.9,
            'actual_success_rate': 99.0,
            'actual_error_rate': 1.0,
            'expected_analysis': 'Burn rate = 10.0 (consuming budget 10x faster, critical)'
        },
        {
            'name': 'Complete Outage',
            'slo_target': 99.9,
            'actual_success_rate': 50.0,
            'actual_error_rate': 50.0,
            'expected_analysis': 'Burn rate = 500.0 (major incident, budget exhausting rapidly)'
        }
    ]

    for scenario in scenarios:
        allowed_error = 100 - scenario['slo_target']
        burn_rate = scenario['actual_error_rate'] / allowed_error

        print(f"\nScenario: {scenario['name']}")
        print(f"  SLO Target: {scenario['slo_target']}%")
        print(f"  Actual Success Rate: {scenario['actual_success_rate']}%")
        print(f"  Actual Error Rate: {scenario['actual_error_rate']}%")
        print(f"  Calculated Burn Rate: {burn_rate:.1f}x")
        print(f"  Analysis: {scenario['expected_analysis']}")
```

## Python ErrorBudgetManager Implementation

### Complete Production-Ready Implementation

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ErrorBudgetConfig:
    """Configuration for error budget calculation."""
    service_name: str
    slo_target: float  # Percentage (e.g., 99.9)
    window_days: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'critical': 0.10,  # 10% remaining
        'warning': 0.25,   # 25% remaining
        'attention': 0.50  # 50% remaining
    })
    burn_rate_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'critical': 14.4,
        'warning': 6.0,
        'attention': 3.0
    })


@dataclass
class ErrorBudgetMetrics:
    """Current error budget metrics."""
    timestamp: datetime
    slo_target: float
    actual_uptime_pct: float
    total_budget_minutes: float
    consumed_minutes: float
    remaining_minutes: float
    consumed_pct: float
    remaining_pct: float
    burn_rate: float
    days_to_exhaustion: float
    status: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'slo_target': self.slo_target,
            'actual_uptime_pct': self.actual_uptime_pct,
            'total_budget_minutes': self.total_budget_minutes,
            'consumed_minutes': self.consumed_minutes,
            'remaining_minutes': self.remaining_minutes,
            'consumed_pct': self.consumed_pct,
            'remaining_pct': self.remaining_pct,
            'burn_rate': self.burn_rate,
            'days_to_exhaustion': self.days_to_exhaustion,
            'status': self.status
        }


class ErrorBudgetManager:
    """
    Production-ready error budget management system.

    Tracks error budget consumption, calculates burn rates, generates alerts,
    and provides budget status for SLO-driven engineering decisions.

    Example:
        >>> config = ErrorBudgetConfig(
        ...     service_name='api-service',
        ...     slo_target=99.9,
        ...     window_days=30
        ... )
        >>> manager = ErrorBudgetManager(config)
        >>>
        >>> # Calculate current status
        >>> status = manager.calculate_error_budget_status(
        ...     start_date=datetime(2025, 1, 1),
        ...     end_date=datetime(2025, 1, 31),
        ...     actual_uptime_minutes=43180  # Out of 44640 total minutes
        ... )
        >>>
        >>> print(f"Budget remaining: {status.remaining_pct:.1f}%")
        >>> print(f"Burn rate: {status.burn_rate:.1f}x")
        >>> print(f"Status: {status.status}")
    """

    def __init__(self, config: ErrorBudgetConfig):
        """
        Initialize error budget manager.

        Args:
            config: Error budget configuration
        """
        self.config = config
        self.total_budget_minutes = self._calculate_total_budget()

        logger.info(
            f"Initialized ErrorBudgetManager for {config.service_name} "
            f"with {config.slo_target}% SLO ({self.total_budget_minutes:.1f} min budget)"
        )

    def _calculate_total_budget(self) -> float:
        """
        Calculate total error budget in minutes.

        Returns:
            Total error budget in minutes
        """
        total_minutes = self.config.window_days * 24 * 60
        allowed_downtime_ratio = 1 - (self.config.slo_target / 100)
        return total_minutes * allowed_downtime_ratio

    def calculate_error_budget_status(
        self,
        start_date: datetime,
        end_date: datetime,
        actual_uptime_minutes: Optional[float] = None,
        actual_uptime_pct: Optional[float] = None,
        total_requests: Optional[int] = None,
        successful_requests: Optional[int] = None
    ) -> ErrorBudgetMetrics:
        """
        Calculate comprehensive error budget status.

        Args:
            start_date: Start of measurement period
            end_date: End of measurement period
            actual_uptime_minutes: Actual uptime in minutes
            actual_uptime_pct: Actual uptime as percentage
            total_requests: Total number of requests (alternative to time-based)
            successful_requests: Successful requests (alternative to time-based)

        Returns:
            ErrorBudgetMetrics with complete status

        Raises:
            ValueError: If insufficient data provided
        """
        # Calculate total time in period
        total_time_delta = end_date - start_date
        total_minutes = total_time_delta.total_seconds() / 60

        # Determine actual uptime
        if actual_uptime_minutes is not None:
            uptime_minutes = actual_uptime_minutes
            actual_uptime_percentage = (uptime_minutes / total_minutes) * 100
        elif actual_uptime_pct is not None:
            actual_uptime_percentage = actual_uptime_pct
            uptime_minutes = total_minutes * (actual_uptime_pct / 100)
        elif total_requests is not None and successful_requests is not None:
            actual_uptime_percentage = (successful_requests / total_requests) * 100
            uptime_minutes = total_minutes * (actual_uptime_percentage / 100)
        else:
            raise ValueError(
                "Must provide either actual_uptime_minutes, actual_uptime_pct, "
                "or (total_requests, successful_requests)"
            )

        # Calculate consumed budget
        expected_uptime_minutes = total_minutes * (self.config.slo_target / 100)
        consumed_minutes = max(0, expected_uptime_minutes - uptime_minutes)

        # Calculate remaining budget
        remaining_minutes = max(0, self.total_budget_minutes - consumed_minutes)
        consumed_pct = min(100, (consumed_minutes / self.total_budget_minutes) * 100)
        remaining_pct = max(0, 100 - consumed_pct)

        # Calculate burn rate
        downtime_minutes = total_minutes - uptime_minutes
        allowed_downtime = total_minutes * (1 - self.config.slo_target / 100)

        if allowed_downtime > 0:
            burn_rate = downtime_minutes / allowed_downtime
        else:
            burn_rate = 0 if downtime_minutes == 0 else float('inf')

        # Calculate days to exhaustion
        if burn_rate > 0 and remaining_pct > 0:
            days_to_exhaustion = (self.config.window_days * (remaining_pct / 100)) / burn_rate
        elif remaining_pct == 0:
            days_to_exhaustion = 0
        else:
            days_to_exhaustion = float('inf')

        # Determine status
        status = self._determine_status(remaining_pct, burn_rate, days_to_exhaustion)

        metrics = ErrorBudgetMetrics(
            timestamp=datetime.now(),
            slo_target=self.config.slo_target,
            actual_uptime_pct=actual_uptime_percentage,
            total_budget_minutes=self.total_budget_minutes,
            consumed_minutes=consumed_minutes,
            remaining_minutes=remaining_minutes,
            consumed_pct=consumed_pct,
            remaining_pct=remaining_pct,
            burn_rate=burn_rate,
            days_to_exhaustion=days_to_exhaustion,
            status=status
        )

        logger.info(
            f"Budget status for {self.config.service_name}: "
            f"{remaining_pct:.1f}% remaining, {burn_rate:.1f}x burn rate, "
            f"status={status}"
        )

        return metrics

    def _determine_status(
        self,
        remaining_pct: float,
        burn_rate: float,
        days_to_exhaustion: float
    ) -> str:
        """
        Determine error budget status.

        Args:
            remaining_pct: Percentage of budget remaining
            burn_rate: Current burn rate
            days_to_exhaustion: Days until budget exhaustion

        Returns:
            Status string: exhausted, critical, warning, attention, or healthy
        """
        if remaining_pct <= 0:
            return 'exhausted'

        # Check critical conditions
        if (remaining_pct <= self.config.alert_thresholds['critical'] * 100 or
            burn_rate >= self.config.burn_rate_thresholds['critical'] or
            days_to_exhaustion <= 3):
            return 'critical'

        # Check warning conditions
        if (remaining_pct <= self.config.alert_thresholds['warning'] * 100 or
            burn_rate >= self.config.burn_rate_thresholds['warning'] or
            days_to_exhaustion <= 7):
            return 'warning'

        # Check attention conditions
        if (remaining_pct <= self.config.alert_thresholds['attention'] * 100 or
            burn_rate >= self.config.burn_rate_thresholds['attention'] or
            days_to_exhaustion <= 15):
            return 'attention'

        return 'healthy'

    def generate_burn_rate_alerts(
        self,
        current_metrics: ErrorBudgetMetrics
    ) -> List[dict]:
        """
        Generate multi-window burn rate alerts.

        Args:
            current_metrics: Current error budget metrics

        Returns:
            List of alert dictionaries
        """
        alerts = []

        # Critical fast burn (14.4x)
        if current_metrics.burn_rate >= self.config.burn_rate_thresholds['critical']:
            alerts.append({
                'severity': 'critical',
                'title': 'Critical Error Budget Burn Rate',
                'description': (
                    f'Service {self.config.service_name} is burning error budget at '
                    f'{current_metrics.burn_rate:.1f}x rate. '
                    f'At this rate, budget will be exhausted in {current_metrics.days_to_exhaustion:.1f} days.'
                ),
                'burn_rate': current_metrics.burn_rate,
                'remaining_pct': current_metrics.remaining_pct,
                'action': 'page',
                'budget_consumed': '2% of monthly budget per hour'
            })

        # Warning fast burn (6x)
        elif current_metrics.burn_rate >= self.config.burn_rate_thresholds['warning']:
            alerts.append({
                'severity': 'warning',
                'title': 'High Error Budget Burn Rate',
                'description': (
                    f'Service {self.config.service_name} is burning error budget at '
                    f'{current_metrics.burn_rate:.1f}x rate. '
                    f'Budget will be exhausted in {current_metrics.days_to_exhaustion:.1f} days if rate continues.'
                ),
                'burn_rate': current_metrics.burn_rate,
                'remaining_pct': current_metrics.remaining_pct,
                'action': 'ticket',
                'budget_consumed': '5% of monthly budget per 6 hours'
            })

        # Attention slow burn (3x)
        elif current_metrics.burn_rate >= self.config.burn_rate_thresholds['attention']:
            alerts.append({
                'severity': 'attention',
                'title': 'Elevated Error Budget Burn Rate',
                'description': (
                    f'Service {self.config.service_name} is burning error budget at '
                    f'{current_metrics.burn_rate:.1f}x rate. '
                    f'Monitor closely.'
                ),
                'burn_rate': current_metrics.burn_rate,
                'remaining_pct': current_metrics.remaining_pct,
                'action': 'monitor',
                'budget_consumed': '10% of monthly budget per day'
            })

        # Low budget warning
        if current_metrics.remaining_pct <= 10 and current_metrics.remaining_pct > 0:
            alerts.append({
                'severity': 'critical',
                'title': 'Error Budget Nearly Exhausted',
                'description': (
                    f'Service {self.config.service_name} has only '
                    f'{current_metrics.remaining_pct:.1f}% error budget remaining '
                    f'({current_metrics.remaining_minutes:.1f} minutes).'
                ),
                'burn_rate': current_metrics.burn_rate,
                'remaining_pct': current_metrics.remaining_pct,
                'action': 'freeze_deployments',
                'budget_consumed': f'{100 - current_metrics.remaining_pct:.1f}% already consumed'
            })

        return alerts

    def get_budget_policy_decision(
        self,
        current_metrics: ErrorBudgetMetrics,
        release_risk: str = 'medium'
    ) -> dict:
        """
        Make release decision based on error budget policy.

        Args:
            current_metrics: Current error budget metrics
            release_risk: Risk level of release (low, medium, high)

        Returns:
            Dictionary with decision and rationale
        """
        decision_matrix = {
            'healthy': {
                'low': 'approve',
                'medium': 'approve',
                'high': 'review'
            },
            'attention': {
                'low': 'approve',
                'medium': 'review',
                'high': 'defer'
            },
            'warning': {
                'low': 'review',
                'medium': 'defer',
                'high': 'block'
            },
            'critical': {
                'low': 'defer',
                'medium': 'block',
                'high': 'block'
            },
            'exhausted': {
                'low': 'block',
                'medium': 'block',
                'high': 'block'
            }
        }

        decision = decision_matrix[current_metrics.status][release_risk]

        rationale = self._build_decision_rationale(
            current_metrics,
            release_risk,
            decision
        )

        return {
            'decision': decision,
            'status': current_metrics.status,
            'release_risk': release_risk,
            'rationale': rationale,
            'metrics': current_metrics.to_dict(),
            'conditions': self._get_approval_conditions(decision, current_metrics)
        }

    def _build_decision_rationale(
        self,
        metrics: ErrorBudgetMetrics,
        risk: str,
        decision: str
    ) -> str:
        """Build human-readable rationale for decision."""
        return (
            f"Decision: {decision.upper()} - "
            f"Budget status is {metrics.status} with {metrics.remaining_pct:.1f}% remaining "
            f"({metrics.remaining_minutes:.1f} minutes). "
            f"Current burn rate is {metrics.burn_rate:.1f}x. "
            f"Release risk assessed as {risk}. "
            f"Projected exhaustion in {metrics.days_to_exhaustion:.1f} days."
        )

    def _get_approval_conditions(
        self,
        decision: str,
        metrics: ErrorBudgetMetrics
    ) -> List[str]:
        """Get conditions for approval."""
        conditions = []

        if decision == 'approve':
            conditions.append('Proceed with normal deployment process')
            conditions.append('Monitor error rates closely post-deployment')

        elif decision == 'review':
            conditions.append('Require manual approval from SRE team')
            conditions.append('Implement additional monitoring')
            conditions.append('Prepare rollback plan')

        elif decision == 'defer':
            conditions.append('Postpone deployment until budget improves')
            conditions.append('Focus on reliability improvements')
            conditions.append('Re-evaluate in 24-48 hours')

        elif decision == 'block':
            conditions.append('Deployment blocked due to error budget policy')
            conditions.append('Feature freeze in effect')
            conditions.append('Only critical bug fixes and reliability improvements allowed')

        return conditions


# Example usage and testing
if __name__ == '__main__':
    # Create configuration
    config = ErrorBudgetConfig(
        service_name='api-service',
        slo_target=99.9,
        window_days=30
    )

    # Initialize manager
    manager = ErrorBudgetManager(config)

    # Calculate status for a scenario
    start = datetime(2025, 1, 1)
    end = datetime(2025, 1, 31)

    # Scenario: 99.85% uptime (slightly below 99.9% SLO)
    total_minutes = 30 * 24 * 60  # 43,200 minutes
    uptime_pct = 99.85

    metrics = manager.calculate_error_budget_status(
        start_date=start,
        end_date=end,
        actual_uptime_pct=uptime_pct
    )

    print("\n=== ERROR BUDGET STATUS ===")
    print(f"Service: {config.service_name}")
    print(f"SLO Target: {config.slo_target}%")
    print(f"Actual Uptime: {metrics.actual_uptime_pct:.2f}%")
    print(f"Budget Consumed: {metrics.consumed_pct:.1f}% ({metrics.consumed_minutes:.1f} min)")
    print(f"Budget Remaining: {metrics.remaining_pct:.1f}% ({metrics.remaining_minutes:.1f} min)")
    print(f"Burn Rate: {metrics.burn_rate:.1f}x")
    print(f"Days to Exhaustion: {metrics.days_to_exhaustion:.1f}")
    print(f"Status: {metrics.status.upper()}")

    # Generate alerts
    alerts = manager.generate_burn_rate_alerts(metrics)
    if alerts:
        print("\n=== ALERTS ===")
        for alert in alerts:
            print(f"\n[{alert['severity'].upper()}] {alert['title']}")
            print(f"  {alert['description']}")
            print(f"  Action: {alert['action']}")

    # Get release decision
    decision = manager.get_budget_policy_decision(metrics, release_risk='medium')
    print("\n=== RELEASE DECISION ===")
    print(f"Decision: {decision['decision'].upper()}")
    print(f"Rationale: {decision['rationale']}")
    print("\nConditions:")
    for condition in decision['conditions']:
        print(f"  - {condition}")
```
