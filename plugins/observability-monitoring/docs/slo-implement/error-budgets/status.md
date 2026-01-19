# Budget Status Determination

## Status Classification System

```python
from enum import Enum
from typing import Optional, List

class BudgetStatus(Enum):
    """Error budget status levels."""
    HEALTHY = "healthy"
    ATTENTION = "attention"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"

class BudgetStatusDeterminer:
    """
    Determine error budget status based on multiple factors.
    """

    def __init__(
        self,
        slo_target: float,
        thresholds: Optional[dict] = None
    ):
        """
        Initialize status determiner.

        Args:
            slo_target: SLO target as percentage
            thresholds: Custom threshold configuration
        """
        self.slo_target = slo_target
        self.thresholds = thresholds or self._default_thresholds()

    def _default_thresholds(self) -> dict:
        """
        Default status thresholds.

        Returns:
            Dictionary with status thresholds
        """
        return {
            'exhausted': {
                'remaining_pct': 0,
                'burn_rate': None,  # Any burn rate
                'days_to_exhaustion': 0
            },
            'critical': {
                'remaining_pct': 10,
                'burn_rate': 5.0,
                'days_to_exhaustion': 3
            },
            'warning': {
                'remaining_pct': 25,
                'burn_rate': 2.0,
                'days_to_exhaustion': 7
            },
            'attention': {
                'remaining_pct': 50,
                'burn_rate': 1.5,
                'days_to_exhaustion': 15
            },
            'healthy': {
                'remaining_pct': 50,
                'burn_rate': 1.0,
                'days_to_exhaustion': 15
            }
        }

    def determine_status(
        self,
        remaining_budget_pct: float,
        current_burn_rate: float,
        days_to_exhaustion: float
    ) -> dict:
        """
        Determine current budget status.

        Args:
            remaining_budget_pct: Percentage of budget remaining (0-100)
            current_burn_rate: Current burn rate multiplier
            days_to_exhaustion: Projected days until budget exhaustion

        Returns:
            Dictionary with status determination
        """
        # Check exhausted
        if remaining_budget_pct <= self.thresholds['exhausted']['remaining_pct']:
            status = BudgetStatus.EXHAUSTED
            severity_score = 100

        # Check critical
        elif (remaining_budget_pct <= self.thresholds['critical']['remaining_pct'] or
              current_burn_rate >= self.thresholds['critical']['burn_rate'] or
              days_to_exhaustion <= self.thresholds['critical']['days_to_exhaustion']):
            status = BudgetStatus.CRITICAL
            severity_score = 80

        # Check warning
        elif (remaining_budget_pct <= self.thresholds['warning']['remaining_pct'] or
              current_burn_rate >= self.thresholds['warning']['burn_rate'] or
              days_to_exhaustion <= self.thresholds['warning']['days_to_exhaustion']):
            status = BudgetStatus.WARNING
            severity_score = 60

        # Check attention
        elif (remaining_budget_pct <= self.thresholds['attention']['remaining_pct'] or
              current_burn_rate >= self.thresholds['attention']['burn_rate'] or
              days_to_exhaustion <= self.thresholds['attention']['days_to_exhaustion']):
            status = BudgetStatus.ATTENTION
            severity_score = 40

        # Healthy
        else:
            status = BudgetStatus.HEALTHY
            severity_score = 20

        return {
            'status': status.value,
            'severity_score': severity_score,
            'remaining_budget_pct': remaining_budget_pct,
            'current_burn_rate': current_burn_rate,
            'days_to_exhaustion': days_to_exhaustion,
            'triggers': self._identify_triggers(
                status,
                remaining_budget_pct,
                current_burn_rate,
                days_to_exhaustion
            ),
            'actions': self._get_recommended_actions(status),
            'color': self._get_status_color(status)
        }

    def _identify_triggers(
        self,
        status: BudgetStatus,
        remaining_pct: float,
        burn_rate: float,
        days_to_exhaustion: float
    ) -> List[str]:
        """Identify which factors triggered the status."""
        triggers = []

        if status == BudgetStatus.EXHAUSTED:
            triggers.append('budget_depleted')
            return triggers

        threshold = self.thresholds.get(status.value, {})

        if remaining_pct <= threshold.get('remaining_pct', 0):
            triggers.append(f'low_remaining_budget_{remaining_pct:.1f}pct')

        if burn_rate >= threshold.get('burn_rate', float('inf')):
            triggers.append(f'high_burn_rate_{burn_rate:.1f}x')

        if days_to_exhaustion <= threshold.get('days_to_exhaustion', float('inf')):
            triggers.append(f'near_exhaustion_{days_to_exhaustion:.1f}_days')

        return triggers

    def _get_recommended_actions(self, status: BudgetStatus) -> List[str]:
        """Get recommended actions for status level."""
        action_map = {
            BudgetStatus.EXHAUSTED: [
                'Feature freeze in effect',
                'Block all non-critical deployments',
                'Focus entirely on reliability improvements',
                'Conduct incident review',
                'Update stakeholders on SLO breach'
            ],
            BudgetStatus.CRITICAL: [
                'Defer risky deployments',
                'Prioritize reliability work',
                'Increase monitoring',
                'Prepare incident response',
                'Notify stakeholders'
            ],
            BudgetStatus.WARNING: [
                'Review pending deployments',
                'Schedule reliability improvements',
                'Analyze recent incidents',
                'Monitor burn rate closely'
            ],
            BudgetStatus.ATTENTION: [
                'Monitor burn rate trends',
                'Review error patterns',
                'Plan reliability work',
                'Continue normal operations with awareness'
            ],
            BudgetStatus.HEALTHY: [
                'Continue normal operations',
                'Maintain vigilance',
                'Consider strategic improvements'
            ]
        }

        return action_map.get(status, [])

    def _get_status_color(self, status: BudgetStatus) -> str:
        """Get color code for status."""
        color_map = {
            BudgetStatus.EXHAUSTED: '#8B0000',  # Dark red
            BudgetStatus.CRITICAL: '#FF0000',   # Red
            BudgetStatus.WARNING: '#FFA500',    # Orange
            BudgetStatus.ATTENTION: '#FFFF00',  # Yellow
            BudgetStatus.HEALTHY: '#00FF00'     # Green
        }

        return color_map.get(status, '#808080')
```
