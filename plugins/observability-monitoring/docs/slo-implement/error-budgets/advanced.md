# Advanced Error Budget Patterns

## Composite Error Budgets

```python
class CompositeErrorBudgetManager:
    """
    Manage error budgets across multiple SLIs with weighted composition.
    """

    def __init__(self, service_name: str):
        """
        Initialize composite manager.

        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        self.sli_managers = {}
        self.weights = {}

    def add_sli(
        self,
        sli_name: str,
        manager: ErrorBudgetManager,
        weight: float = 1.0
    ):
        """
        Add an SLI to the composite budget.

        Args:
            sli_name: Name of the SLI (e.g., 'availability', 'latency')
            manager: ErrorBudgetManager for this SLI
            weight: Weight of this SLI in composite calculation
        """
        self.sli_managers[sli_name] = manager
        self.weights[sli_name] = weight

    def calculate_composite_status(
        self,
        sli_metrics: Dict[str, ErrorBudgetMetrics]
    ) -> dict:
        """
        Calculate composite error budget status.

        Args:
            sli_metrics: Dictionary mapping SLI names to their metrics

        Returns:
            Composite budget status
        """
        total_weight = sum(self.weights.values())

        # Weighted average of budget consumption
        weighted_consumed = sum(
            sli_metrics[sli].consumed_pct * self.weights[sli]
            for sli in sli_metrics
        ) / total_weight

        # Weighted average of burn rate
        weighted_burn_rate = sum(
            sli_metrics[sli].burn_rate * self.weights[sli]
            for sli in sli_metrics
        ) / total_weight

        # Most restrictive status
        statuses = [sli_metrics[sli].status for sli in sli_metrics]
        status_priority = ['exhausted', 'critical', 'warning', 'attention', 'healthy']
        composite_status = next(
            (s for s in status_priority if s in statuses),
            'healthy'
        )

        return {
            'service': self.service_name,
            'composite_consumed_pct': weighted_consumed,
            'composite_remaining_pct': 100 - weighted_consumed,
            'composite_burn_rate': weighted_burn_rate,
            'composite_status': composite_status,
            'sli_details': {
                sli: sli_metrics[sli].to_dict()
                for sli in sli_metrics
            }
        }
```
