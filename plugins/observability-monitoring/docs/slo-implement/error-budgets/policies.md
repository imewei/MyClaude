# Budget-Based Decision Making

## Release Gate Policy

### Automated Release Gate

```python
class ErrorBudgetReleaseGate:
    """
    Automated release gate based on error budget policy.
    """

    def __init__(
        self,
        manager: ErrorBudgetManager,
        policy_config: Optional[dict] = None
    ):
        """
        Initialize release gate.

        Args:
            manager: ErrorBudgetManager instance
            policy_config: Custom policy configuration
        """
        self.manager = manager
        self.policy = policy_config or self._default_policy()

    def _default_policy(self) -> dict:
        """Default error budget policy."""
        return {
            'gates': {
                'healthy': {
                    'min_budget_pct': 50,
                    'max_burn_rate': 1.5,
                    'allowed_risk_levels': ['low', 'medium', 'high']
                },
                'attention': {
                    'min_budget_pct': 25,
                    'max_burn_rate': 2.0,
                    'allowed_risk_levels': ['low', 'medium']
                },
                'warning': {
                    'min_budget_pct': 10,
                    'max_burn_rate': 3.0,
                    'allowed_risk_levels': ['low']
                },
                'critical': {
                    'min_budget_pct': 0,
                    'max_burn_rate': float('inf'),
                    'allowed_risk_levels': []
                },
                'exhausted': {
                    'min_budget_pct': 0,
                    'max_burn_rate': float('inf'),
                    'allowed_risk_levels': []
                }
            },
            'exceptions': {
                'critical_security_fix': True,
                'slo_improvement': True,
                'rollback': True
            }
        }

    def evaluate_release(
        self,
        metrics: ErrorBudgetMetrics,
        release_metadata: dict
    ) -> dict:
        """
        Evaluate if release should be allowed.

        Args:
            metrics: Current error budget metrics
            release_metadata: Release information
                {
                    'risk_level': 'low'|'medium'|'high',
                    'release_type': 'feature'|'bugfix'|'security'|'rollback',
                    'has_tests': bool,
                    'has_rollback_plan': bool
                }

        Returns:
            Evaluation result with decision
        """
        risk_level = release_metadata.get('risk_level', 'medium')
        release_type = release_metadata.get('release_type', 'feature')

        # Check for exceptions
        if self._is_exception_release(release_type):
            return {
                'approved': True,
                'reason': f'Exception: {release_type} releases always allowed',
                'conditions': ['Monitor closely', 'Have rollback ready']
            }

        # Check budget policy
        gate = self.policy['gates'][metrics.status]

        # Check if risk level is allowed
        if risk_level not in gate['allowed_risk_levels']:
            return {
                'approved': False,
                'reason': f'{risk_level.capitalize()} risk releases not allowed when budget status is {metrics.status}',
                'alternative': 'Defer until budget improves or reduce risk level',
                'metrics': metrics.to_dict()
            }

        # Check budget percentage
        if metrics.remaining_pct < gate['min_budget_pct']:
            return {
                'approved': False,
                'reason': f'Insufficient error budget: {metrics.remaining_pct:.1f}% < {gate["min_budget_pct"]}%',
                'alternative': 'Wait for budget to recover',
                'metrics': metrics.to_dict()
            }

        # Check burn rate
        if metrics.burn_rate > gate['max_burn_rate']:
            return {
                'approved': False,
                'reason': f'Burn rate too high: {metrics.burn_rate:.1f}x > {gate["max_burn_rate"]}x',
                'alternative': 'Reduce error rate before new releases',
                'metrics': metrics.to_dict()
            }

        # Additional checks
        conditions = []
        if not release_metadata.get('has_tests'):
            conditions.append('WARNING: No tests detected. Strongly recommend adding tests.')

        if not release_metadata.get('has_rollback_plan'):
            conditions.append('REQUIRED: Must have rollback plan documented')

        return {
            'approved': True,
            'reason': f'Budget policy allows {risk_level} risk releases when status is {metrics.status}',
            'conditions': conditions,
            'metrics': metrics.to_dict()
        }

    def _is_exception_release(self, release_type: str) -> bool:
        """Check if release type is an exception."""
        return self.policy['exceptions'].get(release_type, False)
```

## Error Budget Policies

### Example Policy Document

```yaml
# error_budget_policy.yaml
service: api-service
slo_target: 99.9
window: 30d

policy:
  description: |
    This error budget policy governs release decisions and reliability priorities
    for the API service based on SLO performance and error budget consumption.

  objectives:
    - Maintain 99.9% availability over 30-day rolling window
    - Balance reliability with feature velocity
    - Make data-driven decisions about risk

  gates:
    healthy:
      criteria:
        - budget_remaining >= 50%
        - burn_rate < 1.5x
      allowed_releases:
        - All feature releases
        - All bug fixes
        - Routine maintenance
      deployment_pace: normal

    attention:
      criteria:
        - budget_remaining >= 25% AND < 50%
        - burn_rate >= 1.5x AND < 2.0x
      allowed_releases:
        - Low and medium risk features
        - All bug fixes
        - Critical maintenance
      deployment_pace: cautious
      additional_requirements:
        - Enhanced monitoring for 24h post-deploy
        - Rollback plan documented

    warning:
      criteria:
        - budget_remaining >= 10% AND < 25%
        - burn_rate >= 2.0x AND < 5.0x
      allowed_releases:
        - Low risk features only
        - Bug fixes
        - Reliability improvements
      deployment_pace: minimal
      additional_requirements:
        - SRE approval required
        - Canary deployment mandatory
        - Automated rollback enabled

    critical:
      criteria:
        - budget_remaining < 10%
        - burn_rate >= 5.0x
      allowed_releases:
        - Critical bug fixes only
        - Reliability improvements only
        - Rollbacks
      deployment_pace: frozen
      actions:
        - Feature freeze in effect
        - All hands on reliability
        - Daily status updates

    exhausted:
      criteria:
        - budget_remaining <= 0%
      allowed_releases:
        - Emergency fixes only
        - Rollbacks
      deployment_pace: locked
      actions:
        - Complete deployment freeze
        - Incident declared
        - Executive escalation
        - Post-mortem required

  exceptions:
    - type: critical_security_vulnerability
      override: always_allow
      requirements:
        - Security team approval
        - Accelerated testing

    - type: slo_improvement
      override: always_allow
      requirements:
        - Demonstrated SLO improvement
        - Minimal blast radius

    - type: rollback
      override: always_allow
      requirements:
        - Incident in progress
        - Rollback to known good state

  escalation:
    budget_25pct:
      notify: [team_lead, sre_oncall]
      action: review_pending_releases

    budget_10pct:
      notify: [team_lead, sre_oncall, engineering_manager]
      action: defer_non_critical_releases

    budget_exhausted:
      notify: [team_lead, sre_oncall, engineering_manager, vp_engineering]
      action: feature_freeze_and_postmortem

  review_cadence:
    weekly: budget_status_review
    monthly: policy_effectiveness_review
    quarterly: slo_target_adjustment_review
```
