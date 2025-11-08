# Error Budget Management and Burn Rate Detection

Comprehensive guide to error budget calculation, burn rate monitoring, and budget-based engineering decisions for SLO-driven reliability practices.

## Table of Contents

1. [Error Budget Fundamentals](#error-budget-fundamentals)
2. [Error Budget Mathematics](#error-budget-mathematics)
3. [Burn Rate Concepts](#burn-rate-concepts)
4. [Multi-Window Burn Rate Detection](#multi-window-burn-rate-detection)
5. [Budget Consumption Tracking](#budget-consumption-tracking)
6. [Projected Exhaustion Calculations](#projected-exhaustion-calculations)
7. [Budget Status Determination](#budget-status-determination)
8. [Historical Burn Rate Analysis](#historical-burn-rate-analysis)
9. [Python ErrorBudgetManager Implementation](#python-errorbudgetmanager-implementation)
10. [Burn Rate Alert Generation](#burn-rate-alert-generation)
11. [Error Budget Visualization](#error-budget-visualization)
12. [Budget-Based Decision Making](#budget-based-decision-making)
13. [Error Budget Policies](#error-budget-policies)
14. [Advanced Error Budget Patterns](#advanced-error-budget-patterns)

---

## Error Budget Fundamentals

### What is an Error Budget?

An error budget is the **maximum amount of unreliability** that a service can tolerate while still meeting its SLO. It represents the difference between 100% reliability and your SLO target.

**Key Principles:**
- Error budgets balance reliability with feature velocity
- They provide a shared incentive between product and engineering teams
- They enable data-driven decisions about risk and reliability
- They make reliability a feature, not an afterthought

### Error Budget Philosophy

```
SLO = 99.9% availability
Error Budget = 100% - 99.9% = 0.1% allowed unreliability

In a 30-day month:
Total time = 30 days × 24 hours × 60 minutes = 43,200 minutes
Error budget = 43,200 × 0.001 = 43.2 minutes of downtime allowed
```

**Strategic Uses:**
1. **Feature Velocity**: When budget is healthy, ship faster
2. **Risk Taking**: Budget enables innovation and experimentation
3. **Maintenance Windows**: Planned downtime consumes error budget
4. **Incident Response**: Prioritize based on budget consumption
5. **Release Decisions**: Gate risky releases when budget is low

---

## Error Budget Mathematics

### Basic Formulas

#### 1. Total Error Budget Calculation

```python
def calculate_total_error_budget(slo_target: float, window_days: int) -> dict:
    """
    Calculate total error budget for a given SLO and time window.

    Args:
        slo_target: SLO target as percentage (e.g., 99.9)
        window_days: Time window in days (e.g., 30)

    Returns:
        Dictionary with error budget in various units
    """
    total_minutes = window_days * 24 * 60
    total_seconds = total_minutes * 60

    # Allowed downtime ratio
    allowed_downtime_ratio = 1 - (slo_target / 100)

    # Calculate budget in different units
    budget_minutes = total_minutes * allowed_downtime_ratio
    budget_seconds = total_seconds * allowed_downtime_ratio
    budget_hours = budget_minutes / 60

    return {
        'slo_target_pct': slo_target,
        'window_days': window_days,
        'total_minutes': total_minutes,
        'allowed_downtime_ratio': allowed_downtime_ratio,
        'error_budget_minutes': budget_minutes,
        'error_budget_seconds': budget_seconds,
        'error_budget_hours': budget_hours,
        'error_budget_percentage': (1 - slo_target / 100) * 100
    }

# Examples
budget_99_9 = calculate_total_error_budget(99.9, 30)
# Result: 43.2 minutes per month

budget_99_95 = calculate_total_error_budget(99.95, 30)
# Result: 21.6 minutes per month

budget_99_99 = calculate_total_error_budget(99.99, 30)
# Result: 4.32 minutes per month
```

#### 2. Consumed Budget Calculation

```python
def calculate_consumed_budget(
    total_requests: int,
    failed_requests: int,
    slo_target: float
) -> dict:
    """
    Calculate consumed error budget based on actual failures.

    Args:
        total_requests: Total number of requests
        failed_requests: Number of failed requests
        slo_target: SLO target as percentage

    Returns:
        Dictionary with consumption metrics
    """
    # Actual success rate
    actual_success_rate = ((total_requests - failed_requests) / total_requests) * 100

    # Allowed failures for SLO
    allowed_failures = total_requests * (1 - slo_target / 100)

    # Excess failures beyond budget
    excess_failures = max(0, failed_requests - allowed_failures)

    # Budget consumption percentage
    if allowed_failures > 0:
        consumption_pct = (failed_requests / allowed_failures) * 100
    else:
        consumption_pct = 0 if failed_requests == 0 else float('inf')

    return {
        'total_requests': total_requests,
        'failed_requests': failed_requests,
        'actual_success_rate': actual_success_rate,
        'slo_target': slo_target,
        'allowed_failures': allowed_failures,
        'excess_failures': excess_failures,
        'budget_consumed_pct': min(consumption_pct, 100),
        'budget_exceeded': excess_failures > 0
    }
```

#### 3. Time-Based Budget Consumption

```python
from datetime import datetime, timedelta

def calculate_time_based_consumption(
    start_time: datetime,
    end_time: datetime,
    downtime_minutes: float,
    slo_target: float
) -> dict:
    """
    Calculate error budget consumption based on downtime.

    Args:
        start_time: Period start time
        end_time: Period end time
        downtime_minutes: Actual downtime in minutes
        slo_target: SLO target as percentage

    Returns:
        Dictionary with time-based consumption metrics
    """
    # Total time in minutes
    total_time_delta = end_time - start_time
    total_minutes = total_time_delta.total_seconds() / 60

    # Expected uptime based on SLO
    expected_uptime_minutes = total_minutes * (slo_target / 100)

    # Actual uptime
    actual_uptime_minutes = total_minutes - downtime_minutes

    # Budget consumption
    allowed_downtime = total_minutes * (1 - slo_target / 100)
    consumed_minutes = downtime_minutes

    if allowed_downtime > 0:
        consumption_pct = (consumed_minutes / allowed_downtime) * 100
    else:
        consumption_pct = 0 if consumed_minutes == 0 else float('inf')

    remaining_budget_minutes = max(0, allowed_downtime - consumed_minutes)

    return {
        'period_start': start_time,
        'period_end': end_time,
        'total_minutes': total_minutes,
        'downtime_minutes': downtime_minutes,
        'uptime_minutes': actual_uptime_minutes,
        'allowed_downtime_minutes': allowed_downtime,
        'consumed_budget_minutes': consumed_minutes,
        'remaining_budget_minutes': remaining_budget_minutes,
        'budget_consumed_pct': min(consumption_pct, 100),
        'actual_availability_pct': (actual_uptime_minutes / total_minutes) * 100
    }
```

### Common SLO Targets and Budgets

```python
def get_common_slo_budgets() -> dict:
    """Reference table of common SLO targets and error budgets."""

    slo_configs = {
        '90.0%': {'annual_downtime': '36.5 days', 'monthly_downtime': '3 days', 'weekly_downtime': '16.8 hours'},
        '95.0%': {'annual_downtime': '18.25 days', 'monthly_downtime': '1.5 days', 'weekly_downtime': '8.4 hours'},
        '99.0%': {'annual_downtime': '3.65 days', 'monthly_downtime': '7.2 hours', 'weekly_downtime': '1.68 hours'},
        '99.5%': {'annual_downtime': '1.83 days', 'monthly_downtime': '3.6 hours', 'weekly_downtime': '50.4 minutes'},
        '99.9%': {'annual_downtime': '8.76 hours', 'monthly_downtime': '43.2 minutes', 'weekly_downtime': '10.1 minutes'},
        '99.95%': {'annual_downtime': '4.38 hours', 'monthly_downtime': '21.6 minutes', 'weekly_downtime': '5.04 minutes'},
        '99.99%': {'annual_downtime': '52.6 minutes', 'monthly_downtime': '4.32 minutes', 'weekly_downtime': '1.01 minutes'},
        '99.999%': {'annual_downtime': '5.26 minutes', 'monthly_downtime': '25.9 seconds', 'weekly_downtime': '6.05 seconds'}
    }

    return slo_configs
```

---

## Burn Rate Concepts

### What is Burn Rate?

**Burn rate** is the ratio of the rate at which you're consuming your error budget compared to the expected consumption rate for meeting your SLO exactly.

```
Burn Rate = (Actual Error Rate) / (SLO Error Rate)

Where:
- Burn rate = 1: Consuming budget at expected rate (on track)
- Burn rate > 1: Consuming budget faster than expected (problem)
- Burn rate < 1: Consuming budget slower than expected (healthy)
```

### Burn Rate Mathematics

#### Basic Burn Rate Formula

```python
def calculate_burn_rate(
    actual_error_rate: float,
    slo_target: float
) -> float:
    """
    Calculate error budget burn rate.

    Args:
        actual_error_rate: Current error rate as percentage (e.g., 0.5%)
        slo_target: SLO target as percentage (e.g., 99.9%)

    Returns:
        Burn rate multiplier
    """
    # SLO allows this much error
    allowed_error_rate = 100 - slo_target

    # Burn rate calculation
    if allowed_error_rate > 0:
        burn_rate = actual_error_rate / allowed_error_rate
    else:
        burn_rate = 0 if actual_error_rate == 0 else float('inf')

    return burn_rate

# Examples:
# SLO: 99.9% (allows 0.1% errors)
# Actual: 0.1% errors -> Burn rate = 1 (on track)
# Actual: 0.2% errors -> Burn rate = 2 (2x consumption)
# Actual: 1.0% errors -> Burn rate = 10 (10x consumption)
```

#### Time-to-Exhaustion Based on Burn Rate

```python
def calculate_time_to_exhaustion(
    current_budget_remaining_pct: float,
    burn_rate: float,
    window_days: int
) -> dict:
    """
    Calculate when error budget will be exhausted at current burn rate.

    Args:
        current_budget_remaining_pct: Percentage of budget remaining (0-100)
        burn_rate: Current burn rate multiplier
        window_days: SLO window in days

    Returns:
        Dictionary with exhaustion projections
    """
    if burn_rate <= 0:
        return {
            'time_to_exhaustion_days': float('inf'),
            'time_to_exhaustion_hours': float('inf'),
            'exhaustion_date': None,
            'status': 'healthy'
        }

    # Calculate remaining time at current burn rate
    budget_fraction_remaining = current_budget_remaining_pct / 100
    time_to_exhaustion_days = (window_days * budget_fraction_remaining) / burn_rate
    time_to_exhaustion_hours = time_to_exhaustion_days * 24

    # Project exhaustion date
    exhaustion_date = datetime.now() + timedelta(days=time_to_exhaustion_days)

    # Determine status
    if time_to_exhaustion_days < 1:
        status = 'critical'
    elif time_to_exhaustion_days < 3:
        status = 'warning'
    elif time_to_exhaustion_days < 7:
        status = 'attention'
    else:
        status = 'healthy'

    return {
        'time_to_exhaustion_days': time_to_exhaustion_days,
        'time_to_exhaustion_hours': time_to_exhaustion_hours,
        'exhaustion_date': exhaustion_date,
        'status': status,
        'burn_rate': burn_rate,
        'budget_remaining_pct': current_budget_remaining_pct
    }
```

### Standard Burn Rate Thresholds

#### Google SRE Multi-Window, Multi-Burn-Rate Alerts

```python
def get_standard_burn_rate_configs() -> dict:
    """
    Standard burn rate alert configurations from Google SRE.

    Based on:
    - 30-day SLO window
    - Multi-window detection to reduce false positives
    - Different severities for different burn rates
    """

    return {
        'fast_burn_critical': {
            'burn_rate': 14.4,
            'short_window': '1h',
            'long_window': '5m',
            'budget_consumed': '2% of monthly budget in 1 hour',
            'time_to_exhaustion': '2.08 days',
            'severity': 'critical',
            'action': 'page',
            'description': 'At this rate, entire error budget exhausted in ~2 days'
        },
        'fast_burn_warning': {
            'burn_rate': 6,
            'short_window': '6h',
            'long_window': '30m',
            'budget_consumed': '5% of monthly budget in 6 hours',
            'time_to_exhaustion': '5 days',
            'severity': 'warning',
            'action': 'ticket',
            'description': 'At this rate, entire error budget exhausted in ~5 days'
        },
        'slow_burn_warning': {
            'burn_rate': 3,
            'short_window': '24h',
            'long_window': '2h',
            'budget_consumed': '10% of monthly budget in 24 hours',
            'time_to_exhaustion': '10 days',
            'severity': 'warning',
            'action': 'ticket',
            'description': 'At this rate, entire error budget exhausted in ~10 days'
        },
        'slow_burn_info': {
            'burn_rate': 1,
            'short_window': '3d',
            'long_window': '6h',
            'budget_consumed': '10% of monthly budget in 3 days',
            'time_to_exhaustion': '30 days',
            'severity': 'info',
            'action': 'log',
            'description': 'Consuming budget at expected rate for exact SLO target'
        }
    }
```

### Burn Rate Example Scenarios

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

---

## Multi-Window Burn Rate Detection

### Why Multi-Window Detection?

Single-window burn rate alerts can be noisy. Multi-window detection requires **both** a short window and a long window to exceed thresholds, reducing false positives while maintaining sensitivity.

**Benefits:**
- Reduces alert fatigue from transient spikes
- Confirms sustained high error rates
- Balances sensitivity with specificity
- Aligns with Google SRE best practices

### Multi-Window Implementation

```python
from typing import List, Tuple
import numpy as np

class MultiWindowBurnRateDetector:
    """
    Implements multi-window burn rate detection for error budget alerts.

    Based on Google SRE multi-window, multi-burn-rate alerting.
    """

    def __init__(self, slo_target: float, window_days: int = 30):
        """
        Initialize detector.

        Args:
            slo_target: SLO target as percentage (e.g., 99.9)
            window_days: SLO window in days (default: 30)
        """
        self.slo_target = slo_target
        self.window_days = window_days
        self.allowed_error_rate = 100 - slo_target

        # Define alert tiers
        self.alert_tiers = self._define_alert_tiers()

    def _define_alert_tiers(self) -> List[dict]:
        """
        Define multi-window burn rate alert tiers.

        Returns:
            List of alert tier configurations
        """
        return [
            {
                'name': 'critical_fast_burn',
                'burn_rate': 14.4,
                'short_window_minutes': 60,  # 1 hour
                'long_window_minutes': 5,    # 5 minutes
                'budget_consumed_pct': 2.0,
                'severity': 'critical',
                'for_duration': '2m',
                'action': 'page'
            },
            {
                'name': 'warning_fast_burn',
                'burn_rate': 6.0,
                'short_window_minutes': 360,  # 6 hours
                'long_window_minutes': 30,    # 30 minutes
                'budget_consumed_pct': 5.0,
                'severity': 'warning',
                'for_duration': '15m',
                'action': 'ticket'
            },
            {
                'name': 'warning_slow_burn',
                'burn_rate': 3.0,
                'short_window_minutes': 1440,  # 24 hours
                'long_window_minutes': 120,    # 2 hours
                'budget_consumed_pct': 10.0,
                'severity': 'warning',
                'for_duration': '1h',
                'action': 'ticket'
            },
            {
                'name': 'info_sustained_burn',
                'burn_rate': 1.0,
                'short_window_minutes': 4320,  # 3 days
                'long_window_minutes': 360,    # 6 hours
                'budget_consumed_pct': 10.0,
                'severity': 'info',
                'for_duration': '3h',
                'action': 'log'
            }
        ]

    def check_burn_rate(
        self,
        short_window_error_rate: float,
        long_window_error_rate: float,
        tier_name: str
    ) -> dict:
        """
        Check if burn rate exceeds threshold in both windows.

        Args:
            short_window_error_rate: Error rate in short window (%)
            long_window_error_rate: Error rate in long window (%)
            tier_name: Name of alert tier to check

        Returns:
            Dictionary with burn rate analysis
        """
        # Find tier config
        tier = next((t for t in self.alert_tiers if t['name'] == tier_name), None)
        if not tier:
            raise ValueError(f"Unknown tier: {tier_name}")

        # Calculate burn rates
        short_burn_rate = short_window_error_rate / self.allowed_error_rate
        long_burn_rate = long_window_error_rate / self.allowed_error_rate

        # Check if both windows exceed threshold
        short_exceeds = short_burn_rate >= tier['burn_rate']
        long_exceeds = long_burn_rate >= tier['burn_rate']
        alert_triggered = short_exceeds and long_exceeds

        return {
            'tier': tier_name,
            'severity': tier['severity'],
            'burn_rate_threshold': tier['burn_rate'],
            'short_window_minutes': tier['short_window_minutes'],
            'long_window_minutes': tier['long_window_minutes'],
            'short_window_error_rate': short_window_error_rate,
            'long_window_error_rate': long_window_error_rate,
            'short_burn_rate': short_burn_rate,
            'long_burn_rate': long_burn_rate,
            'short_exceeds': short_exceeds,
            'long_exceeds': long_exceeds,
            'alert_triggered': alert_triggered,
            'budget_consumed_pct': tier['budget_consumed_pct'],
            'action': tier['action'],
            'for_duration': tier['for_duration']
        }

    def check_all_tiers(
        self,
        error_rates: dict
    ) -> List[dict]:
        """
        Check all alert tiers and return triggered alerts.

        Args:
            error_rates: Dictionary with error rates for different windows
                Example: {
                    '5m': 0.5,
                    '30m': 0.4,
                    '1h': 0.3,
                    '2h': 0.25,
                    '6h': 0.2,
                    '24h': 0.15,
                    '3d': 0.1
                }

        Returns:
            List of triggered alerts
        """
        triggered_alerts = []

        for tier in self.alert_tiers:
            # Map window sizes to available error rates
            short_key = self._get_window_key(tier['short_window_minutes'])
            long_key = self._get_window_key(tier['long_window_minutes'])

            if short_key in error_rates and long_key in error_rates:
                result = self.check_burn_rate(
                    error_rates[short_key],
                    error_rates[long_key],
                    tier['name']
                )

                if result['alert_triggered']:
                    triggered_alerts.append(result)

        return triggered_alerts

    def _get_window_key(self, minutes: int) -> str:
        """Convert minutes to window key."""
        if minutes < 60:
            return f'{minutes}m'
        elif minutes < 1440:
            return f'{minutes // 60}h'
        else:
            return f'{minutes // 1440}d'
```

### Prometheus Recording Rules for Multi-Window

```yaml
# prometheus_burn_rate_rules.yaml
groups:
  - name: error_budget_burn_rate
    interval: 30s
    rules:
      # Calculate error rates for different windows
      - record: service:error_rate_5m
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
            /
            sum(rate(http_requests_total[5m])) by (service)
          ) * 100

      - record: service:error_rate_30m
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[30m])) by (service)
            /
            sum(rate(http_requests_total[30m])) by (service)
          ) * 100

      - record: service:error_rate_1h
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[1h])) by (service)
            /
            sum(rate(http_requests_total[1h])) by (service)
          ) * 100

      - record: service:error_rate_2h
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[2h])) by (service)
            /
            sum(rate(http_requests_total[2h])) by (service)
          ) * 100

      - record: service:error_rate_6h
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[6h])) by (service)
            /
            sum(rate(http_requests_total[6h])) by (service)
          ) * 100

      - record: service:error_rate_24h
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[24h])) by (service)
            /
            sum(rate(http_requests_total[24h])) by (service)
          ) * 100

      - record: service:error_rate_3d
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[3d])) by (service)
            /
            sum(rate(http_requests_total[3d])) by (service)
          ) * 100

      # Calculate burn rates for different windows
      - record: service:burn_rate_5m
        expr: |
          service:error_rate_5m / (100 - 99.9)  # Adjust SLO target

      - record: service:burn_rate_1h
        expr: |
          service:error_rate_1h / (100 - 99.9)

      - record: service:burn_rate_6h
        expr: |
          service:error_rate_6h / (100 - 99.9)

      - record: service:burn_rate_24h
        expr: |
          service:error_rate_24h / (100 - 99.9)
```

---

## Budget Consumption Tracking

### Comprehensive Budget Tracker

```python
from datetime import datetime, timedelta
from typing import Optional, List
import json

class BudgetConsumptionTracker:
    """
    Track error budget consumption over time with detailed analytics.
    """

    def __init__(
        self,
        service_name: str,
        slo_target: float,
        window_days: int = 30
    ):
        """
        Initialize budget tracker.

        Args:
            service_name: Name of the service
            slo_target: SLO target as percentage
            window_days: SLO window in days
        """
        self.service_name = service_name
        self.slo_target = slo_target
        self.window_days = window_days
        self.allowed_error_rate = 100 - slo_target

        # Calculate total budget
        total_minutes = window_days * 24 * 60
        self.total_budget_minutes = total_minutes * (self.allowed_error_rate / 100)

        # Consumption history
        self.consumption_events = []

    def record_consumption_event(
        self,
        timestamp: datetime,
        duration_minutes: float,
        incident_id: Optional[str] = None,
        description: Optional[str] = None,
        category: str = 'incident'
    ) -> dict:
        """
        Record an error budget consumption event.

        Args:
            timestamp: When the event occurred
            duration_minutes: How long the event lasted
            incident_id: Optional incident ID
            description: Event description
            category: Event category (incident, deployment, maintenance)

        Returns:
            Dictionary with event details
        """
        event = {
            'timestamp': timestamp,
            'duration_minutes': duration_minutes,
            'incident_id': incident_id,
            'description': description,
            'category': category,
            'budget_consumed_pct': (duration_minutes / self.total_budget_minutes) * 100
        }

        self.consumption_events.append(event)

        return event

    def get_current_consumption(
        self,
        as_of: Optional[datetime] = None
    ) -> dict:
        """
        Calculate current budget consumption.

        Args:
            as_of: Calculate consumption as of this time (default: now)

        Returns:
            Dictionary with consumption metrics
        """
        if as_of is None:
            as_of = datetime.now()

        # Filter events within current window
        window_start = as_of - timedelta(days=self.window_days)
        window_events = [
            e for e in self.consumption_events
            if window_start <= e['timestamp'] <= as_of
        ]

        # Calculate total consumption
        total_consumed_minutes = sum(e['duration_minutes'] for e in window_events)
        remaining_budget_minutes = max(0, self.total_budget_minutes - total_consumed_minutes)
        consumed_pct = (total_consumed_minutes / self.total_budget_minutes) * 100

        # Group by category
        consumption_by_category = {}
        for event in window_events:
            cat = event['category']
            if cat not in consumption_by_category:
                consumption_by_category[cat] = {
                    'count': 0,
                    'total_minutes': 0,
                    'percentage': 0
                }
            consumption_by_category[cat]['count'] += 1
            consumption_by_category[cat]['total_minutes'] += event['duration_minutes']

        # Calculate percentages
        for cat_data in consumption_by_category.values():
            cat_data['percentage'] = (cat_data['total_minutes'] / self.total_budget_minutes) * 100

        return {
            'service': self.service_name,
            'as_of': as_of,
            'window_days': self.window_days,
            'slo_target': self.slo_target,
            'total_budget_minutes': self.total_budget_minutes,
            'consumed_minutes': total_consumed_minutes,
            'remaining_minutes': remaining_budget_minutes,
            'consumed_pct': consumed_pct,
            'remaining_pct': max(0, 100 - consumed_pct),
            'event_count': len(window_events),
            'consumption_by_category': consumption_by_category
        }

    def get_consumption_trend(
        self,
        days_back: int = 90,
        granularity_days: int = 1
    ) -> List[dict]:
        """
        Get historical consumption trend.

        Args:
            days_back: How many days of history to include
            granularity_days: Granularity of trend data points

        Returns:
            List of consumption snapshots over time
        """
        trend = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        current_date = start_date
        while current_date <= end_date:
            snapshot = self.get_current_consumption(as_of=current_date)
            trend.append({
                'date': current_date,
                'consumed_pct': snapshot['consumed_pct'],
                'remaining_pct': snapshot['remaining_pct'],
                'consumed_minutes': snapshot['consumed_minutes']
            })
            current_date += timedelta(days=granularity_days)

        return trend

    def export_consumption_report(
        self,
        output_file: str,
        format: str = 'json'
    ):
        """
        Export consumption data to file.

        Args:
            output_file: Path to output file
            format: Output format (json, csv)
        """
        consumption = self.get_current_consumption()

        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump(consumption, f, indent=2, default=str)
        elif format == 'csv':
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=consumption.keys())
                writer.writeheader()
                writer.writerow(consumption)
```

---

## Projected Exhaustion Calculations

### Exhaustion Projections

```python
import numpy as np
from scipy import stats
from typing import List, Tuple

class BudgetExhaustionPredictor:
    """
    Predict when error budget will be exhausted based on current trends.
    """

    def __init__(self, consumption_tracker: BudgetConsumptionTracker):
        """
        Initialize predictor.

        Args:
            consumption_tracker: BudgetConsumptionTracker instance
        """
        self.tracker = consumption_tracker

    def predict_exhaustion_linear(
        self,
        lookback_days: int = 7
    ) -> dict:
        """
        Predict exhaustion using linear regression on recent consumption.

        Args:
            lookback_days: How many days to use for trend analysis

        Returns:
            Dictionary with prediction details
        """
        # Get consumption trend
        trend = self.tracker.get_consumption_trend(
            days_back=lookback_days,
            granularity_days=1
        )

        if len(trend) < 2:
            return {
                'method': 'linear',
                'prediction': 'insufficient_data',
                'exhaustion_date': None,
                'days_until_exhaustion': None,
                'confidence': 0
            }

        # Prepare data for linear regression
        x = np.array([(t['date'] - trend[0]['date']).days for t in trend])
        y = np.array([t['consumed_pct'] for t in trend])

        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Predict when consumption reaches 100%
        if slope <= 0:
            return {
                'method': 'linear',
                'prediction': 'not_exhausting',
                'exhaustion_date': None,
                'days_until_exhaustion': float('inf'),
                'confidence': abs(r_value),
                'trend': 'improving'
            }

        days_until_100 = (100 - y[-1]) / slope
        exhaustion_date = trend[-1]['date'] + timedelta(days=days_until_100)

        return {
            'method': 'linear',
            'prediction': 'exhaustion_projected',
            'exhaustion_date': exhaustion_date,
            'days_until_exhaustion': days_until_100,
            'confidence': abs(r_value),
            'slope': slope,
            'current_consumption': y[-1],
            'trend': 'degrading'
        }

    def predict_exhaustion_current_rate(self) -> dict:
        """
        Predict exhaustion based on current burn rate.

        Returns:
            Dictionary with prediction details
        """
        consumption = self.tracker.get_current_consumption()

        # Calculate current daily consumption rate
        recent_events = sorted(
            self.tracker.consumption_events,
            key=lambda e: e['timestamp'],
            reverse=True
        )[:10]  # Last 10 events

        if not recent_events:
            return {
                'method': 'current_rate',
                'prediction': 'no_recent_events',
                'exhaustion_date': None,
                'days_until_exhaustion': None
            }

        # Calculate average time between events and average duration
        avg_minutes_per_event = np.mean([e['duration_minutes'] for e in recent_events])

        if len(recent_events) > 1:
            time_diffs = [
                (recent_events[i-1]['timestamp'] - recent_events[i]['timestamp']).total_seconds() / 60
                for i in range(1, len(recent_events))
            ]
            avg_minutes_between_events = np.mean(time_diffs)
        else:
            avg_minutes_between_events = 1440  # Default to 1 day

        # Calculate consumption rate (minutes per day)
        daily_consumption_rate = (avg_minutes_per_event / avg_minutes_between_events) * 1440

        # Project exhaustion
        remaining = consumption['remaining_minutes']
        if daily_consumption_rate > 0:
            days_until_exhaustion = remaining / daily_consumption_rate
            exhaustion_date = datetime.now() + timedelta(days=days_until_exhaustion)
        else:
            days_until_exhaustion = float('inf')
            exhaustion_date = None

        return {
            'method': 'current_rate',
            'prediction': 'exhaustion_projected' if days_until_exhaustion < float('inf') else 'stable',
            'exhaustion_date': exhaustion_date,
            'days_until_exhaustion': days_until_exhaustion,
            'daily_consumption_rate_minutes': daily_consumption_rate,
            'remaining_budget_minutes': remaining
        }

    def get_exhaustion_probability(
        self,
        days_ahead: int = 30
    ) -> dict:
        """
        Calculate probability of budget exhaustion within time period.

        Args:
            days_ahead: Time horizon for prediction

        Returns:
            Dictionary with probability estimates
        """
        # Get multiple predictions
        linear_pred = self.predict_exhaustion_linear()
        rate_pred = self.predict_exhaustion_current_rate()

        # Calculate probability based on predictions
        predictions_exhausting = 0
        total_predictions = 0

        if linear_pred['prediction'] == 'exhaustion_projected':
            total_predictions += 1
            if linear_pred['days_until_exhaustion'] <= days_ahead:
                predictions_exhausting += 1

        if rate_pred['prediction'] == 'exhaustion_projected':
            total_predictions += 1
            if rate_pred['days_until_exhaustion'] <= days_ahead:
                predictions_exhausting += 1

        if total_predictions == 0:
            probability = 0
        else:
            probability = predictions_exhausting / total_predictions

        return {
            'time_horizon_days': days_ahead,
            'exhaustion_probability': probability,
            'linear_prediction': linear_pred,
            'rate_prediction': rate_pred,
            'recommendation': self._get_recommendation(probability)
        }

    def _get_recommendation(self, probability: float) -> str:
        """Get recommendation based on exhaustion probability."""
        if probability >= 0.8:
            return 'CRITICAL: High probability of budget exhaustion. Implement immediate reliability improvements and consider feature freeze.'
        elif probability >= 0.5:
            return 'WARNING: Moderate probability of budget exhaustion. Prioritize reliability work and defer risky releases.'
        elif probability >= 0.2:
            return 'ATTENTION: Some risk of budget exhaustion. Monitor closely and have mitigation plans ready.'
        else:
            return 'HEALTHY: Low probability of budget exhaustion. Continue normal operations.'
```

---

## Budget Status Determination

### Status Classification System

```python
from enum import Enum
from typing import Optional

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

---

## Historical Burn Rate Analysis

### Historical Analysis Engine

```python
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from datetime import datetime, timedelta

class BurnRateHistoricalAnalyzer:
    """
    Analyze historical burn rate patterns to identify trends and anomalies.
    """

    def __init__(self, consumption_tracker: BudgetConsumptionTracker):
        """
        Initialize analyzer.

        Args:
            consumption_tracker: BudgetConsumptionTracker instance
        """
        self.tracker = consumption_tracker

    def analyze_burn_rate_patterns(
        self,
        days_back: int = 90
    ) -> dict:
        """
        Analyze burn rate patterns over time.

        Args:
            days_back: How many days of history to analyze

        Returns:
            Dictionary with pattern analysis
        """
        # Get consumption trend
        trend = self.tracker.get_consumption_trend(
            days_back=days_back,
            granularity_days=1
        )

        if len(trend) < 7:
            return {'error': 'Insufficient data for analysis'}

        # Convert to pandas DataFrame for analysis
        df = pd.DataFrame(trend)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Calculate daily burn rate
        df['daily_consumption'] = df['consumed_minutes'].diff()
        df['burn_rate'] = df['daily_consumption'] / (
            self.tracker.total_budget_minutes / self.tracker.window_days
        )

        # Statistical analysis
        analysis = {
            'period': {
                'start': df.index.min(),
                'end': df.index.max(),
                'days': len(df)
            },
            'burn_rate_stats': {
                'mean': df['burn_rate'].mean(),
                'median': df['burn_rate'].median(),
                'std': df['burn_rate'].std(),
                'min': df['burn_rate'].min(),
                'max': df['burn_rate'].max(),
                'p25': df['burn_rate'].quantile(0.25),
                'p75': df['burn_rate'].quantile(0.75),
                'p95': df['burn_rate'].quantile(0.95),
                'p99': df['burn_rate'].quantile(0.99)
            },
            'consumption_stats': {
                'total_consumed_minutes': df['consumed_minutes'].iloc[-1],
                'total_consumed_pct': df['consumed_pct'].iloc[-1],
                'avg_daily_consumption': df['daily_consumption'].mean(),
                'max_daily_consumption': df['daily_consumption'].max()
            },
            'patterns': self._identify_patterns(df),
            'anomalies': self._detect_anomalies(df),
            'trends': self._analyze_trends(df)
        }

        return analysis

    def _identify_patterns(self, df: pd.DataFrame) -> dict:
        """Identify patterns in burn rate data."""
        patterns = {}

        # Day of week pattern
        df['day_of_week'] = df.index.dayofweek
        dow_pattern = df.groupby('day_of_week')['burn_rate'].mean()

        patterns['day_of_week'] = {
            'Monday': dow_pattern.get(0, 0),
            'Tuesday': dow_pattern.get(1, 0),
            'Wednesday': dow_pattern.get(2, 0),
            'Thursday': dow_pattern.get(3, 0),
            'Friday': dow_pattern.get(4, 0),
            'Saturday': dow_pattern.get(5, 0),
            'Sunday': dow_pattern.get(6, 0)
        }

        # Time-based patterns
        patterns['weekly_pattern'] = self._detect_weekly_pattern(df)
        patterns['monthly_pattern'] = self._detect_monthly_pattern(df)

        return patterns

    def _detect_weekly_pattern(self, df: pd.DataFrame) -> dict:
        """Detect weekly cyclical patterns."""
        # Group by week
        df['week'] = df.index.isocalendar().week
        weekly_avg = df.groupby('week')['burn_rate'].mean()

        return {
            'has_weekly_cycle': weekly_avg.std() > weekly_avg.mean() * 0.3,
            'weekly_variation': weekly_avg.std(),
            'typical_week_burn_rate': weekly_avg.mean()
        }

    def _detect_monthly_pattern(self, df: pd.DataFrame) -> dict:
        """Detect monthly patterns."""
        df['month'] = df.index.month
        monthly_avg = df.groupby('month')['burn_rate'].mean()

        return {
            'has_monthly_cycle': monthly_avg.std() > monthly_avg.mean() * 0.3,
            'monthly_variation': monthly_avg.std(),
            'typical_month_burn_rate': monthly_avg.mean()
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> List[dict]:
        """Detect anomalous burn rate events."""
        anomalies = []

        # Use z-score for anomaly detection
        mean = df['burn_rate'].mean()
        std = df['burn_rate'].std()

        # Find values > 3 standard deviations
        df['z_score'] = (df['burn_rate'] - mean) / std
        anomaly_df = df[abs(df['z_score']) > 3]

        for date, row in anomaly_df.iterrows():
            anomalies.append({
                'date': date,
                'burn_rate': row['burn_rate'],
                'z_score': row['z_score'],
                'severity': 'high' if abs(row['z_score']) > 4 else 'medium'
            })

        return anomalies

    def _analyze_trends(self, df: pd.DataFrame) -> dict:
        """Analyze long-term trends."""
        # Linear trend
        x = np.arange(len(df))
        y = df['burn_rate'].values

        # Remove NaN values
        valid_idx = ~np.isnan(y)
        x_valid = x[valid_idx]
        y_valid = y[valid_idx]

        if len(x_valid) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)

            trend_direction = 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable'

            return {
                'trend_direction': trend_direction,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'projected_7day_change': slope * 7
            }
        else:
            return {'error': 'Insufficient data for trend analysis'}

    def generate_burn_rate_report(
        self,
        days_back: int = 90
    ) -> str:
        """
        Generate human-readable burn rate analysis report.

        Args:
            days_back: How many days to analyze

        Returns:
            Formatted report string
        """
        analysis = self.analyze_burn_rate_patterns(days_back)

        if 'error' in analysis:
            return f"Error: {analysis['error']}"

        report = f"""
ERROR BUDGET BURN RATE ANALYSIS
{'=' * 80}

Service: {self.tracker.service_name}
SLO Target: {self.tracker.slo_target}%
Analysis Period: {analysis['period']['start']} to {analysis['period']['end']} ({analysis['period']['days']} days)

BURN RATE STATISTICS
{'-' * 80}
Mean Burn Rate: {analysis['burn_rate_stats']['mean']:.2f}x
Median Burn Rate: {analysis['burn_rate_stats']['median']:.2f}x
Std Deviation: {analysis['burn_rate_stats']['std']:.2f}x
Min/Max: {analysis['burn_rate_stats']['min']:.2f}x / {analysis['burn_rate_stats']['max']:.2f}x
95th Percentile: {analysis['burn_rate_stats']['p95']:.2f}x
99th Percentile: {analysis['burn_rate_stats']['p99']:.2f}x

CONSUMPTION METRICS
{'-' * 80}
Total Consumed: {analysis['consumption_stats']['total_consumed_minutes']:.1f} minutes ({analysis['consumption_stats']['total_consumed_pct']:.1f}%)
Avg Daily Consumption: {analysis['consumption_stats']['avg_daily_consumption']:.1f} minutes
Max Daily Consumption: {analysis['consumption_stats']['max_daily_consumption']:.1f} minutes

TRENDS
{'-' * 80}
Trend Direction: {analysis['trends']['trend_direction'].upper()}
Statistical Significance: {'YES' if analysis['trends']['is_significant'] else 'NO'} (p={analysis['trends']['p_value']:.4f})
R²: {analysis['trends']['r_squared']:.3f}
Projected 7-Day Change: {analysis['trends']['projected_7day_change']:.2f}x

ANOMALIES DETECTED
{'-' * 80}
"""

        if analysis['anomalies']:
            for anomaly in analysis['anomalies']:
                report += f"  {anomaly['date']}: Burn rate {anomaly['burn_rate']:.2f}x (z-score: {anomaly['z_score']:.2f}, severity: {anomaly['severity']})\n"
        else:
            report += "  No significant anomalies detected.\n"

        return report
```

---

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

---

## Burn Rate Alert Generation

### Prometheus Alert Rules

```yaml
# prometheus_error_budget_alerts.yaml
groups:
  - name: error_budget_alerts
    interval: 30s
    rules:
      # Page: 2% budget in 1 hour (14.4x burn rate)
      - alert: ErrorBudgetFastBurnCritical
        expr: |
          (
            service:burn_rate_5m > 14.4
            AND
            service:burn_rate_1h > 14.4
          )
        for: 2m
        labels:
          severity: critical
          team: sre
          alert_type: error_budget
        annotations:
          summary: "Critical error budget burn for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} is burning error budget at {{ $value }}x rate.

            Details:
            - Burn Rate: {{ $value }}x (threshold: 14.4x)
            - Budget Impact: ~2% of monthly budget consumed per hour
            - Time to Exhaustion: ~2 days at current rate

            IMMEDIATE ACTION REQUIRED:
            - Page on-call engineer
            - Begin incident response
            - Identify root cause
            - Implement mitigation

            Runbook: https://runbooks.company.com/error-budget-critical
            Dashboard: https://grafana.company.com/d/{{ $labels.service }}-slo

      # Ticket: 5% budget in 6 hours (6x burn rate)
      - alert: ErrorBudgetFastBurnWarning
        expr: |
          (
            service:burn_rate_30m > 6
            AND
            service:burn_rate_6h > 6
          )
        for: 15m
        labels:
          severity: warning
          team: sre
          alert_type: error_budget
        annotations:
          summary: "High error budget burn for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} is burning error budget at {{ $value }}x rate.

            Details:
            - Burn Rate: {{ $value }}x (threshold: 6x)
            - Budget Impact: ~5% of monthly budget consumed per 6 hours
            - Time to Exhaustion: ~5 days at current rate

            ACTIONS:
            - Create incident ticket
            - Investigate error patterns
            - Review recent changes
            - Plan mitigation

            Runbook: https://runbooks.company.com/error-budget-warning

      # Monitor: 10% budget in 24 hours (3x burn rate)
      - alert: ErrorBudgetSlowBurn
        expr: |
          (
            service:burn_rate_2h > 3
            AND
            service:burn_rate_24h > 3
          )
        for: 1h
        labels:
          severity: warning
          team: sre
          alert_type: error_budget
        annotations:
          summary: "Elevated error budget burn for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} is burning error budget at {{ $value }}x rate.

            Details:
            - Burn Rate: {{ $value }}x (threshold: 3x)
            - Budget Impact: ~10% of monthly budget consumed per day
            - Time to Exhaustion: ~10 days at current rate

            ACTIONS:
            - Monitor closely
            - Review error trends
            - Plan reliability improvements

      # Budget depletion warning
      - alert: ErrorBudgetLow
        expr: |
          (
            service:error_budget_remaining_pct < 25
            AND
            service:error_budget_remaining_pct > 10
          )
        for: 5m
        labels:
          severity: warning
          team: sre
          alert_type: error_budget
        annotations:
          summary: "Low error budget for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} has {{ $value }}% error budget remaining.

            ACTIONS:
            - Review pending deployments
            - Prioritize reliability work
            - Defer risky changes

      # Budget exhaustion critical
      - alert: ErrorBudgetCritical
        expr: |
          service:error_budget_remaining_pct < 10
        for: 5m
        labels:
          severity: critical
          team: sre
          alert_type: error_budget
        annotations:
          summary: "Critical error budget for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} has only {{ $value }}% error budget remaining.

            IMMEDIATE ACTIONS:
            - Feature freeze
            - Block non-critical deployments
            - Focus on reliability
            - Notify stakeholders

      # Budget exhausted
      - alert: ErrorBudgetExhausted
        expr: |
          service:error_budget_remaining_pct <= 0
        for: 1m
        labels:
          severity: critical
          team: sre
          alert_type: error_budget
        annotations:
          summary: "Error budget EXHAUSTED for {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} has EXHAUSTED its error budget.
            SLO violation in progress.

            MANDATORY ACTIONS:
            - Complete feature freeze
            - Block ALL deployments except critical fixes
            - Incident retrospective required
            - Stakeholder notification
            - SLO review and adjustment
```

### Python Alert Generator

```python
from typing import List, Dict, Optional
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class BurnRateAlertGenerator:
    """
    Generate and send error budget burn rate alerts.
    """

    def __init__(
        self,
        manager: ErrorBudgetManager,
        notification_config: Dict[str, any]
    ):
        """
        Initialize alert generator.

        Args:
            manager: ErrorBudgetManager instance
            notification_config: Notification configuration
        """
        self.manager = manager
        self.config = notification_config

    def check_and_alert(
        self,
        metrics: ErrorBudgetMetrics
    ) -> List[dict]:
        """
        Check metrics and generate alerts if needed.

        Args:
            metrics: Current error budget metrics

        Returns:
            List of alerts that were sent
        """
        alerts = self.manager.generate_burn_rate_alerts(metrics)

        sent_alerts = []
        for alert in alerts:
            # Send alert based on severity
            if alert['severity'] == 'critical':
                self._send_page(alert)
            elif alert['severity'] == 'warning':
                self._send_ticket(alert)
            else:
                self._send_notification(alert)

            sent_alerts.append(alert)

        return sent_alerts

    def _send_page(self, alert: dict):
        """Send page to on-call engineer."""
        logger.critical(f"PAGING: {alert['title']}")

        # Integration with PagerDuty, Opsgenie, etc.
        if 'pagerduty' in self.config:
            self._send_pagerduty(alert)

        # Also send email
        self._send_email(
            to=self.config.get('on_call_email'),
            subject=f"[CRITICAL] {alert['title']}",
            body=self._format_alert_email(alert)
        )

    def _send_ticket(self, alert: dict):
        """Create incident ticket."""
        logger.warning(f"Creating ticket: {alert['title']}")

        # Integration with Jira, GitHub Issues, etc.
        if 'jira' in self.config:
            self._create_jira_ticket(alert)

        # Send email notification
        self._send_email(
            to=self.config.get('team_email'),
            subject=f"[WARNING] {alert['title']}",
            body=self._format_alert_email(alert)
        )

    def _send_notification(self, alert: dict):
        """Send informational notification."""
        logger.info(f"Notification: {alert['title']}")

        # Send to Slack, etc.
        if 'slack' in self.config:
            self._send_slack(alert)

    def _format_alert_email(self, alert: dict) -> str:
        """Format alert as email body."""
        return f"""
ERROR BUDGET ALERT: {alert['title']}

Severity: {alert['severity'].upper()}
Service: {self.manager.config.service_name}
Timestamp: {datetime.now().isoformat()}

{alert['description']}

Metrics:
- Burn Rate: {alert['burn_rate']:.1f}x
- Budget Remaining: {alert['remaining_pct']:.1f}%
- Budget Consumed: {alert['budget_consumed']}

Recommended Action: {alert['action']}

Dashboard: {self.config.get('dashboard_url', 'N/A')}
Runbook: {self.config.get('runbook_url', 'N/A')}
"""

    def _send_email(
        self,
        to: str,
        subject: str,
        body: str
    ):
        """Send email notification."""
        if not self.config.get('smtp_enabled'):
            logger.info(f"Email (disabled): {subject}")
            return

        msg = MIMEMultipart()
        msg['From'] = self.config['smtp_from']
        msg['To'] = to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP(
                self.config['smtp_host'],
                self.config['smtp_port']
            ) as server:
                server.starttls()
                if 'smtp_username' in self.config:
                    server.login(
                        self.config['smtp_username'],
                        self.config['smtp_password']
                    )
                server.send_message(msg)

            logger.info(f"Email sent to {to}: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
```

---

## Error Budget Visualization

### Grafana Dashboard JSON

```python
def create_error_budget_dashboard(
    service_name: str,
    slo_target: float = 99.9
) -> dict:
    """
    Generate complete Grafana dashboard for error budget monitoring.

    Args:
        service_name: Name of the service
        slo_target: SLO target percentage

    Returns:
        Grafana dashboard JSON
    """
    return {
        "dashboard": {
            "title": f"Error Budget - {service_name}",
            "tags": ["slo", "error-budget", service_name],
            "timezone": "browser",
            "refresh": "30s",
            "time": {
                "from": "now-7d",
                "to": "now"
            },
            "panels": [
                # Row 1: Summary Stats
                {
                    "id": 1,
                    "title": "Budget Remaining",
                    "type": "gauge",
                    "gridPos": {"x": 0, "y": 0, "w": 6, "h": 8},
                    "targets": [{
                        "expr": f'''
                        100 * (
                            1 - (
                                (100 - service:success_rate_30d{{service="{service_name}"}}) /
                                (100 - {slo_target})
                            )
                        )
                        ''',
                        "refId": "A"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "min": 0,
                            "max": 100,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "red"},
                                    {"value": 10, "color": "orange"},
                                    {"value": 25, "color": "yellow"},
                                    {"value": 50, "color": "green"}
                                ]
                            }
                        }
                    }
                },
                {
                    "id": 2,
                    "title": "Current Burn Rate",
                    "type": "stat",
                    "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
                    "targets": [{
                        "expr": f'service:burn_rate_1h{{service="{service_name}"}}',
                        "refId": "A"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "short",
                            "decimals": 1,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "green"},
                                    {"value": 1, "color": "yellow"},
                                    {"value": 3, "color": "orange"},
                                    {"value": 6, "color": "red"}
                                ]
                            }
                        }
                    },
                    "options": {
                        "textMode": "value_and_name"
                    }
                },
                {
                    "id": 3,
                    "title": "SLO Compliance (30d)",
                    "type": "stat",
                    "gridPos": {"x": 12, "y": 0, "w": 6, "h": 4},
                    "targets": [{
                        "expr": f'service:success_rate_30d{{service="{service_name}"}}',
                        "refId": "A"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "decimals": 3,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "red"},
                                    {"value": slo_target - 0.1, "color": "orange"},
                                    {"value": slo_target, "color": "green"}
                                ]
                            }
                        }
                    }
                },
                {
                    "id": 4,
                    "title": "Time to Exhaustion",
                    "type": "stat",
                    "gridPos": {"x": 18, "y": 0, "w": 6, "h": 4},
                    "targets": [{
                        "expr": f'''
                        30 * (
                            100 * (
                                1 - (
                                    (100 - service:success_rate_30d{{service="{service_name}"}}) /
                                    (100 - {slo_target})
                                )
                            ) / 100
                        ) / service:burn_rate_1h{{service="{service_name}"}}
                        ''',
                        "refId": "A"
                    }],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "d",
                            "decimals": 1,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "red"},
                                    {"value": 3, "color": "orange"},
                                    {"value": 7, "color": "yellow"},
                                    {"value": 15, "color": "green"}
                                ]
                            }
                        }
                    }
                },

                # Row 2: Burn Rate Trend
                {
                    "id": 10,
                    "title": "Burn Rate Over Time",
                    "type": "graph",
                    "gridPos": {"x": 0, "y": 8, "w": 24, "h": 8},
                    "targets": [
                        {
                            "expr": f'service:burn_rate_5m{{service="{service_name}"}}',
                            "legendFormat": "5min",
                            "refId": "A"
                        },
                        {
                            "expr": f'service:burn_rate_1h{{service="{service_name}"}}',
                            "legendFormat": "1hour",
                            "refId": "B"
                        },
                        {
                            "expr": f'service:burn_rate_6h{{service="{service_name}"}}',
                            "legendFormat": "6hour",
                            "refId": "C"
                        },
                        {
                            "expr": f'service:burn_rate_24h{{service="{service_name}"}}',
                            "legendFormat": "24hour",
                            "refId": "D"
                        }
                    ],
                    "yaxes": [
                        {
                            "format": "short",
                            "label": "Burn Rate (x)",
                            "min": 0,
                            "logBase": 1
                        }
                    ],
                    "seriesOverrides": [
                        {
                            "alias": "5min",
                            "linewidth": 1
                        },
                        {
                            "alias": "1hour",
                            "linewidth": 2
                        }
                    ],
                    "thresholds": [
                        {
                            "value": 1,
                            "op": "gt",
                            "fill": False,
                            "line": True,
                            "colorMode": "custom",
                            "lineColor": "rgba(255, 255, 0, 0.7)"
                        },
                        {
                            "value": 3,
                            "op": "gt",
                            "colorMode": "custom",
                            "lineColor": "rgba(255, 165, 0, 0.7)"
                        },
                        {
                            "value": 14.4,
                            "op": "gt",
                            "colorMode": "custom",
                            "lineColor": "rgba(255, 0, 0, 0.7)"
                        }
                    ]
                },

                # Row 3: Budget Consumption Trend
                {
                    "id": 20,
                    "title": "Error Budget Consumption (30-day rolling)",
                    "type": "graph",
                    "gridPos": {"x": 0, "y": 16, "w": 24, "h": 8},
                    "targets": [{
                        "expr": f'''
                        100 * (
                            (100 - service:success_rate_30d{{service="{service_name}"}}) /
                            (100 - {slo_target})
                        )
                        ''',
                        "legendFormat": "Budget Consumed %",
                        "refId": "A"
                    }],
                    "yaxes": [
                        {
                            "format": "percent",
                            "label": "Budget Consumed",
                            "min": 0,
                            "max": 100
                        }
                    ],
                    "thresholds": [
                        {
                            "value": 50,
                            "colorMode": "custom",
                            "fill": True,
                            "op": "gt",
                            "fillColor": "rgba(255, 255, 0, 0.1)"
                        },
                        {
                            "value": 75,
                            "colorMode": "custom",
                            "fill": True,
                            "op": "gt",
                            "fillColor": "rgba(255, 165, 0, 0.2)"
                        },
                        {
                            "value": 90,
                            "colorMode": "custom",
                            "fill": True,
                            "op": "gt",
                            "fillColor": "rgba(255, 0, 0, 0.3)"
                        }
                    ]
                },

                # Row 4: Error Rate Details
                {
                    "id": 30,
                    "title": "Error Rate by Window",
                    "type": "graph",
                    "gridPos": {"x": 0, "y": 24, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": f'service:error_rate_5m{{service="{service_name}"}}',
                            "legendFormat": "5min",
                            "refId": "A"
                        },
                        {
                            "expr": f'service:error_rate_1h{{service="{service_name}"}}',
                            "legendFormat": "1hour",
                            "refId": "B"
                        },
                        {
                            "expr": f'service:error_rate_24h{{service="{service_name}"}}',
                            "legendFormat": "24hour",
                            "refId": "C"
                        }
                    ],
                    "yaxes": [
                        {
                            "format": "percent",
                            "label": "Error Rate",
                            "min": 0
                        }
                    ]
                },
                {
                    "id": 31,
                    "title": "Request Rate",
                    "type": "graph",
                    "gridPos": {"x": 12, "y": 24, "w": 12, "h": 8},
                    "targets": [{
                        "expr": f'sum(rate(http_requests_total{{service="{service_name}"}}[5m]))',
                        "legendFormat": "Requests/sec",
                        "refId": "A"
                    }],
                    "yaxes": [
                        {
                            "format": "reqps",
                            "label": "Requests/sec",
                            "min": 0
                        }
                    ]
                }
            ],
            "templating": {
                "list": [
                    {
                        "name": "service",
                        "type": "constant",
                        "current": {
                            "value": service_name
                        }
                    },
                    {
                        "name": "slo_target",
                        "type": "constant",
                        "current": {
                            "value": str(slo_target)
                        }
                    }
                ]
            },
            "annotations": {
                "list": [
                    {
                        "datasource": "Prometheus",
                        "enable": True,
                        "expr": f'ALERTS{{alertname=~"ErrorBudget.*", service="{service_name}"}}',
                        "iconColor": "red",
                        "name": "Error Budget Alerts",
                        "tagKeys": "severity",
                        "textFormat": "{{alertname}}",
                        "titleFormat": "Alert"
                    },
                    {
                        "datasource": "Prometheus",
                        "enable": True,
                        "expr": f'changes(service:success_rate_30d{{service="{service_name}"}}[5m]) != 0',
                        "iconColor": "blue",
                        "name": "Deployments",
                        "titleFormat": "Deploy"
                    }
                ]
            }
        }
    }
```

---

## Budget-Based Decision Making

### Release Gate Policy

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

---

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

---

## Advanced Error Budget Patterns

### Composite Error Budgets

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

---

This comprehensive error budget documentation provides production-ready code, formulas, and patterns for implementing robust error budget tracking and burn rate monitoring in SLO-driven reliability practices.
