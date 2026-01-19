# Error Budget Mathematics

## Basic Formulas

### 1. Total Error Budget Calculation

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

### 2. Consumed Budget Calculation

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

### 3. Time-Based Budget Consumption

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

## Burn Rate Mathematics

### Basic Burn Rate Formula

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

### Time-to-Exhaustion Based on Burn Rate

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

## Common SLO Targets and Budgets

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
