# SLO Maintenance and Operations

This document covers error budget management, burn rate alerting, and measurement window selection strategies.

## 6. Error Budget Mathematics

### 6.1 Error Budget Fundamentals

**Core Principle:**
```
Error Budget = 1 - SLO Target
```

The error budget represents the acceptable amount of unreliability, providing a quantitative basis for balancing innovation and reliability.

### 6.2 Error Budget Burn Rate

**Burn Rate Definition:**
```
Burn Rate = (Actual Error Rate) / (Allowed Error Rate)
```

**Interpreting Burn Rates:**
- Burn Rate = 1: Consuming budget at expected rate
- Burn Rate > 1: Consuming budget faster than sustainable
- Burn Rate < 1: Consuming budget slower than expected (good!)

```python
class BurnRateCalculator:
    """Calculate and analyze error budget burn rates"""

    @staticmethod
    def calculate_burn_rate(actual_error_rate, allowed_error_rate):
        """
        Calculate error budget burn rate

        Args:
            actual_error_rate: Current error rate (e.g., 0.002 = 0.2%)
            allowed_error_rate: Target error rate (e.g., 0.001 = 0.1%)

        Returns:
            Burn rate and interpretation
        """
        if allowed_error_rate == 0:
            return float('inf')

        burn_rate = actual_error_rate / allowed_error_rate

        # Calculate budget depletion time
        if burn_rate > 0:
            days_to_depletion = 30 / burn_rate  # Assuming 30-day window
        else:
            days_to_depletion = float('inf')

        return {
            'burn_rate': burn_rate,
            'interpretation': BurnRateCalculator._interpret_burn_rate(burn_rate),
            'days_to_budget_depletion': days_to_depletion,
            'recommended_action': BurnRateCalculator._recommend_action(burn_rate)
        }

    @staticmethod
    def _interpret_burn_rate(burn_rate):
        """Interpret what the burn rate means"""
        if burn_rate <= 0.5:
            return 'Excellent: Using < 50% of allowed error budget'
        elif burn_rate <= 1.0:
            return 'Good: Within error budget'
        elif burn_rate <= 1.5:
            return 'Attention: Burning budget faster than expected'
        elif burn_rate <= 2.0:
            return 'Warning: Significantly elevated burn rate'
        else:
            return 'Critical: Extremely high burn rate'

    @staticmethod
    def _recommend_action(burn_rate):
        """Recommend action based on burn rate"""
        if burn_rate <= 1.0:
            return ['Continue monitoring', 'Consider feature deployment']
        elif burn_rate <= 1.5:
            return ['Investigate error sources', 'Defer risky deployments']
        elif burn_rate <= 2.0:
            return ['Stop feature deployments', 'Focus on reliability', 'Page on-call']
        else:
            return ['Incident response', 'All hands on reliability', 'Consider rollback']

# Example:
result = BurnRateCalculator.calculate_burn_rate(
    actual_error_rate=0.002,  # 0.2% errors
    allowed_error_rate=0.001  # 0.1% allowed (99.9% SLO)
)
# Result: burn_rate = 2.0, days_to_depletion = 15 days
```

### 6.3 Multi-Window Burn Rate Alerting

**Alerting Strategy:**
Use multiple time windows to balance sensitivity and specificity:

```python
class MultiWindowBurnRateAlert:
    """
    Implement multi-window burn rate alerting
    Based on Google SRE practices
    """

    # Alert configurations
    ALERT_CONFIGS = {
        'fast_burn': {
            'description': 'Critical: 2% budget consumed in 1 hour',
            'budget_consumed': 0.02,  # 2% of monthly budget
            'short_window': '5m',
            'long_window': '1h',
            'burn_rate_threshold': 14.4,  # (0.02 * 30 days * 24 hours) / 1 hour
            'severity': 'critical',
            'notification': 'page'
        },
        'slow_burn': {
            'description': 'Warning: 5% budget consumed in 6 hours',
            'budget_consumed': 0.05,  # 5% of monthly budget
            'short_window': '30m',
            'long_window': '6h',
            'burn_rate_threshold': 6.0,  # (0.05 * 30 days * 24 hours) / 6 hours
            'severity': 'warning',
            'notification': 'ticket'
        }
    }

    @staticmethod
    def calculate_threshold_burn_rate(budget_fraction, window_hours, total_window_hours=720):
        """
        Calculate burn rate threshold for alerting

        Args:
            budget_fraction: Fraction of budget (e.g., 0.02 for 2%)
            window_hours: Detection window in hours
            total_window_hours: Total SLO window in hours (default 30 days = 720 hours)

        Formula:
            burn_rate = (budget_fraction * total_window_hours) / window_hours

        Example:
            2% budget in 1 hour:
            burn_rate = (0.02 * 720) / 1 = 14.4x
        """
        burn_rate = (budget_fraction * total_window_hours) / window_hours

        return {
            'burn_rate_threshold': burn_rate,
            'budget_fraction': budget_fraction,
            'window_hours': window_hours,
            'interpretation': f'{budget_fraction*100}% of monthly budget consumed in {window_hours} hours'
        }

    @staticmethod
    def generate_prometheus_alert(alert_type, slo_target=99.9):
        """
        Generate Prometheus alerting rule for multi-window burn rate

        Args:
            alert_type: 'fast_burn' or 'slow_burn'
            slo_target: SLO target percentage (default 99.9)
        """
        config = MultiWindowBurnRateAlert.ALERT_CONFIGS[alert_type]
        error_budget = 1 - (slo_target / 100)

        alert_rule = f"""
# {config['description']}
- alert: ErrorBudget{alert_type.title().replace('_', '')}
  expr: |
    (
      # Short window: {config['short_window']}
      (
        1 - (
          sum(rate(http_requests_total{{status!~"5.."}}[{config['short_window']}]))
          /
          sum(rate(http_requests_total[{config['short_window']}]))
        )
      ) / {error_budget} > {config['burn_rate_threshold']}

      AND

      # Long window: {config['long_window']}
      (
        1 - (
          sum(rate(http_requests_total{{status!~"5.."}}[{config['long_window']}]))
          /
          sum(rate(http_requests_total[{config['long_window']}]))
        )
      ) / {error_budget} > {config['burn_rate_threshold']}
    )
  for: 2m
  labels:
    severity: {config['severity']}
    alert_type: error_budget_burn
  annotations:
    summary: "{config['description']}"
    description: |
      Error budget is being consumed at {{{{ $value }}}}x the normal rate.
      At this rate, {config['budget_consumed']*100}% of the monthly budget
      will be consumed in {config['long_window']}.

      Current burn rate: {{{{ $value | humanize }}}}x
      Threshold: {config['burn_rate_threshold']}x
    runbook_url: https://runbooks.example.com/error-budget-burn
"""

        return alert_rule

# Example usage:
fast_burn_threshold = MultiWindowBurnRateAlert.calculate_threshold_burn_rate(
    budget_fraction=0.02,  # 2% of budget
    window_hours=1  # in 1 hour
)
# Result: 14.4x burn rate threshold

alert_yaml = MultiWindowBurnRateAlert.generate_prometheus_alert('fast_burn')
# Generates complete Prometheus alert rule
```

### 6.4 Error Budget Policy

```python
class ErrorBudgetPolicy:
    """
    Define error budget policies for decision making
    Based on Google SRE practices
    """

    def __init__(self, service_name, slo_target=99.9):
        self.service = service_name
        self.slo_target = slo_target
        self.policy = self._define_policy()

    def _define_policy(self):
        """Define error budget-based policy"""
        return {
            'budget_ranges': {
                'healthy': {
                    'range': (75, 100),  # 75-100% budget remaining
                    'actions': {
                        'deployments': 'approved',
                        'risky_changes': 'approved_with_review',
                        'feature_work': '100%',
                        'reliability_work': '0-20%'
                    }
                },
                'moderate': {
                    'range': (50, 75),  # 50-75% budget remaining
                    'actions': {
                        'deployments': 'approved',
                        'risky_changes': 'requires_approval',
                        'feature_work': '70%',
                        'reliability_work': '30%'
                    }
                },
                'concerning': {
                    'range': (25, 50),  # 25-50% budget remaining
                    'actions': {
                        'deployments': 'requires_approval',
                        'risky_changes': 'blocked',
                        'feature_work': '40%',
                        'reliability_work': '60%'
                    }
                },
                'critical': {
                    'range': (0, 25),  # 0-25% budget remaining
                    'actions': {
                        'deployments': 'blocked',
                        'risky_changes': 'blocked',
                        'feature_work': '0%',
                        'reliability_work': '100%'
                    }
                },
                'exhausted': {
                    'range': (-float('inf'), 0),  # Budget exhausted
                    'actions': {
                        'deployments': 'blocked',
                        'risky_changes': 'blocked',
                        'feature_work': '0%',
                        'reliability_work': '100%',
                        'escalation': 'executive_review'
                    }
                }
            }
        }

    def get_policy_actions(self, budget_remaining_percentage):
        """Get policy actions based on remaining budget"""
        for status, config in self.policy['budget_ranges'].items():
            range_min, range_max = config['range']
            if range_min <= budget_remaining_percentage < range_max:
                return {
                    'status': status,
                    'budget_remaining': budget_remaining_percentage,
                    'actions': config['actions'],
                    'rationale': self._explain_rationale(status, budget_remaining_percentage)
                }

        return None

    def _explain_rationale(self, status, budget_remaining):
        """Explain the rationale for policy actions"""
        explanations = {
            'healthy': f'With {budget_remaining:.1f}% budget remaining, focus on feature velocity',
            'moderate': f'With {budget_remaining:.1f}% budget remaining, balance features and reliability',
            'concerning': f'With {budget_remaining:.1f}% budget remaining, prioritize reliability',
            'critical': f'With {budget_remaining:.1f}% budget remaining, emergency reliability focus',
            'exhausted': 'Budget exhausted. All feature work blocked until reliability improved'
        }
        return explanations.get(status, '')

# Example:
policy = ErrorBudgetPolicy('api-service', slo_target=99.9)
actions = policy.get_policy_actions(budget_remaining_percentage=35)
# Result: 'concerning' status, deployments require approval, 60% reliability work
```

### 6.5 Error Budget Reporting

```python
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class ErrorBudgetReporter:
    """Generate error budget reports and visualizations"""

    def __init__(self, slo_target, window_days=30):
        self.slo_target = slo_target
        self.window_days = window_days
        self.error_budget_total = 1 - (slo_target / 100)

    def calculate_budget_consumption(self, measurements):
        """
        Calculate error budget consumption from measurements

        Args:
            measurements: List of {'timestamp': datetime, 'success': bool}

        Returns:
            Detailed budget consumption analysis
        """
        total_requests = len(measurements)
        failed_requests = sum(1 for m in measurements if not m['success'])

        actual_success_rate = ((total_requests - failed_requests) / total_requests) * 100
        actual_error_rate = 1 - (actual_success_rate / 100)

        # Calculate consumption
        consumed_budget = actual_error_rate
        remaining_budget = self.error_budget_total - consumed_budget
        consumption_percentage = (consumed_budget / self.error_budget_total) * 100

        # Calculate by time period
        start_time = min(m['timestamp'] for m in measurements)
        end_time = max(m['timestamp'] for m in measurements)
        elapsed_hours = (end_time - start_time).total_seconds() / 3600

        burn_rate = (consumed_budget / self.error_budget_total) / (elapsed_hours / (self.window_days * 24))

        return {
            'summary': {
                'slo_target': self.slo_target,
                'actual_performance': actual_success_rate,
                'slo_met': actual_success_rate >= self.slo_target
            },
            'budget': {
                'total': self.error_budget_total,
                'consumed': consumed_budget,
                'remaining': remaining_budget,
                'consumption_percentage': consumption_percentage
            },
            'requests': {
                'total': total_requests,
                'successful': total_requests - failed_requests,
                'failed': failed_requests
            },
            'burn_rate': {
                'current': burn_rate,
                'elapsed_hours': elapsed_hours,
                'projected_exhaustion_days': (self.window_days * remaining_budget) / consumed_budget if consumed_budget > 0 else float('inf')
            }
        }

    def generate_report(self, budget_data):
        """Generate formatted error budget report"""
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              ERROR BUDGET REPORT                             ║
╠══════════════════════════════════════════════════════════════╣
║ SLO Target:              {budget_data['summary']['slo_target']:.3f}%                        ║
║ Actual Performance:      {budget_data['summary']['actual_performance']:.3f}%                        ║
║ Status:                  {'✓ MEETING SLO' if budget_data['summary']['slo_met'] else '✗ MISSING SLO'}                    ║
╠══════════════════════════════════════════════════════════════╣
║ Total Error Budget:      {budget_data['budget']['total']:.6f} ({budget_data['budget']['total']*100:.4f}%)          ║
║ Budget Consumed:         {budget_data['budget']['consumed']:.6f} ({budget_data['budget']['consumption_percentage']:.2f}%)          ║
║ Budget Remaining:        {budget_data['budget']['remaining']:.6f} ({100-budget_data['budget']['consumption_percentage']:.2f}%)          ║
╠══════════════════════════════════════════════════════════════╣
║ Total Requests:          {budget_data['requests']['total']:,}                          ║
║ Failed Requests:         {budget_data['requests']['failed']:,}                          ║
╠══════════════════════════════════════════════════════════════╣
║ Burn Rate:               {budget_data['burn_rate']['current']:.2f}x                           ║
║ Hours Elapsed:           {budget_data['burn_rate']['elapsed_hours']:.1f}                            ║
║ Projected Exhaustion:    {budget_data['burn_rate']['projected_exhaustion_days']:.1f} days                       ║
╚══════════════════════════════════════════════════════════════╝
"""
        return report
```

---

## 7. Measurement Window Selection

### 7.1 Window Types

**Rolling Windows:**
- Continuously updated based on the most recent N days/hours
- Provides real-time view of performance
- More responsive to recent changes
- Smoother transition of data

**Calendar Windows:**
- Fixed periods (weekly, monthly, quarterly)
- Aligns with business reporting cycles
- Easier to understand and communicate
- Clear reset points

### 7.2 Rolling Window Implementation

```python
from collections import deque
from datetime import datetime, timedelta

class RollingWindowSLO:
    """
    Implement rolling window SLO tracking
    """

    def __init__(self, window_days=30):
        self.window_days = window_days
        self.window_seconds = window_days * 24 * 3600
        self.measurements = deque()

    def add_measurement(self, timestamp, success):
        """Add a measurement to the rolling window"""
        measurement = {
            'timestamp': timestamp,
            'success': success
        }
        self.measurements.append(measurement)

        # Remove measurements outside the window
        self._cleanup_old_measurements()

    def _cleanup_old_measurements(self):
        """Remove measurements older than the window"""
        if not self.measurements:
            return

        cutoff_time = datetime.now() - timedelta(days=self.window_days)

        while self.measurements and self.measurements[0]['timestamp'] < cutoff_time:
            self.measurements.popleft()

    def calculate_slo(self):
        """Calculate current SLO over the rolling window"""
        if not self.measurements:
            return None

        total = len(self.measurements)
        successes = sum(1 for m in self.measurements if m['success'])

        success_rate = (successes / total) * 100

        oldest = self.measurements[0]['timestamp']
        newest = self.measurements[-1]['timestamp']
        actual_window_hours = (newest - oldest).total_seconds() / 3600

        return {
            'success_rate': success_rate,
            'total_requests': total,
            'successful_requests': successes,
            'failed_requests': total - successes,
            'window_start': oldest,
            'window_end': newest,
            'actual_window_hours': actual_window_hours
        }

    def get_trend(self, num_points=10):
        """Calculate SLO trend over time"""
        if len(self.measurements) < num_points:
            return []

        chunk_size = len(self.measurements) // num_points
        trends = []

        for i in range(num_points):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = list(self.measurements)[start_idx:end_idx]

            if chunk:
                successes = sum(1 for m in chunk if m['success'])
                rate = (successes / len(chunk)) * 100
                trends.append({
                    'timestamp': chunk[-1]['timestamp'],
                    'success_rate': rate
                })

        return trends
```

### 7.3 Calendar Window Implementation

```python
from datetime import datetime, timedelta
import calendar

class CalendarWindowSLO:
    """
    Implement calendar-based SLO tracking
    """

    def __init__(self, window_type='monthly'):
        """
        Args:
            window_type: 'weekly', 'monthly', 'quarterly'
        """
        self.window_type = window_type
        self.current_period_measurements = []
        self.historical_periods = {}

    def _get_period_key(self, timestamp):
        """Get the period key for a timestamp"""
        if self.window_type == 'weekly':
            # ISO week number
            year, week, _ = timestamp.isocalendar()
            return f'{year}-W{week:02d}'
        elif self.window_type == 'monthly':
            return f'{timestamp.year}-{timestamp.month:02d}'
        elif self.window_type == 'quarterly':
            quarter = (timestamp.month - 1) // 3 + 1
            return f'{timestamp.year}-Q{quarter}'
        else:
            raise ValueError(f'Unknown window type: {self.window_type}')

    def add_measurement(self, timestamp, success):
        """Add a measurement to the appropriate period"""
        period_key = self._get_period_key(timestamp)
        current_period = self._get_period_key(datetime.now())

        measurement = {
            'timestamp': timestamp,
            'success': success
        }

        if period_key == current_period:
            self.current_period_measurements.append(measurement)
        else:
            # Historical period
            if period_key not in self.historical_periods:
                self.historical_periods[period_key] = []
            self.historical_periods[period_key].append(measurement)

    def calculate_current_period_slo(self):
        """Calculate SLO for the current period"""
        if not self.current_period_measurements:
            return None

        total = len(self.current_period_measurements)
        successes = sum(1 for m in self.current_period_measurements if m['success'])

        period_key = self._get_period_key(datetime.now())

        return {
            'period': period_key,
            'success_rate': (successes / total) * 100,
            'total_requests': total,
            'successful_requests': successes,
            'period_type': self.window_type
        }

    def get_period_comparison(self, num_periods=6):
        """Compare SLO across multiple periods"""
        comparisons = []

        # Add current period
        current = self.calculate_current_period_slo()
        if current:
            comparisons.append(current)

        # Add historical periods (sorted by period key)
        sorted_periods = sorted(self.historical_periods.keys(), reverse=True)[:num_periods-1]

        for period_key in sorted_periods:
            measurements = self.historical_periods[period_key]
            total = len(measurements)
            successes = sum(1 for m in measurements if m['success'])

            comparisons.append({
                'period': period_key,
                'success_rate': (successes / total) * 100,
                'total_requests': total,
                'successful_requests': successes,
                'period_type': self.window_type
            })

        return comparisons
```

### 7.4 Window Selection Guidelines

```python
class WindowSelectionGuide:
    """Guide for selecting appropriate measurement windows"""

    @staticmethod
    def recommend_window(service_characteristics):
        """
        Recommend measurement window based on service characteristics

        Args:
            service_characteristics: Dict with service properties

        Returns:
            Recommended window configuration
        """
        recommendations = []

        # Traffic volume considerations
        if service_characteristics.get('requests_per_day', 0) < 1000:
            recommendations.append({
                'window_type': 'rolling',
                'window_size_days': 7,
                'rationale': 'Low traffic volume requires longer window for statistical significance'
            })
        else:
            recommendations.append({
                'window_type': 'rolling',
                'window_size_days': 30,
                'rationale': 'Standard 30-day rolling window for adequate traffic'
            })

        # Business cycle alignment
        if service_characteristics.get('has_business_cycles', False):
            recommendations.append({
                'window_type': 'calendar',
                'window_size': 'monthly',
                'rationale': 'Calendar windows align with business reporting cycles'
            })

        # Seasonality considerations
        if service_characteristics.get('seasonal', False):
            recommendations.append({
                'window_type': 'rolling',
                'window_size_days': 90,
                'rationale': 'Quarterly window smooths seasonal variations'
            })

        # Compliance requirements
        if service_characteristics.get('compliance_required', False):
            recommendations.append({
                'window_type': 'calendar',
                'window_size': 'quarterly',
                'rationale': 'Calendar windows for compliance reporting'
            })

        return recommendations

    @staticmethod
    def calculate_minimum_sample_size(slo_target, confidence_level=0.95, margin_of_error=0.01):
        """
        Calculate minimum sample size for statistically valid SLO

        Args:
            slo_target: Target SLO percentage (e.g., 99.9)
            confidence_level: Confidence level (default 95%)
            margin_of_error: Acceptable margin of error (default 1%)

        Returns:
            Minimum number of samples needed
        """
        import math
        from scipy import stats

        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence_level, 1.96)

        # Expected proportion
        p = slo_target / 100

        # Sample size formula for proportions
        n = (z ** 2 * p * (1 - p)) / (margin_of_error ** 2)

        return {
            'minimum_samples': math.ceil(n),
            'slo_target': slo_target,
            'confidence_level': confidence_level,
            'margin_of_error': margin_of_error,
            'interpretation': f'Need at least {math.ceil(n):,} samples for statistically valid measurement'
        }

# Example usage:
recommendations = WindowSelectionGuide.recommend_window({
    'requests_per_day': 100000,
    'has_business_cycles': True,
    'seasonal': False,
    'compliance_required': True
})

sample_size = WindowSelectionGuide.calculate_minimum_sample_size(
    slo_target=99.9,
    confidence_level=0.95,
    margin_of_error=0.01
)
# Result: Need ~38,400 samples for statistically valid 99.9% SLO measurement
```

### 7.5 Hybrid Window Approach

```python
class HybridWindowSLO:
    """
    Implement hybrid approach using both rolling and calendar windows
    """

    def __init__(self):
        self.rolling_30d = RollingWindowSLO(window_days=30)
        self.calendar_monthly = CalendarWindowSLO(window_type='monthly')
        self.calendar_quarterly = CalendarWindowSLO(window_type='quarterly')

    def add_measurement(self, timestamp, success):
        """Add measurement to all window types"""
        self.rolling_30d.add_measurement(timestamp, success)
        self.calendar_monthly.add_measurement(timestamp, success)
        self.calendar_quarterly.add_measurement(timestamp, success)

    def get_comprehensive_view(self):
        """Get SLO status across all window types"""
        return {
            'rolling_30d': self.rolling_30d.calculate_slo(),
            'current_month': self.calendar_monthly.calculate_current_period_slo(),
            'current_quarter': self.calendar_quarterly.calculate_current_period_slo(),
            'monthly_trend': self.calendar_monthly.get_period_comparison(num_periods=6),
            'quarterly_trend': self.calendar_quarterly.get_period_comparison(num_periods=4)
        }
```
