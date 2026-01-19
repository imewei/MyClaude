# Budget Consumption Tracking

## Comprehensive Budget Tracker

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
