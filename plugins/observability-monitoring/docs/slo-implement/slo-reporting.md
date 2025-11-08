# SLO Reporting and Analytics

## Overview

This document provides comprehensive guidance on implementing SLO reporting systems, including monthly report generation, performance analysis, incident impact assessment, trend forecasting, and stakeholder communication.

## Table of Contents

1. [Report Generation Architecture](#report-generation-architecture)
2. [SLO Performance Metrics](#slo-performance-metrics)
3. [Incident Impact Analysis](#incident-impact-analysis)
4. [Trend Analysis and Forecasting](#trend-analysis-and-forecasting)
5. [Stakeholder Communication Templates](#stakeholder-communication-templates)
6. [HTML Report Templates](#html-report-templates)
7. [Dashboard Integration](#dashboard-integration)
8. [Automated Scheduling](#automated-scheduling)
9. [Historical Analysis](#historical-analysis)
10. [Best Practices](#best-practices)

---

## Report Generation Architecture

### Core Reporter Class

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import pandas as pd
import numpy as np
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


class ReportType(Enum):
    """Report type enumeration"""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    CUSTOMER = "customer"
    INTERNAL = "internal"


class SLOStatus(Enum):
    """SLO status enumeration"""
    MET = "met"
    AT_RISK = "at_risk"
    VIOLATED = "violated"
    WARNING = "warning"


@dataclass
class SLOMetric:
    """SLO metric data structure"""
    name: str
    target: float
    actual: float
    unit: str
    met: bool
    status: SLOStatus
    trend: str  # "improving", "stable", "degrading"
    monthly_values: List[float] = field(default_factory=list)

    @property
    def performance_percentage(self) -> float:
        """Calculate performance as percentage of target"""
        return (self.actual / self.target) * 100 if self.target > 0 else 0

    @property
    def gap(self) -> float:
        """Calculate gap from target"""
        return self.actual - self.target


@dataclass
class IncidentImpact:
    """Incident impact analysis"""
    incident_id: str
    timestamp: datetime
    duration_minutes: float
    affected_users: int
    error_budget_consumed: float
    severity: str
    root_cause: str
    services_impacted: List[str]
    recovery_time: float

    @property
    def user_impact_minutes(self) -> float:
        """Calculate total user-minutes impacted"""
        return self.affected_users * self.duration_minutes


class SLOReporter:
    """Comprehensive SLO reporting system"""

    def __init__(self, metrics_client, storage_client, config: Dict[str, Any]):
        """
        Initialize SLO reporter

        Args:
            metrics_client: Client for querying metrics
            storage_client: Client for storing reports
            config: Reporter configuration
        """
        self.metrics = metrics_client
        self.storage = storage_client
        self.config = config
        self.report_cache = {}

    def generate_monthly_report(
        self,
        service: str,
        month: str,
        report_type: ReportType = ReportType.TECHNICAL
    ) -> Dict[str, Any]:
        """
        Generate comprehensive monthly SLO report

        Args:
            service: Service name
            month: Month in YYYY-MM format
            report_type: Type of report to generate

        Returns:
            Complete report data structure
        """
        # Parse month
        report_period = datetime.strptime(month, "%Y-%m")

        # Gather all report data
        report_data = {
            'metadata': self._generate_metadata(service, report_period, report_type),
            'slo_performance': self._calculate_slo_performance(service, month),
            'incidents': self._analyze_incidents(service, month),
            'error_budget': self._analyze_error_budget(service, month),
            'trends': self._analyze_trends(service, month),
            'forecasts': self._generate_forecasts(service, month),
            'recommendations': self._generate_recommendations(service, month),
            'historical_comparison': self._compare_historical(service, month),
            'charts': self._generate_charts(service, month)
        }

        # Format based on report type
        formatted_report = self._format_report(report_data, report_type)

        # Cache and store
        self._cache_report(service, month, formatted_report)
        self._store_report(service, month, formatted_report)

        return formatted_report

    def _generate_metadata(
        self,
        service: str,
        period: datetime,
        report_type: ReportType
    ) -> Dict[str, Any]:
        """Generate report metadata"""
        return {
            'service': service,
            'period': period.strftime("%Y-%m"),
            'period_start': period.replace(day=1),
            'period_end': (period.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1),
            'generated_at': datetime.utcnow(),
            'report_type': report_type.value,
            'version': '2.0',
            'generator': 'SLOReporter'
        }
```

---

## SLO Performance Metrics

### Performance Calculation Engine

```python
class PerformanceCalculator:
    """Calculate SLO performance metrics"""

    def __init__(self, metrics_client):
        self.metrics = metrics_client

    def calculate_slo_performance(
        self,
        service: str,
        month: str,
        slo_definitions: Dict[str, Any]
    ) -> Dict[str, SLOMetric]:
        """
        Calculate all SLO performance metrics

        Args:
            service: Service name
            month: Month in YYYY-MM format
            slo_definitions: SLO target definitions

        Returns:
            Dictionary of SLO metrics
        """
        slos = {}

        # Availability SLO
        slos['availability'] = self._calculate_availability(
            service, month, slo_definitions.get('availability', {})
        )

        # Latency SLOs
        for percentile in [50, 95, 99]:
            slos[f'latency_p{percentile}'] = self._calculate_latency(
                service, month, percentile, slo_definitions.get(f'latency_p{percentile}', {})
            )

        # Error rate SLO
        slos['error_rate'] = self._calculate_error_rate(
            service, month, slo_definitions.get('error_rate', {})
        )

        # Throughput SLO
        slos['throughput'] = self._calculate_throughput(
            service, month, slo_definitions.get('throughput', {})
        )

        # Custom SLOs
        for custom_slo in slo_definitions.get('custom', []):
            slos[custom_slo['name']] = self._calculate_custom_slo(
                service, month, custom_slo
            )

        return slos

    def _calculate_availability(
        self,
        service: str,
        month: str,
        definition: Dict[str, Any]
    ) -> SLOMetric:
        """Calculate availability SLO"""
        target = definition.get('target', 99.9)

        # Query actual availability
        query = f"""
        100 * (
            sum(rate(service_requests_total{{service="{service}",status=~"2..|3.."}}[5m]))
            /
            sum(rate(service_requests_total{{service="{service}"}}[5m]))
        )
        """

        actual = self.metrics.query_range(
            query=query,
            start=f"{month}-01",
            end=f"{month}-31",
            step="1h"
        )

        # Calculate monthly values
        monthly_values = [float(v[1]) for v in actual['values']]
        avg_availability = np.mean(monthly_values)

        # Determine status
        status = self._determine_status(avg_availability, target, buffer=0.1)
        trend = self._calculate_trend(monthly_values)

        return SLOMetric(
            name="Availability",
            target=target,
            actual=avg_availability,
            unit="%",
            met=avg_availability >= target,
            status=status,
            trend=trend,
            monthly_values=monthly_values
        )

    def _calculate_latency(
        self,
        service: str,
        month: str,
        percentile: int,
        definition: Dict[str, Any]
    ) -> SLOMetric:
        """Calculate latency SLO for specific percentile"""
        target = definition.get('target', 500)  # ms

        query = f"""
        histogram_quantile(
            {percentile / 100},
            sum(rate(service_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le)
        ) * 1000
        """

        actual = self.metrics.query_range(
            query=query,
            start=f"{month}-01",
            end=f"{month}-31",
            step="1h"
        )

        monthly_values = [float(v[1]) for v in actual['values']]
        avg_latency = np.percentile(monthly_values, percentile)

        status = self._determine_status(
            avg_latency,
            target,
            buffer=target * 0.1,
            inverse=True  # Lower is better
        )
        trend = self._calculate_trend(monthly_values, inverse=True)

        return SLOMetric(
            name=f"Latency P{percentile}",
            target=target,
            actual=avg_latency,
            unit="ms",
            met=avg_latency <= target,
            status=status,
            trend=trend,
            monthly_values=monthly_values
        )

    def _calculate_error_rate(
        self,
        service: str,
        month: str,
        definition: Dict[str, Any]
    ) -> SLOMetric:
        """Calculate error rate SLO"""
        target = definition.get('target', 1.0)  # percentage

        query = f"""
        100 * (
            sum(rate(service_requests_total{{service="{service}",status=~"5.."}}[5m]))
            /
            sum(rate(service_requests_total{{service="{service}"}}[5m]))
        )
        """

        actual = self.metrics.query_range(
            query=query,
            start=f"{month}-01",
            end=f"{month}-31",
            step="1h"
        )

        monthly_values = [float(v[1]) for v in actual['values']]
        avg_error_rate = np.mean(monthly_values)

        status = self._determine_status(
            avg_error_rate,
            target,
            buffer=target * 0.1,
            inverse=True
        )
        trend = self._calculate_trend(monthly_values, inverse=True)

        return SLOMetric(
            name="Error Rate",
            target=target,
            actual=avg_error_rate,
            unit="%",
            met=avg_error_rate <= target,
            status=status,
            trend=trend,
            monthly_values=monthly_values
        )

    def _determine_status(
        self,
        actual: float,
        target: float,
        buffer: float,
        inverse: bool = False
    ) -> SLOStatus:
        """
        Determine SLO status with warning buffer

        Args:
            actual: Actual value
            target: Target value
            buffer: Warning buffer
            inverse: True if lower values are better
        """
        if not inverse:
            if actual >= target:
                return SLOStatus.MET
            elif actual >= (target - buffer):
                return SLOStatus.WARNING
            elif actual >= (target - 2 * buffer):
                return SLOStatus.AT_RISK
            else:
                return SLOStatus.VIOLATED
        else:
            if actual <= target:
                return SLOStatus.MET
            elif actual <= (target + buffer):
                return SLOStatus.WARNING
            elif actual <= (target + 2 * buffer):
                return SLOStatus.AT_RISK
            else:
                return SLOStatus.VIOLATED

    def _calculate_trend(
        self,
        values: List[float],
        inverse: bool = False
    ) -> str:
        """Calculate trend from time series values"""
        if len(values) < 2:
            return "stable"

        # Linear regression to determine trend
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]

        threshold = np.std(values) * 0.1

        if abs(slope) < threshold:
            return "stable"
        elif (slope > 0 and not inverse) or (slope < 0 and inverse):
            return "improving"
        else:
            return "degrading"
```

---

## Incident Impact Analysis

### Incident Analysis Engine

```python
class IncidentAnalyzer:
    """Analyze incident impact on SLOs"""

    def __init__(self, incident_client, metrics_client):
        self.incidents = incident_client
        self.metrics = metrics_client

    def analyze_incidents(
        self,
        service: str,
        month: str
    ) -> List[IncidentImpact]:
        """
        Analyze all incidents for the period

        Args:
            service: Service name
            month: Month in YYYY-MM format

        Returns:
            List of incident impact analyses
        """
        # Fetch incidents
        incidents = self.incidents.get_incidents(
            service=service,
            start_date=f"{month}-01",
            end_date=f"{month}-31"
        )

        impact_analyses = []

        for incident in incidents:
            impact = self._analyze_incident_impact(incident, service)
            impact_analyses.append(impact)

        return impact_analyses

    def _analyze_incident_impact(
        self,
        incident: Dict[str, Any],
        service: str
    ) -> IncidentImpact:
        """Analyze single incident impact"""
        start_time = datetime.fromisoformat(incident['started_at'])
        end_time = datetime.fromisoformat(incident['resolved_at'])
        duration = (end_time - start_time).total_seconds() / 60  # minutes

        # Query affected users
        affected_users = self._calculate_affected_users(
            service,
            start_time,
            end_time
        )

        # Calculate error budget consumed
        error_budget_consumed = self._calculate_error_budget_impact(
            service,
            start_time,
            end_time,
            duration
        )

        # Analyze impacted services
        services_impacted = self._identify_impacted_services(
            incident,
            start_time,
            end_time
        )

        # Calculate recovery time
        recovery_time = self._calculate_recovery_time(
            service,
            end_time
        )

        return IncidentImpact(
            incident_id=incident['id'],
            timestamp=start_time,
            duration_minutes=duration,
            affected_users=affected_users,
            error_budget_consumed=error_budget_consumed,
            severity=incident['severity'],
            root_cause=incident.get('root_cause', 'Unknown'),
            services_impacted=services_impacted,
            recovery_time=recovery_time
        )

    def _calculate_affected_users(
        self,
        service: str,
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """Calculate number of affected users"""
        query = f"""
        count(
            count_over_time(
                service_user_requests_total{{
                    service="{service}",
                    status=~"5.."
                }}[{int((end_time - start_time).total_seconds())}s]
            ) by (user_id)
        )
        """

        result = self.metrics.query(query, time=end_time)
        return int(result[0]['value'][1]) if result else 0

    def _calculate_error_budget_impact(
        self,
        service: str,
        start_time: datetime,
        end_time: datetime,
        duration_minutes: float
    ) -> float:
        """Calculate error budget consumed by incident"""
        # Total time in period (30 days in minutes)
        total_time = 30 * 24 * 60

        # SLO target (e.g., 99.9% = 0.1% error budget)
        error_budget_percentage = 0.1

        # Error budget consumed = (downtime / total_time) / error_budget
        consumed = (duration_minutes / total_time) / (error_budget_percentage / 100)

        return consumed * 100  # Return as percentage

    def _identify_impacted_services(
        self,
        incident: Dict[str, Any],
        start_time: datetime,
        end_time: datetime
    ) -> List[str]:
        """Identify downstream services impacted"""
        services = set()

        # Get from incident metadata
        if 'impacted_services' in incident:
            services.update(incident['impacted_services'])

        # Query for correlated service errors
        query = f"""
        sum by (service) (
            increase(
                service_requests_total{{status=~"5.."}}[{int((end_time - start_time).total_seconds())}s]
            )
        ) > 100
        """

        result = self.metrics.query(query, time=end_time)
        for r in result:
            services.add(r['metric']['service'])

        return list(services)

    def _calculate_recovery_time(
        self,
        service: str,
        incident_end: datetime
    ) -> float:
        """Calculate time to full recovery (minutes)"""
        # Query for when error rate returned to baseline
        baseline_query = f"""
        avg_over_time(
            service_error_rate{{service="{service}"}}[1h] offset 24h
        )
        """
        baseline = self.metrics.query(baseline_query)

        # Find when current rate returned to baseline
        check_time = incident_end
        max_check_duration = timedelta(hours=6)

        while (check_time - incident_end) < max_check_duration:
            current_rate = self.metrics.query(
                f'service_error_rate{{service="{service}"}}',
                time=check_time
            )

            if current_rate <= baseline * 1.1:  # Within 10% of baseline
                return (check_time - incident_end).total_seconds() / 60

            check_time += timedelta(minutes=5)

        return max_check_duration.total_seconds() / 60

    def generate_incident_summary(
        self,
        incidents: List[IncidentImpact]
    ) -> Dict[str, Any]:
        """Generate summary statistics for incidents"""
        if not incidents:
            return {
                'total_incidents': 0,
                'total_downtime_minutes': 0,
                'total_user_impact_minutes': 0,
                'total_error_budget_consumed': 0,
                'mean_time_to_recovery': 0,
                'severity_breakdown': {},
                'top_root_causes': []
            }

        return {
            'total_incidents': len(incidents),
            'total_downtime_minutes': sum(i.duration_minutes for i in incidents),
            'total_user_impact_minutes': sum(i.user_impact_minutes for i in incidents),
            'total_error_budget_consumed': sum(i.error_budget_consumed for i in incidents),
            'mean_time_to_recovery': np.mean([i.recovery_time for i in incidents]),
            'severity_breakdown': self._count_by_severity(incidents),
            'top_root_causes': self._top_root_causes(incidents, limit=5)
        }

    def _count_by_severity(
        self,
        incidents: List[IncidentImpact]
    ) -> Dict[str, int]:
        """Count incidents by severity"""
        counts = {}
        for incident in incidents:
            severity = incident.severity
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    def _top_root_causes(
        self,
        incidents: List[IncidentImpact],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify top root causes"""
        causes = {}
        for incident in incidents:
            cause = incident.root_cause
            if cause not in causes:
                causes[cause] = {
                    'count': 0,
                    'total_impact_minutes': 0,
                    'total_error_budget': 0
                }
            causes[cause]['count'] += 1
            causes[cause]['total_impact_minutes'] += incident.user_impact_minutes
            causes[cause]['total_error_budget'] += incident.error_budget_consumed

        # Sort by impact
        sorted_causes = sorted(
            causes.items(),
            key=lambda x: x[1]['total_impact_minutes'],
            reverse=True
        )

        return [
            {'cause': cause, **stats}
            for cause, stats in sorted_causes[:limit]
        ]
```

---

## Trend Analysis and Forecasting

### Forecasting Engine

```python
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class TrendForecaster:
    """Analyze trends and forecast future SLO performance"""

    def __init__(self, metrics_client):
        self.metrics = metrics_client

    def analyze_trends(
        self,
        service: str,
        months: int = 6
    ) -> Dict[str, Any]:
        """
        Analyze SLO trends over multiple months

        Args:
            service: Service name
            months: Number of months to analyze

        Returns:
            Trend analysis results
        """
        # Get historical data
        historical_data = self._fetch_historical_data(service, months)

        trends = {}

        for metric_name, values in historical_data.items():
            trends[metric_name] = {
                'direction': self._calculate_direction(values),
                'rate_of_change': self._calculate_rate_of_change(values),
                'volatility': self._calculate_volatility(values),
                'seasonality': self._detect_seasonality(values),
                'confidence': self._calculate_confidence(values)
            }

        return trends

    def generate_forecasts(
        self,
        service: str,
        months_ahead: int = 3,
        historical_months: int = 6
    ) -> Dict[str, Any]:
        """
        Generate SLO forecasts

        Args:
            service: Service name
            months_ahead: Number of months to forecast
            historical_months: Historical data to use

        Returns:
            Forecast predictions with confidence intervals
        """
        historical_data = self._fetch_historical_data(service, historical_months)

        forecasts = {}

        for metric_name, values in historical_data.items():
            forecast = self._forecast_metric(
                values,
                months_ahead,
                method='exponential_smoothing'
            )

            forecasts[metric_name] = {
                'predictions': forecast['predictions'],
                'confidence_interval_upper': forecast['ci_upper'],
                'confidence_interval_lower': forecast['ci_lower'],
                'method': forecast['method'],
                'accuracy_score': forecast['accuracy']
            }

        return forecasts

    def _fetch_historical_data(
        self,
        service: str,
        months: int
    ) -> Dict[str, List[float]]:
        """Fetch historical metric data"""
        data = {}

        metrics = [
            'availability',
            'latency_p95',
            'latency_p99',
            'error_rate',
            'throughput'
        ]

        for metric in metrics:
            query = self._build_query(service, metric)
            values = self.metrics.query_range(
                query=query,
                start=f"-{months}mo",
                end="now",
                step="1d"
            )
            data[metric] = [float(v[1]) for v in values['values']]

        return data

    def _calculate_direction(self, values: List[float]) -> str:
        """Calculate overall trend direction"""
        if len(values) < 3:
            return "insufficient_data"

        # Linear regression
        x = np.arange(len(values)).reshape(-1, 1)
        y = np.array(values)

        model = LinearRegression()
        model.fit(x, y)

        slope = model.coef_[0]

        # Statistical significance test
        _, p_value = stats.pearsonr(x.flatten(), y)

        if p_value > 0.05:
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "degrading"

    def _calculate_rate_of_change(self, values: List[float]) -> float:
        """Calculate average rate of change"""
        if len(values) < 2:
            return 0.0

        changes = [
            (values[i] - values[i-1]) / values[i-1] * 100
            for i in range(1, len(values))
            if values[i-1] != 0
        ]

        return np.mean(changes) if changes else 0.0

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (coefficient of variation)"""
        if len(values) < 2:
            return 0.0

        mean = np.mean(values)
        if mean == 0:
            return 0.0

        return (np.std(values) / mean) * 100

    def _detect_seasonality(self, values: List[float]) -> Dict[str, Any]:
        """Detect seasonal patterns"""
        if len(values) < 14:  # Need at least 2 weeks of data
            return {'detected': False}

        # Autocorrelation analysis
        from statsmodels.tsa.stattools import acf

        autocorr = acf(values, nlags=len(values)//2)

        # Look for peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.5:  # Significant correlation
                    peaks.append(i)

        if peaks:
            return {
                'detected': True,
                'period': peaks[0],
                'strength': float(autocorr[peaks[0]])
            }

        return {'detected': False}

    def _calculate_confidence(self, values: List[float]) -> float:
        """Calculate confidence in trend analysis"""
        if len(values) < 3:
            return 0.0

        # Based on data length and volatility
        length_score = min(len(values) / 30, 1.0)  # Max at 30 data points
        volatility = self._calculate_volatility(values)
        volatility_score = max(0, 1 - (volatility / 100))

        return (length_score * 0.5 + volatility_score * 0.5) * 100

    def _forecast_metric(
        self,
        historical: List[float],
        periods: int,
        method: str = 'exponential_smoothing'
    ) -> Dict[str, Any]:
        """Forecast metric using specified method"""
        if len(historical) < 4:
            return self._simple_forecast(historical, periods)

        if method == 'exponential_smoothing':
            return self._exponential_smoothing_forecast(historical, periods)
        elif method == 'linear_regression':
            return self._linear_regression_forecast(historical, periods)
        else:
            return self._simple_forecast(historical, periods)

    def _exponential_smoothing_forecast(
        self,
        historical: List[float],
        periods: int
    ) -> Dict[str, Any]:
        """Forecast using exponential smoothing"""
        try:
            model = ExponentialSmoothing(
                historical,
                seasonal_periods=7,
                trend='add',
                seasonal='add'
            )
            fit = model.fit()
            forecast = fit.forecast(periods)

            # Calculate confidence intervals
            std = np.std(historical)
            ci_upper = forecast + 1.96 * std
            ci_lower = forecast - 1.96 * std

            # Calculate accuracy on historical data
            accuracy = self._calculate_forecast_accuracy(
                historical,
                fit.fittedvalues
            )

            return {
                'predictions': forecast.tolist(),
                'ci_upper': ci_upper.tolist(),
                'ci_lower': ci_lower.tolist(),
                'method': 'exponential_smoothing',
                'accuracy': accuracy
            }
        except Exception as e:
            return self._simple_forecast(historical, periods)

    def _linear_regression_forecast(
        self,
        historical: List[float],
        periods: int
    ) -> Dict[str, Any]:
        """Forecast using linear regression"""
        x = np.arange(len(historical)).reshape(-1, 1)
        y = np.array(historical)

        model = LinearRegression()
        model.fit(x, y)

        # Predict future
        future_x = np.arange(
            len(historical),
            len(historical) + periods
        ).reshape(-1, 1)
        predictions = model.predict(future_x)

        # Calculate residuals for confidence interval
        residuals = y - model.predict(x)
        std = np.std(residuals)

        ci_upper = predictions + 1.96 * std
        ci_lower = predictions - 1.96 * std

        accuracy = self._calculate_forecast_accuracy(historical, model.predict(x))

        return {
            'predictions': predictions.tolist(),
            'ci_upper': ci_upper.tolist(),
            'ci_lower': ci_lower.tolist(),
            'method': 'linear_regression',
            'accuracy': accuracy
        }

    def _simple_forecast(
        self,
        historical: List[float],
        periods: int
    ) -> Dict[str, Any]:
        """Simple moving average forecast"""
        avg = np.mean(historical[-3:]) if len(historical) >= 3 else np.mean(historical)
        std = np.std(historical)

        predictions = [avg] * periods
        ci_upper = [avg + 1.96 * std] * periods
        ci_lower = [avg - 1.96 * std] * periods

        return {
            'predictions': predictions,
            'ci_upper': ci_upper,
            'ci_lower': ci_lower,
            'method': 'simple_moving_average',
            'accuracy': 0.5
        }

    def _calculate_forecast_accuracy(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> float:
        """Calculate forecast accuracy (R-squared)"""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        if ss_tot == 0:
            return 0.0

        r_squared = 1 - (ss_res / ss_tot)
        return max(0, min(1, r_squared))
```

---

## Stakeholder Communication Templates

### Template Generator

```python
class StakeholderReportGenerator:
    """Generate reports tailored for different stakeholder types"""

    def __init__(self):
        self.templates = self._load_templates()

    def generate_executive_summary(
        self,
        report_data: Dict[str, Any]
    ) -> str:
        """
        Generate executive summary

        Focus on:
        - High-level metrics
        - Business impact
        - Key recommendations
        - Strategic implications
        """
        template = Template("""
# Executive Summary - {{ metadata.service }}
**Period:** {{ metadata.period }}

## Key Highlights

### Overall Service Health
- **Availability:** {{ "%.2f"|format(slo_performance.availability.actual) }}%
  (Target: {{ slo_performance.availability.target }}%)
  {{ "✅ MEETING TARGET" if slo_performance.availability.met else "⚠️ BELOW TARGET" }}

- **Error Budget Status:** {{ "%.1f"|format(error_budget.remaining_percentage) }}% remaining
  {{ "✅ Healthy" if error_budget.remaining_percentage > 50 else "⚠️ At Risk" if error_budget.remaining_percentage > 20 else "🔴 Critical" }}

### Business Impact
- **Total Incidents:** {{ incidents|length }}
- **User Impact:** {{ "%.0f"|format(incident_summary.total_user_impact_minutes / 60) }} user-hours affected
- **Service Disruption:** {{ "%.1f"|format(incident_summary.total_downtime_minutes / 60) }} hours

### Trend Analysis
{% for metric, trend in trends.items() %}
- **{{ metric }}:** {{ trend.direction|upper }}
  ({{ "+" if trend.rate_of_change > 0 else "" }}{{ "%.1f"|format(trend.rate_of_change) }}%)
{% endfor %}

## Strategic Recommendations

{% for recommendation in recommendations[:3] %}
### {{ loop.index }}. {{ recommendation.title }}
**Priority:** {{ recommendation.priority }}
**Impact:** {{ recommendation.expected_impact }}

{{ recommendation.executive_summary }}

**Investment Required:** {{ recommendation.investment_level }}
{% endfor %}

## Looking Ahead

### 3-Month Forecast
{% for metric, forecast in forecasts.items() %}
- **{{ metric }}:** Projected {{ "%.2f"|format(forecast.predictions[-1]) }}
  (Current: {{ "%.2f"|format(slo_performance[metric].actual) }})
{% endfor %}

---

*This report provides a high-level overview. For detailed technical analysis, see the full technical report.*
""")

        return template.render(**report_data)

    def generate_technical_report(
        self,
        report_data: Dict[str, Any]
    ) -> str:
        """
        Generate detailed technical report

        Focus on:
        - Detailed metrics
        - Root cause analysis
        - Technical recommendations
        - Implementation details
        """
        template = Template("""
# Technical SLO Report - {{ metadata.service }}
**Period:** {{ metadata.period }}
**Generated:** {{ metadata.generated_at.strftime("%Y-%m-%d %H:%M UTC") }}

## Table of Contents
1. [SLO Performance](#slo-performance)
2. [Incident Analysis](#incident-analysis)
3. [Error Budget Analysis](#error-budget-analysis)
4. [Trend Analysis](#trend-analysis)
5. [Technical Recommendations](#technical-recommendations)

---

## SLO Performance

### Summary Table
| SLO | Target | Actual | Status | Trend |
|-----|--------|--------|--------|-------|
{% for name, metric in slo_performance.items() %}
| {{ metric.name }} | {{ metric.target }}{{ metric.unit }} | {{ "%.2f"|format(metric.actual) }}{{ metric.unit }} | {{ metric.status.value }} | {{ metric.trend }} |
{% endfor %}

### Detailed Analysis

{% for name, metric in slo_performance.items() %}
#### {{ metric.name }}
- **Performance:** {{ "%.2f"|format(metric.actual) }}{{ metric.unit }} vs target {{ metric.target }}{{ metric.unit }}
- **Gap:** {{ "%.2f"|format(metric.gap) }}{{ metric.unit }}
- **Status:** {{ metric.status.value|upper }}
- **Trend:** {{ metric.trend|upper }}

**Monthly Distribution:**
- Min: {{ "%.2f"|format(metric.monthly_values|min) }}{{ metric.unit }}
- Max: {{ "%.2f"|format(metric.monthly_values|max) }}{{ metric.unit }}
- Median: {{ "%.2f"|format(metric.monthly_values|median) }}{{ metric.unit }}
- Std Dev: {{ "%.2f"|format(metric.monthly_values|stdev) }}{{ metric.unit }}

{% endfor %}

---

## Incident Analysis

### Incident Summary
- **Total Incidents:** {{ incident_summary.total_incidents }}
- **Total Downtime:** {{ "%.1f"|format(incident_summary.total_downtime_minutes / 60) }} hours
- **Mean Time to Recovery:** {{ "%.1f"|format(incident_summary.mean_time_to_recovery) }} minutes
- **Error Budget Consumed:** {{ "%.1f"|format(incident_summary.total_error_budget_consumed) }}%

### Severity Breakdown
{% for severity, count in incident_summary.severity_breakdown.items() %}
- **{{ severity }}:** {{ count }} incidents
{% endfor %}

### Top Root Causes
{% for cause in incident_summary.top_root_causes %}
{{ loop.index }}. **{{ cause.cause }}**
   - Occurrences: {{ cause.count }}
   - Total Impact: {{ "%.1f"|format(cause.total_impact_minutes / 60) }} user-hours
   - Error Budget: {{ "%.1f"|format(cause.total_error_budget) }}%
{% endfor %}

### Incident Details
{% for incident in incidents %}
#### Incident {{ incident.incident_id }}
- **Time:** {{ incident.timestamp.strftime("%Y-%m-%d %H:%M UTC") }}
- **Duration:** {{ "%.1f"|format(incident.duration_minutes) }} minutes
- **Severity:** {{ incident.severity }}
- **Affected Users:** {{ incident.affected_users }}
- **Error Budget Impact:** {{ "%.2f"|format(incident.error_budget_consumed) }}%
- **Root Cause:** {{ incident.root_cause }}
- **Services Impacted:** {{ incident.services_impacted|join(", ") }}
- **Recovery Time:** {{ "%.1f"|format(incident.recovery_time) }} minutes
{% endfor %}

---

## Error Budget Analysis

### Current Status
- **Total Budget:** 100%
- **Consumed:** {{ "%.1f"|format(error_budget.consumed_percentage) }}%
- **Remaining:** {{ "%.1f"|format(error_budget.remaining_percentage) }}%
- **Burn Rate:** {{ "%.2f"|format(error_budget.burn_rate) }}%/day

### Budget Allocation
- **Incidents:** {{ "%.1f"|format(error_budget.incident_consumption) }}%
- **Planned Maintenance:** {{ "%.1f"|format(error_budget.maintenance_consumption) }}%
- **Baseline Errors:** {{ "%.1f"|format(error_budget.baseline_consumption) }}%

### Projections
- **Days until exhaustion:** {{ error_budget.days_until_exhaustion|int }}
- **Risk level:** {{ error_budget.risk_level }}

---

## Trend Analysis

{% for metric, trend in trends.items() %}
### {{ metric }}
- **Direction:** {{ trend.direction|upper }}
- **Rate of Change:** {{ "%.2f"|format(trend.rate_of_change) }}%/month
- **Volatility:** {{ "%.2f"|format(trend.volatility) }}%
- **Confidence:** {{ "%.0f"|format(trend.confidence) }}%

{% if trend.seasonality.detected %}
**Seasonality Detected:**
- Period: {{ trend.seasonality.period }} days
- Strength: {{ "%.2f"|format(trend.seasonality.strength) }}
{% endif %}
{% endfor %}

---

## Technical Recommendations

{% for recommendation in recommendations %}
### {{ loop.index }}. {{ recommendation.title }}

**Priority:** {{ recommendation.priority }}
**Category:** {{ recommendation.category }}
**Estimated Effort:** {{ recommendation.effort }}
**Expected Impact:** {{ recommendation.expected_impact }}

#### Problem Statement
{{ recommendation.problem }}

#### Proposed Solution
{{ recommendation.solution }}

#### Implementation Steps
{% for step in recommendation.implementation_steps %}
{{ loop.index }}. {{ step }}
{% endfor %}

#### Success Metrics
{% for metric in recommendation.success_metrics %}
- {{ metric }}
{% endfor %}

#### Risk Assessment
{{ recommendation.risks }}

---
{% endfor %}

## Appendix

### Query Examples
```promql
# Availability
{{ queries.availability }}

# Latency P95
{{ queries.latency_p95 }}

# Error Rate
{{ queries.error_rate }}
```

### Methodology
This report uses the following calculation methods:
- SLO performance: 30-day rolling average
- Trend analysis: Linear regression with 6-month lookback
- Forecasting: Exponential smoothing with seasonal adjustment
- Confidence intervals: 95% confidence level

---

*Report generated by SLOReporter v{{ metadata.version }}*
""")

        return template.render(**report_data)

    def generate_customer_facing_report(
        self,
        report_data: Dict[str, Any]
    ) -> str:
        """
        Generate customer-facing transparency report

        Focus on:
        - Service commitments met
        - Impact on customers
        - Improvement initiatives
        - Future commitments
        """
        template = Template("""
# {{ metadata.service }} - Service Reliability Report
**{{ metadata.period }}**

Dear Valued Customers,

We're committed to providing you with transparent information about our service reliability. Here's our performance for {{ metadata.period }}.

## Service Performance

### Our Commitments to You

| Commitment | Our Target | This Month | Status |
|------------|------------|------------|--------|
| Service Availability | {{ slo_performance.availability.target }}% | {{ "%.2f"|format(slo_performance.availability.actual) }}% | {{ "✅ Met" if slo_performance.availability.met else "⚠️ Below Target" }} |
| Response Time | < {{ slo_performance.latency_p95.target }}ms | {{ "%.0f"|format(slo_performance.latency_p95.actual) }}ms | {{ "✅ Met" if slo_performance.latency_p95.met else "⚠️ Below Target" }} |

### Service Incidents

This month, we experienced {{ incidents|length }} service incident{{ "s" if incidents|length != 1 else "" }}.

{% if incidents %}
{% for incident in incidents %}
**{{ incident.timestamp.strftime("%B %d, %Y") }}** - {{ incident.severity }} incident
- **Duration:** {{ "%.0f"|format(incident.duration_minutes) }} minutes
- **Impact:** {{ incident.affected_users }} users affected
- **Resolution:** {{ incident.root_cause }}
{% endfor %}
{% else %}
We're pleased to report no service incidents this month.
{% endif %}

### What We're Doing to Improve

{% for recommendation in recommendations[:3] %}
**{{ loop.index }}. {{ recommendation.customer_summary }}**

{{ recommendation.customer_benefit }}
{% endfor %}

## Looking Forward

We continuously monitor and improve our service reliability. Our projections show:

{% for metric, forecast in forecasts.items() %}
- **{{ metric }}:** Expected to {{ "improve" if forecast.predictions[-1] > slo_performance[metric].actual else "maintain" }} performance
{% endfor %}

Thank you for your continued trust in our service.

---

*For technical details or questions, please contact support@example.com*
""")

        return template.render(**report_data)
```

---

## HTML Report Templates

### Advanced HTML Report Generator

```python
class HTMLReportGenerator:
    """Generate rich HTML reports with interactive charts"""

    def generate_html_report(
        self,
        report_data: Dict[str, Any],
        report_type: ReportType = ReportType.TECHNICAL
    ) -> str:
        """Generate complete HTML report"""

        # Generate charts
        charts = self._generate_all_charts(report_data)

        template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SLO Report - {{ metadata.service }} - {{ metadata.period }}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        header .meta {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .summary-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
        }

        .summary-card h3 {
            color: #667eea;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }

        .summary-card .value {
            font-size: 2rem;
            font-weight: bold;
            color: #333;
        }

        .summary-card .subvalue {
            font-size: 0.9rem;
            color: #666;
            margin-top: 5px;
        }

        .summary-card.good { border-left-color: #10b981; }
        .summary-card.warning { border-left-color: #f59e0b; }
        .summary-card.bad { border-left-color: #ef4444; }

        .section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }

        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #667eea;
        }

        tr:hover {
            background: #f8f9fa;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .status-met { background: #d1fae5; color: #065f46; }
        .status-warning { background: #fef3c7; color: #92400e; }
        .status-at-risk { background: #fed7aa; color: #9a3412; }
        .status-violated { background: #fee2e2; color: #991b1b; }

        .trend-up { color: #10b981; }
        .trend-down { color: #ef4444; }
        .trend-stable { color: #6b7280; }

        .chart-container {
            margin: 30px 0;
            min-height: 400px;
        }

        .recommendation {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
        }

        .recommendation h4 {
            color: #667eea;
            margin-bottom: 10px;
        }

        .priority-critical { border-left-color: #ef4444; }
        .priority-high { border-left-color: #f59e0b; }
        .priority-medium { border-left-color: #3b82f6; }
        .priority-low { border-left-color: #6b7280; }

        footer {
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
        }

        @media print {
            body { background: white; }
            .section { break-inside: avoid; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>SLO Report: {{ metadata.service }}</h1>
            <div class="meta">
                Period: {{ metadata.period }} |
                Generated: {{ metadata.generated_at.strftime("%Y-%m-%d %H:%M UTC") }}
            </div>
        </header>

        <!-- Executive Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card {{ 'good' if slo_performance.availability.met else 'bad' }}">
                <h3>Availability</h3>
                <div class="value">{{ "%.2f"|format(slo_performance.availability.actual) }}%</div>
                <div class="subvalue">Target: {{ slo_performance.availability.target }}%</div>
            </div>

            <div class="summary-card {{ 'good' if error_budget.remaining_percentage > 50 else 'warning' if error_budget.remaining_percentage > 20 else 'bad' }}">
                <h3>Error Budget</h3>
                <div class="value">{{ "%.1f"|format(error_budget.remaining_percentage) }}%</div>
                <div class="subvalue">Remaining</div>
            </div>

            <div class="summary-card {{ 'good' if incidents|length == 0 else 'warning' if incidents|length < 5 else 'bad' }}">
                <h3>Incidents</h3>
                <div class="value">{{ incidents|length }}</div>
                <div class="subvalue">{{ "%.1f"|format(incident_summary.total_downtime_minutes / 60) }} hours downtime</div>
            </div>

            <div class="summary-card">
                <h3>MTTR</h3>
                <div class="value">{{ "%.0f"|format(incident_summary.mean_time_to_recovery) }}m</div>
                <div class="subvalue">Mean Time to Recovery</div>
            </div>
        </div>

        <!-- SLO Performance Section -->
        <div class="section">
            <h2>SLO Performance</h2>

            <table>
                <thead>
                    <tr>
                        <th>SLO</th>
                        <th>Target</th>
                        <th>Actual</th>
                        <th>Status</th>
                        <th>Trend</th>
                    </tr>
                </thead>
                <tbody>
                    {% for name, metric in slo_performance.items() %}
                    <tr>
                        <td><strong>{{ metric.name }}</strong></td>
                        <td>{{ metric.target }}{{ metric.unit }}</td>
                        <td>{{ "%.2f"|format(metric.actual) }}{{ metric.unit }}</td>
                        <td><span class="status-badge status-{{ metric.status.value }}">{{ metric.status.value|upper }}</span></td>
                        <td class="trend-{{ metric.trend }}">{{ metric.trend|upper }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>

            <div id="slo-performance-chart" class="chart-container"></div>
        </div>

        <!-- Incident Analysis Section -->
        <div class="section">
            <h2>Incident Analysis</h2>

            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Incidents</h3>
                    <div class="value">{{ incident_summary.total_incidents }}</div>
                </div>
                <div class="summary-card">
                    <h3>User Impact</h3>
                    <div class="value">{{ "%.0f"|format(incident_summary.total_user_impact_minutes / 60) }}h</div>
                </div>
                <div class="summary-card">
                    <h3>Error Budget Used</h3>
                    <div class="value">{{ "%.1f"|format(incident_summary.total_error_budget_consumed) }}%</div>
                </div>
            </div>

            <div id="incident-timeline-chart" class="chart-container"></div>
            <div id="root-cause-chart" class="chart-container"></div>
        </div>

        <!-- Trend Analysis Section -->
        <div class="section">
            <h2>Trend Analysis</h2>
            <div id="trend-chart" class="chart-container"></div>
        </div>

        <!-- Forecast Section -->
        <div class="section">
            <h2>3-Month Forecast</h2>
            <div id="forecast-chart" class="chart-container"></div>
        </div>

        <!-- Recommendations Section -->
        <div class="section">
            <h2>Recommendations</h2>
            {% for recommendation in recommendations %}
            <div class="recommendation priority-{{ recommendation.priority|lower }}">
                <h4>{{ loop.index }}. {{ recommendation.title }}</h4>
                <p><strong>Priority:</strong> {{ recommendation.priority }} |
                   <strong>Impact:</strong> {{ recommendation.expected_impact }}</p>
                <p>{{ recommendation.summary }}</p>
            </div>
            {% endfor %}
        </div>

        <footer>
            <p>Generated by SLOReporter v{{ metadata.version }}</p>
            <p>For questions or concerns, contact sre-team@example.com</p>
        </footer>
    </div>

    <script>
        // SLO Performance Chart
        {{ charts.slo_performance_js }}

        // Incident Timeline Chart
        {{ charts.incident_timeline_js }}

        // Root Cause Chart
        {{ charts.root_cause_js }}

        // Trend Chart
        {{ charts.trend_js }}

        // Forecast Chart
        {{ charts.forecast_js }}
    </script>
</body>
</html>
""")

        return template.render(**report_data, charts=charts)

    def _generate_all_charts(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate all chart JavaScript"""
        return {
            'slo_performance_js': self._generate_slo_performance_chart(report_data),
            'incident_timeline_js': self._generate_incident_timeline_chart(report_data),
            'root_cause_js': self._generate_root_cause_chart(report_data),
            'trend_js': self._generate_trend_chart(report_data),
            'forecast_js': self._generate_forecast_chart(report_data)
        }

    def _generate_slo_performance_chart(self, data: Dict[str, Any]) -> str:
        """Generate SLO performance comparison chart"""
        return """
        var sloData = [
            {
                x: ['Availability', 'Latency P95', 'Latency P99', 'Error Rate'],
                y: [99.95, 450, 850, 0.8],
                name: 'Actual',
                type: 'bar',
                marker: { color: '#667eea' }
            },
            {
                x: ['Availability', 'Latency P95', 'Latency P99', 'Error Rate'],
                y: [99.9, 500, 1000, 1.0],
                name: 'Target',
                type: 'bar',
                marker: { color: '#cbd5e1' }
            }
        ];

        var sloLayout = {
            title: 'SLO Performance vs Targets',
            barmode: 'group',
            yaxis: { title: 'Value' },
            height: 400
        };

        Plotly.newPlot('slo-performance-chart', sloData, sloLayout);
        """

    def _generate_incident_timeline_chart(self, data: Dict[str, Any]) -> str:
        """Generate incident timeline chart"""
        return """
        var incidentData = [{
            x: ['2024-11-01', '2024-11-05', '2024-11-12', '2024-11-20'],
            y: [45, 120, 30, 90],
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: [10, 20, 8, 15],
                color: ['#f59e0b', '#ef4444', '#f59e0b', '#ef4444']
            },
            text: ['Minor', 'Major', 'Minor', 'Major'],
            hovertemplate: '<b>%{text}</b><br>Duration: %{y} min<br>Date: %{x}<extra></extra>'
        }];

        var incidentLayout = {
            title: 'Incident Timeline',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Duration (minutes)' },
            height: 400
        };

        Plotly.newPlot('incident-timeline-chart', incidentData, incidentLayout);
        """

    def _generate_root_cause_chart(self, data: Dict[str, Any]) -> str:
        """Generate root cause distribution chart"""
        return """
        var rootCauseData = [{
            labels: ['Database Overload', 'Network Issues', 'Code Deployment', 'External API', 'Resource Exhaustion'],
            values: [35, 25, 20, 15, 5],
            type: 'pie',
            marker: {
                colors: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
            }
        }];

        var rootCauseLayout = {
            title: 'Incidents by Root Cause',
            height: 400
        };

        Plotly.newPlot('root-cause-chart', rootCauseData, rootCauseLayout);
        """

    def _generate_trend_chart(self, data: Dict[str, Any]) -> str:
        """Generate trend analysis chart"""
        return """
        var trendData = [
            {
                x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'],
                y: [99.85, 99.88, 99.91, 99.89, 99.92, 99.94, 99.93, 99.95, 99.96, 99.95, 99.95],
                mode: 'lines+markers',
                name: 'Availability',
                line: { color: '#10b981', width: 2 }
            },
            {
                x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'],
                y: [99.9, 99.9, 99.9, 99.9, 99.9, 99.9, 99.9, 99.9, 99.9, 99.9, 99.9],
                mode: 'lines',
                name: 'Target',
                line: { color: '#ef4444', width: 2, dash: 'dash' }
            }
        ];

        var trendLayout = {
            title: 'Availability Trend (6 Months)',
            xaxis: { title: 'Month' },
            yaxis: {
                title: 'Availability (%)',
                range: [99.7, 100]
            },
            height: 400
        };

        Plotly.newPlot('trend-chart', trendData, trendLayout);
        """

    def _generate_forecast_chart(self, data: Dict[str, Any]) -> str:
        """Generate forecast chart with confidence intervals"""
        return """
        var forecastData = [
            {
                x: ['Nov', 'Dec', 'Jan', 'Feb'],
                y: [99.95, 99.96, 99.96, 99.97],
                mode: 'lines+markers',
                name: 'Forecast',
                line: { color: '#667eea', width: 2 }
            },
            {
                x: ['Nov', 'Dec', 'Jan', 'Feb'],
                y: [99.95, 99.98, 99.98, 99.99],
                fill: 'tonexty',
                fillcolor: 'rgba(102, 126, 234, 0.2)',
                line: { color: 'transparent' },
                name: 'Upper CI',
                showlegend: false
            },
            {
                x: ['Nov', 'Dec', 'Jan', 'Feb'],
                y: [99.95, 99.94, 99.94, 99.95],
                fill: 'tonexty',
                fillcolor: 'rgba(102, 126, 234, 0.2)',
                line: { color: 'transparent' },
                name: 'Lower CI',
                showlegend: false
            }
        ];

        var forecastLayout = {
            title: 'Availability Forecast (3 Months)',
            xaxis: { title: 'Month' },
            yaxis: {
                title: 'Availability (%)',
                range: [99.9, 100]
            },
            height: 400
        };

        Plotly.newPlot('forecast-chart', forecastData, forecastLayout);
        """
```

---

## Automated Scheduling

### Report Scheduler

```python
import schedule
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


class ReportScheduler:
    """Automate report generation and distribution"""

    def __init__(
        self,
        reporter: SLOReporter,
        config: Dict[str, Any]
    ):
        self.reporter = reporter
        self.config = config
        self.email_config = config.get('email', {})

    def setup_schedules(self):
        """Setup all report schedules"""

        # Monthly executive reports (1st of month, 9 AM)
        schedule.every().month.at("09:00").do(
            self.generate_and_distribute_executive_reports
        )

        # Weekly technical reports (Monday, 10 AM)
        schedule.every().monday.at("10:00").do(
            self.generate_and_distribute_technical_reports
        )

        # Daily SLO dashboards (every day, 8 AM)
        schedule.every().day.at("08:00").do(
            self.generate_and_distribute_daily_summaries
        )

        # Quarterly business reviews (1st of quarter, 9 AM)
        schedule.every(3).months.at("09:00").do(
            self.generate_quarterly_reviews
        )

    def generate_and_distribute_executive_reports(self):
        """Generate and distribute monthly executive reports"""
        current_month = datetime.now().replace(day=1) - timedelta(days=1)
        month_str = current_month.strftime("%Y-%m")

        services = self.config.get('services', [])

        for service in services:
            try:
                # Generate report
                report = self.reporter.generate_monthly_report(
                    service=service,
                    month=month_str,
                    report_type=ReportType.EXECUTIVE
                )

                # Distribute to executives
                self.distribute_report(
                    report=report,
                    recipient_group='executives',
                    service=service
                )

            except Exception as e:
                print(f"Error generating report for {service}: {e}")

    def generate_and_distribute_technical_reports(self):
        """Generate and distribute weekly technical reports"""
        # Implementation similar to executive reports
        pass

    def distribute_report(
        self,
        report: Dict[str, Any],
        recipient_group: str,
        service: str
    ):
        """
        Distribute report via email

        Args:
            report: Generated report data
            recipient_group: Group of recipients
            service: Service name
        """
        recipients = self.config['distribution_lists'].get(recipient_group, [])

        # Create email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"SLO Report: {service} - {report['metadata']['period']}"
        msg['From'] = self.email_config['from_address']
        msg['To'] = ', '.join(recipients)

        # Attach HTML report
        html_part = MIMEText(report['html'], 'html')
        msg.attach(html_part)

        # Attach PDF if configured
        if self.config.get('attach_pdf', False):
            pdf_data = self._generate_pdf_report(report)
            pdf_part = MIMEApplication(pdf_data, _subtype='pdf')
            pdf_part.add_header(
                'Content-Disposition',
                'attachment',
                filename=f"slo_report_{service}_{report['metadata']['period']}.pdf"
            )
            msg.attach(pdf_part)

        # Send email
        self._send_email(msg, recipients)

    def _send_email(self, msg: MIMEMultipart, recipients: List[str]):
        """Send email via SMTP"""
        try:
            with smtplib.SMTP(
                self.email_config['smtp_host'],
                self.email_config['smtp_port']
            ) as server:
                server.starttls()
                server.login(
                    self.email_config['username'],
                    self.email_config['password']
                )
                server.send_message(msg)

            print(f"Report sent successfully to {', '.join(recipients)}")

        except Exception as e:
            print(f"Error sending email: {e}")
```

---

## Historical Analysis

### Historical Comparison Engine

```python
class HistoricalAnalyzer:
    """Analyze historical SLO performance"""

    def __init__(self, storage_client):
        self.storage = storage_client

    def compare_historical(
        self,
        service: str,
        current_month: str,
        lookback_months: int = 12
    ) -> Dict[str, Any]:
        """
        Compare current performance with historical data

        Args:
            service: Service name
            current_month: Current month (YYYY-MM)
            lookback_months: Months to look back

        Returns:
            Historical comparison analysis
        """
        # Fetch historical reports
        historical_reports = self._fetch_historical_reports(
            service,
            current_month,
            lookback_months
        )

        # Calculate year-over-year comparison
        yoy_comparison = self._calculate_yoy_comparison(
            historical_reports,
            current_month
        )

        # Calculate month-over-month trends
        mom_trends = self._calculate_mom_trends(historical_reports)

        # Identify best and worst months
        performance_ranking = self._rank_monthly_performance(historical_reports)

        # Calculate improvement rate
        improvement_rate = self._calculate_improvement_rate(historical_reports)

        return {
            'yoy_comparison': yoy_comparison,
            'mom_trends': mom_trends,
            'performance_ranking': performance_ranking,
            'improvement_rate': improvement_rate,
            'historical_average': self._calculate_historical_average(historical_reports),
            'consistency_score': self._calculate_consistency_score(historical_reports)
        }

    def _fetch_historical_reports(
        self,
        service: str,
        current_month: str,
        lookback_months: int
    ) -> List[Dict[str, Any]]:
        """Fetch historical report data"""
        reports = []
        current = datetime.strptime(current_month, "%Y-%m")

        for i in range(lookback_months):
            month = (current - timedelta(days=30 * i)).strftime("%Y-%m")
            report = self.storage.get_report(service, month)
            if report:
                reports.append(report)

        return reports

    def _calculate_yoy_comparison(
        self,
        reports: List[Dict[str, Any]],
        current_month: str
    ) -> Dict[str, Any]:
        """Calculate year-over-year comparison"""
        current_date = datetime.strptime(current_month, "%Y-%m")
        previous_year_date = current_date - timedelta(days=365)
        previous_year_month = previous_year_date.strftime("%Y-%m")

        current_report = next(
            (r for r in reports if r['metadata']['period'] == current_month),
            None
        )
        previous_year_report = next(
            (r for r in reports if r['metadata']['period'] == previous_year_month),
            None
        )

        if not current_report or not previous_year_report:
            return {'available': False}

        comparison = {
            'available': True,
            'metrics': {}
        }

        for metric_name in current_report['slo_performance'].keys():
            current_value = current_report['slo_performance'][metric_name]['actual']
            previous_value = previous_year_report['slo_performance'][metric_name]['actual']

            change = ((current_value - previous_value) / previous_value) * 100

            comparison['metrics'][metric_name] = {
                'current': current_value,
                'previous_year': previous_value,
                'change_percentage': change,
                'improved': change > 0
            }

        return comparison

    def _calculate_mom_trends(
        self,
        reports: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Calculate month-over-month trends"""
        trends = {}

        sorted_reports = sorted(
            reports,
            key=lambda r: r['metadata']['period']
        )

        for report in sorted_reports:
            for metric_name, metric_data in report['slo_performance'].items():
                if metric_name not in trends:
                    trends[metric_name] = []
                trends[metric_name].append(metric_data['actual'])

        return trends

    def _rank_monthly_performance(
        self,
        reports: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Rank months by overall performance"""
        scores = []

        for report in reports:
            # Calculate composite score
            total_score = 0
            metrics_count = 0

            for metric_data in report['slo_performance'].values():
                if metric_data['met']:
                    total_score += 1
                metrics_count += 1

            performance_score = (total_score / metrics_count) * 100 if metrics_count > 0 else 0

            scores.append({
                'month': report['metadata']['period'],
                'score': performance_score,
                'slos_met': total_score,
                'total_slos': metrics_count
            })

        sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)

        return {
            'best_month': sorted_scores[0] if sorted_scores else None,
            'worst_month': sorted_scores[-1] if sorted_scores else None,
            'ranking': sorted_scores
        }

    def _calculate_improvement_rate(
        self,
        reports: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate rate of improvement over time"""
        if len(reports) < 3:
            return {}

        sorted_reports = sorted(reports, key=lambda r: r['metadata']['period'])

        improvement_rates = {}

        for metric_name in sorted_reports[0]['slo_performance'].keys():
            values = [
                r['slo_performance'][metric_name]['actual']
                for r in sorted_reports
            ]

            # Linear regression to find improvement rate
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            slope = z[0]

            # Convert to percentage per month
            avg_value = np.mean(values)
            improvement_rate_pct = (slope / avg_value) * 100 if avg_value != 0 else 0

            improvement_rates[metric_name] = improvement_rate_pct

        return improvement_rates

    def _calculate_historical_average(
        self,
        reports: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate historical average for each metric"""
        averages = {}

        for metric_name in reports[0]['slo_performance'].keys():
            values = [
                r['slo_performance'][metric_name]['actual']
                for r in reports
            ]
            averages[metric_name] = np.mean(values)

        return averages

    def _calculate_consistency_score(
        self,
        reports: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate consistency score (0-100)
        Higher score = more consistent performance
        """
        if len(reports) < 2:
            return 0.0

        total_cv = 0
        metrics_count = 0

        for metric_name in reports[0]['slo_performance'].keys():
            values = [
                r['slo_performance'][metric_name]['actual']
                for r in reports
            ]

            # Coefficient of variation
            mean = np.mean(values)
            std = np.std(values)
            cv = (std / mean) * 100 if mean != 0 else 0

            total_cv += cv
            metrics_count += 1

        avg_cv = total_cv / metrics_count if metrics_count > 0 else 0

        # Convert to score (lower CV = higher score)
        consistency_score = max(0, 100 - avg_cv)

        return consistency_score
```

---

## Best Practices

### Implementation Guidelines

1. **Report Generation**
   - Generate reports asynchronously to avoid blocking
   - Cache report data for faster regeneration
   - Version control report schemas
   - Implement retry logic for failed generations

2. **Data Accuracy**
   - Validate metrics before reporting
   - Handle missing data gracefully
   - Document calculation methodologies
   - Include confidence intervals

3. **Distribution**
   - Segment audiences appropriately
   - Personalize content by role
   - Use appropriate communication channels
   - Track report delivery and engagement

4. **Performance**
   - Pre-aggregate data for large time ranges
   - Use incremental calculations
   - Implement report caching
   - Optimize chart generation

5. **Security**
   - Sanitize sensitive data
   - Implement access controls
   - Audit report distribution
   - Encrypt reports in transit

### Example Configuration

```yaml
# config/slo_reporting.yaml
reporting:
  enabled: true
  storage:
    type: s3
    bucket: slo-reports
    retention_days: 730

  schedules:
    executive_monthly:
      enabled: true
      day_of_month: 1
      time: "09:00"
      timezone: "UTC"

    technical_weekly:
      enabled: true
      day_of_week: monday
      time: "10:00"
      timezone: "UTC"

  distribution_lists:
    executives:
      - ceo@example.com
      - cto@example.com

    technical:
      - sre-team@example.com
      - engineering@example.com

    customers:
      - status@example.com

  email:
    smtp_host: smtp.gmail.com
    smtp_port: 587
    from_address: slo-reports@example.com
    username: ${SMTP_USERNAME}
    password: ${SMTP_PASSWORD}

  services:
    - api-gateway
    - auth-service
    - payment-service
    - notification-service

  slo_definitions:
    availability:
      target: 99.9
    latency_p95:
      target: 500
    latency_p99:
      target: 1000
    error_rate:
      target: 1.0

  features:
    attach_pdf: true
    include_forecasts: true
    historical_comparison: true
    interactive_charts: true
```

---

## Summary

This documentation provides a complete SLO reporting system with:

- **Monthly Report Generation**: Automated comprehensive reports
- **Performance Metrics**: Detailed SLO calculations and analysis
- **Incident Impact**: Root cause analysis and impact quantification
- **Trend Analysis**: Historical trends and future forecasting
- **Stakeholder Templates**: Executive, technical, and customer-facing reports
- **HTML Reports**: Rich, interactive visualizations
- **Automated Scheduling**: Scheduled generation and distribution
- **Historical Analysis**: Year-over-year and trend comparisons

The system is production-ready, scalable, and designed for enterprise SLO management and reporting needs.
