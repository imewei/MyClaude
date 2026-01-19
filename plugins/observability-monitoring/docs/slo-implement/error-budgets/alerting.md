# Burn Rate Alerting

## Standard Burn Rate Thresholds

### Google SRE Multi-Window, Multi-Burn-Rate Alerts

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
