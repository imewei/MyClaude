# SLO Framework Implementation

This document provides the Python implementation of the SLO Framework, including the main framework class, tier analysis engine, and user journey templates.

## 8. Python SLOFramework Implementation

### 8.1 Complete SLOFramework Class

```python
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

class SLOFramework:
    """
    Comprehensive SLO framework for designing and implementing SLOs
    """

    def __init__(self, service_name: str):
        self.service = service_name
        self.slos = []
        self.error_budget = None
        self.service_context = None

    def design_slo_framework(self):
        """
        Design comprehensive SLO framework for the service

        Returns:
            Complete SLO specification
        """
        framework = {
            'service_name': self.service,
            'service_context': self._analyze_service_context(),
            'user_journeys': self._identify_user_journeys(),
            'sli_candidates': self._identify_sli_candidates(),
            'slo_targets': self._calculate_slo_targets(),
            'error_budgets': self._define_error_budgets(),
            'measurement_strategy': self._design_measurement_strategy()
        }

        return self._generate_slo_specification(framework)

    def _analyze_service_context(self):
        """Analyze service characteristics for SLO design"""
        return {
            'service_tier': self._determine_service_tier(),
            'user_expectations': self._assess_user_expectations(),
            'business_impact': self._evaluate_business_impact(),
            'technical_constraints': self._identify_constraints(),
            'dependencies': self._map_dependencies()
        }

    def _determine_service_tier(self):
        """Determine appropriate service tier and SLO targets"""
        tiers = {
            'critical': {
                'description': 'Revenue-critical or safety-critical services',
                'availability_target': 99.95,
                'latency_p99': 100,
                'error_rate': 0.001,
                'examples': ['payment processing', 'authentication'],
                'investment_level': 'maximum'
            },
            'essential': {
                'description': 'Core business functionality',
                'availability_target': 99.9,
                'latency_p99': 500,
                'error_rate': 0.01,
                'examples': ['search', 'product catalog'],
                'investment_level': 'high'
            },
            'standard': {
                'description': 'Standard features',
                'availability_target': 99.5,
                'latency_p99': 1000,
                'error_rate': 0.05,
                'examples': ['recommendations', 'analytics'],
                'investment_level': 'moderate'
            },
            'best_effort': {
                'description': 'Non-critical features',
                'availability_target': 99.0,
                'latency_p99': 2000,
                'error_rate': 0.1,
                'examples': ['batch processing', 'reporting'],
                'investment_level': 'minimal'
            }
        }

        # Analyze service characteristics to determine tier
        characteristics = self._analyze_service_characteristics()
        recommended_tier = self._match_tier(characteristics, tiers)

        return {
            'recommended': recommended_tier,
            'rationale': self._explain_tier_selection(characteristics),
            'all_tiers': tiers
        }

    def _analyze_service_characteristics(self):
        """
        Analyze service to determine appropriate tier
        This would be customized based on actual service
        """
        # Example characteristics analysis
        return {
            'revenue_impact': 'high',
            'user_visibility': 'constant',
            'failure_consequence': 'severe',
            'real_time_required': True,
            'compliance_required': False
        }

    def _match_tier(self, characteristics, tiers):
        """Match service characteristics to appropriate tier"""
        # Simplified tier matching logic
        # In production, this would be more sophisticated

        if characteristics.get('revenue_impact') == 'direct' or \
           characteristics.get('failure_consequence') == 'severe':
            return 'critical'
        elif characteristics.get('user_visibility') == 'constant':
            return 'essential'
        elif characteristics.get('user_visibility') == 'frequent':
            return 'standard'
        else:
            return 'best_effort'

    def _explain_tier_selection(self, characteristics):
        """Explain why a tier was selected"""
        reasons = []

        if characteristics.get('revenue_impact') == 'direct':
            reasons.append('Direct revenue impact')
        if characteristics.get('user_visibility') == 'constant':
            reasons.append('High user visibility')
        if characteristics.get('failure_consequence') == 'severe':
            reasons.append('Severe failure consequences')
        if characteristics.get('real_time_required'):
            reasons.append('Real-time requirements')

        return ' | '.join(reasons) if reasons else 'Standard service requirements'

    def _assess_user_expectations(self):
        """Assess what users expect from this service"""
        return {
            'response_time_expectation': '< 1 second',
            'availability_expectation': '24/7',
            'error_tolerance': 'very low',
            'data_freshness': 'real-time'
        }

    def _evaluate_business_impact(self):
        """Evaluate business impact of service degradation"""
        return {
            'revenue_per_hour_of_downtime': 10000,  # Example: $10k/hour
            'user_churn_risk': 'high',
            'brand_reputation_impact': 'significant',
            'competitive_advantage': True
        }

    def _identify_constraints(self):
        """Identify technical constraints affecting SLO targets"""
        return {
            'infrastructure_limitations': [],
            'dependency_constraints': [],
            'budget_constraints': [],
            'team_capacity': 'adequate'
        }

    def _map_dependencies(self):
        """Map service dependencies"""
        return {
            'synchronous_dependencies': [
                {'service': 'database', 'slo': 99.95},
                {'service': 'cache', 'slo': 99.99}
            ],
            'asynchronous_dependencies': [
                {'service': 'queue', 'slo': 99.9}
            ],
            'external_dependencies': [
                {'service': 'payment_gateway', 'slo': 99.9}
            ]
        }

    def _identify_user_journeys(self):
        """Map critical user journeys for SLI selection"""
        journeys = []

        # Example user journey mapping
        journey_template = {
            'name': 'User Login',
            'description': 'User authenticates and accesses dashboard',
            'persona': 'all_users',
            'frequency': 'multiple_daily',
            'business_criticality': 'critical',
            'steps': [
                {
                    'step_number': 1,
                    'name': 'Load login page',
                    'sli_type': 'availability',
                    'threshold': '< 2s load time',
                    'expected_success_rate': 99.95
                },
                {
                    'step_number': 2,
                    'name': 'Submit credentials',
                    'sli_type': 'latency',
                    'threshold': '< 500ms response',
                    'expected_success_rate': 99.9
                },
                {
                    'step_number': 3,
                    'name': 'Validate authentication',
                    'sli_type': 'error_rate',
                    'threshold': '< 0.1% auth failures',
                    'expected_success_rate': 99.9
                },
                {
                    'step_number': 4,
                    'name': 'Load dashboard',
                    'sli_type': 'latency',
                    'threshold': '< 3s full render',
                    'expected_success_rate': 99.5
                }
            ],
            'critical_path': True,
            'business_impact': 'high'
        }

        journeys.append(journey_template)
        return journeys

    def _identify_sli_candidates(self):
        """Identify potential SLIs for the service"""
        return {
            'availability': {
                'definition': 'Percentage of successful requests',
                'measurement': 'HTTP status codes',
                'formula': 'successful_requests / total_requests',
                'recommended': True
            },
            'latency': {
                'definition': 'Request response time',
                'measurement': 'Request duration',
                'formula': 'requests_under_threshold / total_requests',
                'recommended': True,
                'percentiles': [50, 95, 99]
            },
            'error_rate': {
                'definition': 'Percentage of failed requests',
                'measurement': 'Error responses',
                'formula': '1 - (error_requests / total_requests)',
                'recommended': True
            },
            'throughput': {
                'definition': 'Requests processed per second',
                'measurement': 'Request rate',
                'formula': 'requests / time_period',
                'recommended': False
            }
        }

    def _calculate_slo_targets(self):
        """Calculate appropriate SLO targets"""
        tier_info = self._determine_service_tier()
        recommended_tier = tier_info['recommended']
        tier_config = tier_info['all_tiers'][recommended_tier]

        return {
            'availability': {
                'target': tier_config['availability_target'],
                'window': '30d',
                'type': 'rolling'
            },
            'latency_p99': {
                'target': tier_config['latency_p99'],
                'unit': 'milliseconds',
                'window': '30d',
                'type': 'rolling'
            },
            'error_rate': {
                'target': tier_config['error_rate'],
                'window': '30d',
                'type': 'rolling'
            }
        }

    def _define_error_budgets(self):
        """Define error budgets based on SLO targets"""
        targets = self._calculate_slo_targets()

        budgets = {}
        for slo_name, slo_config in targets.items():
            if 'target' in slo_config and slo_name == 'availability':
                error_budget = 1 - (slo_config['target'] / 100)

                # Calculate in time units
                window_days = int(slo_config['window'].replace('d', ''))
                total_minutes = window_days * 24 * 60

                budgets[slo_name] = {
                    'error_budget_ratio': error_budget,
                    'error_budget_percentage': error_budget * 100,
                    'allowed_downtime_minutes': total_minutes * error_budget,
                    'allowed_downtime_hours': (total_minutes * error_budget) / 60,
                    'burn_rate_alerts': self._generate_burn_rate_thresholds(error_budget)
                }

        return budgets

    def _generate_burn_rate_thresholds(self, error_budget):
        """Generate multi-window burn rate alert thresholds"""
        return {
            'critical': {
                'window': '1h',
                'threshold': 14.4,
                'budget_consumed': 0.02,
                'action': 'page'
            },
            'warning': {
                'window': '6h',
                'threshold': 6.0,
                'budget_consumed': 0.05,
                'action': 'ticket'
            }
        }

    def _design_measurement_strategy(self):
        """Design strategy for measuring SLIs"""
        return {
            'data_sources': ['prometheus', 'application_logs'],
            'collection_interval': '30s',
            'aggregation_windows': ['5m', '30m', '1h', '6h', '1d', '30d'],
            'storage_retention': '90d',
            'dashboard_url': f'https://grafana.example.com/d/{self.service}-slo',
            'alert_destinations': ['pagerduty', 'slack']
        }

    def _generate_slo_specification(self, framework):
        """Generate complete SLO specification document"""
        return {
            'version': '1.0',
            'service': self.service,
            'generated_at': datetime.now().isoformat(),
            'framework': framework,
            'implementation': {
                'prometheus_rules': self._generate_prometheus_rules(framework),
                'alert_rules': self._generate_alert_rules(framework),
                'dashboard_config': self._generate_dashboard_config(framework)
            }
        }

    def _generate_prometheus_rules(self, framework):
        """Generate Prometheus recording rules"""
        return """
groups:
  - name: slo_metrics
    interval: 30s
    rules:
      - record: slo:availability:ratio_5m
        expr: |
          sum(rate(http_requests_total{status!~"5.."}[5m])) /
          sum(rate(http_requests_total[5m]))

      - record: slo:latency:p99_5m
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          )
"""

    def _generate_alert_rules(self, framework):
        """Generate Prometheus alert rules"""
        budgets = framework['error_budgets']

        if 'availability' in budgets:
            critical_threshold = budgets['availability']['burn_rate_alerts']['critical']['threshold']

            return f"""
groups:
  - name: slo_alerts
    rules:
      - alert: ErrorBudgetBurnRateCritical
        expr: |
          (1 - slo:availability:ratio_5m) / {budgets['availability']['error_budget_ratio']} > {critical_threshold}
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Critical error budget burn rate"
"""
        return ""

    def _generate_dashboard_config(self, framework):
        """Generate Grafana dashboard configuration"""
        return {
            'title': f'{self.service} SLO Dashboard',
            'panels': [
                {
                    'title': 'SLO Compliance',
                    'type': 'gauge',
                    'query': 'slo:availability:ratio_5m * 100'
                },
                {
                    'title': 'Error Budget',
                    'type': 'gauge',
                    'query': 'error_budget_remaining_percentage'
                },
                {
                    'title': 'Burn Rate',
                    'type': 'graph',
                    'query': 'error_budget_burn_rate'
                }
            ]
        }

# Example usage:
slo_framework = SLOFramework('payment-api')
specification = slo_framework.design_slo_framework()
```

## 9. Tier Analysis and Recommendation Engine

### 9.1 TierAnalyzer Class

```python
class TierAnalyzer:
    """
    Analyze service characteristics and recommend appropriate tier
    """

    def __init__(self):
        self.criteria_weights = {
            'revenue_impact': 0.25,
            'user_visibility': 0.20,
            'failure_severity': 0.20,
            'data_sensitivity': 0.15,
            'regulatory_requirements': 0.10,
            'user_base_size': 0.10
        }

    def analyze_and_recommend(self, service_characteristics):
        """
        Analyze service and recommend tier

        Args:
            service_characteristics: Dict with service properties

        Returns:
            Recommended tier with detailed analysis
        """
        # Calculate score for each tier
        tier_scores = self._calculate_tier_scores(service_characteristics)

        # Get recommended tier
        recommended_tier = max(tier_scores, key=tier_scores.get)

        # Generate detailed analysis
        analysis = {
            'recommended_tier': recommended_tier,
            'confidence_score': tier_scores[recommended_tier],
            'tier_scores': tier_scores,
            'key_factors': self._identify_key_factors(service_characteristics),
            'slo_targets': self._get_tier_targets(recommended_tier),
            'investment_requirements': self._get_investment_requirements(recommended_tier),
            'next_steps': self._generate_next_steps(recommended_tier)
        }

        return analysis

    def _calculate_tier_scores(self, characteristics):
        """Calculate weighted scores for each tier"""
        # Scoring matrix: criteria -> tier -> score
        scoring_matrix = {
            'revenue_impact': {
                'direct': {'critical': 10, 'essential': 5, 'standard': 2, 'best_effort': 0},
                'high': {'critical': 7, 'essential': 10, 'standard': 5, 'best_effort': 2},
                'medium': {'critical': 3, 'essential': 7, 'standard': 10, 'best_effort': 5},
                'low': {'critical': 0, 'essential': 3, 'standard': 7, 'best_effort': 10},
                'none': {'critical': 0, 'essential': 0, 'standard': 5, 'best_effort': 10}
            },
            'user_visibility': {
                'constant': {'critical': 10, 'essential': 8, 'standard': 4, 'best_effort': 0},
                'frequent': {'critical': 7, 'essential': 10, 'standard': 6, 'best_effort': 2},
                'occasional': {'critical': 3, 'essential': 6, 'standard': 10, 'best_effort': 5},
                'rare': {'critical': 0, 'essential': 2, 'standard': 5, 'best_effort': 10}
            },
            'failure_severity': {
                'severe': {'critical': 10, 'essential': 5, 'standard': 2, 'best_effort': 0},
                'moderate': {'critical': 5, 'essential': 10, 'standard': 7, 'best_effort': 3},
                'minor': {'critical': 2, 'essential': 5, 'standard': 10, 'best_effort': 7},
                'negligible': {'critical': 0, 'essential': 2, 'standard': 5, 'best_effort': 10}
            }
        }

        # Calculate scores
        tier_scores = {'critical': 0, 'essential': 0, 'standard': 0, 'best_effort': 0}

        for criterion, weight in self.criteria_weights.items():
            if criterion in characteristics and criterion in scoring_matrix:
                value = characteristics[criterion]
                if value in scoring_matrix[criterion]:
                    for tier, score in scoring_matrix[criterion][value].items():
                        tier_scores[tier] += score * weight

        # Normalize scores to 0-100
        max_score = max(tier_scores.values()) if tier_scores.values() else 1
        normalized_scores = {
            tier: (score / max_score) * 100
            for tier, score in tier_scores.items()
        }

        return normalized_scores

    def _identify_key_factors(self, characteristics):
        """Identify the most important factors influencing tier selection"""
        key_factors = []

        if characteristics.get('revenue_impact') in ['direct', 'high']:
            key_factors.append({
                'factor': 'Revenue Impact',
                'value': characteristics['revenue_impact'],
                'influence': 'Drives toward higher tier'
            })

        if characteristics.get('failure_severity') == 'severe':
            key_factors.append({
                'factor': 'Failure Severity',
                'value': 'severe',
                'influence': 'Requires critical tier'
            })

        if characteristics.get('user_visibility') == 'constant':
            key_factors.append({
                'factor': 'User Visibility',
                'value': 'constant',
                'influence': 'Requires high availability'
            })

        return key_factors

    def _get_tier_targets(self, tier):
        """Get SLO targets for a tier"""
        targets = {
            'critical': {
                'availability': 99.95,
                'latency_p99_ms': 100,
                'error_rate_max': 0.001,
                'downtime_per_month': '22 minutes'
            },
            'essential': {
                'availability': 99.9,
                'latency_p99_ms': 500,
                'error_rate_max': 0.01,
                'downtime_per_month': '43 minutes'
            },
            'standard': {
                'availability': 99.5,
                'latency_p99_ms': 1000,
                'error_rate_max': 0.05,
                'downtime_per_month': '3.6 hours'
            },
            'best_effort': {
                'availability': 99.0,
                'latency_p99_ms': 2000,
                'error_rate_max': 0.1,
                'downtime_per_month': '7.2 hours'
            }
        }

        return targets.get(tier, {})

    def _get_investment_requirements(self, tier):
        """Get investment requirements for a tier"""
        requirements = {
            'critical': {
                'infrastructure': [
                    'Multi-region deployment',
                    'Active-active redundancy',
                    'Automated failover',
                    'Circuit breakers',
                    'Advanced monitoring'
                ],
                'team': [
                    '24/7 on-call rotation',
                    'Dedicated SRE support',
                    'Regular DR drills',
                    'Chaos engineering'
                ],
                'cost_multiplier': '3-4x baseline'
            },
            'essential': {
                'infrastructure': [
                    'Multi-AZ deployment',
                    'Load balancing',
                    'Database replication',
                    'Comprehensive monitoring'
                ],
                'team': [
                    'Business hours on-call',
                    'SLO reviews',
                    'Incident response',
                    'Performance testing'
                ],
                'cost_multiplier': '2-3x baseline'
            },
            'standard': {
                'infrastructure': [
                    'Basic redundancy',
                    'Standard monitoring',
                    'Backup systems'
                ],
                'team': [
                    'Best-effort on-call',
                    'Monthly reviews',
                    'Basic testing'
                ],
                'cost_multiplier': '1.5-2x baseline'
            },
            'best_effort': {
                'infrastructure': [
                    'Single deployment',
                    'Basic monitoring',
                    'Manual recovery'
                ],
                'team': [
                    'No dedicated on-call',
                    'Quarterly reviews',
                    'Manual testing'
                ],
                'cost_multiplier': '1x baseline'
            }
        }

        return requirements.get(tier, {})

    def _generate_next_steps(self, tier):
        """Generate recommended next steps"""
        return [
            f'Review and validate {tier} tier selection with stakeholders',
            'Define specific SLO targets based on tier guidelines',
            'Assess current infrastructure against tier requirements',
            'Identify gaps and create remediation plan',
            'Establish monitoring and alerting',
            'Create runbooks and documentation',
            'Set up SLO dashboard',
            'Schedule regular SLO reviews'
        ]

# Example usage:
analyzer = TierAnalyzer()
recommendation = analyzer.analyze_and_recommend({
    'revenue_impact': 'direct',
    'user_visibility': 'constant',
    'failure_severity': 'severe',
    'data_sensitivity': 'high',
    'regulatory_requirements': 'yes',
    'user_base_size': 'large'
})
```

## 10. User Journey Templates

### 10.1 E-Commerce User Journeys

```python
class ECommerceJourneyTemplates:
    """Pre-built journey templates for e-commerce applications"""

    @staticmethod
    def get_product_browse_journey():
        """Product browsing journey"""
        return {
            'name': 'Product Browse',
            'description': 'User browses product catalog',
            'persona': 'all_users',
            'frequency': 'continuous',
            'business_criticality': 'essential',
            'steps': [
                {
                    'step_number': 1,
                    'name': 'Load homepage',
                    'action': 'GET /',
                    'expected_latency_p95': 1000,
                    'expected_success_rate': 99.9,
                    'sli_type': 'availability'
                },
                {
                    'step_number': 2,
                    'name': 'Search products',
                    'action': 'GET /api/search',
                    'expected_latency_p95': 500,
                    'expected_success_rate': 99.5,
                    'sli_type': 'latency'
                },
                {
                    'step_number': 3,
                    'name': 'View product details',
                    'action': 'GET /api/products/:id',
                    'expected_latency_p95': 300,
                    'expected_success_rate': 99.9,
                    'sli_type': 'availability'
                },
                {
                    'step_number': 4,
                    'name': 'Load product images',
                    'action': 'GET /cdn/images/*',
                    'expected_latency_p95': 200,
                    'expected_success_rate': 99.0,
                    'sli_type': 'availability'
                }
            ]
        }

    @staticmethod
    def get_checkout_journey():
        """Checkout journey"""
        return {
            'name': 'Checkout',
            'description': 'User completes purchase',
            'persona': 'converting_users',
            'frequency': 'multiple_per_day',
            'business_criticality': 'critical',
            'steps': [
                {
                    'step_number': 1,
                    'name': 'View cart',
                    'action': 'GET /api/cart',
                    'expected_latency_p95': 200,
                    'expected_success_rate': 99.95,
                    'sli_type': 'availability'
                },
                {
                    'step_number': 2,
                    'name': 'Enter shipping address',
                    'action': 'POST /api/shipping',
                    'expected_latency_p95': 500,
                    'expected_success_rate': 99.9,
                    'sli_type': 'error_rate'
                },
                {
                    'step_number': 3,
                    'name': 'Select shipping method',
                    'action': 'GET /api/shipping-options',
                    'expected_latency_p95': 300,
                    'expected_success_rate': 99.9,
                    'sli_type': 'availability'
                },
                {
                    'step_number': 4,
                    'name': 'Enter payment info',
                    'action': 'POST /api/payment-method',
                    'expected_latency_p95': 400,
                    'expected_success_rate': 99.95,
                    'sli_type': 'error_rate'
                },
                {
                    'step_number': 5,
                    'name': 'Process payment',
                    'action': 'POST /api/charge',
                    'expected_latency_p95': 2000,
                    'expected_success_rate': 99.95,
                    'sli_type': 'availability',
                    'dependencies': ['payment_gateway', 'fraud_detection']
                },
                {
                    'step_number': 6,
                    'name': 'Confirm order',
                    'action': 'GET /api/order/:id',
                    'expected_latency_p95': 400,
                    'expected_success_rate': 99.99,
                    'sli_type': 'availability'
                }
            ]
        }

    @staticmethod
    def get_user_registration_journey():
        """User registration journey"""
        return {
            'name': 'User Registration',
            'description': 'New user creates account',
            'persona': 'new_users',
            'frequency': 'once_per_user',
            'business_criticality': 'critical',
            'steps': [
                {
                    'step_number': 1,
                    'name': 'Load registration form',
                    'action': 'GET /register',
                    'expected_latency_p95': 1000,
                    'expected_success_rate': 99.9,
                    'sli_type': 'availability'
                },
                {
                    'step_number': 2,
                    'name': 'Validate email',
                    'action': 'POST /api/validate-email',
                    'expected_latency_p95': 300,
                    'expected_success_rate': 99.5,
                    'sli_type': 'latency'
                },
                {
                    'step_number': 3,
                    'name': 'Create account',
                    'action': 'POST /api/users',
                    'expected_latency_p95': 500,
                    'expected_success_rate': 99.95,
                    'sli_type': 'error_rate'
                },
                {
                    'step_number': 4,
                    'name': 'Send verification email',
                    'action': 'POST /api/send-verification',
                    'expected_latency_p95': 1000,
                    'expected_success_rate': 99.0,
                    'sli_type': 'availability',
                    'async': True
                }
            ]
        }

class SaaSJourneyTemplates:
    """Pre-built journey templates for SaaS applications"""

    @staticmethod
    def get_user_login_journey():
        """User login journey"""
        return {
            'name': 'User Login',
            'description': 'User authenticates and accesses dashboard',
            'persona': 'all_users',
            'frequency': 'multiple_daily',
            'business_criticality': 'critical',
            'steps': [
                {
                    'step_number': 1,
                    'name': 'Load login page',
                    'action': 'GET /login',
                    'expected_latency_p95': 800,
                    'expected_success_rate': 99.95,
                    'sli_type': 'availability'
                },
                {
                    'step_number': 2,
                    'name': 'Submit credentials',
                    'action': 'POST /api/auth/login',
                    'expected_latency_p95': 500,
                    'expected_success_rate': 99.9,
                    'sli_type': 'latency'
                },
                {
                    'step_number': 3,
                    'name': 'Fetch user session',
                    'action': 'GET /api/session',
                    'expected_latency_p95': 200,
                    'expected_success_rate': 99.95,
                    'sli_type': 'availability'
                },
                {
                    'step_number': 4,
                    'name': 'Load dashboard',
                    'action': 'GET /dashboard',
                    'expected_latency_p95': 1500,
                    'expected_success_rate': 99.5,
                    'sli_type': 'latency'
                }
            ]
        }

    @staticmethod
    def get_api_integration_journey():
        """API integration journey"""
        return {
            'name': 'API Integration',
            'description': 'External system integrates via API',
            'persona': 'api_users',
            'frequency': 'continuous',
            'business_criticality': 'essential',
            'steps': [
                {
                    'step_number': 1,
                    'name': 'Authenticate',
                    'action': 'POST /api/v1/oauth/token',
                    'expected_latency_p95': 300,
                    'expected_success_rate': 99.95,
                    'sli_type': 'availability'
                },
                {
                    'step_number': 2,
                    'name': 'Fetch data',
                    'action': 'GET /api/v1/data',
                    'expected_latency_p95': 500,
                    'expected_success_rate': 99.9,
                    'sli_type': 'latency'
                },
                {
                    'step_number': 3,
                    'name': 'Create resource',
                    'action': 'POST /api/v1/resources',
                    'expected_latency_p95': 800,
                    'expected_success_rate': 99.5,
                    'sli_type': 'error_rate'
                }
            ]
        }

class DataPipelineJourneyTemplates:
    """Pre-built journey templates for data pipelines"""

    @staticmethod
    def get_batch_processing_journey():
        """Batch data processing journey"""
        return {
            'name': 'Batch Processing',
            'description': 'Process batch of data records',
            'persona': 'system',
            'frequency': 'hourly',
            'business_criticality': 'essential',
            'steps': [
                {
                    'step_number': 1,
                    'name': 'Fetch data from source',
                    'action': 'extract',
                    'expected_latency_p95': 60000,  # 60 seconds
                    'expected_success_rate': 99.5,
                    'sli_type': 'availability'
                },
                {
                    'step_number': 2,
                    'name': 'Transform data',
                    'action': 'transform',
                    'expected_latency_p95': 300000,  # 5 minutes
                    'expected_success_rate': 99.0,
                    'sli_type': 'completeness'
                },
                {
                    'step_number': 3,
                    'name': 'Validate data quality',
                    'action': 'validate',
                    'expected_latency_p95': 30000,  # 30 seconds
                    'expected_success_rate': 95.0,
                    'sli_type': 'quality'
                },
                {
                    'step_number': 4,
                    'name': 'Load to destination',
                    'action': 'load',
                    'expected_latency_p95': 120000,  # 2 minutes
                    'expected_success_rate': 99.5,
                    'sli_type': 'availability'
                }
            ],
            'sli_overrides': {
                'freshness': {
                    'target': 99.0,
                    'threshold_minutes': 30,
                    'description': 'Data processed within 30 minutes'
                }
            }
        }

# Journey template manager
class JourneyTemplateManager:
    """Manage and customize journey templates"""

    def __init__(self):
        self.templates = {
            'ecommerce': ECommerceJourneyTemplates,
            'saas': SaaSJourneyTemplates,
            'data_pipeline': DataPipelineJourneyTemplates
        }

    def get_template(self, category, journey_name):
        """Get a specific journey template"""
        if category in self.templates:
            template_class = self.templates[category]
            method_name = f'get_{journey_name}_journey'

            if hasattr(template_class, method_name):
                return getattr(template_class, method_name)()

        return None

    def list_templates(self):
        """List all available templates"""
        all_templates = {}

        for category, template_class in self.templates.items():
            methods = [m for m in dir(template_class) if m.startswith('get_') and m.endswith('_journey')]
            journey_names = [m.replace('get_', '').replace('_journey', '') for m in methods]
            all_templates[category] = journey_names

        return all_templates

    def customize_template(self, template, customizations):
        """Customize a journey template"""
        customized = template.copy()

        # Apply customizations
        for key, value in customizations.items():
            if key in customized:
                customized[key] = value
            elif key == 'steps':
                # Customize specific steps
                for step_custom in value:
                    step_num = step_custom.get('step_number')
                    for step in customized['steps']:
                        if step['step_number'] == step_num:
                            step.update(step_custom)

        return customized

# Example usage:
manager = JourneyTemplateManager()
templates = manager.list_templates()
# Result: {'ecommerce': ['product_browse', 'checkout', 'user_registration'], ...}

checkout_journey = manager.get_template('ecommerce', 'checkout')
customized_journey = manager.customize_template(checkout_journey, {
    'business_criticality': 'critical',
    'steps': [
        {
            'step_number': 5,
            'expected_latency_p95': 1500  # Slower payment processing
        }
    ]
})
```
