# SLO Design and Architecture

This document covers the service tier classification, user journey mapping, and SLI identification processes.

## 2. Service Tier Classification

### 2.1 Tier Framework Overview

Service tier classification helps determine appropriate SLO targets based on business criticality, user expectations, and technical constraints.

### 2.2 Critical Tier

**Definition**: Revenue-critical or safety-critical services where failures directly impact business viability or user safety.

**Characteristics:**
- Direct revenue generation or protection
- Safety-critical functionality
- No acceptable degradation
- Real-time user interaction
- High visibility to customers

**SLO Targets:**
```python
CRITICAL_TIER = {
    'availability_target': 99.95,      # ~22 minutes downtime/month
    'latency_p50': 50,                 # milliseconds
    'latency_p95': 200,                # milliseconds
    'latency_p99': 500,                # milliseconds
    'error_rate_max': 0.001,           # 0.1% error rate
    'error_budget': 0.05               # 0.05% monthly
}
```

**Examples:**
- Payment processing systems
- Authentication and authorization
- Core trading/transaction systems
- Emergency services
- Medical device control systems

**Investment Requirements:**
- Redundancy across multiple availability zones
- Automated failover and circuit breakers
- Extensive monitoring and alerting
- 24/7 on-call coverage
- Regular disaster recovery drills

### 2.3 Essential Tier

**Definition**: Core business functionality that users depend on regularly but can tolerate brief degradation.

**Characteristics:**
- Core product features
- High user visibility
- Some degradation acceptable
- Impacts user satisfaction but not safety
- Moderate revenue impact

**SLO Targets:**
```python
ESSENTIAL_TIER = {
    'availability_target': 99.9,       # ~43 minutes downtime/month
    'latency_p50': 100,                # milliseconds
    'latency_p95': 500,                # milliseconds
    'latency_p99': 1000,               # milliseconds
    'error_rate_max': 0.01,            # 1% error rate
    'error_budget': 0.1                # 0.1% monthly
}
```

**Examples:**
- Product search and discovery
- User profile management
- Catalog browsing
- Shopping cart functionality
- Content delivery systems

**Investment Requirements:**
- Multi-zone deployment
- Automated monitoring and alerts
- Business hours on-call
- Regular performance testing
- Graceful degradation patterns

### 2.4 Standard Tier

**Definition**: Standard features that enhance user experience but aren't critical to core workflows.

**Characteristics:**
- Nice-to-have features
- Lower user expectations
- Can fail without major impact
- Background or async processing
- Limited revenue impact

**SLO Targets:**
```python
STANDARD_TIER = {
    'availability_target': 99.5,       # ~3.6 hours downtime/month
    'latency_p50': 200,                # milliseconds
    'latency_p95': 1000,               # milliseconds
    'latency_p99': 2000,               # milliseconds
    'error_rate_max': 0.05,            # 5% error rate
    'error_budget': 0.5                # 0.5% monthly
}
```

**Examples:**
- Recommendation engines
- Analytics and reporting
- Social features (likes, comments)
- Notification systems
- A/B testing frameworks

**Investment Requirements:**
- Single-zone deployment acceptable
- Standard monitoring
- Best-effort on-call
- Periodic performance review
- Basic error handling

### 2.5 Best Effort Tier

**Definition**: Non-critical features, batch processes, or internal tools with minimal user impact.

**Characteristics:**
- Internal tools or admin features
- Batch/overnight processing
- Users expect variability
- No real-time requirements
- Negligible revenue impact

**SLO Targets:**
```python
BEST_EFFORT_TIER = {
    'availability_target': 99.0,       # ~7.2 hours downtime/month
    'latency_p50': 500,                # milliseconds
    'latency_p95': 2000,               # milliseconds
    'latency_p99': 5000,               # milliseconds
    'error_rate_max': 0.1,             # 10% error rate
    'error_budget': 1.0                # 1% monthly
}
```

**Examples:**
- Batch report generation
- Data warehouse ETL
- Internal admin dashboards
- Development/staging environments
- Archive/backup systems

**Investment Requirements:**
- Minimal redundancy
- Basic monitoring
- No dedicated on-call
- Manual intervention acceptable
- Simple retry logic

### 2.6 Tier Selection Criteria

**Decision Matrix:**

| Criterion | Critical | Essential | Standard | Best Effort |
|-----------|----------|-----------|----------|-------------|
| Revenue Impact | Direct | High | Medium | Low/None |
| User Visibility | Constant | Frequent | Occasional | Rare |
| Failure Impact | Severe | Moderate | Minor | Negligible |
| Real-time Requirement | Yes | Yes | Sometimes | No |
| Compliance/Safety | Yes | Maybe | No | No |
| User Tolerance | None | Low | Medium | High |
| Investment Level | Maximum | High | Moderate | Minimal |

**Selection Process:**

1. **Assess Business Impact**
   - What happens if this service fails for 1 hour?
   - How many users are affected?
   - What's the revenue impact?

2. **Evaluate User Expectations**
   - What do users expect from this service?
   - How quickly do they expect responses?
   - What error rates are acceptable?

3. **Consider Technical Constraints**
   - What are dependency limitations?
   - What's technically feasible?
   - What's the cost-benefit tradeoff?

4. **Review Regulatory Requirements**
   - Are there compliance mandates?
   - Are there safety requirements?
   - Are there contractual obligations?

### 2.7 Tier Migration Guidelines

**Upgrading a Tier (e.g., Standard → Essential):**
```python
def plan_tier_upgrade(current_tier, target_tier):
    """Plan infrastructure and process changes for tier upgrade"""
    return {
        'infrastructure': [
            'Add redundancy (multi-AZ deployment)',
            'Implement circuit breakers',
            'Add caching layers',
            'Upgrade monitoring systems'
        ],
        'processes': [
            'Establish on-call rotation',
            'Create runbooks',
            'Implement automated rollbacks',
            'Add SLO dashboards'
        ],
        'testing': [
            'Load testing at 2x peak',
            'Chaos engineering tests',
            'Disaster recovery drills',
            'Latency testing across regions'
        ],
        'timeline': '3-6 months',
        'cost_increase': '50-100%'
    }
```

**Downgrading a Tier (e.g., Essential → Standard):**
- Verify with stakeholders that reduced reliability is acceptable
- Communicate changes to users
- Gradually relax SLO targets
- Reduce infrastructure investment accordingly

---

## 3. User Journey Mapping Methodology

### 3.1 What is User Journey Mapping?

User journey mapping identifies critical paths users take through your system to accomplish their goals. Each journey consists of multiple steps, and each step has potential failure modes that impact the overall user experience.

### 3.2 Journey Mapping Process

**Step 1: Identify Key User Personas**
```python
user_personas = {
    'new_user': {
        'goals': ['Sign up', 'Complete onboarding', 'First purchase'],
        'technical_expertise': 'low',
        'tolerance_for_errors': 'very_low'
    },
    'power_user': {
        'goals': ['Repeat purchases', 'Advanced features', 'API access'],
        'technical_expertise': 'high',
        'tolerance_for_errors': 'low'
    },
    'admin': {
        'goals': ['Manage users', 'View analytics', 'Configure settings'],
        'technical_expertise': 'high',
        'tolerance_for_errors': 'medium'
    }
}
```

**Step 2: Map Critical Journeys**

For each persona, identify their most important journeys:
```python
critical_journeys = {
    'user_registration': {
        'persona': 'new_user',
        'frequency': 'once_per_user',
        'business_impact': 'critical',
        'revenue_impact': 'direct'
    },
    'checkout_flow': {
        'persona': 'all_users',
        'frequency': 'multiple_per_day',
        'business_impact': 'critical',
        'revenue_impact': 'direct'
    },
    'product_search': {
        'persona': 'all_users',
        'frequency': 'continuous',
        'business_impact': 'essential',
        'revenue_impact': 'indirect'
    }
}
```

**Step 3: Break Down Journey Steps**

Decompose each journey into discrete, measurable steps:
```python
def map_journey_steps(journey_name):
    """Map individual steps in a user journey"""
    if journey_name == 'checkout_flow':
        return [
            {
                'step_number': 1,
                'name': 'View cart',
                'action': 'GET /api/cart',
                'expected_latency_p95': 200,  # ms
                'expected_success_rate': 99.9
            },
            {
                'step_number': 2,
                'name': 'Enter shipping info',
                'action': 'POST /api/shipping',
                'expected_latency_p95': 500,
                'expected_success_rate': 99.5
            },
            {
                'step_number': 3,
                'name': 'Select payment method',
                'action': 'GET /api/payment-methods',
                'expected_latency_p95': 300,
                'expected_success_rate': 99.9
            },
            {
                'step_number': 4,
                'name': 'Process payment',
                'action': 'POST /api/charge',
                'expected_latency_p95': 2000,
                'expected_success_rate': 99.95
            },
            {
                'step_number': 5,
                'name': 'Confirm order',
                'action': 'GET /api/order/{id}',
                'expected_latency_p95': 400,
                'expected_success_rate': 99.99
            }
        ]
```

**Step 4: Identify Dependencies**

Map service dependencies for each journey step:
```python
def map_dependencies(step):
    """Identify all dependencies for a journey step"""
    dependencies = {
        'synchronous': [],
        'asynchronous': [],
        'external': []
    }

    if step['action'] == 'POST /api/charge':
        dependencies['synchronous'] = [
            {'service': 'payment-gateway', 'criticality': 'critical'},
            {'service': 'fraud-detection', 'criticality': 'critical'},
            {'service': 'inventory-service', 'criticality': 'essential'}
        ]
        dependencies['asynchronous'] = [
            {'service': 'analytics', 'criticality': 'standard'},
            {'service': 'email-service', 'criticality': 'essential'}
        ]
        dependencies['external'] = [
            {'service': 'stripe-api', 'criticality': 'critical'},
            {'service': 'tax-service', 'criticality': 'essential'}
        ]

    return dependencies
```

### 3.3 Journey Success Calculation

**Overall Journey Success Rate:**
```python
def calculate_journey_success_rate(steps):
    """
    Calculate overall journey success rate
    Assumes steps are sequential and independent
    """
    overall_success = 1.0

    for step in steps:
        step_success = step['expected_success_rate'] / 100
        overall_success *= step_success

    return overall_success * 100

# Example: 5-step checkout
checkout_steps = [
    {'expected_success_rate': 99.9},   # Step 1
    {'expected_success_rate': 99.5},   # Step 2
    {'expected_success_rate': 99.9},   # Step 3
    {'expected_success_rate': 99.95},  # Step 4
    {'expected_success_rate': 99.99}   # Step 5
]

overall = calculate_journey_success_rate(checkout_steps)
# Result: ~99.24% overall success rate
```

**Journey Latency Distribution:**
```python
import numpy as np

def calculate_journey_latency(steps, percentile=95):
    """
    Calculate end-to-end journey latency
    Uses sum for sequential steps
    """
    latencies = [step['expected_latency_p95'] for step in steps]

    # For sequential steps, sum the latencies
    total_latency = sum(latencies)

    # For parallel steps, use max
    # total_latency = max(latencies)

    return total_latency

# Example: 5-step checkout
checkout_latency = calculate_journey_latency(checkout_steps)
# Result: 3400ms total p95 latency
```

### 3.4 Journey Prioritization Matrix

**Priority Scoring:**
```python
def calculate_journey_priority(journey):
    """
    Calculate priority score for journey
    Higher score = higher priority for SLO focus
    """
    weights = {
        'business_impact': 0.3,
        'frequency': 0.25,
        'revenue_impact': 0.25,
        'user_visibility': 0.2
    }

    scores = {
        'business_impact': {
            'critical': 10,
            'essential': 7,
            'standard': 4,
            'low': 1
        },
        'frequency': {
            'continuous': 10,
            'multiple_per_day': 8,
            'daily': 6,
            'weekly': 4,
            'monthly': 2,
            'once_per_user': 3
        },
        'revenue_impact': {
            'direct': 10,
            'indirect': 5,
            'none': 0
        },
        'user_visibility': {
            'high': 10,
            'medium': 6,
            'low': 2
        }
    }

    total_score = 0
    for criterion, weight in weights.items():
        if criterion in journey:
            value = journey[criterion]
            score = scores[criterion].get(value, 0)
            total_score += score * weight

    return total_score

# Example usage
checkout_priority = calculate_journey_priority({
    'business_impact': 'critical',
    'frequency': 'multiple_per_day',
    'revenue_impact': 'direct',
    'user_visibility': 'high'
})
# Result: 9.75 out of 10
```

### 3.5 Journey Instrumentation Guide

**Instrumentation Requirements:**
```python
class JourneyInstrumentation:
    def __init__(self, journey_name):
        self.journey_name = journey_name
        self.start_time = None
        self.context = {}

    def start_journey(self, user_id, session_id):
        """Initialize journey tracking"""
        self.start_time = time.time()
        self.context = {
            'journey_id': self.generate_journey_id(),
            'user_id': user_id,
            'session_id': session_id,
            'journey_name': self.journey_name,
            'steps_completed': [],
            'steps_failed': []
        }

        # Emit journey start event
        self.emit_event('journey_started', self.context)

        return self.context['journey_id']

    def track_step(self, step_name, success, latency_ms, metadata=None):
        """Track individual step in journey"""
        step_data = {
            'step_name': step_name,
            'success': success,
            'latency_ms': latency_ms,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

        if success:
            self.context['steps_completed'].append(step_data)
        else:
            self.context['steps_failed'].append(step_data)

        # Emit step event
        self.emit_event('journey_step', {
            **self.context,
            **step_data
        })

    def complete_journey(self, success):
        """Mark journey as complete"""
        total_latency = (time.time() - self.start_time) * 1000

        journey_result = {
            **self.context,
            'success': success,
            'total_latency_ms': total_latency,
            'total_steps': len(self.context['steps_completed']) + len(self.context['steps_failed']),
            'failed_steps': len(self.context['steps_failed'])
        }

        # Emit journey completion event
        self.emit_event('journey_completed', journey_result)

        return journey_result
```

---

## 4. SLI Candidate Identification Process

### 4.1 SLI Selection Framework

The process of identifying appropriate SLIs follows a structured methodology that ensures you measure what matters to users.

### 4.2 Service Type Analysis

**Step 1: Classify Your Service**

```python
class ServiceClassifier:
    @staticmethod
    def classify_service(characteristics):
        """
        Classify service type to recommend appropriate SLIs
        """
        service_types = {
            'request_response': {
                'indicators': ['synchronous', 'api', 'web', 'low_latency'],
                'recommended_slis': ['availability', 'latency', 'error_rate']
            },
            'data_processing': {
                'indicators': ['batch', 'etl', 'pipeline', 'async'],
                'recommended_slis': ['freshness', 'completeness', 'throughput']
            },
            'storage': {
                'indicators': ['database', 'cache', 'object_store'],
                'recommended_slis': ['durability', 'availability', 'latency']
            },
            'streaming': {
                'indicators': ['real_time', 'events', 'messages'],
                'recommended_slis': ['throughput', 'lag', 'delivery_guarantee']
            }
        }

        # Match service characteristics to types
        matched_types = []
        for svc_type, config in service_types.items():
            if any(indicator in characteristics for indicator in config['indicators']):
                matched_types.append({
                    'type': svc_type,
                    'slis': config['recommended_slis']
                })

        return matched_types
```

### 4.3 SLI Candidate Generation

**Request/Response Services:**

```python
class RequestResponseSLIs:
    """SLI candidates for request/response services"""

    @staticmethod
    def get_availability_sli():
        return {
            'name': 'Availability',
            'description': 'Proportion of requests that succeed',
            'formula': 'successful_requests / total_requests',
            'good_event': 'HTTP response status != 5xx',
            'total_events': 'All HTTP requests',
            'prometheus_query': '''
                sum(rate(http_requests_total{status!~"5.."}[5m])) /
                sum(rate(http_requests_total[5m]))
            ''',
            'target_guidance': {
                'critical': 99.95,
                'essential': 99.9,
                'standard': 99.5
            }
        }

    @staticmethod
    def get_latency_sli():
        return {
            'name': 'Latency',
            'description': 'Proportion of requests served within threshold',
            'formula': 'fast_requests / total_requests',
            'good_event': 'Request latency < threshold',
            'total_events': 'All requests',
            'prometheus_query': '''
                sum(rate(http_request_duration_seconds_bucket{le="0.5"}[5m])) /
                sum(rate(http_request_duration_seconds_count[5m]))
            ''',
            'threshold_guidance': {
                'critical': '100ms p99',
                'essential': '500ms p99',
                'standard': '1000ms p99'
            },
            'percentiles': [50, 95, 99, 99.9]
        }

    @staticmethod
    def get_error_rate_sli():
        return {
            'name': 'Error Rate',
            'description': 'Proportion of requests without errors',
            'formula': '1 - (error_requests / total_requests)',
            'good_event': 'HTTP response status < 500',
            'total_events': 'All requests',
            'prometheus_query': '''
                1 - (
                    sum(rate(http_requests_total{status=~"5.."}[5m])) /
                    sum(rate(http_requests_total[5m]))
                )
            ''',
            'considerations': [
                'Exclude health checks',
                'Consider client errors (4xx) separately',
                'Track error categories'
            ]
        }
```

**Data Processing Services:**

```python
class DataProcessingSLIs:
    """SLI candidates for batch/pipeline services"""

    @staticmethod
    def get_freshness_sli():
        return {
            'name': 'Data Freshness',
            'description': 'Proportion of data processed within time budget',
            'formula': 'on_time_batches / total_batches',
            'good_event': 'Batch processing time < SLA',
            'total_events': 'All batches',
            'measurement': '''
                # Track time from data arrival to completion
                freshness_seconds = completion_time - arrival_time
                on_time = freshness_seconds < freshness_sla_seconds
            ''',
            'prometheus_query': '''
                sum(rate(batch_completion_on_time[5m])) /
                sum(rate(batch_completion_total[5m]))
            '''
        }

    @staticmethod
    def get_completeness_sli():
        return {
            'name': 'Data Completeness',
            'description': 'Proportion of records successfully processed',
            'formula': 'processed_records / expected_records',
            'good_event': 'Record processed successfully',
            'total_events': 'All expected records',
            'measurement': '''
                # Compare processed vs expected
                completeness = actual_count / expected_count
            ''',
            'prometheus_query': '''
                sum(rate(records_processed_success[5m])) /
                sum(rate(records_received[5m]))
            '''
        }

    @staticmethod
    def get_correctness_sli():
        return {
            'name': 'Data Correctness',
            'description': 'Proportion of data passing quality checks',
            'formula': 'valid_records / total_records',
            'good_event': 'Record passes validation',
            'total_events': 'All processed records',
            'measurement': '''
                # Validate against business rules
                is_valid = run_data_quality_checks(record)
            ''',
            'prometheus_query': '''
                sum(rate(records_validation_passed[5m])) /
                sum(rate(records_validation_total[5m]))
            '''
        }
```

### 4.4 SLI Evaluation Criteria

**Assessing SLI Quality:**

```python
class SLIEvaluator:
    """Evaluate whether an SLI is well-defined"""

    @staticmethod
    def evaluate_sli(sli_definition):
        """
        Score an SLI definition on multiple criteria
        Returns a score from 0-100
        """
        criteria = {
            'user_centric': {
                'weight': 0.25,
                'questions': [
                    'Does this directly measure user experience?',
                    'Would users notice if this SLI degrades?',
                    'Is this a leading indicator of user satisfaction?'
                ]
            },
            'measurable': {
                'weight': 0.20,
                'questions': [
                    'Can we accurately measure this metric?',
                    'Do we have instrumentation in place?',
                    'Is the measurement reliable and consistent?'
                ]
            },
            'actionable': {
                'weight': 0.20,
                'questions': [
                    'Can we improve this metric through engineering work?',
                    'Is the metric sensitive to our changes?',
                    'Can we identify root causes when it degrades?'
                ]
            },
            'simple': {
                'weight': 0.15,
                'questions': [
                    'Is the definition easy to understand?',
                    'Can stakeholders explain what it means?',
                    'Is the calculation straightforward?'
                ]
            },
            'relevant': {
                'weight': 0.20,
                'questions': [
                    'Does this align with business goals?',
                    'Is this important to users?',
                    'Does this warrant engineering investment?'
                ]
            }
        }

        # In practice, these would be answered yes/no for each SLI
        # and weighted to produce a final score
        return criteria
```

### 4.5 SLI Specification Template

**Complete SLI Specification:**

```python
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SLISpecification:
    """Complete specification for an SLI"""

    name: str
    description: str
    sli_type: str  # availability, latency, error_rate, etc.

    # Measurement
    measurement_method: str
    measurement_interval: str
    measurement_source: str

    # Definition
    numerator_definition: str  # What counts as "good"
    denominator_definition: str  # What's the total
    exclusions: List[str]  # What to exclude from measurement

    # Implementation
    prometheus_query: Optional[str]
    datadog_query: Optional[str]
    custom_implementation: Optional[str]

    # Thresholds
    threshold_value: Optional[float]
    threshold_unit: Optional[str]

    # Metadata
    service_tier: str
    owner: str
    documentation_url: str

    def to_dict(self) -> Dict:
        return {
            'sli_specification': {
                'name': self.name,
                'description': self.description,
                'type': self.sli_type,
                'measurement': {
                    'method': self.measurement_method,
                    'interval': self.measurement_interval,
                    'source': self.measurement_source
                },
                'definition': {
                    'numerator': self.numerator_definition,
                    'denominator': self.denominator_definition,
                    'exclusions': self.exclusions
                },
                'implementation': {
                    'prometheus': self.prometheus_query,
                    'datadog': self.datadog_query,
                    'custom': self.custom_implementation
                },
                'threshold': {
                    'value': self.threshold_value,
                    'unit': self.threshold_unit
                },
                'metadata': {
                    'tier': self.service_tier,
                    'owner': self.owner,
                    'docs': self.documentation_url
                }
            }
        }

# Example usage
api_availability_sli = SLISpecification(
    name='API Availability',
    description='Percentage of API requests that complete successfully',
    sli_type='availability',
    measurement_method='ratio',
    measurement_interval='5m',
    measurement_source='prometheus',
    numerator_definition='Requests with HTTP status 200-499',
    denominator_definition='All HTTP requests excluding health checks',
    exclusions=['/health', '/metrics', '/readiness'],
    prometheus_query='''
        sum(rate(http_requests_total{status!~"5..", endpoint!~"/health|/metrics"}[5m])) /
        sum(rate(http_requests_total{endpoint!~"/health|/metrics"}[5m]))
    ''',
    datadog_query=None,
    custom_implementation=None,
    threshold_value=None,
    threshold_unit=None,
    service_tier='essential',
    owner='platform-team',
    documentation_url='https://wiki.example.com/slis/api-availability'
)
```
