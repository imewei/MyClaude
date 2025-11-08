# SLO Framework: Comprehensive Guide and Implementation

This document provides a complete reference for implementing Service Level Objectives (SLOs), including fundamental concepts, mathematical formulas, Python implementations, and best practices for production systems.

## Table of Contents

1. [SLO Fundamentals and Terminology](#1-slo-fundamentals-and-terminology)
2. [Service Tier Classification](#2-service-tier-classification)
3. [User Journey Mapping Methodology](#3-user-journey-mapping-methodology)
4. [SLI Candidate Identification Process](#4-sli-candidate-identification-process)
5. [SLO Target Calculation Formulas](#5-slo-target-calculation-formulas)
6. [Error Budget Mathematics](#6-error-budget-mathematics)
7. [Measurement Window Selection](#7-measurement-window-selection)
8. [Python SLOFramework Implementation](#8-python-sloframework-implementation)
9. [Tier Analysis and Recommendation Engine](#9-tier-analysis-and-recommendation-engine)
10. [User Journey Templates](#10-user-journey-templates)

---

## 1. SLO Fundamentals and Terminology

### 1.1 Core Concepts

**Service Level Indicator (SLI)**
- A carefully defined quantitative measure of some aspect of the level of service that is provided
- Represents what you measure (e.g., request latency, error rate, throughput)
- Must be measurable, actionable, and meaningful to users
- Should capture user-perceived experience

**Service Level Objective (SLO)**
- A target value or range of values for a service level measured by an SLI
- Represents what you promise internally (e.g., "99.9% of requests succeed")
- Sets clear reliability goals that balance user happiness and development velocity
- Typically expressed as a percentage over a time window

**Service Level Agreement (SLA)**
- An explicit or implicit contract with your users that includes consequences of meeting (or missing) the SLOs
- Represents what you promise externally with legal/financial consequences
- Usually less strict than SLOs to provide a safety buffer
- Often includes remediation steps or compensation for violations

**Error Budget**
- The amount of unreliability you're willing to tolerate
- Calculated as: `Error Budget = 1 - SLO Target`
- Provides a common language between product and engineering
- Enables data-driven decision making about feature velocity vs. reliability

### 1.2 Relationship Between SLI, SLO, and SLA

```
User Experience
      ↓
    Measure (SLI)
      ↓
    Target (SLO)
      ↓
    Promise (SLA)
      ↓
    Consequences
```

**Example Hierarchy:**
- **SLI**: "Percentage of successful HTTP requests"
- **SLO**: "99.9% of requests succeed in a 30-day window"
- **SLA**: "99.5% of requests succeed, or customers receive 10% service credit"
- **Error Budget**: 0.1% (allows ~43 minutes of downtime per month)

### 1.3 Key SLI Types

**Availability SLIs**
```
Availability = (Successful Requests / Total Requests) × 100
```
- Measures whether the service is accessible and functioning
- Typically based on HTTP status codes (non-5xx = success)
- Most fundamental SLI for user-facing services

**Latency SLIs**
```
Latency SLI = (Fast Requests / Total Requests) × 100
```
- Measures how quickly the service responds
- Usually expressed as percentiles (p50, p95, p99)
- Critical for user experience and perceived performance

**Error Rate SLIs**
```
Success Rate = (1 - (Error Requests / Total Requests)) × 100
```
- Measures the proportion of requests without errors
- Can distinguish between client errors (4xx) and server errors (5xx)
- Often the inverse of error rate

**Throughput SLIs**
```
Throughput = Requests Processed / Time Unit
```
- Measures volume of work completed
- Important for batch processing and data pipelines
- Often combined with latency for comprehensive coverage

**Quality SLIs**
```
Quality = (High-Quality Outputs / Total Outputs) × 100
```
- Measures correctness of results
- Applicable to ML models, data pipelines, search relevance
- Requires domain-specific quality metrics

### 1.4 SLO Design Principles

1. **User-Centric**: Measure what users experience, not what's easy to measure
2. **Simple**: Start with few, clear SLOs rather than complex combinations
3. **Achievable**: Set targets that are challenging but realistic
4. **Relevant**: Tie SLOs directly to business outcomes
5. **Measurable**: Ensure you can accurately measure the SLI
6. **Actionable**: Use SLOs to drive concrete engineering decisions

### 1.5 Common Pitfalls to Avoid

- **Over-optimization**: Don't target 100% - it's impossible and wasteful
- **Too many SLOs**: Focus on 2-3 critical SLOs per service
- **Wrong metrics**: Measuring server-side metrics that don't reflect user experience
- **Static targets**: Not reviewing and adjusting SLOs based on reality
- **No consequences**: SLOs without error budgets don't drive behavior

---

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

---

## 5. SLO Target Calculation Formulas

### 5.1 Mathematical Foundations

### 5.1.1 Basic Availability Calculation

**Single Service Availability:**
```
Availability = (Uptime / Total Time) × 100
```

**Converting to "Nines":**
```python
def nines_to_downtime(nines_count, window_days=30):
    """
    Convert availability "nines" to allowed downtime

    Examples:
        99.9%   = 3 nines = 43.2 minutes/month
        99.99%  = 4 nines = 4.32 minutes/month
        99.999% = 5 nines = 26 seconds/month
    """
    availability = float('9' * nines_count) / (10 ** nines_count)
    unavailability = 1 - (availability / 100)

    total_minutes = window_days * 24 * 60
    downtime_minutes = total_minutes * unavailability

    return {
        'availability_percentage': availability,
        'downtime_minutes': downtime_minutes,
        'downtime_hours': downtime_minutes / 60,
        'downtime_seconds': downtime_minutes * 60
    }

# Example outputs:
# 99.9%  → 43.2 minutes downtime per month
# 99.99% → 4.32 minutes downtime per month
```

### 5.1.2 Composite Service Availability

**Serial Dependencies (Chain):**
```
Availability_total = Availability_1 × Availability_2 × ... × Availability_n
```

```python
def calculate_serial_availability(services):
    """
    Calculate availability of services in serial
    Example: API → Database → Cache
    """
    total_availability = 1.0

    for service in services:
        # Convert percentage to decimal
        availability_decimal = service['availability'] / 100
        total_availability *= availability_decimal

    return total_availability * 100

# Example:
services = [
    {'name': 'API', 'availability': 99.9},
    {'name': 'Database', 'availability': 99.95},
    {'name': 'Cache', 'availability': 99.99}
]

result = calculate_serial_availability(services)
# Result: 99.84% (degraded due to chain)
```

**Parallel Dependencies (Redundant):**
```
Unavailability_total = Unavailability_1 × Unavailability_2 × ... × Unavailability_n
Availability_total = 1 - Unavailability_total
```

```python
def calculate_parallel_availability(services):
    """
    Calculate availability of redundant services
    Example: Primary DB OR Secondary DB
    """
    total_unavailability = 1.0

    for service in services:
        # Convert to unavailability
        unavailability = 1 - (service['availability'] / 100)
        total_unavailability *= unavailability

    total_availability = 1 - total_unavailability
    return total_availability * 100

# Example:
redundant_services = [
    {'name': 'Primary', 'availability': 99.9},
    {'name': 'Secondary', 'availability': 99.9}
]

result = calculate_parallel_availability(redundant_services)
# Result: 99.999% (improved due to redundancy)
```

### 5.2 Latency SLO Calculations

**Percentile-Based SLO:**
```python
import numpy as np

def calculate_latency_slo(latencies, threshold_ms, target_percentile=95):
    """
    Calculate what percentage of requests meet latency threshold

    Args:
        latencies: List of latency measurements in milliseconds
        threshold_ms: Maximum acceptable latency
        target_percentile: Target percentile (e.g., 95 for p95)
    """
    # Calculate actual percentile value
    actual_pN_latency = np.percentile(latencies, target_percentile)

    # Calculate percentage meeting threshold
    fast_requests = sum(1 for l in latencies if l <= threshold_ms)
    total_requests = len(latencies)
    percentage_meeting_slo = (fast_requests / total_requests) * 100

    return {
        f'p{target_percentile}_latency_ms': actual_pN_latency,
        'threshold_ms': threshold_ms,
        'percentage_meeting_slo': percentage_meeting_slo,
        'slo_met': actual_pN_latency <= threshold_ms
    }

# Example:
latencies = [50, 75, 100, 150, 200, 250, 300, 400, 500, 1000]
result = calculate_latency_slo(latencies, threshold_ms=500, target_percentile=95)
# Result: p95 = 850ms, 90% meeting 500ms threshold
```

**Compound Latency SLO:**
```python
def calculate_compound_latency_slo(steps):
    """
    Calculate end-to-end latency for multi-step journey
    Assumes sequential steps with independent latencies
    """
    # For sequential steps: sum the percentiles
    total_latency = sum(step['latency_p95'] for step in steps)

    # For probabilistic calculation (more accurate):
    # Use Central Limit Theorem approximation
    mean = sum(step['latency_mean'] for step in steps)
    variance = sum(step['latency_variance'] for step in steps)
    std_dev = np.sqrt(variance)

    # Approximate p95 using normal distribution
    # p95 ≈ mean + 1.645 * std_dev
    p95_approximate = mean + 1.645 * std_dev

    return {
        'simple_sum_p95': total_latency,
        'statistical_p95': p95_approximate,
        'mean': mean,
        'std_dev': std_dev
    }
```

### 5.3 Error Budget Formulas

**Error Budget Calculation:**
```python
class ErrorBudgetCalculator:
    """Calculate and track error budgets"""

    @staticmethod
    def calculate_total_error_budget(slo_target, window_days=30):
        """
        Calculate total error budget for a time window

        Args:
            slo_target: SLO target as percentage (e.g., 99.9)
            window_days: Time window in days (default 30)

        Returns:
            Error budget in various units
        """
        # Calculate allowed downtime
        availability_ratio = slo_target / 100
        allowed_failure_ratio = 1 - availability_ratio

        # Calculate in different time units
        total_minutes = window_days * 24 * 60
        total_seconds = total_minutes * 60
        total_requests = 1000000  # Example: 1M requests/month

        return {
            'slo_target': slo_target,
            'window_days': window_days,
            'error_budget_ratio': allowed_failure_ratio,
            'error_budget_percentage': allowed_failure_ratio * 100,
            'allowed_downtime_minutes': total_minutes * allowed_failure_ratio,
            'allowed_downtime_seconds': total_seconds * allowed_failure_ratio,
            'allowed_downtime_hours': (total_minutes * allowed_failure_ratio) / 60,
            'allowed_failed_requests': total_requests * allowed_failure_ratio
        }

    @staticmethod
    def calculate_remaining_budget(slo_target, actual_performance, elapsed_ratio):
        """
        Calculate remaining error budget

        Args:
            slo_target: Target SLO (e.g., 99.9)
            actual_performance: Actual performance so far (e.g., 99.85)
            elapsed_ratio: Portion of time window elapsed (e.g., 0.5 for halfway)

        Returns:
            Remaining budget and burn rate
        """
        # Total budget
        total_budget = (1 - slo_target / 100)

        # Consumed budget
        consumed_budget = (1 - actual_performance / 100)

        # Remaining budget
        remaining_budget = total_budget - consumed_budget
        remaining_percentage = (remaining_budget / total_budget) * 100

        # Burn rate (how fast we're using the budget)
        expected_consumption = total_budget * elapsed_ratio
        actual_consumption = consumed_budget
        burn_rate = actual_consumption / expected_consumption if expected_consumption > 0 else 0

        # Project when budget will be exhausted
        if burn_rate > 0 and remaining_budget > 0:
            days_remaining = (remaining_budget / consumed_budget) * (elapsed_ratio * 30)
        else:
            days_remaining = float('inf')

        return {
            'total_budget': total_budget,
            'consumed_budget': consumed_budget,
            'remaining_budget': remaining_budget,
            'remaining_percentage': remaining_percentage,
            'burn_rate': burn_rate,
            'days_until_exhaustion': days_remaining,
            'status': ErrorBudgetCalculator._determine_status(remaining_percentage, burn_rate)
        }

    @staticmethod
    def _determine_status(remaining_percentage, burn_rate):
        """Determine error budget health status"""
        if remaining_percentage <= 0:
            return 'exhausted'
        elif burn_rate > 2:
            return 'critical'
        elif burn_rate > 1.5:
            return 'warning'
        elif burn_rate > 1:
            return 'attention'
        else:
            return 'healthy'

# Example usage:
budget = ErrorBudgetCalculator.calculate_total_error_budget(
    slo_target=99.9,
    window_days=30
)
# Result: 43.2 minutes allowed downtime per month

remaining = ErrorBudgetCalculator.calculate_remaining_budget(
    slo_target=99.9,
    actual_performance=99.85,  # Performing below target
    elapsed_ratio=0.5  # Halfway through the month
)
# Result: burn_rate > 1 means using budget faster than expected
```

### 5.4 Request-Based Error Budget

```python
def calculate_request_based_error_budget(total_requests, success_rate_target):
    """
    Calculate error budget in terms of failed requests

    Args:
        total_requests: Expected number of requests in the window
        success_rate_target: Target success rate (e.g., 99.9)

    Returns:
        Number of requests that can fail while meeting SLO
    """
    failure_ratio = 1 - (success_rate_target / 100)
    allowed_failures = int(total_requests * failure_ratio)

    return {
        'total_requests': total_requests,
        'success_rate_target': success_rate_target,
        'allowed_failures': allowed_failures,
        'required_successes': total_requests - allowed_failures,
        'failure_ratio': failure_ratio
    }

# Example:
budget = calculate_request_based_error_budget(
    total_requests=10_000_000,  # 10M requests/month
    success_rate_target=99.9
)
# Result: 10,000 requests can fail (0.1% of 10M)
```

### 5.5 Multi-Window SLO Calculations

```python
def calculate_multi_window_slo(measurements, windows):
    """
    Calculate SLO compliance across multiple time windows

    Args:
        measurements: List of (timestamp, success) tuples
        windows: List of window sizes in minutes

    Returns:
        SLO compliance for each window
    """
    results = {}

    for window_minutes in windows:
        window_seconds = window_minutes * 60
        cutoff_time = time.time() - window_seconds

        # Filter measurements within window
        window_measurements = [
            m for m in measurements
            if m['timestamp'] >= cutoff_time
        ]

        if window_measurements:
            successes = sum(1 for m in window_measurements if m['success'])
            total = len(window_measurements)
            success_rate = (successes / total) * 100
        else:
            success_rate = None

        results[f'{window_minutes}m'] = {
            'success_rate': success_rate,
            'total_requests': len(window_measurements),
            'successful_requests': successes if window_measurements else 0
        }

    return results
```

---

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

---

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

---

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

---

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

---

## Conclusion

This comprehensive SLO framework documentation provides:

1. **Fundamental Understanding**: Clear definitions of SLI, SLO, SLA, and error budgets
2. **Service Classification**: Detailed tier system (critical/essential/standard/best-effort) with specific criteria
3. **User Journey Mapping**: Methodology for identifying and measuring critical user paths
4. **SLI Selection**: Process for choosing meaningful service level indicators
5. **Mathematical Foundation**: Formulas and calculations for SLO targets and error budgets
6. **Error Budget Management**: Mathematics, burn rate calculations, and alerting strategies
7. **Window Selection**: Rolling vs calendar windows with statistical guidance
8. **Production Implementation**: Complete Python classes ready for production use
9. **Tier Analysis Engine**: Automated tier recommendation based on service characteristics
10. **Journey Templates**: Pre-built templates for common application patterns

Use this framework to establish reliability standards, measure service performance, and make data-driven decisions about reliability versus feature development velocity.
