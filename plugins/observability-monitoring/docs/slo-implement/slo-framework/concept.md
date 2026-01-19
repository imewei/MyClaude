# SLO Concepts and Theory

This document covers the fundamental concepts, terminology, and mathematical formulas for Service Level Objectives (SLOs).

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
