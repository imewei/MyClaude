# SLI Measurement Implementation Guide

Comprehensive guide for implementing Service Level Indicators (SLIs) across different service types with production-ready code, Prometheus queries, and testing strategies.

## Table of Contents

1. [SLI Fundamentals](#sli-fundamentals)
2. [SLI Types Overview](#sli-types-overview)
3. [API Service SLIs](#api-service-slis)
4. [Web Application SLIs](#web-application-slis)
5. [Batch Pipeline SLIs](#batch-pipeline-slis)
6. [Streaming Service SLIs](#streaming-service-slis)
7. [Client-Side SLI Measurement](#client-side-sli-measurement)
8. [SLI Implementation Patterns](#sli-implementation-patterns)
9. [Python SLI Classes](#python-sli-classes)
10. [Prometheus Query Examples](#prometheus-query-examples)
11. [SLI Validation and Testing](#sli-validation-and-testing)

---

## SLI Fundamentals

Service Level Indicators (SLIs) are quantitative measures of service behavior that reflect the user experience. They form the foundation for Service Level Objectives (SLOs) and error budget calculations.

### SLI Design Principles

1. **User-Centric**: Measure what users actually experience
2. **Actionable**: Provide signals for engineering decisions
3. **Proportional**: Reflect the proportion of good experiences
4. **Measurable**: Can be accurately and reliably measured
5. **Meaningful**: Correlate with user satisfaction

### SLI Specification Template

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class SLIType(Enum):
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    QUALITY = "quality"
    FRESHNESS = "freshness"
    COMPLETENESS = "completeness"
    CORRECTNESS = "correctness"

@dataclass
class SLISpecification:
    """
    Complete SLI specification for a service
    """
    name: str
    description: str
    sli_type: SLIType

    # Definition
    good_events_definition: str
    total_events_definition: str

    # Implementation
    measurement_source: str  # e.g., "prometheus", "logs", "traces"
    query: str
    aggregation_window: str  # e.g., "5m", "1h"

    # Metadata
    service: str
    owner: str
    runbook_url: Optional[str] = None

    # Validation
    expected_range: tuple = (0.0, 100.0)
    alert_threshold: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for export"""
        return {
            'name': self.name,
            'description': self.description,
            'type': self.sli_type.value,
            'good_events': self.good_events_definition,
            'total_events': self.total_events_definition,
            'measurement': {
                'source': self.measurement_source,
                'query': self.query,
                'window': self.aggregation_window
            },
            'metadata': {
                'service': self.service,
                'owner': self.owner,
                'runbook': self.runbook_url
            }
        }

# Example: API Availability SLI
api_availability_sli = SLISpecification(
    name="api_availability",
    description="Percentage of successful API requests (non-5xx responses)",
    sli_type=SLIType.AVAILABILITY,
    good_events_definition="HTTP requests with status code != 5xx",
    total_events_definition="All HTTP requests",
    measurement_source="prometheus",
    query="""
        sum(rate(http_requests_total{status!~"5.."}[5m])) /
        sum(rate(http_requests_total[5m])) * 100
    """,
    aggregation_window="5m",
    service="api-service",
    owner="platform-team",
    runbook_url="https://runbooks.example.com/api-availability",
    expected_range=(95.0, 100.0),
    alert_threshold=99.0
)
```

---

## SLI Types Overview

### 1. Availability SLI

**Definition**: Proportion of valid requests that were served successfully

**When to Use**:
- Request/response services
- APIs and web applications
- Critical infrastructure services

**Formula**: `successful_requests / total_requests * 100`

**Considerations**:
- Define what "successful" means (2xx only? 2xx + 4xx?)
- Handle edge cases (health checks, retries)
- Consider user vs. system initiated requests

### 2. Latency SLI

**Definition**: Proportion of requests served faster than a threshold

**When to Use**:
- Interactive applications
- Real-time services
- User-facing endpoints

**Formula**: `fast_requests / total_requests * 100`

**Considerations**:
- Choose appropriate percentile (p50, p95, p99)
- Set meaningful thresholds based on user perception
- Account for network latency vs. service latency

### 3. Error Rate SLI

**Definition**: Proportion of requests without errors

**When to Use**:
- Services with complex business logic
- Data processing pipelines
- Integration points

**Formula**: `(1 - error_requests / total_requests) * 100`

**Considerations**:
- Distinguish between client errors (4xx) and server errors (5xx)
- Include business logic errors
- Weight errors by impact

### 4. Throughput SLI

**Definition**: Proportion of time the service meets throughput requirements

**When to Use**:
- High-volume services
- Batch processing systems
- Message queues

**Formula**: `actual_throughput / expected_throughput * 100`

**Considerations**:
- Define expected throughput (requests/sec, messages/sec)
- Account for traffic patterns
- Consider sustainable vs. peak throughput

### 5. Quality SLI

**Definition**: Proportion of outputs that meet quality criteria

**When to Use**:
- ML/AI services
- Data processing pipelines
- Content delivery

**Formula**: `high_quality_outputs / total_outputs * 100`

**Considerations**:
- Define quality metrics (accuracy, precision, completeness)
- Balance quality with performance
- Include human validation where needed

---

## API Service SLIs

### Availability SLI for APIs

```python
class APIAvailabilitySLI:
    """
    Measure API availability based on HTTP status codes
    """

    def __init__(self, prometheus_client, service_name: str):
        self.prom = prometheus_client
        self.service = service_name

    def calculate(self, time_range: str = '5m') -> float:
        """
        Calculate API availability over time range

        Args:
            time_range: Prometheus time range (e.g., '5m', '1h', '1d')

        Returns:
            Availability percentage (0-100)
        """
        query = f"""
        sum(rate(http_requests_total{{
            service="{self.service}",
            status!~"5.."
        }}[{time_range}])) /
        sum(rate(http_requests_total{{
            service="{self.service}"
        }}[{time_range}])) * 100
        """

        result = self.prom.query(query)
        if result and len(result) > 0:
            return float(result[0]['value'][1])
        return 0.0

    def calculate_with_exclusions(
        self,
        time_range: str = '5m',
        exclude_endpoints: List[str] = None,
        exclude_methods: List[str] = None
    ) -> float:
        """
        Calculate availability excluding specific endpoints or methods

        Args:
            time_range: Prometheus time range
            exclude_endpoints: List of endpoint patterns to exclude (regex)
            exclude_methods: List of HTTP methods to exclude

        Returns:
            Availability percentage
        """
        exclude_filters = []

        if exclude_endpoints:
            endpoint_pattern = '|'.join(exclude_endpoints)
            exclude_filters.append(f'endpoint!~"{endpoint_pattern}"')

        if exclude_methods:
            method_pattern = '|'.join(exclude_methods)
            exclude_filters.append(f'method!~"{method_pattern}"')

        filter_str = ','.join(exclude_filters)

        query = f"""
        sum(rate(http_requests_total{{
            service="{self.service}",
            status!~"5..",
            {filter_str}
        }}[{time_range}])) /
        sum(rate(http_requests_total{{
            service="{self.service}",
            {filter_str}
        }}[{time_range}])) * 100
        """

        result = self.prom.query(query)
        return float(result[0]['value'][1]) if result else 0.0

    def calculate_by_endpoint(self, time_range: str = '5m') -> Dict[str, float]:
        """
        Calculate availability broken down by endpoint

        Returns:
            Dictionary mapping endpoint to availability percentage
        """
        query = f"""
        sum(rate(http_requests_total{{
            service="{self.service}",
            status!~"5.."
        }}[{time_range}])) by (endpoint) /
        sum(rate(http_requests_total{{
            service="{self.service}"
        }}[{time_range}])) by (endpoint) * 100
        """

        results = self.prom.query(query)

        availability_by_endpoint = {}
        for result in results:
            endpoint = result['metric'].get('endpoint', 'unknown')
            availability = float(result['value'][1])
            availability_by_endpoint[endpoint] = availability

        return availability_by_endpoint

    def calculate_critical_endpoints_only(
        self,
        critical_endpoints: List[str],
        time_range: str = '5m'
    ) -> float:
        """
        Calculate availability for critical endpoints only

        Args:
            critical_endpoints: List of critical endpoint patterns
            time_range: Prometheus time range

        Returns:
            Availability percentage for critical endpoints
        """
        endpoint_pattern = '|'.join(critical_endpoints)

        query = f"""
        sum(rate(http_requests_total{{
            service="{self.service}",
            status!~"5..",
            endpoint=~"{endpoint_pattern}"
        }}[{time_range}])) /
        sum(rate(http_requests_total{{
            service="{self.service}",
            endpoint=~"{endpoint_pattern}"
        }}[{time_range}])) * 100
        """

        result = self.prom.query(query)
        return float(result[0]['value'][1]) if result else 0.0


# Prometheus Recording Rules for API Availability
API_AVAILABILITY_RECORDING_RULES = """
groups:
  - name: api_availability_sli
    interval: 30s
    rules:
      # Overall API availability
      - record: service:availability:ratio_rate5m
        expr: |
          sum(rate(http_requests_total{status!~"5.."}[5m])) by (service) /
          sum(rate(http_requests_total[5m])) by (service)

      # API availability by endpoint
      - record: service:availability:ratio_rate5m:by_endpoint
        expr: |
          sum(rate(http_requests_total{status!~"5.."}[5m])) by (service, endpoint) /
          sum(rate(http_requests_total[5m])) by (service, endpoint)

      # API availability excluding health checks
      - record: service:availability:ratio_rate5m:no_health
        expr: |
          sum(rate(http_requests_total{
            status!~"5..",
            endpoint!~"/health|/healthz|/ready"
          }[5m])) by (service) /
          sum(rate(http_requests_total{
            endpoint!~"/health|/healthz|/ready"
          }[5m])) by (service)
"""
```

### Latency SLI for APIs

```python
class APILatencySLI:
    """
    Measure API latency based on request duration
    """

    def __init__(self, prometheus_client, service_name: str):
        self.prom = prometheus_client
        self.service = service_name

    def calculate_latency_sli(
        self,
        thresholds_ms: Dict[str, float],
        time_range: str = '5m'
    ) -> Dict[str, Dict]:
        """
        Calculate latency SLI for multiple thresholds

        Args:
            thresholds_ms: Dictionary of percentile to threshold mapping
                          e.g., {'p50': 100, 'p95': 500, 'p99': 1000}
            time_range: Prometheus time range

        Returns:
            Dictionary with latency SLI values for each threshold
        """
        slis = {}

        for percentile, threshold_ms in thresholds_ms.items():
            threshold_sec = threshold_ms / 1000.0

            # Calculate proportion of requests faster than threshold
            query = f"""
            sum(rate(http_request_duration_seconds_bucket{{
                service="{self.service}",
                le="{threshold_sec}"
            }}[{time_range}])) /
            sum(rate(http_request_duration_seconds_count{{
                service="{self.service}"
            }}[{time_range}])) * 100
            """

            result = self.prom.query(query)
            value = float(result[0]['value'][1]) if result else 0.0

            slis[f'latency_{percentile}'] = {
                'value': value,
                'threshold_ms': threshold_ms,
                'met': value >= 95.0  # 95% of requests should be fast
            }

        return slis

    def calculate_percentile_latency(
        self,
        percentiles: List[float],
        time_range: str = '5m'
    ) -> Dict[str, float]:
        """
        Calculate actual latency percentiles

        Args:
            percentiles: List of percentiles (e.g., [0.50, 0.95, 0.99])
            time_range: Prometheus time range

        Returns:
            Dictionary mapping percentile to latency in milliseconds
        """
        results = {}

        for p in percentiles:
            query = f"""
            histogram_quantile({p},
                sum(rate(http_request_duration_seconds_bucket{{
                    service="{self.service}"
                }}[{time_range}])) by (le)
            ) * 1000
            """

            result = self.prom.query(query)
            latency_ms = float(result[0]['value'][1]) if result else 0.0

            percentile_label = f'p{int(p * 100)}'
            results[percentile_label] = latency_ms

        return results

    def calculate_user_centric_latency(
        self,
        include_client_time: bool = True,
        time_range: str = '5m'
    ) -> Dict[str, float]:
        """
        Calculate latency from user perspective including client-side time

        Args:
            include_client_time: Whether to include client-side metrics
            time_range: Prometheus time range

        Returns:
            Dictionary with server and total latency metrics
        """
        # Server-side latency (p95)
        server_query = f"""
        histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket{{
                service="{self.service}"
            }}[{time_range}])) by (le)
        ) * 1000
        """

        server_result = self.prom.query(server_query)
        server_latency = float(server_result[0]['value'][1]) if server_result else 0.0

        results = {
            'server_p95_ms': server_latency
        }

        if include_client_time:
            # Total user-perceived latency (including network, client processing)
            user_query = f"""
            histogram_quantile(0.95,
                sum(rate(user_request_duration_bucket{{
                    service="{self.service}"
                }}[{time_range}])) by (le)
            ) * 1000
            """

            user_result = self.prom.query(user_query)
            user_latency = float(user_result[0]['value'][1]) if user_result else 0.0

            results['user_perceived_p95_ms'] = user_latency
            results['network_overhead_ms'] = user_latency - server_latency

        return results

    def calculate_by_endpoint(
        self,
        percentile: float = 0.95,
        time_range: str = '5m'
    ) -> Dict[str, float]:
        """
        Calculate latency percentile by endpoint

        Args:
            percentile: Percentile to calculate (e.g., 0.95 for p95)
            time_range: Prometheus time range

        Returns:
            Dictionary mapping endpoint to latency in milliseconds
        """
        query = f"""
        histogram_quantile({percentile},
            sum(rate(http_request_duration_seconds_bucket{{
                service="{self.service}"
            }}[{time_range}])) by (endpoint, le)
        ) * 1000
        """

        results = self.prom.query(query)

        latency_by_endpoint = {}
        for result in results:
            endpoint = result['metric'].get('endpoint', 'unknown')
            latency_ms = float(result['value'][1])
            latency_by_endpoint[endpoint] = latency_ms

        return latency_by_endpoint


# Prometheus Recording Rules for API Latency
API_LATENCY_RECORDING_RULES = """
groups:
  - name: api_latency_sli
    interval: 30s
    rules:
      # Latency percentiles
      - record: service:latency_p50:seconds
        expr: |
          histogram_quantile(0.50,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:latency_p95:seconds
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:latency_p99:seconds
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      # Proportion of fast requests (< 500ms)
      - record: service:latency_sli:ratio_rate5m
        expr: |
          sum(rate(http_request_duration_seconds_bucket{le="0.5"}[5m])) by (service) /
          sum(rate(http_request_duration_seconds_count[5m])) by (service)

      # Latency by endpoint
      - record: service:latency_p95:seconds:by_endpoint
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, endpoint, le)
          )
"""
```

### Error Rate SLI for APIs

```python
class APIErrorRateSLI:
    """
    Measure API error rate with categorization
    """

    def __init__(self, prometheus_client, service_name: str):
        self.prom = prometheus_client
        self.service = service_name

    def calculate_error_rate(
        self,
        time_range: str = '5m',
        categorize: bool = True
    ) -> Dict:
        """
        Calculate error rate with optional categorization

        Args:
            time_range: Prometheus time range
            categorize: Whether to break down by error category

        Returns:
            Dictionary with error rates
        """
        results = {}

        if categorize:
            # Different error categories
            error_categories = {
                'client_errors_4xx': 'status=~"4.."',
                'server_errors_5xx': 'status=~"5.."',
                'timeout_errors_504': 'status="504"',
                'gateway_errors_502_503': 'status=~"502|503"',
                'internal_errors_500': 'status="500"'
            }

            for category, filter_expr in error_categories.items():
                query = f"""
                sum(rate(http_requests_total{{
                    service="{self.service}",
                    {filter_expr}
                }}[{time_range}])) /
                sum(rate(http_requests_total{{
                    service="{self.service}"
                }}[{time_range}])) * 100
                """

                result = self.prom.query(query)
                results[category] = float(result[0]['value'][1]) if result else 0.0

        # Overall success rate (excluding 5xx)
        overall_query = f"""
        (1 - sum(rate(http_requests_total{{
            service="{self.service}",
            status=~"5.."
        }}[{time_range}])) /
        sum(rate(http_requests_total{{
            service="{self.service}"
        }}[{time_range}]))) * 100
        """

        overall_result = self.prom.query(overall_query)
        results['overall_success_rate'] = (
            float(overall_result[0]['value'][1]) if overall_result else 0.0
        )

        # Calculate error budget impact
        results['error_rate_5xx'] = 100.0 - results['overall_success_rate']

        return results

    def calculate_by_error_type(self, time_range: str = '5m') -> Dict[str, float]:
        """
        Calculate error rate by custom error types (business logic errors)

        Args:
            time_range: Prometheus time range

        Returns:
            Dictionary mapping error type to error rate
        """
        query = f"""
        sum(rate(application_errors_total{{
            service="{self.service}"
        }}[{time_range}])) by (error_type) /
        sum(rate(http_requests_total{{
            service="{self.service}"
        }}[{time_range}])) * 100
        """

        results = self.prom.query(query)

        error_rates = {}
        for result in results:
            error_type = result['metric'].get('error_type', 'unknown')
            rate = float(result['value'][1])
            error_rates[error_type] = rate

        return error_rates

    def calculate_weighted_error_rate(
        self,
        weights: Dict[str, float],
        time_range: str = '5m'
    ) -> float:
        """
        Calculate weighted error rate based on error severity

        Args:
            weights: Dictionary mapping status code pattern to weight
                    e.g., {'5..': 1.0, '429': 0.5, '4..': 0.1}
            time_range: Prometheus time range

        Returns:
            Weighted error rate
        """
        total_weight = 0.0
        total_requests = 0.0

        for status_pattern, weight in weights.items():
            # Get count of errors matching this pattern
            error_query = f"""
            sum(increase(http_requests_total{{
                service="{self.service}",
                status=~"{status_pattern}"
            }}[{time_range}]))
            """

            error_result = self.prom.query(error_query)
            error_count = float(error_result[0]['value'][1]) if error_result else 0.0

            total_weight += error_count * weight

        # Get total requests
        total_query = f"""
        sum(increase(http_requests_total{{
            service="{self.service}"
        }}[{time_range}]))
        """

        total_result = self.prom.query(total_query)
        total_requests = float(total_result[0]['value'][1]) if total_result else 1.0

        # Calculate weighted error rate
        weighted_rate = (total_weight / total_requests) * 100 if total_requests > 0 else 0.0

        return weighted_rate


# Prometheus Recording Rules for API Error Rate
API_ERROR_RATE_RECORDING_RULES = """
groups:
  - name: api_error_rate_sli
    interval: 30s
    rules:
      # Overall error rate (5xx only)
      - record: service:error_rate_5xx:ratio_rate5m
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service) /
          sum(rate(http_requests_total[5m])) by (service)

      # Success rate (inverse of error rate)
      - record: service:success_rate:ratio_rate5m
        expr: |
          1 - service:error_rate_5xx:ratio_rate5m

      # Error rate by status code
      - record: service:error_rate:ratio_rate5m:by_status
        expr: |
          sum(rate(http_requests_total{status=~"[45].."}[5m])) by (service, status) /
          sum(rate(http_requests_total[5m])) by (service)

      # Error rate by endpoint
      - record: service:error_rate:ratio_rate5m:by_endpoint
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service, endpoint) /
          sum(rate(http_requests_total[5m])) by (service, endpoint)
"""
```

---

## Web Application SLIs

### Core Web Vitals Integration

```python
class WebApplicationSLI:
    """
    SLIs for web applications incorporating Core Web Vitals
    """

    def __init__(self, prometheus_client, app_name: str):
        self.prom = prometheus_client
        self.app = app_name

    def calculate_lcp_sli(
        self,
        threshold_ms: float = 2500,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate Largest Contentful Paint (LCP) SLI

        LCP measures loading performance. Good LCP is < 2.5s

        Args:
            threshold_ms: LCP threshold in milliseconds
            time_range: Prometheus time range

        Returns:
            Dictionary with LCP metrics and SLI
        """
        # Proportion of page loads with good LCP
        sli_query = f"""
        sum(rate(web_vitals_lcp_bucket{{
            app="{self.app}",
            le="{threshold_ms}"
        }}[{time_range}])) /
        sum(rate(web_vitals_lcp_count{{
            app="{self.app}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # Actual p75 LCP
        p75_query = f"""
        histogram_quantile(0.75,
            sum(rate(web_vitals_lcp_bucket{{
                app="{self.app}"
            }}[{time_range}])) by (le)
        )
        """

        p75_result = self.prom.query(p75_query)
        p75_lcp = float(p75_result[0]['value'][1]) if p75_result else 0.0

        return {
            'lcp_sli_percentage': sli_value,
            'lcp_p75_ms': p75_lcp,
            'threshold_ms': threshold_ms,
            'met': sli_value >= 75.0  # 75% of page loads should have good LCP
        }

    def calculate_fid_sli(
        self,
        threshold_ms: float = 100,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate First Input Delay (FID) SLI

        FID measures interactivity. Good FID is < 100ms

        Args:
            threshold_ms: FID threshold in milliseconds
            time_range: Prometheus time range

        Returns:
            Dictionary with FID metrics and SLI
        """
        sli_query = f"""
        sum(rate(web_vitals_fid_bucket{{
            app="{self.app}",
            le="{threshold_ms}"
        }}[{time_range}])) /
        sum(rate(web_vitals_fid_count{{
            app="{self.app}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # Actual p75 FID
        p75_query = f"""
        histogram_quantile(0.75,
            sum(rate(web_vitals_fid_bucket{{
                app="{self.app}"
            }}[{time_range}])) by (le)
        )
        """

        p75_result = self.prom.query(p75_query)
        p75_fid = float(p75_result[0]['value'][1]) if p75_result else 0.0

        return {
            'fid_sli_percentage': sli_value,
            'fid_p75_ms': p75_fid,
            'threshold_ms': threshold_ms,
            'met': sli_value >= 75.0
        }

    def calculate_cls_sli(
        self,
        threshold: float = 0.1,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate Cumulative Layout Shift (CLS) SLI

        CLS measures visual stability. Good CLS is < 0.1

        Args:
            threshold: CLS threshold (dimensionless)
            time_range: Prometheus time range

        Returns:
            Dictionary with CLS metrics and SLI
        """
        sli_query = f"""
        sum(rate(web_vitals_cls_bucket{{
            app="{self.app}",
            le="{threshold}"
        }}[{time_range}])) /
        sum(rate(web_vitals_cls_count{{
            app="{self.app}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # Actual p75 CLS
        p75_query = f"""
        histogram_quantile(0.75,
            sum(rate(web_vitals_cls_bucket{{
                app="{self.app}"
            }}[{time_range}])) by (le)
        )
        """

        p75_result = self.prom.query(p75_query)
        p75_cls = float(p75_result[0]['value'][1]) if p75_result else 0.0

        return {
            'cls_sli_percentage': sli_value,
            'cls_p75': p75_cls,
            'threshold': threshold,
            'met': sli_value >= 75.0
        }

    def calculate_composite_web_vitals_sli(
        self,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate composite SLI based on all Core Web Vitals

        Returns:
            Dictionary with individual and composite SLI scores
        """
        lcp = self.calculate_lcp_sli(time_range=time_range)
        fid = self.calculate_fid_sli(time_range=time_range)
        cls = self.calculate_cls_sli(time_range=time_range)

        # Composite score: all three must meet thresholds
        # This represents the proportion of page loads that pass all vitals
        composite_query = f"""
        sum(rate(web_vitals_lcp_bucket{{
            app="{self.app}",
            le="2500"
        }}[{time_range}]) and
        rate(web_vitals_fid_bucket{{
            app="{self.app}",
            le="100"
        }}[{time_range}]) and
        rate(web_vitals_cls_bucket{{
            app="{self.app}",
            le="0.1"
        }}[{time_range}])) /
        sum(rate(web_vitals_lcp_count{{
            app="{self.app}"
        }}[{time_range}])) * 100
        """

        composite_result = self.prom.query(composite_query)
        composite_value = float(composite_result[0]['value'][1]) if composite_result else 0.0

        return {
            'lcp': lcp,
            'fid': fid,
            'cls': cls,
            'composite_sli': composite_value,
            'all_met': lcp['met'] and fid['met'] and cls['met']
        }


# Prometheus Recording Rules for Web Application SLIs
WEB_APP_RECORDING_RULES = """
groups:
  - name: web_app_sli
    interval: 30s
    rules:
      # LCP SLI (< 2.5s)
      - record: app:lcp_sli:ratio_rate5m
        expr: |
          sum(rate(web_vitals_lcp_bucket{le="2500"}[5m])) by (app) /
          sum(rate(web_vitals_lcp_count[5m])) by (app)

      # FID SLI (< 100ms)
      - record: app:fid_sli:ratio_rate5m
        expr: |
          sum(rate(web_vitals_fid_bucket{le="100"}[5m])) by (app) /
          sum(rate(web_vitals_fid_count[5m])) by (app)

      # CLS SLI (< 0.1)
      - record: app:cls_sli:ratio_rate5m
        expr: |
          sum(rate(web_vitals_cls_bucket{le="0.1"}[5m])) by (app) /
          sum(rate(web_vitals_cls_count[5m])) by (app)

      # Composite Core Web Vitals SLI
      - record: app:web_vitals_composite_sli:ratio_rate5m
        expr: |
          app:lcp_sli:ratio_rate5m *
          app:fid_sli:ratio_rate5m *
          app:cls_sli:ratio_rate5m

      # Page load success rate
      - record: app:page_load_success:ratio_rate5m
        expr: |
          sum(rate(page_loads_total{status="success"}[5m])) by (app) /
          sum(rate(page_loads_total[5m])) by (app)
"""
```

---

## Batch Pipeline SLIs

### Freshness, Completeness, and Accuracy SLIs

```python
class BatchPipelineSLI:
    """
    SLIs for batch processing pipelines
    """

    def __init__(self, prometheus_client, pipeline_name: str):
        self.prom = prometheus_client
        self.pipeline = pipeline_name

    def calculate_freshness_sli(
        self,
        sla_minutes: int = 30,
        time_range: str = '1h'
    ) -> Dict:
        """
        Calculate data freshness SLI

        Measures whether data is processed within the SLA time

        Args:
            sla_minutes: Maximum acceptable processing time
            time_range: Prometheus time range

        Returns:
            Dictionary with freshness metrics
        """
        # Proportion of batches processed within SLA
        sli_query = f"""
        sum(rate(pipeline_batch_processing_duration_bucket{{
            pipeline="{self.pipeline}",
            le="{sla_minutes * 60}"
        }}[{time_range}])) /
        sum(rate(pipeline_batch_processing_duration_count{{
            pipeline="{self.pipeline}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # Actual p95 processing time
        p95_query = f"""
        histogram_quantile(0.95,
            sum(rate(pipeline_batch_processing_duration_bucket{{
                pipeline="{self.pipeline}"
            }}[{time_range}])) by (le)
        ) / 60
        """

        p95_result = self.prom.query(p95_query)
        p95_minutes = float(p95_result[0]['value'][1]) if p95_result else 0.0

        # Data age (time since last successful batch)
        age_query = f"""
        time() - max(pipeline_batch_completion_timestamp{{
            pipeline="{self.pipeline}",
            status="success"
        }})
        """

        age_result = self.prom.query(age_query)
        data_age_seconds = float(age_result[0]['value'][1]) if age_result else 0.0

        return {
            'freshness_sli_percentage': sli_value,
            'p95_processing_time_minutes': p95_minutes,
            'sla_minutes': sla_minutes,
            'current_data_age_minutes': data_age_seconds / 60,
            'met': sli_value >= 99.0 and data_age_seconds < (sla_minutes * 60)
        }

    def calculate_completeness_sli(
        self,
        time_range: str = '1h'
    ) -> Dict:
        """
        Calculate data completeness SLI

        Measures whether all expected data is processed

        Args:
            time_range: Prometheus time range

        Returns:
            Dictionary with completeness metrics
        """
        # Proportion of records successfully processed
        sli_query = f"""
        sum(increase(pipeline_records_processed_total{{
            pipeline="{self.pipeline}",
            status="success"
        }}[{time_range}])) /
        sum(increase(pipeline_records_received_total{{
            pipeline="{self.pipeline}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # Count of records
        received_query = f"""
        sum(increase(pipeline_records_received_total{{
            pipeline="{self.pipeline}"
        }}[{time_range}]))
        """

        processed_query = f"""
        sum(increase(pipeline_records_processed_total{{
            pipeline="{self.pipeline}",
            status="success"
        }}[{time_range}]))
        """

        received = self.prom.query(received_query)
        processed = self.prom.query(processed_query)

        records_received = int(received[0]['value'][1]) if received else 0
        records_processed = int(processed[0]['value'][1]) if processed else 0
        records_lost = records_received - records_processed

        return {
            'completeness_sli_percentage': sli_value,
            'records_received': records_received,
            'records_processed': records_processed,
            'records_lost': records_lost,
            'met': sli_value >= 99.95
        }

    def calculate_accuracy_sli(
        self,
        time_range: str = '1h'
    ) -> Dict:
        """
        Calculate data accuracy/quality SLI

        Measures whether processed data meets quality standards

        Args:
            time_range: Prometheus time range

        Returns:
            Dictionary with accuracy metrics
        """
        # Proportion of records passing validation
        sli_query = f"""
        sum(increase(pipeline_records_validated_total{{
            pipeline="{self.pipeline}",
            result="passed"
        }}[{time_range}])) /
        sum(increase(pipeline_records_validated_total{{
            pipeline="{self.pipeline}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # Validation failure breakdown
        failure_query = f"""
        sum(increase(pipeline_records_validated_total{{
            pipeline="{self.pipeline}",
            result="failed"
        }}[{time_range}])) by (validation_error)
        """

        failure_results = self.prom.query(failure_query)

        failures_by_type = {}
        for result in failure_results:
            error_type = result['metric'].get('validation_error', 'unknown')
            count = int(result['value'][1])
            failures_by_type[error_type] = count

        return {
            'accuracy_sli_percentage': sli_value,
            'validation_failures_by_type': failures_by_type,
            'met': sli_value >= 99.9
        }

    def calculate_composite_pipeline_sli(
        self,
        time_range: str = '1h'
    ) -> Dict:
        """
        Calculate composite SLI for the entire pipeline

        All three aspects (freshness, completeness, accuracy) must be met

        Returns:
            Dictionary with all SLI components and composite score
        """
        freshness = self.calculate_freshness_sli(time_range=time_range)
        completeness = self.calculate_completeness_sli(time_range=time_range)
        accuracy = self.calculate_accuracy_sli(time_range=time_range)

        # Composite: all must be met
        all_met = freshness['met'] and completeness['met'] and accuracy['met']

        # Weighted composite score
        composite_score = (
            freshness['freshness_sli_percentage'] * 0.3 +
            completeness['completeness_sli_percentage'] * 0.4 +
            accuracy['accuracy_sli_percentage'] * 0.3
        )

        return {
            'freshness': freshness,
            'completeness': completeness,
            'accuracy': accuracy,
            'composite_score': composite_score,
            'all_met': all_met
        }


# Prometheus Recording Rules for Batch Pipeline SLIs
BATCH_PIPELINE_RECORDING_RULES = """
groups:
  - name: batch_pipeline_sli
    interval: 60s
    rules:
      # Freshness SLI (< 30 minutes)
      - record: pipeline:freshness_sli:ratio_rate1h
        expr: |
          sum(rate(pipeline_batch_processing_duration_bucket{le="1800"}[1h])) by (pipeline) /
          sum(rate(pipeline_batch_processing_duration_count[1h])) by (pipeline)

      # Completeness SLI
      - record: pipeline:completeness_sli:ratio_rate1h
        expr: |
          sum(increase(pipeline_records_processed_total{status="success"}[1h])) by (pipeline) /
          sum(increase(pipeline_records_received_total[1h])) by (pipeline)

      # Accuracy SLI
      - record: pipeline:accuracy_sli:ratio_rate1h
        expr: |
          sum(increase(pipeline_records_validated_total{result="passed"}[1h])) by (pipeline) /
          sum(increase(pipeline_records_validated_total[1h])) by (pipeline)

      # Data age (minutes since last successful batch)
      - record: pipeline:data_age:minutes
        expr: |
          (time() - max(pipeline_batch_completion_timestamp{status="success"}) by (pipeline)) / 60

      # Composite pipeline SLI
      - record: pipeline:composite_sli:ratio_rate1h
        expr: |
          pipeline:freshness_sli:ratio_rate1h *
          pipeline:completeness_sli:ratio_rate1h *
          pipeline:accuracy_sli:ratio_rate1h
"""
```

---

## Streaming Service SLIs

### Lag, Processing Time, and Ordering SLIs

```python
class StreamingSLI:
    """
    SLIs for streaming/event processing services
    """

    def __init__(self, prometheus_client, service_name: str):
        self.prom = prometheus_client
        self.service = service_name

    def calculate_lag_sli(
        self,
        max_lag_seconds: int = 60,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate consumer lag SLI

        Measures whether the consumer is keeping up with the producer

        Args:
            max_lag_seconds: Maximum acceptable lag
            time_range: Prometheus time range

        Returns:
            Dictionary with lag metrics
        """
        # Proportion of time lag is under threshold
        sli_query = f"""
        sum(rate(stream_consumer_lag_seconds_bucket{{
            service="{self.service}",
            le="{max_lag_seconds}"
        }}[{time_range}])) /
        sum(rate(stream_consumer_lag_seconds_count{{
            service="{self.service}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # Current lag by partition
        current_lag_query = f"""
        max(stream_consumer_lag_seconds{{
            service="{self.service}"
        }}) by (partition)
        """

        lag_results = self.prom.query(current_lag_query)

        lag_by_partition = {}
        max_lag = 0.0
        for result in lag_results:
            partition = result['metric'].get('partition', 'unknown')
            lag = float(result['value'][1])
            lag_by_partition[partition] = lag
            max_lag = max(max_lag, lag)

        return {
            'lag_sli_percentage': sli_value,
            'max_lag_seconds': max_lag,
            'threshold_seconds': max_lag_seconds,
            'lag_by_partition': lag_by_partition,
            'met': sli_value >= 99.0 and max_lag < max_lag_seconds
        }

    def calculate_processing_time_sli(
        self,
        max_processing_ms: int = 1000,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate event processing time SLI

        Measures how quickly events are processed

        Args:
            max_processing_ms: Maximum acceptable processing time
            time_range: Prometheus time range

        Returns:
            Dictionary with processing time metrics
        """
        # Proportion of events processed within threshold
        sli_query = f"""
        sum(rate(stream_event_processing_duration_bucket{{
            service="{self.service}",
            le="{max_processing_ms / 1000}"
        }}[{time_range}])) /
        sum(rate(stream_event_processing_duration_count{{
            service="{self.service}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # p95 processing time
        p95_query = f"""
        histogram_quantile(0.95,
            sum(rate(stream_event_processing_duration_bucket{{
                service="{self.service}"
            }}[{time_range}])) by (le)
        ) * 1000
        """

        p95_result = self.prom.query(p95_query)
        p95_ms = float(p95_result[0]['value'][1]) if p95_result else 0.0

        # Throughput (events/sec)
        throughput_query = f"""
        sum(rate(stream_events_processed_total{{
            service="{self.service}"
        }}[{time_range}]))
        """

        throughput_result = self.prom.query(throughput_query)
        throughput = float(throughput_result[0]['value'][1]) if throughput_result else 0.0

        return {
            'processing_time_sli_percentage': sli_value,
            'p95_processing_time_ms': p95_ms,
            'threshold_ms': max_processing_ms,
            'throughput_events_per_sec': throughput,
            'met': sli_value >= 95.0
        }

    def calculate_ordering_sli(
        self,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate event ordering SLI

        Measures whether events are processed in the correct order

        Args:
            time_range: Prometheus time range

        Returns:
            Dictionary with ordering metrics
        """
        # Proportion of events processed in order
        sli_query = f"""
        sum(rate(stream_events_processed_total{{
            service="{self.service}",
            ordering="correct"
        }}[{time_range}])) /
        sum(rate(stream_events_processed_total{{
            service="{self.service}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # Out-of-order event rate
        ooo_query = f"""
        sum(rate(stream_events_out_of_order_total{{
            service="{self.service}"
        }}[{time_range}]))
        """

        ooo_result = self.prom.query(ooo_query)
        ooo_rate = float(ooo_result[0]['value'][1]) if ooo_result else 0.0

        return {
            'ordering_sli_percentage': sli_value,
            'out_of_order_events_per_sec': ooo_rate,
            'met': sli_value >= 99.9
        }

    def calculate_delivery_sli(
        self,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate event delivery SLI

        Measures successful event delivery (no lost events)

        Args:
            time_range: Prometheus time range

        Returns:
            Dictionary with delivery metrics
        """
        # Proportion of events successfully delivered
        sli_query = f"""
        sum(rate(stream_events_delivered_total{{
            service="{self.service}",
            status="success"
        }}[{time_range}])) /
        sum(rate(stream_events_received_total{{
            service="{self.service}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # Failed delivery rate
        failed_query = f"""
        sum(rate(stream_events_delivered_total{{
            service="{self.service}",
            status="failed"
        }}[{time_range}]))
        """

        failed_result = self.prom.query(failed_query)
        failed_rate = float(failed_result[0]['value'][1]) if failed_result else 0.0

        return {
            'delivery_sli_percentage': sli_value,
            'failed_deliveries_per_sec': failed_rate,
            'met': sli_value >= 99.99
        }


# Prometheus Recording Rules for Streaming SLIs
STREAMING_RECORDING_RULES = """
groups:
  - name: streaming_sli
    interval: 30s
    rules:
      # Lag SLI (< 60 seconds)
      - record: stream:lag_sli:ratio_rate5m
        expr: |
          sum(rate(stream_consumer_lag_seconds_bucket{le="60"}[5m])) by (service) /
          sum(rate(stream_consumer_lag_seconds_count[5m])) by (service)

      # Processing time SLI (< 1 second)
      - record: stream:processing_time_sli:ratio_rate5m
        expr: |
          sum(rate(stream_event_processing_duration_bucket{le="1"}[5m])) by (service) /
          sum(rate(stream_event_processing_duration_count[5m])) by (service)

      # Ordering SLI
      - record: stream:ordering_sli:ratio_rate5m
        expr: |
          sum(rate(stream_events_processed_total{ordering="correct"}[5m])) by (service) /
          sum(rate(stream_events_processed_total[5m])) by (service)

      # Delivery SLI
      - record: stream:delivery_sli:ratio_rate5m
        expr: |
          sum(rate(stream_events_delivered_total{status="success"}[5m])) by (service) /
          sum(rate(stream_events_received_total[5m])) by (service)

      # Current max lag by service
      - record: stream:max_lag:seconds
        expr: |
          max(stream_consumer_lag_seconds) by (service)

      # Throughput (events/sec)
      - record: stream:throughput:events_per_sec
        expr: |
          sum(rate(stream_events_processed_total[5m])) by (service)
"""
```

---

## Client-Side SLI Measurement

### Real User Monitoring (RUM) SLIs

```python
class RealUserMonitoringSLI:
    """
    Client-side SLIs based on Real User Monitoring data
    """

    def __init__(self, prometheus_client, app_name: str):
        self.prom = prometheus_client
        self.app = app_name

    def calculate_navigation_timing_sli(
        self,
        threshold_ms: int = 3000,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate SLI based on Navigation Timing API

        Measures time from navigation start to page load complete

        Args:
            threshold_ms: Maximum acceptable page load time
            time_range: Prometheus time range

        Returns:
            Dictionary with navigation timing metrics
        """
        # Proportion of page loads faster than threshold
        sli_query = f"""
        sum(rate(rum_navigation_timing_bucket{{
            app="{self.app}",
            le="{threshold_ms}"
        }}[{time_range}])) /
        sum(rate(rum_navigation_timing_count{{
            app="{self.app}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # Breaking down timing phases
        phases = {
            'dns': 'domainLookupEnd - domainLookupStart',
            'tcp': 'connectEnd - connectStart',
            'request': 'responseStart - requestStart',
            'response': 'responseEnd - responseStart',
            'dom_processing': 'domContentLoadedEventEnd - domLoading',
            'onload': 'loadEventEnd - loadEventStart'
        }

        phase_metrics = {}
        for phase_name, calculation in phases.items():
            phase_query = f"""
            avg(rum_{phase_name}_duration_ms{{
                app="{self.app}"
            }}[{time_range}])
            """

            phase_result = self.prom.query(phase_query)
            phase_metrics[phase_name] = (
                float(phase_result[0]['value'][1]) if phase_result else 0.0
            )

        return {
            'navigation_timing_sli_percentage': sli_value,
            'threshold_ms': threshold_ms,
            'phase_breakdown': phase_metrics,
            'met': sli_value >= 75.0
        }

    def calculate_ajax_performance_sli(
        self,
        threshold_ms: int = 500,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate SLI for AJAX/API call performance from client side

        Args:
            threshold_ms: Maximum acceptable AJAX response time
            time_range: Prometheus time range

        Returns:
            Dictionary with AJAX performance metrics
        """
        # Proportion of AJAX calls faster than threshold
        sli_query = f"""
        sum(rate(rum_ajax_duration_bucket{{
            app="{self.app}",
            le="{threshold_ms}"
        }}[{time_range}])) /
        sum(rate(rum_ajax_duration_count{{
            app="{self.app}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # AJAX error rate
        error_query = f"""
        sum(rate(rum_ajax_requests_total{{
            app="{self.app}",
            status=~"[45].."
        }}[{time_range}])) /
        sum(rate(rum_ajax_requests_total{{
            app="{self.app}"
        }}[{time_range}])) * 100
        """

        error_result = self.prom.query(error_query)
        error_rate = float(error_result[0]['value'][1]) if error_result else 0.0

        return {
            'ajax_performance_sli_percentage': sli_value,
            'ajax_error_rate_percentage': error_rate,
            'threshold_ms': threshold_ms,
            'met': sli_value >= 95.0 and error_rate < 1.0
        }

    def calculate_javascript_error_sli(
        self,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate SLI for JavaScript errors

        Measures proportion of user sessions without JavaScript errors

        Args:
            time_range: Prometheus time range

        Returns:
            Dictionary with JavaScript error metrics
        """
        # Proportion of sessions without errors
        sli_query = f"""
        sum(rate(rum_sessions_total{{
            app="{self.app}",
            has_js_errors="false"
        }}[{time_range}])) /
        sum(rate(rum_sessions_total{{
            app="{self.app}"
        }}[{time_range}])) * 100
        """

        sli_result = self.prom.query(sli_query)
        sli_value = float(sli_result[0]['value'][1]) if sli_result else 0.0

        # Errors per session
        errors_per_session_query = f"""
        sum(rate(rum_javascript_errors_total{{
            app="{self.app}"
        }}[{time_range}])) /
        sum(rate(rum_sessions_total{{
            app="{self.app}"
        }}[{time_range}]))
        """

        eps_result = self.prom.query(errors_per_session_query)
        errors_per_session = float(eps_result[0]['value'][1]) if eps_result else 0.0

        # Error breakdown by type
        error_types_query = f"""
        topk(5,
            sum(rate(rum_javascript_errors_total{{
                app="{self.app}"
            }}[{time_range}])) by (error_type)
        )
        """

        error_types_results = self.prom.query(error_types_query)

        top_errors = {}
        for result in error_types_results:
            error_type = result['metric'].get('error_type', 'unknown')
            rate = float(result['value'][1])
            top_errors[error_type] = rate

        return {
            'error_free_sessions_percentage': sli_value,
            'errors_per_session': errors_per_session,
            'top_error_types': top_errors,
            'met': sli_value >= 99.0
        }

    def calculate_user_engagement_sli(
        self,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate SLI based on user engagement metrics

        Measures quality of user experience through engagement

        Args:
            time_range: Prometheus time range

        Returns:
            Dictionary with engagement metrics
        """
        # Bounce rate (sessions with only one page view)
        bounce_query = f"""
        sum(rate(rum_sessions_total{{
            app="{self.app}",
            page_views="1"
        }}[{time_range}])) /
        sum(rate(rum_sessions_total{{
            app="{self.app}"
        }}[{time_range}])) * 100
        """

        bounce_result = self.prom.query(bounce_query)
        bounce_rate = float(bounce_result[0]['value'][1]) if bounce_result else 0.0

        # Average session duration
        session_duration_query = f"""
        avg(rum_session_duration_seconds{{
            app="{self.app}"
        }}[{time_range}])
        """

        duration_result = self.prom.query(session_duration_query)
        avg_duration = float(duration_result[0]['value'][1]) if duration_result else 0.0

        # Engaged sessions (> 30 seconds, > 1 page view)
        engaged_query = f"""
        sum(rate(rum_sessions_total{{
            app="{self.app}",
            engaged="true"
        }}[{time_range}])) /
        sum(rate(rum_sessions_total{{
            app="{self.app}"
        }}[{time_range}])) * 100
        """

        engaged_result = self.prom.query(engaged_query)
        engagement_rate = float(engaged_result[0]['value'][1]) if engaged_result else 0.0

        # Engagement SLI: high engagement rate and low bounce rate
        engagement_sli = (engagement_rate + (100 - bounce_rate)) / 2

        return {
            'engagement_sli_percentage': engagement_sli,
            'bounce_rate_percentage': bounce_rate,
            'engagement_rate_percentage': engagement_rate,
            'avg_session_duration_seconds': avg_duration,
            'met': engagement_sli >= 70.0
        }


# Prometheus Recording Rules for RUM SLIs
RUM_RECORDING_RULES = """
groups:
  - name: rum_sli
    interval: 30s
    rules:
      # Navigation timing SLI (< 3 seconds)
      - record: app:navigation_timing_sli:ratio_rate5m
        expr: |
          sum(rate(rum_navigation_timing_bucket{le="3000"}[5m])) by (app) /
          sum(rate(rum_navigation_timing_count[5m])) by (app)

      # AJAX performance SLI (< 500ms)
      - record: app:ajax_performance_sli:ratio_rate5m
        expr: |
          sum(rate(rum_ajax_duration_bucket{le="500"}[5m])) by (app) /
          sum(rate(rum_ajax_duration_count[5m])) by (app)

      # JavaScript error-free sessions
      - record: app:js_error_free_sessions:ratio_rate5m
        expr: |
          sum(rate(rum_sessions_total{has_js_errors="false"}[5m])) by (app) /
          sum(rate(rum_sessions_total[5m])) by (app)

      # Engagement rate
      - record: app:engagement_rate:ratio_rate5m
        expr: |
          sum(rate(rum_sessions_total{engaged="true"}[5m])) by (app) /
          sum(rate(rum_sessions_total[5m])) by (app)

      # Bounce rate
      - record: app:bounce_rate:ratio_rate5m
        expr: |
          sum(rate(rum_sessions_total{page_views="1"}[5m])) by (app) /
          sum(rate(rum_sessions_total[5m])) by (app)
"""
```

---

## SLI Implementation Patterns

### Pattern 1: Request-Based SLI

For synchronous request/response services (APIs, web applications):

```python
class RequestBasedSLI:
    """
    Generic request-based SLI implementation
    """

    def __init__(self, prometheus_client, service_name: str):
        self.prom = prometheus_client
        self.service = service_name

    def calculate_ratio_sli(
        self,
        good_events_query: str,
        total_events_query: str,
        time_range: str = '5m'
    ) -> float:
        """
        Calculate ratio-based SLI

        Args:
            good_events_query: PromQL query for good events
            total_events_query: PromQL query for total events
            time_range: Time range for the query

        Returns:
            SLI value as percentage (0-100)
        """
        good_result = self.prom.query(good_events_query.format(
            service=self.service,
            time_range=time_range
        ))

        total_result = self.prom.query(total_events_query.format(
            service=self.service,
            time_range=time_range
        ))

        good_events = float(good_result[0]['value'][1]) if good_result else 0.0
        total_events = float(total_result[0]['value'][1]) if total_result else 1.0

        return (good_events / total_events) * 100 if total_events > 0 else 0.0


# Usage example
request_sli = RequestBasedSLI(prom_client, "api-service")

# Availability SLI
availability = request_sli.calculate_ratio_sli(
    good_events_query="""
        sum(rate(http_requests_total{{
            service="{service}",
            status!~"5.."
        }}[{time_range}]))
    """,
    total_events_query="""
        sum(rate(http_requests_total{{
            service="{service}"
        }}[{time_range}]))
    """
)
```

### Pattern 2: Window-Based SLI

For batch and data processing systems:

```python
class WindowBasedSLI:
    """
    Window-based SLI for batch processing
    """

    def __init__(self, prometheus_client, service_name: str):
        self.prom = prometheus_client
        self.service = service_name

    def calculate_compliance_sli(
        self,
        compliance_query: str,
        window: str = '1h'
    ) -> Dict:
        """
        Calculate compliance-based SLI over a time window

        Args:
            compliance_query: PromQL query that returns 1 for compliant, 0 for not
            window: Time window for compliance measurement

        Returns:
            Dictionary with compliance percentage and details
        """
        query = compliance_query.format(
            service=self.service,
            window=window
        )

        result = self.prom.query(query)

        # Average compliance over the window
        compliance = float(result[0]['value'][1]) if result else 0.0

        return {
            'compliance_percentage': compliance * 100,
            'window': window,
            'met': compliance >= 0.99
        }


# Usage example
window_sli = WindowBasedSLI(prom_client, "batch-pipeline")

# Freshness compliance
freshness = window_sli.calculate_compliance_sli(
    compliance_query="""
        avg_over_time(
            (pipeline_data_age_seconds{{service="{service}"}} < 1800)[{window}:]
        )
    """
)
```

### Pattern 3: Threshold-Based SLI

For latency and performance metrics:

```python
class ThresholdBasedSLI:
    """
    Threshold-based SLI for latency metrics
    """

    def __init__(self, prometheus_client, service_name: str):
        self.prom = prometheus_client
        self.service = service_name

    def calculate_threshold_sli(
        self,
        metric_query: str,
        threshold: float,
        above_threshold_is_good: bool = False,
        time_range: str = '5m'
    ) -> Dict:
        """
        Calculate SLI based on threshold compliance

        Args:
            metric_query: PromQL query for the metric
            threshold: Threshold value
            above_threshold_is_good: If True, values above threshold are good
            time_range: Time range

        Returns:
            Dictionary with SLI and threshold compliance
        """
        query = metric_query.format(
            service=self.service,
            threshold=threshold,
            time_range=time_range
        )

        result = self.prom.query(query)
        metric_value = float(result[0]['value'][1]) if result else 0.0

        if above_threshold_is_good:
            met = metric_value >= threshold
        else:
            met = metric_value <= threshold

        return {
            'metric_value': metric_value,
            'threshold': threshold,
            'met': met,
            'sli_percentage': 100.0 if met else 0.0
        }


# Usage example
threshold_sli = ThresholdBasedSLI(prom_client, "api-service")

# Latency threshold
latency = threshold_sli.calculate_threshold_sli(
    metric_query="""
        histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket{{
                service="{service}"
            }}[{time_range}])) by (le)
        )
    """,
    threshold=0.5,  # 500ms
    above_threshold_is_good=False
)
```

---

## Python SLI Classes

### Base SLI Class

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

class BaseSLI(ABC):
    """
    Abstract base class for all SLI implementations
    """

    def __init__(
        self,
        prometheus_client,
        service_name: str,
        sli_name: str,
        target_percentage: float = 99.0
    ):
        self.prom = prometheus_client
        self.service = service_name
        self.sli_name = sli_name
        self.target = target_percentage
        self.logger = logging.getLogger(f"SLI.{sli_name}")

    @abstractmethod
    def calculate(self, time_range: str = '5m') -> float:
        """
        Calculate the SLI value

        Args:
            time_range: Prometheus time range

        Returns:
            SLI value as percentage (0-100)
        """
        pass

    def is_met(self, time_range: str = '5m') -> bool:
        """
        Check if SLI target is met

        Returns:
            True if SLI meets target, False otherwise
        """
        value = self.calculate(time_range)
        return value >= self.target

    def get_status(self, time_range: str = '5m') -> Dict:
        """
        Get complete SLI status

        Returns:
            Dictionary with SLI details
        """
        value = self.calculate(time_range)
        met = value >= self.target

        status = {
            'sli_name': self.sli_name,
            'service': self.service,
            'value': value,
            'target': self.target,
            'met': met,
            'gap': self.target - value if not met else 0.0,
            'timestamp': datetime.utcnow().isoformat(),
            'time_range': time_range
        }

        return status

    def calculate_over_windows(
        self,
        windows: List[str]
    ) -> Dict[str, float]:
        """
        Calculate SLI over multiple time windows

        Args:
            windows: List of time windows (e.g., ['5m', '1h', '1d'])

        Returns:
            Dictionary mapping window to SLI value
        """
        results = {}

        for window in windows:
            try:
                value = self.calculate(time_range=window)
                results[window] = value
                self.logger.info(
                    f"{self.sli_name} over {window}: {value:.2f}%"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to calculate {self.sli_name} for window {window}: {e}"
                )
                results[window] = None

        return results

    def validate_query_result(self, result: Any) -> float:
        """
        Validate and extract value from Prometheus query result

        Args:
            result: Prometheus query result

        Returns:
            Extracted float value or 0.0 if invalid
        """
        if not result or len(result) == 0:
            self.logger.warning(f"Empty result for {self.sli_name}")
            return 0.0

        try:
            value = float(result[0]['value'][1])
            if value < 0 or value > 100:
                self.logger.warning(
                    f"SLI value out of range [0, 100]: {value}"
                )
            return value
        except (KeyError, IndexError, ValueError) as e:
            self.logger.error(f"Failed to parse query result: {e}")
            return 0.0


class CompositeSLI(BaseSLI):
    """
    Composite SLI combining multiple SLIs
    """

    def __init__(
        self,
        component_slis: List[BaseSLI],
        weights: Optional[Dict[str, float]] = None,
        aggregation_method: str = 'weighted_average'
    ):
        self.component_slis = component_slis
        self.weights = weights or {
            sli.sli_name: 1.0 for sli in component_slis
        }
        self.aggregation_method = aggregation_method

        # Use first component's service name
        service_name = component_slis[0].service if component_slis else "unknown"

        super().__init__(
            prometheus_client=None,
            service_name=service_name,
            sli_name="composite",
            target_percentage=99.0
        )

    def calculate(self, time_range: str = '5m') -> float:
        """
        Calculate composite SLI from components
        """
        if self.aggregation_method == 'weighted_average':
            return self._weighted_average(time_range)
        elif self.aggregation_method == 'minimum':
            return self._minimum(time_range)
        elif self.aggregation_method == 'product':
            return self._product(time_range)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def _weighted_average(self, time_range: str) -> float:
        """Calculate weighted average of component SLIs"""
        total_weight = sum(self.weights.values())
        weighted_sum = 0.0

        for sli in self.component_slis:
            value = sli.calculate(time_range)
            weight = self.weights.get(sli.sli_name, 1.0)
            weighted_sum += value * weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _minimum(self, time_range: str) -> float:
        """Return minimum of component SLIs"""
        values = [sli.calculate(time_range) for sli in self.component_slis]
        return min(values) if values else 0.0

    def _product(self, time_range: str) -> float:
        """Return product of component SLIs (all must be met)"""
        product = 1.0
        for sli in self.component_slis:
            value = sli.calculate(time_range) / 100.0  # Convert to ratio
            product *= value
        return product * 100.0  # Convert back to percentage
```

### Concrete SLI Implementations

```python
class AvailabilitySLI(BaseSLI):
    """Availability SLI implementation"""

    def calculate(self, time_range: str = '5m') -> float:
        query = f"""
        sum(rate(http_requests_total{{
            service="{self.service}",
            status!~"5.."
        }}[{time_range}])) /
        sum(rate(http_requests_total{{
            service="{self.service}"
        }}[{time_range}])) * 100
        """

        result = self.prom.query(query)
        return self.validate_query_result(result)


class LatencySLI(BaseSLI):
    """Latency SLI implementation"""

    def __init__(
        self,
        prometheus_client,
        service_name: str,
        threshold_ms: float = 500,
        **kwargs
    ):
        super().__init__(prometheus_client, service_name, "latency", **kwargs)
        self.threshold_ms = threshold_ms
        self.threshold_sec = threshold_ms / 1000.0

    def calculate(self, time_range: str = '5m') -> float:
        query = f"""
        sum(rate(http_request_duration_seconds_bucket{{
            service="{self.service}",
            le="{self.threshold_sec}"
        }}[{time_range}])) /
        sum(rate(http_request_duration_seconds_count{{
            service="{self.service}"
        }}[{time_range}])) * 100
        """

        result = self.prom.query(query)
        return self.validate_query_result(result)


class ErrorRateSLI(BaseSLI):
    """Error rate SLI implementation"""

    def calculate(self, time_range: str = '5m') -> float:
        # Calculate success rate (100 - error rate)
        query = f"""
        (1 - sum(rate(http_requests_total{{
            service="{self.service}",
            status=~"5.."
        }}[{time_range}])) /
        sum(rate(http_requests_total{{
            service="{self.service}"
        }}[{time_range}]))) * 100
        """

        result = self.prom.query(query)
        return self.validate_query_result(result)


class ThroughputSLI(BaseSLI):
    """Throughput SLI implementation"""

    def __init__(
        self,
        prometheus_client,
        service_name: str,
        expected_throughput: float,
        **kwargs
    ):
        super().__init__(prometheus_client, service_name, "throughput", **kwargs)
        self.expected_throughput = expected_throughput

    def calculate(self, time_range: str = '5m') -> float:
        query = f"""
        sum(rate(http_requests_total{{
            service="{self.service}"
        }}[{time_range}]))
        """

        result = self.prom.query(query)
        actual_throughput = self.validate_query_result(result)

        # Return percentage of expected throughput achieved
        return min(
            (actual_throughput / self.expected_throughput) * 100,
            100.0
        ) if self.expected_throughput > 0 else 0.0
```

---

## Prometheus Query Examples

### Complete PromQL Query Library

```yaml
# API Service Queries

# 1. Availability
api_availability_5m: |
  sum(rate(http_requests_total{service="api",status!~"5.."}[5m])) /
  sum(rate(http_requests_total{service="api"}[5m])) * 100

# 2. Latency Percentiles
api_latency_p50: |
  histogram_quantile(0.50,
    sum(rate(http_request_duration_seconds_bucket{service="api"}[5m])) by (le)
  ) * 1000

api_latency_p95: |
  histogram_quantile(0.95,
    sum(rate(http_request_duration_seconds_bucket{service="api"}[5m])) by (le)
  ) * 1000

api_latency_p99: |
  histogram_quantile(0.99,
    sum(rate(http_request_duration_seconds_bucket{service="api"}[5m])) by (le)
  ) * 1000

# 3. Error Rate by Category
api_error_rate_5xx: |
  sum(rate(http_requests_total{service="api",status=~"5.."}[5m])) /
  sum(rate(http_requests_total{service="api"}[5m])) * 100

api_error_rate_4xx: |
  sum(rate(http_requests_total{service="api",status=~"4.."}[5m])) /
  sum(rate(http_requests_total{service="api"}[5m])) * 100

# 4. Throughput
api_throughput_requests_per_sec: |
  sum(rate(http_requests_total{service="api"}[5m]))

# 5. Multi-Window Availability
api_availability_1h: |
  sum(rate(http_requests_total{service="api",status!~"5.."}[1h])) /
  sum(rate(http_requests_total{service="api"}[1h])) * 100

api_availability_1d: |
  sum(rate(http_requests_total{service="api",status!~"5.."}[1d])) /
  sum(rate(http_requests_total{service="api"}[1d])) * 100

# Web Application Queries

# 6. Core Web Vitals
web_lcp_p75: |
  histogram_quantile(0.75,
    sum(rate(web_vitals_lcp_bucket{app="web"}[5m])) by (le)
  )

web_fid_p75: |
  histogram_quantile(0.75,
    sum(rate(web_vitals_fid_bucket{app="web"}[5m])) by (le)
  )

web_cls_p75: |
  histogram_quantile(0.75,
    sum(rate(web_vitals_cls_bucket{app="web"}[5m])) by (le)
  )

# 7. Page Load Performance
web_page_load_success_rate: |
  sum(rate(page_loads_total{app="web",status="success"}[5m])) /
  sum(rate(page_loads_total{app="web"}[5m])) * 100

# Batch Pipeline Queries

# 8. Freshness
pipeline_freshness_sli: |
  sum(rate(pipeline_batch_processing_duration_bucket{
    pipeline="data",
    le="1800"
  }[1h])) /
  sum(rate(pipeline_batch_processing_duration_count{pipeline="data"}[1h])) * 100

pipeline_data_age_minutes: |
  (time() - max(pipeline_batch_completion_timestamp{
    pipeline="data",
    status="success"
  })) / 60

# 9. Completeness
pipeline_completeness_sli: |
  sum(increase(pipeline_records_processed_total{
    pipeline="data",
    status="success"
  }[1h])) /
  sum(increase(pipeline_records_received_total{pipeline="data"}[1h])) * 100

# 10. Accuracy
pipeline_accuracy_sli: |
  sum(increase(pipeline_records_validated_total{
    pipeline="data",
    result="passed"
  }[1h])) /
  sum(increase(pipeline_records_validated_total{pipeline="data"}[1h])) * 100

# Streaming Service Queries

# 11. Consumer Lag
stream_lag_seconds: |
  max(stream_consumer_lag_seconds{service="streaming"}) by (partition)

stream_lag_sli: |
  sum(rate(stream_consumer_lag_seconds_bucket{
    service="streaming",
    le="60"
  }[5m])) /
  sum(rate(stream_consumer_lag_seconds_count{service="streaming"}[5m])) * 100

# 12. Processing Time
stream_processing_time_p95: |
  histogram_quantile(0.95,
    sum(rate(stream_event_processing_duration_bucket{
      service="streaming"
    }[5m])) by (le)
  ) * 1000

# 13. Throughput
stream_throughput_events_per_sec: |
  sum(rate(stream_events_processed_total{service="streaming"}[5m]))

# 14. Delivery Success
stream_delivery_sli: |
  sum(rate(stream_events_delivered_total{
    service="streaming",
    status="success"
  }[5m])) /
  sum(rate(stream_events_received_total{service="streaming"}[5m])) * 100

# Error Budget Queries

# 15. Error Budget Burn Rate
error_budget_burn_rate_1h: |
  (1 - (
    sum(increase(http_requests_total{service="api",status!~"5.."}[1h])) /
    sum(increase(http_requests_total{service="api"}[1h]))
  )) / (1 - 0.999)  # For 99.9% SLO

error_budget_burn_rate_6h: |
  (1 - (
    sum(increase(http_requests_total{service="api",status!~"5.."}[6h])) /
    sum(increase(http_requests_total{service="api"}[6h]))
  )) / (1 - 0.999)

# 16. Error Budget Remaining
error_budget_remaining_30d: |
  100 * (1 - (
    (1 - sum(increase(http_requests_total{service="api",status!~"5.."}[30d])) /
         sum(increase(http_requests_total{service="api"}[30d]))) /
    (1 - 0.999)
  ))

# Advanced Queries

# 17. Apdex Score
apdex_score: |
  (
    sum(rate(http_request_duration_seconds_bucket{service="api",le="0.5"}[5m])) +
    0.5 * (
      sum(rate(http_request_duration_seconds_bucket{service="api",le="2"}[5m])) -
      sum(rate(http_request_duration_seconds_bucket{service="api",le="0.5"}[5m]))
    )
  ) / sum(rate(http_request_duration_seconds_count{service="api"}[5m]))

# 18. User-Journey SLI (multi-step)
user_journey_checkout_sli: |
  # All steps must succeed
  min(
    sum(rate(http_requests_total{endpoint="/cart",status!~"5.."}[5m])) /
    sum(rate(http_requests_total{endpoint="/cart"}[5m])),

    sum(rate(http_requests_total{endpoint="/checkout",status!~"5.."}[5m])) /
    sum(rate(http_requests_total{endpoint="/checkout"}[5m])),

    sum(rate(http_requests_total{endpoint="/payment",status!~"5.."}[5m])) /
    sum(rate(http_requests_total{endpoint="/payment"}[5m]))
  ) * 100

# 19. Weighted SLI by Traffic
weighted_availability_by_endpoint: |
  sum(
    (
      rate(http_requests_total{status!~"5.."}[5m]) /
      rate(http_requests_total[5m])
    ) * rate(http_requests_total[5m])
  ) by (endpoint) /
  sum(rate(http_requests_total[5m])) by (endpoint) * 100

# 20. SLI Compliance (binary: 1 if met, 0 if not)
sli_compliance: |
  (
    sum(rate(http_requests_total{service="api",status!~"5.."}[5m])) /
    sum(rate(http_requests_total{service="api"}[5m])) * 100 >= 99.9
  ) * 1
```

---

## SLI Validation and Testing

### SLI Validator

```python
class SLIValidator:
    """
    Validate SLI implementations and data quality
    """

    def __init__(self, prometheus_client):
        self.prom = prometheus_client
        self.logger = logging.getLogger("SLIValidator")

    def validate_sli_implementation(
        self,
        sli: BaseSLI
    ) -> Dict:
        """
        Validate an SLI implementation

        Returns:
            Dictionary with validation results
        """
        results = {
            'sli_name': sli.sli_name,
            'service': sli.service,
            'checks': {},
            'valid': True
        }

        # Check 1: Can calculate SLI
        try:
            value = sli.calculate('5m')
            results['checks']['can_calculate'] = True
            results['sample_value'] = value
        except Exception as e:
            results['checks']['can_calculate'] = False
            results['error'] = str(e)
            results['valid'] = False
            return results

        # Check 2: Value in valid range
        if 0 <= value <= 100:
            results['checks']['value_in_range'] = True
        else:
            results['checks']['value_in_range'] = False
            results['valid'] = False
            results['checks']['value_range_error'] = (
                f"Value {value} outside [0, 100]"
            )

        # Check 3: Has data (not zero)
        if value > 0:
            results['checks']['has_data'] = True
        else:
            results['checks']['has_data'] = False
            self.logger.warning(
                f"SLI {sli.sli_name} returned zero - may indicate no data"
            )

        # Check 4: Consistent across windows
        windows = ['5m', '15m', '1h']
        window_values = sli.calculate_over_windows(windows)

        if all(v is not None for v in window_values.values()):
            results['checks']['consistent_across_windows'] = True
            results['window_values'] = window_values

            # Check for anomalies (values shouldn't differ drastically)
            values = list(window_values.values())
            max_diff = max(values) - min(values)

            if max_diff < 10.0:  # Less than 10% difference
                results['checks']['stable_values'] = True
            else:
                results['checks']['stable_values'] = False
                results['checks']['stability_warning'] = (
                    f"Values vary by {max_diff:.2f}% across windows"
                )
        else:
            results['checks']['consistent_across_windows'] = False
            results['valid'] = False

        return results

    def validate_metric_availability(
        self,
        metric_name: str,
        labels: Dict[str, str],
        time_range: str = '1h'
    ) -> Dict:
        """
        Validate that required metrics are available

        Returns:
            Dictionary with metric availability details
        """
        label_str = ','.join([f'{k}="{v}"' for k, v in labels.items()])

        # Check if metric exists
        query = f'{metric_name}{{{label_str}}}'
        result = self.prom.query(query)

        exists = result is not None and len(result) > 0

        if not exists:
            return {
                'metric': metric_name,
                'exists': False,
                'error': 'Metric not found'
            }

        # Check data freshness
        freshness_query = f'time() - timestamp({metric_name}{{{label_str}}})'
        freshness_result = self.prom.query(freshness_query)

        age_seconds = (
            float(freshness_result[0]['value'][1])
            if freshness_result else float('inf')
        )

        # Check data rate
        rate_query = f'rate({metric_name}{{{label_str}}}[{time_range}])'
        rate_result = self.prom.query(rate_query)

        rate = float(rate_result[0]['value'][1]) if rate_result else 0.0

        return {
            'metric': metric_name,
            'exists': True,
            'age_seconds': age_seconds,
            'is_fresh': age_seconds < 300,  # Less than 5 minutes old
            'rate': rate,
            'has_data': rate > 0
        }

    def test_sli_alerting_threshold(
        self,
        sli: BaseSLI,
        simulated_error_rate: float,
        time_range: str = '5m'
    ) -> Dict:
        """
        Test if SLI properly detects degradation

        This is a simulation to verify alert sensitivity

        Returns:
            Dictionary with test results
        """
        # Calculate baseline SLI
        baseline = sli.calculate(time_range)

        # Simulate degradation
        expected_sli_after_errors = baseline * (1 - simulated_error_rate)

        # Check if this would trigger alert
        would_alert = expected_sli_after_errors < sli.target

        # Calculate how much error rate needed to breach SLO
        current_good_rate = baseline / 100.0
        target_good_rate = sli.target / 100.0

        max_tolerable_error_rate = current_good_rate - target_good_rate

        return {
            'baseline_sli': baseline,
            'simulated_error_rate': simulated_error_rate,
            'expected_sli_after_errors': expected_sli_after_errors,
            'would_trigger_alert': would_alert,
            'max_tolerable_error_rate': max_tolerable_error_rate,
            'alert_sensitivity': 'appropriate' if would_alert else 'may_need_adjustment'
        }


# SLI Testing Framework
class SLITestSuite:
    """
    Comprehensive testing for SLI implementations
    """

    def __init__(self, prometheus_client):
        self.validator = SLIValidator(prometheus_client)
        self.logger = logging.getLogger("SLITestSuite")

    def run_full_test_suite(
        self,
        slis: List[BaseSLI]
    ) -> Dict:
        """
        Run complete test suite for all SLIs

        Returns:
            Dictionary with comprehensive test results
        """
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_slis_tested': len(slis),
            'sli_results': [],
            'overall_status': 'passed'
        }

        for sli in slis:
            self.logger.info(f"Testing SLI: {sli.sli_name}")

            sli_result = {
                'sli_name': sli.sli_name,
                'tests': {}
            }

            # Test 1: Implementation validation
            impl_validation = self.validator.validate_sli_implementation(sli)
            sli_result['tests']['implementation'] = impl_validation

            if not impl_validation['valid']:
                sli_result['status'] = 'failed'
                results['overall_status'] = 'failed'
                results['sli_results'].append(sli_result)
                continue

            # Test 2: Alert sensitivity
            alert_test = self.validator.test_sli_alerting_threshold(
                sli,
                simulated_error_rate=0.01  # 1% error rate
            )
            sli_result['tests']['alert_sensitivity'] = alert_test

            # Test 3: Performance (query execution time)
            import time
            start = time.time()
            sli.calculate('5m')
            execution_time = time.time() - start

            sli_result['tests']['performance'] = {
                'execution_time_seconds': execution_time,
                'acceptable': execution_time < 5.0  # Should complete in < 5s
            }

            # Overall SLI test status
            sli_result['status'] = 'passed'
            results['sli_results'].append(sli_result)

        return results

    def generate_test_report(
        self,
        test_results: Dict
    ) -> str:
        """
        Generate human-readable test report

        Returns:
            Formatted test report
        """
        report = []
        report.append("=" * 60)
        report.append("SLI VALIDATION TEST REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {test_results['timestamp']}")
        report.append(f"Total SLIs Tested: {test_results['total_slis_tested']}")
        report.append(f"Overall Status: {test_results['overall_status'].upper()}")
        report.append("")

        for sli_result in test_results['sli_results']:
            report.append("-" * 60)
            report.append(f"SLI: {sli_result['sli_name']}")
            report.append(f"Status: {sli_result['status'].upper()}")
            report.append("")

            for test_name, test_data in sli_result['tests'].items():
                report.append(f"  {test_name}:")

                if isinstance(test_data, dict):
                    for key, value in test_data.items():
                        report.append(f"    {key}: {value}")
                else:
                    report.append(f"    {test_data}")

                report.append("")

        report.append("=" * 60)

        return "\n".join(report)


# Unit Tests for SLI Classes
def test_availability_sli():
    """Unit test for AvailabilitySLI"""
    # Mock Prometheus client
    class MockPrometheusClient:
        def query(self, query_str):
            # Return 99.5% availability
            return [{'value': [0, '99.5']}]

    prom = MockPrometheusClient()
    sli = AvailabilitySLI(prom, "test-service", target_percentage=99.0)

    # Test calculation
    value = sli.calculate('5m')
    assert value == 99.5, f"Expected 99.5, got {value}"

    # Test target met
    assert sli.is_met('5m'), "Expected SLI to be met"

    # Test status
    status = sli.get_status('5m')
    assert status['met'] == True
    assert status['value'] == 99.5

    print("✓ AvailabilitySLI test passed")


def test_composite_sli():
    """Unit test for CompositeSLI"""
    # Mock Prometheus client
    class MockPrometheusClient:
        def query(self, query_str):
            return [{'value': [0, '99.0']}]

    prom = MockPrometheusClient()

    # Create component SLIs
    availability = AvailabilitySLI(prom, "test-service")
    latency = LatencySLI(prom, "test-service", threshold_ms=500)

    # Create composite
    composite = CompositeSLI(
        component_slis=[availability, latency],
        weights={'availability': 0.6, 'latency': 0.4},
        aggregation_method='weighted_average'
    )

    # Test calculation
    value = composite.calculate('5m')
    assert value == 99.0, f"Expected 99.0, got {value}"

    print("✓ CompositeSLI test passed")


if __name__ == "__main__":
    # Run unit tests
    test_availability_sli()
    test_composite_sli()
    print("\nAll SLI unit tests passed!")
```

---

## Summary

This comprehensive guide covers:

1. **SLI Fundamentals** - Core concepts, design principles, and specifications
2. **SLI Types** - Availability, latency, error rate, throughput, quality, freshness, completeness, correctness
3. **API Service SLIs** - Complete implementations with Prometheus queries for availability, latency, and error rate
4. **Web Application SLIs** - Core Web Vitals (LCP, FID, CLS) integration and page load metrics
5. **Batch Pipeline SLIs** - Freshness, completeness, and accuracy measurements for data pipelines
6. **Streaming Service SLIs** - Lag, processing time, ordering, and delivery metrics for event streams
7. **Client-Side SLI Measurement** - Real User Monitoring (RUM) with navigation timing, AJAX performance, and engagement
8. **SLI Implementation Patterns** - Request-based, window-based, and threshold-based patterns
9. **Python SLI Classes** - Complete class hierarchy with base classes and concrete implementations
10. **Prometheus Query Examples** - 20+ production-ready PromQL queries for all SLI types
11. **SLI Validation and Testing** - Comprehensive testing framework with validators and unit tests

All implementations include:
- Production-ready Python code
- Prometheus recording rules
- Query examples
- Validation strategies
- Testing frameworks

Use these patterns and implementations as a foundation for building robust SLI measurement systems tailored to your specific service types and business requirements.
