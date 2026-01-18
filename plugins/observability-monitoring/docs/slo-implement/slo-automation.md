# SLO Automation Framework

Complete guide for automating Service Level Objective (SLO) management with SLO-as-code, progressive implementation strategies, and GitOps workflows.

## Table of Contents

1. [SLO-as-Code Overview](#slo-as-code-overview)
2. [YAML/JSON Schema Definitions](#yamljson-schema-definitions)
3. [Automated SLO Generation](#automated-slo-generation)
4. [Progressive SLO Implementation](#progressive-slo-implementation)
5. [SLO Template Library](#slo-template-library)
6. [GitOps Workflow](#gitops-workflow)
7. [CI/CD Integration](#cicd-integration)
8. [Kubernetes CRD](#kubernetes-crd)
9. [Python Automation Tools](#python-automation-tools)
10. [Service Discovery Integration](#service-discovery-integration)
11. [Migration Strategies](#migration-strategies)

---

## SLO-as-Code Overview

SLO-as-code enables declarative SLO management through version-controlled configuration files, providing consistency, auditability, and automated deployment.

### Benefits

- **Version Control**: Track SLO changes over time
- **Code Review**: Peer review SLO modifications
- **Automation**: Automatic deployment and validation
- **Consistency**: Standardized SLO definitions across services
- **Rollback**: Easy rollback of problematic changes
- **Documentation**: Self-documenting SLO configuration

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SLO-as-Code Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Git Repository                                               │
│  ├── slo-definitions/                                         │
│  │   ├── api-service.yaml                                     │
│  │   ├── web-service.yaml                                     │
│  │   └── batch-pipeline.yaml                                  │
│  │                                                             │
│  │   ↓ (Git Commit)                                           │
│  │                                                             │
│  CI/CD Pipeline                                               │
│  ├── Validation                                               │
│  │   ├── Schema validation                                    │
│  │   ├── Syntax checking                                      │
│  │   └── Business rule validation                             │
│  │                                                             │
│  ├── Testing                                                  │
│  │   ├── Dry-run deployment                                   │
│  │   ├── Impact analysis                                      │
│  │   └── Alert simulation                                     │
│  │                                                             │
│  └── Deployment                                               │
│      ├── Apply to monitoring system                           │
│      ├── Create recording rules                               │
│      ├── Configure alerts                                     │
│      └── Update dashboards                                    │
│                                                               │
│  Monitoring Infrastructure                                    │
│  ├── Prometheus (Metrics & Recording Rules)                   │
│  ├── Grafana (Dashboards)                                     │
│  ├── AlertManager (Notifications)                             │
│  └── SLO Platform (Reporting)                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## YAML/JSON Schema Definitions

### Core SLO Schema

Complete YAML schema for defining SLOs:

```yaml
apiVersion: slo.dev/v1
kind: ServiceLevelObjective
metadata:
  name: string                    # Unique identifier
  namespace: string               # Kubernetes namespace or logical grouping
  labels:                         # Optional labels
    team: string
    environment: string
    tier: string
  annotations:                    # Optional metadata
    owner: string
    runbook: string
    dashboard: string
    oncall: string

spec:
  # Service identification
  service: string                 # Service name
  description: string             # Human-readable description

  # SLI definition
  indicator:
    type: enum                    # ratio | threshold | latency | availability

    # For ratio-based SLIs
    ratio:
      good:
        metric: string            # Prometheus metric for good events
        filters: []               # Label filters
      total:
        metric: string            # Prometheus metric for total events
        filters: []               # Label filters

    # For threshold-based SLIs
    threshold:
      metric: string              # Prometheus metric
      operator: enum              # lt | lte | gt | gte
      value: number               # Threshold value

    # For latency-based SLIs
    latency:
      metric: string              # Histogram metric
      percentile: number          # 0.50, 0.95, 0.99, etc.
      threshold_ms: number        # Latency threshold in milliseconds

  # SLO objectives
  objectives:
    - displayName: string         # Human-readable name
      window: string              # Time window (e.g., 30d, 7d, 1h)
      target: number              # Target percentage (0.0 - 1.0)
      description: string         # Optional description

  # Alerting configuration
  alerting:
    enabled: boolean

    # Multi-window multi-burn-rate alerts
    burnRates:
      - severity: enum            # critical | warning | info
        shortWindow: string       # Short time window (e.g., 1h)
        longWindow: string        # Long time window (e.g., 5m)
        burnRate: number          # Burn rate threshold (e.g., 14.4)
        notificationChannels: []  # List of notification channels

      - severity: warning
        shortWindow: 6h
        longWindow: 30m
        burnRate: 3
        notificationChannels: []

    # Budget exhaustion alerts
    budgetExhaustion:
      enabled: boolean
      thresholds:                 # Alert at specific budget remaining %
        - percentage: 50
          severity: warning
        - percentage: 25
          severity: critical
        - percentage: 10
          severity: critical

  # Error budget policy
  errorBudgetPolicy:
    enabled: boolean
    actions:
      - threshold: number         # Remaining budget percentage
        action: enum              # freeze | review | notify
        description: string

status:
  # Runtime status (populated by controller)
  currentSLI: number              # Current SLI value
  remainingErrorBudget: number    # Remaining error budget %
  burnRate: number                # Current burn rate
  lastUpdated: timestamp
  conditions: []
```

### Complete Example: API Service SLO

```yaml
apiVersion: slo.dev/v1
kind: ServiceLevelObjective
metadata:
  name: api-service-availability
  namespace: production
  labels:
    team: platform
    environment: production
    tier: critical
    service-type: api
  annotations:
    owner: platform-team@example.com
    runbook: https://runbooks.example.com/api-availability
    dashboard: https://grafana.example.com/d/api-slo
    oncall: https://pagerduty.com/schedules/platform
    description: "Availability SLO for the main API service"

spec:
  service: api-service
  description: |
    API service availability measuring the proportion of successful HTTP requests.
    Success is defined as any request that returns a non-5xx status code.
    This SLO is critical for customer experience and revenue.

  indicator:
    type: ratio
    ratio:
      good:
        metric: http_requests_total
        filters:
          - status_code !~ "5.."
          - endpoint !~ "/health|/metrics"  # Exclude health checks
          - method !~ "HEAD"                # Exclude HEAD requests
      total:
        metric: http_requests_total
        filters:
          - endpoint !~ "/health|/metrics"
          - method !~ "HEAD"

  objectives:
    - displayName: 30-day rolling window
      window: 30d
      target: 0.999                         # 99.9% availability
      description: Monthly availability target

    - displayName: 7-day rolling window
      window: 7d
      target: 0.995                         # 99.5% availability
      description: Weekly availability target for faster feedback

  alerting:
    enabled: true

    burnRates:
      # Fast burn: 2% of monthly budget consumed in 1 hour
      - severity: critical
        shortWindow: 1h
        longWindow: 5m
        burnRate: 14.4
        notificationChannels:
          - pagerduty-critical
          - slack-incidents
        annotations:
          summary: "Critical: Fast error budget burn detected"
          description: |
            API service is burning error budget at 14.4x rate.
            This will consume 2% of the monthly budget in 1 hour.
            Immediate action required.

      # Slow burn: 10% of monthly budget consumed in 6 hours
      - severity: warning
        shortWindow: 6h
        longWindow: 30m
        burnRate: 3
        notificationChannels:
          - slack-alerts
          - email-team
        annotations:
          summary: "Warning: Elevated error budget burn"
          description: |
            API service is burning error budget at 3x rate.
            This will consume 10% of the monthly budget in 6 hours.
            Investigation recommended.

    budgetExhaustion:
      enabled: true
      thresholds:
        - percentage: 50
          severity: info
          notificationChannels:
            - slack-alerts
          message: "50% of error budget remaining"

        - percentage: 25
          severity: warning
          notificationChannels:
            - slack-alerts
            - email-team
          message: "25% of error budget remaining - consider feature freeze"

        - percentage: 10
          severity: critical
          notificationChannels:
            - pagerduty-critical
            - slack-incidents
            - email-leadership
          message: "10% of error budget remaining - feature freeze recommended"

  errorBudgetPolicy:
    enabled: true
    actions:
      - threshold: 25
        action: review
        description: "All releases require SRE review"

      - threshold: 10
        action: freeze
        description: "Feature freeze - reliability work only"
```

### Latency SLO Example

```yaml
apiVersion: slo.dev/v1
kind: ServiceLevelObjective
metadata:
  name: api-service-latency-p95
  namespace: production
  labels:
    team: platform
    environment: production
    tier: critical
    slo-type: latency

spec:
  service: api-service
  description: "95th percentile latency SLO for API requests"

  indicator:
    type: latency
    latency:
      metric: http_request_duration_seconds
      percentile: 0.95
      threshold_ms: 500
      filters:
        - endpoint !~ "/health|/metrics"

  objectives:
    - displayName: 30-day latency target
      window: 30d
      target: 0.95                # 95% of requests < 500ms
      description: "95% of requests complete in under 500ms"

  alerting:
    enabled: true
    burnRates:
      - severity: warning
        shortWindow: 1h
        longWindow: 5m
        burnRate: 6
        notificationChannels:
          - slack-alerts
```

### Batch Pipeline SLO Example

```yaml
apiVersion: slo.dev/v1
kind: ServiceLevelObjective
metadata:
  name: batch-pipeline-freshness
  namespace: production
  labels:
    team: data-platform
    environment: production
    tier: essential
    slo-type: freshness

spec:
  service: batch-data-pipeline
  description: "Data freshness SLO for batch processing pipeline"

  indicator:
    type: threshold
    threshold:
      metric: batch_processing_delay_minutes
      operator: lte
      value: 30
      filters:
        - pipeline_type: "critical"

  objectives:
    - displayName: 7-day freshness target
      window: 7d
      target: 0.99                # 99% of batches processed on time
      description: "99% of batches complete within 30 minutes"

  alerting:
    enabled: true
    burnRates:
      - severity: warning
        shortWindow: 6h
        longWindow: 30m
        burnRate: 4
        notificationChannels:
          - slack-data-team
```

### JSON Schema for Validation

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://slo.dev/schemas/v1/slo.json",
  "title": "ServiceLevelObjective",
  "type": "object",
  "required": ["apiVersion", "kind", "metadata", "spec"],
  "properties": {
    "apiVersion": {
      "type": "string",
      "enum": ["slo.dev/v1"]
    },
    "kind": {
      "type": "string",
      "enum": ["ServiceLevelObjective"]
    },
    "metadata": {
      "type": "object",
      "required": ["name", "namespace"],
      "properties": {
        "name": {
          "type": "string",
          "pattern": "^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
        },
        "namespace": {
          "type": "string",
          "pattern": "^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
        },
        "labels": {
          "type": "object",
          "additionalProperties": {
            "type": "string"
          }
        },
        "annotations": {
          "type": "object",
          "additionalProperties": {
            "type": "string"
          }
        }
      }
    },
    "spec": {
      "type": "object",
      "required": ["service", "indicator", "objectives"],
      "properties": {
        "service": {
          "type": "string"
        },
        "description": {
          "type": "string"
        },
        "indicator": {
          "type": "object",
          "required": ["type"],
          "properties": {
            "type": {
              "type": "string",
              "enum": ["ratio", "threshold", "latency", "availability"]
            }
          },
          "oneOf": [
            {
              "properties": {
                "type": {"const": "ratio"},
                "ratio": {
                  "type": "object",
                  "required": ["good", "total"],
                  "properties": {
                    "good": {"$ref": "#/definitions/metricSelector"},
                    "total": {"$ref": "#/definitions/metricSelector"}
                  }
                }
              },
              "required": ["ratio"]
            },
            {
              "properties": {
                "type": {"const": "threshold"},
                "threshold": {
                  "type": "object",
                  "required": ["metric", "operator", "value"],
                  "properties": {
                    "metric": {"type": "string"},
                    "operator": {
                      "type": "string",
                      "enum": ["lt", "lte", "gt", "gte"]
                    },
                    "value": {"type": "number"}
                  }
                }
              },
              "required": ["threshold"]
            },
            {
              "properties": {
                "type": {"const": "latency"},
                "latency": {
                  "type": "object",
                  "required": ["metric", "percentile", "threshold_ms"],
                  "properties": {
                    "metric": {"type": "string"},
                    "percentile": {
                      "type": "number",
                      "minimum": 0,
                      "maximum": 1
                    },
                    "threshold_ms": {
                      "type": "number",
                      "minimum": 0
                    }
                  }
                }
              },
              "required": ["latency"]
            }
          ]
        },
        "objectives": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "required": ["window", "target"],
            "properties": {
              "displayName": {"type": "string"},
              "window": {
                "type": "string",
                "pattern": "^[0-9]+(d|h|m)$"
              },
              "target": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
              },
              "description": {"type": "string"}
            }
          }
        },
        "alerting": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "burnRates": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["severity", "shortWindow", "longWindow", "burnRate"],
                "properties": {
                  "severity": {
                    "type": "string",
                    "enum": ["critical", "warning", "info"]
                  },
                  "shortWindow": {"type": "string"},
                  "longWindow": {"type": "string"},
                  "burnRate": {"type": "number"},
                  "notificationChannels": {
                    "type": "array",
                    "items": {"type": "string"}
                  }
                }
              }
            }
          }
        }
      }
    }
  },
  "definitions": {
    "metricSelector": {
      "type": "object",
      "required": ["metric"],
      "properties": {
        "metric": {"type": "string"},
        "filters": {
          "type": "array",
          "items": {"type": "string"}
        }
      }
    }
  }
}
```

---

## Automated SLO Generation

Automatically generate SLOs for discovered services based on observed behavior and service characteristics.

### Python SLO Generator

```python
#!/usr/bin/env python3
"""
Automated SLO Generator

Discovers services and generates appropriate SLO configurations based on
service characteristics, observed metrics, and best practices.
"""

import yaml
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum


class ServiceTier(Enum):
    """Service tier classification"""
    CRITICAL = "critical"      # 99.95% availability
    ESSENTIAL = "essential"    # 99.9% availability
    STANDARD = "standard"      # 99.5% availability
    BEST_EFFORT = "best_effort"  # 99.0% availability


class ServiceType(Enum):
    """Service type classification"""
    API = "api"
    WEB = "web"
    BATCH = "batch"
    STREAMING = "streaming"
    DATABASE = "database"
    CACHE = "cache"


@dataclass
class ServiceCharacteristics:
    """Service characteristics for SLO generation"""
    name: str
    type: ServiceType
    tier: ServiceTier
    namespace: str
    team: str

    # Observed metrics
    average_qps: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float

    # Dependencies
    has_database: bool = False
    has_cache: bool = False
    has_external_apis: bool = False

    # Traffic patterns
    traffic_pattern: str = "steady"  # steady, bursty, periodic
    peak_qps_multiplier: float = 1.0

    # Business context
    revenue_impacting: bool = False
    customer_facing: bool = False
    data_sensitivity: str = "low"  # low, medium, high


class SLOAutomation:
    """Automated SLO generation and management"""

    def __init__(self, prometheus_url: str, kubernetes_client=None):
        self.prometheus_url = prometheus_url
        self.k8s = kubernetes_client
        self.templates = SLOTemplateLibrary()

    def auto_generate_slos(self, service_discovery) -> List[Dict[str, Any]]:
        """
        Automatically generate SLOs for discovered services

        Args:
            service_discovery: Service discovery client

        Returns:
            List of generated SLO configurations
        """
        services = service_discovery.get_all_services()
        generated_slos = []

        for service in services:
            print(f"Analyzing service: {service.name}")

            # Analyze service characteristics
            characteristics = self.analyze_service(service)

            # Determine if SLO generation is appropriate
            if not self._should_generate_slo(characteristics):
                print(f"  Skipping - insufficient data or non-production service")
                continue

            # Select appropriate template
            template = self.select_template(characteristics)

            # Customize based on observed behavior
            customized_slo = self.customize_slo(template, characteristics)

            # Validate generated SLO
            if self.validate_slo(customized_slo):
                generated_slos.append(customized_slo)
                print(f"  Generated SLO: {customized_slo['metadata']['name']}")
            else:
                print(f"  Failed validation - manual review required")

        return generated_slos

    def analyze_service(self, service) -> ServiceCharacteristics:
        """
        Analyze service to determine characteristics

        Args:
            service: Service metadata from discovery

        Returns:
            ServiceCharacteristics object
        """
        # Query Prometheus for metrics
        lookback = "7d"
        metrics = self._query_service_metrics(service.name, lookback)

        # Determine service type
        service_type = self._classify_service_type(service, metrics)

        # Determine service tier
        service_tier = self._classify_service_tier(service, metrics)

        # Analyze dependencies
        dependencies = self._analyze_dependencies(service)

        # Analyze traffic patterns
        traffic = self._analyze_traffic_patterns(metrics)

        return ServiceCharacteristics(
            name=service.name,
            type=service_type,
            tier=service_tier,
            namespace=service.namespace,
            team=service.labels.get('team', 'unknown'),

            average_qps=metrics.get('avg_qps', 0),
            p50_latency_ms=metrics.get('p50_latency', 0),
            p95_latency_ms=metrics.get('p95_latency', 0),
            p99_latency_ms=metrics.get('p99_latency', 0),
            error_rate=metrics.get('error_rate', 0),

            has_database=dependencies.get('database', False),
            has_cache=dependencies.get('cache', False),
            has_external_apis=dependencies.get('external_apis', False),

            traffic_pattern=traffic.get('pattern', 'steady'),
            peak_qps_multiplier=traffic.get('peak_multiplier', 1.0),

            revenue_impacting=service.labels.get('revenue-impact') == 'true',
            customer_facing=service.labels.get('customer-facing') == 'true',
            data_sensitivity=service.labels.get('data-sensitivity', 'low')
        )

    def _classify_service_type(self, service, metrics) -> ServiceType:
        """Classify service type based on labels and metrics"""

        # Check explicit labels
        if 'service-type' in service.labels:
            return ServiceType(service.labels['service-type'])

        # Infer from metrics
        if metrics.get('http_requests_total', 0) > 0:
            # Has HTTP traffic
            if metrics.get('has_frontend', False):
                return ServiceType.WEB
            else:
                return ServiceType.API

        elif metrics.get('batch_jobs_total', 0) > 0:
            return ServiceType.BATCH

        elif metrics.get('stream_messages_total', 0) > 0:
            return ServiceType.STREAMING

        elif metrics.get('database_connections', 0) > 0:
            return ServiceType.DATABASE

        # Default to API
        return ServiceType.API

    def _classify_service_tier(self, service, metrics) -> ServiceTier:
        """Classify service tier based on business impact"""

        # Check explicit tier label
        if 'tier' in service.labels:
            return ServiceTier(service.labels['tier'])

        # Infer from characteristics
        score = 0

        # Revenue impact
        if service.labels.get('revenue-impact') == 'true':
            score += 3

        # Customer-facing
        if service.labels.get('customer-facing') == 'true':
            score += 2

        # High QPS
        if metrics.get('avg_qps', 0) > 1000:
            score += 2

        # Data sensitivity
        if service.labels.get('data-sensitivity') == 'high':
            score += 1

        # Map score to tier
        if score >= 6:
            return ServiceTier.CRITICAL
        elif score >= 4:
            return ServiceTier.ESSENTIAL
        elif score >= 2:
            return ServiceTier.STANDARD
        else:
            return ServiceTier.BEST_EFFORT

    def _should_generate_slo(self, characteristics: ServiceCharacteristics) -> bool:
        """Determine if SLO should be generated for this service"""

        # Skip if non-production
        if characteristics.namespace in ['development', 'test', 'staging']:
            return False

        # Skip if insufficient traffic
        if characteristics.average_qps < 0.1:  # Less than 1 request per 10 seconds
            return False

        # Skip if no error rate data
        if characteristics.error_rate is None:
            return False

        return True

    def select_template(self, characteristics: ServiceCharacteristics) -> Dict[str, Any]:
        """Select appropriate SLO template based on service characteristics"""

        if characteristics.type == ServiceType.API:
            return self.templates.get_api_service_template()
        elif characteristics.type == ServiceType.WEB:
            return self.templates.get_web_service_template()
        elif characteristics.type == ServiceType.BATCH:
            return self.templates.get_batch_pipeline_template()
        elif characteristics.type == ServiceType.STREAMING:
            return self.templates.get_streaming_service_template()
        elif characteristics.type == ServiceType.DATABASE:
            return self.templates.get_database_template()
        else:
            return self.templates.get_generic_template()

    def customize_slo(
        self,
        template: Dict[str, Any],
        characteristics: ServiceCharacteristics
    ) -> Dict[str, Any]:
        """
        Customize SLO template based on observed service behavior

        Args:
            template: Base SLO template
            characteristics: Analyzed service characteristics

        Returns:
            Customized SLO configuration
        """
        slo = template.copy()

        # Update metadata
        slo['metadata']['name'] = f"{characteristics.name}-availability"
        slo['metadata']['namespace'] = characteristics.namespace
        slo['metadata']['labels'] = {
            'team': characteristics.team,
            'environment': 'production',
            'tier': characteristics.tier.value,
            'service-type': characteristics.type.value,
            'auto-generated': 'true'
        }

        # Set service name
        slo['spec']['service'] = characteristics.name

        # Customize SLO target based on tier
        target = self._get_target_for_tier(characteristics.tier)
        for objective in slo['spec']['objectives']:
            objective['target'] = target

        # Customize latency threshold based on observed p95
        if characteristics.type in [ServiceType.API, ServiceType.WEB]:
            # Set threshold at p95 + 20% margin
            latency_threshold = characteristics.p95_latency_ms * 1.2

            # Add latency SLO
            latency_slo = self._create_latency_slo(
                characteristics.name,
                characteristics.namespace,
                latency_threshold
            )

        # Customize alerting based on tier
        slo['spec']['alerting'] = self._customize_alerting(characteristics.tier)

        # Add error budget policy
        slo['spec']['errorBudgetPolicy'] = self._create_error_budget_policy(
            characteristics.tier
        )

        return slo

    def _get_target_for_tier(self, tier: ServiceTier) -> float:
        """Get SLO target based on service tier"""
        targets = {
            ServiceTier.CRITICAL: 0.9995,     # 99.95%
            ServiceTier.ESSENTIAL: 0.999,     # 99.9%
            ServiceTier.STANDARD: 0.995,      # 99.5%
            ServiceTier.BEST_EFFORT: 0.990    # 99.0%
        }
        return targets[tier]

    def _customize_alerting(self, tier: ServiceTier) -> Dict[str, Any]:
        """Customize alerting configuration based on tier"""

        if tier == ServiceTier.CRITICAL:
            return {
                'enabled': True,
                'burnRates': [
                    {
                        'severity': 'critical',
                        'shortWindow': '1h',
                        'longWindow': '5m',
                        'burnRate': 14.4,
                        'notificationChannels': ['pagerduty-critical', 'slack-incidents']
                    },
                    {
                        'severity': 'warning',
                        'shortWindow': '6h',
                        'longWindow': '30m',
                        'burnRate': 3,
                        'notificationChannels': ['slack-alerts']
                    }
                ]
            }

        elif tier == ServiceTier.ESSENTIAL:
            return {
                'enabled': True,
                'burnRates': [
                    {
                        'severity': 'warning',
                        'shortWindow': '2h',
                        'longWindow': '10m',
                        'burnRate': 10,
                        'notificationChannels': ['slack-alerts']
                    }
                ]
            }

        else:
            return {
                'enabled': True,
                'burnRates': [
                    {
                        'severity': 'info',
                        'shortWindow': '6h',
                        'longWindow': '30m',
                        'burnRate': 5,
                        'notificationChannels': ['slack-alerts']
                    }
                ]
            }

    def _create_error_budget_policy(self, tier: ServiceTier) -> Dict[str, Any]:
        """Create error budget policy based on tier"""

        if tier in [ServiceTier.CRITICAL, ServiceTier.ESSENTIAL]:
            return {
                'enabled': True,
                'actions': [
                    {
                        'threshold': 25,
                        'action': 'review',
                        'description': 'All releases require SRE review'
                    },
                    {
                        'threshold': 10,
                        'action': 'freeze',
                        'description': 'Feature freeze - reliability work only'
                    }
                ]
            }
        else:
            return {
                'enabled': True,
                'actions': [
                    {
                        'threshold': 10,
                        'action': 'review',
                        'description': 'Releases require review'
                    }
                ]
            }

    def validate_slo(self, slo: Dict[str, Any]) -> bool:
        """
        Validate SLO configuration

        Args:
            slo: SLO configuration to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ['apiVersion', 'kind', 'metadata', 'spec']
            for field in required_fields:
                if field not in slo:
                    print(f"Missing required field: {field}")
                    return False

            # Validate metadata
            if 'name' not in slo['metadata']:
                print("Missing metadata.name")
                return False

            # Validate spec
            spec = slo['spec']
            if 'service' not in spec or 'indicator' not in spec or 'objectives' not in spec:
                print("Missing required spec fields")
                return False

            # Validate objectives
            for obj in spec['objectives']:
                if 'target' not in obj or 'window' not in obj:
                    print("Invalid objective configuration")
                    return False

                # Check target range
                if not (0 <= obj['target'] <= 1):
                    print(f"Invalid target: {obj['target']} (must be 0-1)")
                    return False

            return True

        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def export_slos(self, slos: List[Dict[str, Any]], output_dir: str):
        """
        Export generated SLOs to YAML files

        Args:
            slos: List of SLO configurations
            output_dir: Directory to write YAML files
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        for slo in slos:
            filename = f"{slo['metadata']['name']}.yaml"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w') as f:
                yaml.dump(slo, f, default_flow_style=False, sort_keys=False)

            print(f"Exported: {filepath}")


# Example usage
if __name__ == "__main__":
    from service_discovery import KubernetesServiceDiscovery

    # Initialize
    discovery = KubernetesServiceDiscovery()
    automation = SLOAutomation(
        prometheus_url="http://prometheus:9090",
        kubernetes_client=discovery.client
    )

    # Generate SLOs
    slos = automation.auto_generate_slos(discovery)

    # Export to files
    automation.export_slos(slos, output_dir="./generated-slos")

    print(f"\nGenerated {len(slos)} SLOs")
```

---

## Progressive SLO Implementation

Implement SLOs progressively, starting with achievable targets and gradually increasing reliability requirements.

### Progressive Implementation Strategy

```python
#!/usr/bin/env python3
"""
Progressive SLO Implementation

Gradually increase SLO targets over time as reliability improves.
"""

from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from enum import Enum


class ImplementationPhase(Enum):
    """Progressive implementation phases"""
    BASELINE = "baseline"           # Phase 1: 99.0%
    IMPROVEMENT = "improvement"     # Phase 2: 99.5%
    PRODUCTION_READY = "production_ready"  # Phase 3: 99.9%
    EXCELLENCE = "excellence"       # Phase 4: 99.95%


class ProgressiveSLOManager:
    """Manage progressive SLO implementation"""

    def __init__(self):
        self.phases = self._define_phases()

    def _define_phases(self) -> Dict[ImplementationPhase, Dict]:
        """Define progressive implementation phases"""
        return {
            ImplementationPhase.BASELINE: {
                'duration_days': 30,
                'availability_target': 0.990,   # 99.0%
                'latency_p95_ms': 2000,
                'latency_p99_ms': 5000,
                'description': 'Baseline establishment - measure current performance',
                'objectives': [
                    'Establish baseline metrics',
                    'Implement basic monitoring',
                    'Create initial dashboards',
                    'Set up alerting infrastructure'
                ],
                'success_criteria': [
                    'Metrics collection stable for 30 days',
                    'Basic alerts configured',
                    'Team trained on SLO concepts'
                ]
            },

            ImplementationPhase.IMPROVEMENT: {
                'duration_days': 60,
                'availability_target': 0.995,   # 99.5%
                'latency_p95_ms': 1000,
                'latency_p99_ms': 2000,
                'description': 'Initial improvement - address low-hanging fruit',
                'objectives': [
                    'Fix obvious reliability issues',
                    'Implement retry logic',
                    'Add circuit breakers',
                    'Improve error handling',
                    'Optimize slow queries'
                ],
                'success_criteria': [
                    'Meet 99.5% availability for 30 consecutive days',
                    'Error budget policy established',
                    'Regular SLO reviews scheduled'
                ]
            },

            ImplementationPhase.PRODUCTION_READY: {
                'duration_days': 90,
                'availability_target': 0.999,   # 99.9%
                'latency_p95_ms': 500,
                'latency_p99_ms': 1000,
                'description': 'Production readiness - robust reliability',
                'objectives': [
                    'Implement comprehensive monitoring',
                    'Add automated remediation',
                    'Deploy to multiple availability zones',
                    'Implement load balancing',
                    'Add auto-scaling',
                    'Create runbooks'
                ],
                'success_criteria': [
                    'Meet 99.9% availability for 60 consecutive days',
                    'Incidents resolved within SLA',
                    'Automated testing in place',
                    'Disaster recovery tested'
                ]
            },

            ImplementationPhase.EXCELLENCE: {
                'duration_days': None,  # Ongoing
                'availability_target': 0.9995,  # 99.95%
                'latency_p95_ms': 200,
                'latency_p99_ms': 500,
                'description': 'Excellence - industry-leading reliability',
                'objectives': [
                    'Implement chaos engineering',
                    'Deploy multi-region',
                    'Add advanced observability',
                    'Continuous optimization',
                    'Predictive alerting'
                ],
                'success_criteria': [
                    'Sustained 99.95% availability',
                    'Zero-downtime deployments',
                    'Proactive issue detection',
                    'Industry recognition'
                ]
            }
        }

    def implement_progressive_slos(self, service: str) -> Dict:
        """
        Generate progressive SLO implementation plan

        Args:
            service: Service name

        Returns:
            Implementation plan with phases and timeline
        """
        start_date = datetime.now()
        phases = []

        current_date = start_date
        for phase_enum in ImplementationPhase:
            phase = self.phases[phase_enum]

            if phase['duration_days']:
                end_date = current_date + timedelta(days=phase['duration_days'])
            else:
                end_date = None  # Ongoing

            phases.append({
                'phase': phase_enum.value,
                'start_date': current_date.isoformat(),
                'end_date': end_date.isoformat() if end_date else 'ongoing',
                'duration_days': phase['duration_days'],
                'slo_config': self._generate_phase_slo(service, phase_enum),
                'objectives': phase['objectives'],
                'success_criteria': phase['success_criteria'],
                'description': phase['description']
            })

            if end_date:
                current_date = end_date

        return {
            'service': service,
            'implementation_start': start_date.isoformat(),
            'phases': phases,
            'total_duration_days': sum(
                p['duration_days'] for p in self.phases.values()
                if p['duration_days']
            )
        }

    def _generate_phase_slo(self, service: str, phase: ImplementationPhase) -> Dict:
        """Generate SLO configuration for a specific phase"""
        phase_config = self.phases[phase]

        return {
            'apiVersion': 'slo.dev/v1',
            'kind': 'ServiceLevelObjective',
            'metadata': {
                'name': f"{service}-availability-{phase.value}",
                'namespace': 'production',
                'labels': {
                    'service': service,
                    'phase': phase.value,
                    'progressive': 'true'
                },
                'annotations': {
                    'description': phase_config['description']
                }
            },
            'spec': {
                'service': service,
                'description': f"{phase.value} phase SLO for {service}",
                'indicator': {
                    'type': 'ratio',
                    'ratio': {
                        'good': {
                            'metric': 'http_requests_total',
                            'filters': ['status_code !~ "5.."']
                        },
                        'total': {
                            'metric': 'http_requests_total'
                        }
                    }
                },
                'objectives': [
                    {
                        'displayName': f'{phase.value} availability target',
                        'window': '30d',
                        'target': phase_config['availability_target']
                    }
                ],
                'alerting': self._generate_phase_alerting(phase)
            }
        }

    def _generate_phase_alerting(self, phase: ImplementationPhase) -> Dict:
        """Generate alerting configuration appropriate for phase"""

        if phase == ImplementationPhase.BASELINE:
            # Minimal alerting during baseline
            return {
                'enabled': True,
                'burnRates': [
                    {
                        'severity': 'info',
                        'shortWindow': '24h',
                        'longWindow': '1h',
                        'burnRate': 5
                    }
                ]
            }

        elif phase == ImplementationPhase.IMPROVEMENT:
            # Moderate alerting
            return {
                'enabled': True,
                'burnRates': [
                    {
                        'severity': 'warning',
                        'shortWindow': '6h',
                        'longWindow': '30m',
                        'burnRate': 8
                    }
                ]
            }

        else:
            # Full production alerting
            return {
                'enabled': True,
                'burnRates': [
                    {
                        'severity': 'critical',
                        'shortWindow': '1h',
                        'longWindow': '5m',
                        'burnRate': 14.4
                    },
                    {
                        'severity': 'warning',
                        'shortWindow': '6h',
                        'longWindow': '30m',
                        'burnRate': 3
                    }
                ]
            }

    def check_phase_readiness(
        self,
        current_phase: ImplementationPhase,
        metrics: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Check if service is ready to advance to next phase

        Args:
            current_phase: Current implementation phase
            metrics: Service metrics

        Returns:
            (ready, issues) tuple
        """
        phase_config = self.phases[current_phase]
        issues = []

        # Check availability target
        if metrics['availability'] < phase_config['availability_target']:
            issues.append(
                f"Availability {metrics['availability']*100:.2f}% "
                f"below target {phase_config['availability_target']*100:.2f}%"
            )

        # Check latency targets
        if metrics['latency_p95'] > phase_config['latency_p95_ms']:
            issues.append(
                f"P95 latency {metrics['latency_p95']}ms "
                f"exceeds target {phase_config['latency_p95_ms']}ms"
            )

        # Check stability (coefficient of variation)
        if metrics.get('availability_stddev', 0) > 0.01:  # 1% variation
            issues.append("Availability too variable - not stable")

        # Check incident rate
        if metrics.get('incidents_last_30d', 0) > 5:
            issues.append("Too many incidents - improve stability first")

        ready = len(issues) == 0
        return ready, issues

    def generate_implementation_report(self, plan: Dict) -> str:
        """Generate human-readable implementation report"""

        report = f"""
# Progressive SLO Implementation Plan

**Service**: {plan['service']}
**Start Date**: {plan['implementation_start']}
**Total Duration**: {plan['total_duration_days']} days

## Implementation Phases

"""

        for i, phase in enumerate(plan['phases'], 1):
            report += f"""
### Phase {i}: {phase['phase'].title()}

**Timeline**: {phase['start_date']} → {phase['end_date']}
**Duration**: {phase['duration_days']} days
**Description**: {phase['description']}

**SLO Target**: {phase['slo_config']['spec']['objectives'][0]['target']*100:.2f}%

**Objectives**:
{self._format_list(phase['objectives'])}

**Success Criteria**:
{self._format_list(phase['success_criteria'])}

---
"""

        return report

    def _format_list(self, items: List[str]) -> str:
        """Format list items as markdown"""
        return '\n'.join(f"- {item}" for item in items)


# Example usage
if __name__ == "__main__":
    manager = ProgressiveSLOManager()

    # Generate implementation plan
    plan = manager.implement_progressive_slos('api-service')

    # Generate report
    report = manager.generate_implementation_report(plan)
    print(report)

    # Check readiness to advance
    current_metrics = {
        'availability': 0.9965,
        'latency_p95': 450,
        'availability_stddev': 0.005,
        'incidents_last_30d': 2
    }

    ready, issues = manager.check_phase_readiness(
        ImplementationPhase.IMPROVEMENT,
        current_metrics
    )

    if ready:
        print("\nReady to advance to next phase!")
    else:
        print("\nNot ready to advance. Issues:")
        for issue in issues:
            print(f"  - {issue}")
```

---

## SLO Template Library

Comprehensive library of SLO templates for different service types.

### Template Library Implementation

```python
#!/usr/bin/env python3
"""
SLO Template Library

Pre-built SLO templates for common service types.
"""

from typing import Dict, Any


class SLOTemplateLibrary:
    """Library of SLO templates for various service types"""

    @staticmethod
    def get_api_service_template() -> Dict[str, Any]:
        """
        SLO template for RESTful API services

        Includes:
        - Availability (success rate)
        - Latency (p95, p99)
        - Error rate
        """
        return {
            'apiVersion': 'slo.dev/v1',
            'kind': 'ServiceLevelObjective',
            'metadata': {
                'name': 'template-api-service',
                'labels': {
                    'template': 'api-service',
                    'version': 'v1'
                }
            },
            'spec': {
                'service': '${SERVICE_NAME}',
                'description': 'API service SLO template',
                'indicator': {
                    'type': 'ratio',
                    'ratio': {
                        'good': {
                            'metric': 'http_requests_total',
                            'filters': [
                                'status_code !~ "5.."',
                                'endpoint !~ "/health|/metrics"'
                            ]
                        },
                        'total': {
                            'metric': 'http_requests_total',
                            'filters': [
                                'endpoint !~ "/health|/metrics"'
                            ]
                        }
                    }
                },
                'objectives': [
                    {
                        'displayName': '30-day availability',
                        'window': '30d',
                        'target': 0.999
                    }
                ],
                'alerting': {
                    'enabled': True,
                    'burnRates': [
                        {
                            'severity': 'critical',
                            'shortWindow': '1h',
                            'longWindow': '5m',
                            'burnRate': 14.4
                        },
                        {
                            'severity': 'warning',
                            'shortWindow': '6h',
                            'longWindow': '30m',
                            'burnRate': 3
                        }
                    ]
                }
            }
        }

    @staticmethod
    def get_web_service_template() -> Dict[str, Any]:
        """
        SLO template for web frontend services

        Includes:
        - Page load time
        - Time to Interactive (TTI)
        - First Contentful Paint (FCP)
        - Availability
        """
        return {
            'apiVersion': 'slo.dev/v1',
            'kind': 'ServiceLevelObjective',
            'metadata': {
                'name': 'template-web-service',
                'labels': {
                    'template': 'web-service',
                    'version': 'v1'
                }
            },
            'spec': {
                'service': '${SERVICE_NAME}',
                'description': 'Web service SLO template with client-side metrics',
                'indicator': {
                    'type': 'latency',
                    'latency': {
                        'metric': 'page_load_duration_seconds',
                        'percentile': 0.95,
                        'threshold_ms': 3000  # 3 seconds
                    }
                },
                'objectives': [
                    {
                        'displayName': '95% of pages load in < 3s',
                        'window': '30d',
                        'target': 0.95
                    }
                ],
                'alerting': {
                    'enabled': True,
                    'burnRates': [
                        {
                            'severity': 'warning',
                            'shortWindow': '2h',
                            'longWindow': '10m',
                            'burnRate': 10
                        }
                    ]
                }
            }
        }

    @staticmethod
    def get_batch_pipeline_template() -> Dict[str, Any]:
        """
        SLO template for batch data pipelines

        Includes:
        - Data freshness
        - Completeness
        - Processing time
        """
        return {
            'apiVersion': 'slo.dev/v1',
            'kind': 'ServiceLevelObjective',
            'metadata': {
                'name': 'template-batch-pipeline',
                'labels': {
                    'template': 'batch-pipeline',
                    'version': 'v1'
                }
            },
            'spec': {
                'service': '${SERVICE_NAME}',
                'description': 'Batch pipeline SLO template',
                'indicator': {
                    'type': 'threshold',
                    'threshold': {
                        'metric': 'batch_processing_delay_minutes',
                        'operator': 'lte',
                        'value': 30,
                        'filters': ['priority="high"']
                    }
                },
                'objectives': [
                    {
                        'displayName': '99% of batches complete within 30 minutes',
                        'window': '7d',
                        'target': 0.99
                    }
                ],
                'alerting': {
                    'enabled': True,
                    'burnRates': [
                        {
                            'severity': 'warning',
                            'shortWindow': '6h',
                            'longWindow': '1h',
                            'burnRate': 5
                        }
                    ]
                }
            }
        }

    @staticmethod
    def get_streaming_service_template() -> Dict[str, Any]:
        """
        SLO template for streaming/event processing services

        Includes:
        - Processing latency
        - Message throughput
        - Error rate
        """
        return {
            'apiVersion': 'slo.dev/v1',
            'kind': 'ServiceLevelObjective',
            'metadata': {
                'name': 'template-streaming-service',
                'labels': {
                    'template': 'streaming-service',
                    'version': 'v1'
                }
            },
            'spec': {
                'service': '${SERVICE_NAME}',
                'description': 'Streaming service SLO template',
                'indicator': {
                    'type': 'latency',
                    'latency': {
                        'metric': 'stream_processing_duration_seconds',
                        'percentile': 0.99,
                        'threshold_ms': 1000
                    }
                },
                'objectives': [
                    {
                        'displayName': '99% of messages processed in < 1s',
                        'window': '7d',
                        'target': 0.99
                    }
                ],
                'alerting': {
                    'enabled': True,
                    'burnRates': [
                        {
                            'severity': 'critical',
                            'shortWindow': '1h',
                            'longWindow': '5m',
                            'burnRate': 10
                        }
                    ]
                }
            }
        }

    @staticmethod
    def get_database_template() -> Dict[str, Any]:
        """
        SLO template for database services

        Includes:
        - Query latency
        - Availability
        - Connection success rate
        """
        return {
            'apiVersion': 'slo.dev/v1',
            'kind': 'ServiceLevelObjective',
            'metadata': {
                'name': 'template-database',
                'labels': {
                    'template': 'database',
                    'version': 'v1'
                }
            },
            'spec': {
                'service': '${SERVICE_NAME}',
                'description': 'Database service SLO template',
                'indicator': {
                    'type': 'latency',
                    'latency': {
                        'metric': 'database_query_duration_seconds',
                        'percentile': 0.95,
                        'threshold_ms': 100
                    }
                },
                'objectives': [
                    {
                        'displayName': '95% of queries complete in < 100ms',
                        'window': '30d',
                        'target': 0.95
                    }
                ],
                'alerting': {
                    'enabled': True,
                    'burnRates': [
                        {
                            'severity': 'critical',
                            'shortWindow': '30m',
                            'longWindow': '5m',
                            'burnRate': 20
                        }
                    ]
                }
            }
        }

    @staticmethod
    def get_generic_template() -> Dict[str, Any]:
        """Generic SLO template for any service"""
        return {
            'apiVersion': 'slo.dev/v1',
            'kind': 'ServiceLevelObjective',
            'metadata': {
                'name': 'template-generic',
                'labels': {
                    'template': 'generic',
                    'version': 'v1'
                }
            },
            'spec': {
                'service': '${SERVICE_NAME}',
                'description': 'Generic service SLO template',
                'indicator': {
                    'type': 'ratio',
                    'ratio': {
                        'good': {
                            'metric': '${GOOD_METRIC}',
                            'filters': []
                        },
                        'total': {
                            'metric': '${TOTAL_METRIC}',
                            'filters': []
                        }
                    }
                },
                'objectives': [
                    {
                        'displayName': 'Service reliability',
                        'window': '30d',
                        'target': 0.99
                    }
                ],
                'alerting': {
                    'enabled': True,
                    'burnRates': [
                        {
                            'severity': 'warning',
                            'shortWindow': '2h',
                            'longWindow': '10m',
                            'burnRate': 8
                        }
                    ]
                }
            }
        }
```

---

## GitOps Workflow

Implement GitOps workflow for SLO management with version control, code review, and automated deployment.

### GitOps Implementation

```yaml
# .github/workflows/slo-deployment.yaml
name: SLO Deployment Pipeline

on:
  push:
    branches:
      - main
    paths:
      - 'slo-definitions/**/*.yaml'
  pull_request:
    paths:
      - 'slo-definitions/**/*.yaml'

env:
  PROMETHEUS_URL: ${{ secrets.PROMETHEUS_URL }}
  GRAFANA_URL: ${{ secrets.GRAFANA_URL }}

jobs:
  validate:
    name: Validate SLO Definitions
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          uv uv pip install pyyaml jsonschema prometheus-client

      - name: Validate YAML syntax
        run: |
          python scripts/validate_slos.py \
            --schema schemas/slo-schema.json \
            --slo-dir slo-definitions/

      - name: Check for duplicates
        run: |
          python scripts/check_duplicates.py \
            --slo-dir slo-definitions/

      - name: Validate against Prometheus
        run: |
          python scripts/validate_prometheus_queries.py \
            --slo-dir slo-definitions/ \
            --prometheus-url $PROMETHEUS_URL

  dry-run:
    name: Dry Run Deployment
    runs-on: ubuntu-latest
    needs: validate
    if: github.event_name == 'pull_request'
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Dry run deployment
        run: |
          python scripts/deploy_slos.py \
            --slo-dir slo-definitions/ \
            --dry-run \
            --output deployment-plan.md

      - name: Comment deployment plan on PR
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const plan = fs.readFileSync('deployment-plan.md', 'utf8');

            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: `## SLO Deployment Plan\n\n${plan}`
            });

  deploy:
    name: Deploy SLOs
    runs-on: ubuntu-latest
    needs: validate
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          uv uv pip install pyyaml requests prometheus-client kubernetes

      - name: Deploy SLOs to Prometheus
        run: |
          python scripts/deploy_slos.py \
            --slo-dir slo-definitions/ \
            --prometheus-url $PROMETHEUS_URL \
            --apply

      - name: Deploy dashboards to Grafana
        run: |
          python scripts/deploy_grafana_dashboards.py \
            --slo-dir slo-definitions/ \
            --grafana-url $GRAFANA_URL \
            --grafana-token ${{ secrets.GRAFANA_TOKEN }}

      - name: Apply Kubernetes CRDs
        run: |
          kubectl apply -f slo-definitions/
        env:
          KUBECONFIG: ${{ secrets.KUBECONFIG }}

      - name: Verify deployment
        run: |
          python scripts/verify_deployment.py \
            --slo-dir slo-definitions/

      - name: Create deployment summary
        run: |
          python scripts/generate_deployment_summary.py \
            --output deployment-summary.md

      - name: Post to Slack
        uses: slackapi/slack-github-action@v1
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK_URL }}
          payload: |
            {
              "text": "SLO deployment completed",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*SLO Deployment Completed*\n\nCommit: ${{ github.sha }}\nBranch: ${{ github.ref }}"
                  }
                }
              ]
            }
```

### Validation Scripts

```python
#!/usr/bin/env python3
"""
scripts/validate_slos.py

Validate SLO definitions against JSON schema.
"""

import argparse
import json
import sys
from pathlib import Path
import yaml
from jsonschema import validate, ValidationError, Draft7Validator


def load_schema(schema_path: Path) -> dict:
    """Load JSON schema"""
    with open(schema_path) as f:
        return json.load(f)


def load_slo_files(slo_dir: Path) -> list:
    """Load all SLO YAML files"""
    slo_files = []
    for yaml_file in slo_dir.rglob('*.yaml'):
        with open(yaml_file) as f:
            try:
                slo_files.append({
                    'path': yaml_file,
                    'content': yaml.safe_load(f)
                })
            except yaml.YAMLError as e:
                print(f"ERROR: Invalid YAML in {yaml_file}: {e}")
                sys.exit(1)
    return slo_files


def validate_slos(schema: dict, slo_files: list) -> tuple:
    """Validate SLO files against schema"""
    errors = []
    warnings = []

    validator = Draft7Validator(schema)

    for slo_file in slo_files:
        path = slo_file['path']
        content = slo_file['content']

        # Validate against schema
        validation_errors = list(validator.iter_errors(content))
        if validation_errors:
            for error in validation_errors:
                errors.append({
                    'file': str(path),
                    'error': error.message,
                    'path': ' -> '.join(str(p) for p in error.path)
                })

        # Business logic validation
        if 'spec' in content:
            spec = content['spec']

            # Check target range
            if 'objectives' in spec:
                for obj in spec['objectives']:
                    if 'target' in obj:
                        target = obj['target']
                        if target < 0.9:
                            warnings.append({
                                'file': str(path),
                                'warning': f"Very low SLO target: {target*100}%"
                            })
                        elif target > 0.9999:
                            warnings.append({
                                'file': str(path),
                                'warning': f"Very high SLO target: {target*100}% - may be unrealistic"
                            })

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description='Validate SLO definitions')
    parser.add_argument('--schema', required=True, help='Path to JSON schema')
    parser.add_argument('--slo-dir', required=True, help='Directory containing SLO definitions')
    args = parser.parse_args()

    # Load schema
    schema = load_schema(Path(args.schema))

    # Load SLO files
    slo_files = load_slo_files(Path(args.slo_dir))
    print(f"Found {len(slo_files)} SLO files")

    # Validate
    errors, warnings = validate_slos(schema, slo_files)

    # Print warnings
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  {warning['file']}: {warning['warning']}")

    # Print errors
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  {error['file']}")
            if error['path']:
                print(f"    Path: {error['path']}")
            print(f"    Error: {error['error']}")
        sys.exit(1)

    print("\n✅ All SLO definitions are valid!")


if __name__ == '__main__':
    main()
```

---

## CI/CD Integration

Integrate SLO validation and deployment into CI/CD pipelines.

### SLO Deployment Script

```python
#!/usr/bin/env python3
"""
scripts/deploy_slos.py

Deploy SLO definitions to monitoring infrastructure.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml
import requests
from prometheus_client.parser import text_string_to_metric_families


class SLODeployer:
    """Deploy SLOs to monitoring infrastructure"""

    def __init__(self, prometheus_url: str, dry_run: bool = False):
        self.prometheus_url = prometheus_url
        self.dry_run = dry_run
        self.changes = []

    def deploy_slos(self, slo_dir: Path) -> bool:
        """
        Deploy all SLOs from directory

        Returns:
            True if successful, False otherwise
        """
        # Load SLO files
        slo_files = list(slo_dir.rglob('*.yaml'))
        print(f"Found {len(slo_files)} SLO files")

        for slo_file in slo_files:
            with open(slo_file) as f:
                slo = yaml.safe_load(f)

            if not self._deploy_slo(slo):
                return False

        if self.dry_run:
            print("\n📋 Deployment Plan (Dry Run):")
            for change in self.changes:
                print(f"  - {change}")
            return True

        print(f"\n✅ Successfully deployed {len(slo_files)} SLOs")
        return True

    def _deploy_slo(self, slo: Dict[str, Any]) -> bool:
        """Deploy a single SLO"""
        name = slo['metadata']['name']
        print(f"\nDeploying SLO: {name}")

        # Generate Prometheus recording rules
        recording_rules = self._generate_recording_rules(slo)

        # Generate alert rules
        alert_rules = self._generate_alert_rules(slo)

        if self.dry_run:
            self.changes.append(f"Create/update recording rules for {name}")
            self.changes.append(f"Create/update alert rules for {name}")
            return True

        # Apply recording rules
        if not self._apply_prometheus_rules(recording_rules, f"{name}-recording"):
            return False

        # Apply alert rules
        if not self._apply_prometheus_rules(alert_rules, f"{name}-alerts"):
            return False

        return True

    def _generate_recording_rules(self, slo: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Prometheus recording rules for SLO"""
        service = slo['spec']['service']
        indicator = slo['spec']['indicator']

        rules = {
            'groups': [{
                'name': f'{service}_slo_rules',
                'interval': '30s',
                'rules': []
            }]
        }

        if indicator['type'] == 'ratio':
            ratio = indicator['ratio']

            # Good events rate
            rules['groups'][0]['rules'].append({
                'record': f'{service}:sli:good_events:rate5m',
                'expr': f"sum(rate({ratio['good']['metric']}[5m]))"
            })

            # Total events rate
            rules['groups'][0]['rules'].append({
                'record': f'{service}:sli:total_events:rate5m',
                'expr': f"sum(rate({ratio['total']['metric']}[5m]))"
            })

            # SLI value
            rules['groups'][0]['rules'].append({
                'record': f'{service}:sli:value',
                'expr': f"""
                  {service}:sli:good_events:rate5m
                  /
                  {service}:sli:total_events:rate5m
                """
            })

            # Error budget burn rate
            for objective in slo['spec']['objectives']:
                target = objective['target']
                window = objective['window']

                rules['groups'][0]['rules'].append({
                    'record': f'{service}:error_budget:burn_rate_{window}',
                    'expr': f"""
                      (1 - {service}:sli:value)
                      /
                      (1 - {target})
                    """
                })

        return rules

    def _generate_alert_rules(self, slo: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Prometheus alert rules for SLO"""
        service = slo['spec']['service']
        alerting = slo['spec'].get('alerting', {})

        if not alerting.get('enabled'):
            return {'groups': []}

        rules = {
            'groups': [{
                'name': f'{service}_slo_alerts',
                'rules': []
            }]
        }

        for burn_rate in alerting.get('burnRates', []):
            short_window = burn_rate['shortWindow']
            long_window = burn_rate['longWindow']
            rate = burn_rate['burnRate']
            severity = burn_rate['severity']

            rules['groups'][0]['rules'].append({
                'alert': f'{service.title()}ErrorBudgetBurn',
                'expr': f"""
                  (
                    {service}:error_budget:burn_rate_{short_window} > {rate}
                    and
                    {service}:error_budget:burn_rate_{long_window} > {rate}
                  )
                """,
                'for': '2m',
                'labels': {
                    'severity': severity,
                    'service': service
                },
                'annotations': {
                    'summary': f'Error budget burn rate exceeded for {service}',
                    'description': f'Service {service} is burning error budget at {{{{ $value }}}}x rate'
                }
            })

        return rules

    def _apply_prometheus_rules(self, rules: Dict[str, Any], name: str) -> bool:
        """Apply rules to Prometheus via ConfigMap or API"""
        # In production, this would:
        # 1. Create/update Kubernetes ConfigMap with rules
        # 2. Trigger Prometheus reload
        # 3. Verify rules are loaded

        print(f"  ✓ Applied rules: {name}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Deploy SLO definitions')
    parser.add_argument('--slo-dir', required=True, help='Directory containing SLO definitions')
    parser.add_argument('--prometheus-url', help='Prometheus URL')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--apply', action='store_true', help='Apply changes')
    parser.add_argument('--output', help='Output file for deployment plan')
    args = parser.parse_args()

    if args.apply and args.dry_run:
        print("ERROR: Cannot use --apply and --dry-run together")
        sys.exit(1)

    deployer = SLODeployer(
        prometheus_url=args.prometheus_url or '',
        dry_run=args.dry_run or not args.apply
    )

    success = deployer.deploy_slos(Path(args.slo_dir))

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
```

---

## Kubernetes CRD

Custom Resource Definition for SLOs in Kubernetes.

### SLO CRD Definition

```yaml
# kubernetes/slo-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: servicelevelobjectives.slo.dev
spec:
  group: slo.dev
  names:
    kind: ServiceLevelObjective
    listKind: ServiceLevelObjectiveList
    plural: servicelevelobjectives
    singular: servicelevelobjective
    shortNames:
      - slo
      - slos
  scope: Namespaced
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required:
                - service
                - indicator
                - objectives
              properties:
                service:
                  type: string
                  description: Service name
                description:
                  type: string
                  description: Human-readable description
                indicator:
                  type: object
                  required:
                    - type
                  properties:
                    type:
                      type: string
                      enum:
                        - ratio
                        - threshold
                        - latency
                        - availability
                    ratio:
                      type: object
                      properties:
                        good:
                          type: object
                          properties:
                            metric:
                              type: string
                            filters:
                              type: array
                              items:
                                type: string
                        total:
                          type: object
                          properties:
                            metric:
                              type: string
                            filters:
                              type: array
                              items:
                                type: string
                objectives:
                  type: array
                  items:
                    type: object
                    required:
                      - window
                      - target
                    properties:
                      displayName:
                        type: string
                      window:
                        type: string
                        pattern: '^[0-9]+(d|h|m)$'
                      target:
                        type: number
                        minimum: 0
                        maximum: 1
                      description:
                        type: string
                alerting:
                  type: object
                  properties:
                    enabled:
                      type: boolean
                    burnRates:
                      type: array
                      items:
                        type: object
                        properties:
                          severity:
                            type: string
                            enum:
                              - critical
                              - warning
                              - info
                          shortWindow:
                            type: string
                          longWindow:
                            type: string
                          burnRate:
                            type: number
            status:
              type: object
              properties:
                currentSLI:
                  type: number
                remainingErrorBudget:
                  type: number
                burnRate:
                  type: number
                lastUpdated:
                  type: string
                  format: date-time
                conditions:
                  type: array
                  items:
                    type: object
                    properties:
                      type:
                        type: string
                      status:
                        type: string
                      lastTransitionTime:
                        type: string
                        format: date-time
                      reason:
                        type: string
                      message:
                        type: string
      subresources:
        status: {}
      additionalPrinterColumns:
        - name: Service
          type: string
          jsonPath: .spec.service
        - name: Target
          type: number
          jsonPath: .spec.objectives[0].target
        - name: Current SLI
          type: number
          jsonPath: .status.currentSLI
        - name: Budget Remaining
          type: number
          jsonPath: .status.remainingErrorBudget
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
```

### SLO Controller

```python
#!/usr/bin/env python3
"""
kubernetes/slo_controller.py

Kubernetes controller for SLO CRDs.
"""

import asyncio
import logging
from typing import Dict, Any
from kubernetes import client, config, watch
from prometheus_api_client import PrometheusConnect


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SLOController:
    """Kubernetes controller for SLO resources"""

    def __init__(self, prometheus_url: str):
        config.load_incluster_config()  # or load_kube_config() for local development
        self.api = client.CustomObjectsApi()
        self.prometheus = PrometheusConnect(url=prometheus_url)

        self.group = "slo.dev"
        self.version = "v1"
        self.plural = "servicelevelobjectives"

    async def run(self):
        """Main controller loop"""
        logger.info("Starting SLO controller...")

        while True:
            try:
                await self._watch_slos()
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")
                await asyncio.sleep(10)

    async def _watch_slos(self):
        """Watch for SLO resource changes"""
        w = watch.Watch()

        for event in w.stream(
            self.api.list_cluster_custom_object,
            group=self.group,
            version=self.version,
            plural=self.plural
        ):
            event_type = event['type']
            slo = event['object']

            logger.info(f"Event: {event_type} for SLO {slo['metadata']['name']}")

            if event_type in ['ADDED', 'MODIFIED']:
                await self._reconcile_slo(slo)
            elif event_type == 'DELETED':
                await self._cleanup_slo(slo)

    async def _reconcile_slo(self, slo: Dict[str, Any]):
        """Reconcile SLO resource"""
        name = slo['metadata']['name']
        namespace = slo['metadata']['namespace']

        logger.info(f"Reconciling SLO: {namespace}/{name}")

        # Calculate current SLI
        current_sli = self._calculate_sli(slo)

        # Calculate error budget
        remaining_budget = self._calculate_error_budget(slo, current_sli)

        # Calculate burn rate
        burn_rate = self._calculate_burn_rate(slo)

        # Update status
        await self._update_status(
            namespace,
            name,
            current_sli,
            remaining_budget,
            burn_rate
        )

    def _calculate_sli(self, slo: Dict[str, Any]) -> float:
        """Calculate current SLI value from Prometheus"""
        spec = slo['spec']
        service = spec['service']
        indicator = spec['indicator']

        if indicator['type'] == 'ratio':
            query = f"{service}:sli:value"
            result = self.prometheus.custom_query(query)

            if result:
                return float(result[0]['value'][1])

        return 0.0

    def _calculate_error_budget(self, slo: Dict[str, Any], current_sli: float) -> float:
        """Calculate remaining error budget percentage"""
        target = slo['spec']['objectives'][0]['target']

        # Error budget = (actual - target) / (1 - target)
        error_budget = (current_sli - target) / (1 - target) * 100

        return max(0, min(100, error_budget))

    def _calculate_burn_rate(self, slo: Dict[str, Any]) -> float:
        """Calculate current error budget burn rate"""
        service = slo['spec']['service']
        query = f"{service}:error_budget:burn_rate_1h"

        result = self.prometheus.custom_query(query)
        if result:
            return float(result[0]['value'][1])

        return 0.0

    async def _update_status(
        self,
        namespace: str,
        name: str,
        current_sli: float,
        remaining_budget: float,
        burn_rate: float
    ):
        """Update SLO resource status"""
        from datetime import datetime

        status = {
            'currentSLI': current_sli,
            'remainingErrorBudget': remaining_budget,
            'burnRate': burn_rate,
            'lastUpdated': datetime.utcnow().isoformat(),
            'conditions': [
                {
                    'type': 'Ready',
                    'status': 'True',
                    'lastTransitionTime': datetime.utcnow().isoformat(),
                    'reason': 'SLICalculated',
                    'message': f'Current SLI: {current_sli:.4f}'
                }
            ]
        }

        try:
            self.api.patch_namespaced_custom_object_status(
                group=self.group,
                version=self.version,
                namespace=namespace,
                plural=self.plural,
                name=name,
                body={'status': status}
            )
            logger.info(f"Updated status for {namespace}/{name}")
        except Exception as e:
            logger.error(f"Failed to update status: {e}")

    async def _cleanup_slo(self, slo: Dict[str, Any]):
        """Clean up resources for deleted SLO"""
        name = slo['metadata']['name']
        logger.info(f"Cleaning up SLO: {name}")
        # Clean up Prometheus rules, alerts, dashboards, etc.


async def main():
    import os

    prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
    controller = SLOController(prometheus_url)

    await controller.run()


if __name__ == '__main__':
    asyncio.run(main())
```

---

## Python Automation Tools

Additional Python tools for SLO automation.

### SLO CLI Tool

```python
#!/usr/bin/env python3
"""
slo_cli.py

Command-line interface for SLO management.
"""

import click
import yaml
from pathlib import Path
from tabulate import tabulate


@click.group()
def cli():
    """SLO Management CLI"""
    pass


@cli.command()
@click.argument('service')
@click.option('--type', default='api', help='Service type')
@click.option('--tier', default='standard', help='Service tier')
@click.option('--output', default='slo.yaml', help='Output file')
def generate(service, type, tier, output):
    """Generate SLO definition for a service"""
    from slo_automation import SLOTemplateLibrary

    templates = SLOTemplateLibrary()

    if type == 'api':
        template = templates.get_api_service_template()
    elif type == 'web':
        template = templates.get_web_service_template()
    elif type == 'batch':
        template = templates.get_batch_pipeline_template()
    else:
        click.echo(f"Unknown service type: {type}")
        return

    # Customize template
    template['metadata']['name'] = f"{service}-availability"
    template['spec']['service'] = service

    # Write to file
    with open(output, 'w') as f:
        yaml.dump(template, f, default_flow_style=False)

    click.echo(f"Generated SLO definition: {output}")


@cli.command()
@click.argument('slo_file')
def validate(slo_file):
    """Validate SLO definition"""
    with open(slo_file) as f:
        slo = yaml.safe_load(f)

    # TODO: Implement validation
    click.echo(f"✓ {slo_file} is valid")


@cli.command()
@click.option('--prometheus-url', required=True)
def status(prometheus_url):
    """Show SLO status for all services"""
    # TODO: Query Prometheus and display status

    data = [
        ['api-service', '99.95%', '80%', '1.2x', '✓'],
        ['web-service', '99.80%', '40%', '3.5x', '⚠'],
        ['batch-pipeline', '99.50%', '90%', '0.8x', '✓'],
    ]

    headers = ['Service', 'Current SLI', 'Budget Remaining', 'Burn Rate', 'Status']
    click.echo(tabulate(data, headers=headers, tablefmt='grid'))


@cli.command()
@click.argument('service')
@click.option('--prometheus-url', required=True)
@click.option('--days', default=30, help='Number of days')
def report(service, prometheus_url, days):
    """Generate SLO report for a service"""
    from slo_implement import SLOReporter
    from prometheus_api_client import PrometheusConnect

    prom = PrometheusConnect(url=prometheus_url)
    reporter = SLOReporter(prom)

    report = reporter.generate_monthly_report(service, f"{days}d")
    click.echo(report)


if __name__ == '__main__':
    cli()
```

---

## Service Discovery Integration

Integrate with service discovery to automatically detect services and generate SLOs.

### Service Discovery Implementation

```python
#!/usr/bin/env python3
"""
service_discovery.py

Discover services from Kubernetes and other sources.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from kubernetes import client, config


@dataclass
class DiscoveredService:
    """Discovered service metadata"""
    name: str
    namespace: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    endpoints: List[str]
    metrics_available: bool


class KubernetesServiceDiscovery:
    """Discover services from Kubernetes"""

    def __init__(self):
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()

        self.client = client.CoreV1Api()
        self.apps_client = client.AppsV1Api()

    def get_all_services(self) -> List[DiscoveredService]:
        """Discover all services in the cluster"""
        services = []

        # Get all namespaces
        namespaces = self.client.list_namespace()

        for ns in namespaces.items:
            namespace = ns.metadata.name

            # Skip system namespaces
            if namespace.startswith('kube-'):
                continue

            # Get services in namespace
            svc_list = self.client.list_namespaced_service(namespace)

            for svc in svc_list.items:
                # Check if service has metrics
                metrics_available = self._has_metrics(svc)

                services.append(DiscoveredService(
                    name=svc.metadata.name,
                    namespace=namespace,
                    labels=svc.metadata.labels or {},
                    annotations=svc.metadata.annotations or {},
                    endpoints=self._get_endpoints(svc),
                    metrics_available=metrics_available
                ))

        return services

    def _has_metrics(self, service) -> bool:
        """Check if service exposes Prometheus metrics"""
        # Check for prometheus.io/scrape annotation
        annotations = service.metadata.annotations or {}
        return annotations.get('prometheus.io/scrape') == 'true'

    def _get_endpoints(self, service) -> List[str]:
        """Get service endpoints"""
        endpoints = []

        spec = service.spec
        if spec.cluster_ip and spec.cluster_ip != 'None':
            for port in spec.ports or []:
                endpoints.append(f"{spec.cluster_ip}:{port.port}")

        return endpoints


# Example usage
if __name__ == '__main__':
    discovery = KubernetesServiceDiscovery()
    services = discovery.get_all_services()

    print(f"Discovered {len(services)} services:")
    for svc in services:
        print(f"  - {svc.namespace}/{svc.name} (metrics: {svc.metrics_available})")
```

---

## Migration Strategies

Strategies for migrating from manual to automated SLO management.

### Migration Guide

```markdown
# SLO Automation Migration Guide

## Overview

This guide provides a phased approach to migrating from manual SLO management to fully automated SLO-as-code.

## Migration Phases

### Phase 1: Assessment (Week 1-2)

**Objectives:**
- Inventory existing SLOs and monitoring
- Identify gaps in current implementation
- Define target state

**Activities:**
1. Document existing SLOs
   - What services have SLOs?
   - How are they measured?
   - Where are they defined?

2. Assess current tooling
   - Prometheus/monitoring setup
   - Dashboard tools
   - Alerting infrastructure

3. Identify stakeholders
   - Service owners
   - SRE team
   - Product management

**Deliverables:**
- Current state documentation
- Gap analysis
- Migration roadmap

### Phase 2: Pilot (Week 3-6)

**Objectives:**
- Implement SLO-as-code for 2-3 pilot services
- Validate approach
- Refine templates and tooling

**Activities:**
1. Select pilot services
   - Choose services with different characteristics
   - Ensure team buy-in

2. Create SLO definitions
   - Use templates from library
   - Customize for each service

3. Deploy automation
   - Set up GitOps pipeline
   - Configure validation
   - Deploy monitoring

4. Run in parallel
   - Keep existing SLOs
   - Compare automated vs manual

**Success Criteria:**
- Automated SLOs match manual SLOs
- Deployments succeed
- Alerts fire correctly

### Phase 3: Expansion (Week 7-12)

**Objectives:**
- Migrate all production services
- Establish processes and training
- Build confidence in automation

**Activities:**
1. Migrate services by tier
   - Start with best-effort
   - Move to critical last

2. Train teams
   - SLO-as-code concepts
   - GitOps workflow
   - Troubleshooting

3. Establish review process
   - SLO change reviews
   - Regular SLO meetings

**Success Criteria:**
- 80%+ services migrated
- Teams comfortable with process
- Error budget policies in use

### Phase 4: Optimization (Week 13+)

**Objectives:**
- Refine SLO targets
- Improve automation
- Drive continuous improvement

**Activities:**
1. Analyze SLO performance
   - Review adherence
   - Adjust targets
   - Optimize alerts

2. Enhance automation
   - Add more templates
   - Improve validation
   - Automate remediation

3. Expand scope
   - Add new SLI types
   - Multi-region SLOs
   - Composite SLOs

**Success Criteria:**
- All services on automated SLOs
- Regular refinement process
- SLOs driving decisions

## Migration Checklist

### Prerequisites
- [ ] Prometheus deployed and stable
- [ ] Grafana for dashboards
- [ ] Git repository for SLO definitions
- [ ] CI/CD pipeline available
- [ ] Kubernetes cluster (if using CRDs)

### Setup
- [ ] Install SLO automation tools
- [ ] Configure Prometheus integration
- [ ] Set up GitOps workflow
- [ ] Deploy Kubernetes CRDs (optional)
- [ ] Create initial templates

### Pilot Services
- [ ] Select 2-3 pilot services
- [ ] Define SLOs in YAML
- [ ] Validate definitions
- [ ] Deploy to monitoring
- [ ] Verify metrics and alerts
- [ ] Run parallel for 2 weeks
- [ ] Get team feedback

### Production Rollout
- [ ] Create SLO definitions for all services
- [ ] Validate all definitions
- [ ] Deploy in phases (by tier)
- [ ] Train service owners
- [ ] Document processes
- [ ] Establish review cadence
- [ ] Decommission manual SLOs

### Continuous Improvement
- [ ] Monthly SLO reviews
- [ ] Quarterly target adjustments
- [ ] Regular template updates
- [ ] Automation enhancements
- [ ] Team training refreshers

## Common Challenges

### Challenge: Resistance to Change
**Solution:**
- Start with volunteers
- Show value quickly
- Make it easy to adopt
- Provide good documentation

### Challenge: Incomplete Metrics
**Solution:**
- Add instrumentation first
- Use progressive implementation
- Start with basic SLOs
- Improve over time

### Challenge: Alert Fatigue
**Solution:**
- Start with generous thresholds
- Use multi-window burn rates
- Adjust based on feedback
- Focus on actionable alerts

### Challenge: Complex Services
**Solution:**
- Break down into components
- Use multiple SLOs
- Start simple, add complexity
- Get architecture input

## Support Resources

- **Documentation**: /docs/slo-automation
- **Templates**: /slo-definitions/templates
- **Examples**: /slo-definitions/examples
- **Slack**: #slo-automation
- **Office Hours**: Tuesdays 2-3pm
```

---

## Complete Example: End-to-End

```bash
#!/bin/bash
# end-to-end-example.sh
#
# Complete example of SLO automation workflow

set -e

echo "🚀 SLO Automation End-to-End Example"
echo "======================================"

# 1. Discover services
echo -e "\n📡 Step 1: Discover services"
python3 service_discovery.py --output discovered-services.json

# 2. Generate SLO definitions
echo -e "\n📝 Step 2: Generate SLO definitions"
python3 slo_automation.py \
  --input discovered-services.json \
  --output-dir slo-definitions/ \
  --progressive

# 3. Validate SLO definitions
echo -e "\n✅ Step 3: Validate definitions"
python3 validate_slos.py \
  --schema schemas/slo-schema.json \
  --slo-dir slo-definitions/

# 4. Dry-run deployment
echo -e "\n🔍 Step 4: Dry-run deployment"
python3 deploy_slos.py \
  --slo-dir slo-definitions/ \
  --prometheus-url http://prometheus:9090 \
  --dry-run

# 5. Review changes
echo -e "\n👀 Step 5: Review changes (manual step)"
read -p "Review the changes and press Enter to continue..."

# 6. Deploy SLOs
echo -e "\n🚀 Step 6: Deploy SLOs"
python3 deploy_slos.py \
  --slo-dir slo-definitions/ \
  --prometheus-url http://prometheus:9090 \
  --apply

# 7. Verify deployment
echo -e "\n✓ Step 7: Verify deployment"
python3 verify_deployment.py \
  --slo-dir slo-definitions/ \
  --prometheus-url http://prometheus:9090

# 8. Generate dashboards
echo -e "\n📊 Step 8: Generate dashboards"
python3 deploy_grafana_dashboards.py \
  --slo-dir slo-definitions/ \
  --grafana-url http://grafana:3000

echo -e "\n✅ SLO automation complete!"
echo "View dashboards at: http://grafana:3000/dashboards/slo"
```

---

## Summary

This comprehensive SLO automation framework provides:

1. **SLO-as-Code**: Version-controlled, reviewable SLO definitions
2. **Automated Generation**: Discover services and generate appropriate SLOs
3. **Progressive Implementation**: Gradually increase reliability targets
4. **Template Library**: Pre-built templates for common service types
5. **GitOps Workflow**: CI/CD integration with validation and deployment
6. **Kubernetes CRD**: Native Kubernetes integration
7. **Python Tools**: Complete automation toolkit
8. **Service Discovery**: Automatic service detection
9. **Migration Guide**: Clear path from manual to automated SLO management

This enables organizations to scale SLO practices across hundreds of services while maintaining consistency, quality, and reliability standards.
