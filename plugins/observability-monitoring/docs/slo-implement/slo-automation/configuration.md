# SLO Configuration & Schema
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

