# SLO Automation Implementation
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

