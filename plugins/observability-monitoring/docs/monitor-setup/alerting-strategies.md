# Alerting Strategies: Complete Guide to Alertmanager and Alert Management

## Table of Contents

1. [Alertmanager Architecture](#alertmanager-architecture)
2. [Alert Routing Strategies](#alert-routing-strategies)
3. [Grouping and Inhibition Rules](#grouping-and-inhibition-rules)
4. [Notification Channels](#notification-channels)
5. [Alert Rule Design Patterns](#alert-rule-design-patterns)
6. [Multi-Window Multi-Burn-Rate Alerting](#multi-window-multi-burn-rate-alerting)
7. [Runbook Automation](#runbook-automation)
8. [Alert Fatigue Prevention](#alert-fatigue-prevention)
9. [SLO-Based Alerting](#slo-based-alerting)
10. [Incident Escalation and On-Call](#incident-escalation-and-on-call)
11. [Alert Testing and Validation](#alert-testing-and-validation)

---

## Alertmanager Architecture

### Core Concepts

Alertmanager handles alerts sent by client applications such as Prometheus. It takes care of:

- **Deduplication**: Grouping identical alerts
- **Grouping**: Batching related alerts together
- **Routing**: Sending alerts to the correct receiver
- **Silencing**: Temporarily muting alerts
- **Inhibition**: Suppressing alerts based on other active alerts

### Architecture Overview

```yaml
# alertmanager.yml - Complete configuration
global:
  # Global settings
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alertmanager@example.com'
  smtp_auth_username: 'alertmanager@example.com'
  smtp_auth_password: 'your-password'
  smtp_require_tls: true
  http_config:
    follow_redirects: true
    enable_http2: true

# Template files for custom notifications
templates:
  - '/etc/alertmanager/templates/*.tmpl'

# Alert routing tree
route:
  receiver: 'default-receiver'
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h

  routes:
    # Critical production alerts
    - match:
        severity: critical
        environment: production
      receiver: 'pagerduty-critical'
      group_wait: 10s
      group_interval: 5m
      repeat_interval: 4h
      continue: true

    # Database alerts
    - match_re:
        service: '(postgres|mysql|mongodb|redis)'
      receiver: 'database-team'
      group_by: ['alertname', 'instance', 'database']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 3h
      routes:
        - match:
            severity: critical
          receiver: 'database-oncall'
          group_wait: 10s
          repeat_interval: 30m

    # Application alerts
    - match_re:
        service: '(api|web|worker)'
      receiver: 'app-team-slack'
      group_by: ['alertname', 'service', 'instance']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 4h
      routes:
        - match:
            severity: critical
          receiver: 'app-oncall-pagerduty'
        - match:
            severity: warning
          receiver: 'app-team-slack'

    # Infrastructure alerts
    - match:
        team: infrastructure
      receiver: 'infra-team'
      group_by: ['alertname', 'cluster', 'instance']
      routes:
        - match:
            severity: critical
          receiver: 'infra-oncall'
          repeat_interval: 30m
        - match:
            severity: warning
          receiver: 'infra-slack'
          repeat_interval: 6h

    # Security alerts
    - match:
        team: security
      receiver: 'security-team'
      group_wait: 5s
      group_interval: 1m
      repeat_interval: 1h
      continue: false

    # SLO burn rate alerts
    - match:
        alert_type: slo_burn_rate
      receiver: 'slo-alerts'
      group_by: ['alertname', 'service', 'slo']
      group_wait: 5s
      group_interval: 2m
      repeat_interval: 1h
      routes:
        - match:
            severity: critical
          receiver: 'slo-critical-oncall'
        - match:
            severity: warning
          receiver: 'slo-warning-slack'

# Inhibition rules
inhibit_rules:
  # Inhibit warning if critical is firing
  - source_match:
      severity: critical
    target_match:
      severity: warning
    equal: ['alertname', 'instance', 'service']

  # Inhibit instance alerts if cluster is down
  - source_match:
      alertname: ClusterDown
    target_match_re:
      alertname: '(InstanceDown|HighCPU|HighMemory)'
    equal: ['cluster']

  # Inhibit downstream service alerts if upstream is down
  - source_match:
      alertname: DatabaseDown
    target_match_re:
      alertname: '(APISlowResponse|HighErrorRate)'
    equal: ['environment', 'cluster']

  # Inhibit individual disk alerts if node is down
  - source_match:
      alertname: NodeDown
    target_match_re:
      alertname: '(DiskSpaceLow|DiskIOHigh)'
    equal: ['instance']

# Receivers configuration
receivers:
  # Default receiver
  - name: 'default-receiver'
    slack_configs:
      - channel: '#alerts-general'
        title: 'Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
        send_resolved: true

  # PagerDuty for critical alerts
  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        severity: '{{ .CommonLabels.severity }}'
        description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'
        details:
          firing: '{{ .Alerts.Firing | len }}'
          resolved: '{{ .Alerts.Resolved | len }}'
          details: '{{ .CommonAnnotations.description }}'
        client: 'Alertmanager'
        client_url: '{{ .ExternalURL }}'

  # Database team notifications
  - name: 'database-team'
    slack_configs:
      - channel: '#database-alerts'
        title: 'Database Alert: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Labels.alertname }}
          *Severity:* {{ .Labels.severity }}
          *Instance:* {{ .Labels.instance }}
          *Database:* {{ .Labels.database }}
          *Description:* {{ .Annotations.description }}
          {{ end }}
        send_resolved: true
    email_configs:
      - to: 'database-team@example.com'
        headers:
          Subject: '[{{ .Status | toUpper }}] Database Alert: {{ .GroupLabels.alertname }}'

  - name: 'database-oncall'
    pagerduty_configs:
      - service_key: 'DATABASE_ONCALL_KEY'
        severity: 'critical'
    slack_configs:
      - channel: '#database-critical'
        title: 'CRITICAL Database Alert'
        send_resolved: true

  # Application team notifications
  - name: 'app-team-slack'
    slack_configs:
      - channel: '#app-alerts'
        title: 'Application Alert: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Service:* {{ .Labels.service }}
          *Severity:* {{ .Labels.severity }}
          *Summary:* {{ .Annotations.summary }}
          *Runbook:* {{ .Annotations.runbook_url }}
          {{ end }}
        send_resolved: true
        actions:
          - type: button
            text: 'View Runbook'
            url: '{{ (index .Alerts 0).Annotations.runbook_url }}'
          - type: button
            text: 'View Dashboard'
            url: '{{ (index .Alerts 0).Annotations.dashboard_url }}'

  - name: 'app-oncall-pagerduty'
    pagerduty_configs:
      - service_key: 'APP_ONCALL_KEY'
        severity: '{{ .CommonLabels.severity }}'

  # Infrastructure team
  - name: 'infra-team'
    webhook_configs:
      - url: 'http://webhook-receiver:8080/alerts'
        send_resolved: true
        http_config:
          basic_auth:
            username: 'webhook-user'
            password: 'webhook-pass'

  - name: 'infra-oncall'
    pagerduty_configs:
      - service_key: 'INFRA_ONCALL_KEY'
    slack_configs:
      - channel: '#infra-critical'

  - name: 'infra-slack'
    slack_configs:
      - channel: '#infra-alerts'

  # Security team
  - name: 'security-team'
    pagerduty_configs:
      - service_key: 'SECURITY_ONCALL_KEY'
        severity: 'critical'
    slack_configs:
      - channel: '#security-alerts'
        title: 'SECURITY ALERT: {{ .GroupLabels.alertname }}'
        send_resolved: true
    email_configs:
      - to: 'security@example.com'
        headers:
          Subject: '[SECURITY] {{ .Status | toUpper }}: {{ .GroupLabels.alertname }}'
        html: '{{ template "email.default.html" . }}'

  # SLO alerts
  - name: 'slo-alerts'
    slack_configs:
      - channel: '#slo-alerts'
        title: 'SLO Burn Rate Alert'
        text: |
          {{ range .Alerts }}
          *Service:* {{ .Labels.service }}
          *SLO:* {{ .Labels.slo }}
          *Burn Rate:* {{ .Labels.burn_rate }}
          *Budget Remaining:* {{ .Annotations.budget_remaining }}
          {{ end }}

  - name: 'slo-critical-oncall'
    pagerduty_configs:
      - service_key: 'SLO_ONCALL_KEY'

  - name: 'slo-warning-slack'
    slack_configs:
      - channel: '#slo-warnings'
```

### High Availability Setup

```yaml
# alertmanager-ha.yml
# Run multiple Alertmanager instances with clustering

global:
  resolve_timeout: 5m

# Cluster configuration for HA
cluster:
  listen-address: '0.0.0.0:9094'
  peers:
    - 'alertmanager-1:9094'
    - 'alertmanager-2:9094'
    - 'alertmanager-3:9094'
  peer-timeout: 15s
  gossip-interval: 200ms
  push-pull-interval: 1m

# Data persistence
data:
  retention: 120h

# API configuration
api:
  v2:
    timeout: 30s
```

---

## Alert Routing Strategies

### Hierarchical Routing

```yaml
route:
  receiver: 'default'
  group_by: ['alertname']

  routes:
    # Environment-based routing
    - match:
        environment: production
      receiver: 'prod-team'
      group_by: ['alertname', 'service']

      routes:
        # Severity-based sub-routing
        - match:
            severity: critical
          receiver: 'prod-oncall'
          group_wait: 5s
          repeat_interval: 15m

        - match:
            severity: warning
          receiver: 'prod-slack'
          group_wait: 30s
          repeat_interval: 4h

    - match:
        environment: staging
      receiver: 'staging-team'
      repeat_interval: 12h
```

### Team-Based Routing

```yaml
routes:
  # Platform team
  - match_re:
      team: 'platform'
    receiver: 'platform-team'
    group_by: ['alertname', 'cluster']
    routes:
      - match_re:
          service: '(kubernetes|etcd|calico)'
        receiver: 'platform-k8s-team'
      - match_re:
          service: '(prometheus|grafana|alertmanager)'
        receiver: 'platform-monitoring-team'

  # Data team
  - match_re:
      team: 'data'
    receiver: 'data-team'
    group_by: ['alertname', 'pipeline']
    routes:
      - match:
          severity: critical
        receiver: 'data-oncall'
```

### Service Mesh Routing

```yaml
routes:
  # Istio/Service Mesh alerts
  - match:
      service_mesh: istio
    receiver: 'service-mesh-team'
    group_by: ['alertname', 'namespace', 'service']
    routes:
      - match_re:
          alertname: '(HighRequestLatency|HighErrorRate)'
        receiver: 'app-owners'
        group_by: ['service', 'namespace']
      - match_re:
          alertname: '(CircuitBreakerOpen|ControlPlaneDown)'
        receiver: 'platform-oncall'
```

---

## Grouping and Inhibition Rules

### Advanced Grouping Strategies

```yaml
# Dynamic grouping based on labels
route:
  # Group by service and instance for most alerts
  group_by: ['alertname', 'service', 'instance']

  routes:
    # Group cluster-wide alerts by cluster only
    - match_re:
        scope: 'cluster'
      group_by: ['alertname', 'cluster']

    # Group multi-instance alerts differently
    - match_re:
        alertname: '(HighCPUAcrossCluster|HighMemoryAcrossCluster)'
      group_by: ['alertname', 'cluster']
      group_wait: 2m  # Wait longer to batch instances

    # Don't group security alerts
    - match:
        team: security
      group_by: ['alertname']  # Each alert separately
```

### Comprehensive Inhibition Rules

```yaml
inhibit_rules:
  # Node-level inhibitions
  - source_match:
      alertname: NodeDown
    target_match_re:
      alertname: '(HighCPU|HighMemory|DiskSpaceLow|NetworkErrors)'
    equal: ['instance', 'cluster']

  # Service-level inhibitions
  - source_match:
      alertname: ServiceDown
    target_match_re:
      alertname: '(HighLatency|HighErrorRate|LowThroughput)'
    equal: ['service', 'environment']

  # Database-specific inhibitions
  - source_match:
      alertname: PostgresDown
    target_match_re:
      alertname: '(PostgresReplicationLag|PostgresSlowQueries|PostgresConnectionsHigh)'
    equal: ['instance', 'database']

  # Cascade inhibitions (upstream affects downstream)
  - source_match:
      alertname: LoadBalancerDown
    target_match_re:
      alertname: '(BackendUnhealthy|HighResponseTime)'
    equal: ['cluster', 'environment']

  # Maintenance window inhibitions
  - source_match:
      alertname: MaintenanceMode
    target_match_re:
      severity: '(warning|info)'
    equal: ['service', 'environment']

  # Severity-based inhibitions
  - source_match:
      severity: critical
    target_match:
      severity: warning
    equal: ['alertname', 'service', 'instance']

  - source_match:
      severity: warning
    target_match:
      severity: info
    equal: ['alertname', 'service', 'instance']

  # Network partitioning inhibitions
  - source_match:
      alertname: NetworkPartition
    target_match_re:
      alertname: '(NodeUnreachable|ServiceTimeout|ConnectionRefused)'
    equal: ['cluster', 'zone']
```

---

## Notification Channels

### Slack Configuration

```yaml
# Advanced Slack notifications
receivers:
  - name: 'slack-production'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK'
        channel: '#prod-alerts'
        username: 'Alertmanager'
        icon_emoji: ':warning:'
        title: |
          [{{ .Status | toUpper }}{{ if eq .Status "firing" }}:{{ .Alerts.Firing | len }}{{ end }}] {{ .GroupLabels.alertname }}
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Labels.alertname }}
          *Severity:* {{ .Labels.severity }}
          *Environment:* {{ .Labels.environment }}
          *Service:* {{ .Labels.service }}
          *Instance:* {{ .Labels.instance }}

          *Summary:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}

          *Started:* {{ .StartsAt | since }}
          {{ if .EndsAt }}*Ended:* {{ .EndsAt | since }}{{ end }}

          *Labels:*
          {{ range .Labels.SortedPairs }} - {{ .Name }}: {{ .Value }}
          {{ end }}
          {{ end }}
        send_resolved: true
        actions:
          - type: button
            text: 'View Runbook :book:'
            url: '{{ (index .Alerts 0).Annotations.runbook_url }}'
            style: 'primary'
          - type: button
            text: 'View Dashboard :chart_with_upwards_trend:'
            url: '{{ (index .Alerts 0).Annotations.dashboard_url }}'
          - type: button
            text: 'Silence :mute:'
            url: '{{ .ExternalURL }}/#/silences/new?filter=%7B{{ range .CommonLabels.SortedPairs }}{{ .Name }}%3D"{{ .Value }}"{{ end }}%7D'
            style: 'danger'

        # Color coding
        color: |
          {{ if eq .Status "firing" }}
            {{ if eq .CommonLabels.severity "critical" }}danger{{ else if eq .CommonLabels.severity "warning" }}warning{{ else }}#439FE0{{ end }}
          {{ else }}good{{ end }}

        # Footer
        footer: 'Alertmanager'
        footer_icon: 'https://example.com/alertmanager-icon.png'
```

### PagerDuty Configuration

```yaml
receivers:
  - name: 'pagerduty-production'
    pagerduty_configs:
      - routing_key: 'YOUR_PAGERDUTY_INTEGRATION_KEY'
        severity: '{{ .CommonLabels.severity }}'
        description: |
          {{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}
        details:
          firing: '{{ .Alerts.Firing | len }}'
          resolved: '{{ .Alerts.Resolved | len }}'
          num_alerts: '{{ .Alerts | len }}'
          group_labels: '{{ .GroupLabels.SortedPairs.Values }}'
          common_labels: '{{ .CommonLabels.SortedPairs.Values }}'
          common_annotations: '{{ .CommonAnnotations.SortedPairs.Values }}'
          description: '{{ .CommonAnnotations.description }}'
          runbook: '{{ .CommonAnnotations.runbook_url }}'
          dashboard: '{{ .CommonAnnotations.dashboard_url }}'
        client: 'Prometheus Alertmanager'
        client_url: '{{ .ExternalURL }}'
        send_resolved: true

        # Custom links
        links:
          - href: '{{ .CommonAnnotations.runbook_url }}'
            text: 'Runbook'
          - href: '{{ .CommonAnnotations.dashboard_url }}'
            text: 'Dashboard'
          - href: '{{ .CommonAnnotations.logs_url }}'
            text: 'Logs'
```

### Email Configuration

```yaml
receivers:
  - name: 'email-oncall'
    email_configs:
      - to: 'oncall@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'alertmanager@example.com'
        auth_identity: 'alertmanager@example.com'
        auth_password: 'your-app-password'
        require_tls: true

        headers:
          Subject: '[{{ .Status | toUpper }}] {{ .GroupLabels.alertname }} ({{ .Alerts.Firing | len }} firing)'
          From: 'Alertmanager <alertmanager@example.com>'
          Reply-To: 'devops@example.com'

        html: |
          <!DOCTYPE html>
          <html>
          <head>
            <style>
              body { font-family: Arial, sans-serif; }
              .alert { border: 1px solid #ddd; margin: 10px; padding: 10px; }
              .critical { background-color: #ffebee; border-color: #f44336; }
              .warning { background-color: #fff3e0; border-color: #ff9800; }
              .label { font-weight: bold; }
            </style>
          </head>
          <body>
            <h2>Alert: {{ .GroupLabels.alertname }}</h2>
            <p>Status: <strong>{{ .Status | toUpper }}</strong></p>
            <p>Firing Alerts: {{ .Alerts.Firing | len }}</p>
            <p>Resolved Alerts: {{ .Alerts.Resolved | len }}</p>

            {{ range .Alerts }}
            <div class="alert {{ .Labels.severity }}">
              <h3>{{ .Labels.alertname }}</h3>
              <p><span class="label">Severity:</span> {{ .Labels.severity }}</p>
              <p><span class="label">Service:</span> {{ .Labels.service }}</p>
              <p><span class="label">Instance:</span> {{ .Labels.instance }}</p>
              <p><span class="label">Summary:</span> {{ .Annotations.summary }}</p>
              <p><span class="label">Description:</span> {{ .Annotations.description }}</p>
              <p><span class="label">Runbook:</span> <a href="{{ .Annotations.runbook_url }}">{{ .Annotations.runbook_url }}</a></p>
              <p><span class="label">Started:</span> {{ .StartsAt }}</p>
              {{ if .EndsAt }}<p><span class="label">Ended:</span> {{ .EndsAt }}</p>{{ end }}
            </div>
            {{ end }}
          </body>
          </html>

        send_resolved: true
```

### Webhook Configuration

```yaml
receivers:
  - name: 'webhook-receiver'
    webhook_configs:
      - url: 'http://webhook-service:8080/alerts'
        send_resolved: true
        http_config:
          basic_auth:
            username: 'webhook-user'
            password: 'webhook-password'
          tls_config:
            insecure_skip_verify: false
            ca_file: '/etc/ssl/certs/ca.pem'
        max_alerts: 0  # Send all alerts

      # Microsoft Teams webhook
      - url: 'https://outlook.office.com/webhook/YOUR_TEAMS_WEBHOOK'
        send_resolved: true
        http_config:
          follow_redirects: true

      # Custom incident management system
      - url: 'https://incident-manager.example.com/api/v1/alerts'
        send_resolved: true
        http_config:
          bearer_token: 'YOUR_API_TOKEN'
          tls_config:
            server_name: 'incident-manager.example.com'
```

---

## Alert Rule Design Patterns

### Symptom-Based Alerts

```yaml
# prometheus-alerts.yml
groups:
  - name: symptom-based-alerts
    interval: 30s
    rules:
      # User-facing symptoms
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
            /
            sum(rate(http_requests_total[5m])) by (service)
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          team: platform
          alert_type: symptom
        annotations:
          summary: "High error rate on {{ $labels.service }}"
          description: "{{ $labels.service }} is experiencing {{ $value | humanizePercentage }} error rate (threshold: 5%)"
          runbook_url: "https://runbooks.example.com/high-error-rate"
          dashboard_url: "https://grafana.example.com/d/service-overview?var-service={{ $labels.service }}"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
          ) > 1.0
        for: 5m
        labels:
          severity: warning
          team: platform
          alert_type: symptom
        annotations:
          summary: "High latency on {{ $labels.service }}"
          description: "{{ $labels.service }} 95th percentile latency is {{ $value }}s (threshold: 1s)"
          runbook_url: "https://runbooks.example.com/high-latency"

      - alert: LowThroughput
        expr: |
          sum(rate(http_requests_total[5m])) by (service) < 10
        for: 15m
        labels:
          severity: warning
          team: platform
          alert_type: symptom
        annotations:
          summary: "Low throughput on {{ $labels.service }}"
          description: "{{ $labels.service }} is receiving only {{ $value }} req/s (expected: >10 req/s)"
```

### Resource-Based Alerts

```yaml
groups:
  - name: resource-alerts
    interval: 30s
    rules:
      - alert: HighCPUUsage
        expr: |
          100 - (avg by (instance, service) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 10m
        labels:
          severity: warning
          team: infrastructure
          alert_type: resource
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage is {{ $value | humanize }}% on {{ $labels.instance }} (service: {{ $labels.service }})"
          runbook_url: "https://runbooks.example.com/high-cpu"

      - alert: HighMemoryUsage
        expr: |
          (
            node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes
          ) / node_memory_MemTotal_bytes * 100 > 85
        for: 10m
        labels:
          severity: warning
          team: infrastructure
          alert_type: resource
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage is {{ $value | humanize }}% on {{ $labels.instance }}"

      - alert: DiskSpaceLow
        expr: |
          (
            node_filesystem_avail_bytes{fstype!~"tmpfs|fuse.*"}
            /
            node_filesystem_size_bytes{fstype!~"tmpfs|fuse.*"}
          ) * 100 < 15
        for: 5m
        labels:
          severity: critical
          team: infrastructure
          alert_type: resource
        annotations:
          summary: "Disk space low on {{ $labels.instance }}"
          description: "Disk {{ $labels.mountpoint }} on {{ $labels.instance }} has only {{ $value | humanize }}% free space"
          runbook_url: "https://runbooks.example.com/disk-space-low"
```

### Application-Specific Alerts

```yaml
groups:
  - name: application-alerts
    interval: 30s
    rules:
      # Database alerts
      - alert: PostgresReplicationLag
        expr: |
          pg_replication_lag_seconds > 300
        for: 5m
        labels:
          severity: critical
          team: database
          service: postgres
        annotations:
          summary: "PostgreSQL replication lag on {{ $labels.instance }}"
          description: "Replication lag is {{ $value }}s on {{ $labels.instance }} (database: {{ $labels.database }})"
          runbook_url: "https://runbooks.example.com/postgres-replication-lag"

      - alert: PostgresConnectionsHigh
        expr: |
          sum(pg_stat_activity_count) by (instance) / sum(pg_settings_max_connections) by (instance) > 0.8
        for: 5m
        labels:
          severity: warning
          team: database
          service: postgres
        annotations:
          summary: "High connection count on PostgreSQL {{ $labels.instance }}"
          description: "Connection usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}"

      # Queue alerts
      - alert: QueueDepthHigh
        expr: |
          rabbitmq_queue_messages_ready > 10000
        for: 10m
        labels:
          severity: warning
          team: platform
          service: rabbitmq
        annotations:
          summary: "High queue depth in {{ $labels.queue }}"
          description: "Queue {{ $labels.queue }} has {{ $value }} messages ready (threshold: 10000)"
          runbook_url: "https://runbooks.example.com/queue-depth-high"

      # Cache alerts
      - alert: RedisMemoryHigh
        expr: |
          redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
          team: platform
          service: redis
        annotations:
          summary: "Redis memory usage high on {{ $labels.instance }}"
          description: "Memory usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}"

      - alert: RedisCacheHitRateLow
        expr: |
          rate(redis_keyspace_hits_total[5m]) / (rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m])) < 0.8
        for: 15m
        labels:
          severity: warning
          team: platform
          service: redis
        annotations:
          summary: "Low cache hit rate on {{ $labels.instance }}"
          description: "Cache hit rate is {{ $value | humanizePercentage }} on {{ $labels.instance }} (expected: >80%)"
```

---

## Multi-Window Multi-Burn-Rate Alerting

### SLO Burn Rate Concepts

Multi-window multi-burn-rate alerting is based on Google's SRE Workbook methodology for detecting SLO violations at different time scales.

```yaml
# SLO configuration
groups:
  - name: slo-burn-rate-alerts
    interval: 30s
    rules:
      # Fast burn (1 hour to exhaust 30-day error budget)
      - alert: SLOBurnRateCritical
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[1h])) by (service)
            /
            sum(rate(http_requests_total[1h])) by (service)
          ) > (14.4 * 0.001)  # 14.4x burn rate for 99.9% SLO
          and
          (
            sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
            /
            sum(rate(http_requests_total[5m])) by (service)
          ) > (14.4 * 0.001)
        for: 2m
        labels:
          severity: critical
          team: platform
          alert_type: slo_burn_rate
          window: fast
          burn_rate: critical
        annotations:
          summary: "Critical SLO burn rate on {{ $labels.service }}"
          description: |
            {{ $labels.service }} is burning through error budget at 14.4x the acceptable rate.
            At this rate, the entire 30-day error budget will be exhausted in 2 hours.
            Current error rate: {{ $value | humanizePercentage }}
          runbook_url: "https://runbooks.example.com/slo-burn-rate"
          dashboard_url: "https://grafana.example.com/d/slo-dashboard?var-service={{ $labels.service }}"

      # Medium burn (6 hours to exhaust error budget)
      - alert: SLOBurnRateHigh
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[6h])) by (service)
            /
            sum(rate(http_requests_total[6h])) by (service)
          ) > (6 * 0.001)  # 6x burn rate
          and
          (
            sum(rate(http_requests_total{status=~"5.."}[30m])) by (service)
            /
            sum(rate(http_requests_total[30m])) by (service)
          ) > (6 * 0.001)
        for: 15m
        labels:
          severity: warning
          team: platform
          alert_type: slo_burn_rate
          window: medium
          burn_rate: high
        annotations:
          summary: "High SLO burn rate on {{ $labels.service }}"
          description: |
            {{ $labels.service }} is burning through error budget at 6x the acceptable rate.
            At this rate, the entire 30-day error budget will be exhausted in 5 days.
            Current error rate: {{ $value | humanizePercentage }}

      # Slow burn (3 days to exhaust error budget)
      - alert: SLOBurnRateMedium
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[1d])) by (service)
            /
            sum(rate(http_requests_total[1d])) by (service)
          ) > (3 * 0.001)  # 3x burn rate
          and
          (
            sum(rate(http_requests_total{status=~"5.."}[2h])) by (service)
            /
            sum(rate(http_requests_total[2h])) by (service)
          ) > (3 * 0.001)
        for: 1h
        labels:
          severity: warning
          team: platform
          alert_type: slo_burn_rate
          window: slow
          burn_rate: medium
        annotations:
          summary: "Medium SLO burn rate on {{ $labels.service }}"
          description: |
            {{ $labels.service }} is burning through error budget at 3x the acceptable rate.
            At this rate, the entire 30-day error budget will be exhausted in 10 days.
```

### Multi-Window SLO Implementation

```yaml
groups:
  - name: availability-slo
    interval: 30s
    rules:
      # Recording rules for SLO calculations
      - record: service:slo_errors:rate5m
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)

      - record: service:slo_requests:rate5m
        expr: |
          sum(rate(http_requests_total[5m])) by (service)

      - record: service:slo_error_ratio:rate5m
        expr: |
          service:slo_errors:rate5m / service:slo_requests:rate5m

      # Multi-window alerts for 99.9% SLO (0.1% error budget)

      # Page-worthy: 2% budget consumed in 1 hour
      - alert: ErrorBudgetBurn1h
        expr: |
          (
            service:slo_error_ratio:rate5m > (14.4 * 0.001)
            and
            sum(rate(http_requests_total{status=~"5.."}[1h])) by (service)
            /
            sum(rate(http_requests_total[1h])) by (service)
            > (14.4 * 0.001)
          )
        for: 2m
        labels:
          severity: critical
          slo: availability
          window: 1h
          burn_rate: "14.4"
        annotations:
          summary: "2% error budget consumed in 1 hour ({{ $labels.service }})"
          description: "At current burn rate, error budget will be exhausted in 2.08 hours"

      # Page-worthy: 5% budget consumed in 6 hours
      - alert: ErrorBudgetBurn6h
        expr: |
          (
            service:slo_error_ratio:rate5m > (6 * 0.001)
            and
            sum(rate(http_requests_total{status=~"5.."}[6h])) by (service)
            /
            sum(rate(http_requests_total[6h])) by (service)
            > (6 * 0.001)
          )
        for: 15m
        labels:
          severity: critical
          slo: availability
          window: 6h
          burn_rate: "6"
        annotations:
          summary: "5% error budget consumed in 6 hours ({{ $labels.service }})"
          description: "At current burn rate, error budget will be exhausted in 5 days"

      # Ticket-worthy: 10% budget consumed in 3 days
      - alert: ErrorBudgetBurn3d
        expr: |
          (
            service:slo_error_ratio:rate5m > (1 * 0.001)
            and
            sum(rate(http_requests_total{status=~"5.."}[3d])) by (service)
            /
            sum(rate(http_requests_total[3d])) by (service)
            > (1 * 0.001)
          )
        for: 1h
        labels:
          severity: warning
          slo: availability
          window: 3d
          burn_rate: "1"
        annotations:
          summary: "10% error budget consumed in 3 days ({{ $labels.service }})"
          description: "At current burn rate, error budget will be exhausted in 30 days"
```

### Latency SLO Burn Rate

```yaml
groups:
  - name: latency-slo-burn-rate
    interval: 30s
    rules:
      # Recording rules for latency SLO (95th percentile < 500ms)
      - record: service:slo_latency:p95_5m
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
          )

      # Fast burn rate alert
      - alert: LatencySLOBurnRateCritical
        expr: |
          (
            service:slo_latency:p95_5m > 0.5  # 500ms threshold
            and
            histogram_quantile(0.95,
              sum(rate(http_request_duration_seconds_bucket[1h])) by (le, service)
            ) > 0.5
          )
        for: 5m
        labels:
          severity: critical
          slo: latency
          burn_rate: critical
        annotations:
          summary: "Critical latency SLO burn rate ({{ $labels.service }})"
          description: "95th percentile latency is {{ $value }}s (threshold: 0.5s)"

      # Medium burn rate alert
      - alert: LatencySLOBurnRateHigh
        expr: |
          (
            service:slo_latency:p95_5m > 0.5
            and
            histogram_quantile(0.95,
              sum(rate(http_request_duration_seconds_bucket[6h])) by (le, service)
            ) > 0.5
          )
        for: 30m
        labels:
          severity: warning
          slo: latency
          burn_rate: high
        annotations:
          summary: "High latency SLO burn rate ({{ $labels.service }})"
          description: "95th percentile latency is {{ $value }}s over 6 hours"
```

---

## Runbook Automation

### Runbook Structure

```markdown
# Runbook: High Error Rate

## Alert Details
- **Alert Name**: HighErrorRate
- **Severity**: Critical
- **Team**: Platform
- **SLO Impact**: Availability SLO

## Symptoms
- Error rate > 5% for 5 minutes
- Users experiencing service failures
- Increased 5xx responses

## Diagnosis Steps

### 1. Check Service Health
```bash
# Check if service is running
kubectl get pods -l app=myservice -n production

# Check recent deployments
kubectl rollout history deployment/myservice -n production

# Check service logs
kubectl logs -l app=myservice -n production --tail=100 --since=10m
```

### 2. Check Dependencies
```bash
# Check database connectivity
kubectl exec -it myservice-pod -n production -- psql -h postgres -U app -c "SELECT 1"

# Check Redis connectivity
kubectl exec -it myservice-pod -n production -- redis-cli -h redis ping

# Check external API status
curl -s https://status.external-api.com/api/v1/status
```

### 3. Review Recent Changes
```bash
# Check recent commits
git log --oneline --since="1 hour ago"

# Check recent deployments
kubectl get events -n production --sort-by='.lastTimestamp' | grep deployment
```

## Automated Remediation

### Auto-Remediation Script
```bash
#!/bin/bash
# auto-remediate-high-error-rate.sh

SERVICE="myservice"
NAMESPACE="production"
ERROR_THRESHOLD=0.05

# Get current error rate
ERROR_RATE=$(curl -s 'http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])' | jq -r '.data.result[0].value[1]')

if (( $(echo "$ERROR_RATE > $ERROR_THRESHOLD" | bc -l) )); then
    echo "Error rate $ERROR_RATE exceeds threshold. Attempting remediation..."

    # Step 1: Restart unhealthy pods
    UNHEALTHY_PODS=$(kubectl get pods -n $NAMESPACE -l app=$SERVICE -o json | jq -r '.items[] | select(.status.containerStatuses[0].ready==false) | .metadata.name')

    if [ -n "$UNHEALTHY_PODS" ]; then
        echo "Restarting unhealthy pods: $UNHEALTHY_PODS"
        echo "$UNHEALTHY_PODS" | xargs -I {} kubectl delete pod {} -n $NAMESPACE
        sleep 30
    fi

    # Step 2: Check if error rate improved
    NEW_ERROR_RATE=$(curl -s 'http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])' | jq -r '.data.result[0].value[1]')

    if (( $(echo "$NEW_ERROR_RATE < $ERROR_THRESHOLD" | bc -l) )); then
        echo "Remediation successful. Error rate reduced to $NEW_ERROR_RATE"
        exit 0
    else
        echo "Remediation failed. Escalating to on-call engineer."
        # Trigger escalation
        curl -X POST https://pagerduty.com/api/v1/escalate \
            -H "Authorization: Bearer $PAGERDUTY_TOKEN" \
            -d '{"incident_key": "high-error-rate-'$SERVICE'", "escalation_level": 2}'
        exit 1
    fi
fi
```

### Webhook-Triggered Automation
```yaml
# webhook-receiver configuration
receivers:
  - name: 'auto-remediation-webhook'
    webhook_configs:
      - url: 'http://remediation-service:8080/remediate'
        send_resolved: false
        http_config:
          bearer_token: 'remediation-token'
        max_alerts: 1
```

```python
# remediation-service.py
from flask import Flask, request
import subprocess
import json

app = Flask(__name__)

@app.route('/remediate', methods=['POST'])
def remediate():
    alert_data = request.json

    for alert in alert_data.get('alerts', []):
        alertname = alert['labels'].get('alertname')
        service = alert['labels'].get('service')

        if alertname == 'HighErrorRate':
            result = subprocess.run(
                ['/scripts/auto-remediate-high-error-rate.sh', service],
                capture_output=True,
                text=True
            )

            return {
                'status': 'remediation_attempted',
                'alertname': alertname,
                'service': service,
                'result': result.stdout
            }

    return {'status': 'no_action_taken'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```
```

### Runbook Links in Alerts

```yaml
annotations:
  runbook_url: "https://runbooks.example.com/alerts/{{ $labels.alertname | toLower }}"
  runbook_steps: |
    1. Check service health: kubectl get pods -l app={{ $labels.service }}
    2. Review logs: kubectl logs -l app={{ $labels.service }} --tail=100
    3. Check dependencies: kubectl get svc,endpoints
    4. Review recent changes: git log --since="1 hour ago"
    5. If issue persists, execute: /scripts/remediate-{{ $labels.alertname | toLower }}.sh
```

---

## Alert Fatigue Prevention

### Alert Tuning Strategies

```yaml
groups:
  - name: tuned-alerts
    interval: 30s
    rules:
      # Use appropriate thresholds and durations
      - alert: HighCPUTuned
        expr: |
          100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 15m  # Long duration to avoid noise
        labels:
          severity: warning
        annotations:
          summary: "Sustained high CPU on {{ $labels.instance }}"
          description: "CPU has been >80% for 15 minutes"

      # Combine multiple signals to reduce false positives
      - alert: ServiceDegradedComposite
        expr: |
          (
            rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01
            and
            histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
            and
            rate(http_requests_total[5m]) > 10
          )
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Service degradation detected"
          description: "Multiple signals indicate service issues"

      # Time-based alert suppression (business hours only)
      - alert: NonCriticalIssue
        expr: |
          some_metric > threshold
          and
          hour() >= 9 and hour() < 18  # 9 AM to 6 PM
          and
          day_of_week() > 0 and day_of_week() < 6  # Monday to Friday
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Non-critical issue (business hours only)"
```

### Dynamic Thresholds

```yaml
groups:
  - name: dynamic-threshold-alerts
    interval: 30s
    rules:
      # Statistical anomaly detection
      - alert: AnomalousTraffic
        expr: |
          abs(
            rate(http_requests_total[5m])
            -
            avg_over_time(rate(http_requests_total[5m])[1h:5m])
          ) > (
            3 * stddev_over_time(rate(http_requests_total[5m])[1h:5m])
          )
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Anomalous traffic pattern detected"
          description: "Traffic deviates >3 standard deviations from 1-hour average"

      # Time-series prediction
      - alert: CapacityPrediction
        expr: |
          predict_linear(node_filesystem_avail_bytes[1h], 24 * 3600) < 0
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Disk will be full in <24 hours"
          description: "Based on current trends, disk {{ $labels.mountpoint }} will be full"
```

### Alert Deduplication

```yaml
# Alertmanager grouping for deduplication
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h

  routes:
    # Group instance-level alerts by service
    - match_re:
        alertname: '(HighCPU|HighMemory|DiskSpaceLow)'
      group_by: ['alertname', 'service']
      group_wait: 2m  # Wait to collect multiple instances
```

### Maintenance Windows

```yaml
# Silence alerts during maintenance
# Apply via Alertmanager API or UI

# Example: Silence all alerts for a service during deployment
{
  "matchers": [
    {
      "name": "service",
      "value": "myservice",
      "isRegex": false
    }
  ],
  "startsAt": "2025-11-07T10:00:00Z",
  "endsAt": "2025-11-07T11:00:00Z",
  "createdBy": "deployment-automation",
  "comment": "Automated deployment in progress"
}
```

```bash
# Create silence via amtool
amtool silence add \
    service=myservice \
    --duration=1h \
    --comment="Deployment in progress" \
    --author="deploy-bot"
```

---

## SLO-Based Alerting Integration

### SLO Definition and Tracking

```yaml
groups:
  - name: slo-definitions
    interval: 30s
    rules:
      # Availability SLO: 99.9% (43.2 minutes downtime per month)
      - record: slo:availability:target
        expr: 0.999

      - record: slo:availability:error_budget
        expr: 1 - slo:availability:target

      # Current error rate
      - record: slo:availability:error_rate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[30d])) by (service)
          /
          sum(rate(http_requests_total[30d])) by (service)

      # Remaining error budget (0-1 range)
      - record: slo:availability:budget_remaining
        expr: |
          (slo:availability:error_budget - slo:availability:error_rate) / slo:availability:error_budget

      # Latency SLO: 95% of requests < 500ms
      - record: slo:latency:target
        expr: 0.95

      - record: slo:latency:p95
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[30d])) by (le, service)
          )

      - record: slo:latency:target_threshold
        expr: 0.5  # 500ms

      # SLO compliance (1 = compliant, 0 = non-compliant)
      - record: slo:latency:compliant
        expr: |
          slo:latency:p95 < slo:latency:target_threshold

      # Alert on SLO budget consumption
      - alert: SLOBudget90PercentConsumed
        expr: |
          slo:availability:budget_remaining < 0.1
        for: 5m
        labels:
          severity: critical
          alert_type: slo
        annotations:
          summary: "90% of error budget consumed for {{ $labels.service }}"
          description: |
            {{ $labels.service }} has consumed 90% of its 30-day error budget.
            Remaining budget: {{ $value | humanizePercentage }}
          runbook_url: "https://runbooks.example.com/slo-budget-exhausted"

      - alert: SLOBudget50PercentConsumed
        expr: |
          slo:availability:budget_remaining < 0.5
        for: 15m
        labels:
          severity: warning
          alert_type: slo
        annotations:
          summary: "50% of error budget consumed for {{ $labels.service }}"
          description: "Remaining budget: {{ $value | humanizePercentage }}"
```

### SLO Dashboard Integration

```yaml
annotations:
  dashboard_url: "https://grafana.example.com/d/slo-dashboard?var-service={{ $labels.service }}&var-window=30d"
  slo_target: "99.9%"
  slo_current: "{{ with query \"slo:availability:error_rate\" }}{{ . | first | value | humanizePercentage }}{{ end }}"
  budget_remaining: "{{ with query \"slo:availability:budget_remaining\" }}{{ . | first | value | humanizePercentage }}{{ end }}"
```

---

## Incident Escalation and On-Call

### Escalation Policy Configuration

```yaml
# PagerDuty escalation policy
receivers:
  - name: 'escalation-level-1'
    pagerduty_configs:
      - service_key: 'L1_ONCALL_KEY'
        severity: 'critical'
        details:
          escalation_level: '1'
          escalation_policy: 'Platform Team L1'

  - name: 'escalation-level-2'
    pagerduty_configs:
      - service_key: 'L2_ONCALL_KEY'
        severity: 'critical'
        details:
          escalation_level: '2'
          escalation_policy: 'Platform Team L2'

  - name: 'escalation-level-3'
    pagerduty_configs:
      - service_key: 'MANAGEMENT_KEY'
        severity: 'critical'
        details:
          escalation_level: '3'
          escalation_policy: 'Management Escalation'

# Time-based escalation routing
route:
  routes:
    - match:
        severity: critical
      receiver: 'escalation-level-1'
      repeat_interval: 15m
      continue: true

    # Auto-escalate after 30 minutes
    - match:
        severity: critical
      receiver: 'escalation-level-2'
      group_wait: 30m
      repeat_interval: 15m
```

### On-Call Rotation Integration

```yaml
# Dynamic on-call routing based on schedule
receivers:
  - name: 'dynamic-oncall'
    webhook_configs:
      - url: 'http://oncall-router:8080/route'
        send_resolved: true
        http_config:
          bearer_token: 'router-token'
```

```python
# oncall-router service
from flask import Flask, request
import requests
from datetime import datetime

app = Flask(__name__)

ONCALL_SCHEDULES = {
    'platform': {
        'monday-friday-daytime': 'team-slack',
        'monday-friday-nighttime': 'oncall-pagerduty',
        'weekend': 'oncall-pagerduty'
    }
}

def get_current_schedule(team):
    now = datetime.now()
    day_of_week = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour

    if day_of_week >= 5:  # Weekend
        return ONCALL_SCHEDULES[team]['weekend']
    elif 9 <= hour < 18:  # Business hours
        return ONCALL_SCHEDULES[team]['monday-friday-daytime']
    else:  # Nighttime
        return ONCALL_SCHEDULES[team]['monday-friday-nighttime']

@app.route('/route', methods=['POST'])
def route_alert():
    alert_data = request.json
    team = alert_data['commonLabels'].get('team', 'platform')

    target_receiver = get_current_schedule(team)

    # Forward to appropriate receiver
    if target_receiver == 'oncall-pagerduty':
        # Send to PagerDuty
        requests.post(
            'https://events.pagerduty.com/v2/enqueue',
            json={
                'routing_key': 'ONCALL_KEY',
                'event_action': 'trigger',
                'payload': alert_data
            }
        )
    else:
        # Send to Slack
        requests.post(
            'https://hooks.slack.com/services/YOUR/WEBHOOK',
            json={'text': str(alert_data)}
        )

    return {'status': 'routed', 'receiver': target_receiver}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

---

## Alert Testing and Validation

### Alert Unit Testing

```yaml
# promtool test rules alert-tests.yml
rule_files:
  - prometheus-alerts.yml

evaluation_interval: 1m

tests:
  # Test high error rate alert
  - interval: 1m
    input_series:
      - series: 'http_requests_total{status="500", service="api"}'
        values: '0+60x10'  # 60 requests/min for 10 minutes
      - series: 'http_requests_total{status="200", service="api"}'
        values: '0+100x10'  # 100 requests/min for 10 minutes

    alert_rule_test:
      - eval_time: 5m
        alertname: HighErrorRate
        exp_alerts:
          - exp_labels:
              severity: critical
              service: api
              team: platform
            exp_annotations:
              summary: "High error rate on api"

  # Test SLO burn rate alert
  - interval: 1m
    input_series:
      - series: 'http_requests_total{status="500", service="api"}'
        values: '0+100x60'  # 100 errors/min
      - series: 'http_requests_total{status="200", service="api"}'
        values: '0+900x60'  # 900 success/min

    alert_rule_test:
      - eval_time: 10m
        alertname: SLOBurnRateCritical
        exp_alerts:
          - exp_labels:
              severity: critical
              service: api
              alert_type: slo_burn_rate
```

### Integration Testing

```bash
#!/bin/bash
# test-alerting-pipeline.sh

set -e

echo "Testing Alertmanager routing..."

# Test critical alert routing
amtool alert add \
    alertname="TestCriticalAlert" \
    severity="critical" \
    service="test-service" \
    --annotation=summary="Test critical alert" \
    --end=5m

# Verify alert was received
sleep 10
ALERTS=$(amtool alert query alertname="TestCriticalAlert")

if echo "$ALERTS" | grep -q "TestCriticalAlert"; then
    echo "✓ Alert created successfully"
else
    echo "✗ Alert creation failed"
    exit 1
fi

# Test silencing
echo "Testing alert silencing..."
SILENCE_ID=$(amtool silence add alertname="TestCriticalAlert" --duration=10m --comment="Test silence")

# Verify silence was created
if amtool silence query | grep -q "$SILENCE_ID"; then
    echo "✓ Silence created successfully"
else
    echo "✗ Silence creation failed"
    exit 1
fi

# Clean up
amtool silence expire "$SILENCE_ID"
echo "✓ All tests passed"
```

### Load Testing

```python
# load-test-alertmanager.py
import requests
import concurrent.futures
import time

ALERTMANAGER_URL = "http://alertmanager:9093/api/v1/alerts"

def send_alert(alert_id):
    alert = [{
        "labels": {
            "alertname": f"LoadTest{alert_id}",
            "severity": "warning",
            "service": "load-test"
        },
        "annotations": {
            "summary": f"Load test alert {alert_id}"
        }
    }]

    response = requests.post(ALERTMANAGER_URL, json=alert)
    return response.status_code == 200

def main():
    num_alerts = 1000
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(send_alert, range(num_alerts)))

    end_time = time.time()
    success_count = sum(results)

    print(f"Sent {success_count}/{num_alerts} alerts successfully")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Rate: {num_alerts / (end_time - start_time):.2f} alerts/second")

if __name__ == "__main__":
    main()
```

### End-to-End Validation

```bash
#!/bin/bash
# e2e-alert-validation.sh

# 1. Trigger test alert in Prometheus
curl -X POST http://prometheus:9090/api/v1/alerts \
    -d 'alert=TestE2EAlert&severity=critical'

# 2. Wait for alert to fire
sleep 60

# 3. Check Alertmanager received it
ALERT_PRESENT=$(curl -s http://alertmanager:9093/api/v1/alerts | \
    jq '.data[] | select(.labels.alertname=="TestE2EAlert")')

if [ -z "$ALERT_PRESENT" ]; then
    echo "✗ Alert not found in Alertmanager"
    exit 1
fi

echo "✓ Alert found in Alertmanager"

# 4. Check notification was sent (check webhook receiver)
NOTIFICATION_SENT=$(curl -s http://webhook-receiver:8080/alerts | \
    jq '.[] | select(.labels.alertname=="TestE2EAlert")')

if [ -z "$NOTIFICATION_SENT" ]; then
    echo "✗ Notification not sent"
    exit 1
fi

echo "✓ Notification sent successfully"
echo "✓ E2E validation passed"
```

---

## Summary

This comprehensive guide covers:

1. **Alertmanager Architecture**: Complete configuration with HA setup
2. **Alert Routing**: Hierarchical, team-based, and service mesh routing
3. **Grouping and Inhibition**: Advanced rules to reduce noise
4. **Notification Channels**: Slack, PagerDuty, email, and webhooks
5. **Alert Rule Patterns**: Symptom-based, resource-based, and application-specific
6. **Multi-Window Multi-Burn-Rate**: Google SRE-style SLO alerting
7. **Runbook Automation**: Auto-remediation and webhook integration
8. **Alert Fatigue Prevention**: Tuning, deduplication, and dynamic thresholds
9. **SLO-Based Alerting**: Error budget tracking and compliance monitoring
10. **Incident Escalation**: Dynamic on-call routing and escalation policies
11. **Alert Testing**: Unit tests, integration tests, and E2E validation

Use these patterns to build a production-ready alerting system that balances responsiveness with maintainability.
