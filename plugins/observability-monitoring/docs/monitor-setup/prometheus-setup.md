# Prometheus Setup and Configuration

Comprehensive guide for setting up Prometheus metrics collection, exporters, recording rules, and federation for production monitoring.

## Table of Contents

1. [Prometheus Architecture](#prometheus-architecture)
2. [Global Configuration](#global-configuration)
3. [Scrape Configurations](#scrape-configurations)
4. [Service Discovery](#service-discovery)
5. [Recording Rules](#recording-rules)
6. [Alerting Configuration](#alerting-configuration)
7. [Exporters](#exporters)
8. [Storage and Retention](#storage-and-retention)
9. [Federation](#federation)
10. [Best Practices](#best-practices)

---

## Prometheus Architecture

### Core Components

**Prometheus Server**:
- Time-series database with efficient storage
- Pull-based metrics collection via HTTP
- PromQL query language
- Alert rule evaluation
- Local storage with optional remote storage integration

**Key Characteristics**:
- Multi-dimensional data model (metrics identified by name + key-value labels)
- Flexible query language (PromQL)
- No dependency on distributed storage (autonomous single-server nodes)
- Pull model over HTTP
- Time series collection via intermediate push gateway
- Service discovery or static configuration for targets

### Data Model

**Metric Types**:

1. **Counter**: Cumulative metric that only increases (requests, errors)
2. **Gauge**: Single numerical value that can go up or down (CPU, memory)
3. **Histogram**: Samples observations and counts them in buckets (request durations)
4. **Summary**: Similar to histogram but calculates quantiles on client side

**Metric Naming Convention**:
```
<namespace>_<subsystem>_<name>_<unit>

Examples:
- http_requests_total (counter)
- http_request_duration_seconds (histogram)
- process_cpu_seconds_total (counter)
- node_memory_MemAvailable_bytes (gauge)
```

---

## Global Configuration

### Basic prometheus.yml

```yaml
# prometheus.yml
global:
  # How frequently to scrape targets by default
  scrape_interval: 15s

  # How frequently to evaluate rules
  evaluation_interval: 15s

  # External labels to attach to all time series and alerts
  external_labels:
    cluster: 'production'
    region: 'us-east-1'
    environment: 'prod'
    datacenter: 'dc1'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - 'alertmanager:9093'
      timeout: 10s
      api_version: v2

# Load rules from these files
rule_files:
  - "alerts/*.yml"
  - "recording_rules/*.yml"
  - "slo_rules/*.yml"

# Scrape configuration for targets
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### Advanced Global Configuration

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s  # Global timeout for scrape requests

  external_labels:
    cluster: 'production'
    region: 'us-east-1'
    monitor: 'prod-prometheus-01'

  # Query log file path
  query_log_file: /var/log/prometheus/query.log

# Remote write for long-term storage (Thanos, Cortex, M3DB)
remote_write:
  - url: "http://thanos-receive:19291/api/v1/receive"
    queue_config:
      capacity: 10000
      max_shards: 50
      min_shards: 1
      max_samples_per_send: 5000
      batch_send_deadline: 5s
      min_backoff: 30ms
      max_backoff: 100ms
    write_relabel_configs:
      - source_labels: [__name__]
        regex: 'expensive_metric.*'
        action: drop

# Remote read for querying historical data
remote_read:
  - url: "http://thanos-query:10902/api/v1/query"
    read_recent: true
```

---

## Scrape Configurations

### Static Targets

```yaml
scrape_configs:
  - job_name: 'api-servers'
    scrape_interval: 10s
    scrape_timeout: 5s
    metrics_path: '/metrics'
    scheme: 'https'

    static_configs:
      - targets:
          - 'api-1.example.com:443'
          - 'api-2.example.com:443'
          - 'api-3.example.com:443'
        labels:
          environment: 'production'
          tier: 'api'

      - targets:
          - 'api-1.staging.example.com:443'
        labels:
          environment: 'staging'
          tier: 'api'

    # TLS configuration
    tls_config:
      ca_file: /etc/prometheus/ca.crt
      cert_file: /etc/prometheus/client.crt
      key_file: /etc/prometheus/client.key
      insecure_skip_verify: false

    # Relabeling
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        regex: '([^:]+)(:[0-9]+)?'
        replacement: '${1}'
```

### File-Based Service Discovery

```yaml
scrape_configs:
  - job_name: 'file-sd'
    file_sd_configs:
      - files:
          - 'targets/api-*.json'
          - 'targets/web-*.yml'
        refresh_interval: 30s
```

**Target file example (targets/api-production.json)**:
```json
[
  {
    "targets": ["api-1.example.com:9090", "api-2.example.com:9090"],
    "labels": {
      "job": "api",
      "environment": "production",
      "tier": "backend"
    }
  },
  {
    "targets": ["api-3.example.com:9090"],
    "labels": {
      "job": "api",
      "environment": "production",
      "tier": "backend",
      "special": "canary"
    }
  }
]
```

---

## Service Discovery

### Kubernetes Service Discovery

```yaml
scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - default
            - production
            - monitoring

    relabel_configs:
      # Only scrape pods with annotation prometheus.io/scrape: "true"
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true

      # Use custom port if prometheus.io/port annotation exists
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

      # Use custom path if prometheus.io/path annotation exists
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

      # Add Kubernetes labels as Prometheus labels
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)

      # Add namespace label
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace

      # Add pod name label
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name

      # Add container name label
      - source_labels: [__meta_kubernetes_pod_container_name]
        action: replace
        target_label: kubernetes_container_name
```

### Kubernetes Service Discovery (Services)

```yaml
scrape_configs:
  - job_name: 'kubernetes-services'
    kubernetes_sd_configs:
      - role: service

    metrics_path: /probe
    params:
      module: [http_2xx]

    relabel_configs:
      # Only probe services with annotation prometheus.io/probe: "true"
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_probe]
        action: keep
        regex: true

      # Set target to service address
      - source_labels: [__address__]
        target_label: __param_target

      # Set actual scrape target to blackbox exporter
      - target_label: __address__
        replacement: blackbox-exporter:9115

      # Add service labels
      - source_labels: [__param_target]
        target_label: instance
      - action: labelmap
        regex: __meta_kubernetes_service_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_service_name]
        target_label: kubernetes_service_name
```

### Consul Service Discovery

```yaml
scrape_configs:
  - job_name: 'consul-services'
    consul_sd_configs:
      - server: 'consul.example.com:8500'
        datacenter: 'dc1'
        services: ['api', 'web', 'database']
        tags: ['production', 'monitoring']
        scheme: 'https'
        tls_config:
          ca_file: /etc/prometheus/consul-ca.crt

    relabel_configs:
      # Use service name as job label
      - source_labels: [__meta_consul_service]
        target_label: job

      # Add all Consul tags
      - source_labels: [__meta_consul_tags]
        target_label: tags
        regex: ',(.*),'

      # Add node name
      - source_labels: [__meta_consul_node]
        target_label: node
```

### EC2 Service Discovery (AWS)

```yaml
scrape_configs:
  - job_name: 'ec2-instances'
    ec2_sd_configs:
      - region: us-east-1
        access_key: YOUR_ACCESS_KEY
        secret_key: YOUR_SECRET_KEY
        port: 9090
        filters:
          - name: tag:Environment
            values:
              - production
          - name: instance-state-name
            values:
              - running

    relabel_configs:
      # Use instance ID as instance label
      - source_labels: [__meta_ec2_instance_id]
        target_label: instance

      # Use private IP
      - source_labels: [__meta_ec2_private_ip]
        target_label: __address__
        replacement: '${1}:9090'

      # Add availability zone
      - source_labels: [__meta_ec2_availability_zone]
        target_label: availability_zone

      # Add EC2 tags as labels
      - action: labelmap
        regex: __meta_ec2_tag_(.+)
```

---

## Recording Rules

Recording rules allow you to precompute frequently needed or computationally expensive expressions and save the result as a new time series.

### Basic Recording Rules

```yaml
# recording_rules/api_rules.yml
groups:
  - name: api_recording_rules
    interval: 30s
    rules:
      # Request rate per service
      - record: service:http_requests:rate5m
        expr: |
          sum(rate(http_requests_total[5m])) by (service, method, route)

      # Error rate per service
      - record: service:http_errors:rate5m
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)

      # Error ratio
      - record: service:http_error_ratio:rate5m
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          /
          sum(rate(http_requests_total[5m])) by (service)

      # Request duration p50, p95, p99
      - record: service:http_request_duration_seconds:p50
        expr: |
          histogram_quantile(0.50,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:http_request_duration_seconds:p95
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:http_request_duration_seconds:p99
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )
```

### RED Metrics Recording Rules

```yaml
# recording_rules/red_metrics.yml
groups:
  - name: red_metrics
    interval: 30s
    rules:
      # Rate: Request rate
      - record: service:request_rate:5m
        expr: |
          sum(rate(http_requests_total[5m])) by (service, method)

      # Errors: Error rate
      - record: service:error_rate:5m
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          /
          sum(rate(http_requests_total[5m])) by (service)

      # Duration: Latency percentiles
      - record: service:request_duration_p50:5m
        expr: |
          histogram_quantile(0.50,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:request_duration_p95:5m
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:request_duration_p99:5m
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )
```

### USE Metrics Recording Rules (Utilization, Saturation, Errors)

```yaml
# recording_rules/use_metrics.yml
groups:
  - name: use_metrics
    interval: 30s
    rules:
      # CPU Utilization
      - record: instance:cpu_utilization:rate5m
        expr: |
          1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)

      # Memory Utilization
      - record: instance:memory_utilization:ratio
        expr: |
          (
            node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes
          )
          /
          node_memory_MemTotal_bytes

      # Disk Utilization
      - record: instance:disk_utilization:ratio
        expr: |
          (
            node_filesystem_size_bytes{fstype!~"tmpfs|fuse.lxcfs"} -
            node_filesystem_free_bytes{fstype!~"tmpfs|fuse.lxcfs"}
          )
          /
          node_filesystem_size_bytes{fstype!~"tmpfs|fuse.lxcfs"}

      # Network Saturation (errors + drops)
      - record: instance:network_saturation:rate5m
        expr: |
          (
            rate(node_network_receive_errs_total[5m]) +
            rate(node_network_receive_drop_total[5m]) +
            rate(node_network_transmit_errs_total[5m]) +
            rate(node_network_transmit_drop_total[5m])
          )
          /
          (
            rate(node_network_receive_packets_total[5m]) +
            rate(node_network_transmit_packets_total[5m])
          )
```

### Aggregation Recording Rules

```yaml
# recording_rules/aggregations.yml
groups:
  - name: aggregation_rules
    interval: 1m
    rules:
      # Cluster-wide metrics
      - record: cluster:cpu_usage:sum
        expr: |
          sum(rate(node_cpu_seconds_total{mode!="idle"}[5m]))

      - record: cluster:memory_usage:sum
        expr: |
          sum(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)

      - record: cluster:network_receive_bytes:rate5m
        expr: |
          sum(rate(node_network_receive_bytes_total[5m]))

      - record: cluster:network_transmit_bytes:rate5m
        expr: |
          sum(rate(node_network_transmit_bytes_total[5m]))

      # Per-namespace aggregations (Kubernetes)
      - record: namespace:pod_cpu:sum
        expr: |
          sum(rate(container_cpu_usage_seconds_total[5m])) by (namespace)

      - record: namespace:pod_memory:sum
        expr: |
          sum(container_memory_working_set_bytes) by (namespace)

      # Per-service aggregations
      - record: service:http_requests_total:sum
        expr: |
          sum(rate(http_requests_total[5m])) by (service)

      - record: service:http_request_duration_seconds:avg
        expr: |
          avg(rate(http_request_duration_seconds_sum[5m])) by (service)
          /
          avg(rate(http_request_duration_seconds_count[5m])) by (service)
```

---

## Alerting Configuration

### Connecting to Alertmanager

```yaml
# prometheus.yml
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - 'alertmanager-1:9093'
            - 'alertmanager-2:9093'
            - 'alertmanager-3:9093'
      timeout: 10s
      api_version: v2
      path_prefix: /
      scheme: http

    # Kubernetes service discovery for Alertmanager
    - kubernetes_sd_configs:
        - role: pod
          namespaces:
            names:
              - monitoring
      relabel_configs:
        - source_labels: [__meta_kubernetes_pod_label_app]
          action: keep
          regex: alertmanager
        - source_labels: [__meta_kubernetes_pod_container_port_number]
          action: keep
          regex: "9093"
```

### Alert Rules

```yaml
# alerts/api_alerts.yml
groups:
  - name: api_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          /
          sum(rate(http_requests_total[5m])) by (service)
          > 0.05
        for: 5m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "High error rate on {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} has error rate of {{ $value | humanizePercentage }}
            for more than 5 minutes.
            Current value: {{ $value }}
            Runbook: https://runbooks.example.com/high-error-rate

      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          ) > 1
        for: 10m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "High latency on {{ $labels.service }}"
          description: |
            Service {{ $labels.service }} p95 latency is {{ $value | humanizeDuration }}
            which exceeds threshold of 1s for more than 10 minutes.

      - alert: APIDown
        expr: |
          up{job="api"} == 0
        for: 1m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "API instance {{ $labels.instance }} is down"
          description: |
            API instance {{ $labels.instance }} has been down for more than 1 minute.
            Immediate action required.
```

---

## Exporters

### Node Exporter (Infrastructure Metrics)

**Installation**:
```bash
# Download
wget https://github.com/prometheus/node_exporter/releases/download/v1.6.1/node_exporter-1.6.1.linux-amd64.tar.gz
tar xvfz node_exporter-1.6.1.linux-amd64.tar.gz
cd node_exporter-1.6.1.linux-amd64

# Run as systemd service
sudo cp node_exporter /usr/local/bin/
```

**Systemd service** (`/etc/systemd/system/node_exporter.service`):
```ini
[Unit]
Description=Node Exporter
After=network.target

[Service]
Type=simple
User=node_exporter
ExecStart=/usr/local/bin/node_exporter \
  --collector.filesystem.mount-points-exclude='^/(sys|proc|dev|host|etc)($$|/)' \
  --collector.netclass.ignored-devices='^(veth.*|docker.*|br-.*)$$' \
  --collector.netdev.device-exclude='^(veth.*|docker.*|br-.*)$$' \
  --collector.textfile.directory=/var/lib/node_exporter/textfile_collector

Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

**Prometheus scrape config**:
```yaml
scrape_configs:
  - job_name: 'node-exporter'
    static_configs:
      - targets:
          - 'node1:9100'
          - 'node2:9100'
          - 'node3:9100'
    relabel_configs:
      - source_labels: [__address__]
        regex: '([^:]+)(:[0-9]+)?'
        target_label: instance
        replacement: '${1}'
```

### Blackbox Exporter (Endpoint Probing)

**Configuration** (`blackbox.yml`):
```yaml
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      method: GET
      valid_status_codes: [200]
      fail_if_ssl: false
      fail_if_not_ssl: false
      preferred_ip_protocol: "ip4"

  http_post_2xx:
    prober: http
    http:
      method: POST
      headers:
        Content-Type: application/json
      body: '{"health": "check"}'

  tcp_connect:
    prober: tcp
    timeout: 5s

  icmp:
    prober: icmp
    timeout: 5s
    icmp:
      preferred_ip_protocol: "ip4"

  dns_query:
    prober: dns
    dns:
      query_name: "example.com"
      query_type: "A"
```

**Prometheus scrape config**:
```yaml
scrape_configs:
  - job_name: 'blackbox-http'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
          - https://api.example.com/health
          - https://www.example.com
          - https://app.example.com
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115
```

### Custom Application Exporter (Python Example)

```python
# custom_exporter.py
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
import time
import random

# Create metrics
request_duration = Histogram(
    'myapp_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

request_count = Counter(
    'myapp_requests_total',
    'Total request count',
    ['method', 'endpoint', 'status']
)

active_connections = Gauge(
    'myapp_active_connections',
    'Number of active connections'
)

queue_size = Gauge(
    'myapp_queue_size',
    'Current queue size'
)

# Custom business metrics
revenue_total = Counter(
    'myapp_revenue_dollars_total',
    'Total revenue in dollars',
    ['product']
)

active_users = Gauge(
    'myapp_active_users',
    'Number of currently active users'
)

def collect_metrics():
    """Simulate metric collection"""
    while True:
        # Simulate varying metrics
        active_connections.set(random.randint(50, 200))
        queue_size.set(random.randint(0, 100))
        active_users.set(random.randint(100, 1000))

        # Simulate requests
        endpoint = random.choice(['/api/users', '/api/products', '/api/orders'])
        method = random.choice(['GET', 'POST', 'PUT'])
        status = random.choices([200, 201, 400, 500], weights=[70, 15, 10, 5])[0]

        duration = random.uniform(0.01, 2.0)
        request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        request_count.labels(method=method, endpoint=endpoint, status=status).inc()

        # Simulate revenue
        if random.random() < 0.1:  # 10% chance of purchase
            product = random.choice(['basic', 'pro', 'enterprise'])
            amount = {'basic': 9.99, 'pro': 29.99, 'enterprise': 99.99}[product]
            revenue_total.labels(product=product).inc(amount)

        time.sleep(1)

if __name__ == '__main__':
    # Start metrics server on port 8000
    start_http_server(8000)
    print("Custom exporter started on :8000")
    collect_metrics()
```

**Prometheus scrape config**:
```yaml
scrape_configs:
  - job_name: 'custom-app'
    static_configs:
      - targets:
          - 'app1:8000'
          - 'app2:8000'
    scrape_interval: 10s
```

---

## Storage and Retention

### Local Storage Configuration

```yaml
# Prometheus startup flags
--storage.tsdb.path=/var/lib/prometheus/data
--storage.tsdb.retention.time=30d
--storage.tsdb.retention.size=100GB
--storage.tsdb.wal-compression
--storage.tsdb.min-block-duration=2h
--storage.tsdb.max-block-duration=2h
```

**Docker example**:
```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:v2.45.0
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=90d'
      - '--storage.tsdb.retention.size=200GB'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"

volumes:
  prometheus-data:
```

### Remote Storage (Thanos)

**Prometheus configuration**:
```yaml
global:
  external_labels:
    cluster: 'production'
    replica: '01'

remote_write:
  - url: "http://thanos-receive:19291/api/v1/receive"
    queue_config:
      capacity: 10000
      max_shards: 50
      min_shards: 1
      max_samples_per_send: 5000
    write_relabel_configs:
      - source_labels: [__name__]
        regex: 'go_.*|process_.*'
        action: drop

remote_read:
  - url: "http://thanos-query:10902/api/v1/query"
    read_recent: true
```

**Thanos Sidecar** (for object storage):
```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:v2.45.0
    # ... prometheus config

  thanos-sidecar:
    image: thanosio/thanos:v0.31.0
    command:
      - sidecar
      - --tsdb.path=/prometheus
      - --prometheus.url=http://prometheus:9090
      - --grpc-address=0.0.0.0:10901
      - --http-address=0.0.0.0:10902
      - --objstore.config-file=/etc/thanos/bucket.yml
    volumes:
      - prometheus-data:/prometheus
      - ./thanos-bucket.yml:/etc/thanos/bucket.yml
    ports:
      - "10901:10901"
      - "10902:10902"
```

**Object storage config** (`thanos-bucket.yml`):
```yaml
type: S3
config:
  bucket: "prometheus-metrics"
  endpoint: "s3.amazonaws.com"
  region: "us-east-1"
  access_key: "YOUR_ACCESS_KEY"
  secret_key: "YOUR_SECRET_KEY"
```

---

## Federation

Federation allows a Prometheus server to scrape selected time series from another Prometheus server.

### Hierarchical Federation

**Prometheus config on central server**:
```yaml
scrape_configs:
  - job_name: 'federate-dc1'
    scrape_interval: 30s
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        # Federate all recording rules
        - '{__name__=~"job:.*"}'
        - '{__name__=~"service:.*"}'
        - '{__name__=~"instance:.*"}'
        # Federate critical raw metrics
        - 'up'
        - 'node_cpu_seconds_total'
        - 'node_memory_MemAvailable_bytes'
    static_configs:
      - targets:
          - 'prometheus-dc1:9090'
        labels:
          datacenter: 'dc1'

  - job_name: 'federate-dc2'
    scrape_interval: 30s
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{__name__=~"job:.*"}'
        - '{__name__=~"service:.*"}'
    static_configs:
      - targets:
          - 'prometheus-dc2:9090'
        labels:
          datacenter: 'dc2'
```

### Cross-Service Federation

```yaml
scrape_configs:
  - job_name: 'federate-api-metrics'
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{job="api"}'
        - '{job="api-gateway"}'
    static_configs:
      - targets:
          - 'prometheus-prod-1:9090'
          - 'prometheus-prod-2:9090'
```

---

## Best Practices

### 1. Metric Naming Conventions

```
# Good metric names
http_requests_total
http_request_duration_seconds
process_cpu_seconds_total
node_memory_MemAvailable_bytes

# Bad metric names
http_requests (missing unit)
requestDuration (camelCase not recommended)
memory_available_mb (use base units: bytes, seconds)
```

### 2. Label Best Practices

**Good labels** (low cardinality):
```prometheus
http_requests_total{method="GET", route="/api/users", status="200"}
http_requests_total{method="POST", route="/api/orders", status="201"}
```

**Bad labels** (high cardinality):
```prometheus
# DON'T use user IDs, email addresses, timestamps as labels
http_requests_total{user_id="12345", email="user@example.com"}
```

**Cardinality check**:
```promql
# Count unique label combinations
count(http_requests_total)

# Count by specific label
count by(method) (http_requests_total)
```

### 3. Recording Rule Naming

```yaml
# Format: level:metric:operations
# level: aggregation level (cluster, service, instance, etc.)
# metric: metric name
# operations: operations applied (rate5m, sum, avg, etc.)

# Examples:
service:http_requests:rate5m
instance:cpu_usage:ratio
cluster:memory_usage:sum
namespace:pod_cpu:sum
```

### 4. Scrape Interval Tuning

```yaml
global:
  scrape_interval: 15s  # Default for most metrics

scrape_configs:
  # Fast-changing metrics (queues, active connections)
  - job_name: 'realtime-metrics'
    scrape_interval: 5s
    static_configs:
      - targets: ['queue-service:9090']

  # Slow-changing metrics (configuration, versions)
  - job_name: 'static-metrics'
    scrape_interval: 60s
    static_configs:
      - targets: ['config-service:9090']

  # Infrastructure metrics
  - job_name: 'node-exporter'
    scrape_interval: 30s
    static_configs:
      - targets: ['node1:9100', 'node2:9100']
```

### 5. Resource Management

**Memory estimation**:
```
Memory = (number_of_time_series × samples_per_second × retention_seconds × bytes_per_sample)

Example:
- 1 million time series
- 1 sample per 15 seconds = 0.067 samples/sec
- 30 days retention = 2,592,000 seconds
- 2 bytes per sample (compressed)

Memory ≈ 1,000,000 × 0.067 × 2,592,000 × 2 / 1024^3 ≈ 323 GB

Rule of thumb: Allocate 50% more than calculated for overhead
Recommended: 500 GB RAM
```

**Disk estimation**:
```
Disk = Memory × 2 (for WAL and compaction overhead)

Example: 323 GB × 2 = 646 GB
Recommended: 1 TB SSD
```

### 6. High Availability Setup

**Multi-replica Prometheus**:
```yaml
# Prometheus replica 1
global:
  external_labels:
    cluster: 'production'
    replica: 'prometheus-01'

# Prometheus replica 2
global:
  external_labels:
    cluster: 'production'
    replica: 'prometheus-02'
```

**Alertmanager clustering**:
```yaml
# Alertmanager config
cluster:
  listen_address: "0.0.0.0:9094"
  peers:
    - "alertmanager-1:9094"
    - "alertmanager-2:9094"
    - "alertmanager-3:9094"
```

### 7. Security Best Practices

**Enable TLS**:
```yaml
# Prometheus startup flags
--web.config.file=/etc/prometheus/web-config.yml

# web-config.yml
tls_server_config:
  cert_file: /etc/prometheus/certs/prometheus.crt
  key_file: /etc/prometheus/certs/prometheus.key
  client_ca_file: /etc/prometheus/certs/ca.crt
  client_auth_type: "RequireAndVerifyClientCert"
```

**Enable authentication**:
```yaml
# web-config.yml
basic_auth_users:
  admin: $2y$10$hashed_password_here
  readonly: $2y$10$another_hashed_password

# Generate password hash:
# htpasswd -nBC 10 "" | tr -d ':\n'
```

### 8. Performance Optimization

**Query optimization**:
```promql
# BAD: Aggregates over all samples in range
sum(http_requests_total[5m])

# GOOD: Aggregates over rate (reduces data points)
sum(rate(http_requests_total[5m]))

# BAD: Multiple expensive queries
histogram_quantile(0.95, http_request_duration_seconds_bucket[5m])
histogram_quantile(0.99, http_request_duration_seconds_bucket[5m])

# GOOD: Use recording rules for expensive queries
service:request_duration_p95:5m  # Pre-computed
service:request_duration_p99:5m  # Pre-computed
```

**Metric reduction**:
```yaml
# Drop unused metrics
metric_relabel_configs:
  - source_labels: [__name__]
    regex: 'go_gc_.*|go_memstats_.*|process_.*'
    action: drop

# Aggregate high-cardinality labels
metric_relabel_configs:
  - source_labels: [__name__, le]
    separator: ';'
    regex: 'http_request_duration_seconds_bucket;(0.005|0.01|0.025)'
    action: drop  # Drop unnecessary histogram buckets
```

---

This comprehensive Prometheus setup guide covers production-grade configuration, best practices, and optimization strategies for scalable monitoring infrastructure.
