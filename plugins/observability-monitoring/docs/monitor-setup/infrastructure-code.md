# Infrastructure as Code for Observability Monitoring

Complete infrastructure templates for deploying Prometheus, Grafana, Jaeger, and Alertmanager across multiple cloud providers and environments.

## Table of Contents

- [Terraform Modules](#terraform-modules)
- [Helm Charts](#helm-charts)
- [Docker Compose](#docker-compose)
- [AWS CloudWatch Integration](#aws-cloudwatch-integration)
- [Azure Monitor Integration](#azure-monitor-integration)
- [GCP Operations Integration](#gcp-operations-integration)
- [Multi-Cloud Deployment](#multi-cloud-deployment)
- [Cost Optimization](#cost-optimization)
- [Security and Access Control](#security-and-access-control)
- [Backup and Disaster Recovery](#backup-and-disaster-recovery)

## Terraform Modules

### Prometheus Module

Complete Terraform module for deploying Prometheus with persistent storage and high availability.

```hcl
# modules/prometheus/main.tf

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

variable "namespace" {
  description = "Kubernetes namespace for Prometheus"
  type        = string
  default     = "monitoring"
}

variable "retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 15
}

variable "storage_class" {
  description = "Storage class for persistent volumes"
  type        = string
  default     = "gp3"
}

variable "storage_size" {
  description = "Storage size for Prometheus data"
  type        = string
  default     = "100Gi"
}

variable "replicas" {
  description = "Number of Prometheus replicas for HA"
  type        = number
  default     = 2
}

variable "remote_write_endpoints" {
  description = "Remote write endpoints for long-term storage"
  type = list(object({
    url = string
    basic_auth = optional(object({
      username = string
      password = string
    }))
  }))
  default = []
}

resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "monitoring"                   = "true"
    }
  }
}

resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "prometheus"
  version    = "25.3.1"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name

  values = [
    yamlencode({
      server = {
        retention = "${var.retention_days}d"

        persistentVolume = {
          enabled      = true
          storageClass = var.storage_class
          size         = var.storage_size
        }

        replicaCount = var.replicas

        resources = {
          requests = {
            cpu    = "500m"
            memory = "2Gi"
          }
          limits = {
            cpu    = "2000m"
            memory = "8Gi"
          }
        }

        remoteWrite = [
          for endpoint in var.remote_write_endpoints : {
            url = endpoint.url
            basicAuth = endpoint.basic_auth != null ? {
              username = {
                name = kubernetes_secret.remote_write_creds[endpoint.url].metadata[0].name
                key  = "username"
              }
              password = {
                name = kubernetes_secret.remote_write_creds[endpoint.url].metadata[0].name
                key  = "password"
              }
            } : null
          }
        ]

        service = {
          type = "ClusterIP"
        }
      }

      alertmanager = {
        enabled = true

        persistentVolume = {
          enabled      = true
          storageClass = var.storage_class
          size         = "10Gi"
        }

        replicaCount = 2

        config = {
          global = {
            resolve_timeout = "5m"
          }

          route = {
            group_by        = ["alertname", "cluster", "service"]
            group_wait      = "10s"
            group_interval  = "10s"
            repeat_interval = "12h"
            receiver        = "default"
          }

          receivers = [
            {
              name = "default"
            }
          ]
        }
      }

      pushgateway = {
        enabled = true

        resources = {
          requests = {
            cpu    = "100m"
            memory = "128Mi"
          }
        }
      }

      nodeExporter = {
        enabled = true
      }

      kubeStateMetrics = {
        enabled = true
      }
    })
  ]

  depends_on = [kubernetes_namespace.monitoring]
}

resource "kubernetes_secret" "remote_write_creds" {
  for_each = {
    for endpoint in var.remote_write_endpoints :
    endpoint.url => endpoint if endpoint.basic_auth != null
  }

  metadata {
    name      = "prometheus-remote-write-${md5(each.key)}"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  data = {
    username = base64encode(each.value.basic_auth.username)
    password = base64encode(each.value.basic_auth.password)
  }
}

output "prometheus_endpoint" {
  description = "Prometheus server endpoint"
  value       = "http://prometheus-server.${var.namespace}.svc.cluster.local"
}

output "alertmanager_endpoint" {
  description = "Alertmanager endpoint"
  value       = "http://prometheus-alertmanager.${var.namespace}.svc.cluster.local"
}
```

### Grafana Module

```hcl
# modules/grafana/main.tf

variable "namespace" {
  description = "Kubernetes namespace for Grafana"
  type        = string
  default     = "monitoring"
}

variable "admin_password" {
  description = "Admin password for Grafana"
  type        = string
  sensitive   = true
}

variable "domain" {
  description = "Domain for Grafana ingress"
  type        = string
  default     = "grafana.example.com"
}

variable "prometheus_url" {
  description = "Prometheus datasource URL"
  type        = string
}

variable "enable_oauth" {
  description = "Enable OAuth authentication"
  type        = bool
  default     = false
}

variable "oauth_config" {
  description = "OAuth configuration"
  type = object({
    client_id     = string
    client_secret = string
    auth_url      = string
    token_url     = string
    api_url       = string
  })
  default = null
}

resource "helm_release" "grafana" {
  name       = "grafana"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  version    = "7.0.8"
  namespace  = var.namespace

  values = [
    yamlencode({
      adminPassword = var.admin_password

      persistence = {
        enabled      = true
        storageClass = "gp3"
        size         = "10Gi"
      }

      datasources = {
        "datasources.yaml" = {
          apiVersion = 1
          datasources = [
            {
              name      = "Prometheus"
              type      = "prometheus"
              url       = var.prometheus_url
              access    = "proxy"
              isDefault = true
            }
          ]
        }
      }

      dashboardProviders = {
        "dashboardproviders.yaml" = {
          apiVersion = 1
          providers = [
            {
              name            = "default"
              orgId           = 1
              folder          = ""
              type            = "file"
              disableDeletion = false
              editable        = true
              options = {
                path = "/var/lib/grafana/dashboards/default"
              }
            }
          ]
        }
      }

      dashboards = {
        default = {
          kubernetes-cluster = {
            gnetId     = 7249
            revision   = 1
            datasource = "Prometheus"
          }
          kubernetes-pods = {
            gnetId     = 6417
            revision   = 1
            datasource = "Prometheus"
          }
          node-exporter = {
            gnetId     = 1860
            revision   = 31
            datasource = "Prometheus"
          }
        }
      }

      ingress = {
        enabled = true
        annotations = {
          "kubernetes.io/ingress.class"                = "nginx"
          "cert-manager.io/cluster-issuer"             = "letsencrypt-prod"
          "nginx.ingress.kubernetes.io/ssl-redirect"   = "true"
        }
        hosts = [var.domain]
        tls = [
          {
            secretName = "grafana-tls"
            hosts      = [var.domain]
          }
        ]
      }

      resources = {
        requests = {
          cpu    = "250m"
          memory = "512Mi"
        }
        limits = {
          cpu    = "1000m"
          memory = "2Gi"
        }
      }

      "grafana.ini" = merge(
        {
          server = {
            root_url = "https://${var.domain}"
          }

          analytics = {
            reporting_enabled = false
            check_for_updates = false
          }

          security = {
            admin_password = var.admin_password
          }
        },
        var.enable_oauth && var.oauth_config != null ? {
          "auth.generic_oauth" = {
            enabled        = true
            name           = "OAuth"
            client_id      = var.oauth_config.client_id
            client_secret  = var.oauth_config.client_secret
            auth_url       = var.oauth_config.auth_url
            token_url      = var.oauth_config.token_url
            api_url        = var.oauth_config.api_url
            scopes         = "openid profile email"
            allow_sign_up  = true
          }
        } : {}
      )
    })
  ]
}

output "grafana_url" {
  description = "Grafana URL"
  value       = "https://${var.domain}"
}
```

### Jaeger Module

```hcl
# modules/jaeger/main.tf

variable "namespace" {
  description = "Kubernetes namespace for Jaeger"
  type        = string
  default     = "tracing"
}

variable "storage_type" {
  description = "Storage backend type (elasticsearch, cassandra, memory)"
  type        = string
  default     = "elasticsearch"
}

variable "elasticsearch_url" {
  description = "Elasticsearch URL for storage"
  type        = string
  default     = ""
}

variable "retention_days" {
  description = "Trace retention period in days"
  type        = number
  default     = 7
}

resource "kubernetes_namespace" "tracing" {
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
      "tracing"                      = "true"
    }
  }
}

resource "helm_release" "jaeger" {
  name       = "jaeger"
  repository = "https://jaegertracing.github.io/helm-charts"
  chart      = "jaeger"
  version    = "0.71.11"
  namespace  = kubernetes_namespace.tracing.metadata[0].name

  values = [
    yamlencode({
      provisionDataStore = {
        cassandra = false
        elasticsearch = var.storage_type == "elasticsearch"
      }

      storage = {
        type = var.storage_type

        elasticsearch = var.storage_type == "elasticsearch" ? {
          host = var.elasticsearch_url

          indexCleaner = {
            enabled      = true
            numberOfDays = var.retention_days
            schedule     = "0 2 * * *"
          }
        } : null
      }

      agent = {
        enabled = true

        daemonset = {
          useHostPort = true
        }

        resources = {
          requests = {
            cpu    = "100m"
            memory = "128Mi"
          }
          limits = {
            cpu    = "500m"
            memory = "512Mi"
          }
        }
      }

      collector = {
        enabled      = true
        replicaCount = 2

        service = {
          type = "ClusterIP"
        }

        resources = {
          requests = {
            cpu    = "500m"
            memory = "1Gi"
          }
          limits = {
            cpu    = "2000m"
            memory = "4Gi"
          }
        }

        autoscaling = {
          enabled                        = true
          minReplicas                    = 2
          maxReplicas                    = 10
          targetCPUUtilizationPercentage = 80
        }
      }

      query = {
        enabled      = true
        replicaCount = 2

        service = {
          type = "ClusterIP"
        }

        ingress = {
          enabled = true
          annotations = {
            "kubernetes.io/ingress.class" = "nginx"
          }
          hosts = ["jaeger.example.com"]
        }

        resources = {
          requests = {
            cpu    = "250m"
            memory = "512Mi"
          }
          limits = {
            cpu    = "1000m"
            memory = "2Gi"
          }
        }
      }
    })
  ]
}

output "jaeger_collector_endpoint" {
  description = "Jaeger collector endpoint"
  value       = "http://jaeger-collector.${var.namespace}.svc.cluster.local:14268"
}

output "jaeger_query_endpoint" {
  description = "Jaeger query endpoint"
  value       = "http://jaeger-query.${var.namespace}.svc.cluster.local"
}
```

## Helm Charts

### Custom Monitoring Stack Chart

```yaml
# charts/monitoring-stack/Chart.yaml

apiVersion: v2
name: monitoring-stack
description: Complete observability stack with Prometheus, Grafana, and Jaeger
type: application
version: 1.0.0
appVersion: "1.0"

dependencies:
  - name: prometheus
    version: "25.3.1"
    repository: "https://prometheus-community.github.io/helm-charts"
    condition: prometheus.enabled

  - name: grafana
    version: "7.0.8"
    repository: "https://grafana.github.io/helm-charts"
    condition: grafana.enabled

  - name: jaeger
    version: "0.71.11"
    repository: "https://jaegertracing.github.io/helm-charts"
    condition: jaeger.enabled

  - name: loki
    version: "5.38.0"
    repository: "https://grafana.github.io/helm-charts"
    condition: loki.enabled
```

```yaml
# charts/monitoring-stack/values.yaml

global:
  storageClass: gp3
  domain: monitoring.example.com

prometheus:
  enabled: true

  server:
    retention: "15d"
    persistentVolume:
      size: 100Gi

    resources:
      requests:
        cpu: 500m
        memory: 2Gi
      limits:
        cpu: 2000m
        memory: 8Gi

    extraScrapeConfigs: |
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__

  alertmanager:
    enabled: true
    persistentVolume:
      size: 10Gi

grafana:
  enabled: true

  adminPassword: changeme

  persistence:
    enabled: true
    size: 10Gi

  datasources:
    datasources.yaml:
      apiVersion: 1
      datasources:
        - name: Prometheus
          type: prometheus
          url: http://{{ .Release.Name }}-prometheus-server
          isDefault: true
        - name: Loki
          type: loki
          url: http://{{ .Release.Name }}-loki:3100
        - name: Jaeger
          type: jaeger
          url: http://{{ .Release.Name }}-jaeger-query:16686

  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
        - name: 'default'
          orgId: 1
          folder: ''
          type: file
          disableDeletion: false
          editable: true
          options:
            path: /var/lib/grafana/dashboards/default

jaeger:
  enabled: true

  storage:
    type: elasticsearch

  collector:
    replicaCount: 2
    autoscaling:
      enabled: true
      minReplicas: 2
      maxReplicas: 10

loki:
  enabled: true

  persistence:
    enabled: true
    size: 50Gi

  config:
    limits_config:
      retention_period: 168h  # 7 days
```

### ServiceMonitor CRD

```yaml
# charts/monitoring-stack/templates/servicemonitor.yaml

apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {{ include "monitoring-stack.fullname" . }}
  labels:
    {{- include "monitoring-stack.labels" . | nindent 4 }}
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: {{ include "monitoring-stack.name" . }}
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
      scheme: http
```

## Docker Compose

### Complete Local Development Stack

```yaml
# docker-compose.yml

version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=15d'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
    volumes:
      - ./config/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - monitoring
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

  alertmanager:
    image: prom/alertmanager:v0.26.0
    container_name: alertmanager
    command:
      - '--config.file=/etc/alertmanager/config.yml'
      - '--storage.path=/alertmanager'
    volumes:
      - ./config/alertmanager:/etc/alertmanager
      - alertmanager_data:/alertmanager
    ports:
      - "9093:9093"
    networks:
      - monitoring
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.1.5
    container_name: grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_ROOT_URL=http://localhost:3000
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - ./config/grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - monitoring
    restart: unless-stopped
    depends_on:
      - prometheus

  jaeger:
    image: jaegertracing/all-in-one:1.50
    container_name: jaeger
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"
      - "4317:4317"
      - "4318:4318"
    networks:
      - monitoring
    restart: unless-stopped

  loki:
    image: grafana/loki:2.9.2
    container_name: loki
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - ./config/loki:/etc/loki
      - loki_data:/loki
    ports:
      - "3100:3100"
    networks:
      - monitoring
    restart: unless-stopped

  promtail:
    image: grafana/promtail:2.9.2
    container_name: promtail
    command: -config.file=/etc/promtail/config.yml
    volumes:
      - ./config/promtail:/etc/promtail
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    networks:
      - monitoring
    restart: unless-stopped
    depends_on:
      - loki

  node-exporter:
    image: prom/node-exporter:v1.6.1
    container_name: node-exporter
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--path.rootfs=/host'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/host:ro
    ports:
      - "9100:9100"
    networks:
      - monitoring
    restart: unless-stopped

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.2
    container_name: cadvisor
    privileged: true
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    ports:
      - "8080:8080"
    networks:
      - monitoring
    restart: unless-stopped

volumes:
  prometheus_data:
    driver: local
  alertmanager_data:
    driver: local
  grafana_data:
    driver: local
  loki_data:
    driver: local

networks:
  monitoring:
    driver: bridge
```

### Prometheus Configuration

```yaml
# config/prometheus/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'local'
    environment: 'development'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alerts/*.yml'

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'grafana'
    static_configs:
      - targets: ['grafana:3000']

  - job_name: 'alertmanager'
    static_configs:
      - targets: ['alertmanager:9093']

  - job_name: 'jaeger'
    static_configs:
      - targets: ['jaeger:14269']
```

## AWS CloudWatch Integration

### Terraform Module for CloudWatch Integration

```hcl
# modules/aws-cloudwatch/main.tf

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention period"
  type        = number
  default     = 7
}

variable "enable_container_insights" {
  description = "Enable Container Insights for EKS"
  type        = bool
  default     = true
}

# CloudWatch Log Group for EKS
resource "aws_cloudwatch_log_group" "eks_cluster" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = var.log_retention_days

  tags = {
    Name        = "${var.cluster_name}-logs"
    Environment = terraform.workspace
  }
}

# Container Insights for EKS
resource "aws_eks_addon" "cloudwatch_observability" {
  count = var.enable_container_insights ? 1 : 0

  cluster_name = var.cluster_name
  addon_name   = "amazon-cloudwatch-observability"

  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
}

# IAM Role for CloudWatch Agent
resource "aws_iam_role" "cloudwatch_agent" {
  name = "${var.cluster_name}-cloudwatch-agent"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")}"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:amazon-cloudwatch:cloudwatch-agent"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "cloudwatch_agent_policy" {
  role       = aws_iam_role.cloudwatch_agent.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "${var.cluster_name}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "node_cpu_utilization"
  namespace           = "ContainerInsights"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "Alert when CPU exceeds 80%"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    ClusterName = var.cluster_name
  }
}

resource "aws_cloudwatch_metric_alarm" "high_memory" {
  alarm_name          = "${var.cluster_name}-high-memory"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "node_memory_utilization"
  namespace           = "ContainerInsights"
  period              = 300
  statistic           = "Average"
  threshold           = 85
  alarm_description   = "Alert when memory exceeds 85%"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    ClusterName = var.cluster_name
  }
}

# SNS Topic for Alerts
resource "aws_sns_topic" "alerts" {
  name = "${var.cluster_name}-cloudwatch-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = "alerts@example.com"
}

data "aws_caller_identity" "current" {}
data "aws_eks_cluster" "cluster" {
  name = var.cluster_name
}

output "log_group_name" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.eks_cluster.name
}

output "cloudwatch_agent_role_arn" {
  description = "IAM role ARN for CloudWatch agent"
  value       = aws_iam_role.cloudwatch_agent.arn
}
```

### CloudWatch Agent Configuration

```yaml
# config/cloudwatch-agent-config.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: cloudwatch-agent-config
  namespace: amazon-cloudwatch
data:
  cwagentconfig.json: |
    {
      "agent": {
        "region": "us-east-1"
      },
      "logs": {
        "metrics_collected": {
          "kubernetes": {
            "cluster_name": "my-cluster",
            "metrics_collection_interval": 60
          }
        },
        "force_flush_interval": 5
      },
      "metrics": {
        "namespace": "ContainerInsights",
        "metrics_collected": {
          "cpu": {
            "measurement": [
              {"name": "cpu_usage_idle", "rename": "CPU_USAGE_IDLE", "unit": "Percent"},
              {"name": "cpu_usage_nice", "unit": "Percent"},
              "cpu_usage_guest"
            ],
            "metrics_collection_interval": 60,
            "totalcpu": false
          },
          "disk": {
            "measurement": [
              {"name": "used_percent", "rename": "DISK_USED", "unit": "Percent"}
            ],
            "metrics_collection_interval": 60,
            "resources": ["*"]
          },
          "mem": {
            "measurement": [
              {"name": "mem_used_percent", "rename": "MEMORY_USED", "unit": "Percent"}
            ],
            "metrics_collection_interval": 60
          }
        }
      }
    }
```

## Azure Monitor Integration

### Terraform Module for Azure Monitor

```hcl
# modules/azure-monitor/main.tf

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
}

variable "cluster_name" {
  description = "AKS cluster name"
  type        = string
}

variable "retention_days" {
  description = "Log retention period"
  type        = number
  default     = 30
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "monitoring" {
  name                = "${var.cluster_name}-logs"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "PerGB2018"
  retention_in_days   = var.retention_days

  tags = {
    Environment = terraform.workspace
    ManagedBy   = "terraform"
  }
}

# Container Insights Solution
resource "azurerm_log_analytics_solution" "container_insights" {
  solution_name         = "ContainerInsights"
  location              = var.location
  resource_group_name   = var.resource_group_name
  workspace_resource_id = azurerm_log_analytics_workspace.monitoring.id
  workspace_name        = azurerm_log_analytics_workspace.monitoring.name

  plan {
    publisher = "Microsoft"
    product   = "OMSGallery/ContainerInsights"
  }
}

# Application Insights
resource "azurerm_application_insights" "monitoring" {
  name                = "${var.cluster_name}-appinsights"
  location            = var.location
  resource_group_name = var.resource_group_name
  workspace_id        = azurerm_log_analytics_workspace.monitoring.id
  application_type    = "web"

  tags = {
    Environment = terraform.workspace
    ManagedBy   = "terraform"
  }
}

# Diagnostic Settings for AKS
resource "azurerm_monitor_diagnostic_setting" "aks" {
  name                       = "${var.cluster_name}-diagnostics"
  target_resource_id         = data.azurerm_kubernetes_cluster.aks.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.monitoring.id

  enabled_log {
    category = "kube-apiserver"
  }

  enabled_log {
    category = "kube-controller-manager"
  }

  enabled_log {
    category = "kube-scheduler"
  }

  enabled_log {
    category = "kube-audit"
  }

  enabled_log {
    category = "cluster-autoscaler"
  }

  metric {
    category = "AllMetrics"
    enabled  = true
  }
}

# Alert Rules
resource "azurerm_monitor_metric_alert" "cpu_high" {
  name                = "${var.cluster_name}-cpu-high"
  resource_group_name = var.resource_group_name
  scopes              = [data.azurerm_kubernetes_cluster.aks.id]
  description         = "CPU usage is above 80%"
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"

  criteria {
    metric_namespace = "Microsoft.ContainerService/managedClusters"
    metric_name      = "node_cpu_usage_percentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }

  action {
    action_group_id = azurerm_monitor_action_group.alerts.id
  }
}

resource "azurerm_monitor_metric_alert" "memory_high" {
  name                = "${var.cluster_name}-memory-high"
  resource_group_name = var.resource_group_name
  scopes              = [data.azurerm_kubernetes_cluster.aks.id]
  description         = "Memory usage is above 85%"
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"

  criteria {
    metric_namespace = "Microsoft.ContainerService/managedClusters"
    metric_name      = "node_memory_working_set_percentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 85
  }

  action {
    action_group_id = azurerm_monitor_action_group.alerts.id
  }
}

# Action Group
resource "azurerm_monitor_action_group" "alerts" {
  name                = "${var.cluster_name}-alerts"
  resource_group_name = var.resource_group_name
  short_name          = "monitoring"

  email_receiver {
    name          = "email-alerts"
    email_address = "alerts@example.com"
  }

  webhook_receiver {
    name        = "slack-alerts"
    service_uri = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  }
}

data "azurerm_kubernetes_cluster" "aks" {
  name                = var.cluster_name
  resource_group_name = var.resource_group_name
}

output "workspace_id" {
  description = "Log Analytics workspace ID"
  value       = azurerm_log_analytics_workspace.monitoring.workspace_id
}

output "instrumentation_key" {
  description = "Application Insights instrumentation key"
  value       = azurerm_application_insights.monitoring.instrumentation_key
  sensitive   = true
}
```

## GCP Operations Integration

### Terraform Module for GCP Operations

```hcl
# modules/gcp-operations/main.tf

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
}

variable "log_retention_days" {
  description = "Log retention period"
  type        = number
  default     = 30
}

# Enable required APIs
resource "google_project_service" "monitoring" {
  project = var.project_id
  service = "monitoring.googleapis.com"

  disable_on_destroy = false
}

resource "google_project_service" "logging" {
  project = var.project_id
  service = "logging.googleapis.com"

  disable_on_destroy = false
}

resource "google_project_service" "cloudtrace" {
  project = var.project_id
  service = "cloudtrace.googleapis.com"

  disable_on_destroy = false
}

# Log Sink for GKE
resource "google_logging_project_sink" "gke_logs" {
  name        = "${var.cluster_name}-logs"
  destination = "storage.googleapis.com/${google_storage_bucket.logs.name}"

  filter = <<-EOT
    resource.type="k8s_cluster"
    resource.labels.cluster_name="${var.cluster_name}"
  EOT

  unique_writer_identity = true
}

# Storage Bucket for Logs
resource "google_storage_bucket" "logs" {
  name     = "${var.project_id}-${var.cluster_name}-logs"
  location = var.region

  lifecycle_rule {
    condition {
      age = var.log_retention_days
    }
    action {
      type = "Delete"
    }
  }

  uniform_bucket_level_access = true
}

resource "google_storage_bucket_iam_member" "log_writer" {
  bucket = google_storage_bucket.logs.name
  role   = "roles/storage.objectCreator"
  member = google_logging_project_sink.gke_logs.writer_identity
}

# Monitoring Alert Policies
resource "google_monitoring_alert_policy" "cpu_high" {
  display_name = "${var.cluster_name} - High CPU Usage"
  combiner     = "OR"

  conditions {
    display_name = "CPU usage above 80%"

    condition_threshold {
      filter          = "resource.type=\"k8s_node\" AND resource.labels.cluster_name=\"${var.cluster_name}\" AND metric.type=\"kubernetes.io/node/cpu/allocatable_utilization\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "memory_high" {
  display_name = "${var.cluster_name} - High Memory Usage"
  combiner     = "OR"

  conditions {
    display_name = "Memory usage above 85%"

    condition_threshold {
      filter          = "resource.type=\"k8s_node\" AND resource.labels.cluster_name=\"${var.cluster_name}\" AND metric.type=\"kubernetes.io/node/memory/allocatable_utilization\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.85

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
}

# Notification Channels
resource "google_monitoring_notification_channel" "email" {
  display_name = "Email Alerts"
  type         = "email"

  labels = {
    email_address = "alerts@example.com"
  }
}

# Custom Dashboard
resource "google_monitoring_dashboard" "cluster_overview" {
  dashboard_json = jsonencode({
    displayName = "${var.cluster_name} Overview"
    mosaicLayout = {
      columns = 12
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "CPU Usage"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"k8s_node\" AND resource.labels.cluster_name=\"${var.cluster_name}\" AND metric.type=\"kubernetes.io/node/cpu/allocatable_utilization\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
              }]
            }
          }
        },
        {
          width  = 6
          height = 4
          xPos   = 6
          widget = {
            title = "Memory Usage"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"k8s_node\" AND resource.labels.cluster_name=\"${var.cluster_name}\" AND metric.type=\"kubernetes.io/node/memory/allocatable_utilization\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
              }]
            }
          }
        }
      ]
    }
  })
}

output "log_bucket" {
  description = "Log storage bucket"
  value       = google_storage_bucket.logs.name
}

output "dashboard_url" {
  description = "Monitoring dashboard URL"
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.cluster_overview.id}"
}
```

## Multi-Cloud Deployment

### Unified Monitoring Stack

```hcl
# main.tf - Multi-cloud deployment

module "monitoring_aws" {
  count  = var.deploy_aws ? 1 : 0
  source = "./modules/aws-cloudwatch"

  cluster_name         = var.aws_cluster_name
  region               = var.aws_region
  log_retention_days   = var.log_retention_days
  enable_container_insights = true
}

module "monitoring_azure" {
  count  = var.deploy_azure ? 1 : 0
  source = "./modules/azure-monitor"

  resource_group_name = var.azure_resource_group
  location            = var.azure_location
  cluster_name        = var.azure_cluster_name
  retention_days      = var.log_retention_days
}

module "monitoring_gcp" {
  count  = var.deploy_gcp ? 1 : 0
  source = "./modules/gcp-operations"

  project_id         = var.gcp_project_id
  region             = var.gcp_region
  cluster_name       = var.gcp_cluster_name
  log_retention_days = var.log_retention_days
}

# Central Prometheus with Remote Write
module "prometheus_central" {
  source = "./modules/prometheus"

  namespace        = "monitoring"
  retention_days   = var.log_retention_days
  storage_class    = "fast-ssd"
  storage_size     = "500Gi"
  replicas         = 3

  remote_write_endpoints = concat(
    var.deploy_aws ? [{
      url = "https://aps-workspaces.${var.aws_region}.amazonaws.com/workspaces/${module.monitoring_aws[0].workspace_id}/api/v1/remote_write"
      basic_auth = {
        username = var.aws_prometheus_username
        password = var.aws_prometheus_password
      }
    }] : [],
    var.deploy_azure ? [{
      url = module.monitoring_azure[0].prometheus_endpoint
    }] : [],
    var.deploy_gcp ? [{
      url = "https://monitoring.googleapis.com:443/v1/projects/${var.gcp_project_id}/location/global/prometheus/api/v1/write"
    }] : []
  )
}

# Centralized Grafana
module "grafana_central" {
  source = "./modules/grafana"

  namespace      = "monitoring"
  admin_password = var.grafana_admin_password
  domain         = var.grafana_domain
  prometheus_url = module.prometheus_central.prometheus_endpoint

  enable_oauth = true
  oauth_config = {
    client_id     = var.oauth_client_id
    client_secret = var.oauth_client_secret
    auth_url      = var.oauth_auth_url
    token_url     = var.oauth_token_url
    api_url       = var.oauth_api_url
  }
}
```

## Cost Optimization

### Resource Right-Sizing

```hcl
# modules/cost-optimization/main.tf

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

locals {
  # Environment-based resource allocation
  resource_profiles = {
    dev = {
      prometheus_replicas    = 1
      prometheus_storage     = "50Gi"
      prometheus_cpu         = "500m"
      prometheus_memory      = "2Gi"
      grafana_replicas       = 1
      retention_days         = 7
    }
    staging = {
      prometheus_replicas    = 2
      prometheus_storage     = "100Gi"
      prometheus_cpu         = "1000m"
      prometheus_memory      = "4Gi"
      grafana_replicas       = 1
      retention_days         = 14
    }
    prod = {
      prometheus_replicas    = 3
      prometheus_storage     = "500Gi"
      prometheus_cpu         = "2000m"
      prometheus_memory      = "8Gi"
      grafana_replicas       = 2
      retention_days         = 30
    }
  }

  profile = local.resource_profiles[var.environment]
}

# Auto-scaling based on metrics
resource "kubernetes_horizontal_pod_autoscaler_v2" "prometheus" {
  metadata {
    name      = "prometheus-server"
    namespace = "monitoring"
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = "prometheus-server"
    }

    min_replicas = local.profile.prometheus_replicas
    max_replicas = local.profile.prometheus_replicas * 3

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 70
        }
      }
    }

    metric {
      type = "Resource"
      resource {
        name = "memory"
        target {
          type                = "Utilization"
          average_utilization = 80
        }
      }
    }
  }
}

# Storage lifecycle policies
resource "kubernetes_manifest" "storage_lifecycle" {
  manifest = {
    apiVersion = "v1"
    kind       = "ConfigMap"
    metadata = {
      name      = "prometheus-storage-lifecycle"
      namespace = "monitoring"
    }
    data = {
      retention_policy = yamlencode({
        retention = "${local.profile.retention_days}d"
        compaction = {
          enabled = true
          interval = "2h"
        }
      })
    }
  }
}
```

## Security and Access Control

### RBAC Configuration

```yaml
# config/rbac/monitoring-rbac.yaml

apiVersion: v1
kind: ServiceAccount
metadata:
  name: prometheus
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus
rules:
  - apiGroups: [""]
    resources:
      - nodes
      - nodes/proxy
      - nodes/metrics
      - services
      - endpoints
      - pods
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources:
      - configmaps
    verbs: ["get"]
  - apiGroups: ["networking.k8s.io"]
    resources:
      - ingresses
    verbs: ["get", "list", "watch"]
  - nonResourceURLs: ["/metrics"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus
subjects:
  - kind: ServiceAccount
    name: prometheus
    namespace: monitoring
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: grafana
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: grafana
  namespace: monitoring
rules:
  - apiGroups: [""]
    resources:
      - configmaps
      - secrets
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: grafana
  namespace: monitoring
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: grafana
subjects:
  - kind: ServiceAccount
    name: grafana
    namespace: monitoring
```

### Network Policies

```yaml
# config/security/network-policies.yaml

apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: prometheus-network-policy
  namespace: monitoring
spec:
  podSelector:
    matchLabels:
      app: prometheus
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
        - podSelector:
            matchLabels:
              app: grafana
      ports:
        - protocol: TCP
          port: 9090
  egress:
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 443
        - protocol: TCP
          port: 80
    - to:
        - podSelector:
            matchLabels:
              app: alertmanager
      ports:
        - protocol: TCP
          port: 9093
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: grafana-network-policy
  namespace: monitoring
spec:
  podSelector:
    matchLabels:
      app: grafana
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 3000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: prometheus
      ports:
        - protocol: TCP
          port: 9090
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 443
```

## Backup and Disaster Recovery

### Velero Backup Configuration

```yaml
# config/backup/velero-backup.yaml

apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: monitoring-daily-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  template:
    includedNamespaces:
      - monitoring
    includedResources:
      - persistentvolumeclaims
      - persistentvolumes
      - configmaps
      - secrets
    labelSelector:
      matchLabels:
        backup: "true"
    snapshotVolumes: true
    ttl: 720h  # 30 days
---
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: monitoring-weekly-backup
  namespace: velero
spec:
  schedule: "0 3 * * 0"  # Weekly on Sunday at 3 AM
  template:
    includedNamespaces:
      - monitoring
    snapshotVolumes: true
    ttl: 2160h  # 90 days
```

### Prometheus Snapshot Script

```bash
#!/bin/bash
# scripts/prometheus-snapshot.sh

set -euo pipefail

PROMETHEUS_POD=$(kubectl get pods -n monitoring -l app=prometheus-server -o jsonpath='{.items[0].metadata.name}')
BACKUP_DIR="/backups/prometheus/$(date +%Y%m%d-%H%M%S)"
S3_BUCKET="s3://my-monitoring-backups"

echo "Creating Prometheus snapshot..."
kubectl exec -n monitoring "$PROMETHEUS_POD" -- \
  curl -X POST http://localhost:9090/api/v1/admin/tsdb/snapshot

SNAPSHOT_NAME=$(kubectl exec -n monitoring "$PROMETHEUS_POD" -- \
  ls -t /prometheus/snapshots | head -1)

echo "Copying snapshot to local backup directory..."
mkdir -p "$BACKUP_DIR"
kubectl cp "monitoring/$PROMETHEUS_POD:/prometheus/snapshots/$SNAPSHOT_NAME" "$BACKUP_DIR"

echo "Uploading to S3..."
aws s3 sync "$BACKUP_DIR" "$S3_BUCKET/$(date +%Y%m%d-%H%M%S)/"

echo "Cleaning up old backups (keeping last 30 days)..."
find /backups/prometheus -type d -mtime +30 -exec rm -rf {} +

echo "Backup completed successfully!"
```

### Disaster Recovery Terraform

```hcl
# modules/disaster-recovery/main.tf

variable "backup_bucket" {
  description = "S3 bucket for backups"
  type        = string
}

variable "backup_retention_days" {
  description = "Backup retention period"
  type        = number
  default     = 30
}

# S3 Bucket for Backups
resource "aws_s3_bucket" "backups" {
  bucket = var.backup_bucket

  tags = {
    Purpose = "monitoring-backups"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "delete-old-backups"
    status = "Enabled"

    expiration {
      days = var.backup_retention_days
    }
  }

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"

    transition {
      days          = 7
      storage_class = "GLACIER"
    }
  }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Backup CronJob
resource "kubernetes_cron_job_v1" "prometheus_backup" {
  metadata {
    name      = "prometheus-backup"
    namespace = "monitoring"
  }

  spec {
    schedule = "0 2 * * *"

    job_template {
      metadata {}

      spec {
        template {
          metadata {}

          spec {
            service_account_name = "prometheus-backup"

            container {
              name  = "backup"
              image = "amazon/aws-cli:latest"

              command = ["/bin/sh", "-c"]
              args = [
                <<-EOT
                  apk add --no-cache curl &&
                  SNAPSHOT=$(curl -X POST http://prometheus-server:9090/api/v1/admin/tsdb/snapshot | jq -r '.data.name') &&
                  kubectl cp monitoring/prometheus-server-0:/prometheus/snapshots/$SNAPSHOT /tmp/snapshot &&
                  aws s3 sync /tmp/snapshot ${var.backup_bucket}/$(date +%Y%m%d-%H%M%S)/
                EOT
              ]

              env {
                name  = "AWS_REGION"
                value = "us-east-1"
              }
            }

            restart_policy = "OnFailure"
          }
        }
      }
    }
  }
}

output "backup_bucket_name" {
  description = "Backup S3 bucket name"
  value       = aws_s3_bucket.backups.id
}
```

## Complete Deployment Example

```bash
#!/bin/bash
# deploy-monitoring.sh

set -euo pipefail

ENVIRONMENT=${1:-dev}
CLOUD_PROVIDER=${2:-aws}

echo "Deploying monitoring stack for $ENVIRONMENT on $CLOUD_PROVIDER..."

# Initialize Terraform
terraform init

# Select workspace
terraform workspace select "$ENVIRONMENT" || terraform workspace new "$ENVIRONMENT"

# Plan deployment
terraform plan \
  -var="environment=$ENVIRONMENT" \
  -var="deploy_$CLOUD_PROVIDER=true" \
  -out=tfplan

# Apply deployment
terraform apply tfplan

# Wait for pods to be ready
kubectl wait --for=condition=ready pod \
  -l app=prometheus-server \
  -n monitoring \
  --timeout=300s

kubectl wait --for=condition=ready pod \
  -l app=grafana \
  -n monitoring \
  --timeout=300s

# Get access information
echo "Monitoring stack deployed successfully!"
echo "Grafana URL: $(terraform output -raw grafana_url)"
echo "Prometheus URL: $(kubectl get svc -n monitoring prometheus-server -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')"
echo "Jaeger URL: $(kubectl get svc -n tracing jaeger-query -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')"
```

This comprehensive infrastructure-as-code documentation provides production-ready templates for deploying complete observability monitoring across multiple cloud providers with proper security, backup, and cost optimization strategies.
